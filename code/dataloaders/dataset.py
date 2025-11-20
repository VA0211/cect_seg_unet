import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import augmentations
from augmentations.ctaugment import OPS
import matplotlib.pyplot as plt
from PIL import Image

import nibabel as nib
import pandas as pd
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split

import pandas as pd
import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import sys

class LiverTumorPatientSliceDataset(Dataset):
    def __init__(self, 
                 metadata_csv, 
                 split="train",             # "train", "val", or "test"
                 val_ratio=0.2,             # % of training set for validation
                 test_ratio=0.1,            # % of total data for testing
                 random_seed=42,
                 output_size=(256, 256),
                 augment=True):
        
        self.split = split.lower()
        self.output_size = output_size
        self.augment = augment
        
        # 1. Load Metadata
        df = pd.read_csv(metadata_csv)
        
        # 2. STRICT CLEANING (Matching your manual check)
        # Only keep rows where mask_path is a valid string
        df = df[df['mask_path'].apply(lambda x: isinstance(x, str) and len(x) > 0)]
        
        # 3. Patient-Level Split (Only on patients who actually have masks)
        unique_patients = sorted(df['patient_id'].unique()) # Sort ensures reproducibility
        
        train_ids, test_ids = train_test_split(
            unique_patients, 
            test_size=test_ratio, 
            random_state=random_seed
        )
        
        # Split remaining train_ids into actual train and val
        # Correct math: If we want 20% val ratio, it applies to the remaining 90%
        val_size_adjusted = val_ratio / (1.0 - test_ratio)
        train_ids, val_ids = train_test_split(
            train_ids, 
            test_size=val_size_adjusted, 
            random_state=random_seed
        )

        # Select ID list based on split
        if self.split == 'train':
            self.target_ids = train_ids
        elif self.split == 'val':
            self.target_ids = val_ids
        elif self.split == 'test':
            self.target_ids = test_ids
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")
            
        # Filter DataFrame to only this split's patients
        self.df_split = df[df['patient_id'].isin(self.target_ids)].copy()
        
        # 4. Collect Slices
        # We run the collection logic immediately to populate the index
        self.slice_infos = self._collect_slices()
        
        # Print Stats
        print(f"[{self.split.upper()}] Patients: {len(self.target_ids)} | Slices: {len(self.slice_infos)}")

    def _collect_slices(self):
        slice_info_list = []
        
        # Iterate rows in this split
        for _, row in self.df_split.iterrows():
            ct_path = row['ct_path']
            mask_path = row['mask_path']
            
            try:
                # Fast Load: We rely on the mask to find valid slices
                mask_proxy = nib.load(mask_path)
                
                # Get data (this reads from disk)
                # Optimization: If your NII files are huge, this is the bottleneck.
                mask_data = mask_proxy.get_fdata()
                
                # Logic from your manual check:
                # Collapse H,W to check if any pixel in the slice is > 0
                # axis=(0,1) means we look at the XY plane for each Z
                tumor_slices = np.any(mask_data > 0, axis=(0, 1))
                
                # Get indices where True
                valid_z_indices = np.where(tumor_slices)[0]
                
                for z in valid_z_indices:
                    slice_info_list.append({
                        'ct_path': ct_path,
                        'mask_path': mask_path,
                        'slice_idx': int(z)
                    })
                    
            except Exception as e:
                print(f"Error reading {mask_path}: {e}")
                continue
                
        return slice_info_list

    def __len__(self):
        return len(self.slice_infos)

    def resize(self, img):
        h, w = img.shape
        zoom_factor = (self.output_size[0] / h, self.output_size[1] / w)
        return zoom(img, zoom_factor, order=1)

    def __getitem__(self, idx):
        info = self.slice_infos[idx]
        ct_path = info['ct_path']
        mask_path = info['mask_path']
        slice_idx = info['slice_idx']

        # Load specific slice from volume
        # Note: nibabel caches, but loading full volume every time is slow.
        # Ideally, convert data to .npy 2D slices offline for speed.
        ct_vol = nib.load(ct_path).dataobj
        mask_vol = nib.load(mask_path).dataobj
        
        # Use slicing to read only needed data if supported, otherwise loads all
        ct_slice = np.array(ct_vol[:, :, slice_idx]).astype(np.float32)
        mask_slice = np.array(mask_vol[:, :, slice_idx]).astype(np.float32)

        # Handle 4D data (H, W, 1) -> (H, W)
        if ct_slice.ndim > 2: ct_slice = ct_slice.squeeze()
        if mask_slice.ndim > 2: mask_slice = mask_slice.squeeze()

        # Normalize
        mean = np.mean(ct_slice)
        std = np.std(ct_slice)
        ct_slice = (ct_slice - mean) / (std + 1e-8)
        
        # Binarize Mask
        mask_slice = (mask_slice > 0).astype(np.float32)

        # Augmentation (Train only)
        if self.augment and self.split == "train":
            if random.random() > 0.5: # Vertical
                ct_slice = np.flip(ct_slice, axis=0).copy()
                mask_slice = np.flip(mask_slice, axis=0).copy()
            if random.random() > 0.5: # Horizontal
                ct_slice = np.flip(ct_slice, axis=1).copy()
                mask_slice = np.flip(mask_slice, axis=1).copy()

        # Resize
        if self.output_size:
            ct_slice = self.resize(ct_slice)
            mask_slice = self.resize(mask_slice)

        ct_tensor = torch.from_numpy(ct_slice).unsqueeze(0).float() # (1, H, W)
        mask_tensor = torch.from_numpy(mask_slice).long()           # (H, W)

        return {
            "image": ct_tensor,
            "label": mask_tensor,
            "idx": idx
        }
    
class LiverTumorSliceDatasetPatient(Dataset):
    def __init__(self, 
                 split_file,              # Path to "patient_file_splits.csv"
                 split,                   # "train", "val", or "test"
                 cect_root_dirs,        
                 mask_dir,
                 phase="P",              # <-- NEW PARAMETER (e.g., 'P', 'C1')
                 output_size=(256, 256),    
                 augment=True):
        
        self.cect_root_dirs = cect_root_dirs
        self.mask_dir = mask_dir
        self.output_size = output_size
        self.split = split.lower()
        self.augment = augment and self.split == "train"
        self.phase = phase # <-- Store the phase

        try:
            # Load the master split file
            all_splits_df = pd.read_csv(split_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Master split file not found: {split_file}")
        
        # Filter for the *current* split (e.g., 'train')
        df_this_split = all_splits_df[all_splits_df['split'] == self.split]
        
        # --- NEW FILTERING LOGIC ---
        if self.phase is not None:
            # Define extraction function
            def get_phase_from_file(file_name):
                try:
                    return file_name.split('_ct_')[-1].split('.')[0]
                except Exception:
                    return None
            
            # Apply it to create a new 'phase' column
            df_this_split = df_this_split.copy() # Avoid SettingWithCopyWarning
            df_this_split['extracted_phase'] = df_this_split['file_name'].apply(get_phase_from_file)
            
            # Filter the dataframe for the specified phase
            df_this_split = df_this_split[df_this_split['extracted_phase'] == self.phase]
            
            if df_this_split.empty:
                print(f"Warning: No files found for split '{self.split}' and phase '{self.phase}'", file=sys.stderr)
        # --- END NEW LOGIC ---

        # Create a set of all possible file name variations
        self.valid_files = set()
        for f in df_this_split['file_name'].tolist():
            self.valid_files.add(f)
            if f.endswith('.nii.gz'):
                # Add the .nii version
                self.valid_files.add(f.replace('.nii.gz', '.nii'))
            elif f.endswith('.nii'):
                # Add the .nii.gz version
                self.valid_files.add(f + '.gz')
        
        # Get patient list (just for the print message)
        patient_count = len(df_this_split['patient_id'].unique())

        if not self.valid_files:
            print(f"Warning: No valid file names were processed for split '{self.split}'", file=sys.stderr)

        # Gather all valid slices
        self.slice_infos = self._collect_slices()

        print(f"[{self.split.upper()}] Loaded {len(self.slice_infos)} slices from {patient_count} patients.")

    def _collect_slices(self):
        slice_info_list = []
        for root_dir in self.cect_root_dirs:
            for root, dirs, files in os.walk(root_dir):
                # Skip mask-related directories
                if "mask_files" in root.lower() or "liver_mask_files" in root.lower():
                    continue
                    
                for file in files:
                    # This check will now work for both .nii and .nii.gz
                    if file not in self.valid_files:
                        continue

                    ct_path = os.path.join(root, file)
                    # Create mask name (works for both .nii and .nii.gz)
                    mask_filename = file.replace("ct", "mask")
                    mask_path = os.path.join(self.mask_dir, mask_filename)

                    if not os.path.exists(mask_path):
                        # Try the other extension just in case
                        if mask_path.endswith('.nii'):
                            mask_path += '.gz'
                        elif mask_path.endswith('.nii.gz'):
                            mask_path = mask_path.replace('.nii.gz', '.nii')

                        if not os.path.exists(mask_path):
                            # print(f"Warning: Missing mask for {ct_path}")
                            continue

                    try:
                        ct_img = nib.load(ct_path)
                        mask_img = nib.load(mask_path)
                        ct_shape = ct_img.shape
                        mask_shape = mask_img.shape
                    except Exception:
                        continue # Skip corrupted files

                    if ct_shape[2] != mask_shape[2]:
                        continue
                    
                    mask_data = mask_img.get_fdata()

                    for z in range(ct_shape[2]):
                        if np.any(mask_data[:, :, z]):
                            slice_info_list.append((ct_path, mask_path, z))
        return slice_info_list

    def __len__(self):
        return len(self.slice_infos)

    def resize(self, img):
        h, w = img.shape
        return zoom(img, (self.output_size[0] / h, self.output_size[1] / w), order=1)

    def __getitem__(self, idx):
        ct_path, mask_path, slice_idx = self.slice_infos[idx]

        try:
            # FIX 1: Cast to float32 immediately to solve dtype=object error
            ct = nib.load(ct_path).get_fdata().astype(np.float32)
            mask = nib.load(mask_path).get_fdata().astype(np.uint8) 
        except Exception as e:
            print(f"Error loading file: {ct_path} or {mask_path}")
            raise e

        # Handle 4D/5D scans
        if ct.ndim != 3:
            if ct.ndim == 5:
                ct = ct[:, :, :, 0, 0] 
            elif ct.ndim == 4:
                ct = ct[:, :, :, 0] 
            ct = ct.astype(np.float32) # Ensure cast is maintained

        ct_slice = ct[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx]

        # Normalize
        ct_slice = (ct_slice - np.mean(ct_slice)) / (np.std(ct_slice) + 1e-8)
        
        # Binarize
        mask_slice = (mask_slice > 0).astype(np.float32)

        if self.augment:
            if random.random() > 0.5:
                ct_slice = np.flip(ct_slice, axis=0).copy()
                mask_slice = np.flip(mask_slice, axis=0).copy()
            if random.random() > 0.5:
                ct_slice = np.flip(ct_slice, axis=1).copy()
                mask_slice = np.flip(mask_slice, axis=1).copy()

        # FIX 2: Use correct resize order for mask
        if self.output_size:
            ct_slice = self.resize(ct_slice)
            mask_slice = self.resize(mask_slice)

        ct_tensor = torch.tensor(ct_slice, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask_slice.astype(np.uint8), dtype=torch.uint8) 

        return {
            "image": ct_tensor,
            "label": mask_tensor,
            "idx": idx
        }

class LiverTumorSliceDataset(Dataset):
    def __init__(self, 
                 metadata_csv,        
                 cect_root_dirs,       
                 mask_dir,
                 split="train",             # "train", "val", or "test"
                 val_ratio=0.2,             # % for validation
                 test_ratio=0.1,            # % for testing
                 random_seed=42,            # reproducible split
                 output_size=(256, 256),    # target resize
                 augment=True):
        
        self.cect_root_dirs = cect_root_dirs
        self.mask_dir = mask_dir
        self.output_size = output_size
        self.augment = augment
        self.split = split.lower()

        # Load and filter valid metadata
        df = pd.read_csv(metadata_csv)
        df = df[df['label'] == True]
        self.valid_files = df['file_name'].tolist()

        # Gather all valid slices
        all_slices = self._collect_slices()

        train_val_slices, test_slices = train_test_split(
            all_slices,
            test_size=test_ratio,
            random_state=random_seed,
            shuffle=True
        )
        train_slices, val_slices = train_test_split(
            train_val_slices,
            test_size=val_ratio / (1.0 - test_ratio),  # correct val ratio inside train+val
            random_state=random_seed,
            shuffle=True
        )

        if self.split == "train":
            self.slice_infos = train_slices
        elif self.split == "val":
            self.slice_infos = val_slices
        elif self.split == "test":
            self.slice_infos = test_slices
        else:
            raise ValueError(f"Unknown split: {self.split}")

        print(f"[{self.split.upper()}] Total {len(self.slice_infos)} slices")

    def _collect_slices(self):
        slice_info_list = []
        for root_dir in self.cect_root_dirs:
            for root, dirs, files in os.walk(root_dir):
                if os.path.basename(root).endswith("mask_files"):
                    continue
                for file in files:
                    if not (file.endswith('.nii') or file.endswith('.nii.gz')):
                        continue
                    if file not in self.valid_files:
                        continue

                    ct_path = os.path.join(root, file)
                    mask_name = file.replace("ct", "mask")
                    mask_path = os.path.join(self.mask_dir, mask_name)

                    if not os.path.exists(mask_path):
                        continue

                    try:
                        ct = nib.load(ct_path).get_fdata()
                        mask = nib.load(mask_path).get_fdata()
                    except Exception:
                        continue

                    if ct.shape[2] != mask.shape[2]:
                        continue

                    for z in range(ct.shape[2]):
                        if np.any(mask[:, :, z]):
                            slice_info_list.append((ct_path, mask_path, z))

        return slice_info_list

    def __len__(self):
        return len(self.slice_infos)

    def resize(self, img):
        h, w = img.shape
        return zoom(img, (self.output_size[0] / h, self.output_size[1] / w), order=1)

    def __getitem__(self, idx):
        ct_path, mask_path, slice_idx = self.slice_infos[idx]

        ct = nib.load(ct_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        if ct.ndim != 3:
            ct = ct[:, :, :, 0, 0]

        ct_slice = ct[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx]

        ct_slice = (ct_slice - np.mean(ct_slice)) / (np.std(ct_slice) + 1e-8)
        mask_slice = (mask_slice > 0).astype(np.float32)

        if self.augment and self.split == "train":
            if random.random() > 0.5:
                ct_slice = np.flip(ct_slice, axis=0).copy()
                mask_slice = np.flip(mask_slice, axis=0).copy()
            if random.random() > 0.5:
                ct_slice = np.flip(ct_slice, axis=1).copy()
                mask_slice = np.flip(mask_slice, axis=1).copy()

        if self.output_size:
            ct_slice = self.resize(ct_slice)
            mask_slice = self.resize(mask_slice)

        ct_tensor = torch.tensor(ct_slice, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask_slice, dtype=torch.uint8)

        return {
            "image": ct_tensor,
            "label": mask_tensor,
            "idx": idx
        }

class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            # # 计算要选择的数据数量（10%）
            # num_samples = len(self.sample_list)
            # num_samples_to_select = int(1 * num_samples)
            # # 随机选择数据
            # selected_samples = random.sample(self.sample_list, num_samples_to_select)
            # self.sample_list = selected_samples

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample

class BaseDataSets_Synapse(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            # # 计算要选择的数据数量（10%）
            # num_samples = len(self.sample_list)
            # num_samples_to_select = int(1 * num_samples)
            # # 随机选择数据
            # selected_samples = random.sample(self.sample_list, num_samples_to_select)
            # self.sample_list = selected_samples

        elif self.split == "val":
            with open(self._base_dir + "/val.txt", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = np.load(self._base_dir + "/train_npz/{}.npz".format(case))
        else:
            if self.split == "val":
                h5f = h5py.File(self._base_dir + "/test_vol_h5/{}.npy.h5".format(case))
            else:
                h5f = h5py.File(self._base_dir + "/test_vol_h5/{}.npy.h5".format(case))
                
        image = np.array(h5f["image"])
        label = np.array(h5f["label"])
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample

def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


class CTATransform(object):
    def __init__(self, output_size, cta):
        self.output_size = output_size
        self.cta = cta

    def __call__(self, sample, ops_weak, ops_strong):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        to_tensor = transforms.ToTensor()

        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        # apply augmentations
        image_weak = augmentations.cta_apply(transforms.ToPILImage()(image), ops_weak)
        image_strong = augmentations.cta_apply(image_weak, ops_strong)
        label_aug = augmentations.cta_apply(transforms.ToPILImage()(label), ops_weak)
        label_aug = to_tensor(label_aug).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()

        sample = {
            "image_weak": to_tensor(image_weak),
            "image_strong": to_tensor(image_strong),
            "label_aug": label_aug,
        }
        return sample

    def cta_apply(self, pil_img, ops):
        if ops is None:
            return pil_img
        for op, args in ops:
            pil_img = OPS[op].f(pil_img, *args)
        return pil_img

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample


class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
