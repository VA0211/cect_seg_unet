import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Assuming your custom dataset loader is in a file named 'dataset.py'
# in a 'dataloaders' folder. Adjust the import if your structure is different.
from dataloaders.dataset import LiverTumorSliceDataset

# --- Image Saving Functions (from your code, with one correction) ---

def save_base_image(image, save_path=None):
    """Saves the base grayscale CT image."""
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    if save_path:
        # Corrected: the first argument to savefig should be the path.
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_truth_tumor_masks(image, tumor_mask=None, alpha=0.4, save_path=None):
    """Saves the CT image with the ground truth tumor mask overlaid."""
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    if tumor_mask is not None:
        plt.imshow(tumor_mask, cmap='Reds', alpha=alpha)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def process_dataset(output_dir):
    """
    Loads the entire dataset (train, validation, test), calculates tumor area
    for each slice, saves the base image and an overlay image, and compiles
    all information into a single CSV file.
    """

    csv_data = '/kaggle/input/cect-liver-2/file_check.csv'
    cect_root_dirs = ["/kaggle/input/cect-liver-1", "/kaggle/input/cect-liver-2"]
    mask_dir = "/kaggle/input/cect-liver-2/mask_files/mask_files"
    img_size = 256
    
    TUMOR_CLASS_INDEX = 1

    print("--- Starting Full Dataset Processing ---")
    print(f"Output directory: {output_dir}")
    print(f"Tumor Class Index: {TUMOR_CLASS_INDEX}")

    all_results = []
    splits_to_process = ['train', 'val', 'test']

    # --- Loop through each dataset split ---
    for split in splits_to_process:
        print(f"\nProcessing '{split}' split...")

        # Create specific output directories for this split
        image_save_dir = os.path.join(output_dir, split, 'images')
        overlay_save_dir = os.path.join(output_dir, split, 'tumor_mask')
        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(overlay_save_dir, exist_ok=True)

        # --- Dataset and DataLoader Setup for the current split ---
        db_split = LiverTumorSliceDataset(
            metadata_csv=csv_data,
            cect_root_dirs=cect_root_dirs,
            mask_dir=mask_dir,
            split=split,
            val_ratio=0.2,
            test_ratio=0.1,
            random_seed=42,
            output_size=(img_size, img_size),
            augment=False
        )
        
        loader = DataLoader(db_split, batch_size=1, shuffle=False, num_workers=1)
        print(f"Found {len(db_split)} slices.")

        # --- Processing Loop for the current split ---
        for i, batch in enumerate(tqdm(loader, desc=f"Processing {split}")):
            image_tensor = batch["image"]
            label_tensor = batch["label"]

            # Convert tensors to numpy arrays
            img_np = image_tensor.squeeze().cpu().numpy()
            label_np = label_tensor.squeeze().cpu().numpy()

            # Calculate tumor area
            tumor_area = np.sum(label_np == TUMOR_CLASS_INDEX)

            # Define file paths for the images to be saved
            image_filename = f"slice_{i:04d}.png"
            image_path = os.path.join(image_save_dir, image_filename)
            overlay_path = os.path.join(overlay_save_dir, image_filename)

            # Save the base image and the overlay image
            save_base_image(img_np, image_path)
            save_truth_tumor_masks(img_np, tumor_mask=label_np, save_path=overlay_path)

            # Store the results
            all_results.append({
                'slice_index': i,
                'split': split,
                'tumor_area_pixels': tumor_area,
            })

    # --- Finalize and Save CSV ---
    print("\n--- Aggregating results and saving CSV ---")
    df = pd.DataFrame(all_results)
    
    # Reorder columns for clarity
    df = df[['split', 'slice_index', 'tumor_area_pixels']]

    csv_save_path = os.path.join(output_dir, 'full_dataset_tumor_areas.csv')
    df.to_csv(csv_save_path, index=False)

    print(f"\nProcessing complete!")
    print(f"Total slices processed: {len(df)}")
    print(f"Results summary saved to: {csv_save_path}")
    print("------------------------------------------")


if __name__ == '__main__':
    process_dataset("/kaggle/working/")