import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# Assuming your custom dataset loader is in a file named 'dataset.py'
# in a 'dataloaders' folder. Adjust the import if your structure is different.
from dataloaders.dataset import LiverTumorSliceDataset

def calculate_tumor_areas(args):
    """
    Loads a dataset of medical images and masks, calculates the pixel area
    of the tumor in each ground truth mask, and saves the results to a CSV file.
    """
    csv_data = '/kaggle/input/cect-liver-2/file_check.csv'
    cect_root_dirs = ["/kaggle/input/cect-liver-1", "/kaggle/input/cect-liver-2"]
    mask_dir = "/kaggle/input/cect-liver-2/mask_files/mask_files"
    img_size = 256

    TUMOR_CLASS_INDEX = 1
    # ----------------------------------------------------

    # --- Dataset and DataLoader Setup ---
    # This sets up the dataset loader to read your test data split.
    db_test = LiverTumorSliceDataset(
        metadata_csv=csv_data,
        cect_root_dirs=cect_root_dirs,
        mask_dir=mask_dir,
        split="test",
        val_ratio=0.2,
        test_ratio=0.1,
        random_seed=42,
        output_size=(img_size, img_size),
        augment=False
    )

    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    print(f"Found {len(db_test)} slices in the test dataset.")

    print("--- Starting Tumor Area Calculation ---")
    print(f"Looking for tumor pixels with index: {TUMOR_CLASS_INDEX}")

    # --- Area Calculation Loop ---
    all_tumor_areas = []
    # Loop through each slice in the dataset
    for i, batch in enumerate(tqdm(test_loader, desc="Calculating Areas")):
        label = batch["label"] # Get the ground truth mask

        # Convert the torch tensor to a numpy array for processing
        label_np = label.squeeze().cpu().numpy()

        # Count the number of pixels that match the tumor class index
        tumor_area = np.sum(label_np == TUMOR_CLASS_INDEX)
        all_tumor_areas.append(tumor_area)

    # --- Save Results to CSV ---
    # Create a pandas DataFrame to store the results
    df = pd.DataFrame({
        'test_slice_index': range(len(all_tumor_areas)),
        'tumor_area_pixels': all_tumor_areas
    })

    # Save the DataFrame to the specified output CSV file
    output_path = './tumor_areas.csv'
    df.to_csv(output_path, index=False)

    print("\n--- Calculation Complete ---")
    print(f"Results saved to: {output_path}")
    print(f"Total slices processed: {len(all_tumor_areas)}")
    print(f"Mean tumor area: {df['Tumor_Area_Pixels'].mean():.2f} pixels")
    print("----------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Tumor Pixel Area from Ground Truth Masks")
    # Argument to specify where to save the output CSV file
    parser.add_argument('--output_csv', type=str, default='./tumor_areas.csv',
                        help='Path to save the output CSV file with tumor areas.')

    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    calculate_tumor_areas(args)
