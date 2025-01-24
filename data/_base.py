# Define transformation

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kstest
import numpy as np

num_workers = os.cpu_count()
# num_workers = 16

class MyDataset(Dataset):
    def __init__(self, data_dir, transforms=None, strength=None):
        """
        Args:
            data_dir (str) : Path to the directory containing 'label.csv' + images
            transforms (callable, optional): Optional transform to be applied on a sample.
            strength (float in [0, 1], optional): If provided, filter data by standardized Illusion_Strength 
                                                  close to this value (e.g., round by 0.1).
        """
        self.data_dir = data_dir
        self.data_info = self.get_img_info(data_dir)
        self.transforms = transforms
        self.strength = strength

        # Optional: If you want to filter rows by strength with 0.1 "step" or tolerance:
        if self.strength is not None:
            # Example: keep items where standardized Illusion_Strength is within ±0.05 of the requested strength
            lower_bound = self.strength - 0.05
            upper_bound = self.strength + 0.05
            self.data_info = self.data_info[
                (self.data_info['Illusion_Strength_Std'] >= lower_bound) &
                (self.data_info['Illusion_Strength_Std'] <= upper_bound)
            ].reset_index(drop=True)

    def __getitem__(self, item):
        item = int(item)  # Ensure item is an integer
        row = self.data_info.iloc[item]
        path_img, label = row['name'], row['label']

        label = torch.tensor(int(label))  # Convert label to a tensor
        path_img = os.path.join(self.data_dir, path_img)
        image = Image.open(path_img).convert('RGB')

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.data_info)

    def get_img_info(self, data_dir):
        # Load CSV
        path_dir = os.path.join(data_dir, 'label.csv')
        df = pd.read_csv(path_dir)
        
        col = 'Illusion_Strength'
        strength_values = df[col].values.astype(float)

        # -----------------------------------------------------
        # Step 1) Min–Max scale to [0,1]
        # -----------------------------------------------------
        min_val = strength_values.min()
        max_val = strength_values.max()
        denom = (max_val - min_val) if (max_val - min_val) != 0 else 1e-8
        strength_01 = (strength_values - min_val) / denom

        # -----------------------------------------------------
        # Step 2) Shift from [0,1] -> [-1,1] 
        # -----------------------------------------------------
        strength_neg1_1 = 2.0 * strength_01 - 1.0

        # -----------------------------------------------------
        # Step 3) Candidate transforms
        # -----------------------------------------------------
        def transform_arcsin(x):
            # arcsin domain is [-1,1], clamp to avoid numerical issues
            x_clamped = np.clip(x, -1, 1)
            return np.arcsin(x_clamped)
        
        def transform_arccos(x):
            # arccos domain is [-1,1], clamp to avoid numerical issues
            x_clamped = np.clip(x, -1, 1)
            return np.arccos(x_clamped)
        
        def transform_none(x):
            return x
        
        # We'll store each transform in a dictionary
        transforms = {
            'arcsin': transform_arcsin,
            'arccos': transform_arccos,
            'none':   transform_none
        }

        # -----------------------------------------------------
        # Step 4) For each transform, apply + re-scale to [0,1],
        #         then check uniformity with KS test.
        # -----------------------------------------------------
        best_transform = None
        best_p_value = -1
        best_data_01 = None

        for name, func in transforms.items():
            try:
                # Apply the transform
                transformed = func(strength_neg1_1)

                # Some transforms (arcsin/arccos) will produce values in [ -π/2..π/2 ] or [0..π].
                # Now we min-max scale that result again to [0,1].
                scaler = MinMaxScaler(feature_range=(0,1))
                transformed_01 = scaler.fit_transform(transformed.reshape(-1,1)).flatten()
                
                # Use KS test to check how "uniform" it is in [0,1].
                # Higher p-value => closer to uniform distribution => "better linearization"
                _, p_value = kstest(transformed_01, 'uniform')

                # Decide if this is the best so far
                if p_value > best_p_value:
                    best_p_value = p_value
                    best_transform = name
                    best_data_01 = transformed_01
            except ValueError:
                # If arcsin or arccos domain errors occur, skip
                continue

        print(f"\n[INFO] Chosen transform for Illusion_Strength: '{best_transform}' (KS p-value={best_p_value:.4g})")

        # -----------------------------------------------------
        # Store the standardized column in the DataFrame
        # -----------------------------------------------------
        if best_data_01 is None:
            # Fallback: if everything failed (extremely unlikely),
            # just store original scaled [0,1] data.
            df['Illusion_Strength_Std'] = strength_01
        else:
            df['Illusion_Strength_Std'] = best_data_01

        return df


# Custom Dataset to hold tensors
class TensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
    def __len__(self):
        return len(self.labels)