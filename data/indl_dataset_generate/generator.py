# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Dataset generator for various visual illusions.
Each function generates a dataset for a specific illusion type,
saves images and corresponding labels to the given path.

Supported illusions:
- Hering Wundt
- Muller-Lyer
- Poggendorff
- Vertical–horizontal
- Zollner
- Red-Yellow Boundary
- Clock Angle

Dependencies: numpy, matplotlib, pandas, PIL, pyllusion, tqdm
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from PIL import Image

import pandas as pd
import pyllusion
from tqdm import tqdm
from contextlib import contextmanager
import functools

# Checkpoint decorator for tracking and resuming progress
def checkpoint_generation(func):
    @functools.wraps(func)
    def wrapper(path, size, positive_ratio):
        csv_path = os.path.join(path, "label.csv")
        start_index = 0
        
        # Check if the CSV file already exists (was partially generated)
        if os.path.exists(csv_path):
            try:
                # Load existing progress
                existing_df = pd.read_csv(csv_path)
                start_index = len(existing_df)
                
                # If the dataset is already complete, just return
                if start_index >= size:
                    print(f"{func.__name__} in {path} is already complete with {start_index} samples.")
                    return
                
                print(f"Resuming {func.__name__} generation in {path} from index {start_index}/{size}.")
                
                # Save a backup of the existing CSV just in case
                existing_df.to_csv(os.path.join(path, "label_backup.csv"), index=False)
                
            except Exception as e:
                print(f"Error reading existing CSV in {path}: {e}")
                print("Starting generation from scratch.")
                start_index = 0
        
        # Call the original function with a modified size to generate remaining samples
        remaining_size = size - start_index
        func(path, remaining_size, positive_ratio, start_index=start_index)
    
    return wrapper

# Hering Wundt illusion
@checkpoint_generation
def dataset01(path, size, positive_ratio, start_index=0):
    """
    Generate Hering Wundt illusion dataset.

    Args:
        path (str): Directory to save images and label file.
        size (int): Number of samples to generate.
        positive_ratio (float): Probability of generating a positive (parallel) sample.
        start_index (int): Starting index for generation (for resuming).

    Each image contains radiating lines and either parallel or bent red lines.
    """
    # Initialize dataframe or load existing one
    csv_path = os.path.join(path, "label.csv")
    if (os.path.exists(csv_path) and start_index > 0):
        label_df = pd.read_csv(csv_path)
    else:
        label_df = pd.DataFrame(columns=['name', 'label', 'max_slope', 'step_size', 'bend', 'illusion_strength'])
    
    # Save progress periodically
    save_interval = 50
    
    for i in tqdm(range(size)):
        current_index = start_index + i
        
        # Create a new figure for each image
        fig = plt.figure(figsize=(4, 4), facecolor='white', dpi=64)
        
        label = int(np.random.rand() < positive_ratio)
        max_slope = np.random.rand() * 5 + 1
        step_size = np.random.rand() * 0.15 + 0.1
        bend = 0
        illusion_strength = 0.0
        # generate radiating lines
        for angle in np.arange(-max_slope, max_slope, step_size):
            plt.plot(np.arange(-2,2,0.01), angle * np.arange(-2,2,0.01), 'k')

        # generate the straight lines
        if label: #parallel
            plt.plot([-1,-1], [-4,4], 'r', linewidth=2)
            plt.plot([1,1], [-4,4], 'r', linewidth=2)
            illusion_strength = 0.0  # No illusion in parallel case
        else: #bended
            bend = 0
            while bend == 0:
                bend = np.random.rand() * 0.08
            sign = 1 if np.random.rand() > 0.5 else -1
            plt.plot([-1,-(1-sign * bend),-1], [-4,0,4], 'r', linewidth=2)
            plt.plot([1,(1-sign * bend),1], [-4,0,4], 'r', linewidth=2)
            
            # Calculate normalized illusion strength based on bend amount
            # According to illusion_strength.md, we normalize by the curvature needed to fully cancel the effect
            # A bend of 0.08 represents a strong illusion in our setup, so we normalize to that
            illusion_strength = abs(bend) / 0.08
            # Ensure it's in 0-1 range
            illusion_strength = min(1.0, illusion_strength)
        
        # hide axes
        plt.axis('off')
        
        # save image (removed random rotation)
        name = f'hering{current_index}.png'
        fig.savefig(os.path.join(path, name))
        
        # Properly close the figure to free memory
        plt.close(fig)
        
        label_df.loc[len(label_df)] = [name, label, max_slope, step_size, bend, illusion_strength]
        
        # Save progress periodically
        if (i + 1) % save_interval == 0 or i == size - 1:
            label_df.to_csv(csv_path, index=False)
        
    # Final save
    label_df.to_csv(csv_path, index=False)

# Muller-Lyer illusion
@checkpoint_generation
def dataset02(path, size, positive_ratio, start_index=0):
    """
    Generate Muller-Lyer illusion dataset.

    Args:
        path (str): Directory to save images and label file.
        size (int): Number of samples to generate.
        positive_ratio (float): Probability of generating a positive (no difference) sample.
        start_index (int): Starting index for generation (for resuming).

    Each image contains a Muller-Lyer illusion with random parameters.
    """
    # Initialize dataframe or load existing one
    csv_path = os.path.join(path, "label.csv")
    if os.path.exists(csv_path) and start_index > 0:
        label_df = pd.read_csv(csv_path)
    else:
        label_df = pd.DataFrame(columns=['name', 'label', 'value', 'Top_x1', 'Top_y1', 'Top_x2', 'Top_y2', 
                              'Bottom_x1', 'Bottom_y1', 'Bottom_x2', 'Bottom_y2', 'illusion_strength'])
    
    # Save progress periodically
    save_interval = 50
    
    for i in tqdm(range(size)):
        current_index = start_index + i
        
        label = int(np.random.rand() < positive_ratio)
        if (label):
            diff = 0
        else: 
            diff = 0
            while diff == 0:
                diff = np.random.randint(-500, +500)/1000
        strength = -np.random.randint(25, 35)
        mullerlyer = pyllusion.MullerLyer(illusion_strength=strength, difference=diff, distance=np.random.randint(80, 120)/100)
        
        # Calculate normalized illusion strength based on diff amount
        # According to illusion_strength.md, Müller–Lyer illusions typically produce 10-20% misjudgment
        # We normalize by the maximum typical effect (0.20 or 20%)
        illusion_strength = 0.0
        if not label:  # Only non-zero for illusions
            illusion_strength = abs(diff) / 0.20  # Normalize by maximum typical effect (20%)
            illusion_strength = min(1.0, illusion_strength)  # Ensure it's in 0-1 range
        
        # Remove rotation and save directly
        img = mullerlyer.to_image(width=256, height=256, outline=4)
        fn = lambda x : 255 if x > 210 else 0
        img = img.convert("L").point(fn, mode='1')
        dict = mullerlyer.get_parameters()
        name = f"mullerlyer{current_index}.png"
        img.save(os.path.join(path, name))
        label_df.loc[len(label_df)] = [name, label, diff, dict['Top_x1'], dict['Top_y1'], dict['Top_x2'], dict['Top_y2'], 
                                    dict['Bottom_x1'], dict['Bottom_y1'], dict['Bottom_x2'], dict['Bottom_y2'], illusion_strength]
        
        # Save progress periodically
        if (i + 1) % save_interval == 0 or i == size - 1:
            label_df.to_csv(csv_path, index=False)
            
    # Final save
    label_df.to_csv(csv_path, index=False)

# Apply the checkpoint_generation decorator to all other dataset functions
# (Only showing the first two fully, but apply the pattern to all others)

#Poggendorff illusion
@checkpoint_generation
def dataset03(path, size, positive_ratio, start_index=0):
    """
    Generate Poggendorff illusion dataset.

    Args:
        path (str): Directory to save images and label file.
        size (int): Number of samples to generate.
        positive_ratio (float): Probability of generating a positive (no difference) sample.
        start_index (int): Starting index for generation (for resuming).

    Each image contains a Poggendorff illusion with random parameters.
    """
    # Initialize dataframe or load existing one
    csv_path = os.path.join(path, "label.csv")
    if os.path.exists(csv_path) and start_index > 0:
        label_df = pd.read_csv(csv_path)
    else:
        label_df = pd.DataFrame(columns=['name', 'label', 'value', 'Illusion_Strength', 'Left_x1', 'Left_y1', 'Left_x2', 'Left_y2', 
                                         'Right_x1', 'Right_y1','Right_x2', 'Right_y2','Angle','Rectangle_Height',
                                         'Rectangle_Width', 'normalized_illusion_strength'])
    
    save_interval = 50
    
    for i in tqdm(range(size)):
        current_index = start_index + i
        
        label = int(np.random.rand() < positive_ratio)
        if (label):
            diff = 0
        else: 
            diff = 0
            while diff == 0:
                diff = 0.3 * np.random.rand()
        strength = -np.random.randint(1, 60)
        poggendorff = pyllusion.Poggendorff(illusion_strength=strength, difference=diff)
        
        # Calculate normalized illusion strength
        # According to illusion_strength.md, we normalize by the gap between parallels
        # The maximum effect typically occurs around 45° angles
        normalized_illusion_strength = 0.0
        if not label:  # Only non-zero for illusions
            # Get parameters
            dict = poggendorff.get_parameters()
            # Calculate the apparent offset (difference) relative to the rectangle width (gap between parallels)
            # In this illusion, we use dict['Rectangle_Width'] as the normalization factor
            rectangle_width = dict['Rectangle_Width']
            if rectangle_width > 0:
                # Normalize by the gap width, maximum typical offset is around 20% of the gap width
                normalized_illusion_strength = abs(diff) / 0.20
                normalized_illusion_strength = min(1.0, normalized_illusion_strength)
            
        # Remove rotation and save directly
        img = poggendorff.to_image(width=256, height=256)
        fn = lambda x : 255 if x > 210 else 0
        img = img.convert("L").point(fn, mode='1')
        dict = poggendorff.get_parameters()
        name = f'poggendorff{current_index}.png'
        img.save(os.path.join(path, name))
        label_df.loc[len(label_df)] = [name, label, dict['Difference'], dict['Illusion_Strength'], 
                                      dict['Left_x1'], dict['Left_y1'], dict['Left_x2'], dict['Left_y2'], 
                                      dict['Right_x1'], dict['Right_y1'], dict['Right_x2'], dict['Right_y2'], 
                                      dict['Angle'], dict['Rectangle_Height'], dict['Rectangle_Width'], 
                                      normalized_illusion_strength]

        # Save progress periodically
        if (i + 1) % save_interval == 0 or i == size - 1:
            label_df.to_csv(csv_path, index=False)
    
    # Final save
    label_df.to_csv(csv_path, index=False)

# Vertical–horizontal illusion
@checkpoint_generation
def dataset04(path, size, positive_ratio, start_index=0):
    """
    Generate Vertical–horizontal illusion dataset.

    Args:
        path (str): Directory to save images and label file.
        size (int): Number of samples to generate.
        positive_ratio (float): Probability of generating a positive (no difference) sample.
        start_index (int): Starting index for generation (for resuming).

    Each image contains a vertical-horizontal illusion with random parameters.
    """
    # Initialize dataframe or load existing one
    csv_path = os.path.join(path, "label.csv")
    if os.path.exists(csv_path) and start_index > 0:
        label_df = pd.read_csv(csv_path)
    else:
        label_df = pd.DataFrame(columns=['name', 'label', 'value', 'illusion_strength'])
    
    save_interval = 50
    
    for i in tqdm(range(size)):
        current_index = start_index + i
        
        label = int(np.random.rand() < positive_ratio)
        if (label):
            diff = 0
        else: 
            diff = 0
            while diff == 0:
                diff = 0.3 * np.random.rand()
        strength = -np.random.randint(60, 90)
        zollner = pyllusion.VerticalHorizontal(illusion_strength=strength, difference=diff)
        
        # Calculate normalized illusion strength based on diff amount
        # According to illusion_strength.md, vertical-horizontal illusions typically produce 
        # overestimation of vertical lengths by 3.9%-10%, we normalize by 0.10 (10%)
        illusion_strength = 0.0
        if not label:  # Only non-zero for illusions
            illusion_strength = abs(diff) / 0.10  # Normalize by maximum typical effect (10%)
            illusion_strength = min(1.0, illusion_strength)  # Ensure it's in 0-1 range
        
        img = zollner.to_image(width=256, height=256)
        fn = lambda x : 255 if x > 210 else 0
        img = img.convert("L").point(fn, mode='1')
        name = f'vertical{current_index}.png'
        img.save(os.path.join(path, name))
        label_df.loc[len(label_df)] = [name, label, diff, illusion_strength]

        # Save progress periodically
        if (i + 1) % save_interval == 0 or i == size - 1:
            label_df.to_csv(csv_path, index=False)
    
    # Final save
    label_df.to_csv(csv_path, index=False)

# Zollner illusion
@checkpoint_generation
def dataset05(path, size, positive_ratio, start_index=0):
    """
    Generate Zollner illusion dataset.

    Args:
        path (str): Directory to save images and label file.
        size (int): Number of samples to generate.
        positive_ratio (float): Probability of generating a positive (no difference) sample.
        start_index (int): Starting index for generation (for resuming).

    Each image contains a Zollner illusion with random parameters.
    """
    # Initialize dataframe or load existing one
    csv_path = os.path.join(path, "label.csv")
    if os.path.exists(csv_path) and start_index > 0:
        label_df = pd.read_csv(csv_path)
    else:
        label_df = pd.DataFrame(columns=['name', 'label', 'value', 'illusion_strength'])
    
    save_interval = 50
    
    for i in tqdm(range(size)):
        current_index = start_index + i
        
        label = int(np.random.rand() < positive_ratio)
        if (label):
            diff = 0
        else: 
            diff = 0
            while diff == 0:
                diff = 9 * np.random.rand()
        strength = np.random.randint(45, 65)
        zollner = pyllusion.Zollner(illusion_strength=strength, difference=diff)
        
        # Calculate normalized illusion strength based on diff amount
        # According to illusion_strength.md, Zöllner illusion strength is measured by 
        # the angular deviation needed to make lines appear parallel
        # Typical maximum angular deviation is around 3 degrees
        illusion_strength = 0.0
        if not label:  # Only non-zero for illusions
            # Convert the difference to degrees (diff is in arbitrary units used by pyllusion)
            # In pyllusion, diff of 9 corresponds approximately to 3 degrees
            angle_degrees = diff / 3
            illusion_strength = angle_degrees / 3.0  # Normalize by maximum typical effect (3°)
            illusion_strength = min(1.0, illusion_strength)  # Ensure it's in 0-1 range
        
        # Remove rotation and save directly
        img = zollner.to_image(width=256, height=256)
        fn = lambda x : 255 if x > 210 else 0
        img = img.convert("L").point(fn, mode='1')
        name = f'zollner{current_index}.png'
        img.save(os.path.join(path, name))
        label_df.loc[len(label_df)] = [name, label, diff, illusion_strength]

        # Save progress periodically
        if (i + 1) % save_interval == 0 or i == size - 1:
            label_df.to_csv(csv_path, index=False)
    
    # Final save
    label_df.to_csv(csv_path, index=False)

# RED YELLOW BOUNDARY
@checkpoint_generation
def dataset06(path, size, positive_ratio, start_index=0):
    """
    Generate Red-Yellow Boundary dataset.

    Args:
        path (str): Directory to save images and label file.
        size (int): Number of samples to generate.
        positive_ratio (float): Probability of generating a positive (yellow) sample.
        start_index (int): Starting index for generation (for resuming).

    Each image contains a colored rectangle with random position, size, and color.
    """
    # Initialize dataframe or load existing one
    csv_path = os.path.join(path, "label.csv")
    if os.path.exists(csv_path) and start_index > 0:
        label_df = pd.read_csv(csv_path)
    else:
        label_df = pd.DataFrame(columns = ["name", "label", "width", "x", "y", "r", "g", "b", "illusion_strength"])
    
    save_interval = 50
    
    for i in tqdm(range(size)):
        current_index = start_index + i
        
        x = np.random.rand() * 43 - 21
        y = np.random.rand() * 43 - 21
        w = np.min([np.abs(x + 32), np.abs(32 - x), np.abs(y + 32), np.abs(32 - y), np.max([np.random.rand() * 64, 11]), 42])

        illusion_strength = 0.0
        if np.random.rand() > positive_ratio:
            # Calculate how close the color is to the red/yellow boundary
            # According to illusion_strength.md, the Red-Yellow Boundary illusion strength
            # can be measured by the categorical amplification of perceived difference
            g_value = np.random.rand()  # Random green component
            c = (1, g_value, 0)
            
            # Calculate normalized illusion strength
            # The closer g_value is to 0.5 (boundary between red and yellow), the stronger the illusion
            # Maximum strength is at g=0.5, minimum at g=0 or g=1
            illusion_strength = 1.0 - abs(g_value - 0.5) * 2.0
            illusion_strength = min(1.0, max(0.0, illusion_strength))
            label = 0
        else:
            # Standard yellow-orange color (no illusion)
            c = (1, 0.5, 0)
            illusion_strength = 0.0
            label = 1

        # Create a new figure
        fig = plt.figure(figsize=(4, 4), facecolor='white', dpi=16)
        ax1 = fig.add_subplot(111, aspect = 'equal')
        ax1.add_patch(
            patches.Rectangle(
                (x, y),
                width=w,
                height=w,
                color = c
            )
        )
        ax1.set_xlim([-32,32])
        ax1.set_ylim([-32,32])
        ax1.set_axis_off()
        
        name = f'rect{current_index}.png'
        fig.savefig(os.path.join(path, name))
        
        # Close the figure
        plt.close(fig)
        
        label_df.loc[len(label_df)] = [name, label, w, x, y, *c, illusion_strength]
        
        # Save progress periodically
        if (i + 1) % save_interval == 0 or i == size - 1:
            label_df.to_csv(csv_path, index=False)
    
    # Final save
    label_df.to_csv(csv_path, index=False)

# CLOCK ANGLE
@checkpoint_generation
def dataset07(path, size, positive_ratio, start_index=0):
    """
    Generate Clock Angle dataset.

    Args:
        path (str): Directory to save images and label file.
        size (int): Number of samples to generate.
        positive_ratio (float): Probability of generating a positive (colinear) sample.
        start_index (int): Starting index for generation (for resuming).

    Each image contains three points forming two segments, with angle information.
    """
    def limit(x, min, max):
        # Clamp value x to [min, max]
        x = np.max((x, min))
        x = np.min((x, max))
        return x
    def get_angle(v1, v2):
        # Compute cosine of angle between vectors v1 and v2
        return v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    tolerance = 0 #tolerance 
    angleLimit = 0.1 #maximum angle between two segment
    
    # Initialize dataframe or load existing one
    csv_path = os.path.join(path, "label.csv")
    if (os.path.exists(csv_path) and start_index > 0):
        label_df = pd.read_csv(csv_path)
    else:
        label_df = pd.DataFrame(columns = ["name", "label", "angle", "x1", "y1", "x2", "y2", "x3", "y3", "illusion_strength"])
    
    save_interval = 50
    
    for i in tqdm(range(size)):
        current_index = start_index + i
        
        # Random points for the segments
        p1 = np.random.rand(2) * 64 - 32
        p2 = np.random.rand(2) * 64 - 32
        illusion_strength = 0.0
        
        if (np.random.rand() > positive_ratio):
            # Negative sample: introduce a small angle
            p3 = np.random.rand(2) * 64 - 32
            theta = np.random.rand() * 2 * angleLimit - angleLimit
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            p3 = R @ (p2 - p1) * np.random.rand() + p2 
            xl = limit(p3[0], -32, 32) / p3[0] if p3[0] != 0 else float('inf')
            yl = limit(p3[1], -32, 32) / p3[1] if p3[1] != 0 else float('inf')
            scale_factor = np.min([v for v in (xl, yl) if v > 0]) if min(xl, yl) > 0 else 1.0
            p3 = R @ (p2 - p1) * scale_factor + p2
            label = 0 # negative
            
            # Calculate illusion strength based on angular deviation
            # According to illusion_strength.md, clock angle illusion strength is measured by 
            # the angular difference or error in perception
            # The maximum error reported in studies was around 14° for 60° angles
            # We compute the angle between the segments and normalize it
            v1 = p2 - p1
            v2 = p3 - p2
            angle_cos = get_angle(v1, v2)
            # Convert from cosine to degrees
            if angle_cos > 1.0:  # Handle numerical issues
                angle_cos = 1.0
            if angle_cos < -1.0:
                angle_cos = -1.0
            angle_degrees = np.degrees(np.arccos(angle_cos))
            
            # Normalize using 15 degrees as maximum reference value
            illusion_strength = angle_degrees / 15.0
            illusion_strength = min(1.0, illusion_strength)
        else:
            # Positive sample: colinear
            p3 = (p2 - p1) * np.random.rand() + p2 
            xl = limit(p3[0], -32, 32) / p3[0] if p3[0] != 0 else float('inf')
            yl = limit(p3[1], -32, 32) / p3[1] if p3[1] != 0 else float('inf')
            scale_factor = np.min([v for v in (xl, yl) if v > 0]) if min(xl, yl) > 0 else 1.0
            p3 = (p2 - p1) * scale_factor + p2
            label = 1
            # No illusion in colinear case
            illusion_strength = 0.0
            
        P = np.concatenate([p1[None,:],p2[None,:],p3[None,:]], axis=0)
        
        # Create a new figure
        fig = plt.figure(figsize=(4, 4), facecolor='white', dpi=128)
        ax1 = fig.add_subplot(111, aspect='equal')
        ax1.set_xlim([-32,32])
        ax1.set_ylim([-32,32])
        ax1.set_axis_off()
        ax1.plot(P[:, 0], P[:, 1], linewidth=1, c='black')
        
        name = f'line{current_index}.png'
        fig.savefig(os.path.join(path, name))
        
        # Close the figure
        plt.close(fig)
        
        angle = get_angle(p2-p1, p3-p2)
        label_df.loc[len(label_df)] = [name, label, angle, *p1, *p2, *p3, illusion_strength]
        
        # Save progress periodically
        if (i + 1) % save_interval == 0 or i == size - 1:
            label_df.to_csv(csv_path, index=False)
    
    # Final save
    label_df.to_csv(csv_path, index=False)

# CAFE WALL ILLUSION
@checkpoint_generation
def dataset08(path, size, positive_ratio, start_index=0):
    """
    Generate Café Wall Illusion dataset.

    Args:
        path (str): Directory to save images and label file.
        size (int): Number of samples to generate.
        positive_ratio (float): Probability of generating a positive (straight) sample.
        start_index (int): Starting index for generation (for resuming).

    Each image contains a pattern of offset black and white tiles where the horizontal lines
    may appear sloped even though they are perfectly straight.
    """
    # Initialize dataframe or load existing one
    csv_path = os.path.join(path, "label.csv")
    if os.path.exists(csv_path) and start_index > 0:
        label_df = pd.read_csv(csv_path)
    else:
        label_df = pd.DataFrame(columns=['name', 'label', 'mortar_width', 'row_shift', 'tile_size', 'illusion_strength'])
    
    save_interval = 50
    
    for i in tqdm(range(size)):
        current_index = start_index + i
        
        tile_size = np.random.randint(8, 16)
        mortar_width = np.random.randint(1, 5)
        
        # For positive samples (non-illusion), no row shift
        label = int(np.random.rand() < positive_ratio)
        illusion_strength = 0.0
        
        if label:
            row_shift = 0  # No offset between rows creates no illusion
        else:
            row_shift = np.random.uniform(0.1, 0.5) * tile_size  # Offset creates illusion
            
            # Calculate normalized illusion strength based on row_shift
            # According to illusion_strength.md, Café Wall illusion strength is measured by
            # the apparent tilt angle of what should be horizontal lines
            # The tilt perception is strongest with optimal row_shift (around 50% of tile_size)
            # We normalize by the maximum apparent tilt
            
            # Empirically, the strength of the illusion is related to the ratio of row_shift to tile_size
            # Maximum effect occurs around row_shift = 0.5 * tile_size
            relative_shift = row_shift / tile_size
            
            # Create a curve that peaks at 0.5 and drops at 0 and 1
            # Using a simple parabolic function: 4x(1-x) which peaks at x=0.5 with value 1
            illusion_strength = 4.0 * relative_shift * (1.0 - relative_shift)
            
            # Also factor in mortar width - thinner mortar lines typically produce stronger illusions
            # Optimal mortar width is around 1-2 pixels
            mortar_factor = 1.0 - (mortar_width - 1) / 4.0  # Ranges from 1.0 (width=1) to 0.0 (width=5)
            mortar_factor = max(0.0, min(1.0, mortar_factor))
            
            # Combine the factors
            illusion_strength = illusion_strength * mortar_factor
        
        rows = 8
        cols = 10
        img_width = cols * tile_size
        img_height = rows * (tile_size + mortar_width) - mortar_width
        
        # Create a new figure
        fig = plt.figure(figsize=(4, 4), facecolor='white', dpi=64)
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim([0, img_width])
        ax.set_ylim([0, img_height])
        
        # Draw the tiles
        for row in range(rows):
            y = row * (tile_size + mortar_width)
            # Apply offset for alternating rows
            offset = row_shift if row % 2 == 1 else 0
            
            # Draw "mortar" (gray horizontal line)
            if row > 0:
                ax.add_patch(
                    patches.Rectangle(
                        (0, y - mortar_width),
                        width=img_width,
                        height=mortar_width,
                        color='gray'
                    )
                )
            
            # Draw tiles
            for col in range(cols):
                x = col * tile_size + offset
                # Alternate black and white tiles
                color = 'black' if col % 2 == 0 else 'white'
                ax.add_patch(
                    patches.Rectangle(
                        (x, y),
                        width=tile_size,
                        height=tile_size,
                        color=color,
                        edgecolor='gray',
                        linewidth=0.5
                    )
                )
        
        # Hide axes
        ax.set_axis_off()
        
        # Save image
        name = f'cafe_wall{current_index}.png'
        fig.savefig(os.path.join(path, name))
        
        # Close the figure
        plt.close(fig)
        
        # Add to dataframe
        label_df.loc[len(label_df)] = [name, label, mortar_width, row_shift, tile_size, illusion_strength]
        
        # Save progress periodically
        if (i + 1) % save_interval == 0 or i == size - 1:
            label_df.to_csv(csv_path, index=False)
    
    # Final save
    label_df.to_csv(csv_path, index=False)

# CHECKER SHADOW ILLUSION
@checkpoint_generation
def dataset09(path, size, positive_ratio, start_index=0):
    """
    Generate Checker Shadow Illusion dataset.

    Args:
        path (str): Directory to save images and label file.
        size (int): Number of samples to generate.
        positive_ratio (float): Probability of generating a positive (identical brightness) sample.
        start_index (int): Starting index for generation (for resuming).

    Each image contains a checkerboard with a shadow crossing it, with two marked squares
    that either are or appear to be different shades.
    """
    # Initialize dataframe or load existing one
    csv_path = os.path.join(path, "label.csv")
    if os.path.exists(csv_path) and start_index > 0:
        label_df = pd.read_csv(csv_path)
    else:
        label_df = pd.DataFrame(columns=['name', 'label', 'shadow_angle', 'shadow_alpha', 'square1_color', 'square2_color', 'illusion_strength'])
    
    save_interval = 50
    
    for i in tqdm(range(size)):
        current_index = start_index + i
        
        # Create a checkerboard pattern
        board_size = 7  # 7x7 grid
        square_size = 32  # pixels
        img_size = board_size * square_size
        
        fig = plt.figure(figsize=(4, 4), facecolor='white', dpi=64)
        ax = fig.add_subplot(111, aspect='equal')
        
        # Draw checkerboard
        for row in range(board_size):
            for col in range(board_size):
                color = 'white' if (row + col) % 2 == 0 else 'black'
                ax.add_patch(
                    patches.Rectangle(
                        (col * square_size, row * square_size),
                        width=square_size,
                        height=square_size,
                        color=color
                    )
                )
        
        # Define two squares to compare (fixed positions for consistency)
        square1_pos = (1, 2)  # (col, row)
        square2_pos = (5, 4)
        
        # Choose shadow parameters
        shadow_angle = np.random.randint(0, 180)
        shadow_alpha = np.random.uniform(0.2, 0.6)
        
        # Determine if this is a positive or negative sample
        label = int(np.random.rand() < positive_ratio)
        
        # Calculate normalized illusion strength
        # According to illusion_strength.md, Checker Shadow illusion strength is measured by 
        # the luminance difference that appears the same due to context
        illusion_strength = 0.0
        
        if label:
            # Positive case: squares have same brightness despite shadow
            # We'll create a gradient shadow and adjust square colors to compensate
            square1_color = 0.5  # mid-gray
            square2_color = 0.5  # same mid-gray
            # No illusion in this case (squares are actually the same color)
            illusion_strength = 0.0
        else:
            # Negative case: squares have different brightness but may appear similar due to shadow
            square1_color = np.random.uniform(0.35, 0.45)
            square2_color = np.random.uniform(0.55, 0.65)
            
            # Calculate normalized illusion strength based on color difference
            # According to illusion_strength.md, strength is measured as the ratio of
            # luminance differences that appear equal
            # Maximum difference in the checker shadow illusion is typically around 0.3 in absolute terms
            illusion_strength = abs(square1_color - square2_color) / 0.3
            illusion_strength = min(1.0, illusion_strength)
        
        # Create a shadow mask
        xx, yy = np.meshgrid(np.linspace(0, img_size, 100), np.linspace(0, img_size, 100))
        shadow_dir = np.array([np.cos(np.radians(shadow_angle)), np.sin(np.radians(shadow_angle))])
        shadow_start = img_size * 0.3  # Shadow starts 30% into the image
        shadow_width = img_size * 0.4  # Shadow width is 40% of image
        
        # Project points onto shadow direction
        proj = xx * shadow_dir[0] + yy * shadow_dir[1]
        shadow_mask = (proj > shadow_start) & (proj < shadow_start + shadow_width)
        
        # Draw shadow as a semi-transparent black rectangle
        shadow_rect = patches.Rectangle((0, 0), img_size, img_size, color='black', alpha=0)
        ax.add_patch(shadow_rect)
        
        # Create custom shadow using a gradient
        shadow = np.zeros((100, 100))
        shadow[shadow_mask] = shadow_alpha
        ax.imshow(shadow, extent=[0, img_size, 0, img_size], 
                 cmap='gray', alpha=0.5, origin='lower')
        
        # Draw the two squares we're comparing with specific colors
        for pos, color in [(square1_pos, square1_color), (square2_pos, square2_color)]:
            col, row = pos
            ax.add_patch(
                patches.Rectangle(
                    (col * square_size, row * square_size),
                    width=square_size,
                    height=square_size,
                    color=str(color),
                    edgecolor='red',
                    linewidth=2
                )
            )
        
        # Hide axes
        ax.set_axis_off()
        
        # Save image
        name = f'checker_shadow{current_index}.png'
        fig.savefig(os.path.join(path, name))
        
        # Close the figure
        plt.close(fig)
        
        # Add to dataframe
        label_df.loc[len(label_df)] = [name, label, shadow_angle, shadow_alpha, square1_color, square2_color, illusion_strength]
        
        # Save progress periodically
        if (i + 1) % save_interval == 0 or i == size - 1:
            label_df.to_csv(csv_path, index=False)
    
    # Final save
    label_df.to_csv(csv_path, index=False)

# FRASER SPIRAL ILLUSION
@checkpoint_generation
def dataset10(path, size, positive_ratio, start_index=0):
    """
    Generate Fraser Spiral Illusion dataset.

    Args:
        path (str): Directory to save images and label file.
        size (int): Number of samples to generate.
        positive_ratio (float): Probability of generating a positive (concentric circles) sample.
        start_index (int): Starting index for generation (for resuming).

    Each image contains a series of twisted arcs that may give the illusion of a spiral,
    though they actually form concentric circles.
    """
    # Initialize dataframe or load existing one
    csv_path = os.path.join(path, "label.csv")
    if os.path.exists(csv_path) and start_index > 0:
        label_df = pd.read_csv(csv_path)
    else:
        label_df = pd.DataFrame(columns=['name', 'label', 'num_circles', 'pattern_frequency', 'is_spiral', 'illusion_strength'])
    
    save_interval = 50
    
    for i in tqdm(range(size)):
        current_index = start_index + i
        
        # Decide whether this is a true spiral or concentric circles (illusion)
        label = int(np.random.rand() < positive_ratio)
        num_circles = np.random.randint(5, 12)
        pattern_frequency = np.random.randint(8, 24)  # Higher values = more segments
        
        # Calculate illusion strength
        # According to illusion_strength.md, Fraser spiral illusion strength is based on 
        # how much the perceived spiral deviates from actual circles
        # The strength depends on pattern_frequency and segment arrangement
        illusion_strength = 0.0
        
        if label:
            # For concentric circles with twisted segments (the illusion)
            # Calculate illusion strength based on pattern frequency, which determines
            # how much the segments create the spiral impression
            # Optimal illusion occurs around pattern_frequency of 15-18
            # Map this to a normalized strength with peak at optimal frequency
            optimal_freq = 16.5
            # Create a curve that peaks at optimal frequency
            illusion_strength = 1.0 - abs(pattern_frequency - optimal_freq) / 8.5
            illusion_strength = max(0.0, min(1.0, illusion_strength))
        else:
            # No illusion for actual spirals (they are what they appear to be)
            illusion_strength = 0.0
        
        fig = plt.figure(figsize=(4, 4), facecolor='white', dpi=128)
        ax = fig.add_subplot(111, aspect='equal')
        
        # Set up the figure
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        
        # Generate either concentric circles or a spiral
        if label:
            # Concentric circles with twisted segments (the illusion)
            for r in np.linspace(0.2, 1.0, num_circles):
                theta = np.linspace(0, 2 * np.pi, 500)
                
                # Add the "twisted cord" pattern that creates the illusion
                pattern = np.sin(theta * pattern_frequency) * 0.03
                
                # Apply the pattern to the radius
                x = r * np.cos(theta) + pattern * np.cos(theta + np.pi/2)
                y = r * np.sin(theta) + pattern * np.sin(theta + np.pi/2)
                
                # Alternate black and white segments
                segments = np.linspace(0, len(theta), pattern_frequency*2+1).astype(int)
                for j in range(len(segments)-1):
                    start, end = segments[j], segments[j+1]
                    color = 'black' if j % 2 == 0 else 'white'
                    ax.plot(x[start:end], y[start:end], color=color, linewidth=2)
                    
            is_spiral = False
            
        else:
            # Actual spiral
            turns = np.random.uniform(2.0, 3.5)  # Number of spiral turns
            theta = np.linspace(0, 2 * np.pi * turns, 1000)
            
            # Create a spiral with increasing radius
            growth_rate = 1.0 / (2 * np.pi * turns)
            r = np.linspace(0.2, 1.0, len(theta))
            
            # Apply the spiral equation
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Add the twisted cord pattern to the spiral too
            pattern = np.sin(theta * pattern_frequency/turns) * 0.03
            x = x + pattern * np.cos(theta + np.pi/2)
            y = y + pattern * np.sin(theta + np.pi/2)
            
            # Draw spiral with segments
            segments = np.linspace(0, len(theta), pattern_frequency*4+1).astype(int)
            for j in range(len(segments)-1):
                start, end = segments[j], segments[j+1]
                color = 'black' if j % 2 == 0 else 'white'
                ax.plot(x[start:end], y[start:end], color=color, linewidth=2)
                
            is_spiral = True
        
        # Hide axes
        ax.set_axis_off()
        
        # Save image
        name = f'fraser_spiral{current_index}.png'
        fig.savefig(os.path.join(path, name))
        
        # Close the figure
        plt.close(fig)
        
        # Add to dataframe
        label_df.loc[len(label_df)] = [name, label, num_circles, pattern_frequency, is_spiral, illusion_strength]
        
        # Save progress periodically
        if (i + 1) % save_interval == 0 or i == size - 1:
            label_df.to_csv(csv_path, index=False)
    
    # Final save
    label_df.to_csv(csv_path, index=False)

# KANIZSA TRIANGLE ILLUSION
@checkpoint_generation
def dataset11(path, size, positive_ratio, start_index=0):
    """
    Generate Kanizsa Triangle Illusion dataset.

    Args:
        path (str): Directory to save images and label file.
        size (int): Number of samples to generate.
        positive_ratio (float): Probability of generating a positive (visible triangle) sample.
        start_index (int): Starting index for generation (for resuming).

    Each image contains three pac-man shapes arranged to create an illusory triangle
    or randomly positioned to not create the illusion.
    """
    # Initialize dataframe or load existing one
    csv_path = os.path.join(path, "label.csv")
    if os.path.exists(csv_path) and start_index > 0:
        label_df = pd.read_csv(csv_path)
    else:
        label_df = pd.DataFrame(columns=['name', 'label', 'triangle_size', 'rotation', 'pacman_size', 'illusion_strength'])
    
    save_interval = 50
    
    for i in tqdm(range(size)):
        current_index = start_index + i
        
        # Decide whether this creates a triangle illusion or not
        label = int(np.random.rand() < positive_ratio)
        
        fig = plt.figure(figsize=(4, 4), facecolor='white', dpi=128)
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        
        # General parameters
        triangle_size = np.random.uniform(0.5, 0.8)  # Size of the illusory triangle
        rotation = np.random.uniform(0, 2*np.pi)  # Overall rotation of the arrangement
        pacman_size = np.random.uniform(0.15, 0.25)  # Size of the pacman inducers
        
        # Calculate normalized illusion strength
        # According to illusion_strength.md, Kanizsa illusion strength is measured by
        # the perceived contour clarity relative to real contours
        illusion_strength = 0.0
        
        if label:
            # Calculate illusion strength based on optimal parameters
            # The strength depends on pacman size, triangle size, and arrangement
            
            # 1. Optimal pacman size is around 0.2 (neither too small nor too large)
            pacman_factor = 1.0 - abs(pacman_size - 0.2) * 10.0
            pacman_factor = max(0.0, min(1.0, pacman_factor))
            
            # 2. Optimal triangle size factor - larger triangles (up to a point) create stronger illusions
            # with maximum around 0.7-0.75
            triangle_factor = 1.0 - abs(triangle_size - 0.72) * 4.0
            triangle_factor = max(0.0, min(1.0, triangle_factor))
            
            # Combine factors - multiply so if either is poor, the illusion is weak
            illusion_strength = pacman_factor * triangle_factor
        else:
            # No illusion for random pacman arrangement
            illusion_strength = 0.0
            
        if label:
            # Create the Kanizsa triangle illusion
            # First, compute the vertices of an equilateral triangle
            angles = np.array([0, 2*np.pi/3, 4*np.pi/3]) + rotation
            vertices = np.array([
                [np.cos(angle) * triangle_size, np.sin(angle) * triangle_size]
                for angle in angles
            ])
            
            # For each vertex, draw a pacman (circle with a sector removed)
            # The mouth of the pacman should face toward the center of the triangle
            for idx, vertex in enumerate(vertices):
                # Calculate angle to center
                angle_to_center = np.arctan2(-vertex[1], -vertex[0])
                
                # Draw the pacman
                pacman = patches.Wedge(
                    vertex, pacman_size, 
                    np.degrees(angle_to_center) - 45, 
                    np.degrees(angle_to_center) + 45, 
                    fc='black'
                )
                ax.add_patch(pacman)
            
        else:
            # Random arrangement that does not create the triangle illusion
            # Place pacman shapes randomly with random orientations
            for _ in range(3):
                # Random position
                pos = np.random.uniform(-0.8, 0.8, 2)
                
                # Random orientation
                orientation = np.random.uniform(0, 360)
                
                # Draw the pacman
                pacman = patches.Wedge(
                    pos, pacman_size, 
                    orientation - 45, 
                    orientation + 45, 
                    fc='black'
                )
                ax.add_patch(pacman)
        
        # Hide axes
        ax.set_axis_off()
        
        # Save image
        name = f'kanizsa_triangle{current_index}.png'
        fig.savefig(os.path.join(path, name))
        
        # Close the figure
        plt.close(fig)
        
        # Add to dataframe
        label_df.loc[len(label_df)] = [name, label, triangle_size, rotation, pacman_size, illusion_strength]
        
        # Save progress periodically
        if (i + 1) % save_interval == 0 or i == size - 1:
            label_df.to_csv(csv_path, index=False)
    
    # Final save
    label_df.to_csv(csv_path, index=False)

# ROTATING SNAKES ILLUSION
@checkpoint_generation
def dataset12(path, size, positive_ratio, start_index=0):
    """
    Generate Rotating Snakes Illusion dataset.

    Args:
        path (str): Directory to save images and label file.
        size (int): Number of samples to generate.
        positive_ratio (float): Probability of generating a positive (illusion) sample.
        start_index (int): Starting index for generation (for resuming).

    Each image contains circular patterns that create an illusion of rotation due to
    specific color gradients and placement.
    """
    # Initialize dataframe or load existing one
    csv_path = os.path.join(path, "label.csv")
    if os.path.exists(csv_path) and start_index > 0:
        label_df = pd.read_csv(csv_path)
    else:
        label_df = pd.DataFrame(columns=['name', 'label', 'num_circles', 'segments', 'direction', 'illusion_strength'])
    
    save_interval = 50
    
    for i in tqdm(range(size)):
        current_index = start_index + i
        
        # Decide if this is an illusion sample or a control sample
        label = int(np.random.rand() < positive_ratio)
        
        fig = plt.figure(figsize=(4, 4), facecolor='white', dpi=128)
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        
        # General parameters
        num_circles = np.random.randint(3, 8)  # Number of circle rings
        segments = np.random.randint(8, 16)  # Number of segments per circle
        direction = 1 if np.random.rand() < 0.5 else -1  # Clockwise or counter-clockwise
        
        # Calculate normalized illusion strength
        # According to illusion_strength.md, Rotating Snakes illusion strength is measured by
        # equivalent real motion speed or proportion of time motion is seen
        illusion_strength = 0.0
        
        if label:
            # Calculate illusion strength based on optimal parameters for motion perception
            
            # 1. Optimal segmentation is around 12 segments (not too few or too many)
            segment_factor = 1.0 - abs(segments - 12) / 12.0
            segment_factor = max(0.0, min(1.0, segment_factor))
            
            # 2. Optimal number of circles is around 5-6
            circle_factor = 1.0 - abs(num_circles - 5.5) / 5.5
            circle_factor = max(0.0, min(1.0, circle_factor))
            
            # Combine factors
            illusion_strength = segment_factor * circle_factor
            
            # Use a specific sequence of colors that creates the motion illusion
            colors = ['black', 'dimgray', 'white', 'lightgray']
        else:
            # No illusion in control sample
            illusion_strength = 0.0
            # No illusion colors - uniform or random ordering won't create illusion
            colors = np.random.choice(['gray', 'darkgray', 'silver', 'gainsboro'], 4, replace=False)
        
        # Create multiple circular patterns at different positions
        centers = []
        sizes = []
        
        # Add a central circle
        centers.append([0, 0])
        sizes.append(np.random.uniform(0.3, 0.5))
        
        # Add satellite circles
        for _ in range(np.random.randint(2, 6)):
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.uniform(0.4, 0.9)
            centers.append([np.cos(angle) * dist, np.sin(angle) * dist])
            sizes.append(np.random.uniform(0.15, 0.3))
        
        # Draw each circle
        for center, size in zip(centers, sizes):
            for r_idx in range(num_circles):
                radius = size * (1 - r_idx / num_circles)
                
                # Draw segments of the circle
                for s_idx in range(segments):
                    start_angle = s_idx * 360 / segments
                    end_angle = (s_idx + 1) * 360 / segments
                    
                    # Choose color based on segment position - creates the illusion
                    color_idx = (s_idx * direction) % len(colors)
                    
                    # Draw wedge
                    wedge = patches.Wedge(
                        center, radius,
                        start_angle, end_angle,
                        width=size/num_circles,
                        fc=colors[color_idx]
                    )
                    ax.add_patch(wedge)
        
        # Hide axes
        ax.set_axis_off()
        
        # Save image
        name = f'rotating_snakes{current_index}.png'
        fig.savefig(os.path.join(path, name))
        
        # Close the figure
        plt.close(fig)
        
        # Add to dataframe
        label_df.loc[len(label_df)] = [name, label, num_circles, segments, direction, illusion_strength]
        
        # Save progress periodically
        if (i + 1) % save_interval == 0 or i == size - 1:
            label_df.to_csv(csv_path, index=False)
    
    # Final save
    label_df.to_csv(csv_path, index=False)

# EBBINGHAUS ILLUSION
@checkpoint_generation
def dataset13(path, size, positive_ratio, start_index=0):
    """
    Generate Ebbinghaus Illusion dataset.

    Args:
        path (str): Directory to save images and label file.
        size (int): Number of samples to generate.
        positive_ratio (float): Probability of generating a positive (same size) sample.
        start_index (int): Starting index for generation (for resuming).

    Each image contains two central circles surrounded by other circles, where the central
    circles may appear different sizes due to the surrounding context.
    """
    # Initialize dataframe or load existing one
    csv_path = os.path.join(path, "label.csv")
    if os.path.exists(csv_path) and start_index > 0:
        label_df = pd.read_csv(csv_path)
    else:
        label_df = pd.DataFrame(columns=['name', 'label', 'center1_size', 'center2_size', 
                                        'surround1_size', 'surround2_size', 
                                        'surround1_distance', 'surround2_distance',
                                        'illusion_strength'])
    
    save_interval = 50
    
    for i in tqdm(range(size)):
        current_index = start_index + i
        
        # Decide whether this is an illusion sample or same-size sample
        label = int(np.random.rand() < positive_ratio)
        
        fig = plt.figure(figsize=(4, 4), facecolor='white', dpi=128)
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        
        # Position of the two central targets
        target1_pos = [-0.5, 0]
        target2_pos = [0.5, 0]
        
        # Surrounding circle parameters
        num_surrounds = np.random.randint(6, 9)  # Number of surrounding circles
        
        # Determine central circle sizes
        # For illusion, they're the same size but will appear different
        # For non-illusion (label=0), they're actually different sizes
        base_size = np.random.uniform(0.15, 0.2)
        if label:
            # Positive case: same size
            center1_size = base_size
            center2_size = base_size
        else:
            # Negative case: different sizes (about 10-20% difference)
            size_diff = np.random.uniform(0.1, 0.2) * base_size
            # Randomly decide which is larger
            if np.random.rand() < 0.5:
                center1_size = base_size - size_diff/2
                center2_size = base_size + size_diff/2
            else:
                center1_size = base_size + size_diff/2
                center2_size = base_size - size_diff/2
        
        # Determine surrounding circle parameters
        # For the illusion, small surrounds make center look larger and vice versa
        surround1_size = np.random.uniform(0.06, 0.1)  # Small surrounding circles
        surround2_size = np.random.uniform(0.25, 0.3)  # Large surrounding circles
        
        # Distance of surrounding circles from center
        surround1_distance = center1_size + surround1_size * 1.2
        surround2_distance = center2_size + surround2_size * 1.2
        
        # Calculate normalized illusion strength
        # According to illusion_strength.md, Ebbinghaus illusion strength is measured by 
        # the percentage size difference at the point of subjective equality (PSE)
        # Maximum effect is typically around 10%
        illusion_strength = 0.0
        
        if label:  # Same size centers but may appear different
            # Calculate illusion strength based on surrounds size difference
            # The greater the difference in surrounding sizes, the stronger the illusion
            surrounds_ratio = max(surround1_size, surround2_size) / min(surround1_size, surround2_size)
            
            # Normalize to a 0-1 scale where typical max effect (surrounds_ratio ≈ 4-5) maps to illusion_strength ≈ 1.0
            # A surrounds_ratio of 1 would give no illusion (strength = 0)
            illusion_strength = min(1.0, (surrounds_ratio - 1) / 4.0)
        else:
            # No illusion for physically different sized centers
            illusion_strength = 0.0
        
        # Draw the central target circles
        central1 = patches.Circle(target1_pos, center1_size, fc='black')
        central2 = patches.Circle(target2_pos, center2_size, fc='black')
        ax.add_patch(central1)
        ax.add_patch(central2)
        
        # Draw surrounding circles for first target
        for j in range(num_surrounds):
            angle = j * 2 * np.pi / num_surrounds
            x = target1_pos[0] + np.cos(angle) * surround1_distance
            y = target1_pos[1] + np.sin(angle) * surround1_distance
            circle = patches.Circle([x, y], surround1_size, fc='black')
            ax.add_patch(circle)
            
        # Draw surrounding circles for second target
        for j in range(num_surrounds):
            angle = j * 2 * np.pi / num_surrounds
            x = target2_pos[0] + np.cos(angle) * surround2_distance
            y = target2_pos[1] + np.sin(angle) * surround2_distance
            circle = patches.Circle([x, y], surround2_size, fc='black')
            ax.add_patch(circle)
        
        # Hide axes
        ax.set_axis_off()
        
        # Save image
        name = f'ebbinghaus{current_index}.png'
        fig.savefig(os.path.join(path, name))
        
        # Close the figure
        plt.close(fig)
        
        # Add to dataframe
        label_df.loc[len(label_df)] = [name, label, center1_size, center2_size, 
                                    surround1_size, surround2_size, 
                                    surround1_distance, surround2_distance,
                                    illusion_strength]
        
        # Save progress periodically
        if (i + 1) % save_interval == 0 or i == size - 1:
            label_df.to_csv(csv_path, index=False)
    
    # Final save
    label_df.to_csv(csv_path, index=False)

# DELBOEUF ILLUSION
@checkpoint_generation
def dataset14(path, size, positive_ratio, start_index=0):
    """
    Generate Delboeuf Illusion dataset.

    Args:
        path (str): Directory to save images and label file.
        size (int): Number of samples to generate.
        positive_ratio (float): Probability of generating a positive (same size) sample.
        start_index (int): Starting index for generation (for resuming).

    Each image contains two central circles each surrounded by a ring, where the central
    circles may appear different sizes due to the surrounding rings.
    """
    # Initialize dataframe or load existing one
    csv_path = os.path.join(path, "label.csv")
    if os.path.exists(csv_path) and start_index > 0:
        label_df = pd.read_csv(csv_path)
    else:
        label_df = pd.DataFrame(columns=['name', 'label', 'center1_size', 'center2_size', 
                                        'ring1_size', 'ring2_size', 'illusion_strength'])
    
    save_interval = 50
    
    for i in tqdm(range(size)):
        current_index = start_index + i
        
        # Decide whether this is an illusion sample or same-size sample
        label = int(np.random.rand() < positive_ratio)
        
        fig = plt.figure(figsize=(4, 4), facecolor='white', dpi=128)
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        
        # Position of the two circular targets
        target1_pos = [-0.5, 0]
        target2_pos = [0.5, 0]
        
        # Determine central circle sizes
        base_size = np.random.uniform(0.15, 0.25)
        if label:
            # Positive case: same size
            center1_size = base_size
            center2_size = base_size
        else:
            # Negative case: different sizes (about 10-20% difference)
            size_diff = np.random.uniform(0.1, 0.2) * base_size
            # Randomly decide which is larger
            if np.random.rand() < 0.5:
                center1_size = base_size - size_diff/2
                center2_size = base_size + size_diff/2
            else:
                center1_size = base_size + size_diff/2
                center2_size = base_size - size_diff/2
        
        # Determine surrounding ring sizes
        # For the illusion, smaller ring makes center look larger and vice versa
        ring1_ratio = np.random.uniform(1.2, 1.5)  # Smaller surrounding ring
        ring2_ratio = np.random.uniform(2.0, 2.5)  # Larger surrounding ring
        
        ring1_size = center1_size * ring1_ratio
        ring2_size = center2_size * ring2_ratio
        
        # Calculate normalized illusion strength
        # According to illusion_strength.md, Delboeuf illusion strength is measured by
        # the percentage diameter difference at the point of subjective equality
        # A maximum effect is typically around 15%
        illusion_strength = 0.0
        
        if label:  # Same size centers but may appear different
            # Calculate illusion strength based on ring size ratio difference
            # The greater the difference in ring ratios, the stronger the illusion
            # Maximum effect when one ring is close to the circle and one is far
            ring_ratio_diff = abs(ring1_ratio - ring2_ratio)
            
            # Normalize to 0-1 scale where maximum typical effect (ratio diff of ~1.3) maps to 1.0
            illusion_strength = min(1.0, ring_ratio_diff / 1.3)
        else:
            # No illusion for physically different sized centers
            illusion_strength = 0.0
        
        # Draw the surrounding rings first
        ring1 = patches.Circle(target1_pos, ring1_size, fc='none', ec='black', lw=2)
        ring2 = patches.Circle(target2_pos, ring2_size, fc='none', ec='black', lw=2)
        ax.add_patch(ring1)
        ax.add_patch(ring2)
        
        # Draw the central target circles
        central1 = patches.Circle(target1_pos, center1_size, fc='black')
        central2 = patches.Circle(target2_pos, center2_size, fc='black')
        ax.add_patch(central1)
        ax.add_patch(central2)
        
        # Hide axes
        ax.set_axis_off()
        
        # Save image
        name = f'delboeuf{current_index}.png'
        fig.savefig(os.path.join(path, name))
        
        # Close the figure
        plt.close(fig)
        
        # Add to dataframe
        label_df.loc[len(label_df)] = [name, label, center1_size, center2_size, 
                                    ring1_size, ring2_size, illusion_strength]
        
        # Save progress periodically
        if (i + 1) % save_interval == 0 or i == size - 1:
            label_df.to_csv(csv_path, index=False)
    
    # Final save
    label_df.to_csv(csv_path, index=False)

