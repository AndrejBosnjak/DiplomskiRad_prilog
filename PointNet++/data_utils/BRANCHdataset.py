import os
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class CustomPLYDataset(Dataset):
    def __init__(self, root_dir, npoints=2500, split='train', normal_channel=False):
        self.root_dir = root_dir
        self.npoints = npoints
        self.normal_channel = normal_channel
        self.file_paths = []

        # Load file paths for the dataset split (train/test/val)
        if split == 'train':
            split_dir = os.path.join(self.root_dir, 'train')
        elif split == 'test':
            split_dir = os.path.join(self.root_dir, 'test')
        elif split == 'val':
            split_dir = os.path.join(self.root_dir, 'val')
        elif split == 'trainval':
            train_dir = os.path.join(self.root_dir, 'train')
            val_dir = os.path.join(self.root_dir, 'val')
            self.file_paths = [os.path.join(train_dir, fname) for fname in os.listdir(train_dir) if fname.endswith('.ply')] + \
                              [os.path.join(val_dir, fname) for fname in os.listdir(val_dir) if fname.endswith('.ply')]
            return
        else:
            raise ValueError(f"Invalid split: {split}")

        self.file_paths = [os.path.join(split_dir, fname) for fname in os.listdir(split_dir) if fname.endswith('.ply')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        ply_path = self.file_paths[idx]

        # Load the .ply file
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)  # Nx3 array of xyz coordinates
        colors = np.asarray(pcd.colors)  # Nx3 array of RGB values

        # Assign labels based on colors
        labels = self.color_to_label(colors)

        # Normalize the points
        points = pc_normalize(points)

        # Resample points to a fixed size (self.npoints)
        choice = np.random.choice(len(points), self.npoints, replace=True)
        points = points[choice, :]
        labels = labels[choice]

        if self.normal_channel:
            normals = np.asarray(pcd.normals)
            if normals.size > 0:  # If normals exist in the point cloud
                normals = normals[choice, :]
                points = np.hstack((points, normals))  # Concatenate normals if required
            else:
                raise ValueError("Normal channel requested but normals not found in point cloud")

        return points, labels, ply_path

    def color_to_label(self, colors):
        """
        Convert the RGB color values into corresponding labels.
        """
        labels = np.zeros((colors.shape[0]), dtype=np.int32)
        
        red_blue_mask = (colors[:, 0] == 1) & (colors[:, 2] == 1)
        yellow_mask = (colors[:, 0] == 1) & (colors[:, 1] == 1) 
        
        labels[red_blue_mask] = 0  # Red/blue is label 0
        labels[yellow_mask] = 1  # Yellow is label 1
        
        return labels
