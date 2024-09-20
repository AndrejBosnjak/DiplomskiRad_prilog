import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation

class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(PointNetSegmentation, self).__init__()
        
        input_channels = 6 if normal_channel else 3

        self.normal_channel = normal_channel
        
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512, radius_list=[0.1, 0.2, 0.4], nsample_list=[32, 64, 128],
            in_channel=input_channels, mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=128, radius_list=[0.4, 0.8], nsample_list=[64, 128],
            in_channel=128 + 128 + 64, mlp_list=[[128, 128, 256], [128, 196, 256]]
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True
        )
        # Feature Propagation layers
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + input_channels, mlp=[128, 128])

        # Convolutional layers for output
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
    def forward(self, xyz, cls_label=None):
        """
        Forward pass
        xyz: point cloud data of shape [B, C, N] where B is batch size, C is number of channels (coordinates), N is the number of points
        cls_label: (Optional) Class label for each point in training mode [B, 16] (one-hot vector) (not needed in inference mode)
        """
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]  # Use only the XYZ coordinates
        else:
            l0_points = xyz
            l0_xyz = xyz  # Full input as XYZ

        # Set abstraction layers
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Feature propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        
        # If class labels are provided, concatenate to the input (for training)
        if cls_label is not None:
            cls_label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)
            l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot, l0_xyz, l0_points], 1), l1_points)
        else:
            l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        feat = F.relu(self.bn1(self.conv1(l0_points)))
        feat = self.drop1(feat)
        x = self.conv2(feat)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)  # Permute to [B, N, num_classes] for per-point classification
        return x, l3_points


class PointNetSegmentationLoss(nn.Module):
    def __init__(self):
        super(PointNetSegmentationLoss, self).__init__()

    def forward(self, pred, target):
        """
        Compute the segmentation loss.
        pred: Predictions [B, N, num_classes]
        target: Ground truth labels [B, N] (class per point)
        """
        loss = F.nll_loss(pred, target)
        return loss
