import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch.nn.functional as F
from data_utils.BRANCHdataset import CustomPLYDataset
from models.pointnet2_part_seg_msg_new import PointNetSegmentation 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

visualize = False
visualize_conf_matrix_on_data = True

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True  # For reproducibility in convolutional layers
    torch.backends.cudnn.benchmark = False     # Disables the auto-tuner for convolution layers

set_random_seed(1)

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate metrics for positive class (1)
    precision_pos = precision_score(y_true, y_pred, average='binary', pos_label=1)
    recall_pos = recall_score(y_true, y_pred, average='binary', pos_label=1)
    f1_pos = f1_score(y_true, y_pred, average='binary', pos_label=1)
    
    # Calculate metrics for negative class (0)
    precision_neg = precision_score(y_true, y_pred, average='binary', pos_label=0)
    recall_neg = recall_score(y_true, y_pred, average='binary', pos_label=0)
    f1_neg = f1_score(y_true, y_pred, average='binary', pos_label=0)

    return accuracy, precision_pos, recall_pos, f1_pos, precision_neg, recall_neg, f1_neg

# Load model and checkpoint
def load_model(checkpoint_path, input_channels):
    model = PointNetSegmentation(normal_channel=False, num_classes=2) 
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

# Evaluation function
def evaluate_model(model, test_loader):
    y_true = []
    y_pred = []
    all_data_orig = []
    all_ply_paths = []
    
    with torch.no_grad():  # Disable gradients during evaluation
        for data, labels, plyPath in test_loader:
            data_orig = data.cpu().numpy()
            data, labels = data.to(device), labels.to(device)
            all_data_orig.append(data_orig)
            points = data.float().cuda()
            target = labels.long().cuda()
            points = points.transpose(2, 1)
            outputs, _ = model(points)
            seg_pred = outputs.contiguous().view(-1, 2)
            target = target.view(-1, 1)[:, 0]  # Flatten target labels
            pred_choice = seg_pred.data.max(1)[1]  # Get the predicted class
            pred_choice_numpy = pred_choice.cpu().detach().numpy()
            if visualize:
                batch_size, num_points, _ = data_orig.shape
                visualize_point_clouds(data_orig, pred_choice_numpy, num_points, batch_size, plyPath)
            if visualize_conf_matrix_on_data:
                visualize_classification_results(data_orig, target.cpu().numpy(), pred_choice.cpu().numpy())
            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred_choice.cpu().numpy())


    accuracy, precision_pos, recall_pos, f1_pos, precision_neg, recall_neg, f1_neg = calculate_metrics(y_true, y_pred)

    conf_matrix = confusion_matrix(y_true, y_pred)

    return accuracy, precision_pos, recall_pos, f1_pos, precision_neg, recall_neg, f1_neg, conf_matrix


def visualize_classification_results(data_orig, y_true, y_pred, batch_size=4, num_points=10000):
    """
    Visualize true positives (TP), true negatives (TN), false positives (FP), 
    and false negatives (FN) on the point cloud for each batch.

    Args:
    - data_orig (numpy array): The original point cloud of shape (batch_size, num_points, 3 or 6)
    - y_true (numpy array): Ground truth labels of shape (batch_size * num_points,)
    - y_pred (numpy array): Predicted labels of shape (batch_size * num_points,)
    - batch_size (int): Number of point clouds in the batch
    - num_points (int): Number of points per point cloud
    """
    for i in range(batch_size):
        # Extract the current point cloud, predictions, and true labels
        point_cloud = data_orig[i]
        y_true_i = y_true[i * num_points: (i + 1) * num_points]
        y_pred_i = y_pred[i * num_points: (i + 1) * num_points]


        # Initialize the color array (default is black for all points)
        colors = np.zeros((num_points, 3))

        # True Positives (TP): Predicted 1, Actual 1 -> Green
        tp_indices = np.where((y_pred_i == 1) & (y_true_i == 1))[0]
        colors[tp_indices] = [0, 1, 0]  # Green

        # True Negatives (TN): Predicted 0, Actual 0 -> Gray
        tn_indices = np.where((y_pred_i == 0) & (y_true_i == 0))[0]
        colors[tn_indices] = [0.5, 0.5, 0.5]  # Gray

        # False Positives (FP): Predicted 1, Actual 0 -> Red
        fp_indices = np.where((y_pred_i == 1) & (y_true_i == 0))[0]
        colors[fp_indices] = [1, 0, 0]  # Red

        # False Negatives (FN): Predicted 0, Actual 1 -> Blue
        fn_indices = np.where((y_pred_i == 0) & (y_true_i == 1))[0]
        colors[fn_indices] = [0, 0, 1]  # Blue

        # Create an Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        
        # If point_cloud has 6 channels (XYZ + normals), extract XYZ for visualization
        if point_cloud.shape[1] == 6:
            points = point_cloud[:, :3]  # Use only XYZ coordinates
        else:
            points = point_cloud  # Assume already XYZ coordinates
        
        # Set the points and colors for the point cloud
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Visualize the point cloud
        print(f"Visualizing point cloud {i+1}/{batch_size}")
        print("Precision: " + str(precision_score(y_true_i, y_pred_i)))
        print("Recall: " + str(recall_score(y_true_i, y_pred_i)))
        print("F1 score: " + str(f1_score(y_true_i, y_pred_i)))
        o3d.visualization.draw_geometries([pcd], window_name=f"Classification Results - Point Cloud {i+1}")



def run_evaluation():
    # Settings
    root = 'data/BRANCH_Dataset/ply_data'
    test_dataset = CustomPLYDataset(root_dir=root, npoints=10000, split='test', normal_channel=False)  # Define your dataset class here
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    input_channels = 3 
    checkpoint_path = "log/point_classification/pointnet2_part_seg_msg/checkpoints/best_model.pth"  # Update with the actual path
    model = load_model(checkpoint_path, input_channels)

    # Run evaluation
    accuracy, precision_pos, recall_pos, f1_pos, precision_neg, recall_neg, f1_neg, conf_matrix = evaluate_model(model, test_loader)
    
    print(f"Accuracy: {accuracy:.4f}")

    print(f"Precision_pos: {precision_pos:.4f}")
    print(f"Recall_pos: {recall_pos:.4f}")
    print(f"F1 Score_pos: {f1_pos:.4f}")

    print(f"Precision_neg: {precision_neg:.4f}")
    print(f"Recall_neg: {recall_neg:.4f}")
    print(f"F1 Score_neg: {f1_neg:.4f}")

    # Display confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)

    #conf_matrix = np.array([[63325, 3790], [4143, 8742]])
        # Visualize confusion matrix using matplotlib
    plt.figure(figsize=(5, 5))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matrica zabune')
    plt.colorbar()

    tick_marks = np.arange(2)  # Assuming binary classification, update if needed for more classes
    plt.xticks(tick_marks, ["Neorezane", "Orezane"], rotation=45)
    plt.yticks(tick_marks, ["Neorezane", "Orezane"])

    # Annotate each cell with the number of cases
    thresh = conf_matrix.max() / 2.  # Threshold for text color (white or black)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Istinita oznaka')
    plt.xlabel('Predviđena oznaka')
    plt.show()

def visualize_point_clouds(points_orig, pred_choice_numpy, num_points, batch_size, ply_path):
    for i in range(batch_size):

        point_cloud = points_orig[i]
        pred_choice_i = pred_choice_numpy[i * num_points: (i + 1) * num_points]

        # Map predictions to colors
        color_map = {0: [1, 0, 0],  # Red for class 0
                    1: [0, 1, 0]}  # Green for class 1

        colors = np.array([color_map[class_label] for class_label in pred_choice_i])

        # Create Open3D point cloud with colors
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_orig = o3d.io.read_point_cloud(ply_path[i])
        pcd_orig = modify_point_cloud_colors(pcd_orig)
        o3d.visualization.draw_geometries([pcd_orig], window_name="Istinite oznake")
        o3d.visualization.draw_geometries([pcd], window_name="Predviđene oznake")

def modify_point_cloud_colors(pcd):
    # Convert point cloud colors to numpy array
    colors = np.asarray(pcd.colors)

    # Define color mappings
    color_map = {
        (1.0, 0.0, 0.0): (1.0, 0.0, 0.0),  # Red stays Red
        (0.0, 0.0, 1.0): (1.0, 0.0, 0.0),  # Blue turns into Red
        (1.0, 1.0, 0.0): (0.0, 1.0, 0.0)   # Yellow turns into Green
    }

    # Apply color mapping
    for i, color in enumerate(colors):
        color_tuple = tuple(color)
        if color_tuple in color_map:
            colors[i] = color_map[color_tuple]

    # Update point cloud colors
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_evaluation()
