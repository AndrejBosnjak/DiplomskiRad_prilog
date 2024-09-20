import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
from pathlib import Path
from tqdm import tqdm
from data_utils.BRANCHdataset import CustomPLYDataset
import open3d as o3d
from models.pointnet2_part_seg_msg_new import PointNetSegmentation
from models.pointnet2_part_seg_msg_new import PointNetSegmentationLoss
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

visualize_train = False
visualize_test = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if y.is_cuda:
        return new_y.cuda()
    return new_y

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_part_seg_msg_new', help='model name')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size during training')
    parser.add_argument('--epoch', default=100, type=int, help='Number of epochs to run')
    parser.add_argument('--learning_rate', default=0.005, type=float, help='Initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='Specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer: Adam or SGD')
    parser.add_argument('--log_dir', type=str, default="pointnet2_part_seg_msg", help='Log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--npoint', type=int, default=10000, help='Number of points in point cloud')
    parser.add_argument('--normal', action='store_true', default=False, help='Use normals in input')
    parser.add_argument('--step_size', type=int, default=20, help='Decay step for learning rate decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Decay rate for learning rate decay')
    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        #print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('point_classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETERS...')
    log_string(args)

    '''DATASET LOADING'''
    root = 'data/BRANCH_Dataset/ply_data'

    num_cpu_cores = os.cpu_count()

    TRAIN_DATASET = CustomPLYDataset(root_dir=root, npoints=args.npoint, split='train', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=num_cpu_cores, drop_last=True)
    VAL_DATASET = CustomPLYDataset(root_dir=root, npoints=args.npoint, split='val', normal_channel=args.normal)
    valDataLoader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=num_cpu_cores)
    log_string(f"The number of training data is: {len(TRAIN_DATASET)}")
    log_string(f"The number of val data is: {len(VAL_DATASET)}")

    num_classes = 2  # Two classes: pruned and not pruned

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    classifier = MODEL.PointNetSegmentation(num_classes, normal_channel=args.normal).cuda()
    criterion = MODEL.PointNetSegmentationLoss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Using pre-trained model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    '''TRAINING'''
    LEARNING_RATE_CLIP = 1e-5
    best_acc = 0

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []
        total_loss = 0  # Initialize total_loss for the epoch
        num_batches = len(trainDataLoader)
        
        log_string(f'Epoch {epoch+1}/{args.epoch}:')

        classifier = classifier.train()

        ''' Train one epoch '''
        for points, target, ply_path in tqdm(trainDataLoader, total=num_batches):
            optimizer.zero_grad()
            points_orig = points.cpu().numpy()
            points = points.float().cuda()
            target = target.long().cuda()
            points = points.transpose(2, 1)  # [B, C, N]

            # Predict for each point
            seg_pred, _ = classifier(points)  # Predict for each point
            seg_pred = seg_pred.contiguous().view(-1, num_classes)
            target = target.view(-1, 1)[:, 0]  # Flatten target labels
            pred_choice = seg_pred.data.max(1)[1]  # Get the predicted class

            if visualize_train:
                batch_size, num_points, _ = points_orig.shape
                pred_choice_numpy = pred_choice.cpu().detach().numpy()
                visualize_point_clouds(points_orig, pred_choice_numpy, num_points, batch_size, ply_path)

            # Compute the accuracy
            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))

            # Compute loss
            loss = criterion(seg_pred, target)
            total_loss += loss.item()  # Accumulate loss
            loss.backward()
            optimizer.step()

        # Compute the average loss for the epoch
        avg_loss = total_loss / num_batches

        # Log the mean accuracy and average loss for the epoch
        train_instance_acc = np.mean(mean_correct)
        log_string(f'Train accuracy: {train_instance_acc:.5f}')
        log_string(f'Average Loss: {avg_loss:.5f}')

        '''VALIDATION'''
        with torch.no_grad():
            total_correct = 0
            total_seen = 0
            total_loss = 0.0  # To accumulate the total loss
            all_targets = []  # List to accumulate true labels
            all_predictions = []  # List to accumulate predicted labels
            epoch_recall = []
            epoch_precision = []
            epoch_f1 = []
            predictions_and_labels = []  # List to accumulate predictions and labels             
            classifier = classifier.eval()

            for points, target, ply_path in tqdm(valDataLoader, total=len(valDataLoader)):
                points_orig = points.cpu().numpy()
                points = points.float().cuda()
                target = target.long().cuda()
                points = points.transpose(2, 1)
                print(len(target))

                seg_pred, _ = classifier(points)
                seg_pred = seg_pred.contiguous().view(-1, num_classes)
                pred_choice = seg_pred.data.max(1)[1]
                target = target.view(-1, 1)[:, 0]

                # Compute loss
                loss = criterion(seg_pred, target)
                total_loss += loss.item() 

                # Collect predictions and targets for metrics calculation
                pred_choice_numpy = pred_choice.cpu().detach().numpy()
                target_numpy = target.cpu().detach().numpy()

                all_targets.extend(target_numpy)
                all_predictions.extend(pred_choice_numpy)

                # Temporary lists to hold precision, recall, and F1 for each point cloud
                precision_list = []
                recall_list = []
                f1_list = []

                for i in range(len(ply_path)):
                    filepath = ply_path[i]
                    predictions = pred_choice_numpy[i * args.npoint:(i + 1) * args.npoint]
                    true_labels = target_numpy[i * args.npoint:(i + 1) * args.npoint]
                    
                    # Calculate metrics for the individual point cloud
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        true_labels, predictions, average='binary', zero_division=0)
                    
                    precision_list.append(precision)
                    recall_list.append(recall)
                    f1_list.append(f1)

                    # Calculate confusion matrix for the individual point cloud
                    confusion_mat = confusion_matrix(true_labels, predictions)
                    TN, FP, FN, TP = confusion_mat.ravel() if confusion_mat.size == 4 else (0, 0, 0, 0)

                    log_string(f'ply_file: "{filepath}", TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')

                    # Only collect during the validation process
                    predictions_and_labels.append((filepath, predictions, true_labels))

                # Calculate average metrics for the batch
                avg_precision = np.mean(precision_list)
                avg_recall = np.mean(recall_list)
                avg_f1 = np.mean(f1_list)

                epoch_precision.append(avg_precision)
                epoch_recall.append(avg_recall)
                epoch_f1.append(avg_f1)

                log_string(f'Average Precision for batch: {avg_precision:.5f}')
                log_string(f'Average Recall for batch: {avg_recall:.5f}')
                log_string(f'Average F1 Score for batch: {avg_f1:.5f}')

                if visualize_test:
                    batch_size, num_points, _ = points_orig.shape
                    visualize_point_clouds(points_orig, pred_choice_numpy, num_points, batch_size, ply_path)

                correct = pred_choice.eq(target.data).cpu().sum()  # Computes element-wise equality
                total_correct += correct
                total_seen += target.numel()  # Returns the total number of elements 

            if total_seen == 0:
                print("Warning: Total seen is 0. Cannot calculate test accuracy.")
                test_acc = 0 
            else:
                test_acc = total_correct / float(total_seen)

            avg_loss = total_loss / num_batches

            log_string(f'Test accuracy: {test_acc:.5f}')
            log_string(f'Test loss: {avg_loss:.5f}')  # Log the average loss
            log_string(f'Average Precision: {np.mean(epoch_precision):.5f}')
            log_string(f'Average Recall: {np.mean(epoch_recall):.5f}')
            log_string(f'Average F1 Score: {np.mean(epoch_f1):.5f}')


            if test_acc > best_acc:
                best_acc = test_acc
                log_string('Saving best model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                state = {
                    'epoch': epoch,
                    'train_acc': train_instance_acc,
                    'test_acc': test_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string(f'Best Test Accuracy: {best_acc:.5f} at Epoch {epoch}')

                with open(exp_dir.joinpath('best_epoch_labels.txt'), 'w') as f:
                    for filepath, prediction, true_label in predictions_and_labels:
                        f.write(f'"{filepath}", {prediction.tolist()}\n')
                        f.write(f'True labels for "{filepath}": {true_label.tolist()}\n')


def visualize_point_clouds(points_orig, pred_choice_numpy, num_points, batch_size, ply_path):
    for i in range(batch_size):

        point_cloud = points_orig[i]
        pred_choice_i = pred_choice_numpy[i * num_points: (i + 1) * num_points]

        # Count class instances
        class_0_count = np.count_nonzero(pred_choice_i == 0)
        class_1_count = np.count_nonzero(pred_choice_i == 1)

        print(f"Point Cloud {i}: Class 0 count:", class_0_count)
        print(f"Point Cloud {i}: Class 1 count:", class_1_count)

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

if __name__ == '__main__':
    args = parse_args()
    main(args)
