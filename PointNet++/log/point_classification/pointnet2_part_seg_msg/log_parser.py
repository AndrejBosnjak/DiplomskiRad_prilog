import numpy as np
import re
import matplotlib.pyplot as plt

def parse_log_file(log_file_path):
    # Initialize lists to hold the values
    train_accuracy = []
    train_loss = []
    test_accuracy = []
    test_loss = []
    avg_precision = []
    avg_recall = []
    avg_f1_score = []

    # Regular expressions to match the required lines
    train_acc_re = re.compile(r'Train accuracy: (\d+\.\d+)')
    train_loss_re = re.compile(r'Average Loss: (\d+\.\d+)')
    test_acc_re = re.compile(r'Test accuracy: (\d+\.\d+)')
    test_loss_re = re.compile(r'Test loss: (\d+\.\d+)')
    avg_precision_re = re.compile(r'Average Precision: (\d+\.\d+)')
    avg_recall_re = re.compile(r'Average Recall: (\d+\.\d+)')
    avg_f1_score_re = re.compile(r'Average F1 Score: (\d+\.\d+)')

    # Read the log file
    with open(log_file_path, 'r') as file:
        for line in file:
            # Check for each pattern and append the value to the corresponding list
            train_acc_match = train_acc_re.search(line)
            if train_acc_match:
                train_accuracy.append(float(train_acc_match.group(1)))
                continue

            train_loss_match = train_loss_re.search(line)
            if train_loss_match:
                train_loss.append(float(train_loss_match.group(1)))
                continue

            test_acc_match = test_acc_re.search(line)
            if test_acc_match:
                test_accuracy.append(float(test_acc_match.group(1)))
                continue

            test_loss_match = test_loss_re.search(line)
            if test_loss_match:
                test_loss.append(float(test_loss_match.group(1)))
                continue

            avg_precision_match = avg_precision_re.search(line)
            if avg_precision_match:
                avg_precision.append(float(avg_precision_match.group(1)))
                continue

            avg_recall_match = avg_recall_re.search(line)
            if avg_recall_match:
                avg_recall.append(float(avg_recall_match.group(1)))
                continue

            avg_f1_score_match = avg_f1_score_re.search(line)
            if avg_f1_score_match:
                avg_f1_score.append(float(avg_f1_score_match.group(1)))
                continue

    # Convert lists to numpy arrays
    train_accuracy = np.array(train_accuracy)
    train_loss = np.array(train_loss)
    test_accuracy = np.array(test_accuracy)
    test_loss = np.array(test_loss)
    avg_precision = np.array(avg_precision)
    avg_recall = np.array(avg_recall)
    avg_f1_score = np.array(avg_f1_score)

    return (train_accuracy, train_loss, test_accuracy, test_loss,
            avg_precision, avg_recall, avg_f1_score)

def plot_metrics(train_accuracy, train_loss, test_accuracy, test_loss, avg_precision, avg_recall, avg_f1_score):
    # List of metrics and their corresponding titles
    metrics = [
        (train_accuracy, 'Točnost na skupu za treniranje', 'Epoha', 'Točnost', 'train_acc.png'),
        (train_loss, 'Vrijednost funkcije gubitka tijekom treniranja', 'Epoha', 'Vrijednost gubitka', 'train_loss.png'),
        (test_accuracy, 'Točnost na skupu za validaciju', 'Epoha', 'Točnost', 'val_acc.png'),
        (test_loss, 'Vrijednost funkcije gubitka tijekom validacije', 'Epoha', 'Vrijednost gubitka', 'val_loss.png'),
        (avg_precision, 'Prosječna preciznost', 'Epoha', 'Preciznost', 'avg_precision.png'),
        (avg_recall, 'Prosječni odziv', 'Epoha', 'Odziv', 'avg_recall.png'),
        (avg_f1_score, 'Prosječni F1 rezultat', 'Epoha', 'F1 rezultat', 'avg_f1.png')
    ]

    # Create a plot for each metric
    for i, (data, title, xlabel, ylabel, filename) in enumerate(metrics):
        plt.figure(i + 1)
        plt.plot(data, linestyle='-', color='b')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()  # Show the plot in the window

log_file_path = 'log/point_classification/pointnet2_part_seg_msg/logs/pointnet2_part_seg_msg_new.txt'
train_accuracy, train_loss, test_accuracy, test_loss, avg_precision, avg_recall, avg_f1_score = parse_log_file(log_file_path)

plot_metrics(train_accuracy, train_loss, test_accuracy, test_loss, avg_precision, avg_recall, avg_f1_score)
