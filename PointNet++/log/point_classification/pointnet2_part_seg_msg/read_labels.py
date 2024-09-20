import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def extract_labels_from_best_epoch(file_path):
    predicted_labels = []
    true_labels = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace
        
        # Check if the line contains predicted or true labels
        if line.startswith('"') and ', [' in line:
            # Extract predicted labels
            parts = line.split(', [')
            if len(parts) == 2:
                labels_str = parts[1].strip('[]')  # Remove the brackets
                labels = np.array(labels_str.split(',')).astype(int)
                predicted_labels.append(labels)

        elif 'True labels for' in line:
            # Extract true labels
            start_index = line.index(':') + 2  # Skip to after ': '
            labels_str = line[start_index:].strip('[]')  # Remove the brackets
            labels = np.array(labels_str.split(',')).astype(int)
            true_labels.append(labels)

    return predicted_labels, true_labels

def segment_labels(labels, num_segments=8):
    segmented = []
    for label_array in labels:
        # Calculate segment size
        segment_size = len(label_array) // num_segments
        # Create segments
        segments = [label_array[i * segment_size:(i + 1) * segment_size] for i in range(num_segments)]
        
        # Add the remainder to the last segment if necessary
        if len(label_array) % num_segments != 0:
            segments[-1] = np.concatenate((segments[-1], label_array[num_segments * segment_size:]))
        
        segmented.append(segments)
    return segmented

def plot_confusion_matrix(predicted_labels, true_labels):
    # Flatten the arrays to create single lists for true and predicted labels
    all_predicted = np.concatenate(predicted_labels)
    all_true = np.concatenate(true_labels)

    # Compute confusion matrix
    cm = confusion_matrix(all_true, all_predicted)

    # Calculate metrics
    accuracy = accuracy_score(all_true, all_predicted)
    precision = precision_score(all_true, all_predicted, average='weighted')
    recall = recall_score(all_true, all_predicted, average='weighted')
    f1 = f1_score(all_true, all_predicted, average='weighted')

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matrica zabune')
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Neorezane', 'Orezane'])
    plt.yticks(tick_marks, ['Neorezane', 'Orezane'])

    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j], horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('Istinita oznaka')
    plt.xlabel('Predviđena oznaka')
    plt.tight_layout()
    plt.show()

# Example usage
file_path = 'best_epoch_labels.txt'  # Replace with your file path
predicted_labels, true_labels = extract_labels_from_best_epoch(file_path)

# Segment labels into 8 parts
segmented_predicted = segment_labels(predicted_labels)
segmented_true = segment_labels(true_labels)

# Plot confusion matrix for each segment
for i in range(len(segmented_predicted)):
    plot_confusion_matrix(segmented_predicted[i], segmented_true[i])
