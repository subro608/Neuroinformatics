import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from eeg_dataset import EEGDataset
from eeg_net import EEGNet
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Create output directories
if not os.path.exists("images"):
    os.makedirs("images")
if not os.path.exists("test_results"):
    os.makedirs("test_results")

# Model params
num_chans = 19
timepoints = 1425
num_classes = 3
F1 = 5
D = 5
F2 = 25
dropout_rate = 0.5


def test_model(data_type="test_cross"):
    # Load model
    model_file = "models/eegnet_5fold_train7.pth"
    try:
        model = EEGNet(
            num_channels=num_chans,
            timepoints=timepoints,
            num_classes=num_classes,
            F1=F1,
            D=D,
            F2=F2,
            dropout_rate=dropout_rate,
        )
        model.load_state_dict(torch.load(model_file))
        print(f"Model loaded successfully from {model_file}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Data loading
    data_dir = "model-data"
    data_file = "labels.json"

    with open(os.path.join(data_dir, data_file), "r") as file:
        data_info = json.load(file)

    test_data = [d for d in data_info if d["type"] == data_type]
    test_dataset = EEGDataset(data_dir, test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    # Count samples per class
    total_a = sum(1 for d in test_data if d["label"] == "A")
    total_c = sum(1 for d in test_data if d["label"] == "C")
    total_f = sum(1 for d in test_data if d["label"] == "F")

    print(f"\nTesting on {data_type} data:")
    print(f"Test dataset: {len(test_dataset)} samples")
    print(f"Test dataloader: {len(test_dataloader)} batches")
    print(f"Test dataloader batch size: {test_dataloader.batch_size}")
    print(f"Class distribution - A: {total_a}, C: {total_c}, F: {total_f}\n")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize metrics storage
    all_labels = []
    all_probs = []
    a_probs = []
    c_probs = []
    f_probs = []

    # Testing loop
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        correct_a = 0
        a_as_c = 0
        a_as_f = 0
        correct_c = 0
        c_as_a = 0
        c_as_f = 0
        correct_f = 0
        f_as_a = 0
        f_as_c = 0

        for eeg_data, labels in tqdm(test_dataloader, desc="Testing"):
            eeg_data, labels = eeg_data.to(device), labels.to(device)
            outputs = model.forward(eeg_data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store probabilities
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
            a_probs.extend(probs[:, 0])
            c_probs.extend(probs[:, 1])
            f_probs.extend(probs[:, 2])

            # Count predictions
            for i in range(labels.size(0)):
                if labels[i] == 0:  # True class is A
                    if predicted[i] == 0:
                        correct_a += 1
                    elif predicted[i] == 1:
                        a_as_c += 1
                    else:
                        a_as_f += 1
                elif labels[i] == 1:  # True class is C
                    if predicted[i] == 1:
                        correct_c += 1
                    elif predicted[i] == 0:
                        c_as_a += 1
                    else:
                        c_as_f += 1
                else:  # True class is F
                    if predicted[i] == 2:
                        correct_f += 1
                    elif predicted[i] == 0:
                        f_as_a += 1
                    else:
                        f_as_c += 1

    # Calculate metrics
    accuracy = correct / total if total > 0 else 0

    # Create confusion matrix
    confusion_matrix = np.zeros((3, 3))
    confusion_matrix[0, 0] = correct_a
    confusion_matrix[0, 1] = a_as_c
    confusion_matrix[0, 2] = a_as_f
    confusion_matrix[1, 0] = c_as_a
    confusion_matrix[1, 1] = correct_c
    confusion_matrix[1, 2] = c_as_f
    confusion_matrix[2, 0] = f_as_a
    confusion_matrix[2, 1] = f_as_c
    confusion_matrix[2, 2] = correct_f

    # Binary classification accuracies
    accuracy_ad_cn = (
        (correct_a + correct_c) / (total_a + total_c) if (total_a + total_c) > 0 else 0
    )
    accuracy_ftd_cn = (
        (correct_c + correct_f) / (total_c + total_f) if (total_c + total_f) > 0 else 0
    )
    accuracy_ad_ftd = (
        (correct_a + correct_f) / (total_a + total_f) if (total_a + total_f) > 0 else 0
    )

    # Calculate per-class metrics (fixed to match test_mvt.py calculation method)
    # Class A
    precision_a = (
        correct_a / (correct_a + c_as_a + f_as_a)
        if (correct_a + c_as_a + f_as_a) > 0
        else 0
    )
    recall_a = (
        correct_a / (correct_a + a_as_c + a_as_f)
        if (correct_a + a_as_c + a_as_f) > 0
        else 0
    )
    f1_a = (
        2 * precision_a * recall_a / (precision_a + recall_a)
        if (precision_a + recall_a) > 0
        else 0
    )
    sensitivity_a = correct_a / total_a if total_a > 0 else 0
    specificity_a = (
        (correct_c + correct_f) / (total_c + total_f) if (total_c + total_f) > 0 else 0
    )

    # Class C
    precision_c = (
        correct_c / (correct_c + a_as_c + f_as_c)
        if (correct_c + a_as_c + f_as_c) > 0
        else 0
    )
    recall_c = (
        correct_c / (correct_c + c_as_a + c_as_f)
        if (correct_c + c_as_a + c_as_f) > 0
        else 0
    )
    f1_c = (
        2 * precision_c * recall_c / (precision_c + recall_c)
        if (precision_c + recall_c) > 0
        else 0
    )
    sensitivity_c = correct_c / total_c if total_c > 0 else 0
    specificity_c = (
        (correct_a + correct_f) / (total_a + total_f) if (total_a + total_f) > 0 else 0
    )

    # Class F
    precision_f = (
        correct_f / (correct_f + a_as_f + c_as_f)
        if (correct_f + a_as_f + c_as_f) > 0
        else 0
    )
    recall_f = (
        correct_f / (correct_f + f_as_a + f_as_c)
        if (correct_f + f_as_a + f_as_c) > 0
        else 0
    )
    f1_f = (
        2 * precision_f * recall_f / (precision_f + recall_f)
        if (precision_f + recall_f) > 0
        else 0
    )
    sensitivity_f = correct_f / total_f if total_f > 0 else 0
    specificity_f = (
        (correct_a + correct_c) / (total_a + total_c) if (total_a + total_c) > 0 else 0
    )

    # Mean metrics
    mAP = (precision_a + precision_c + precision_f) / 3
    mAR = (recall_a + recall_c + recall_f) / 3
    mF1 = (f1_a + f1_c + f1_f) / 3

    # Save results to file
    results_file = f"test_results/eegnet_results_{data_type}.txt"
    with open(results_file, "w") as f:
        f.write(f"EEGNet Test Results for {data_type}\n\n")
        f.write(f"Correct: {correct}, Total: {total}\n")
        f.write(
            f"Correct A: {correct_a}, A as C: {a_as_c}, A as F: {a_as_f}, Total A: {total_a}\n"
        )
        f.write(
            f"Correct C: {correct_c}, C as A: {c_as_a}, C as F: {c_as_f}, Total C: {total_c}\n"
        )
        f.write(
            f"Correct F: {correct_f}, F as A: {f_as_a}, F as C: {f_as_c}, Total F: {total_f}\n"
        )
        f.write(f"Overall Accuracy: {100 * accuracy:.4f}%\n\n")
        f.write("Binary Classification Accuracies:\n")
        f.write(f"AD vs. CN: {100 * accuracy_ad_cn:.4f}%\n")
        f.write(f"FTD vs. CN: {100 * accuracy_ftd_cn:.4f}%\n")
        f.write(f"AD vs. FTD: {100 * accuracy_ad_ftd:.4f}%\n\n")
        f.write("Per-Class Metrics:\n")
        f.write(
            f"Class A - Precision: {100 * precision_a:.4f}%, Recall: {100 * recall_a:.4f}%, F1: {100 * f1_a:.4f}%, "
            f"Sensitivity: {100 * sensitivity_a:.4f}%, Specificity: {100 * specificity_a:.4f}%\n"
        )
        f.write(
            f"Class C - Precision: {100 * precision_c:.4f}%, Recall: {100 * recall_c:.4f}%, F1: {100 * f1_c:.4f}%, "
            f"Sensitivity: {100 * sensitivity_c:.4f}%, Specificity: {100 * specificity_c:.4f}%\n"
        )
        f.write(
            f"Class F - Precision: {100 * precision_f:.4f}%, Recall: {100 * recall_f:.4f}%, F1: {100 * f1_f:.4f}%, "
            f"Sensitivity: {100 * sensitivity_f:.4f}%, Specificity: {100 * specificity_f:.4f}%\n\n"
        )
        f.write("Mean Metrics:\n")
        f.write(
            f"mAP: {100 * mAP:.4f}%, mAR: {100 * mAR:.4f}%, mF1: {100 * mF1:.4f}%\n"
        )

    # Print results
    print(f"\nResults saved to {results_file}")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=["A", "C", "F"],
        yticklabels=["A", "C", "F"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({data_type})")
    plt.savefig(f"images/eegnet_confusion_matrix_{data_type}.png")
    plt.close()

    # Plot ROC curves
    all_probs = np.array(all_probs)

    fpr_a, tpr_a, _ = roc_curve(all_labels, a_probs, pos_label=0)
    fpr_c, tpr_c, _ = roc_curve(all_labels, c_probs, pos_label=1)
    fpr_f, tpr_f, _ = roc_curve(all_labels, f_probs, pos_label=2)

    roc_auc_a, roc_auc_c, roc_auc_f = roc_auc_score(
        all_labels, all_probs, multi_class="ovr", average=None
    )

    plt.figure(figsize=(10, 8))
    plt.plot(fpr_a, tpr_a, color="darkorange", lw=2, label=f"AUC A: {roc_auc_a:.4f}")
    plt.plot(fpr_c, tpr_c, color="green", lw=2, label=f"AUC C: {roc_auc_c:.4f}")
    plt.plot(fpr_f, tpr_f, color="red", lw=2, label=f"AUC F: {roc_auc_f:.4f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves ({data_type})")
    plt.legend(loc="lower right")
    plt.savefig(f"images/eegnet_roc_curves_{data_type}.png")
    plt.close()

    return accuracy, mF1


if __name__ == "__main__":
    # Test both cross-subject and within-subject
    cross_acc, cross_f1 = test_model("test_cross")
    within_acc, within_f1 = test_model("test_within")

    print("\nFinal Results:")
    print(
        f"Cross-subject - Accuracy: {100 * cross_acc:.2f}%, F1: {100 * cross_f1:.2f}%"
    )
    print(
        f"Within-subject - Accuracy: {100 * within_acc:.2f}%, F1: {100 * within_f1:.2f}%"
    )
