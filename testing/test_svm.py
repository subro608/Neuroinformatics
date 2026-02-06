import json
import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from eeg_svm_dataset import EEGSVMDataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

# Ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Create output directories
if not os.path.exists("images"):
    os.makedirs("images")
if not os.path.exists("test_results"):
    os.makedirs("test_results")


def test_model(model_file, data_type="test_cross"):
    """
    Test the SVM model on the specified data type.

    Args:
        model_file (str): Path to the saved model file
        data_type (str): Type of test data ('test_cross' or 'test_within')

    Returns:
        tuple: (accuracy, f1_score) on the test data
    """
    # Load model
    try:
        with open(model_file, "rb") as f:
            model_data = pickle.load(f)

        # Extract the components instead of using the pipeline
        svm_model = model_data["model"].model  # The actual SVC model
        scaler = model_data["scaler"]  # The fitted scaler

        print(f"Model loaded successfully from {model_file}")

        # Print model parameters
        params = model_data["params"]
        print(
            f"Model parameters: kernel={params['kernel']}, C={params['C']}, gamma={params['gamma']}"
        )

    except Exception as e:
        print(f"Error loading model: {e}")
        return 0, 0

    # Data loading
    data_dir = "model-data"
    data_file = "labels.json"

    with open(os.path.join(data_dir, data_file), "r") as file:
        data_info = json.load(file)

    test_data = [d for d in data_info if d["type"] == data_type]

    # Count samples per class
    total_a = sum(1 for d in test_data if d["label"] == "A")
    total_c = sum(1 for d in test_data if d["label"] == "C")
    total_f = sum(1 for d in test_data if d["label"] == "F")

    print(f"\nTesting on {data_type} data:")
    print(f"Test dataset: {len(test_data)} samples")
    print(f"Class distribution - A: {total_a}, C: {total_c}, F: {total_f}\n")

    # Create dataset
    test_dataset = EEGSVMDataset(data_dir, test_data)
    X_test, y_test = test_dataset.load_data()

    # Apply scaling manually
    X_test_scaled = scaler.transform(X_test)

    # Testing
    print("Evaluating model...")
    y_pred = svm_model.predict(X_test_scaled)
    y_pred_proba = svm_model.predict_proba(X_test_scaled)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Extract elements from confusion matrix
    correct_a = cm[0, 0]
    a_as_c = cm[0, 1]
    a_as_f = cm[0, 2]

    c_as_a = cm[1, 0]
    correct_c = cm[1, 1]
    c_as_f = cm[1, 2]

    f_as_a = cm[2, 0]
    f_as_c = cm[2, 1]
    correct_f = cm[2, 2]

    # Total correct and total
    correct = correct_a + correct_c + correct_f
    total = len(y_test)

    # Binary classification accuracies
    # A vs C
    a_vs_c_mask = np.where((y_test == 0) | (y_test == 1))[0]
    accuracy_ad_cn = accuracy_score(y_test[a_vs_c_mask], y_pred[a_vs_c_mask])

    # C vs F
    c_vs_f_mask = np.where((y_test == 1) | (y_test == 2))[0]
    accuracy_ftd_cn = accuracy_score(y_test[c_vs_f_mask], y_pred[c_vs_f_mask])

    # A vs F
    a_vs_f_mask = np.where((y_test == 0) | (y_test == 2))[0]
    accuracy_ad_ftd = accuracy_score(y_test[a_vs_f_mask], y_pred[a_vs_f_mask])

    # Calculate per-class metrics
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
    results_file = f"test_results/svm_results_{data_type}.txt"
    with open(results_file, "w") as f:
        f.write(f"SVM Test Results for {data_type}\n\n")
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
        cm,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=["A", "C", "F"],
        yticklabels=["A", "C", "F"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"SVM Confusion Matrix ({data_type})")
    cm_path = f"images/svm_confusion_matrix_{data_type}.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # Get class probabilities for ROC curves
    a_probs = y_pred_proba[:, 0]
    c_probs = y_pred_proba[:, 1]
    f_probs = y_pred_proba[:, 2]

    # Plot ROC curves
    fpr_a, tpr_a, _ = roc_curve(y_test, a_probs, pos_label=0)
    fpr_c, tpr_c, _ = roc_curve(y_test, c_probs, pos_label=1)
    fpr_f, tpr_f, _ = roc_curve(y_test, f_probs, pos_label=2)

    # Calculate AUC for each class (one-vs-rest)
    roc_auc_a = roc_auc_score((y_test == 0).astype(int), a_probs)
    roc_auc_c = roc_auc_score((y_test == 1).astype(int), c_probs)
    roc_auc_f = roc_auc_score((y_test == 2).astype(int), f_probs)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr_a, tpr_a, color="darkorange", lw=2, label=f"AUC A: {roc_auc_a:.4f}")
    plt.plot(fpr_c, tpr_c, color="green", lw=2, label=f"AUC C: {roc_auc_c:.4f}")
    plt.plot(fpr_f, tpr_f, color="red", lw=2, label=f"AUC F: {roc_auc_f:.4f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"SVM ROC Curves ({data_type})")
    plt.legend(loc="lower right")
    roc_path = f"images/svm_roc_curves_{data_type}.png"
    plt.savefig(roc_path)
    plt.close()
    print(f"ROC curves saved to {roc_path}")

    return accuracy, mF1


if __name__ == "__main__":
    # Path to your trained model
    model_file = "svm-models/svm_model_20250226_225329_fold4.pkl"  # Update this with your actual model path

    # Test both cross-subject and within-subject
    print("\nTesting SVM model on cross-subject data...")
    cross_acc, cross_f1 = test_model(model_file, "test_cross")

    print("\nTesting SVM model on within-subject data...")
    within_acc, within_f1 = test_model(model_file, "test_within")

    print("\nFinal Results:")
    print(
        f"Cross-subject - Accuracy: {100 * cross_acc:.2f}%, F1: {100 * cross_f1:.2f}%"
    )
    print(
        f"Within-subject - Accuracy: {100 * within_acc:.2f}%, F1: {100 * within_f1:.2f}%"
    )
