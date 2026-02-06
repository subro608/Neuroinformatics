import argparse
import json
import os
import pickle
import random
import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from eeg_svm import EEGSVM
from eeg_svm_dataset import EEGSVMDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set random seed
random.seed(42)
np.random.seed(42)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train SVM model on EEG data with K-fold cross-validation"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="model-data",
        help="Directory containing the EEG data files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="svm-models",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--folds", type=int, default=5, help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        choices=["rbf", "linear", "poly", "sigmoid"],
        help="SVM kernel type",
    )
    parser.add_argument(
        "--C", type=float, default=1.0, help="SVM regularization parameter"
    )
    parser.add_argument(
        "--gamma", type=str, default="scale", help="SVM kernel coefficient"
    )
    parser.add_argument(
        "--balance_classes",
        action="store_true",
        help="Use class weights to handle imbalanced data",
    )

    return parser.parse_args()


def train_svm_kfold():
    """Train SVM model on EEG data using K-fold cross-validation"""
    args = parse_args()

    # Create output directories
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    images_dir = os.path.join(args.output_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_dir, f"training_log_{timestamp}.txt")

    def log(message):
        print(message)
        with open(log_file, "a") as f:
            f.write(message + "\n")

    log(f"Starting SVM training with {args.folds}-fold cross-validation at {timestamp}")
    log(f"Arguments: {args}")

    # Load data
    log("Loading dataset...")
    dataset = EEGSVMDataset(args.data_dir)

    # Load all training data
    with open(os.path.join(args.data_dir, "labels.json"), "r") as file:
        data_info = json.load(file)

    train_data = [d for d in data_info if d["type"] == "train"]

    # Separate training data by class for balancing
    train_data_A = [d for d in train_data if d["label"] == "A"]
    train_data_C = [d for d in train_data if d["label"] == "C"]
    train_data_F = [d for d in train_data if d["label"] == "F"]

    # Determine the minimum number of samples for balancing
    min_samples = min(
        (len(train_data_A) + len(train_data_C)) / 2,
        (len(train_data_A) + len(train_data_F)) / 2,
        (len(train_data_C) + len(train_data_F)) / 2,
    )

    a_index = int(min(min_samples, len(train_data_A)))
    c_index = int(min(min_samples, len(train_data_C)))
    f_index = int(min(min_samples, len(train_data_F)))

    # Randomly sample from each class to create a balanced training set
    balanced_train_data = (
        random.sample(train_data_A, a_index)
        + random.sample(train_data_C, c_index)
        + random.sample(train_data_F, f_index)
    )

    log(
        f"Before Balancing - A: {len(train_data_A)}, C: {len(train_data_C)}, F: {len(train_data_F)}"
    )
    log(f"After Balancing - A: {a_index}, C: {c_index}, F: {f_index}")
    log(f"Total: {len(balanced_train_data)}")

    # Create a dataset using the balanced training data
    balanced_dataset = EEGSVMDataset(args.data_dir, balanced_train_data)
    X, y = balanced_dataset.load_data()

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    # Class weights for imbalanced data
    class_weight = None
    if args.balance_classes:
        class_counts = np.bincount(y)
        class_weight = {
            i: len(y) / (len(np.unique(y)) * count)
            for i, count in enumerate(class_counts)
        }
        log(f"Using class weights: {class_weight}")

    # Training parameters
    log("\nTraining Parameters:")
    log(f"Kernel: {args.kernel}")
    log(f"C: {args.C}")
    log(f"Gamma: {args.gamma}")
    log(f"Class weights: {class_weight}")

    # Initialize arrays to store metrics
    fold_accuracies = []
    fold_train_times = []
    best_fold_model = None
    best_fold_scaler = None
    best_fold_accuracy = 0

    # Training loop
    for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
        fold_start_time = time.time()
        log(f"\nFold {fold + 1}/{args.folds}")

        # Split data
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        log(f"Train set: {X_train.shape}, Valid set: {X_valid.shape}")

        # Class distribution in training set
        train_class_counts = {label: (y_train == label).sum() for label in range(3)}
        log(
            f"Train class distribution - A: {train_class_counts[0]}, C: {train_class_counts[1]}, F: {train_class_counts[2]}"
        )

        # Create scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)

        # Create and train model
        log(f"Training SVM model for fold {fold + 1}...")
        model = SVC(
            kernel=args.kernel,
            C=args.C,
            gamma=args.gamma,
            class_weight=class_weight,
            probability=True,
            random_state=42,
        )

        # Train model
        train_start_time = time.time()
        model.fit(X_train_scaled, y_train)
        train_end_time = time.time()
        train_time = train_end_time - train_start_time

        # Evaluate on validation set
        y_pred = model.predict(X_valid_scaled)
        accuracy = accuracy_score(y_valid, y_pred)

        log(
            f"Fold {fold + 1} - Validation Accuracy: {accuracy:.4f}, Training Time: {train_time:.2f}s"
        )

        # Store metrics
        fold_accuracies.append(accuracy)
        fold_train_times.append(train_time)

        # Keep track of best model
        if accuracy > best_fold_accuracy:
            best_fold_accuracy = accuracy
            best_fold_model = model
            best_fold_scaler = scaler
            best_fold = fold + 1
            log(
                f"New best model from fold {best_fold} with accuracy {best_fold_accuracy:.4f}"
            )

        fold_end_time = time.time()
        log(f"Fold {fold + 1} completed in {fold_end_time - fold_start_time:.2f}s")

    # Calculate average metrics
    avg_accuracy = np.mean(fold_accuracies)
    avg_train_time = np.mean(fold_train_times)

    log("\nTraining complete!")
    log(f"Average validation accuracy: {avg_accuracy:.4f}")
    log(f"Average training time: {avg_train_time:.2f}s")
    log(f"Best fold: {best_fold} with accuracy {best_fold_accuracy:.4f}")

    # Fixed: Create a properly initialized best model
    best_model = EEGSVM(
        kernel=args.kernel, C=args.C, gamma=args.gamma, class_weight=class_weight
    )

    # Create a properly fitted pipeline
    pipeline = Pipeline(
        [
            ("scaler", best_fold_scaler),  # Use the pre-fitted scaler
            ("svm", best_fold_model),  # Use the pre-fitted SVM model
        ]
    )

    # Set the pipeline in the model
    best_model.pipeline = pipeline

    # Save best model
    model_path = os.path.join(
        args.output_dir, f"svm_model_{timestamp}_fold{best_fold}.pkl"
    )
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": best_model,
                "scaler": best_fold_scaler,
                "params": {
                    "kernel": args.kernel,
                    "C": args.C,
                    "gamma": args.gamma,
                    "class_weight": class_weight,
                },
                "fold_accuracies": fold_accuracies,
                "best_fold": best_fold,
                "best_accuracy": best_fold_accuracy,
            },
            f,
        )

    log(f"Best model saved to {model_path}")

    # Plot fold accuracies
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, args.folds + 1), fold_accuracies)
    plt.xlabel("Fold")
    plt.ylabel("Validation Accuracy")
    plt.title(f"SVM {args.folds}-Fold Cross-Validation Accuracy")
    plt.xticks(range(1, args.folds + 1))
    plt.ylim([0, 1])
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for i, acc in enumerate(fold_accuracies):
        plt.text(i + 1, acc + 0.01, f"{acc:.4f}", ha="center", va="bottom")

    plt.tight_layout()
    accuracy_plot_path = os.path.join(
        images_dir, f"svm_fold_accuracies_{timestamp}.png"
    )
    plt.savefig(accuracy_plot_path)
    plt.close()

    log(f"Fold accuracies plot saved to {accuracy_plot_path}")

    # Now test on both test sets using the best model
    log("\nEvaluating best model on test sets...")

    # Test on cross-subject data
    X_test_cross, y_test_cross = dataset.load_data("test_cross")
    cross_results = best_model.evaluate(X_test_cross, y_test_cross)

    log("\nCross-subject test results:")
    log(f"Accuracy: {cross_results['accuracy']:.4f}")
    log(f"F1 Score: {cross_results['f1']:.4f}")

    log("Binary classification accuracies:")
    log(f"A vs. C: {cross_results['binary_accuracies']['A_vs_C']:.4f}")
    log(f"C vs. F: {cross_results['binary_accuracies']['C_vs_F']:.4f}")
    log(f"A vs. F: {cross_results['binary_accuracies']['A_vs_F']:.4f}")

    # Confusion matrix for cross-subject
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cross_results["confusion_matrix"],
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=["A", "C", "F"],
        yticklabels=["A", "C", "F"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Cross-subject)")
    cm_cross_path = os.path.join(
        images_dir, f"svm_confusion_matrix_cross_{timestamp}.png"
    )
    plt.savefig(cm_cross_path)
    plt.close()

    # Test on within-subject data
    X_test_within, y_test_within = dataset.load_data("test_within")
    within_results = best_model.evaluate(X_test_within, y_test_within)

    log("\nWithin-subject test results:")
    log(f"Accuracy: {within_results['accuracy']:.4f}")
    log(f"F1 Score: {within_results['f1']:.4f}")

    log("Binary classification accuracies:")
    log(f"A vs. C: {within_results['binary_accuracies']['A_vs_C']:.4f}")
    log(f"C vs. F: {within_results['binary_accuracies']['C_vs_F']:.4f}")
    log(f"A vs. F: {within_results['binary_accuracies']['A_vs_F']:.4f}")

    # Confusion matrix for within-subject
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        within_results["confusion_matrix"],
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=["A", "C", "F"],
        yticklabels=["A", "C", "F"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Within-subject)")
    cm_within_path = os.path.join(
        images_dir, f"svm_confusion_matrix_within_{timestamp}.png"
    )
    plt.savefig(cm_within_path)
    plt.close()

    # Save final results
    results = {
        "timestamp": timestamp,
        "params": {
            "kernel": args.kernel,
            "C": args.C,
            "gamma": args.gamma,
            "class_weight": class_weight,
            "folds": args.folds,
        },
        "fold_accuracies": fold_accuracies,
        "best_fold": best_fold,
        "best_accuracy": best_fold_accuracy,
        "cross_subject_results": {
            "accuracy": cross_results["accuracy"],
            "f1": cross_results["f1"],
            "confusion_matrix": cross_results["confusion_matrix"].tolist(),
            "binary_accuracies": cross_results["binary_accuracies"],
        },
        "within_subject_results": {
            "accuracy": within_results["accuracy"],
            "f1": within_results["f1"],
            "confusion_matrix": within_results["confusion_matrix"].tolist(),
            "binary_accuracies": within_results["binary_accuracies"],
        },
    }

    results_path = os.path.join(args.output_dir, f"svm_results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    log(f"\nFinal results saved to {results_path}")

    return best_model, results


if __name__ == "__main__":
    # Import here to avoid unused import warning when imported as module
    import seaborn as sns

    train_svm_kfold()
