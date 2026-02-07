import json
import os
import time
import warnings

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DISEASE_NAMES = {
    "A": "Alzheimer's Disease (AD)",
    "C": "Control (CN)",
    "F": "Frontotemporal Dementia (FTD)",
}


def evaluate_model(model, dataloader, device, class_names=["A", "C", "F"]):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels, all_probs = [], [], []
    class_correct = {cls: 0 for cls in class_names}
    class_total = {cls: 0 for cls in class_names}
    class_as_other = {
        "A": {"as_C": 0, "as_F": 0},
        "C": {"as_A": 0, "as_F": 0},
        "F": {"as_A": 0, "as_C": 0},
    }
    confusion = np.zeros((len(class_names), len(class_names)), dtype=int)
    inference_times = []

    with torch.no_grad():
        for inputs_dict, labels in tqdm(dataloader, desc="Evaluating"):
            raw_eeg = inputs_dict["raw_eeg"].to(device, non_blocking=True)
            scale_inputs = {
                k: v.to(device, non_blocking=True)
                for k, v in inputs_dict.items()
                if k.startswith("scale_")
            }
            if "adjacency" in inputs_dict:
                scale_inputs["adjacency"] = inputs_dict["adjacency"].to(
                    device, non_blocking=True
                )
            if "spatial_positions" in inputs_dict:
                scale_inputs["spatial_positions"] = inputs_dict["spatial_positions"].to(
                    device, non_blocking=True
                )
            labels = labels.to(device, non_blocking=True)

            start_time = time.time()
            with torch.amp.autocast(device_type=device.type):
                outputs = model(raw_eeg, scale_inputs)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            probs = torch.softmax(outputs.float(), dim=1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for i in range(labels.size(0)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                confusion[true_label][pred_label] += 1
                true_cls = class_names[true_label]
                pred_cls = class_names[pred_label]
                class_total[true_cls] += 1
                if true_label == pred_label:
                    class_correct[true_cls] += 1
                else:
                    if true_cls == "A":
                        class_as_other["A"]["as_C" if pred_cls == "C" else "as_F"] += 1
                    elif true_cls == "C":
                        class_as_other["C"]["as_A" if pred_cls == "A" else "as_F"] += 1
                    else:
                        class_as_other["F"]["as_A" if pred_cls == "A" else "as_C"] += 1

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    class_accuracy = {
        cls: (
            100.0 * class_correct[cls] / class_total[cls]
            if class_total[cls] > 0
            else 0.0
        )
        for cls in class_names
    }

    # Binary accuracies
    a_c_correct = class_correct["A"] + class_correct["C"]
    a_c_total = class_total["A"] + class_total["C"]
    a_c_accuracy = 100.0 * a_c_correct / a_c_total if a_c_total > 0 else 0.0
    a_f_correct = class_correct["A"] + class_correct["F"]
    a_f_total = class_total["A"] + class_total["F"]
    a_f_accuracy = 100.0 * a_f_correct / a_f_total if a_f_total > 0 else 0.0
    c_f_correct = class_correct["C"] + class_correct["F"]
    c_f_total = class_total["C"] + class_total["F"]
    c_f_accuracy = 100.0 * c_f_correct / c_f_total if c_f_total > 0 else 0.0

    # Balanced accuracy
    balanced_accuracy = (
        np.mean([class_accuracy[cls] for cls in class_names]) if total > 0 else 0.0
    )

    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average=None,
        labels=list(range(len(class_names))),
        zero_division=0,
    )

    # ROC-AUC and PR-AUC
    all_probs = np.array(all_probs)
    all_labels_one_hot = np.eye(len(class_names))[all_labels]
    roc_auc = {
        cls: roc_auc_score(all_labels_one_hot[:, i], all_probs[:, i])
        for i, cls in enumerate(class_names)
        if class_total[cls] > 0
    }
    pr_auc = {
        cls: average_precision_score(all_labels_one_hot[:, i], all_probs[:, i])
        for i, cls in enumerate(class_names)
        if class_total[cls] > 0
    }

    class_metrics = {}
    for i, cls in enumerate(class_names):
        tn = sum(class_correct[c] for c in class_names if c != cls)
        fp = sum(class_as_other[c][f"as_{cls}"] for c in class_names if c != cls)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        class_metrics[cls] = {
            "precision": class_precision[i],
            "recall": class_recall[i],
            "f1": class_f1[i],
            "accuracy": class_accuracy[cls] / 100.0,
            "sensitivity": class_recall[i],
            "specificity": specificity,
            "roc_auc": roc_auc.get(cls, 0.0),
            "pr_auc": pr_auc.get(cls, 0.0),
        }

    avg_inference_time = np.mean(inference_times) * 1000  # ms

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "confusion_matrix": confusion,
        "class_metrics": class_metrics,
        "binary_accuracies": {
            "A_vs_C": a_c_accuracy,
            "A_vs_F": a_f_accuracy,
            "C_vs_F": c_f_accuracy,
        },
        "inference_time": avg_inference_time,
        "class_totals": class_total,
        "class_correct": class_correct,
        "class_as_other": class_as_other,
    }


def test_model(data_type, model_path):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f"results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    dim = int(os.environ.get("EEG_DIM", 64))
    num_channels, num_classes, scales = 19, 3, [3, 4, 5]
    class_names = ["A", "C", "F"]

    # Dynamic batch size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        free_mem, _ = torch.cuda.mem_get_info()
        batch_size = 16 if free_mem > 8e9 else 8 if free_mem > 4e9 else 4
    else:
        batch_size = 2
    print(f"Using device: {device}, Batch size: {batch_size}")

    # Load model
    from eeg_multi_scale_graph_spectral_advanced import create_model

    try:
        model = create_model(
            num_channels=num_channels, num_classes=num_classes, dim=dim, scales=scales
        )
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        model.to(device)
        print(
            f"Model loaded from {model_path} with {sum(p.numel() for p in model.parameters()):,} parameters"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Load data
    data_dir = os.environ.get("EEG_DATA_DIR", "model_data")
    labels_path = os.path.join(data_dir, "labels.json")
    if not os.path.exists(labels_path):
        print(f"Error: labels.json not found")
        return None

    with open(labels_path, "r") as file:
        data_info = json.load(file)
    test_data = [
        d
        for d in data_info
        if d.get("type") == data_type and d.get("label") in class_names
    ]
    verified_data = [
        d for d in test_data if os.path.exists(os.path.join(data_dir, d["file_name"]))
    ]

    class_counts = {
        cls: sum(1 for d in verified_data if d["label"] == cls) for cls in class_names
    }
    print(
        f"Dataset {data_type}: {len(verified_data)} samples, Distribution: {class_counts}"
    )

    from eeg_dataset_multiscalegraph_spectral_advanced import SpatialSpectralEEGDataset

    test_dataset = SpatialSpectralEEGDataset(
        data_dir=data_dir,
        data_info=verified_data,
        scales=scales,
        time_length=2000,
        adjacency_type="combined",
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
        num_workers=2,
    )

    # Evaluate
    print(f"\nEvaluating on {data_type} dataset...")
    results = evaluate_model(model, test_dataloader, device, class_names)

    # Save results
    results_file_txt = os.path.join(output_dir, f"{data_type}_results.txt")
    results_file_json = os.path.join(output_dir, f"{data_type}_results.json")
    with open(results_file_txt, "w") as f:
        f.write(
            f"EVALUATION RESULTS: {data_type.upper()}\n\nModel: {model_path}\nSamples: {len(test_dataset)}\n"
            f"Class Distribution: {class_counts}\n\n"
        )
        f.write(
            f"Accuracy: {results['accuracy']:.2f}%\nBalanced Accuracy: {results['balanced_accuracy']:.2f}%\n"
            f"Precision: {results['precision']:.2f}%\nRecall: {results['recall']:.2f}%\nF1: {results['f1']:.2f}%\n"
            f"Inference Time: {results['inference_time']:.2f} ms\n\n"
        )
        f.write("BINARY ACCURACIES\n")
        for key, val in results["binary_accuracies"].items():
            f.write(f"{key}: {val:.2f}%\n")
        f.write("\nPER-CLASS METRICS\n")
        for cls in class_names:
            metrics = results["class_metrics"][cls]
            f.write(
                f"{cls} ({DISEASE_NAMES[cls]}):\n  Total: {results['class_totals'][cls]}\n"
                f"  Correct: {results['class_correct'][cls]}\n"
            )
            for k, v in results["class_as_other"][cls].items():
                f.write(
                    f"  Misclassified {k}: {v} ({100 * v / results['class_totals'][cls]:.2f}%)\n"
                )
            f.write(
                f"  Precision: {metrics['precision']*100:.2f}%\n  Recall: {metrics['recall']*100:.2f}%\n"
                f"  F1: {metrics['f1']*100:.2f}%\n  Accuracy: {metrics['accuracy']*100:.2f}%\n"
                f"  Specificity: {metrics['specificity']*100:.2f}%\n  ROC-AUC: {metrics['roc_auc']*100:.2f}%\n"
                f"  PR-AUC: {metrics['pr_auc']*100:.2f}%\n\n"
            )
        f.write("CONFUSION MATRIX\n")
        f.write(f"  {'':6s} {'A':^6s} {'C':^6s} {'F':^6s}\n")
        for i, row in enumerate(results["confusion_matrix"]):
            f.write(f"  {class_names[i]:6s} {row[0]:^6d} {row[1]:^6d} {row[2]:^6d}\n")

    with open(results_file_json, "w") as f:
        json.dump(
            {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in results.items()
            },
            f,
            indent=4,
        )

    print(f"Results saved to {results_file_txt} and {results_file_json}")
    return results


def main():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f"spatial_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.environ.get("EEG_MODEL_PATH", "")
    if not model_path or not os.path.exists(model_path):
        # Search for best_model_overall.pth
        for root, _, files in os.walk("."):
            for file in files:
                if file.endswith(".pth") and "best_model_overall" in file:
                    model_path = os.path.join(root, file)
                    break
            if model_path and os.path.exists(model_path):
                break
        else:
            print("No suitable model found. Set EEG_MODEL_PATH env var.")
            return

    print(f"Using model: {model_path}")
    results = {}
    for data_type in ["test_cross", "test_within"]:
        print(f"\nEvaluating {data_type}...")
        results[data_type] = test_model(data_type, model_path)

    # Comparison summary
    if any(results.values()):
        summary_file = os.path.join(output_dir, "comparison_summary.txt")
        with open(summary_file, "w") as f:
            f.write(f"COMPARISON SUMMARY\nModel: {model_path}\n\n")
            f.write(
                f"{'Dataset':15s} {'Accuracy':10s} {'Bal.Acc':10s} {'Precision':10s} {'Recall':10s} {'F1':10s}\n"
            )
            f.write("-" * 60 + "\n")
            for dataset, result in results.items():
                if result:
                    f.write(
                        f"{dataset:15s} {result['accuracy']:.2f}%{'':<2s} {result['balanced_accuracy']:.2f}%{'':<2s} "
                        f"{result['precision']:.2f}%{'':<2s} {result['recall']:.2f}%{'':<2s} {result['f1']:.2f}%\n"
                    )
            f.write("\nBINARY ACCURACIES\n")
            f.write(
                f"{'Dataset':15s} {'AD vs CN':12s} {'AD vs FTD':12s} {'CN vs FTD':12s}\n"
            )
            f.write("-" * 60 + "\n")
            for dataset, result in results.items():
                if result:
                    f.write(
                        f"{dataset:15s} {result['binary_accuracies']['A_vs_C']:.2f}%{'':<5s} "
                        f"{result['binary_accuracies']['A_vs_F']:.2f}%{'':<5s} {result['binary_accuracies']['C_vs_F']:.2f}%\n"
                    )
            f.write("\nPER-CLASS COMPARISON\n")
            for cls in ["A", "C", "F"]:
                f.write(
                    f"\n{cls} ({DISEASE_NAMES[cls]})\n{'Dataset':15s} {'Prec':8s} {'Rec':8s} {'F1':8s} "
                    f"{'Acc':8s} {'Spec':8s} {'ROC-AUC':8s} {'PR-AUC':8s}\n"
                )
                f.write("-" * 70 + "\n")
                for dataset, result in results.items():
                    if result and cls in result["class_metrics"]:
                        m = result["class_metrics"][cls]
                        f.write(
                            f"{dataset:15s} {m['precision']*100:.2f}%{'':<2s} {m['recall']*100:.2f}%{'':<2s} "
                            f"{m['f1']*100:.2f}%{'':<2s} {m['accuracy']*100:.2f}%{'':<2s} {m['specificity']*100:.2f}%{'':<2s} "
                            f"{m['roc_auc']*100:.2f}%{'':<2s} {m['pr_auc']*100:.2f}%\n"
                        )

        print(f"\nSummary saved to {summary_file}")

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
