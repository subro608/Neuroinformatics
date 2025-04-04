import torch
import numpy as np
import os
import json
import warnings
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import time

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Disease names for readability
DISEASE_NAMES = {
    'A': 'Alzheimer\'s Disease (AD)',
    'C': 'Control (CN)',
    'F': 'Frontotemporal Dementia (FTD)'
}

def evaluate_model(model, dataloader, device, class_names=['A', 'C', 'F']):
    """Evaluate the model on a dataset"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    # For per-class metrics
    class_correct = {cls: 0 for cls in class_names}
    class_total = {cls: 0 for cls in class_names}
    class_as_other = {
        'A': {'as_C': 0, 'as_F': 0},
        'C': {'as_A': 0, 'as_F': 0},
        'F': {'as_A': 0, 'as_C': 0}
    }
    
    confusion = np.zeros((len(class_names), len(class_names)), dtype=int)
    
    # Inference time tracking
    inference_times = []
    
    with torch.no_grad():
        for inputs_dict, labels in tqdm(dataloader, desc="Evaluating"):
            # Move data to device
            raw_eeg = inputs_dict['raw_eeg'].to(device)
            scale_inputs = {k: v.to(device) for k, v in inputs_dict.items() if k.startswith('scale_')}
            labels = labels.to(device)
            
            # Measure inference time
            start_time = time.time()
            
            # Forward pass
            outputs = model(raw_eeg, scale_inputs)
            
            # Record inference time
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Get predictions
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update confusion matrix
            for i in range(labels.size(0)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                confusion[true_label][pred_label] += 1
                
                # Per-class tracking
                true_cls = class_names[true_label]
                pred_cls = class_names[pred_label]
                
                class_total[true_cls] += 1
                
                if true_label == pred_label:
                    class_correct[true_cls] += 1
                else:
                    # Track misclassifications
                    if true_cls == 'A':
                        if pred_cls == 'C':
                            class_as_other['A']['as_C'] += 1
                        else:
                            class_as_other['A']['as_F'] += 1
                    elif true_cls == 'C':
                        if pred_cls == 'A':
                            class_as_other['C']['as_A'] += 1
                        else:
                            class_as_other['C']['as_F'] += 1
                    else:  # F
                        if pred_cls == 'A':
                            class_as_other['F']['as_A'] += 1
                        else:
                            class_as_other['F']['as_C'] += 1
            
            # Store results
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = 100. * correct / total if total > 0 else 0.0
    
    # Calculate per-class metrics
    class_accuracy = {cls: 100. * class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0 
                      for cls in class_names}
    
    # Calculate binary classification accuracies
    # A vs C (AD vs Control)
    a_c_correct = class_correct['A'] + class_correct['C']
    a_c_total = class_total['A'] + class_total['C']
    a_c_accuracy = 100. * a_c_correct / a_c_total if a_c_total > 0 else 0.0
    
    # A vs F (AD vs FTD)
    a_f_correct = class_correct['A'] + class_correct['F']
    a_f_total = class_total['A'] + class_total['F']
    a_f_accuracy = 100. * a_f_correct / a_f_total if a_f_total > 0 else 0.0
    
    # C vs F (Control vs FTD)
    c_f_correct = class_correct['C'] + class_correct['F']
    c_f_total = class_total['C'] + class_total['F']
    c_f_accuracy = 100. * c_f_correct / c_f_total if c_f_total > 0 else 0.0
    
    # Calculate precision, recall, F1
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, 
            average='weighted',
            zero_division=0
        )
        
        # Per-class precision, recall, F1
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, 
            average=None,
            labels=list(range(len(class_names))),
            zero_division=0
        )
        
        class_metrics = {}
        for i, cls in enumerate(class_names):
            class_metrics[cls] = {
                'precision': class_precision[i],
                'recall': class_recall[i],
                'f1': class_f1[i],
                'accuracy': class_accuracy[cls] / 100.0,
                'sensitivity': class_recall[i]
            }
            
            # Calculate specificity
            if cls == 'A':
                tn = class_correct['C'] + class_correct['F']
                fp = class_as_other['C']['as_A'] + class_as_other['F']['as_A']
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            elif cls == 'C':
                tn = class_correct['A'] + class_correct['F']
                fp = class_as_other['A']['as_C'] + class_as_other['F']['as_C']
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:  # F
                tn = class_correct['A'] + class_correct['C']
                fp = class_as_other['A']['as_F'] + class_as_other['C']['as_F']
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
            class_metrics[cls]['specificity'] = specificity
    except:
        precision, recall, f1 = 0, 0, 0
        class_metrics = {cls: {'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0, 
                              'sensitivity': 0, 'specificity': 0} for cls in class_names}
    
    # Average inference time
    avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
    
    return {
        'accuracy': accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'confusion_matrix': confusion,
        'class_metrics': class_metrics,
        'binary_accuracies': {
            'A_vs_C': a_c_accuracy,
            'A_vs_F': a_f_accuracy,
            'C_vs_F': c_f_accuracy
        },
        'inference_time': avg_inference_time,
        'class_totals': class_total,
        'class_correct': class_correct,
        'class_as_other': class_as_other
    }

def test_model(data_type, model_path):
    """Test model on specified dataset (test_cross or test_within)"""
    # Create output directory with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f'results_{timestamp}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Model parameters
    num_channels = 19
    num_classes = 3
    dim = 256
    scales = [3, 4, 5]
    batch_size = 16
    
    # Class names
    class_names = ['A', 'C', 'F']
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    try:
        from eeg_multi_time_graph_spectral import create_model
        
        # Create model instance
        model = create_model(
            num_channels=num_channels,
            num_classes=num_classes,
            dim=dim,
            scales=scales
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model = checkpoint
        
        model = model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Load data
    try:
        # Find data directory
        data_dir = 'model-data'
        if not os.path.exists(data_dir):
            for alt_dir in ['data', '.', '..']:
                if os.path.exists(alt_dir):
                    data_dir = alt_dir
                    break
        
        # Check for labels.json
        labels_path = os.path.join(data_dir, 'labels.json')
        if not os.path.exists(labels_path):
            parent_dir = os.path.dirname(os.path.abspath(data_dir))
            labels_path = os.path.join(parent_dir, 'labels.json')
        
        # Load data info
        with open(labels_path, 'r') as file:
            data_info = json.load(file)
        
        # Filter data for specified test type
        test_data = [d for d in data_info if 'type' in d and d['type'] == data_type]
        
        # Verify file existence
        verified_data = []
        for item in test_data:
            if 'file_name' in item and 'label' in item and item['label'] in class_names:
                file_path = os.path.join(data_dir, item['file_name'])
                file_exists = False
                
                # Check various file extensions
                for ext in ['', '.set', '.npy', '.pkl']:
                    if os.path.exists(file_path + ext):
                        item['file_name'] = item['file_name'] + ext if ext else item['file_name']
                        verified_data.append(item)
                        file_exists = True
                        break
        
        # Print class distribution
        class_counts = {cls: 0 for cls in class_names}
        for d in verified_data:
            class_counts[d['label']] += 1
        
        # Create dataset
        from eeg_dataset_multitimegraph_spectral import MultiScaleTimeSpectralEEGDataset
        
        class RobustMultiScaleEEGDataset(MultiScaleTimeSpectralEEGDataset):
            def __getitem__(self, idx):
                try:
                    return super().__getitem__(idx)
                except:
                    sample = {
                        'raw_eeg': torch.zeros(19, self.time_length),
                        'adjacency': torch.FloatTensor(self.adjacency)
                    }
                    for scale in self.scales:
                        sample[f'scale_{scale}'] = torch.zeros(19, scale)
                    return sample, self.labels[idx]
        
        # Create dataset and dataloader
        test_dataset = RobustMultiScaleEEGDataset(data_dir, verified_data, scales=scales)
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=batch_size,
            num_workers=0,
            shuffle=False
        )
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Run evaluation
    try:
        results = evaluate_model(model, test_dataloader, device, class_names)
        
        # Save results to file
        results_file = os.path.join(output_dir, f'{data_type}_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"EVALUATION RESULTS: {data_type.upper()}\n\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Samples: {len(test_dataset)}\n")
            f.write(f"Class Distribution: {class_counts}\n\n")
            
            f.write("OVERALL RESULTS\n")
            f.write(f"Accuracy: {results['accuracy']:.2f}%\n")
            f.write(f"Precision: {results['precision']:.4f}%\n")
            f.write(f"Recall: {results['recall']:.4f}%\n")
            f.write(f"F1 Score: {results['f1']:.4f}%\n")
            f.write(f"Average Inference Time: {results['inference_time']:.2f} ms\n\n")
            
            f.write("BINARY CLASSIFICATION ACCURACIES\n")
            f.write(f"AD vs. Control: {results['binary_accuracies']['A_vs_C']:.2f}%\n")
            f.write(f"AD vs. FTD: {results['binary_accuracies']['A_vs_F']:.2f}%\n")
            f.write(f"Control vs. FTD: {results['binary_accuracies']['C_vs_F']:.2f}%\n\n")
            
            f.write("PER-CLASS RESULTS\n")
            for cls in class_names:
                metrics = results['class_metrics'][cls]
                f.write(f"{cls} ({DISEASE_NAMES[cls]}):\n")
                f.write(f"  Total Samples: {results['class_totals'][cls]}\n")
                f.write(f"  Correctly Classified: {results['class_correct'][cls]}\n")
                
                if cls == 'A':
                    f.write(f"  Misclassified as Control: {results['class_as_other'][cls]['as_C']}\n")
                    f.write(f"  Misclassified as FTD: {results['class_as_other'][cls]['as_F']}\n")
                elif cls == 'C':
                    f.write(f"  Misclassified as AD: {results['class_as_other'][cls]['as_A']}\n")
                    f.write(f"  Misclassified as FTD: {results['class_as_other'][cls]['as_F']}\n")
                else:  # F
                    f.write(f"  Misclassified as AD: {results['class_as_other'][cls]['as_A']}\n")
                    f.write(f"  Misclassified as Control: {results['class_as_other'][cls]['as_C']}\n")
                
                f.write(f"  Precision: {metrics['precision'] * 100:.4f}%\n")
                f.write(f"  Recall (Sensitivity): {metrics['recall'] * 100:.4f}%\n")
                f.write(f"  F1 Score: {metrics['f1'] * 100:.4f}%\n")
                f.write(f"  Accuracy: {metrics['accuracy'] * 100:.4f}%\n")
                f.write(f"  Specificity: {metrics['specificity'] * 100:.4f}%\n\n")
            
            f.write("CONFUSION MATRIX\n")
            confusion = results['confusion_matrix']
            f.write(f"  {'':6s} {class_names[0]:^6s} {class_names[1]:^6s} {class_names[2]:^6s}\n")
            for i, row in enumerate(confusion):
                f.write(f"  {class_names[i]:6s} {row[0]:^6d} {row[1]:^6d} {row[2]:^6d}\n")
        
        print(f"Results for {data_type} saved to {results_file}")
        return results
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None

def main():
    """Main function to evaluate both test sets"""
    # Create timestamp for output directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f'results_{timestamp}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Use the specified model path
    model_path = 'mvt_timegraph_20250325_021450\\models\\best_model_overall.pth'
    
    print(f"Using model: {model_path}")
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Test both datasets
    results = {}
    
    # Test cross-subject
    print("\nEvaluating cross-subject generalization (test_cross)...")
    results['test_cross'] = test_model('test_cross', model_path)
    
    # Test within-subject
    print("\nEvaluating within-subject performance (test_within)...")
    results['test_within'] = test_model('test_within', model_path)
    
    # Create comparison summary
    if results['test_cross'] or results['test_within']:
        summary_file = os.path.join(output_dir, 'comparison_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("COMPARISON OF TEST RESULTS\n\n")
            f.write(f"Model: {model_path}\n\n")
            
            # Table headers
            f.write(f"{'Dataset':15s} {'Accuracy':10s} {'Precision':10s} {'Recall':10s} {'F1 Score':10s}\n")
            f.write("-"*60 + "\n")
            
            # Dataset results
            for dataset_name, result in results.items():
                if result:
                    f.write(f"{dataset_name:15s} {result['accuracy']:.2f}%{'':<5s} "
                            f"{result['precision']:.2f}%{'':<2s} "
                            f"{result['recall']:.2f}%{'':<2s} "
                            f"{result['f1']:.2f}%\n")
            
            # Add binary classification accuracies
            f.write("\nBINARY CLASSIFICATION ACCURACIES\n")
            f.write(f"{'Dataset':15s} {'AD vs CN':12s} {'AD vs FTD':12s} {'CN vs FTD':12s}\n")
            f.write("-"*60 + "\n")
            
            for dataset_name, result in results.items():
                if result:
                    f.write(f"{dataset_name:15s} "
                            f"{result['binary_accuracies']['A_vs_C']:.2f}%{'':<5s} "
                            f"{result['binary_accuracies']['A_vs_F']:.2f}%{'':<5s} "
                            f"{result['binary_accuracies']['C_vs_F']:.2f}%\n")
            
            # Per-class comparison
            class_names = ['A', 'C', 'F']
            
            f.write("\nPER-CLASS COMPARISON\n\n")
            
            for cls in class_names:
                f.write(f"Class: {cls} ({DISEASE_NAMES[cls]})\n")
                f.write("-"*60 + "\n")
                f.write(f"{'Dataset':15s} {'Precision':10s} {'Recall':10s} {'F1 Score':10s} {'Accuracy':10s} {'Specificity':12s}\n")
                f.write("-"*70 + "\n")
                
                for dataset_name, result in results.items():
                    if result and cls in result['class_metrics']:
                        metrics = result['class_metrics'][cls]
                        f.write(f"{dataset_name:15s} "
                                f"{metrics['precision']*100:.2f}%{'':<2s} "
                                f"{metrics['recall']*100:.2f}%{'':<2s} "
                                f"{metrics['f1']*100:.2f}%{'':<2s} "
                                f"{metrics['accuracy']*100:.2f}%{'':<2s} "
                                f"{metrics['specificity']*100:.2f}%\n")
                f.write("\n")
        
        print(f"\nComparison summary saved to {summary_file}")
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()