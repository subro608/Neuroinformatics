import torch
import os
import json
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

# Import our custom modules
from eeg_multiscale_spectral import create_model, MultiScaleGraphTransformer
from eeg_dataset_multiscale_spectral import MultiScaleSpectralEEGDataset

# Ignore warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Create output directories
for dir_path in ['test_results/multiscale', 'connectivity']:
   if not os.path.exists(dir_path):
       os.makedirs(dir_path)

# Model parameters
num_channels = 19
num_classes = 3  # A (AD), C (Control), F (FTD)
dim = 256
scales = [3, 4, 5]  # Multi-scale approach

# Define abbreviations for clarity
DISEASE_NAMES = {
    'A': 'Alzheimer\'s Disease (AD)',
    'C': 'Control (CN)',
    'F': 'Frontotemporal Dementia (FTD)'
}

# Default electrode positions (if not available from dataset module)
DEFAULT_ELECTRODE_POSITIONS = {
    'Fp1': {'x': -0.25, 'y': 0.7},
    'Fp2': {'x': 0.25, 'y': 0.7},
    'F3': {'x': -0.4, 'y': 0.5},
    'F4': {'x': 0.4, 'y': 0.5},
    'C3': {'x': -0.5, 'y': 0},
    'C4': {'x': 0.5, 'y': 0},
    'P3': {'x': -0.4, 'y': -0.5},
    'P4': {'x': 0.4, 'y': -0.5},
    'O1': {'x': -0.25, 'y': -0.7},
    'O2': {'x': 0.25, 'y': -0.7},
    'F7': {'x': -0.65, 'y': 0.3},
    'F8': {'x': 0.65, 'y': 0.3},
    'T3': {'x': -0.75, 'y': 0},
    'T4': {'x': 0.75, 'y': 0},
    'T5': {'x': -0.65, 'y': -0.3},
    'T6': {'x': 0.65, 'y': -0.3},
    'Fz': {'x': 0, 'y': 0.5},
    'Cz': {'x': 0, 'y': 0},
    'Pz': {'x': 0, 'y': -0.5}
}

# Try to import electrode positions, fallback to default if not available
try:
    from dataset import ELECTRODE_POSITIONS
except ImportError:
    ELECTRODE_POSITIONS = DEFAULT_ELECTRODE_POSITIONS
    print("Warning: Using default electrode positions. Import from dataset module failed.")

def visualize_brain_connectivity(adjacency, electrode_positions, save_path, title="Brain Connectivity"):
    """Visualize brain connectivity from adjacency matrix"""
    plt.figure(figsize=(12, 10))
    
    # Plot electrodes
    x_coords = []
    y_coords = []
    labels = []
    
    for label, pos in electrode_positions.items():
        x_coords.append(pos['x'])
        y_coords.append(pos['y'])
        labels.append(label)
    
    plt.scatter(x_coords, y_coords, s=120, c='blue', alpha=0.7)
    
    # Add electrode labels
    for i, label in enumerate(labels):
        plt.text(x_coords[i], y_coords[i], label, 
                fontsize=12, ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Plot head outline
    circle = plt.Circle((0, 0), 0.85, fill=False, color='black', linewidth=2)
    plt.gca().add_patch(circle)
    
    # Add connections based on adjacency weights
    adj_np = adjacency.cpu().numpy() if torch.is_tensor(adjacency) else adjacency
    
    # Normalize adjacency
    max_weight = np.max(adj_np)
    min_weight = np.min(adj_np)
    
    # Only draw connections above threshold
    threshold = max_weight * 0.5
    
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):  # Only draw each connection once
            weight = adj_np[i, j]
            if weight > threshold:
                # Normalize weight for line width and color
                norm_weight = (weight - min_weight) / (max_weight - min_weight)
                plt.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]],
                        'r-', alpha=min(0.9, norm_weight),
                        linewidth=norm_weight * 5)
    
    # Set plot properties
    plt.axis('equal')
    plt.title(title)
    plt.grid(False)
    plt.axis('off')
    
    # Fix colorbar issue by explicitly creating the mappable with an existing axes
    ax = plt.gca()
    norm = plt.Normalize(vmin=min_weight, vmax=max_weight)
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
    sm.set_array([])
    
    # Create colorbar with explicit axes position
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Connection Strength', fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def find_model_checkpoint():
    """Find the latest model checkpoint in common directories"""
    # Define potential model directories to search
    potential_dirs = [
        'spectral_multiscale_20250323_210620\models',
    ]
    
    # Check for best model first
    for dir_path in potential_dirs:
        best_model_path = os.path.join(dir_path, 'best_model_overall.pth')
        if os.path.exists(best_model_path):
            print(f"Found best model at {best_model_path}")
            return best_model_path
    
    # Check for fold-specific models
    for dir_path in potential_dirs:
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                if file.startswith('best_model_fold') and file.endswith('.pth'):
                    model_path = os.path.join(dir_path, file)
                    print(f"Found fold model at {model_path}")
                    return model_path
    
    # Check for checkpoints
    for dir_path in potential_dirs:
        checkpoint_dir = dir_path.replace('models', 'checkpoints')
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if checkpoints:
                latest = sorted(checkpoints)[-1]
                checkpoint_path = os.path.join(checkpoint_dir, latest)
                print(f"Found checkpoint at {checkpoint_path}")
                return checkpoint_path
    
    return None

def test_model(data_type='test_cross', model_path=None, model_type='multiscale'):
    """Test model on the specified dataset"""
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f'test_results/{model_type}_{timestamp}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Select model checkpoint
    if model_path is None:
        model_path = find_model_checkpoint()
        if model_path is None:
            raise FileNotFoundError("Could not find model checkpoint. Please specify model_path parameter.")

    print(f"Loading model from: {model_path}")
    
    try:
        # Initialize model
        model = create_model(
            model_type=model_type,
            num_channels=num_channels, 
            num_classes=num_classes, 
            dim=dim,
            num_clusters=scales
        )
        
        # Load model with appropriate method based on checkpoint format
        try:
            # Try loading as state dict
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Check if checkpoint is a state dict or contains one
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Loaded model from state_dict")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded model directly")
                
        except Exception as e:
            print(f"Error in first loading attempt: {e}")
            # Try loading as full model
            model = torch.load(model_path, map_location='cpu')
            print("Loaded full model object")
        
        # Print model info for debugging
        print(f"Model type: {type(model).__name__}")
        print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

    # Data loading - first check if data directory exists
    data_dir = 'model-data'
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} not found. Please ensure the correct data path.")
    
    data_file = 'labels.json'
    if not os.path.exists(os.path.join(data_dir, data_file)):
        raise FileNotFoundError(f"Labels file {data_file} not found in {data_dir}.")

    # Load data info
    with open(os.path.join(data_dir, data_file), 'r') as file:
        data_info = json.load(file)

    # Filter data based on type
    test_data = [d for d in data_info if d['type'] == data_type]
    if not test_data:
        # If no data of specified type, try to use any available test data
        print(f"No data found with type '{data_type}', searching for alternative test data...")
        test_data = [d for d in data_info if 'test' in d['type']]
        if test_data:
            data_type = test_data[0]['type']
            print(f"Using {data_type} data instead.")
        else:
            raise ValueError("No test data available in the dataset.")
    
    # For binary comparison if needed
    binary_a_vs_c = [d for d in test_data if d['label'] in ['A', 'C']]
    binary_a_vs_f = [d for d in test_data if d['label'] in ['A', 'F']]
    binary_c_vs_f = [d for d in test_data if d['label'] in ['C', 'F']]

    # Create dataset
    try:
        test_dataset = MultiScaleSpectralEEGDataset(data_dir, test_data, scales=scales)
        print(f"Successfully created dataset with {len(test_dataset)} samples")
        
        # Check if any samples are available
        if len(test_dataset) == 0:
            raise ValueError("Dataset contains no samples.")
        
        # Create dataloader
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        # Additional diagnostics
        print(f"Data directory contents: {os.listdir(data_dir)}")
        raise

    # Count samples per class
    total_a = sum(1 for d in test_data if d['label'] == 'A')
    total_c = sum(1 for d in test_data if d['label'] == 'C')
    total_f = sum(1 for d in test_data if d['label'] == 'F')

    print(f'\nTesting on {data_type} data:')
    print(f'Test dataset: {len(test_dataset)} samples')
    print(f'Test dataloader: {len(test_dataloader)} batches')
    print(f'Test dataloader batch size: {test_dataloader.batch_size}')
    print(f'Class distribution - A: {total_a}, C: {total_c}, F: {total_f}\n')

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    # Initialize metrics storage
    all_labels = []
    all_preds = []
    
    # Store some adjacency matrices for visualization
    adjacency_by_class = {
        'A': [],  # Alzheimer's
        'C': [],  # Control
        'F': []   # FTD
    }
    class_labels = ['A', 'C', 'F']

    # Inference time tracking
    inference_times = []

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
        
        # Wrap in try/except for better error handling
        try:
            for batch_idx, (inputs_dict, labels) in enumerate(tqdm(test_dataloader, desc=f"Testing {model_type} model")):
                # Debug info for first batch
                if batch_idx == 0:
                    print(f"Keys in batch: {list(inputs_dict.keys())}")
                    for key in inputs_dict:
                        if isinstance(inputs_dict[key], torch.Tensor):
                            print(f"{key} shape: {inputs_dict[key].shape}")
                
                # For multiscale model, we need all scales
                scale_features = {k: v.to(device) for k, v in inputs_dict.items() 
                                if k.startswith('scale_')}
                adjacency = inputs_dict['adjacency'].to(device)
                
                # Start timing
                start_time = time.time()
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast():
                    # Use embeddings_dict for multiscale model
                    outputs = model(embeddings_dict=scale_features, adjacency=adjacency)
                
                # Record inference time
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Process outputs
                labels = labels.to(device)
                _, predicted = torch.max(outputs, 1)
                
                # Store some adjacency matrices for visualization
                for i in range(min(3, labels.size(0))):  # Store up to 3 samples per batch
                    class_idx = labels[i].item()
                    if len(adjacency_by_class[class_labels[class_idx]]) < 5:  # Limit to 5 samples per class
                        adjacency_by_class[class_labels[class_idx]].append(adjacency[i].cpu().numpy())
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Store predictions
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

                # Count predictions
                for i in range(labels.size(0)):
                    if labels[i] == 0:  # True class is A (Alzheimer's)
                        if predicted[i] == 0:
                            correct_a += 1
                        elif predicted[i] == 1:
                            a_as_c += 1
                        else:
                            a_as_f += 1
                    elif labels[i] == 1:  # True class is C (Control)
                        if predicted[i] == 1:
                            correct_c += 1
                        elif predicted[i] == 0:
                            c_as_a += 1
                        else:
                            c_as_f += 1
                    else:  # True class is F (FTD)
                        if predicted[i] == 2:
                            correct_f += 1
                        elif predicted[i] == 0:
                            f_as_a += 1
                        else:
                            f_as_c += 1
        except Exception as e:
            print(f"Error during testing: {e}")
            import traceback
            traceback.print_exc()
            return None, None

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

    # Binary classification accuracies - particularly important for Alzheimer's
    accuracy_ad_cn = (correct_a + correct_c) / (total_a + total_c) if (total_a + total_c) > 0 else 0
    accuracy_ftd_cn = (correct_c + correct_f) / (total_c + total_f) if (total_c + total_f) > 0 else 0
    accuracy_ad_ftd = (correct_a + correct_f) / (total_a + total_f) if (total_a + total_f) > 0 else 0

    # Calculate per-class metrics
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, labels=[0, 1, 2], zero_division=0
        )
        
        # Create a dictionary for easy reference
        class_metrics = {
            'A': {'precision': precision[0], 'recall': recall[0], 'f1': f1[0]},
            'C': {'precision': precision[1], 'recall': recall[1], 'f1': f1[1]},
            'F': {'precision': precision[2], 'recall': recall[2], 'f1': f1[2]}
        }
        
        # Add sensitivities and specificities
        class_metrics['A']['sensitivity'] = correct_a / total_a if total_a > 0 else 0
        class_metrics['A']['specificity'] = (correct_c + correct_f) / (total_c + total_f) if (total_c + total_f) > 0 else 0
        
        class_metrics['C']['sensitivity'] = correct_c / total_c if total_c > 0 else 0
        class_metrics['C']['specificity'] = (correct_a + correct_f) / (total_a + total_f) if (total_a + total_f) > 0 else 0
        
        class_metrics['F']['sensitivity'] = correct_f / total_f if total_f > 0 else 0
        class_metrics['F']['specificity'] = (correct_a + correct_c) / (total_a + total_c) if (total_a + total_c) > 0 else 0

        # Mean metrics
        mAP = np.mean([class_metrics[c]['precision'] for c in class_metrics])
        mAR = np.mean([class_metrics[c]['recall'] for c in class_metrics])
        mF1 = np.mean([class_metrics[c]['f1'] for c in class_metrics])
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        # Set default metrics in case of error
        mAP = 0
        mAR = 0 
        mF1 = 0
        class_metrics = {
            'A': {'precision': 0, 'recall': 0, 'f1': 0, 'sensitivity': 0, 'specificity': 0},
            'C': {'precision': 0, 'recall': 0, 'f1': 0, 'sensitivity': 0, 'specificity': 0},
            'F': {'precision': 0, 'recall': 0, 'f1': 0, 'sensitivity': 0, 'specificity': 0}
        }
    
    # Average inference time
    avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms

    # Save results to file
    results_file = f'{output_dir}/results_{model_type}_{data_type}.txt'
    with open(results_file, 'w') as f:
        f.write(f'{model_type.upper()} Model Test Results for {data_type}\n\n')
        f.write(f'Model Path: {model_path}\n')
        f.write(f'Scales: {scales}\n')
        f.write(f'Average Inference Time: {avg_inference_time:.2f} ms per batch\n\n')
        f.write(f'Correct: {correct}, Total: {total}\n')
        f.write(f'Correct A: {correct_a}, A as C: {a_as_c}, A as F: {a_as_f}, Total A: {total_a}\n')
        f.write(f'Correct C: {correct_c}, C as A: {c_as_a}, C as F: {c_as_f}, Total C: {total_c}\n')
        f.write(f'Correct F: {correct_f}, F as A: {f_as_a}, F as C: {f_as_c}, Total F: {total_f}\n')
        f.write(f'Overall Accuracy: {100 * accuracy:.4f}%\n\n')
        f.write('Binary Classification Accuracies:\n')
        f.write(f'AD vs. CN: {100 * accuracy_ad_cn:.4f}%\n')
        f.write(f'FTD vs. CN: {100 * accuracy_ftd_cn:.4f}%\n')
        f.write(f'AD vs. FTD: {100 * accuracy_ad_ftd:.4f}%\n\n')
        f.write('Per-Class Metrics:\n')
        
        for cls in ['A', 'C', 'F']:
            metrics = class_metrics[cls]
            disease = 'Alzheimer\'s' if cls == 'A' else 'Control' if cls == 'C' else 'FTD'
            f.write(f'Class {cls} ({disease}) - Precision: {100 * metrics["precision"]:.4f}%, '
                    f'Recall: {100 * metrics["recall"]:.4f}%, F1: {100 * metrics["f1"]:.4f}%, '
                    f'Sensitivity: {100 * metrics["sensitivity"]:.4f}%, '
                    f'Specificity: {100 * metrics["specificity"]:.4f}%\n')
        
        f.write('\nMean Metrics:\n')
        f.write(f'mAP: {100 * mAP:.4f}%, mAR: {100 * mAR:.4f}%, mF1: {100 * mF1:.4f}%\n')
        f.write(f'Inference Speed: {1000/avg_inference_time:.2f} samples/second\n')

    # Print results
    print(f'\nResults saved to {results_file}')

    # Plot confusion matrix
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues',
                    xticklabels=['AD', 'CN', 'FTD'], yticklabels=['AD', 'CN', 'FTD'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{model_type.upper()} Model Confusion Matrix ({data_type})')
        plt.savefig(f'{output_dir}/confusion_matrix_{model_type}_{data_type}.png')
        plt.close()
        print(f"Saved confusion matrix to {output_dir}/confusion_matrix_{model_type}_{data_type}.png")
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")
    
    # Plot connectivity differences between classes
    try:
        if adjacency_by_class['A'] and adjacency_by_class['C']:
            # AD vs Control
            ad_avg = np.mean(adjacency_by_class['A'], axis=0)
            cn_avg = np.mean(adjacency_by_class['C'], axis=0)
            diff_ad_cn = ad_avg - cn_avg
            
            # Visualize as a brain map
            visualize_brain_connectivity(
                diff_ad_cn, 
                ELECTRODE_POSITIONS,
                f'{output_dir}/brain_diff_AD_CN_{data_type}.png',
                f'Connectivity Difference: AD vs CN ({data_type})'
            )
            print(f"Saved AD vs CN connectivity difference visualization")
    except Exception as e:
        print(f"Error visualizing connectivity differences: {e}")
        
    # Additional comparison: FTD vs Control if available
    try:
        if adjacency_by_class['F'] and adjacency_by_class['C']:
            ftd_avg = np.mean(adjacency_by_class['F'], axis=0)
            cn_avg = np.mean(adjacency_by_class['C'], axis=0)
            diff_ftd_cn = ftd_avg - cn_avg
            
            # Visualize as a brain map
            visualize_brain_connectivity(
                diff_ftd_cn, 
                ELECTRODE_POSITIONS,
                f'{output_dir}/brain_diff_FTD_CN_{data_type}.png',
                f'Connectivity Difference: FTD vs CN ({data_type})'
            )
            print(f"Saved FTD vs CN connectivity difference visualization")
    except Exception as e:
        print(f"Error visualizing FTD vs CN connectivity differences: {e}")
    
    return accuracy, mF1

def main():
    """Main function to run the testing script"""
    print("EEG Classification Model Testing")
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Default settings
    model_type = 'multiscale'  # Use multiscale as default
    model_path = find_model_checkpoint()  # Automatically find model
    
    # Test on both datasets
    for data_type in ['test_cross', 'test_within']:
        print(f"\n{'-'*50}")
        print(f"Testing {model_type.upper()} model on {data_type} data...")
        print(f"Model path: {model_path if model_path else 'Not found - will search'}")
        
        # Run the test
        accuracy, f1 = test_model(
            data_type=data_type,
            model_path=model_path,
            model_type=model_type
        )
        
        if accuracy is not None:
            print(f"\nFinal Results for {model_type.upper()} model on {data_type}:")
            print(f"Accuracy: {accuracy * 100:.2f}%")
            print(f"F1 Score: {f1 * 100:.2f}%")
    
    print("\nTesting completed!")

if __name__ == "__main__":
    import argparse
    import sys
    
    # Check if arguments are provided
    if len(sys.argv) > 1:
        # Use argparse for command-line options
        parser = argparse.ArgumentParser(description='Test EEG classification models')
        parser.add_argument('--model_type', type=str, default='multiscale', 
                            choices=['transformer', 'graph', 'multiscale'],
                            help='Model type to test')
        parser.add_argument('--model_path', type=str, default=None,
                            help='Path to model checkpoint')
        
        args = parser.parse_args()
        
        # Run tests for both data types
        for data_type in ['test_cross', 'test_within']:
            print(f"\n{'-'*50}")
            print(f"Testing {args.model_type.upper()} model on {data_type} data...")
            
            accuracy, f1 = test_model(
                data_type=data_type,
                model_path=args.model_path,
                model_type=args.model_type
            )
            
            if accuracy is not None:
                print(f"\nFinal Results for {args.model_type.upper()} model on {data_type}:")
                print(f"Accuracy: {accuracy * 100:.2f}%")
                print(f"F1 Score: {f1 * 100:.2f}%")
        
        print("\nTesting completed successfully!")
    else:
        # If no arguments provided, run the main function
        main()