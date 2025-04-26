import torch
import eeg_multiscale_spectral

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Print parameters by layer/module for detailed analysis
    print(f"\nParameter count by module:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name}: {module_params:,} parameters ({module_params/total_params*100:.2f}%)")
    
    # Format with commas for readability
    print(f"\nTotal trainable parameters: {total_params:,}")
    
    return total_params

def test_model_parameters():
    # Define model parameters
    num_channels = 19
    num_classes = 3
    dim = 256
    scales = [3, 4, 5]
    
    # Test MultiScale model
    multiscale_model = eeg_multiscale_spectral.create_model(
        model_type='multiscale',
        num_channels=num_channels, 
        num_classes=num_classes, 
        dim=dim,
        num_clusters=scales
    )
    print("===== MultiScale Model Parameters =====")
    multiscale_params = count_parameters(multiscale_model)
    
    # Test Graph model
    graph_model = eeg_multiscale_spectral.create_model(
        model_type='graph',
        num_channels=num_channels, 
        num_classes=num_classes, 
        dim=dim,
        num_clusters=5
    )
    print("\n===== Graph Model Parameters =====")
    graph_params = count_parameters(graph_model)
    
    # Test Transformer model
    transformer_model = eeg_multiscale_spectral.create_model(
        model_type='transformer',
        num_channels=num_channels, 
        num_classes=num_classes, 
        dim=dim,
        num_clusters=5
    )
    print("\n===== Transformer Model Parameters =====")
    transformer_params = count_parameters(transformer_model)
    
    # Compare models
    print("\n===== Model Comparison =====")
    print(f"MultiScale model: {multiscale_params:,} parameters")
    print(f"Graph model:      {graph_params:,} parameters")
    print(f"Transformer model: {transformer_params:,} parameters")

if __name__ == "__main__":
    test_model_parameters()