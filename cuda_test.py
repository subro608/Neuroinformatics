import torch

def test_cuda():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # Get the number of CUDA devices
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
        
        # Get the name of the current CUDA device
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA device name: {device_name}")
        
        # Try to create a tensor on CUDA
        try:
            # Create a small tensor and move it to CUDA
            x = torch.ones(5, 5).cuda()
            # Perform a simple operation
            y = x + x
            print(f"Tensor created on CUDA - device: {y.device}")
            print("CUDA test successful!")
            return True
        except Exception as e:
            print(f"Error using CUDA: {e}")
            return False
    else:
        print("CUDA is not available on this system")
        return False

if __name__ == "__main__":
    test_cuda()