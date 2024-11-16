import torch
import torch.nn as nn
from torchvision import datasets, transforms
from train import SimpleCNN
import glob
import pytest
from tqdm import tqdm

def get_latest_model():
    model_files = glob.glob('models/mnist_model_*.pth')
    return max(model_files) if model_files else None

def test_model_architecture():
    model = SimpleCNN()
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Output shape should be (1, 10)"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, should be less than 25000"

def test_model_accuracy():
    print("\n" + "="*70)  # Add a separator line
    print("RUNNING MODEL ACCURACY TEST")
    print("="*70 + "\n")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    # Load latest model
    model_path = get_latest_model()
    assert model_path is not None, "No trained model found"
    print(f"Using model: {model_path}")
    
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    correct = 0
    total = 0
    
    print("\nEvaluating model on test dataset...")
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print("\n" + "="*70)
    print(f"TEST RESULTS:")
    print(f"Total test samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("="*70 + "\n")
    
    assert accuracy > 80, f"Model accuracy is {accuracy:.2f}%, should be > 80%"

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__]) 