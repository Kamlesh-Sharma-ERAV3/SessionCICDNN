import torch
import torch.nn as nn
from torchvision import datasets, transforms
from train import SimpleCNN
import glob
import pytest
from tqdm import tqdm
import os

def get_latest_model():
    model_files = glob.glob('models/mnist_model_*.pth')
    return max(model_files) if model_files else None

def get_device():
    if torch.backends.mps.is_available():
        try:
            # Test MPS availability with a small tensor
            device = torch.device("mps")
            torch.zeros(1).to(device)
            return device
        except:
            print("Warning: MPS (Metal) device found but unusable, falling back to CPU")
            return torch.device("cpu")
    return torch.device("cpu")

def get_best_model():
    """Get the model with the highest accuracy or create new if none exists"""
    best_model_path = 'models/mnist_model_best.pth'
    latest_model_path = 'models/mnist_model_latest.pth'
    
    if not os.path.exists(latest_model_path):
        return None, None
        
    latest_checkpoint = torch.load(latest_model_path)
    latest_accuracy = latest_checkpoint['accuracy']
    
    # If best model doesn't exist or latest is better, update best
    if not os.path.exists(best_model_path):
        torch.save(latest_checkpoint, best_model_path)
        return latest_model_path, latest_accuracy
        
    best_checkpoint = torch.load(best_model_path)
    best_accuracy = best_checkpoint['accuracy']
    
    if latest_accuracy > best_accuracy:
        print(f"\nNew best model found! Accuracy: {latest_accuracy:.2f}% (previous: {best_accuracy:.2f}%)")
        torch.save(latest_checkpoint, best_model_path)
        return latest_model_path, latest_accuracy
    else:
        print(f"\nKeeping previous best model. Best accuracy: {best_accuracy:.2f}% (current: {latest_accuracy:.2f}%)")
        return best_model_path, best_accuracy

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
    print("\n" + "="*70)
    print("RUNNING MODEL ACCURACY TEST")
    print("="*70 + "\n")
    
    device = get_device()
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    # Load best model
    model_path, best_accuracy = get_best_model()
    assert model_path is not None, "No model found"
    print(f"Using model: {model_path}")
    
    model = SimpleCNN().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
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