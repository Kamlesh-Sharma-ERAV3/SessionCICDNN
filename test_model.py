import torch
import torch.nn as nn
from torchvision import datasets, transforms
from train import SimpleCNN
import glob
import pytest
from tqdm import tqdm
import os
import numpy as np
import cv2
from train import transform, load_and_preprocess_image
import albumentations as A

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

def test_rotation_angle():
    """Test if rotation stays within specified limits"""
    # Create a simple test image with a horizontal line
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[45:55, :, :] = 255  # Thicker horizontal line for better detection
    
    # Apply augmentation multiple times to ensure consistent behavior
    max_height_diff = 0
    num_tests = 10
    
    for _ in range(num_tests):
        augmented = transform(image=test_image)['image']
        # Find white pixels
        white_pixels = np.where(augmented[:, :, 0] == 255)[0]
        if len(white_pixels) > 0:
            height_diff = max(white_pixels) - min(white_pixels)
            max_height_diff = max(max_height_diff, height_diff)
    
    # Calculate maximum expected height difference for 30-degree rotation
    # Using trigonometry: opposite = adjacent * tan(theta)
    max_expected_diff = int(100 * np.tan(np.radians(31)))  # 100 is image width
    
    assert max_height_diff < max_expected_diff, \
        f"Rotation angle too large. Got {np.degrees(np.arctan(max_height_diff/100)):.1f}°, expected ≤30°"

def test_image_shape_preservation():
    """Test if the augmented image maintains the same shape as input"""
    test_image = np.zeros((64, 64, 3), dtype=np.uint8)
    augmented = transform(image=test_image)['image']
    
    assert test_image.shape == augmented.shape
    assert augmented.dtype == np.uint8

def test_rotation_boundaries():
    """Test if rotation respects both positive and negative angle limits"""
    # Create a test image with a cross pattern
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[45:55, :, :] = 255  # Horizontal line
    test_image[:, 45:55, :] = 255  # Vertical line
    
    # Collect multiple augmentations to check rotation distribution
    rotations = []
    num_samples = 50
    
    for _ in range(num_samples):
        augmented = transform(image=test_image)['image']
        
        # Find the angle of rotation by analyzing the horizontal line
        y_coords, _ = np.where(augmented[:, :, 0] == 255)
        if len(y_coords) > 0:
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            height_diff = max_y - min_y
            # Approximate angle using arctan
            if height_diff > 0:
                angle = np.abs(np.degrees(np.arctan(height_diff / 100)))
                rotations.append(angle)
    
    # Check if all rotations are within limits
    max_rotation = max(rotations) if rotations else 0
    assert max_rotation <= 31, f"Maximum rotation {max_rotation:.1f}° exceeds limit of 30°"
    
    # Check if we have a good distribution of angles
    assert len(rotations) > 0, "No valid rotations detected"
    mean_rotation = np.mean(rotations)
    assert mean_rotation > 5, f"Mean rotation {mean_rotation:.1f}° seems too low"

def test_horizontal_flip():
    """Test if horizontal flipping works correctly"""
    # Create a test image with an asymmetric pattern
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Create an 'L' shape on the left side
    test_image[30:70, 20:30, :] = 255  # Vertical line
    test_image[60:70, 20:50, :] = 255  # Horizontal line
    
    # Apply augmentation
    augmented = transform(image=test_image)['image']
    
    # The 'L' shape should now be on the right side
    # Check vertical line position
    vertical_line_original = test_image[40, 20:30, 0]  # Sample from original vertical line
    vertical_line_flipped = augmented[40, -30:-20, 0]  # Sample from expected flipped position
    
    # Check horizontal line position
    horiz_line_original = test_image[60:70, 20:50, 0]  # Sample from original horizontal line
    horiz_line_flipped = augmented[60:70, -50:-20, 0]  # Sample from expected flipped position
    
    # Verify the flip
    assert np.array_equal(vertical_line_original, vertical_line_flipped), "Vertical line not correctly flipped"
    assert np.array_equal(horiz_line_original, horiz_line_flipped), "Horizontal line not correctly flipped"
    
    # Verify that the pattern is not in the original position
    assert not np.array_equal(test_image[:, :20, :], augmented[:, :20, :]), "Image should be flipped"

def test_brightness_adjustment():
    """Test if brightness adjustment works"""
    # Create a gray test image
    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 127
    
    # Apply augmentation
    augmented = transform(image=test_image)['image']
    
    # Check if brightness changed
    original_mean = np.mean(test_image)
    augmented_mean = np.mean(augmented)
    
    # Verify brightness changed but within reasonable limits
    assert abs(original_mean - augmented_mean) > 1, "Brightness should change"
    assert abs(original_mean - augmented_mean) < 51, "Brightness change should be within limits"

def test_gaussian_noise():
    """Test if Gaussian noise is added"""
    # Create a uniform test image
    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 127
    
    # Apply augmentation
    augmented = transform(image=test_image)['image']
    
    # Calculate standard deviation of the difference
    diff = augmented.astype(float) - test_image.astype(float)
    std_dev = np.std(diff)
    
    # Verify noise was added
    assert std_dev > 0, "No noise detected"
    assert std_dev < 50, "Noise level too high"

def test_random_crop():
    """Test if random crop works correctly"""
    # Create test image with specific size
    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 127
    
    # Apply augmentation
    augmented = transform(image=test_image)['image']
    
    # Check if dimensions are correct
    assert augmented.shape == (80, 80, 3), "Crop dimensions incorrect"
    assert augmented.dtype == np.uint8, "Image type should remain uint8"

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__]) 