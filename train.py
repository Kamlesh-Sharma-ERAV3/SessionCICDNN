import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime
import os
import ssl
import urllib.request
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)  # 28x28 -> 26x26
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3)  # 26x26 -> 24x24
        self.pool = nn.MaxPool2d(2, 2)  # 24x24 -> 12x12
        self.pool2 = nn.MaxPool2d(2, 2)  # 12x12 -> 6x6
        self.fc1 = nn.Linear(8 * 6 * 6, 20)
        self.fc2 = nn.Linear(20, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(-1, 8 * 6 * 6)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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

def train():
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Train for 1 epoch
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    print("\nStarting training...")
    progress_bar = tqdm(train_loader, desc='Training', unit='batch')
    
    for data, target in progress_bar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Update progress bar with loss and accuracy
        accuracy = 100 * correct / total
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'accuracy': f'{accuracy:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    final_accuracy = 100 * correct / total
    print(f'\nTraining completed:')
    print(f'Average loss: {avg_loss:.4f}')
    print(f'Training accuracy: {final_accuracy:.2f}%')
    
    # Save model with timestamp and accuracy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists('models'):
        os.makedirs('models')
    model_path = f'models/mnist_model_{timestamp}_acc{final_accuracy:.1f}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    return model

if __name__ == "__main__":
    train() 