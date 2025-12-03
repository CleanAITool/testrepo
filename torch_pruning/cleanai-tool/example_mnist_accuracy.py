"""
CleanAI - Simple Accuracy Comparison
====================================

Basit CNN modeli ile MNIST üzerinde hızlı accuracy karşılaştırması.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

# cleanai-tool import
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pruner import StructuredPruner
from importance import WeightActivationImportance


class SimpleCNN(nn.Module):
    """Basit CNN modeli"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_mnist(batch_size=128):
    """MNIST dataset yükle"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return trainloader, testloader


def train_model(model, trainloader, device, epochs=3):
    """Model eğit"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 99:
                print(f'  Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(trainloader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0


def evaluate_accuracy(model, dataloader, device):
    """Model accuracy'sini hesapla"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100.0 * correct / total


def count_parameters(model):
    """Model parametrelerini say"""
    return sum(p.numel() for p in model.parameters())


def main():
    print("=" * 70)
    print("CleanAI - Simple Accuracy Comparison (MNIST)".center(70))
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    # Dataset yükle
    print("\nLoading MNIST dataset...")
    trainloader, testloader = load_mnist(batch_size=256)
    print(f"✓ Train samples: {len(trainloader.dataset)}")
    print(f"✓ Test samples: {len(testloader.dataset)}")
    
    # Model oluştur ve eğit
    print("\nCreating and training model...")
    model = SimpleCNN().to(device)
    original_params = count_parameters(model)
    print(f"✓ Model created with {original_params:,} parameters")
    
    print("\nTraining original model (3 epochs)...")
    train_model(model, trainloader, device, epochs=3)
    
    # Original model accuracy
    print("\n" + "=" * 70)
    print("Evaluating Original Model".center(70))
    print("=" * 70)
    
    start_time = time.time()
    original_accuracy = evaluate_accuracy(model, testloader, device)
    original_time = time.time() - start_time
    
    print(f"\n✓ Original Accuracy: {original_accuracy:.2f}%")
    print(f"✓ Inference time: {original_time:.2f}s")
    print(f"✓ Parameters: {original_params:,}")
    
    # Pruning uygula
    print("\n" + "=" * 70)
    print("Applying Structured Pruning".center(70))
    print("=" * 70)
    
    importance = WeightActivationImportance(
        weight_ratio=0.5,
        activation_ratio=0.5,
        p=2,
        normalize=True
    )
    
    example_inputs = torch.randn(1, 1, 28, 28).to(device)
    pruner = StructuredPruner(
        model=model,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=0.5,  # %50 pruning
        device=device,
        ignored_layers=[model.fc2]  # Output layer'ı prune etme
    )
    
    # Activation topla
    print("\nCollecting activations...")
    pruner.collect_activations(trainloader, max_batches=20)
    
    # Pruning yap
    print("\nApplying pruning...")
    pruned_model = pruner.prune()
    pruned_params = count_parameters(pruned_model)
    
    print(f"\n✓ Pruning completed")
    print(f"✓ Parameters: {original_params:,} -> {pruned_params:,}")
    print(f"✓ Reduction: {original_params - pruned_params:,} ({(1 - pruned_params/original_params)*100:.2f}%)")
    
    # Pruned model accuracy (before fine-tuning)
    print("\n" + "=" * 70)
    print("Evaluating Pruned Model (Before Fine-tuning)".center(70))
    print("=" * 70)
    
    pruned_accuracy_before = evaluate_accuracy(pruned_model, testloader, device)
    print(f"\n✓ Pruned Accuracy (before FT): {pruned_accuracy_before:.2f}%")
    
    # Fine-tune pruned model
    print("\n" + "=" * 70)
    print("Fine-tuning Pruned Model".center(70))
    print("=" * 70)
    
    print("\nFine-tuning for 3 epochs...")
    train_model(pruned_model, trainloader, device, epochs=3)
    
    # Pruned model accuracy (after fine-tuning)
    print("\n" + "=" * 70)
    print("Evaluating Pruned Model (After Fine-tuning)".center(70))
    print("=" * 70)
    
    start_time = time.time()
    pruned_accuracy = evaluate_accuracy(pruned_model, testloader, device)
    pruned_time = time.time() - start_time
    
    print(f"\n✓ Pruned Accuracy (after FT): {pruned_accuracy:.2f}%")
    print(f"✓ Inference time: {pruned_time:.2f}s")
    print(f"✓ Parameters: {pruned_params:,}")
    
    # Final karşılaştırma
    print("\n" + "=" * 70)
    print("Final Comparison".center(70))
    print("=" * 70)
    
    print("\n{:<30} {:>15} {:>15} {:>15}".format("Metric", "Original", "Pruned (BF)", "Pruned (AF)"))
    print("-" * 70)
    print("{:<30} {:>14.2f}% {:>14.2f}% {:>14.2f}%".format(
        "Accuracy", original_accuracy, pruned_accuracy_before, pruned_accuracy))
    print("{:<30} {:>15,} {:>15,} {:>15,}".format(
        "Parameters", original_params, pruned_params, pruned_params))
    print("{:<30} {:>14.2f}s {:>15} {:>14.2f}s".format(
        "Inference Time", original_time, "-", pruned_time))
    
    accuracy_drop = original_accuracy - pruned_accuracy
    param_reduction = (1 - pruned_params/original_params) * 100
    speedup = original_time / pruned_time
    
    print("\n{:<30} {:>15}".format("Final Accuracy Drop", f"{accuracy_drop:.2f}%"))
    print("{:<30} {:>15}".format("Parameter Reduction", f"{param_reduction:.2f}%"))
    print("{:<30} {:>15}".format("Speedup", f"{speedup:.2f}x"))
    
    print("\n" + "=" * 70)
    print("Experiment Complete!".center(70))
    print("=" * 70)
    
    # Özet
    if accuracy_drop < 1.0 and param_reduction > 30:
        print("\n✓ Excellent result! High compression with minimal accuracy loss.")
    elif accuracy_drop < 2.0 and param_reduction > 40:
        print("\n✓ Great result! Good compression with acceptable accuracy loss.")
    else:
        print(f"\n✓ Completed. Accuracy drop: {accuracy_drop:.2f}%, Compression: {param_reduction:.2f}%")


if __name__ == '__main__':
    main()
