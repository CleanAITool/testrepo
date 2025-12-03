"""
CleanAI - VGG11 on CIFAR-10 Accuracy Comparison
================================================

VGG11 modeli ile CIFAR-10 üzerinde pruning öncesi/sonrası accuracy karşılaştırması.
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


class VGG11(nn.Module):
    """VGG11 architecture for CIFAR-10"""
    def __init__(self, num_classes=10):
        super(VGG11, self).__init__()
        # VGG11 config: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_cifar10(batch_size=128):
    """CIFAR-10 dataset yükle"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return trainloader, testloader


def train_model(model, trainloader, device, epochs=5):
    """Model eğit"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 100 == 99:
                acc = 100.0 * correct / total
                print(f'  Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(trainloader)}], '
                      f'Loss: {running_loss/100:.4f}, Acc: {acc:.2f}%')
                running_loss = 0.0
        
        scheduler.step()
        print(f'  Epoch {epoch+1} completed')


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
    print("CleanAI - VGG11 on CIFAR-10 Accuracy Comparison".center(70))
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    # Dataset yükle
    print("\nLoading CIFAR-10 dataset...")
    trainloader, testloader = load_cifar10(batch_size=128)
    print(f"✓ Train samples: {len(trainloader.dataset)}")
    print(f"✓ Test samples: {len(testloader.dataset)}")
    
    # Model oluştur ve eğit
    print("\nCreating and training VGG11...")
    model = VGG11(num_classes=10).to(device)
    original_params = count_parameters(model)
    print(f"✓ Model created with {original_params:,} parameters")
    
    print("\nTraining original model (5 epochs)...")
    train_model(model, trainloader, device, epochs=5)
    
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
    
    # Output layer'ı ignore et
    output_layer = model.classifier[-1]
    
    example_inputs = torch.randn(1, 3, 32, 32).to(device)
    pruner = StructuredPruner(
        model=model,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=0.4,  # %40 pruning
        device=device,
        ignored_layers=[output_layer]
    )
    
    # Activation topla
    print("\nCollecting activations...")
    pruner.collect_activations(trainloader, max_batches=50)
    
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
    
    print("\nFine-tuning for 5 epochs...")
    train_model(pruned_model, trainloader, device, epochs=5)
    
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
    elif accuracy_drop < 3.0 and param_reduction > 30:
        print("\n✓ Great result! Good compression with acceptable accuracy loss.")
    else:
        print(f"\n✓ Completed. Accuracy drop: {accuracy_drop:.2f}%, Compression: {param_reduction:.2f}%")
    
    # Model kaydet
    print("\nSaving pruned model...")
    torch.save(pruned_model.state_dict(), 'pruned_vgg11_cifar10.pth')
    print("✓ Model saved to 'pruned_vgg11_cifar10.pth'")


if __name__ == '__main__':
    main()
