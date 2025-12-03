"""
CleanAI - Pretrained Model Accuracy Comparison
==============================================

Pretrained ResNet18 modelinde pruning öncesi ve sonrası accuracy karşılaştırması.
"""
import torch
import torch.nn as nn
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


def load_cifar10(batch_size=128, num_workers=2):
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
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return trainloader, testloader


def evaluate_accuracy(model, dataloader, device, max_batches=None):
    """Model accuracy'sini hesapla"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy


def count_parameters(model):
    """Model parametrelerini say"""
    return sum(p.numel() for p in model.parameters())


def main():
    print("=" * 70)
    print("CleanAI - Pretrained Model Accuracy Comparison".center(70))
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    # Dataset yükle
    print("\nLoading CIFAR-10 dataset...")
    trainloader, testloader = load_cifar10(batch_size=128, num_workers=0)
    print(f"✓ Train samples: {len(trainloader.dataset)}")
    print(f"✓ Test samples: {len(testloader.dataset)}")
    
    # Pretrained ResNet18 yükle
    print("\nLoading pretrained ResNet18...")
    model = torchvision.models.resnet18(pretrained=True)
    
    # CIFAR-10 için son katmanı değiştir
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    
    original_params = count_parameters(model)
    print(f"✓ Model loaded with {original_params:,} parameters")
    
    # Fine-tune (kısa bir eğitim - opsiyonel)
    print("\nFine-tuning on CIFAR-10 (1 epoch)...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    model.train()
    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 50 == 0:
            print(f"  Batch [{batch_idx}/{len(trainloader)}] Loss: {loss.item():.4f}")
    
    print("✓ Fine-tuning completed")
    
    # Original model accuracy
    print("\n" + "=" * 70)
    print("Evaluating Original Model".center(70))
    print("=" * 70)
    
    start_time = time.time()
    original_accuracy = evaluate_accuracy(model, testloader, device)
    original_time = time.time() - start_time
    
    print(f"\n✓ Original Model Accuracy: {original_accuracy:.2f}%")
    print(f"✓ Inference time: {original_time:.2f}s")
    print(f"✓ Parameters: {original_params:,}")
    
    # Pruning uygula
    print("\n" + "=" * 70)
    print("Applying Structured Pruning".center(70))
    print("=" * 70)
    
    # Importance scorer oluştur (weight + activation)
    importance = WeightActivationImportance(
        weight_ratio=0.5,
        activation_ratio=0.5,
        p=2,
        normalize=True
    )
    
    # Pruner oluştur
    example_inputs = torch.randn(1, 3, 32, 32).to(device)
    pruner = StructuredPruner(
        model=model,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=0.3,  # %30 pruning
        device=device
    )
    
    # Activation topla (training set'ten subset)
    print("\nCollecting activations from training data...")
    pruner.collect_activations(trainloader, max_batches=50)
    print("✓ Activations collected")
    
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
    print(f"\n✓ Pruned Model Accuracy (before FT): {pruned_accuracy_before:.2f}%")
    
    # Fine-tune pruned model
    print("\n" + "=" * 70)
    print("Fine-tuning Pruned Model".center(70))
    print("=" * 70)
    
    optimizer = torch.optim.SGD(pruned_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    
    for epoch in range(5):
        print(f"\nEpoch {epoch+1}/5")
        pruned_model.train()
        
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = pruned_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"  Batch [{batch_idx}/{len(trainloader)}] Loss: {loss.item():.4f}")
        
        scheduler.step()
        
        # Evaluate after each epoch
        epoch_acc = evaluate_accuracy(pruned_model, testloader, device)
        print(f"  Epoch {epoch+1} Accuracy: {epoch_acc:.2f}%")
    
    print("\n✓ Fine-tuning completed")
    
    # Pruned model accuracy (after fine-tuning)
    print("\n" + "=" * 70)
    print("Evaluating Pruned Model (After Fine-tuning)".center(70))
    print("=" * 70)
    
    start_time = time.time()
    pruned_accuracy = evaluate_accuracy(pruned_model, testloader, device)
    pruned_time = time.time() - start_time
    
    print(f"\n✓ Pruned Model Accuracy (after FT): {pruned_accuracy:.2f}%")
    print(f"✓ Inference time: {pruned_time:.2f}s")
    print(f"✓ Parameters: {pruned_params:,}")
    
    # Final karşılaştırma
    print("\n" + "=" * 70)
    print("Final Comparison".center(70))
    print("=" * 70)
    
    print("\n{:<25} {:>20} {:>20}".format("Metric", "Original", "Pruned"))
    print("-" * 70)
    print("{:<25} {:>19.2f}% {:>19.2f}%".format("Accuracy", original_accuracy, pruned_accuracy))
    print("{:<25} {:>20,} {:>20,}".format("Parameters", original_params, pruned_params))
    print("{:<25} {:>19.2f}s {:>19.2f}s".format("Inference Time", original_time, pruned_time))
    
    accuracy_drop = original_accuracy - pruned_accuracy
    param_reduction = (1 - pruned_params/original_params) * 100
    speedup = original_time / pruned_time
    
    print("\n{:<25} {:>20}".format("Accuracy Drop", f"{accuracy_drop:.2f}%"))
    print("{:<25} {:>20}".format("Parameter Reduction", f"{param_reduction:.2f}%"))
    print("{:<25} {:>20}".format("Speedup", f"{speedup:.2f}x"))
    
    print("\n" + "=" * 70)
    
    # Model kaydet
    print("\nSaving pruned model...")
    torch.save(pruned_model.state_dict(), 'pruned_resnet18_cifar10.pth')
    print("✓ Model saved to 'pruned_resnet18_cifar10.pth'")
    
    print("\n" + "=" * 70)
    print("Experiment Complete!".center(70))
    print("=" * 70)


if __name__ == '__main__':
    main()
