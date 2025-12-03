"""
CleanAI - Real-World Model: ResNet50 on ImageNet Subset
========================================================

Pretrained ResNet50 modeli ile gerçek dünya veri seti (ImageNet subset) üzerinde 
pruning öncesi/sonrası accuracy karşılaştırması.

Not: Full ImageNet çok büyük olduğu için ImageNette (ImageNet'in 10 sınıflı subset'i) kullanılıyor.
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import time
import os

# cleanai-tool import
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pruner import StructuredPruner
from importance import WeightActivationImportance
import glob


def load_latest_checkpoint(checkpoint_dir, pattern):
    """Load the latest checkpoint matching pattern"""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, pattern))
    if not checkpoints:
        return None
    latest = max(checkpoints, key=os.path.getctime)
    return latest


def load_imagenet_subset(batch_size=16, num_train_samples=25000, num_val_samples=5000):
    """
    ImageNet benzeri dataset yükle (Tiny ImageNet veya ImageNet subset)
    Eğer yoksa CIFAR-100'ü ImageNet gibi kullan (daha gerçekçi, 100 sınıf)
    """
    # ImageNet preprocessing - smaller image size for memory efficiency
    transform_train = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    print("Using CIFAR-100 as ImageNet substitute (100 classes, similar complexity)")
    
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    # Subset oluştur (daha hızlı test için)
    if num_train_samples < len(trainset):
        indices = torch.randperm(len(trainset))[:num_train_samples].tolist()
        trainset = Subset(trainset, indices)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    valset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_val
    )
    
    if num_val_samples < len(valset):
        indices = torch.randperm(len(valset))[:num_val_samples].tolist()
        valset = Subset(valset, indices)
    
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return trainloader, valloader


def train_model(model, trainloader, device, epochs=10, lr=0.001, save_path=None):
    """Model eğit - with gradient accumulation and memory optimization"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_loss = float('inf')
    accumulation_steps = 2  # Simulate batch_size=32 with 16x2
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_loss_sum = 0.0
        
        optimizer.zero_grad()
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps  # Normalize loss
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()  # Clear cache periodically
            
            running_loss += loss.item() * accumulation_steps
            epoch_loss_sum += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 50 == 49:
                acc = 100.0 * correct / total
                avg_loss = running_loss / 50
                print(f'  Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(trainloader)}], '
                      f'Loss: {avg_loss:.4f}, Acc: {acc:.2f}%')
                running_loss = 0.0
        
        # Epoch sonu özeti
        epoch_acc = 100.0 * correct / total
        epoch_loss = epoch_loss_sum / len(trainloader)
        print(f'  ✓ Epoch [{epoch+1}/{epochs}] completed - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
        
        scheduler.step()
        
        # En iyi model checkpoint
        if save_path and epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint_path = save_path.replace('.pth', f'_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'accuracy': epoch_acc,
            }, checkpoint_path)
            print(f'  ✓ Checkpoint saved: {checkpoint_path}')


def evaluate_accuracy(model, dataloader, device, max_batches=None):
    """Model accuracy'sini hesapla"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Periodic memory cleanup
            if i % 20 == 0:
                torch.cuda.empty_cache()
    
    return 100.0 * correct / total


def count_parameters(model):
    """Model parametrelerini say"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    print("=" * 80)
    print("CleanAI - Real-World Model: ResNet50 on ImageNet-like Dataset".center(80))
    print("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    # Dataset yükle
    print("\nLoading dataset...")
    trainloader, valloader = load_imagenet_subset(
        batch_size=16,  # Reduced for memory efficiency
        num_train_samples=25000,  # 25K training samples (half of CIFAR-100)
        num_val_samples=5000      # 5K validation samples
    )
    print(f"✓ Train samples: {len(trainloader.dataset)}")
    print(f"✓ Validation samples: {len(valloader.dataset)}")
    
    # Pretrained ResNet50 yükle
    print("\nLoading pretrained ResNet50...")
    model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
    
    # CIFAR-100 için (100 classes) son katmanı değiştir
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 100)
    model = model.to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"✓ Model loaded with {total_params:,} parameters ({trainable_params:,} trainable)")
    
    # Checkpoint dizini oluştur
    os.makedirs('./checkpoints', exist_ok=True)
    
    # Check if training checkpoint exists
    latest_checkpoint = load_latest_checkpoint('./checkpoints', 'resnet50_original_epoch*.pth')
    
    if latest_checkpoint:
        print("\n" + "=" * 80)
        print("Loading Pre-trained Checkpoint".center(80))
        print("=" * 80)
        print(f"\n✓ Found checkpoint: {os.path.basename(latest_checkpoint)}")
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch_trained = checkpoint.get('epoch', 0) + 1
        print(f"✓ Loaded model trained for {epoch_trained} epochs")
        print(f"✓ Last training accuracy: {checkpoint.get('accuracy', 0):.2f}%")
        print(f"✓ Last training loss: {checkpoint.get('loss', 0):.4f}")
    else:
        # Fine-tune on CIFAR-100
        print("\n" + "=" * 80)
        print("Fine-tuning ResNet50 on CIFAR-100 (Transfer Learning)".center(80))
        print("=" * 80)
        print("\nFine-tuning for 7 epochs...")
        train_model(model, trainloader, device, epochs=7, lr=0.001, 
                    save_path='./checkpoints/resnet50_original.pth')
    
    # Original model kaydet
    print("\nSaving original fine-tuned model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': 100,
        'architecture': 'resnet50'
    }, './checkpoints/resnet50_finetuned_original.pth')
    print("✓ Original model saved to './checkpoints/resnet50_finetuned_original.pth'")
    
    # Original model accuracy
    print("\n" + "=" * 80)
    print("Evaluating Original Model".center(80))
    print("=" * 80)
    
    start_time = time.time()
    original_accuracy = evaluate_accuracy(model, valloader, device)
    original_time = time.time() - start_time
    
    print(f"\n✓ Original Model Accuracy: {original_accuracy:.2f}%")
    print(f"✓ Inference time: {original_time:.2f}s")
    print(f"✓ Parameters: {total_params:,}")
    
    # Pruning uygula
    print("\n" + "=" * 80)
    print("Applying Structured Pruning".center(80))
    print("=" * 80)
    
    importance = WeightActivationImportance(
        weight_ratio=0.5,
        activation_ratio=0.5,
        p=2,
        normalize=True
    )
    
    # Output layer'ı ignore et
    example_inputs = torch.randn(1, 3, 112, 112).to(device)
    pruner = StructuredPruner(
        model=model,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=0.05,  # %5 pruning (more conservative for better recovery)
        device=device,
        ignored_layers=[model.fc]
    )
    
    # Activation topla
    print("\nCollecting activations from training data...")
    pruner.collect_activations(trainloader, max_batches=100)
    print("✓ Activations collected")
    
    # Pruning yap
    print("\nApplying pruning...")
    pruned_model = pruner.prune()
    pruned_params, pruned_trainable = count_parameters(pruned_model)
    
    print(f"\n✓ Pruning completed")
    print(f"✓ Parameters: {total_params:,} -> {pruned_params:,}")
    print(f"✓ Reduction: {total_params - pruned_params:,} ({(1 - pruned_params/total_params)*100:.2f}%)")
    
    # Pruned model kaydet (before fine-tuning)
    print("\nSaving pruned model (before fine-tuning)...")
    torch.save({
        'model_state_dict': pruned_model.state_dict(),
        'num_classes': 100,
        'architecture': 'resnet50_pruned',
        'pruning_ratio': 0.3,
        'original_params': total_params,
        'pruned_params': pruned_params
    }, './checkpoints/resnet50_pruned_before_ft.pth')
    print("✓ Pruned model saved to './checkpoints/resnet50_pruned_before_ft.pth'")
    
    # Pruned model accuracy (before fine-tuning)
    print("\n" + "=" * 80)
    print("Evaluating Pruned Model (Before Fine-tuning)".center(80))
    print("=" * 80)
    
    pruned_accuracy_before = evaluate_accuracy(pruned_model, valloader, device)
    print(f"\n✓ Pruned Model Accuracy (before FT): {pruned_accuracy_before:.2f}%")
    print(f"✓ Accuracy drop: {original_accuracy - pruned_accuracy_before:.2f}%")
    
    # Fine-tune pruned model
    print("\n" + "=" * 80)
    print("Fine-tuning Pruned Model".center(80))
    print("=" * 80)
    
    print("\nFine-tuning pruned model for 5 epochs...")
    train_model(pruned_model, trainloader, device, epochs=5, lr=0.001,
                save_path='./checkpoints/resnet50_pruned_finetuned.pth')
    
    # Final pruned model kaydet
    print("\nSaving final pruned model (after fine-tuning)...")
    torch.save({
        'model_state_dict': pruned_model.state_dict(),
        'num_classes': 100,
        'architecture': 'resnet50_pruned',
        'pruning_ratio': 0.3,
        'original_params': total_params,
        'pruned_params': pruned_params,
        'original_accuracy': original_accuracy,
        'pruned_accuracy_before_ft': pruned_accuracy_before,
    }, './checkpoints/resnet50_pruned_final.pth')
    print("✓ Final pruned model saved to './checkpoints/resnet50_pruned_final.pth'")
    
    # Pruned model accuracy (after fine-tuning)
    print("\n" + "=" * 80)
    print("Evaluating Pruned Model (After Fine-tuning)".center(80))
    print("=" * 80)
    
    start_time = time.time()
    pruned_accuracy = evaluate_accuracy(pruned_model, valloader, device)
    pruned_time = time.time() - start_time
    
    print(f"\n✓ Pruned Model Accuracy (after FT): {pruned_accuracy:.2f}%")
    print(f"✓ Inference time: {pruned_time:.2f}s")
    print(f"✓ Parameters: {pruned_params:,}")
    
    # Final karşılaştırma
    print("\n" + "=" * 80)
    print("Final Comparison".center(80))
    print("=" * 80)
    
    print("\n{:<35} {:>18} {:>18} {:>18}".format(
        "Metric", "Original", "Pruned (BF)", "Pruned (AF)"))
    print("-" * 80)
    print("{:<35} {:>17.2f}% {:>17.2f}% {:>17.2f}%".format(
        "Accuracy", original_accuracy, pruned_accuracy_before, pruned_accuracy))
    print("{:<35} {:>18,} {:>18,} {:>18,}".format(
        "Parameters", total_params, pruned_params, pruned_params))
    print("{:<35} {:>17.2f}s {:>18} {:>17.2f}s".format(
        "Inference Time", original_time, "-", pruned_time))
    
    accuracy_drop = original_accuracy - pruned_accuracy
    param_reduction = (1 - pruned_params/total_params) * 100
    speedup = original_time / pruned_time
    
    print("\n{:<35} {:>18}".format("Final Accuracy Drop", f"{accuracy_drop:.2f}%"))
    print("{:<35} {:>18}".format("Parameter Reduction", f"{param_reduction:.2f}%"))
    print("{:<35} {:>18}".format("Speedup", f"{speedup:.2f}x"))
    print("{:<35} {:>18}".format("Accuracy Recovery", 
                                 f"{pruned_accuracy - pruned_accuracy_before:.2f}%"))
    
    print("\n" + "=" * 80)
    print("Experiment Complete!".center(80))
    print("=" * 80)
    
    # Özet
    if accuracy_drop < 1.0 and param_reduction > 20:
        print("\n✓ Excellent result! High compression with minimal accuracy loss.")
    elif accuracy_drop < 3.0 and param_reduction > 20:
        print("\n✓ Great result! Good compression with acceptable accuracy loss.")
    elif accuracy_drop < 5.0 and param_reduction > 25:
        print("\n✓ Good result! Decent compression with reasonable accuracy trade-off.")
    else:
        print(f"\n✓ Completed. Accuracy drop: {accuracy_drop:.2f}%, Compression: {param_reduction:.2f}%")
    
    # Saved files özeti
    print("\n" + "=" * 80)
    print("Saved Checkpoints".center(80))
    print("=" * 80)
    print("\n1. ./checkpoints/resnet50_finetuned_original.pth - Original fine-tuned model")
    print("2. ./checkpoints/resnet50_pruned_before_ft.pth - Pruned model (before fine-tuning)")
    print("3. ./checkpoints/resnet50_pruned_final.pth - Final pruned model (after fine-tuning)")
    print("\n✓ All checkpoints saved successfully!")


if __name__ == '__main__':
    main()
