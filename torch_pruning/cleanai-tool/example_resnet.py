"""
CleanAI Structured Pruning - ResNet Örneği
==========================================

Bu örnek, ResNet18 modeli üzerinde structured pruning gösterir.
Skip connections ve residual block'ları doğru şekilde handle eder.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pruner import StructuredPruner
from importance import WeightActivationImportance


def count_parameters(model):
    """Model parametrelerini say"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def create_imagenet_dummy_data(num_samples=200, img_size=224):
    """ImageNet boyutunda dummy veri"""
    images = torch.randn(num_samples, 3, img_size, img_size)
    labels = torch.randint(0, 1000, (num_samples,))
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    return dataloader


def main():
    print("\n" + "="*80)
    print(" " * 25 + "CleanAI ResNet18 Pruning Example")
    print("="*80 + "\n")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # ResNet18 modeli yükle
    print("Loading ResNet18 (pretrained=False)...")
    model = models.resnet18(pretrained=False)
    model = model.to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"✓ ResNet18 loaded")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}\n")
    
    # Test verisi oluştur
    print("Creating dummy ImageNet data...")
    test_loader = create_imagenet_dummy_data(num_samples=200, img_size=224)
    print(f"✓ Created dataloader with {len(test_loader)} batches\n")
    
    # Importance scorer
    print("Configuring importance scorer...")
    importance = WeightActivationImportance(
        weight_ratio=0.6,       # Ağırlığa daha fazla önem
        activation_ratio=0.4,   # Aktivasyona daha az önem
        p=2,
        normalize=True
    )
    print(f"✓ {importance}\n")
    
    # Belirli layer'ları ignore et
    # İlk conv ve son fc layer'ı koruyalım
    ignored_layers = [
        model.conv1,      # İlk conv layer
        model.fc,         # Son fully connected layer
    ]
    
    print("Setting up pruner...")
    print(f"Ignored layers: {len(ignored_layers)}")
    print(f"  - {model.conv1}")
    print(f"  - {model.fc}")
    print()
    
    # Layer-specific pruning ratios (opsiyonel)
    # ResNet'in katmanlarına göre farklı oranlar verebiliriz
    layer_pruning_ratios = {
        # İlk katmanlara daha az, son katmanlara daha fazla pruning
        model.layer1: 0.2,   # %20 pruning
        model.layer2: 0.3,   # %30 pruning
        model.layer3: 0.4,   # %40 pruning
        model.layer4: 0.35,  # %35 pruning
    }
    
    # Pruner oluştur
    pruner = StructuredPruner(
        model=model,
        example_inputs=torch.randn(1, 3, 224, 224),
        importance=importance,
        pruning_ratio=0.3,  # Default %30
        layer_pruning_ratios=layer_pruning_ratios,
        ignored_layers=ignored_layers,
        device=device
    )
    print()
    
    # Aktivasyonları topla
    print("Collecting activations...")
    pruner.collect_activations(test_loader, num_batches=20)
    print()
    
    # Pruning uygula
    print("Starting structured pruning...\n")
    pruned_model = pruner.prune()
    
    # Model özeti
    pruner.print_summary()
    
    # Final sonuçlar
    pruned_total, pruned_trainable = count_parameters(pruned_model)
    
    print("\n" + "="*80)
    print(" " * 30 + "Pruning Results")
    print("="*80)
    print(f"{'Metric':<30} {'Before':<20} {'After':<20} {'Reduction':<20}")
    print("-"*80)
    print(f"{'Total Parameters':<30} {total_params:<20,} {pruned_total:<20,} "
          f"{total_params - pruned_total:,} ({(1 - pruned_total/total_params)*100:.2f}%)")
    print(f"{'Trainable Parameters':<30} {trainable_params:<20,} {pruned_trainable:<20,} "
          f"{trainable_params - pruned_trainable:,} ({(1 - pruned_trainable/trainable_params)*100:.2f}%)")
    print("="*80 + "\n")
    
    # Forward pass testi
    print("Testing forward pass...")
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    model.eval()
    
    with torch.no_grad():
        output_original = model(dummy_input)
        output_pruned = pruned_model(dummy_input)
    
    print(f"✓ Original output shape: {output_original.shape}")
    print(f"✓ Pruned output shape:   {output_pruned.shape}")
    print(f"✓ Forward pass successful!\n")
    
    # Opsiyonel: Model'i kaydet
    save_path = "resnet18_pruned.pth"
    print(f"Saving pruned model to {save_path}...")
    torch.save(pruned_model.state_dict(), save_path)
    print(f"✓ Model saved!\n")
    
    print("="*80)
    print(" " * 30 + "Pruning Complete!")
    print("="*80 + "\n")
    
    # Ekstra bilgiler
    print("Notes:")
    print("  - Pruned model'i fine-tuning yaparak accuracy'yi geri kazanabilirsiniz")
    print("  - Farklı pruning ratios ile deneme yaparak optimal noktayı bulabilirsiniz")
    print("  - İlk ve son layer'lar ignore edildi (accuracy için önemli)")
    print()


if __name__ == '__main__':
    main()
