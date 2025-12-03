"""
CleanAI Structured Pruning Tool - Basit Örnek
============================================

Bu örnek, basit bir CNN modeli üzerinde structured pruning nasıl yapılacağını gösterir.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# CleanAI import - proje içinden
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pruner import StructuredPruner
from importance import WeightActivationImportance


# Basit bir CNN modeli
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 10)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def count_parameters(model):
    """Model parametrelerini say"""
    return sum(p.numel() for p in model.parameters())


def create_dummy_data(num_samples=100, img_size=32):
    """Test için dummy veri oluştur"""
    images = torch.randn(num_samples, 3, img_size, img_size)
    labels = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    return dataloader


def main():
    print("="*70)
    print(" " * 20 + "CleanAI Structured Pruning")
    print("="*70 + "\n")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Model oluştur
    print("Creating model...")
    model = SimpleCNN().to(device)
    initial_params = count_parameters(model)
    print(f"✓ Model created with {initial_params:,} parameters\n")
    
    # Dummy test verisi
    print("Creating dummy test data...")
    test_loader = create_dummy_data(num_samples=100, img_size=32)
    print(f"✓ Created dataloader with {len(test_loader)} batches\n")
    
    # Importance scorer oluştur
    print("Creating importance scorer...")
    importance = WeightActivationImportance(
        weight_ratio=0.5,       # %50 ağırlık
        activation_ratio=0.5,   # %50 aktivasyon
        p=2,                    # L2 norm
        normalize=True
    )
    print(f"✓ {importance}\n")
    
    # Pruner oluştur
    print("Creating structured pruner...")
    pruner = StructuredPruner(
        model=model,
        example_inputs=torch.randn(1, 3, 32, 32),
        importance=importance,
        pruning_ratio=0.3,  # %30 pruning
        device=device
    )
    print()
    
    # Aktivasyonları topla
    print("Collecting activations from test data...")
    pruner.collect_activations(test_loader, num_batches=10)
    print()
    
    # Pruning uygula
    print("Applying pruning...\n")
    pruned_model = pruner.prune()
    
    # Özet
    pruner.print_summary()
    
    # Final karşılaştırma
    final_params = count_parameters(pruned_model)
    reduction = (1 - final_params / initial_params) * 100
    
    print("\n" + "="*70)
    print(" " * 25 + "Final Results")
    print("="*70)
    print(f"Original parameters:  {initial_params:,}")
    print(f"Pruned parameters:    {final_params:,}")
    print(f"Reduction:            {initial_params - final_params:,} ({reduction:.2f}%)")
    print("="*70 + "\n")
    
    # Test forward pass
    print("Testing forward pass on pruned model...")
    dummy_input = torch.randn(2, 3, 32, 32).to(device)
    with torch.no_grad():
        output = pruned_model(dummy_input)
    print(f"✓ Forward pass successful! Output shape: {output.shape}\n")
    
    print("="*70)
    print(" " * 25 + "Pruning Complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
