"""
CleanAI Quick Start - 5 Dakikada Pruning
========================================

En hızlı şekilde pruning yapmak için bu dosyayı çalıştırın!
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# cleanai-tool import
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pruner import StructuredPruner
from importance import WeightActivationImportance


# 1. Basit bir model tanımla
class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 2. Model oluştur
print("="*50)
print("CleanAI Quick Start")
print("="*50)

model = TinyNet()
print(f"\n✓ Model created: {sum(p.numel() for p in model.parameters()):,} params")

# 3. Pruner oluştur (activation toplamadan sadece weight-based)
importance = WeightActivationImportance(
    weight_ratio=1.0,      # Sadece weight kullan
    activation_ratio=0.0,
    normalize=True
)

pruner = StructuredPruner(
    model=model,
    example_inputs=torch.randn(1, 3, 32, 32),
    importance=importance,
    pruning_ratio=0.4,  # %40 pruning
    device='cpu'
)

# 4. Prune (aktivasyon toplamadan)
print("\n" + "="*50)
print("Pruning without activation collection")
print("="*50 + "\n")

pruned_model = pruner.prune()

# 5. Sonuçlar
print(f"\n✓ Pruned model: {sum(p.numel() for p in pruned_model.parameters()):,} params")

# 6. Test forward
dummy = torch.randn(2, 3, 32, 32)
output = pruned_model(dummy)
print(f"✓ Output shape: {output.shape}")

print("\n" + "="*50)
print("Done! That was easy 🎉")
print("="*50 + "\n")

print("For more examples, check:")
print("  - example_basic.py    : CNN with activation collection")
print("  - example_resnet.py   : ResNet18 pruning")
print()
