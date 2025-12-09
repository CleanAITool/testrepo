"""
Neuron Coverage Pruning - Quick Start Example

Basit bir CNN modelinde Neuron Coverage bazlı pruning örneği.
MNIST dataset kullanarak hızlı test.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import sys
import os

# CleanAI-Tool import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pruner.structured_pruner import StructuredPruner
from importance.neuron_coverage import NeuronCoverageImportance


# Basit CNN modeli
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    print("="*70)
    print("NEURON COVERAGE PRUNING - QUICK START")
    print("="*70 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # 1. Dataset
    print("Loading MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 2. Model oluştur ve eğit
    print("Creating and training model...\n")
    model = SimpleCNN().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Hızlı eğitim (3 epoch)
    model.train()
    for epoch in range(3):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # 3. Original accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    acc_before = 100. * correct / total
    params_before = sum(p.numel() for p in model.parameters())
    
    print(f"\n✓ Original Model:")
    print(f"  Accuracy: {acc_before:.2f}%")
    print(f"  Parameters: {params_before:,}\n")
    
    # 4. Neuron Coverage Importance Scorer
    print("="*70)
    print("NEURON COVERAGE ANALYSIS")
    print("="*70 + "\n")
    
    importance = NeuronCoverageImportance(
        threshold=0.0,          # ReLU sonrası
        metric='coverage',      # Hibrit: frequency × strength
        percentile_threshold=20, # Alt %20'yi penalize et
        normalize=True
    )
    
    # Aktivasyonları topla
    importance.register_activation_hooks(model)
    importance.collect_activations(
        model=model,
        dataloader=test_loader,
        max_batches=20,  # İlk 20 batch
        device=device
    )
    
    # Coverage raporu
    importance.print_coverage_report()
    
    # 5. Pruning
    print("\n" + "="*70)
    print("PRUNING")
    print("="*70 + "\n")
    
    example_inputs = torch.randn(1, 1, 28, 28).to(device)
    
    # Output layer'ı koru
    ignored_layers = [model.classifier[3]]
    
    pruner = StructuredPruner(
        model=model,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=0.4,  # %40 pruning
        ignored_layers=ignored_layers,
        device=device
    )
    
    print("Applying pruning...")
    pruned_model = pruner.prune()
    
    importance.remove_hooks()
    
    params_after = sum(p.numel() for p in pruned_model.parameters())
    reduction = (1 - params_after / params_before) * 100
    
    print(f"\n✓ Pruning completed!")
    print(f"  Parameters: {params_before:,} → {params_after:,}")
    print(f"  Reduction: {reduction:.2f}%\n")
    
    # 6. Pruned accuracy (before fine-tuning)
    pruned_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = pruned_model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    acc_pruned = 100. * correct / total
    print(f"Pruned Model Accuracy (before FT): {acc_pruned:.2f}%")
    print(f"Accuracy drop: {acc_before - acc_pruned:.2f}%\n")
    
    # 7. Fine-tuning
    print("Fine-tuning (2 epochs)...")
    optimizer = optim.Adam(pruned_model.parameters(), lr=0.0005)
    
    pruned_model.train()
    for epoch in range(2):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = pruned_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'FT Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # 8. Final accuracy
    pruned_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = pruned_model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    acc_final = 100. * correct / total
    
    # 9. Sonuçlar
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Original Model:    Acc: {acc_before:.2f}%,  Params: {params_before:,}")
    print(f"Pruned Model:      Acc: {acc_final:.2f}%,   Params: {params_after:,}")
    print(f"Compression:       {reduction:.2f}% parameter reduction")
    print(f"Accuracy Change:   {acc_final - acc_before:+.2f}%")
    print("="*70)


if __name__ == '__main__':
    main()
