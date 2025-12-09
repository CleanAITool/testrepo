"""
Neuron Coverage vs Weight-Activation Comparison

Bu script iki yöntemin nasıl farklı channel'ları seçtiğini gösterir.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pruner.structured_pruner import StructuredPruner
from importance.weight_activation import WeightActivationImportance
from importance.neuron_coverage import NeuronCoverageImportance


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


def train_quick(model, train_loader, device, epochs=2):
    """Hızlı eğitim"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 50:  # Sadece 50 batch
                break
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} completed")


def get_importance_scores_and_selected_channels(model, test_loader, device, importance_type='weight_activation'):
    """
    Belirtilen importance yöntemi ile her layer için önem skorlarını ve
    seçilecek channel'ları hesapla
    """
    model_copy = type(model)()  # Yeni model instance
    model_copy.load_state_dict(model.state_dict())
    model_copy = model_copy.to(device)
    
    example_inputs = torch.randn(1, 1, 28, 28).to(device)
    
    # Importance scorer oluştur
    if importance_type == 'weight_activation':
        importance = WeightActivationImportance(
            weight_ratio=0.5,
            activation_ratio=0.5
        )
        importance.register_activation_hooks(model_copy)
        importance.collect_activations(
            model=model_copy,
            dataloader=test_loader,
            max_batches=20,
            device=device
        )
    else:  # neuron_coverage
        importance = NeuronCoverageImportance(
            threshold=0.0,
            metric='coverage',
            percentile_threshold=25,
            normalize=True
        )
        importance.register_activation_hooks(model_copy)
        importance.collect_activations(
            model=model_copy,
            dataloader=test_loader,
            max_batches=20,
            device=device
        )
    
    # Pruner oluştur
    ignored_layers = [model_copy.fc2]
    
    pruner = StructuredPruner(
        model=model_copy,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=0.3,
        ignored_layers=ignored_layers,
        device=device
    )
    
    # Her group için importance skorlarını ve prune edilecek channel'ları kaydet
    all_groups = pruner.DG.get_all_prunable_groups()
    
    layer_info = {}
    
    for group in all_groups:
        root_module = group[0].dep.target.module
        
        # Module ismini bul
        module_name = None
        for name, module in model_copy.named_modules():
            if module is root_module:
                module_name = name
                break
        
        if module_name is None or module_name == '':
            continue
        
        # Importance hesapla
        importance_scores = pruner._compute_importance_for_group(group)
        
        if importance_scores is None:
            continue
        
        # Convert to torch tensor if numpy array
        if isinstance(importance_scores, np.ndarray):
            importance_scores_tensor = torch.from_numpy(importance_scores).float()
        else:
            importance_scores_tensor = importance_scores
        
        num_channels = len(importance_scores)
        pruning_ratio = pruner._get_pruning_ratio_for_layer(root_module)
        
        # Prune edilecek channel'ları seç
        prune_idxs = pruner._select_channels_to_prune(
            importance_scores_tensor,
            num_channels,
            pruning_ratio
        )
        
        # Convert to numpy for storage and comparison
        if isinstance(importance_scores, torch.Tensor):
            importance_scores_np = importance_scores.cpu().numpy()
        else:
            importance_scores_np = importance_scores
        
        layer_info[module_name] = {
            'num_channels': num_channels,
            'importance_scores': importance_scores_np,
            'prune_idxs': prune_idxs,
            'pruning_ratio': pruning_ratio
        }
    
    importance.remove_hooks()
    
    return layer_info


def compare_methods(model, test_loader, device):
    """İki yöntemi karşılaştır"""
    print("\n" + "="*80)
    print("COMPUTING IMPORTANCE SCORES WITH WEIGHT-ACTIVATION METHOD")
    print("="*80)
    
    wa_info = get_importance_scores_and_selected_channels(
        model, test_loader, device, 'weight_activation'
    )
    
    print("\n" + "="*80)
    print("COMPUTING IMPORTANCE SCORES WITH NEURON COVERAGE METHOD")
    print("="*80)
    
    nc_info = get_importance_scores_and_selected_channels(
        model, test_loader, device, 'neuron_coverage'
    )
    
    # Karşılaştırma
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    for layer_name in wa_info.keys():
        if layer_name not in nc_info:
            continue
        
        wa_data = wa_info[layer_name]
        nc_data = nc_info[layer_name]
        
        print(f"\n{'='*80}")
        print(f"Layer: {layer_name}")
        print(f"{'='*80}")
        print(f"Total channels: {wa_data['num_channels']}")
        print(f"Pruning ratio: {wa_data['pruning_ratio']:.2%}")
        print(f"Channels to prune: {len(wa_data['prune_idxs'])}")
        
        # Importance skorlarının korelasyonu
        wa_scores = wa_data['importance_scores']
        nc_scores = nc_data['importance_scores']
        
        # Normalize et (karşılaştırma için)
        wa_norm = (wa_scores - wa_scores.min()) / (wa_scores.max() - wa_scores.min() + 1e-8)
        nc_norm = (nc_scores - nc_scores.min()) / (nc_scores.max() - nc_scores.min() + 1e-8)
        
        # Korelasyon hesapla
        correlation = np.corrcoef(wa_norm, nc_norm)[0, 1]
        
        print(f"\nImportance Score Correlation: {correlation:.4f}")
        
        # Seçilen channel'ların overlap'i
        wa_set = set(wa_data['prune_idxs'])
        nc_set = set(nc_data['prune_idxs'])
        
        overlap = wa_set.intersection(nc_set)
        overlap_ratio = len(overlap) / len(wa_set) if len(wa_set) > 0 else 0
        
        print(f"\nSelected Channels Overlap:")
        print(f"  Weight-Activation selected: {sorted(list(wa_set))}")
        print(f"  Neuron Coverage selected:   {sorted(list(nc_set))}")
        print(f"  Common channels: {sorted(list(overlap))}")
        print(f"  Overlap ratio: {overlap_ratio:.2%}")
        
        # Top 5 ve Bottom 5 skorları göster
        print(f"\nTop 5 Important Channels (KEEP):")
        wa_top5_idx = np.argsort(wa_scores)[-5:][::-1]
        nc_top5_idx = np.argsort(nc_scores)[-5:][::-1]
        
        print(f"  Weight-Activation: {wa_top5_idx.tolist()} (scores: {wa_scores[wa_top5_idx]})")
        print(f"  Neuron Coverage:   {nc_top5_idx.tolist()} (scores: {nc_scores[nc_top5_idx]})")
        
        print(f"\nBottom 5 Important Channels (PRUNE):")
        wa_bot5_idx = np.argsort(wa_scores)[:5]
        nc_bot5_idx = np.argsort(nc_scores)[:5]
        
        print(f"  Weight-Activation: {wa_bot5_idx.tolist()} (scores: {wa_scores[wa_bot5_idx]})")
        print(f"  Neuron Coverage:   {nc_bot5_idx.tolist()} (scores: {nc_scores[nc_bot5_idx]})")
        
        # Eğer overlap çok yüksekse uyarı
        if overlap_ratio > 0.8:
            print(f"\n⚠️  WARNING: Very high overlap ({overlap_ratio:.0%})!")
            print(f"    The two methods are selecting almost the same channels.")


def main():
    print("="*80)
    print("WEIGHT-ACTIVATION vs NEURON COVERAGE COMPARISON")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}\n")
    
    # Dataset
    print("Loading MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Model
    print("Creating and training model...")
    model = SimpleCNN().to(device)
    train_quick(model, train_loader, device, epochs=2)
    
    # Karşılaştırma
    compare_methods(model, test_loader, device)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Insights:")
    print("• High correlation → Methods agree on channel importance")
    print("• High overlap → Same channels selected for pruning")
    print("• Low overlap → Methods prioritize different features")
    print("="*80)


if __name__ == '__main__':
    main()
