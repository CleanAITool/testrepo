"""
Neuron Coverage bazlı pruning örneği

Bu örnek, Neuron Coverage metriğini kullanarak CIFAR-10 üzerinde 
VGG11 modelini prune eder.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg11
import sys
import os

# CleanAI-Tool import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.graph import DependencyGraph
from pruner.structured_pruner import StructuredPruner
from importance.neuron_coverage import NeuronCoverageImportance


def prepare_cifar10(batch_size=128):
    """CIFAR-10 dataset hazırla"""
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
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return trainloader, testloader


def train_model(model, trainloader, criterion, optimizer, epochs=5, device='cuda'):
    """Model eğit"""
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(trainloader)}], '
                      f'Loss: {running_loss/100:.3f}, Acc: {100.*correct/total:.2f}%')
                running_loss = 0.0
        
        print(f'Epoch {epoch+1} completed. Train Accuracy: {100.*correct/total:.2f}%')


def evaluate_accuracy(model, testloader, device='cuda'):
    """Model accuracy hesapla"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def count_parameters(model):
    """Model parametre sayısını hesapla"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("="*70)
    print("NEURON COVERAGE PRUNING EXAMPLE - VGG11 on CIFAR-10")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # 1. Dataset hazırla
    print("Loading CIFAR-10 dataset...")
    trainloader, testloader = prepare_cifar10(batch_size=128)
    print("✓ Dataset loaded\n")
    
    # 2. Model oluştur ve eğit
    print("Creating VGG11 model...")
    model = vgg11(pretrained=False, num_classes=10)
    model = model.to(device)
    
    params_before = count_parameters(model)
    print(f"✓ Model created. Parameters: {params_before:,}\n")
    
    # Initial training
    print("Training original model (10 epochs)...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    train_model(model, trainloader, criterion, optimizer, epochs=10, device=device)
    
    # Evaluate before pruning
    print("\nEvaluating original model...")
    acc_before = evaluate_accuracy(model, testloader, device)
    print(f"✓ Original Model Accuracy: {acc_before:.2f}%\n")
    
    # 3. Neuron Coverage Importance Scorer oluştur
    print("="*70)
    print("NEURON COVERAGE ANALYSIS")
    print("="*70)
    
    # Farklı metric'leri test et
    metrics = ['frequency', 'strength', 'coverage']
    
    for metric_type in metrics:
        print(f"\n--- Testing metric: {metric_type} ---\n")
        
        # Importance scorer oluştur
        importance = NeuronCoverageImportance(
            threshold=0.0,           # ReLU için 0.0
            metric=metric_type,      # 'frequency', 'strength', veya 'coverage'
            percentile_threshold=25, # Alt %25'lik dilimi penalize et
            normalize=True
        )
        
        # Hook'ları kaydet
        importance.register_activation_hooks(model)
        
        # Aktivasyonları topla
        print(f"Collecting activations with {metric_type} metric...")
        importance.collect_activations(
            model=model,
            dataloader=testloader,
            max_batches=50,  # Memory için sınırla
            device=device
        )
        
        # Coverage raporunu yazdır
        importance.print_coverage_report()
        
        # Hook'ları temizle
        importance.remove_hooks()
    
    # 4. En iyi metrik ile pruning yap (coverage)
    print("\n" + "="*70)
    print("PRUNING WITH COVERAGE METRIC")
    print("="*70 + "\n")
    
    # Yeni importance scorer oluştur
    importance = NeuronCoverageImportance(
        threshold=0.0,
        metric='coverage',  # Hibrit metrik
        percentile_threshold=25,
        normalize=True
    )
    
    # Hook'ları kaydet
    importance.register_activation_hooks(model)
    
    # Aktivasyonları topla
    print("Collecting activations for pruning...")
    importance.collect_activations(
        model=model,
        dataloader=testloader,
        max_batches=50,
        device=device
    )
    
    # Coverage istatistikleri
    stats_before = importance.get_coverage_statistics()
    print(f"\nCoverage before pruning: {stats_before['coverage_ratio']*100:.2f}%")
    
    # 5. Pruner oluştur
    example_inputs = torch.randn(1, 3, 32, 32).to(device)
    
    # Son classifier katmanını koru
    ignored_layers = [model.classifier[6]]  # Son Linear layer
    
    pruner = StructuredPruner(
        model=model,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=0.3,  # %30 pruning
        ignored_layers=ignored_layers,
        device=device
    )
    
    print("\n" + "="*70)
    print("PRUNING EXECUTION")
    print("="*70)
    
    # 6. Prune
    print("\nApplying structured pruning...")
    pruned_model = pruner.prune()
    
    # Hook'ları temizle
    importance.remove_hooks()
    
    params_after = count_parameters(pruned_model)
    reduction = (1 - params_after / params_before) * 100
    
    print(f"\n✓ Pruning completed!")
    print(f"Parameters: {params_before:,} → {params_after:,}")
    print(f"Reduction: {reduction:.2f}%\n")
    
    # 7. Pruning sonrası coverage
    print("Analyzing coverage after pruning...")
    
    # Yeni importance scorer oluştur (pruned model için)
    importance_after = NeuronCoverageImportance(
        threshold=0.0,
        metric='coverage',
        normalize=True
    )
    
    importance_after.register_activation_hooks(pruned_model)
    importance_after.collect_activations(
        model=pruned_model,
        dataloader=testloader,
        max_batches=50,
        device=device
    )
    
    stats_after = importance_after.get_coverage_statistics()
    print(f"Coverage after pruning: {stats_after['coverage_ratio']*100:.2f}%")
    
    importance_after.remove_hooks()
    
    # 8. Pruned model accuracy (fine-tuning öncesi)
    print("\nEvaluating pruned model (before fine-tuning)...")
    acc_pruned = evaluate_accuracy(pruned_model, testloader, device)
    print(f"Pruned Model Accuracy: {acc_pruned:.2f}%")
    print(f"Accuracy drop: {acc_before - acc_pruned:.2f}%\n")
    
    # 9. Fine-tuning
    print("="*70)
    print("FINE-TUNING")
    print("="*70 + "\n")
    
    print("Fine-tuning pruned model (5 epochs)...")
    optimizer = optim.SGD(pruned_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    train_model(pruned_model, trainloader, criterion, optimizer, epochs=5, device=device)
    
    # 10. Final accuracy
    print("\nEvaluating fine-tuned model...")
    acc_final = evaluate_accuracy(pruned_model, testloader, device)
    
    # 11. Sonuçlar
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Original Model:")
    print(f"  - Parameters:  {params_before:,}")
    print(f"  - Accuracy:    {acc_before:.2f}%")
    print(f"  - Coverage:    {stats_before['coverage_ratio']*100:.2f}%")
    print(f"\nPruned Model (after fine-tuning):")
    print(f"  - Parameters:  {params_after:,}")
    print(f"  - Accuracy:    {acc_final:.2f}%")
    print(f"  - Coverage:    {stats_after['coverage_ratio']*100:.2f}%")
    print(f"\nCompression:")
    print(f"  - Parameter Reduction: {reduction:.2f}%")
    print(f"  - Accuracy Change:     {acc_final - acc_before:+.2f}%")
    print(f"  - Coverage Change:     {(stats_after['coverage_ratio'] - stats_before['coverage_ratio'])*100:+.2f}%")
    print("="*70)


if __name__ == '__main__':
    main()
