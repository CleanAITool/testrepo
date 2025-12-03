# CleanAI Structured Pruning Tool

PyTorch modelleri üzerinde structured pruning (kanal/filtre/layer bazlı budama) yapan tamamen bağımsız bir araç.

## Özellikler

- ✅ **Sıfırdan Implementation**: Sadece PyTorch ve temel Python kütüphaneleri
- ✅ **Weight-Activation Hybrid Scoring**: Hem ağırlık hem aktivasyon bazlı önem hesaplama
- ✅ **Autograd-based Tracing**: Otomatik bağımlılık grafiği oluşturma
- ✅ **Structured Pruning**: Gerçek tensor slicing ile yapısal budama
- ✅ **Dependency Resolution**: Katmanlar arası bağımlılıkları otomatik çözme

## Desteklenen Katmanlar

- `nn.Conv2d`, `nn.ConvTranspose2d`
- `nn.Linear`
- `nn.BatchNorm2d`, `nn.BatchNorm1d`
- `nn.LayerNorm`
- `nn.GroupNorm`
- Depthwise Convolutions

## Kurulum

Bu proje bağımsızdır, herhangi bir dış pruning kütüphanesi gerektirmez.

```python
# Proje içinden import
from cleanai_tool import StructuredPruner, WeightActivationImportance
```

## Kullanım

### Temel Kullanım

```python
import torch
import torch.nn as nn
from cleanai_tool import StructuredPruner, WeightActivationImportance

# Model tanımlama
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(128, 10)
)

# Test verisi
test_loader = torch.utils.data.DataLoader(...)

# Pruner oluşturma
importance = WeightActivationImportance(
    weight_ratio=0.5,      # Ağırlık etkisi
    activation_ratio=0.5    # Aktivasyon etkisi
)

pruner = StructuredPruner(
    model=model,
    example_inputs=torch.randn(1, 3, 32, 32),
    importance=importance,
    pruning_ratio=0.3  # %30 kanal budama
)

# Aktivasyonları toplama (test veri seti üzerinde)
pruner.collect_activations(test_loader, num_batches=10)

# Pruning uygulama
pruned_model = pruner.prune()

print(f"Orijinal parametreler: {sum(p.numel() for p in model.parameters())}")
print(f"Budanmış parametreler: {sum(p.numel() for p in pruned_model.parameters())}")
```

### İleri Seviye Kullanım

```python
# Katman-spesifik pruning oranları
pruner = StructuredPruner(
    model=model,
    example_inputs=torch.randn(1, 3, 32, 32),
    importance=importance,
    pruning_ratio=0.3,
    layer_pruning_ratios={
        model[0]: 0.2,  # İlk conv katmanı daha az buda
        model[3]: 0.4   # İkinci conv katmanı daha fazla buda
    }
)

# Bazı katmanları yok say
pruner = StructuredPruner(
    model=model,
    example_inputs=torch.randn(1, 3, 32, 32),
    importance=importance,
    pruning_ratio=0.3,
    ignored_layers=[model[0], model[-1]]  # İlk ve son katman
)
```

## Pruning Algoritması

### 1. Graph Tracing (Autograd-based)

Model üzerinde forward pass yapılır ve `grad_fn` objeleri ile computational graph oluşturulur.

### 2. Dependency Resolution

Katmanlar arası bağımlılıklar tespit edilir:

- Conv → BN → ReLU
- Concat, Split gibi operasyonlar
- Skip connections

### 3. Importance Scoring

**Weight-Activation Hybrid Method:**

```
importance(channel_i) = α × ||W_i|| + β × ||A_i||
```

- `W_i`: i. kanalın ağırlık normu (L2)
- `A_i`: i. kanalın aktivasyon ortalaması
- `α, β`: ağırlık ve aktivasyon oranları (α + β = 1)

### 4. Channel Selection & Pruning

- En düşük öneme sahip kanallar seçilir
- Bağımlı katmanlar birlikte budanır
- Tensor slicing ile gerçek yapısal değişiklik yapılır

## Mimari

```
cleanai-tool/
├── core/
│   ├── graph.py         # Dependency graph
│   ├── node.py          # Graph nodes
│   ├── dependency.py    # Dependencies
│   └── group.py         # Pruning groups
├── importance/
│   └── weight_activation.py  # Hybrid importance scorer
├── pruner/
│   ├── functions.py     # Pruning functions (Conv, Linear, etc.)
│   └── structured_pruner.py  # Ana pruner sınıfı
└── utils/
    └── helpers.py       # Yardımcı fonksiyonlar
```

## Örnekler

`examples/` klasöründe detaylı örnekler bulabilirsiniz:

- ResNet pruning
- VGG pruning
- Custom model pruning

## Sınırlamalar

- Şu anda sadece single-layer RNN/LSTM desteği
- Multi-head attention tam olarak desteklenmemektedir
- Dynamic control flow içeren modeller desteklenmez

## Lisans

MIT License
