"""
CleanAI Structured Pruning Tool - Kullanım Kılavuzu
==================================================

## İçindekiler

1. Kurulum
2. Temel Kullanım
3. İleri Seviye Kullanım
4. API Referansı
5. Örnekler
6. Sorun Giderme

---

## 1. Kurulum

CleanAI Tool bağımsız bir projedir, sadece PyTorch gerektirir:

```bash
# Gereksinimler
pip install torch torchvision
```

Proje içinden import:

```python
from cleanai_tool import StructuredPruner, WeightActivationImportance
```

---

## 2. Temel Kullanım

### Adım 1: Model ve Veri Hazırlama

```python
import torch
import torch.nn as nn

# Modelinizi oluşturun
model = YourModel()

# Test/validation dataloader
test_loader = torch.utils.data.DataLoader(...)
```

### Adım 2: Importance Scorer Oluşturma

```python
from cleanai_tool import WeightActivationImportance

importance = WeightActivationImportance(
    weight_ratio=0.5,       # Ağırlık etkisi: %50
    activation_ratio=0.5,   # Aktivasyon etkisi: %50
    p=2,                    # L2 norm
    normalize=True          # Normalize et
)
```

### Adım 3: Pruner Oluşturma

```python
from cleanai_tool import StructuredPruner

pruner = StructuredPruner(
    model=model,
    example_inputs=torch.randn(1, 3, 224, 224),  # Örnek input
    importance=importance,
    pruning_ratio=0.3  # %30 pruning
)
```

### Adım 4: Aktivasyonları Toplama

```python
# Test verisi üzerinden aktivasyonları topla
pruner.collect_activations(test_loader, num_batches=10)
```

### Adım 5: Pruning Uygulama

```python
# Pruning'i uygula
pruned_model = pruner.prune()

# Özet göster
pruner.print_summary()
```

---

## 3. İleri Seviye Kullanım

### Layer-Specific Pruning Ratios

Farklı katmanlara farklı pruning oranları uygulama:

```python
layer_pruning_ratios = {
    model.layer1: 0.2,   # %20 pruning
    model.layer2: 0.3,   # %30 pruning
    model.layer3: 0.4,   # %40 pruning
}

pruner = StructuredPruner(
    model=model,
    example_inputs=example_input,
    importance=importance,
    pruning_ratio=0.3,  # Default
    layer_pruning_ratios=layer_pruning_ratios
)
```

### Belirli Layer'ları İgnore Etme

```python
ignored_layers = [
    model.conv1,    # İlk conv layer
    model.fc,       # Son fully connected
]

pruner = StructuredPruner(
    model=model,
    example_inputs=example_input,
    importance=importance,
    pruning_ratio=0.3,
    ignored_layers=ignored_layers
)
```

### Özelleştirilmiş Importance Scorer

Sadece ağırlık veya sadece aktivasyon kullanma:

```python
# Sadece Weight-based
weight_only = WeightActivationImportance(
    weight_ratio=1.0,
    activation_ratio=0.0
)

# Sadece Activation-based
activation_only = WeightActivationImportance(
    weight_ratio=0.0,
    activation_ratio=1.0
)

# Ağırlığa daha fazla önem
weight_heavy = WeightActivationImportance(
    weight_ratio=0.7,
    activation_ratio=0.3
)
```

---

## 4. API Referansı

### StructuredPruner

```python
StructuredPruner(
    model: nn.Module,                    # Pruning yapılacak model
    example_inputs: torch.Tensor,        # Graph tracing için örnek input
    importance: WeightActivationImportance,  # Importance scorer
    pruning_ratio: float = 0.3,          # Global pruning oranı
    layer_pruning_ratios: Dict = None,   # Layer-spesifik oranlar
    ignored_layers: List = None,         # İhmal edilecek layer'lar
    device: str = 'cuda'                 # Cihaz
)
```

**Metodlar:**

- `collect_activations(dataloader, num_batches)`: Aktivasyonları topla
- `prune()`: Pruning'i uygula, pruned model'i döndür
- `print_summary()`: Model özetini yazdır
- `get_pruned_model_summary()`: Özet dictionary döndür

### WeightActivationImportance

```python
WeightActivationImportance(
    weight_ratio: float = 0.5,      # Ağırlık oranı
    activation_ratio: float = 0.5,  # Aktivasyon oranı
    p: int = 2,                     # Norm derecesi
    normalize: bool = True          # Normalize et
)
```

**Metodlar:**

- `collect_activations(model, dataloader, num_batches, device)`: Aktivasyonları topla
- `register_activation_hooks(model)`: Hook'ları kaydet
- `remove_activation_hooks()`: Hook'ları kaldır
- `clear_activation_cache()`: Cache'i temizle

---

## 5. Örnekler

### Örnek 1: Basit CNN

```python
from cleanai_tool import StructuredPruner, WeightActivationImportance

# Model
model = SimpleCNN()

# Pruner
importance = WeightActivationImportance()
pruner = StructuredPruner(
    model=model,
    example_inputs=torch.randn(1, 3, 32, 32),
    importance=importance,
    pruning_ratio=0.3
)

# Aktivasyonları topla
pruner.collect_activations(test_loader, num_batches=10)

# Prune
pruned_model = pruner.prune()
```

### Örnek 2: ResNet18

```python
import torchvision.models as models

# ResNet18
model = models.resnet18(pretrained=True)

# İlk ve son layer'ları koru
ignored = [model.conv1, model.fc]

# Pruner
pruner = StructuredPruner(
    model=model,
    example_inputs=torch.randn(1, 3, 224, 224),
    importance=WeightActivationImportance(weight_ratio=0.6, activation_ratio=0.4),
    pruning_ratio=0.35,
    ignored_layers=ignored
)

# Prune
pruner.collect_activations(val_loader, num_batches=20)
pruned_model = pruner.prune()

# Kaydet
torch.save(pruned_model.state_dict(), 'resnet18_pruned.pth')
```

---

## 6. Sorun Giderme

### Sorun: "Module is not in dependency graph"

**Çözüm:** Model'in forward pass'ı sırasında bu module kullanılmıyor olabilir.
ignored_layers'a ekleyin veya model mimarisini kontrol edin.

### Sorun: "Could not compute importance"

**Çözüm:** Bu layer için weight/activation bulunamadı. `collect_activations()`
çağrıldığından emin olun.

### Sorun: Memory Overflow

**Çözüm:**

- `num_batches` parametresini azaltın
- Daha küçük batch size kullanın
- Aktivasyon cache'i temizleyin: `importance.clear_activation_cache()`

### Sorun: Forward pass hatası (pruning sonrası)

**Çözüm:**

- Skip connection'lar veya concat operasyonları sorun çıkarabilir
- İlgili layer'ları ignored_layers'a ekleyin
- Model mimarisini kontrol edin

---

## Desteklenen Layer'lar

✓ Conv2d
✓ ConvTranspose2d
✓ Linear
✓ BatchNorm2d / BatchNorm1d
✓ LayerNorm
✓ GroupNorm
✓ Depthwise Convolutions

## Limitasyonlar

- Multi-head Attention tam desteklenmez
- Recursive modeller (RNN/LSTM) sınırlı destek
- Dynamic control flow desteklenmez

---

## İletişim & Katkı

Sorularınız veya katkılarınız için:

- Issues: GitHub Issues
- Pull Requests: Hoş geldiniz!

---

© 2024 CleanAI Team
"""
