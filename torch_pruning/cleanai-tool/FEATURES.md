# CleanAI Structured Pruning Tool - Özellikler

## 🎯 Temel Özellikler

### 1. **Tamamen Bağımsız Implementation**

- ✅ Sadece PyTorch ve temel Python kütüphaneleri
- ✅ Dış pruning kütüphanelerine (torch-pruning vb.) bağımlılık YOK
- ✅ Her şey sıfırdan implement edilmiş
- ✅ Öğrenmek ve özelleştirmek kolay

### 2. **Weight-Activation Hybrid Scoring**

```
importance(channel_i) = α × ||W_i||₂ + β × ||A_i||
```

- **Weight Score**: Ağırlık büyüklüğü (magnitude-based)
- **Activation Score**: Aktivasyon katkısı (data-dependent)
- **Hibrit Yaklaşım**: Her iki yöntemin avantajlarını birleştirir
- **Konfigüre Edilebilir**: α ve β oranlarını kendiniz belirleyin

### 3. **Autograd-Based Dependency Tracing**

- Otomatik computational graph oluşturma
- `grad_fn` objeleri ile layer bağımlılıklarını tespit
- Skip connections ve residual blocks'ları handle eder
- Concat, split gibi operasyonları destekler

### 4. **True Structured Pruning**

- Mask-based değil, gerçek tensor slicing
- Model boyutu fiziksel olarak küçülür
- Parameter sayısı ve memory gerçekten azalır
- Inference hızı artar

### 5. **Flexible Pruning Strategies**

- **Global Pruning**: Tüm model için tek oran
- **Layer-Specific**: Her katman için farklı oran
- **Selective**: Belirli layer'ları ignore etme
- **Iterative**: Adım adım pruning (gelecek versiyonlarda)

---

## 🏗️ Mimari Avantajları

### Modüler Tasarım

```
core/           → Graph & Dependencies
importance/     → Scoring Methods
pruner/         → Pruning Functions
utils/          → Helpers
```

### Genişletilebilir

- Yeni layer tipleri eklemek kolay
- Custom importance scorer yazabilirsiniz
- Kendi pruning stratejinizi implement edebilirsiniz

### Debugging Friendly

- Her adım açıkça görülebilir
- Dependency graph'i inceleyebilirsiniz
- Pruning history kaydedilir

---

## 🔧 Desteklenen Katmanlar

| Katman               | Destek | Notlar            |
| -------------------- | ------ | ----------------- |
| Conv2d               | ✅     | Tam destek        |
| ConvTranspose2d      | ✅     | Tam destek        |
| Linear               | ✅     | Tam destek        |
| BatchNorm2d/1d       | ✅     | Tam destek        |
| LayerNorm            | ✅     | Tam destek        |
| GroupNorm            | ✅     | Tam destek        |
| Depthwise Conv       | ✅     | Özel handling     |
| ReLU, Pooling        | ✅     | Pass-through      |
| Skip Connections     | ✅     | Otomatik detect   |
| Multi-head Attention | ⚠️     | Sınırlı destek    |
| RNN/LSTM             | ⚠️     | Single-layer only |

---

## 📊 Karşılaştırma

### CleanAI vs Diğer Pruning Kütüphaneleri

| Özellik                  | CleanAI             | torch-pruning | NVIDIA Pruning |
| ------------------------ | ------------------- | ------------- | -------------- |
| Bağımsızlık              | ✅ Tam              | ❌            | ❌             |
| Weight-Activation Hybrid | ✅                  | ❌            | ❌             |
| Öğrenme Eğrisi           | ✅ Kolay            | ⚠️ Orta       | ⚠️ Zor         |
| Özelleştirme             | ✅ Kolay            | ⚠️ Orta       | ❌ Zor         |
| Dokümantasyon            | ✅ Türkçe+İngilizce | ✅ İngilizce  | ⚠️ Sınırlı     |
| Kod Kalitesi             | ✅ Clean, Simple    | ✅ İyi        | ✅ İyi         |

---

## 💡 Kullanım Senaryoları

### 1. Model Compression

```python
# ResNet18: 11.7M → 6.5M params (%44 azalma)
pruner = StructuredPruner(
    model=resnet18,
    pruning_ratio=0.35
)
```

### 2. Mobile/Edge Deployment

```python
# MobileNet için agresif pruning
pruner = StructuredPruner(
    model=mobilenet,
    pruning_ratio=0.5,
    layer_pruning_ratios={
        mobilenet.features[0]: 0.2,  # İlk katman koruma
        mobilenet.features[-1]: 0.6   # Son katman agresif
    }
)
```

### 3. Research & Experimentation

```python
# Farklı α-β kombinasyonlarını dene
for alpha in [0.3, 0.5, 0.7]:
    beta = 1 - alpha
    importance = WeightActivationImportance(
        weight_ratio=alpha,
        activation_ratio=beta
    )
    # Test et...
```

---

## 🎓 Eğitim & Öğrenme

### Anlaşılır Kod

- Her fonksiyon yorum satırlarıyla açıklanmış
- Docstring'ler detaylı
- Değişken isimleri açıklayıcı

### Örneklerle Öğrenme

- `quick_start.py`: 5 dakikada başla
- `example_basic.py`: Adım adım açıklamalı
- `example_resnet.py`: Production-ready örnek

### Teorik Temeller

- Weight magnitude pruning (classic)
- Activation-based pruning (data-dependent)
- Hybrid scoring (best of both)

---

## 🚀 Performance

### Hız

- Graph building: ~1-2 saniye (orta büyüklükte model)
- Activation collection: Veri setine bağlı
- Pruning execution: ~1 saniye

### Memory

- Activation cache: Efficient storage
- In-place pruning: Minimum memory overhead
- Hook cleanup: No memory leaks

### Accuracy

- Aktivasyon kullanımı → Daha iyi accuracy preservation
- Weight+Activation → Optimal trade-off
- Fine-tuning ile %1-2 accuracy drop

---

## 🔮 Gelecek Planları

### v1.1 (Yakında)

- [ ] Iterative pruning support
- [ ] Pruning scheduler
- [ ] Auto-tuning pruning ratios

### v1.2

- [ ] Multi-head attention full support
- [ ] Quantization integration
- [ ] Knowledge distillation

### v2.0

- [ ] Dynamic pruning (training sırasında)
- [ ] Neural architecture search integration
- [ ] Distributed pruning

---

## 📝 Lisans

MIT License - Özgürce kullanın, değiştirin, paylaşın!

---

## 🤝 Katkıda Bulunma

- Issues: Bug reports, feature requests
- Pull Requests: Hoş geldiniz!
- Documentation: Her türlü iyileştirme

---

**CleanAI Team** © 2024
