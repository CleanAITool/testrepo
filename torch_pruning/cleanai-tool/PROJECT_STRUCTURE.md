"""
CleanAI Structured Pruning Tool - Proje Yapısı
==============================================

cleanai-tool/
│
├── README.md # Proje ana dokümantasyonu
├── USAGE_GUIDE.md # Detaylı kullanım kılavuzu
├── LICENSE # MIT License
├── **init**.py # Ana package init
│
├── core/ # Temel modüller
│ ├── **init**.py
│ ├── node.py # Graph node'ları
│ ├── dependency.py # Layer bağımlılıkları
│ ├── group.py # Pruning grupları
│ └── graph.py # Dependency graph (autograd tracing)
│
├── importance/ # Önem skorları
│ ├── **init**.py
│ └── weight_activation.py # Weight-Activation hibrit skoru
│
├── pruner/ # Pruning implementasyonları
│ ├── **init**.py
│ ├── functions.py # Layer-spesifik pruning fonksiyonları
│ └── structured_pruner.py # Ana pruner sınıfı
│
├── utils/ # Yardımcı fonksiyonlar
│ └── **init**.py
│
├── example_basic.py # Basit CNN örneği
└── example_resnet.py # ResNet18 örneği

## Temel Akış:

1. Graph Building (core/graph.py)

   - Forward hooks ile model trace'leme
   - grad_fn objeleri ile computational graph
   - Node ve Dependency oluşturma

2. Importance Scoring (importance/weight_activation.py)

   - Weight magnitude hesaplama (L2 norm)
   - Activation toplama (forward hooks)
   - Hibrit skor: α×weight + β×activation

3. Pruning (pruner/structured_pruner.py)

   - Pruning grupları oluşturma
   - Channel seçimi (en düşük skorlular)
   - Tensor slicing ile gerçek budama

4. Pruning Functions (pruner/functions.py)
   - Conv2d, Linear, BatchNorm, LayerNorm vb.
   - Depthwise convolutions
   - Parameter ve gradient güncelleme

## Dependency Resolution:

Conv2d(64, 128) → BN(128) → ReLU → Conv2d(128, 256)
│ │
└─── output channels (64) ──────────┘
│
├── triggers BN input channels
└── triggers next Conv2d input channels

## Weight-Activation Scoring:

importance(ch_i) = α × ||W_i||₂ + β × ||A_i||

- W_i: i. kanalın ağırlık normu
- A_i: i. kanalın ortalama aktivasyonu
- α + β = 1

## Kullanım:

from cleanai_tool import StructuredPruner, WeightActivationImportance

# Setup

importance = WeightActivationImportance(weight_ratio=0.5, activation_ratio=0.5)
pruner = StructuredPruner(model, example_inputs, importance, pruning_ratio=0.3)

# Collect activations

pruner.collect_activations(test_loader, num_batches=10)

# Prune

pruned_model = pruner.prune()
"""
