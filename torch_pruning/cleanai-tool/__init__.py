"""
CleanAI Structured Pruning Tool
================================

Tamamen PyTorch ve temel Python kütüphaneleri kullanılarak geliştirilmiş
structured pruning aracı. Weight-Activation hibrit önem skoru kullanır.

Temel Kullanım:
--------------
    import torch
    from cleanai-tool import StructuredPruner, WeightActivationImportance
    
    # Model ve veri hazırlama
    model = YourModel()
    test_loader = your_test_dataloader
    
    # Pruner oluşturma
    importance = WeightActivationImportance()
    pruner = StructuredPruner(
        model=model,
        importance=importance,
        pruning_ratio=0.3
    )
    
    # Aktivasyonları toplama
    pruner.collect_activations(test_loader)
    
    # Pruning uygulama
    pruner.prune()
"""

__version__ = "1.0.0"
__author__ = "CleanAI Team"

from .pruner.structured_pruner import StructuredPruner
from .importance.weight_activation import WeightActivationImportance
from .core.graph import DependencyGraph
from .core.group import PruningGroup

__all__ = [
    'StructuredPruner',
    'WeightActivationImportance',
    'DependencyGraph',
    'PruningGroup',
]
