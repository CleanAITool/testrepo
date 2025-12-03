"""
Node: Dependency Graph'teki bir katmanı/operasyonu temsil eden sınıf
"""
import torch.nn as nn
from typing import List, Optional


class Node:
    """
    Dependency graph'te bir düğüm (node). Her node bir PyTorch modülünü 
    veya operasyonu (concat, split vb.) temsil eder.
    
    Attributes:
        module: PyTorch modülü veya operasyon
        grad_fn: Autograd grad_fn objesi
        name: Modül ismi
        inputs: Bu node'a giriş veren node'lar
        outputs: Bu node'dan çıkış alan node'lar
        dependencies: Bu node ile ilişkili dependency listesi
        pruning_dim: Pruning yapılacak dimension (genellikle 1 veya 0)
        enable_index_mapping: Concat/split için index mapping aktif mi
    """
    
    def __init__(
        self,
        module: nn.Module,
        grad_fn=None,
        name: Optional[str] = None
    ):
        self.module = module
        self.grad_fn = grad_fn
        self.name = name or str(type(module).__name__)
        
        # Graph connections
        self.inputs: List[Node] = []
        self.outputs: List[Node] = []
        
        # Dependencies (edges in the graph)
        self.dependencies: List = []  # List[Dependency]
        
        # Pruning dimension (0 for output channels, 1 for input channels typically)
        self.pruning_dim = 1
        
        # Index mapping için concat/split operasyonlarında kullanılır
        self.enable_index_mapping = True
        
    def add_input(self, node: 'Node'):
        """Bir input node ekle"""
        if node not in self.inputs:
            self.inputs.append(node)
    
    def add_output(self, node: 'Node'):
        """Bir output node ekle"""
        if node not in self.outputs:
            self.outputs.append(node)
    
    def __repr__(self):
        return f"Node({self.name})"
    
    def __str__(self):
        return self.name
    
    def __hash__(self):
        return id(self.module)
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.module is other.module
