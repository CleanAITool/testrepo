"""
Dependency: Katmanlar arası bağımlılıkları temsil eden sınıf
"""
from typing import Callable, List

try:
    from core.node import Node
except ImportError:
    from .node import Node


class Dependency:
    """
    İki node arasındaki bağımlılık (dependency). 
    
    Eğer A -> B şeklinde bir dependency varsa:
    - A'da pruning yapıldığında (trigger), B'de de pruning yapılmalı (handler)
    
    Attributes:
        trigger: Bu dependency'yi tetikleyen pruning fonksiyonu
        handler: Tetiklendiğinde çalışacak pruning fonksiyonu
        source: Kaynak node (pruning'in başladığı yer)
        target: Hedef node (pruning'in uygulanacağı yer)
        index_mapping: Concat/split gibi işlemlerde index dönüşümü için
    """
    
    def __init__(
        self,
        trigger: Callable,
        handler: Callable,
        source: Node,
        target: Node
    ):
        self.trigger = trigger
        self.handler = handler
        self.source = source
        self.target = target
        
        # Index mapping fonksiyonları (concat/split için)
        # [0]: current -> standard coordinate
        # [1]: standard -> target coordinate
        self.index_mapping = [None, None]
    
    def __call__(self, idxs: List[int]):
        """
        Dependency'yi çalıştır: handler fonksiyonunu idxs ile çağır
        """
        if len(idxs) == 0:
            return
        
        # Pruning dimension'ı target node'dan al ve pruner instance'ına set et
        # handler bir bound method, __self__ ile pruner instance'a erişebiliriz
        if hasattr(self.handler, '__self__'):
            self.handler.__self__.pruning_dim = self.target.pruning_dim
        
        result = self.handler(self.target.module, idxs)
        return result
    
    def is_triggered_by(self, pruning_fn: Callable) -> bool:
        """Bu dependency belirli bir pruning fonksiyonu tarafından tetiklenir mi?"""
        return pruning_fn == self.trigger
    
    def __repr__(self):
        trigger_name = self.trigger.__name__ if self.trigger else "None"
        handler_name = self.handler.__name__ if self.handler else "None"
        return f"Dependency({trigger_name}@{self.source.name} -> {handler_name}@{self.target.name})"
    
    def __eq__(self, other):
        if not isinstance(other, Dependency):
            return False
        return (
            self.source == other.source and
            self.target == other.target and
            self.trigger == other.trigger and
            self.handler == other.handler
        )
    
    def __hash__(self):
        return hash((id(self.source.module), id(self.target.module), 
                     id(self.trigger), id(self.handler)))
    
    @property
    def layer(self):
        """Hedef modülün alias'ı"""
        return self.target.module
    
    @property
    def pruning_fn(self):
        """Handler fonksiyonun alias'ı"""
        return self.handler
