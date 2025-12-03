"""
PruningGroup: Birlikte pruning yapılması gereken katman ve indeks çiftleri
"""
from typing import List, Tuple

try:
    from core.dependency import Dependency
except ImportError:
    from .dependency import Dependency


class GroupItem:
    """Bir pruning grubundaki tek bir öğe"""
    def __init__(self, dep: Dependency, idxs: List[int]):
        self.dep = dep
        self.idxs = idxs
    
    def __repr__(self):
        return f"GroupItem({self.dep}, idxs={self.idxs})"


class PruningGroup:
    """
    Pruning Group: Birlikte pruning yapılması gereken dependency ve index çiftleri.
    
    Örnek: Conv2d(3, 64) -> BN(64) -> ReLU
    Bu durumda Conv'un output channels'ı ve BN'nin channels'ı birlikte pruning edilmeli.
    
    Group = [(Conv.prune_out_channels, [0,1,2]), (BN.prune_channels, [0,1,2])]
    """
    
    def __init__(self):
        self._items: List[GroupItem] = []
        self._dependency_graph = None
    
    def add(self, dep: Dependency, idxs: List[int]):
        """Gruba yeni bir dependency-index çifti ekle"""
        self._items.append(GroupItem(dep, idxs))
    
    def merge(self, dep: Dependency, idxs: List[int]):
        """
        Eğer aynı dependency varsa, indeksleri merge et (birleştir)
        """
        for i, item in enumerate(self._items):
            if (item.dep.target == dep.target and 
                item.dep.handler == dep.handler):
                # Aynı dependency bulundu, indeksleri merge et
                merged_idxs = sorted(list(set(item.idxs + idxs)))
                self._items[i].idxs = merged_idxs
                return
        # Bulunamadı, yeni ekle
        self.add(dep, idxs)
    
    def prune(self):
        """Gruptaki tüm dependency'leri uygula (pruning yap)"""
        for item in self._items:
            if len(item.idxs) > 0:
                item.dep(item.idxs)
    
    def __len__(self):
        return len(self._items)
    
    def __getitem__(self, idx):
        return self._items[idx]
    
    def __setitem__(self, idx, value):
        self._items[idx] = value
    
    def __iter__(self):
        return iter(self._items)
    
    def __repr__(self):
        lines = ["\n" + "="*50]
        lines.append(" " * 15 + "Pruning Group")
        lines.append("="*50)
        for i, item in enumerate(self._items):
            lines.append(f"[{i}] {item.dep} | idxs: {len(item.idxs)}")
        lines.append("="*50)
        return "\n".join(lines)
    
    def details(self):
        """Detaylı bilgi göster (indekslerle birlikte)"""
        lines = ["\n" + "="*50]
        lines.append(" " * 15 + "Pruning Group (Detailed)")
        lines.append("="*50)
        for i, item in enumerate(self._items):
            root_indicator = " (ROOT)" if i == 0 else ""
            lines.append(f"[{i}] {item.dep}{root_indicator}")
            lines.append(f"     Indices ({len(item.idxs)}): {item.idxs}")
        lines.append("="*50)
        return "\n".join(lines)
