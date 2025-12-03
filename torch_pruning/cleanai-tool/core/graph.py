"""
DependencyGraph: Autograd ile model tracing ve dependency graph oluşturma
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Union, Tuple
import warnings

try:
    from core.node import Node
    from core.dependency import Dependency
    from core.group import PruningGroup
    from pruner.functions import get_pruner, is_prunable
except ImportError:
    from .node import Node
    from .dependency import Dependency
    from .group import PruningGroup
    from ..pruner.functions import get_pruner, is_prunable


class DependencyGraph:
    """
    Dependency Graph: Model üzerinde forward pass yaparak autograd ile 
    computational graph çıkarır ve layer dependencies oluşturur.
    """
    
    def __init__(self):
        self.model = None
        self.module2node: Dict[nn.Module, Node] = {}
        self._module2name: Dict[nn.Module, str] = {}
        self._grad_fn2module: Dict = {}
        self._ignored_layers: List[nn.Module] = []
        
    def build(
        self,
        model: nn.Module,
        example_inputs: torch.Tensor,
        ignored_layers: Optional[List[nn.Module]] = None,
        forward_fn: Optional[Callable] = None
    ) -> 'DependencyGraph':
        """
        Model üzerinde tracing yaparak dependency graph oluştur
        
        Args:
            model: Pruning yapılacak PyTorch modeli
            example_inputs: Tracing için örnek input tensörü
            ignored_layers: İhmal edilecek layer'lar
            forward_fn: Custom forward fonksiyonu (opsiyonel)
        
        Returns:
            Self (chaining için)
        """
        self.model = model
        self._module2name = {m: name for name, m in model.named_modules()}
        
        if ignored_layers:
            self._ignored_layers = ignored_layers
        
        # 1. Trace the model
        self._trace_model(model, example_inputs, forward_fn)
        
        # 2. Build dependencies
        self._build_dependencies()
        
        return self
    
    def _trace_model(
        self,
        model: nn.Module,
        example_inputs: torch.Tensor,
        forward_fn: Optional[Callable] = None
    ):
        """
        Model'i trace et: forward hooks kullanarak her module'ün grad_fn'ini yakala
        """
        model.eval()
        visited = {}
        
        def _record_grad_fn(module, inputs, outputs):
            """Her module'ün grad_fn'ini kaydet"""
            if module not in visited:
                visited[module] = 0
            visited[module] += 1
            
            # Output'u normalize et
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # grad_fn'i kaydet
            if hasattr(outputs, 'grad_fn') and outputs.grad_fn is not None:
                self._grad_fn2module[outputs.grad_fn] = module
        
        # Prunable layer'lara hook ekle
        hooks = []
        for module in model.modules():
            if is_prunable(module) and module not in self._ignored_layers:
                hook = module.register_forward_hook(_record_grad_fn)
                hooks.append(hook)
        
        # Forward pass yap (grad_fn için grad enabled olmalı)
        if forward_fn is not None:
            outputs = forward_fn(model, example_inputs)
        else:
            if isinstance(example_inputs, dict):
                outputs = model(**example_inputs)
            elif isinstance(example_inputs, (list, tuple)):
                outputs = model(*example_inputs)
            else:
                outputs = model(example_inputs)
        
        # Hook'ları temizle
        for hook in hooks:
            hook.remove()
        
        # Reused modules (birden fazla kez kullanılan)
        reused = [m for m, count in visited.items() if count > 1]
        
        # Computational graph'i oluştur
        if isinstance(outputs, (list, tuple)):
            for out in outputs:
                if isinstance(out, torch.Tensor):
                    self._build_computational_graph(out, reused)
        else:
            self._build_computational_graph(outputs, reused)
    
    def _build_computational_graph(
        self,
        output_tensor: torch.Tensor,
        reused_modules: List[nn.Module]
    ):
        """
        Output tensor'dan geriye doğru grad_fn'leri takip ederek graph oluştur
        """
        if not hasattr(output_tensor, 'grad_fn') or output_tensor.grad_fn is None:
            return
        
        visited = set()
        stack = [output_tensor.grad_fn]
        
        while stack:
            grad_fn = stack.pop()
            
            if grad_fn in visited:
                continue
            visited.add(grad_fn)
            
            # Bu grad_fn'e karşılık gelen module'ü bul
            module = self._grad_fn2module.get(grad_fn, None)
            
            if module is not None:
                # Node oluştur
                if module not in self.module2node or module in reused_modules:
                    node = Node(
                        module=module,
                        grad_fn=grad_fn,
                        name=self._module2name.get(module, str(type(module).__name__))
                    )
                    if module not in self.module2node:
                        self.module2node[module] = node
                else:
                    node = self.module2node[module]
            
            # next_functions'ı takip et
            if hasattr(grad_fn, 'next_functions'):
                for next_fn, _ in grad_fn.next_functions:
                    if next_fn is not None:
                        # AccumulateGrad (leaf) kontrolü
                        if hasattr(next_fn, 'variable'):
                            # Bu bir leaf parameter
                            continue
                        
                        # Input module'ü bul (ara operasyonları atlayarak)
                        input_module = self._find_nearest_module(next_fn)
                        
                        if input_module is not None and module is not None:
                            # Node bağlantılarını kur
                            if input_module not in self.module2node:
                                # grad_fn'i modülden bul (tersine çevir)
                                input_grad_fn = None
                                for gfn, mod in self._grad_fn2module.items():
                                    if mod == input_module:
                                        input_grad_fn = gfn
                                        break
                                
                                input_node = Node(
                                    module=input_module,
                                    grad_fn=input_grad_fn,
                                    name=self._module2name.get(input_module, str(type(input_module).__name__))
                                )
                                self.module2node[input_module] = input_node
                            else:
                                input_node = self.module2node[input_module]
                            
                            # Bağlantıları ekle
                            node = self.module2node[module]
                            node.add_input(input_node)
                            input_node.add_output(node)
                        
                        stack.append(next_fn)
    
    def _find_nearest_module(self, grad_fn) -> Optional[nn.Module]:
        """
        Bir grad_fn'den geriye doğru giderek en yakın prunable module'ü bul.
        ReLU, MaxPool gibi ara operasyonları atlar.
        """
        if grad_fn is None:
            return None
        
        # Direkt module varsa döndür
        if grad_fn in self._grad_fn2module:
            return self._grad_fn2module[grad_fn]
        
        # BFS ile geri git
        visited = set()
        queue = [grad_fn]
        
        while queue:
            current_fn = queue.pop(0)
            
            if current_fn in visited:
                continue
            visited.add(current_fn)
            
            # Leaf mi kontrol et
            if hasattr(current_fn, 'variable'):
                continue
            
            # Module varsa döndür
            if current_fn in self._grad_fn2module:
                return self._grad_fn2module[current_fn]
            
            # next_functions'ları ekle
            if hasattr(current_fn, 'next_functions'):
                for next_fn, _ in current_fn.next_functions:
                    if next_fn is not None:
                        queue.append(next_fn)
        
        return None
    
    def _build_dependencies(self):
        """
        Node'lar arasındaki dependency'leri oluştur
        """
        for module, node in self.module2node.items():
            pruner = get_pruner(module)
            if pruner is None:
                continue
            
            # Inter-layer dependencies (layer'lar arası)
            # Rule 1: Input connections
            for input_node in node.inputs:
                input_pruner = get_pruner(input_node.module)
                if input_pruner is None:
                    continue
                
                # input_node'un output'u prune edildiğinde, bu node'un input'u prune edilmeli
                dep = Dependency(
                    trigger=input_pruner.prune_out_channels,
                    handler=pruner.prune_in_channels,
                    source=input_node,
                    target=node
                )
                node.dependencies.append(dep)
            
            # Rule 2: Output connections
            for output_node in node.outputs:
                output_pruner = get_pruner(output_node.module)
                if output_pruner is None:
                    continue
                
                # Bu node'un output'u prune edildiğinde, output_node'un input'u prune edilmeli
                dep = Dependency(
                    trigger=pruner.prune_out_channels,
                    handler=output_pruner.prune_in_channels,
                    source=node,
                    target=output_node
                )
                node.dependencies.append(dep)
            
            # Intra-layer dependency (layer içi)
            # BatchNorm gibi layer'larda input ve output aynı
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
                dep = Dependency(
                    trigger=pruner.prune_out_channels,
                    handler=pruner.prune_in_channels,
                    source=node,
                    target=node
                )
                node.dependencies.append(dep)
    
    def get_pruning_group(
        self,
        module: nn.Module,
        pruning_fn: Callable,
        idxs: List[int]
    ) -> PruningGroup:
        """
        Bir module ve pruning fonksiyonu için pruning group oluştur.
        Bu grup, birlikte prune edilmesi gereken tüm layer'ları içerir.
        
        Args:
            module: Root module (pruning'in başlayacağı layer)
            pruning_fn: Pruning fonksiyonu (prune_out_channels veya prune_in_channels)
            idxs: Prune edilecek channel indeksleri
        
        Returns:
            PruningGroup: Birlikte prune edilecek dependency'ler
        """
        if module not in self.module2node:
            raise ValueError(f"Module {module} is not in dependency graph")
        
        root_node = self.module2node[module]
        group = PruningGroup()
        group._dependency_graph = self
        
        # Root dependency'yi ekle
        root_dep = Dependency(pruning_fn, pruning_fn, root_node, root_node)
        group.add(root_dep, idxs)
        
        # BFS ile dependency'leri takip et
        visited = set()
        queue = [(root_dep, idxs)]
        
        while queue:
            current_dep, current_idxs = queue.pop(0)
            current_node = current_dep.target
            
            # Bu node'un tüm dependency'lerini kontrol et
            # Hem source node'dan (trigger edilen) hem de target node'dan (handler'ın uygulandığı)
            nodes_to_check = [current_node]
            if current_dep.source != current_node and current_dep.source not in visited:
                nodes_to_check.append(current_dep.source)
            
            for check_node in nodes_to_check:
                if check_node in visited:
                    continue
                visited.add(check_node)
                
                for dep in check_node.dependencies:
                    # Eğer bu dependency current handler tarafından tetikleniyorsa
                    if dep.trigger == current_dep.handler:
                        # Bu dependency tetikleniyor
                        new_idxs = current_idxs.copy()
                        
                        # Index mapping varsa uygula (concat/split için)
                        for mapping in dep.index_mapping:
                            if mapping is not None:
                                new_idxs = mapping(new_idxs)
                        
                        if len(new_idxs) == 0:
                            continue
                        
                        # Gruba ekle (merge ile aynı dependency varsa birleştir)
                        group.merge(dep, new_idxs)
                        
                        # Queue'ya ekle
                        queue.append((dep, new_idxs))
        
        return group
    
    def get_out_channels(self, module: nn.Module) -> Optional[int]:
        """Bir module'ün output channel sayısını getir"""
        pruner = get_pruner(module)
        if pruner is None:
            return None
        return pruner.get_out_channels(module)
    
    def get_in_channels(self, module: nn.Module) -> Optional[int]:
        """Bir module'ün input channel sayısını getir"""
        pruner = get_pruner(module)
        if pruner is None:
            return None
        return pruner.get_in_channels(module)
    
    def get_all_prunable_groups(self) -> List[PruningGroup]:
        """
        Model'deki tüm prunable group'ları getir
        """
        groups = []
        visited = set()
        
        for module, node in self.module2node.items():
            if module in visited or module in self._ignored_layers:
                continue
            
            pruner = get_pruner(module)
            if pruner is None:
                continue
            
            # Conv ve Linear layer'lar için group oluştur
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                out_channels = pruner.get_out_channels(module)
                if out_channels is None:
                    continue
                
                # Tüm channel'larla group oluştur
                group = self.get_pruning_group(
                    module,
                    pruner.prune_out_channels,
                    list(range(out_channels))
                )
                
                # Bu group'taki tüm module'leri visited'e ekle
                for item in group:
                    visited.add(item.dep.target.module)
                
                groups.append(group)
        
        return groups
