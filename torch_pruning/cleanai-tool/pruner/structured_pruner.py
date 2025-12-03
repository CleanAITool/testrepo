"""
StructuredPruner: Ana pruning sınıfı
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Union
import warnings

try:
    from core.graph import DependencyGraph
    from core.group import PruningGroup
    from importance.weight_activation import WeightActivationImportance
    from pruner.functions import get_pruner
except ImportError:
    from ..core.graph import DependencyGraph
    from ..core.group import PruningGroup
    from ..importance.weight_activation import WeightActivationImportance
    from ..pruner.functions import get_pruner


class StructuredPruner:
    """
    Structured Pruner: Model üzerinde structured pruning uygular
    
    Bu sınıf:
    1. Dependency graph oluşturur (autograd tracing ile)
    2. Pruning group'ları belirler
    3. Importance score hesaplar (Weight-Activation hybrid)
    4. En düşük skorlu channel'ları pruning eder
    
    Args:
        model: Pruning yapılacak PyTorch modeli
        example_inputs: Graph tracing için örnek input
        importance: Importance scorer (default: WeightActivationImportance)
        pruning_ratio: Global pruning oranı (0.0 - 1.0)
        layer_pruning_ratios: Layer-spesifik pruning oranları
        ignored_layers: İhmal edilecek layer'lar
        device: Cihaz (cuda/cpu)
    """
    
    def __init__(
        self,
        model: nn.Module,
        example_inputs: torch.Tensor,
        importance: Optional[WeightActivationImportance] = None,
        pruning_ratio: float = 0.3,
        layer_pruning_ratios: Optional[Dict[nn.Module, float]] = None,
        ignored_layers: Optional[List[nn.Module]] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.pruning_ratio = pruning_ratio
        self.layer_pruning_ratios = layer_pruning_ratios or {}
        self.ignored_layers = ignored_layers or []
        self.example_inputs = example_inputs.to(device)  # Validation için sakla
        
        # Importance scorer
        if importance is None:
            self.importance = WeightActivationImportance()
        else:
            self.importance = importance
        
        # Dependency graph oluştur
        print("Building dependency graph...")
        self.DG = DependencyGraph()
        self.DG.build(
            model=model,
            example_inputs=self.example_inputs,
            ignored_layers=self.ignored_layers
        )
        print(f"✓ Dependency graph built: {len(self.DG.module2node)} nodes")
        
        # Initial channel counts
        self.initial_channels: Dict[nn.Module, int] = {}
        for module in self.DG.module2node.keys():
            pruner = get_pruner(module)
            if pruner:
                out_ch = pruner.get_out_channels(module)
                if out_ch:
                    self.initial_channels[module] = out_ch
    
    def collect_activations(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: Optional[int] = None,
        max_batches: Optional[int] = None
    ):
        """
        Test veri seti üzerinden aktivasyonları topla
        
        Args:
            dataloader: PyTorch DataLoader
            num_batches: Kaç batch kullanılacak (None = tümü) - deprecated, max_batches kullan
            max_batches: Kaç batch kullanılacak (None = tümü)
        """
        # Backward compatibility
        if max_batches is None:
            max_batches = num_batches
        
        print("Collecting activations...")
        self.importance.collect_activations(
            model=self.model,
            dataloader=dataloader,
            num_batches=max_batches,
            device=self.device
        )
        print(f"✓ Activations collected from {max_batches or len(dataloader)} batches")
    
    def _get_pruning_ratio_for_layer(self, module: nn.Module) -> float:
        """Bir layer için pruning ratio'yu getir"""
        return self.layer_pruning_ratios.get(module, self.pruning_ratio)
    
    def _compute_importance_for_group(self, group: PruningGroup) -> torch.Tensor:
        """Bir group için importance score hesapla"""
        importance = self.importance(group)
        
        if importance is None:
            warnings.warn(f"Could not compute importance for group: {group}")
            return None
        
        return importance
    
    def _select_channels_to_prune(
        self,
        importance: torch.Tensor,
        num_channels: int,
        pruning_ratio: float
    ) -> List[int]:
        """
        Importance skorlarına göre prune edilecek channel'ları seç
        
        Args:
            importance: Channel importance skorları [num_channels]
            num_channels: Toplam channel sayısı
            pruning_ratio: Pruning oranı
        
        Returns:
            prune_idxs: Prune edilecek channel indeksleri
        """
        num_to_prune = int(num_channels * pruning_ratio)
        
        if num_to_prune == 0:
            return []
        
        # En düşük skorlu channel'ları seç
        _, sorted_idxs = torch.sort(importance)
        prune_idxs = sorted_idxs[:num_to_prune].tolist()
        
        return prune_idxs
    
    def prune(self) -> nn.Module:
        """
        Model'i prune et
        
        Returns:
            pruned_model: Pruning yapılmış model
        """
        print("\n" + "="*60)
        print("Starting Structured Pruning")
        print("="*60)
        
        # Tüm prunable group'ları al
        all_groups = self.DG.get_all_prunable_groups()
        print(f"\nFound {len(all_groups)} prunable groups")
        
        total_pruned = 0
        total_channels = 0
        
        # Her group için pruning yap
        for i, group in enumerate(all_groups):
            # Root module'ü al
            root_module = group[0].dep.target.module
            root_idxs = group[0].idxs
            num_channels = len(root_idxs)
            
            # Pruning ratio
            pruning_ratio = self._get_pruning_ratio_for_layer(root_module)
            
            if pruning_ratio == 0:
                continue
            
            total_channels += num_channels
            
            # Importance hesapla
            importance = self._compute_importance_for_group(group)
            
            if importance is None:
                continue
            
            # Prune edilecek channel'ları seç
            prune_idxs = self._select_channels_to_prune(
                importance,
                num_channels,
                pruning_ratio
            )
            
            if len(prune_idxs) == 0:
                continue
            
            total_pruned += len(prune_idxs)
            
            # Yeni group oluştur (sadece prune edilecek channel'larla)
            pruner = get_pruner(root_module)
            prune_group = self.DG.get_pruning_group(
                root_module,
                pruner.prune_out_channels,
                prune_idxs
            )
            
            # Pruning uygula
            prune_group.prune()
            
            # Progress
            module_name = self.DG._module2name.get(root_module, str(type(root_module).__name__))
            print(f"[{i+1}/{len(all_groups)}] {module_name}: "
                  f"{num_channels} -> {num_channels - len(prune_idxs)} channels "
                  f"(pruned {len(prune_idxs)})")
        
        # İstatistikler
        print("="*60)
        print("Pruning Summary")
        print("="*60)
        print(f"Total channels before: {total_channels}")
        print(f"Total channels pruned: {total_pruned}")
        if total_channels > 0:
            print(f"Effective pruning ratio: {total_pruned / total_channels:.2%}")
        else:
            print("Warning: No prunable channels found!")
        
        # Model parametrelerini say
        total_params_before = sum(p.numel() for p in self.model.parameters())
        
        # Yeniden hesapla (pruning sonrası)
        total_params_after = sum(p.numel() for p in self.model.parameters())
        
        print(f"\nParameters before: {total_params_before:,}")
        print(f"Parameters after: {total_params_after:,}")
        print(f"Parameters reduced: {total_params_before - total_params_after:,} "
              f"({(1 - total_params_after/total_params_before):.2%})")
        print("="*60 + "\n")
        
        # Model'i validate et ve gerekirse linear layer'ları fix et
        self._fix_linear_layers_if_needed()
        
        return self.model
    
    def _fix_linear_layers_if_needed(self):
        """
        Pruning sonrası linear layer'ların input shape'lerini kontrol et ve gerekirse düzelt.
        Bu, flatten/view sonrası linear layer'lar için otomatik shape adaptation sağlar.
        """
        try:
            # Test forward pass
            with torch.no_grad():
                self.model.eval()
                _ = self.model(self.example_inputs)
            # Başarılı, düzeltmeye gerek yok
        except RuntimeError as e:
            if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                print("\n⚠ Detected shape mismatch in linear layers, auto-fixing...")
                
                # Pre-hook kullanarak linear layer'ların gerçek input shape'lerini yakala
                linear_actual_inputs = {}
                hooks = []
                
                def pre_hook_fn(name):
                    def fn(module, inp):
                        if isinstance(module, nn.Linear) and len(inp) > 0:
                            # Flatten edilmiş input shape'i
                            if inp[0].dim() > 2:
                                actual_in = inp[0].view(inp[0].size(0), -1).shape[1]
                            else:
                                actual_in = inp[0].shape[-1]
                            linear_actual_inputs[name] = actual_in
                    return fn
                
                # Linear layer'lara pre-hook ekle
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Linear):
                        h = module.register_forward_pre_hook(pre_hook_fn(name))
                        hooks.append(h)
                
                # Forward pass yap (hata olsa bile hook'lar çalışır)
                try:
                    with torch.no_grad():
                        _ = self.model(self.example_inputs)
                except:
                    pass  # Hata bekleniyor
                
                # Hook'ları temizle
                for h in hooks:
                    h.remove()
                
                # Linear layer'ları düzelt
                fixed_count = 0
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Linear) and name in linear_actual_inputs:
                        actual_in = linear_actual_inputs[name]
                        if actual_in != module.in_features:
                            print(f"  Fixing {name}: {module.in_features} -> {actual_in} input features")
                            
                            # Yeni weight oluştur (Xavier initialization)
                            new_weight = nn.Parameter(torch.empty(module.out_features, actual_in, device=module.weight.device))
                            nn.init.xavier_uniform_(new_weight)
                            module.weight = new_weight
                            module.in_features = actual_in
                            fixed_count += 1
                
                print(f"✓ Fixed {fixed_count} linear layer(s)\n")
                
                # Tekrar validate et (cascade fix için)
                if fixed_count > 0:
                    self._fix_linear_layers_if_needed()  # Recursive call
            else:
                # Başka bir hata, raise et
                raise
    
    def get_pruned_model_summary(self) -> Dict:
        """
        Pruning sonrası model özeti
        
        Returns:
            summary: Model özet bilgileri
        """
        summary = {
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'layers': {}
        }
        
        for module, node in self.DG.module2node.items():
            pruner = get_pruner(module)
            if pruner:
                module_name = self.DG._module2name.get(module, str(type(module).__name__))
                
                initial_ch = self.initial_channels.get(module, None)
                current_ch = pruner.get_out_channels(module)
                
                if initial_ch and current_ch:
                    summary['layers'][module_name] = {
                        'type': type(module).__name__,
                        'initial_channels': initial_ch,
                        'current_channels': current_ch,
                        'pruned_channels': initial_ch - current_ch,
                        'pruning_ratio': (initial_ch - current_ch) / initial_ch
                    }
        
        return summary
    
    def print_summary(self):
        """Model özetini yazdır"""
        summary = self.get_pruned_model_summary()
        
        print("\n" + "="*70)
        print(" " * 25 + "Model Summary")
        print("="*70)
        print(f"{'Layer':<30} {'Type':<15} {'Initial':<10} {'Current':<10} {'Pruned':<10}")
        print("-"*70)
        
        for name, info in summary['layers'].items():
            print(f"{name:<30} {info['type']:<15} "
                  f"{info['initial_channels']:<10} "
                  f"{info['current_channels']:<10} "
                  f"{info['pruned_channels']:<10}")
        
        print("="*70)
        print(f"Total Parameters: {summary['total_params']:,}")
        print("="*70 + "\n")
