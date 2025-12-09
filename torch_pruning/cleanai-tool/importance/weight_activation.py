"""
Weight-Activation Hybrid Importance: 
Ağırlık büyüklüğü ve aktivasyon katkısını birleştiren hibrit önem skoru
"""
import torch
import torch.nn as nn
from typing import Dict, Optional

try:
    from core.group import PruningGroup
    from pruner.functions import get_pruner
except ImportError:
    from ..core.group import PruningGroup
    from ..pruner.functions import get_pruner


class WeightActivationImportance:
    """
    Weight-Activation Hybrid Importance Scorer
    
    Önem skoru hesaplama formülü:
    importance(channel_i) = α × weight_score(i) + β × activation_score(i)
    
    - weight_score: Ağırlık normları (L2 norm)
    - activation_score: Aktivasyon katkısı (ortalama aktivasyon büyüklüğü)
    - α, β: ağırlık ve aktivasyon oranları (α + β = 1)
    
    Attributes:
        weight_ratio: Ağırlık skorunun önem oranı (default: 0.5)
        activation_ratio: Aktivasyon skorunun önem oranı (default: 0.5)
        p: Norm derecesi (L1: p=1, L2: p=2) (default: 2)
        normalize: Skorları normalize et (default: True)
    """
    
    def __init__(
        self,
        weight_ratio: float = 0.5,
        activation_ratio: float = 0.5,
        p: int = 2,
        normalize: bool = True
    ):
        assert abs(weight_ratio + activation_ratio - 1.0) < 1e-6, \
            "weight_ratio + activation_ratio toplamı 1 olmalı"
        
        self.weight_ratio = weight_ratio
        self.activation_ratio = activation_ratio
        self.p = p
        self.normalize = normalize
        
        # Aktivasyon değerlerini sakla
        self._activation_cache: Dict[nn.Module, torch.Tensor] = {}
        self._hooks = []
    
    def register_activation_hooks(self, model: nn.Module):
        """
        Model'e activation hook'ları ekle. Forward pass sırasında 
        aktivasyonları yakalayacak.
        """
        self.clear_activation_cache()
        
        def _hook_fn(module, input, output):
            """Aktivasyonları yakala"""
            if isinstance(output, tuple):
                output = output[0]
            
            # Detach et ve CPU'ya taşı (memory için)
            act = output.detach()
            
            # Spatial dimension'ları ortala (Conv için)
            if len(act.shape) == 4:  # [B, C, H, W]
                act = act.mean(dim=[0, 2, 3])  # [C]
            elif len(act.shape) == 3:  # [B, L, C]
                act = act.mean(dim=[0, 1])  # [C]
            elif len(act.shape) == 2:  # [B, C]
                act = act.mean(dim=0)  # [C]
            
            # Cache'e ekle (accumulate)
            if module in self._activation_cache:
                self._activation_cache[module] += act.abs()
            else:
                self._activation_cache[module] = act.abs()
        
        # Prunable layer'lara hook ekle
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                hook = module.register_forward_hook(_hook_fn)
                self._hooks.append(hook)
    
    def remove_activation_hooks(self):
        """Activation hook'ları kaldır"""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def remove_hooks(self):
        """Alias for remove_activation_hooks (API compatibility)"""
        self.remove_activation_hooks()
    
    def clear_activation_cache(self):
        """Activation cache'i temizle"""
        self._activation_cache = {}
    
    def collect_activations(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_batches: Optional[int] = None,
        max_batches: Optional[int] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Bir dataloader üzerinden forward pass yaparak aktivasyonları topla
        
        Args:
            model: PyTorch modeli
            dataloader: Veri yükleyici
            num_batches: Kaç batch kullanılacak (None = tümü) - deprecated, use max_batches
            max_batches: Kaç batch kullanılacak (None = tümü)
            device: Cihaz (cuda/cpu)
        """
        # Backward compatibility: max_batches takes precedence
        if max_batches is None:
            max_batches = num_batches
        model.eval()
        self.register_activation_hooks(model)
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if max_batches is not None and i >= max_batches:
                    break
                
                # Batch'i device'a taşı
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                elif isinstance(batch, dict):
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                             for k, v in batch.items()}
                else:
                    inputs = batch.to(device)
                
                # Forward pass
                _ = model(inputs)
        
        # Hook'ları kaldır
        self.remove_activation_hooks()
        
        # Normalize activations (batch sayısına böl)
        num_samples = min(max_batches, len(dataloader)) if max_batches else len(dataloader)
        for module in self._activation_cache:
            self._activation_cache[module] /= num_samples
    
    def _compute_weight_importance(
        self,
        module: nn.Module,
        idxs: list
    ) -> torch.Tensor:
        """
        Ağırlık bazlı önem skoru hesapla
        
        Weight importance = ||W_i||_p (L-p norm of weights)
        """
        pruner = get_pruner(module)
        if pruner is None:
            return None
        
        # Conv2d/Linear için weight shape: [out_ch, in_ch, ...]
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            if hasattr(module, 'transposed') and module.transposed:
                # ConvTranspose: [in_ch, out_ch, ...]
                weights = module.weight.data.transpose(0, 1)[idxs]
            else:
                weights = module.weight.data[idxs]
            
            # Flatten spatial dimensions
            weights = weights.flatten(1)  # [selected_channels, in_ch * kH * kW]
        
        elif isinstance(module, nn.Linear):
            weights = module.weight.data[idxs]  # [selected_channels, in_features]
        
        else:
            # BatchNorm, LayerNorm gibi
            if hasattr(module, 'weight') and module.weight is not None:
                weights = module.weight.data[idxs].unsqueeze(1)
            else:
                return None
        
        # L-p norm hesapla
        importance = weights.abs().pow(self.p).sum(dim=1)  # [selected_channels]
        
        if self.p != 1:
            importance = importance.pow(1.0 / self.p)
        
        return importance
    
    def _compute_activation_importance(
        self,
        module: nn.Module,
        idxs: list
    ) -> Optional[torch.Tensor]:
        """
        Aktivasyon bazlı önem skoru hesapla
        
        Activation importance = mean(|A_i|) (ortalama aktivasyon büyüklüğü)
        """
        if module not in self._activation_cache:
            return None
        
        # Cache'den aktivasyonları al
        activations = self._activation_cache[module][idxs]
        return activations
    
    def __call__(self, group: PruningGroup) -> torch.Tensor:
        """
        Bir pruning group için hibrit önem skoru hesapla
        
        Args:
            group: PruningGroup objesi
        
        Returns:
            importance_scores: Her channel için önem skoru [num_channels]
        """
        # İlk item'ı root olarak al
        root_item = group[0]
        root_module = root_item.dep.target.module
        root_idxs = root_item.idxs
        
        num_channels = len(root_idxs)
        device = next(root_module.parameters()).device
        
        # Weight importance
        weight_imp = self._compute_weight_importance(root_module, root_idxs)
        if weight_imp is None:
            return None
        weight_imp = weight_imp.to(device)
        
        # Activation importance
        activation_imp = self._compute_activation_importance(root_module, root_idxs)
        
        # Hibrit skor hesapla
        if activation_imp is not None:
            activation_imp = activation_imp.to(device)
            
            # Normalize (opsiyonel)
            if self.normalize:
                if weight_imp.max() > 0:
                    weight_imp = weight_imp / (weight_imp.max() + 1e-8)
                if activation_imp.max() > 0:
                    activation_imp = activation_imp / (activation_imp.max() + 1e-8)
            
            # Hibrit skor
            importance = (
                self.weight_ratio * weight_imp + 
                self.activation_ratio * activation_imp
            )
        else:
            # Aktivasyon yoksa sadece weight kullan
            importance = weight_imp
        
        return importance
    
    def __repr__(self):
        return (f"WeightActivationImportance("
                f"weight_ratio={self.weight_ratio}, "
                f"activation_ratio={self.activation_ratio}, "
                f"p={self.p}, "
                f"normalize={self.normalize})")
