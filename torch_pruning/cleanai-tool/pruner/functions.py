"""
Pruning Functions: Her katman tipi için pruning fonksiyonları
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Callable


class BasePruner(ABC):
    """
    Temel pruner sınıfı. Her katman tipi için prune_out_channels ve 
    prune_in_channels fonksiyonları implement edilmeli.
    """
    
    def __init__(self, pruning_dim: int = 1):
        self.pruning_dim = pruning_dim
    
    @abstractmethod
    def prune_out_channels(self, layer: nn.Module, idxs: List[int]) -> nn.Module:
        """Output channels'ı prune et"""
        pass
    
    @abstractmethod
    def prune_in_channels(self, layer: nn.Module, idxs: List[int]) -> nn.Module:
        """Input channels'ı prune et"""
        pass
    
    @abstractmethod
    def get_out_channels(self, layer: nn.Module) -> int:
        """Output channel sayısını getir"""
        pass
    
    @abstractmethod
    def get_in_channels(self, layer: nn.Module) -> int:
        """Input channel sayısını getir"""
        pass
    
    def _prune_parameter(self, param: nn.Parameter, keep_idxs: List[int], dim: int) -> nn.Parameter:
        """
        Bir parametreyi belirtilen dimension'da prune et
        """
        keep_idxs_tensor = torch.LongTensor(keep_idxs).to(param.device)
        pruned = torch.index_select(param.data, dim, keep_idxs_tensor).contiguous()
        new_param = nn.Parameter(pruned)
        
        # Gradient varsa onu da prune et
        if param.grad is not None:
            pruned_grad = torch.index_select(param.grad, dim, keep_idxs_tensor)
            new_param.grad = pruned_grad
        
        return new_param


class Conv2dPruner(BasePruner):
    """Conv2d ve ConvTranspose2d için pruner"""
    
    def prune_out_channels(self, layer: nn.Module, idxs: List[int]) -> nn.Module:
        keep_idxs = sorted(list(set(range(layer.out_channels)) - set(idxs)))
        
        # Output channels güncelle
        layer.out_channels = len(keep_idxs)
        
        # Weight'i prune et
        if hasattr(layer, 'transposed') and layer.transposed:
            # ConvTranspose: weight shape is [in_channels, out_channels, kH, kW]
            layer.weight = self._prune_parameter(layer.weight, keep_idxs, dim=1)
        else:
            # Conv2d: weight shape is [out_channels, in_channels, kH, kW]
            layer.weight = self._prune_parameter(layer.weight, keep_idxs, dim=0)
        
        # Bias varsa onu da prune et
        if layer.bias is not None:
            layer.bias = self._prune_parameter(layer.bias, keep_idxs, dim=0)
        
        return layer
    
    def prune_in_channels(self, layer: nn.Module, idxs: List[int]) -> nn.Module:
        keep_idxs = sorted(list(set(range(layer.in_channels)) - set(idxs)))
        
        # Input channels güncelle
        layer.in_channels = len(keep_idxs)
        
        # Group conv için özel işlem
        if layer.groups > 1:
            keep_idxs = keep_idxs[:len(keep_idxs) // layer.groups]
        
        # Weight'i prune et
        if hasattr(layer, 'transposed') and layer.transposed:
            layer.weight = self._prune_parameter(layer.weight, keep_idxs, dim=0)
        else:
            layer.weight = self._prune_parameter(layer.weight, keep_idxs, dim=1)
        
        return layer
    
    def get_out_channels(self, layer: nn.Module) -> int:
        return layer.out_channels
    
    def get_in_channels(self, layer: nn.Module) -> int:
        return layer.in_channels


class DepthwiseConv2dPruner(Conv2dPruner):
    """Depthwise Convolution için pruner (groups == in_channels == out_channels)"""
    
    def prune_out_channels(self, layer: nn.Module, idxs: List[int]) -> nn.Module:
        keep_idxs = sorted(list(set(range(layer.out_channels)) - set(idxs)))
        
        # Depthwise conv: in_channels = out_channels = groups
        layer.out_channels = len(keep_idxs)
        layer.in_channels = len(keep_idxs)
        layer.groups = len(keep_idxs)
        
        # Weight prune (depthwise: [out_channels, 1, kH, kW])
        layer.weight = self._prune_parameter(layer.weight, keep_idxs, dim=0)
        
        if layer.bias is not None:
            layer.bias = self._prune_parameter(layer.bias, keep_idxs, dim=0)
        
        return layer
    
    # Depthwise conv için in ve out aynı
    prune_in_channels = prune_out_channels


class LinearPruner(BasePruner):
    """nn.Linear için pruner"""
    
    def prune_out_channels(self, layer: nn.Linear, idxs: List[int]) -> nn.Linear:
        keep_idxs = sorted(list(set(range(layer.out_features)) - set(idxs)))
        
        layer.out_features = len(keep_idxs)
        
        # Weight: [out_features, in_features]
        layer.weight = self._prune_parameter(layer.weight, keep_idxs, dim=0)
        
        if layer.bias is not None:
            layer.bias = self._prune_parameter(layer.bias, keep_idxs, dim=0)
        
        return layer
    
    def prune_in_channels(self, layer: nn.Linear, idxs: List[int]) -> nn.Linear:
        # idxs boşsa hiçbir şey yapma
        if len(idxs) == 0:
            return layer
            
        keep_idxs = sorted(list(set(range(layer.in_features)) - set(idxs)))
        
        new_in_features = len(keep_idxs)
        
        # Eğer in_features değişiyorsa, weight'i prune et
        if new_in_features != layer.in_features:
            layer.in_features = new_in_features
            layer.weight = self._prune_parameter(layer.weight, keep_idxs, dim=1)
        
        return layer
    
    def get_out_channels(self, layer: nn.Linear) -> int:
        return layer.out_features
    
    def get_in_channels(self, layer: nn.Linear) -> int:
        return layer.in_features


class BatchNorm2dPruner(BasePruner):
    """BatchNorm2d/BatchNorm1d için pruner"""
    
    def prune_out_channels(self, layer: nn.Module, idxs: List[int]) -> nn.Module:
        keep_idxs = sorted(list(set(range(layer.num_features)) - set(idxs)))
        
        layer.num_features = len(keep_idxs)
        
        # Running statistics
        if layer.track_running_stats:
            layer.running_mean = layer.running_mean.data[keep_idxs]
            layer.running_var = layer.running_var.data[keep_idxs]
        
        # Learnable parameters
        if layer.affine:
            layer.weight = self._prune_parameter(layer.weight, keep_idxs, dim=0)
            layer.bias = self._prune_parameter(layer.bias, keep_idxs, dim=0)
        
        return layer
    
    # BatchNorm için in ve out aynı
    prune_in_channels = prune_out_channels
    
    def get_out_channels(self, layer: nn.Module) -> int:
        return layer.num_features
    
    def get_in_channels(self, layer: nn.Module) -> int:
        return layer.num_features


class LayerNormPruner(BasePruner):
    """LayerNorm için pruner"""
    
    def __init__(self, pruning_dim: int = -1):
        super().__init__(pruning_dim)
    
    def prune_out_channels(self, layer: nn.LayerNorm, idxs: List[int]) -> nn.LayerNorm:
        pruning_dim = self.pruning_dim
        
        if len(layer.normalized_shape) < abs(pruning_dim):
            return layer
        
        num_features = layer.normalized_shape[pruning_dim]
        keep_idxs = sorted(list(set(range(num_features)) - set(idxs)))
        
        if layer.elementwise_affine:
            layer.weight = self._prune_parameter(layer.weight, keep_idxs, dim=pruning_dim)
            if layer.bias is not None:
                layer.bias = self._prune_parameter(layer.bias, keep_idxs, dim=pruning_dim)
        
        # normalized_shape'i güncelle
        if pruning_dim != -1:
            layer.normalized_shape = (
                layer.normalized_shape[:pruning_dim] + 
                (len(keep_idxs),) + 
                layer.normalized_shape[pruning_dim+1:]
            )
        else:
            layer.normalized_shape = layer.normalized_shape[:pruning_dim] + (len(keep_idxs),)
        
        return layer
    
    prune_in_channels = prune_out_channels
    
    def get_out_channels(self, layer: nn.LayerNorm) -> int:
        return layer.normalized_shape[self.pruning_dim]
    
    def get_in_channels(self, layer: nn.LayerNorm) -> int:
        return layer.normalized_shape[self.pruning_dim]


class GroupNormPruner(BasePruner):
    """GroupNorm için pruner"""
    
    def prune_out_channels(self, layer: nn.GroupNorm, idxs: List[int]) -> nn.GroupNorm:
        keep_idxs = sorted(list(set(range(layer.num_channels)) - set(idxs)))
        
        layer.num_channels = len(keep_idxs)
        
        if layer.affine:
            layer.weight = self._prune_parameter(layer.weight, keep_idxs, dim=0)
            layer.bias = self._prune_parameter(layer.bias, keep_idxs, dim=0)
        
        return layer
    
    prune_in_channels = prune_out_channels
    
    def get_out_channels(self, layer: nn.GroupNorm) -> int:
        return layer.num_channels
    
    def get_in_channels(self, layer: nn.GroupNorm) -> int:
        return layer.num_channels


# Pruner sözlüğü: layer type -> pruner instance
PRUNER_DICT = {
    nn.Conv2d: Conv2dPruner(),
    nn.ConvTranspose2d: Conv2dPruner(),
    nn.Linear: LinearPruner(),
    nn.BatchNorm2d: BatchNorm2dPruner(),
    nn.BatchNorm1d: BatchNorm2dPruner(),
    nn.LayerNorm: LayerNormPruner(),
    nn.GroupNorm: GroupNormPruner(),
}


def get_pruner(layer: nn.Module) -> BasePruner:
    """Bir layer için uygun pruner'ı getir"""
    layer_type = type(layer)
    
    # Depthwise conv kontrolü
    if isinstance(layer, nn.Conv2d):
        if layer.groups == layer.out_channels and layer.out_channels > 1:
            return DepthwiseConv2dPruner()
    
    return PRUNER_DICT.get(layer_type, None)


def is_prunable(layer: nn.Module) -> bool:
    """Bir layer prunable mı?"""
    return get_pruner(layer) is not None
