"""
NeuronCoverageImportance: Neuron Coverage bazlı importance scoring

Bu yöntem, test veri seti üzerinde nöronların ne sıklıkla aktif olduğunu
ölçerek önem skorları hesaplar. Nadiren aktif olan nöronlar düşük önem 
skoruna sahip olur ve pruning için aday olurlar.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np


class NeuronCoverageImportance:
    """
    Neuron Coverage bazlı importance scorer
    
    Metrikler:
    1. Activation Frequency: Nöronun kaç kez threshold'u geçtiği
    2. Activation Strength: Aktif olduğunda ortalama aktivasyon değeri
    3. Coverage Score: frequency × strength (hibrit skor)
    
    Args:
        threshold: Aktivasyon threshold'u (varsayılan: 0.0, ReLU için)
        metric: 'frequency', 'strength', veya 'coverage' (hibrit)
        percentile_threshold: Üst persentil için threshold (0-100)
        normalize: Skorları normalize et
    """
    
    def __init__(
        self,
        threshold: float = 0.0,
        metric: str = 'coverage',  # 'frequency', 'strength', 'coverage'
        percentile_threshold: Optional[float] = None,
        normalize: bool = True
    ):
        self.threshold = threshold
        self.metric = metric
        self.percentile_threshold = percentile_threshold
        self.normalize = normalize
        
        # Aktivasyon istatistikleri
        self.activation_counts: Dict[nn.Module, torch.Tensor] = {}
        self.activation_sums: Dict[nn.Module, torch.Tensor] = {}
        self.total_samples: int = 0
        self.hooks = []
        
        # Dynamic threshold (eğer percentile kullanılıyorsa)
        self.dynamic_thresholds: Dict[nn.Module, float] = {}
    
    def register_activation_hooks(self, model: nn.Module):
        """Model'e aktivasyon toplama hook'ları ekle"""
        self.hooks = []
        
        def make_hook(module):
            def hook(m, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                
                if not isinstance(output, torch.Tensor):
                    return
                
                # Detach and move to CPU to save memory
                act = output.detach()
                
                # Channel-wise işlem
                if len(act.shape) == 4:  # Conv: [B, C, H, W]
                    # Her channel için spatial boyutta max aktivasyonu al
                    act_max = act.max(dim=3)[0].max(dim=2)[0]  # [B, C]
                    
                    if module not in self.activation_counts:
                        num_channels = act.shape[1]
                        self.activation_counts[module] = torch.zeros(num_channels, device=act.device)
                        self.activation_sums[module] = torch.zeros(num_channels, device=act.device)
                    
                    # Threshold kontrolü
                    active_mask = (act_max > self.threshold).float()  # [B, C]
                    self.activation_counts[module] += active_mask.sum(dim=0)  # [C]
                    
                    # Aktif olduğunda strength topla
                    active_values = act_max * active_mask  # [B, C]
                    self.activation_sums[module] += active_values.sum(dim=0)  # [C]
                
                elif len(act.shape) == 2:  # Linear: [B, Features]
                    if module not in self.activation_counts:
                        num_features = act.shape[1]
                        self.activation_counts[module] = torch.zeros(num_features, device=act.device)
                        self.activation_sums[module] = torch.zeros(num_features, device=act.device)
                    
                    # Threshold kontrolü
                    active_mask = (act > self.threshold).float()  # [B, F]
                    self.activation_counts[module] += active_mask.sum(dim=0)  # [F]
                    
                    # Aktif olduğunda strength topla
                    active_values = act * active_mask  # [B, F]
                    self.activation_sums[module] += active_values.sum(dim=0)  # [F]
            
            return hook
        
        # Conv ve Linear katmanlara hook ekle
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hook = module.register_forward_hook(make_hook(module))
                self.hooks.append(hook)
        
        return self
    
    def collect_activations(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        max_batches: Optional[int] = None,
        device: str = 'cuda'
    ):
        """
        Dataloader üzerinden aktivasyonları topla
        
        Args:
            model: PyTorch modeli
            dataloader: DataLoader
            max_batches: Maksimum batch sayısı (None = tümü)
            device: Cihaz
        """
        model.eval()
        self.total_samples = 0
        
        print(f"Collecting neuron coverage statistics...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                # Input'u al
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0].to(device)
                    batch_size = inputs.shape[0]
                else:
                    inputs = batch.to(device)
                    batch_size = inputs.shape[0]
                
                # Forward pass
                try:
                    outputs = model(inputs)
                    self.total_samples += batch_size
                except Exception as e:
                    print(f"Warning: Forward pass failed on batch {batch_idx}: {e}")
                    continue
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1} batches...")
        
        print(f"✓ Collected activations from {self.total_samples} samples")
        
        # Percentile threshold hesapla (eğer isteniyorsa)
        if self.percentile_threshold is not None:
            self._compute_dynamic_thresholds()
        
        return self
    
    def _compute_dynamic_thresholds(self):
        """Percentile bazlı dynamic threshold hesapla"""
        print(f"Computing {self.percentile_threshold}th percentile thresholds...")
        
        for module, counts in self.activation_counts.items():
            if self.total_samples > 0:
                # Aktivasyon frekanslarını hesapla
                frequencies = counts.cpu().numpy() / self.total_samples
                
                # Percentile threshold
                threshold_val = np.percentile(frequencies, self.percentile_threshold)
                self.dynamic_thresholds[module] = threshold_val
                
                print(f"  {module.__class__.__name__}: threshold = {threshold_val:.4f}")
    
    def remove_hooks(self):
        """Hook'ları temizle"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def remove_activation_hooks(self):
        """Alias for remove_hooks (API compatibility)"""
        self.remove_hooks()
    
    def __call__(
        self,
        group,
        ch_groups: int = 1
    ) -> Optional[np.ndarray]:
        """
        Bir pruning group için importance skorları hesapla
        
        Args:
            group: PruningGroup instance
            ch_groups: Channel grouping (depthwise conv için)
        
        Returns:
            Importance skorları (düşük = daha az önemli = prune adayı)
        """
        # Root module'ü bul
        root_module = None
        for item in group:
            if isinstance(item.dep.target.module, (nn.Conv2d, nn.Linear)):
                root_module = item.dep.target.module
                break
        
        if root_module is None:
            return None
        
        # Bu module için coverage istatistikleri var mı?
        if root_module not in self.activation_counts:
            print(f"Warning: No coverage stats for {root_module.__class__.__name__}")
            return None
        
        counts = self.activation_counts[root_module].cpu().numpy()
        sums = self.activation_sums[root_module].cpu().numpy()
        
        if self.total_samples == 0:
            return None
        
        # Metric'e göre skor hesapla
        if self.metric == 'frequency':
            # Sadece aktivasyon frekansı
            scores = counts / self.total_samples
        
        elif self.metric == 'strength':
            # Sadece ortalama aktivasyon gücü (aktif olduğunda)
            # Hiç aktif olmayanlar için 0
            scores = np.zeros_like(sums)
            active_mask = counts > 0
            scores[active_mask] = sums[active_mask] / counts[active_mask]
        
        elif self.metric == 'coverage':
            # Hibrit: frequency × average strength
            frequency = counts / self.total_samples
            
            # Average strength hesapla
            avg_strength = np.zeros_like(sums)
            active_mask = counts > 0
            avg_strength[active_mask] = sums[active_mask] / counts[active_mask]
            
            # Combine
            scores = frequency * avg_strength
        
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        # Dynamic threshold uygula (eğer varsa)
        if self.percentile_threshold is not None and root_module in self.dynamic_thresholds:
            threshold = self.dynamic_thresholds[root_module]
            # Threshold'un altındaki nöronları penalize et
            below_threshold = (counts / self.total_samples) < threshold
            scores[below_threshold] *= 0.1  # Düşük skor ver
        
        # Normalize
        if self.normalize and scores.max() > 0:
            scores = scores / scores.max()
        
        # Channel grouping (depthwise için)
        if ch_groups > 1:
            scores = scores.reshape(ch_groups, -1).mean(axis=1)
        
        return scores
    
    def get_coverage_statistics(self) -> Dict[str, float]:
        """
        Coverage istatistiklerini döndür
        
        Returns:
            Dict: 'total_neurons', 'active_neurons', 'coverage_ratio', 'avg_frequency'
        """
        total_neurons = 0
        active_neurons = 0
        total_frequency = 0.0
        
        for module, counts in self.activation_counts.items():
            num_channels = len(counts)
            total_neurons += num_channels
            
            # En az bir kez aktif olanlar
            active_mask = counts > 0
            active_neurons += active_mask.sum().item()
            
            # Ortalama frekans
            if self.total_samples > 0:
                frequencies = counts.cpu().numpy() / self.total_samples
                total_frequency += frequencies.sum()
        
        coverage_ratio = active_neurons / total_neurons if total_neurons > 0 else 0.0
        avg_frequency = total_frequency / total_neurons if total_neurons > 0 else 0.0
        
        return {
            'total_neurons': total_neurons,
            'active_neurons': active_neurons,
            'coverage_ratio': coverage_ratio,
            'avg_frequency': avg_frequency
        }
    
    def print_coverage_report(self):
        """Coverage raporu yazdır"""
        stats = self.get_coverage_statistics()
        
        print("\n" + "="*60)
        print("NEURON COVERAGE REPORT")
        print("="*60)
        print(f"Total Neurons:     {stats['total_neurons']}")
        print(f"Active Neurons:    {stats['active_neurons']}")
        print(f"Coverage Ratio:    {stats['coverage_ratio']*100:.2f}%")
        print(f"Avg Frequency:     {stats['avg_frequency']*100:.2f}%")
        print("="*60)
        
        # Per-layer detayları
        print("\nPer-Layer Coverage:")
        print("-"*60)
        
        for module, counts in self.activation_counts.items():
            if self.total_samples == 0:
                continue
            
            num_channels = len(counts)
            active_mask = counts > 0
            active_count = active_mask.sum().item()
            frequencies = counts.cpu().numpy() / self.total_samples
            
            layer_name = module.__class__.__name__
            coverage = active_count / num_channels * 100
            avg_freq = frequencies.mean() * 100
            
            print(f"{layer_name:20s} | Channels: {num_channels:4d} | "
                  f"Active: {active_count:4d} ({coverage:5.1f}%) | "
                  f"Avg Freq: {avg_freq:5.2f}%")
        
        print("-"*60 + "\n")
