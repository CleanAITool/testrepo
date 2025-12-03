Benim için PyTorch tabanlı, tamamen kendi implementasyonumuza sahip bir structured pruning aracı geliştirmeni istiyorum.
Dışarıdan torch-pruning veya benzeri hiçbir pruning kütüphanesi kullanılmayacak.
Sadece PyTorch’ın kendi modülleri ve gerekli temel Python kütüphaneleri kullanılabilir.

Hedef

PyTorch modelleri üzerinde structured pruning (kanal, filtre, layer, neuron group vb.) yapan bir Python projesi geliştir.

Pruning kararları için Weight Magnitude – Activation kuralını uygula.

Bu kuralın özü:

Ağırlık büyüklüğü küçük olan

Aktivasyon katkısı düşük olan kanallar/filtreler prune edilmeye daha uygundur.

Kısaca hem ağırlığın etkisine hem de aktivasyon gücüne bakarak bir önem skoru oluşturulacak.

Girdi / Kullanıcıdan Alınacaklar

Kullanıcı sadece eğitimli PyTorch modelini ve test datasetini verecek.

Bu test seti üzerinden aktivasyonlar çıkarılacak ve Weight Magnitude – Activation kuralı uygulanacak.

Beklenen Çıktı

Pruned edilmiş yeni PyTorch modeli.

Hangi kanallar/filtrelerin prune edildiğini ve önem skorlarını gösteren kısa bir özet raporu.

Genel Beklentiler

Tüm pruning işlemleri bizim tarafımızdan implement edilecek (dependency resolution, katman küçültme, parametre silme, tensor shape güncellemeleri).

Tamamen “mask-based” olmayan, gerçek yapısal pruning (tensor slicing) uygulanmalı.

Kod okunabilir, modüler ve genişletilebilir olacak.

Forward uyumluluğu korunacak, prune sonrası model sorunsuz çalışmalı.

Bu bilgileri kullanarak, baştan sona çalışan bir CleanAI Structured Pruning aracı geliştir.

========================================================================================

## Pipeline Snapshot

- Capture a trace of the model with `torch.autograd` to recover how every tensor flows between modules and tensor ops.
- Convert that trace into a dependency graph whose edges describe "if this tensor loses channels, those other tensors must drop the equivalent entries".
- Keep explicit index mappings so that composite ops such as concatenation, splitting, reshaping, unbinding, expansion, or slicing know how to translate channel indices.
- Form pruning groups: each group starts from a root tensor slice and bundles every coupled tensor/parameter that must shrink together.
- Score the channels in each group with an importance estimator (e.g., magnitude, Taylor, Hessian, random, LayerNorm-aware norms, LAMP-style scaling) and remove the least useful ones by rewriting PyTorch modules in-place.
- Log the operations so the exact pruning history can be replayed or serialized.

## 1. Graph Capture via Autograd Hooks

- The library registers forward hooks on every prunable `nn.Module` subclass (convs, linears, norms, embeddings, attention blocks, RNN cells, plus simple tensor ops modeled as lightweight `nn.Module` placeholders).
- Each hook records the `grad_fn` that produces the module output. After a single forward pass on example inputs (optionally wrapped by a custom `forward_fn` or `output_transform`), `grad_fn` objects can be walked backwards to reconstruct the computation graph.
- Unknown ops default to "elementwise" nodes so they do not interfere with channel counts. Special placeholders model concat, split, reshape/view, unbind, expand, slice, and the final model outputs.
- Parameters that do not belong to a standard module ("unwrapped" tensors) are detected by subtracting the parameters seen inside supported modules from the global `model.named_parameters()` list. Their last non-singleton dimension becomes the default pruning dimension.

## 2. Building the Dependency Graph

- Each recorded module (or synthetic op) is stored as a node containing pointers to its input/output nodes, the originating module instance, and bookkeeping such as pruning dimension and whether index mapping is allowed.
- For every tensor connection A→B, the graph adds two dependency rules: pruning A's outputs forces the matching inputs of B to shrink, and pruning B's inputs must respect upstream outputs. Intra-layer coupling (e.g., between weights and biases) is handled by simply using the same pruning callback for both directions.
- The system supports custom pruning callbacks: users can map a PyTorch module class or instance to tailored logic, and all descendants of that module are excluded from tracing so the custom rule stays in control.

## 3. Shape and Index Tracking

- To figure out how tensor indices propagate through structural transforms, the code inspects the saved metadata on `grad_fn` objects (e.g., `_saved_split_sizes`, `_saved_dim`, `_saved_start/_saved_end/_saved_step` for slices, `_saved_self_sizes` for views, `_saved_self_sym_sizes` for expands). When those hints are unavailable, it falls back to recursive channel inference by walking neighbors in the graph.
- Each dependency carries up to two mapping callables that convert "root" channel indices (defined at the group origin) into the coordinate system of the downstream module. Dedicated mapping objects exist for concat, split, flatten/reshape, unbind, expand, and slice so that even complex cascades like split→concat→reshape are handled deterministically.
- A hybrid index object keeps both the current coordinate and the original root index. This is crucial when multiple fan-outs/fan-ins need to merge pruning decisions or when concatenations change ordering.

## 4. Forming Pruning Groups

- Requesting a pruning operation (e.g., removing specific output channels of a convolution) spawns a fresh group. The root dependency is added, and a depth-first walk follows every dependency that is triggered by the same pruning callback, applying index mappings on the fly.
- Dependencies are merged so that identical module/function pairs share one entry with a unified index list. Layers listed as "ignored" or with insufficient remaining channels are skipped to prevent over-pruning.
- Groups may include tensors that are not attached to modules, such as standalone `nn.Parameter` objects. These are reassigned into the owning module via attribute writes after pruning so `state_dict` stays consistent.

## 5. Executing Structural Changes

- Each supported layer type has a pruning routine that keeps the desired indices with `torch.index_select`, rewrites `nn.Parameter` objects (and their gradients if present), and updates metadata like `out_channels`, `in_channels`, `groups`, `normalized_shape`, or embedding/table dimensions.
- Convolutions handle regular, transposed, depthwise, and grouped variants. Normalization layers adjust running statistics and affine parameters. Dense layers manage linear and attention projections. Recurrent layers currently expect single-layer configurations.
- For modules with grouped behavior (depthwise/group convolutions, group norms, multi-head attention, GQA-style expansions), index mappings and pruning functions respect the grouping factor so structural validity is preserved.
- During pruning the graph stores a history entry containing the module path, whether the operation acted on inputs or outputs, and the pruned indices. This enables replay, serialization, or rollback.

## 6. Channel Importance Estimation

- Importance calculators operate on whole groups so that correlated tensors share a single score vector. They iterate over each dependency, gather the relevant weight slices, flatten across spatial dimensions, and compute statistics such as L1/L2 norms, BN scaling magnitudes, geometric-median distances (`torch.cdist`), Taylor first-order scores (`weight * grad`), or Hessian approximations (via batched Hessian-vector products).
- Scores from multiple tensors are reduced onto the root indices using scatter-add, scatter-max, or custom reductions, optionally normalized (mean, max, gaussian z-score, sentinel quantile, or LAMP cumulative scaling).
- Because groups can include non-standard ops, estimators simply skip entries without usable tensors, returning `None` when no parameter contributed; callers must then decide whether to postpone pruning for that group.

## 7. Persistence and Replay

- After each pruning action, the library stores the tuple `(module_path, is_output_pruning, indices)` inside an internal history list.
- Serialization helpers expose `state_dict`-like APIs to dump the current pruning state along with model weights. Loading the history replays every recorded pruning action on a fresh model instance before applying weights, guaranteeing structural alignment without manual bookkeeping.

## 8. PyTorch Ingredients Worth Reusing

- `nn.Module.register_forward_hook` for tracing which module created each tensor.
- `torch.autograd.Function` metadata (`grad_fn.next_functions`, `_saved_*` fields) for shape/offset inference.
- Lightweight `nn.Module` subclasses to represent pure tensor ops (concat, split, reshape, unbind, expand, slice, output) so the dependency graph treats them uniformly with learned layers.
- `torch.index_select` and parameter reassignment to rewrite weights/biases while preserving gradients.
- Scatter/gather operations (`scatter_add_`, `scatter_`, `index_select`) to merge importance scores from multiple tensors back to root indices.
- Persistent pruning logs stored alongside checkpoints so pruning can be replayed deterministically.

## 9. Re-Implementing the Approach

1. Use a single forward pass with hooks to record `grad_fn` pointers for every tensor-producing module.
2. Walk `grad_fn.next_functions` backward to recreate the mixed graph of modules and tensor ops.
3. For each connection, register how pruning should propagate (output→input and input→output) and stash mapping callbacks for nontrivial ops.
4. Provide pruning routines for each layer type you care about, all of which accept a tensor plus index list and rewrite parameters/metadata accordingly.
5. When pruning is requested, build a group rooted at the chosen module and propagate through the dependency graph until every coupled tensor is covered.
6. Compute importance per group, pick the lowest-scoring indices, and call the pruning routines. Record everything for future replay.
