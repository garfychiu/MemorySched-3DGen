<div align="center">

# A Unified Training-Free Memory Scheduling Framework for Single-Image 3D Generation

</div>

## Abstract

Single-image 3D mesh generation is in high demand for applications such as AR/VR and digital twins. However, existing two-stage models—comprising multi-view diffusion and mesh reconstruction—incur enormous memory footprints during inference, far exceeding the capacity of mainstream consumer-grade GPUs, which severely hinders their deployment on edge devices. Existing training-free optimization methods either rely on retraining, neglect module-wise sensitivity differences, or lack an analyzable memory cost model, making it difficult to balance compression ratio and reconstruction fidelity. To address these issues, we propose a unified training-free memory-lightweight framework that modifies only the forward computation logic without altering any pretrained weights. First, we perform module-level numerical sensitivity stratification and apply FP8 storage-computation separated quantization only to low-sensitivity layers, reducing the peak memory of the multi-view diffusion stage. For the mesh reconstruction stage, our core innovation introduces a dimensionless chunking balance factor λ, which decouples the original O(G³) memory complexity of grid decoding into O(λG³). Based on the mathematical equivalence between chunk-wise and full-batch decoding, we establish a resolution-unified memory cost model and derive a closed-form solution for the two-stage memory equilibrium point λ₀. This critical point strictly defines the benefit boundary of chunking, ensuring that the global peak memory automatically converges to the hardware-reachable minimum without redundant iterations, thereby achieving Pareto-optimal memory-speed co-scheduling. Taking InstantMesh as an example, our framework adapts to full-range GPUs from 6 GB to 24 GB according to the λ₀ criterion. At 128 resolution, it reduces the peak memory from 19.63 GB to 5.86 GB (-70.1%), while maintaining an F-Score of 0.902 (near-lossless), enabling high-precision models to run stably on 8 GB or even 6 GB consumer cards. Furthermore, when transferred to two distinct architectures, the peak memory of Fancy123 drops from 19.63 GB to 10.15 GB (-48.3%), and that of Unique3D from 22.17 GB to 12.80 GB (-42.2%), with negligible degradation in geometric accuracy and texture details, fully demonstrating the generality of our theory. This work provides a new inference-side scheduling paradigm with closed-form optimal interpretability for 3D generation under resource-constrained scenarios.

## Highlights

- **Training-free**: modifies only the forward computation logic, without altering any pretrained weights.
- **Module-level sensitivity stratification**: applies FP8 storage-computation separated quantization only to low-sensitivity layers, reducing the peak memory of the multi-view diffusion stage.
- **Chunking balance factor λ**: decouples the original O(G³) grid-decoding memory into O(λG³), with a closed-form two-stage memory equilibrium point λ₀.
- **Pareto-optimal memory-speed co-scheduling**: global peak memory converges to the hardware-reachable minimum without redundant iterations.
- **InstantMesh**: peak memory 19.63 GB → 5.86 GB (-70.1%) at 128 resolution, F-Score 0.902 (near-lossless), runs on 8 GB or even 6 GB consumer GPUs.
- **Generalizes across architectures**: Fancy123 19.63 GB → 10.15 GB (-48.3%), Unique3D 22.17 GB → 12.80 GB (-42.2%), with negligible degradation in geometry and texture.
