# LLM Quantization: Error, Outliers, and Hardware Performance

This repository studies how quantization error develops and scales, from synthetic PyTorch tensors to real LLM layers and hardware inference.

## Project Features

* **Baseline Testing:** Custom INT8 and INT4 symmetric quantization built in PyTorch to measure baseline error without relying on quantization frameworks.
* **Outlier Analysis:** Tracks how isolated extreme weights distort global scale factors and increase tensor-wide Mean Squared Error (MSE).
* **Mitigation:** Implements group quantization to limit outlier distortion to smaller regions of a tensor.
* **Real-Model Validation:** Evaluates error propagation across the core linear layers of TinyLlama-1.1B.
* **Hardware Benchmarking:** GGUF inference speed profiling on an AMD Radeon RX 9070 XT (RDNA 4 architecture).

## Underlying Mathematics

The baseline implementation uses symmetric max-based quantization. For a floating-point weight tensor $W$, the scale factor ($s$) is derived from the largest absolute weight:

$$ s = \frac{\max(|W|)}{q_{\max}} $$

where $q_{\max}=127$ for signed INT8 and $q_{\max}=7$ for signed symmetric INT4. Quantized weights ($q$) are mapped via:

$$ q = \mathrm{round}\left(\frac{W}{s}\right) $$

To evaluate precision loss, weights are dequantized ($\hat{W} = q \cdot s$) and evaluated against the original tensor using mean squared error (MSE), while also tracking the rate at which nonzero weights are rounded to zero.

## The Outlier Problem and Mitigation

Symmetric max-based quantization faces significant precision loss when handling extreme outliers. Because signed INT4 offers only 15 discrete levels (-7 to 7), a single high-magnitude weight can increase the scale factor for the entire tensor. Smaller nonzero weights are then mapped onto a coarser grid, causing more of them to round to zero.

To mitigate this, this implementation partitions tensors into local blocks **(group quantization).** Assigning independent scale factors confines an outlier's effect to its group rather than allowing it to determine the scale of the full tensor. To isolate the effect of outliers from bit width alone, controlled synthetic experiments injected individual outliers of increasing magnitude while holding the quantization method and bit width fixed.

The figure below compares INT4 quantization error across different group sizes as outlier magnitude increases. Smaller groups restrict each outlier to a smaller set of weights, while global quantization forces the entire tensor to share a single scale.

![Quantization error vs. group size](results/mse_vs_group_size.png)

## Real-Model Validation

The synthetic results were then tested across the linear weight matrices of TinyLlama-1.1B. Global INT4 quantization showed substantial variation between matrix families. Matrices containing more extreme weights relative to their standard deviation generally experienced greater zeroing and benefitted more from localized scaling.

Across the analyzed matrices, the max-to-standard-deviation ratio was strongly associated with global INT4 zeroing (Spearman ρ = 0.999) and with the MSE improvement obtained using group-64 quantization (ρ = 0.953). The real-model results show the same behavior as the controlled experiments: extreme values can dominate a shared scale and reduce the effective precision available to the rest of the tensor.

To test whether this relationship also appeared in real model weights, each matrix's maximum absolute weight was normalized by its standard deviation and compared with its global INT4 zeroing rate. The resulting relationship is shown below.

![Outlier severity vs. INT4 zeroing](results/max_std_vs_global_int4_zeroing.png)

The same outlier-severity measure was then compared with the reduction in MSE obtained by switching from global INT4 to group-64 quantization. This tests whether matrices with more extreme outliers also benefit more from localized scaling.

![Outlier severity vs. group quantization improvement](results/max_std_vs_grouping_improvement.png)

## Hardware Performance

Finally, GGUF quantizations of TinyLlama-1.1B were benchmarked locally on an AMD Radeon RX 9070 XT (RDNA 4 architecture) to measure how lower precision affected actual inference performance.

Lower-precision models reduced model size and achieved higher token-generation throughput. However, prompt-processing (prefill) performance did not improve monotonically as model size decreased, showing that smaller quantized models do not necessarily produce proportional speed gains across every stage of inference.

The table below reports generation throughput from the 128-token benchmark.

| Format | Model Size (GB) | Generation (tokens/s) | Speedup vs. Q8_0 |
|---|---:|---:|---:|
| Q8_0 | 1.169 | 388.5 | — |
| Q6_K | 0.903 | 464.6 | +19.6% |
| Q5_K_M | 0.781 | 487.8 | +25.6% |
| Q4_K_M | 0.667 | 529.8 | +36.4% |

The same generation results are plotted below to show the throughput trend as quantization becomes more aggressive.

![GPU generation throughput](results/gpu_generation_throughput_by_quantization.png)
