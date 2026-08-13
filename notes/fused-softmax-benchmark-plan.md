# Fused Softmax Benchmark Plan

## Target

Benchmark fused row-wise softmax on fp32 tensors and compare against `torch.softmax`.

## Shape

- M = 4096
- N = 4096
- dtype = fp32

## Byte model

For X in R^(M x N), fp32 means 4 bytes per element.

Naive softmax estimated memory traffic:

```text

read X, write shifted

read shifted, write exp

read exp, write Y
```



## Metrics to report

| Metric | Meaning |

|---|---|

| median time | Typical runtime across repeated benchmark iterations |

| p90 or standard deviation | Timing stability / noise |

| estimated bytes moved | Bytes predicted by the memory-traffic model |

| effective bandwidth | estimated bytes / median time |

| speedup vs torch | torch median time / triton median time |

| correctness max error vs torch | max absolute difference between Triton output and torch output |