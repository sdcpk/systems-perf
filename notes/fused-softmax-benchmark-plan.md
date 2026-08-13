# Fused Softmax Benchmark Plan

## Target

Benchmark fused row-wise softmax on fp32 tensors and compare against `torch.softmax`.

## Shape

- M = 4096
- N = 4096
- dtype = fp32

## Byte model

For `X` in `R^(M x N)`, fp32 means 4 bytes per element.

Naive softmax estimated memory traffic:

```text
read X, write shifted
read shifted, write exp
read exp, write Y
```

So:

```text
Naive bytes = 24MN
```

Fused softmax estimated memory traffic:

```text
read X
write Y
```

So:

```text
Fused bytes = 8MN
```

For `M = N = 4096`:

```text
MN = 2^24 elements
Naive bytes = 24 * 2^24 bytes = 384 MiB
Fused bytes = 8 * 2^24 bytes = 128 MiB
```

RTX 4090 memory bandwidth:

```text
1008 GB/s
```

Fused lower bound:

```text
128 MiB / 1008 GB/s ≈ 0.13 ms
```

## Metrics to report

| Metric | Meaning |
|---|---|
| median time | Typical runtime across repeated benchmark iterations |
| p90 or standard deviation | Timing stability / noise |
| estimated bytes moved |ytes predicted by the memory-traffic model |
| effective bandwidth | estimated bytes / median time |
| speedup vs torch | torch median time / triton median time |
| correctness max error vs torch | max absolute difference between Triton output and torch output |

## Success condition

The Triton fused softmax runs correctly and reports effective bandwidth.

It does not need to beat `torch.softmax` yet.
