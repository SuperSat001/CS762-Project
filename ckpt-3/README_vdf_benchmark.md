# Class-Group VDF Benchmark for RANDAO

## Quick Start

```bash
# Correctness tests only (< 1 min)
sage vdf_benchmark.sage --test

# Quick benchmark: 256-bit discriminant, T = 2^18 (< 2 min)
sage vdf_benchmark.sage

# Custom parameters
sage vdf_benchmark.sage --bits 512 --T_exp 16

# Full suite across multiple sizes with extrapolation (10-30 min)
sage vdf_benchmark.sage --full
```

## What It Benchmarks

| Component | What's measured |
|---|---|
| Discriminant generation | Time to find a prime p ≡ 3 (mod 4) of target bit-size |
| Hash-to-class-group | Prime-form construction + Tonelli-Shanks + reduction + squaring |
| VDF evaluation | T sequential NUDUPL squarings: g^{2^T} |
| Wesolowski proof | Proof generation (≈ same cost as VDF eval) |
| Verification | π^ℓ · g^r = y check (should be ~1 ms) |
| Seed derivation | Hash VDF output to 256-bit proposer seed |

## Expected Output (approximate)

For 256-bit discriminant, T = 2^18:
- Squarings/sec: ~200,000 - 500,000
- VDF time: 0.5 - 1.3 seconds
- Verification: < 5 ms

For 1024-bit discriminant, T = 2^14:
- Squarings/sec: ~10,000 - 30,000
- VDF time: 0.5 - 1.6 seconds
- Verification: < 20 ms

The `--full` flag extrapolates to 2048-bit production parameters.

## Dependencies

- SageMath >= 9.0 (includes BinaryQF, number theory)
- No external packages needed

## Files Produced

- `vdf_benchmark_results.json`: machine-readable results (with `--full`)
- Console output with timing tables and extrapolations
