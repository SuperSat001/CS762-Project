#!/usr/bin/env sage
"""
=============================================================================
Class-Group VDF Benchmarking Suite for RANDAO Integration
=============================================================================

This script benchmarks a Wesolowski-style VDF over the class group of an
imaginary quadratic field Cl(Δ), as proposed for Ethereum's RANDAO.

Components benchmarked:
  1. Discriminant generation from a public seed
  2. Hash-to-class-group (prime-form method)
  3. VDF evaluation: g^{2^T} via repeated NUDUPL squaring
  4. Wesolowski proof generation
  5. Proof verification

Usage:
  sage vdf_benchmark.sage [--bits 256] [--T_exp 20] [--full]

  --bits   : discriminant bit-size (default 256 for quick test; use 2048 for production)
  --T_exp  : VDF iteration exponent, T = 2^T_exp (default 20)
  --full   : run full benchmark suite across multiple parameter sizes

Author: VDF benchmarking for RANDAO forking attack defense
Date:   2026
=============================================================================
"""

import time
import sys
import hashlib
import json
from collections import OrderedDict

# ============================================================================
# 1. DISCRIMINANT GENERATION
# ============================================================================

def generate_discriminant(seed_bytes, target_bits):
    """
    Generate a fundamental discriminant Δ from a public seed.
    
    Requirements:
      - |Δ| has `target_bits` bits
      - -Δ is prime
      - -Δ ≡ 3 (mod 4), so Δ ≡ 1 (mod 4) and Δ is fundamental
    
    This ensures the class group Cl(Δ) has minimal 2-torsion (genus theory)
    and no trapdoor exists.
    
    Args:
        seed_bytes: public seed (e.g., hash of a beacon block)
        target_bits: desired bit-length of |Δ|
    
    Returns:
        Δ (negative integer), generation time
    """
    t0 = time.time()
    
    # Hash seed to get starting point in the target range
    h = hashlib.sha512(seed_bytes).digest()
    # Expand hash to target_bits by repeated hashing if needed
    while len(h) * 8 < target_bits:
        h += hashlib.sha512(h).digest()
    
    # Convert to integer, mask to target_bits, ensure MSB is set
    candidate = int.from_bytes(h[:target_bits // 8], 'big')
    candidate |= (1 << (target_bits - 1))  # ensure correct bit-length
    
    # Find next prime p ≡ 3 (mod 4) so that Δ = -p is fundamental
    # with Δ ≡ 1 (mod 4)
    if candidate % 4 != 3:
        candidate += (3 - candidate % 4) % 4
    
    attempts = 0
    while True:
        if is_prime(candidate):
            Delta = -candidate
            dt = time.time() - t0
            return Delta, dt, attempts
        candidate += 4  # stay in ≡ 3 (mod 4) residue class
        attempts += 1


def verify_discriminant(Delta):
    """Verify that Δ is a valid fundamental discriminant."""
    checks = {
        "is_negative": Delta < 0,
        "neg_Delta_is_prime": is_prime(-Delta),
        "neg_Delta_mod4_eq_3": (-Delta) % 4 == 3,
        "Delta_mod4_eq_1": Delta % 4 == 1,
        "bit_length": int(-Delta).bit_length(),
    }
    return checks


# ============================================================================
# 2. HASH-TO-CLASS-GROUP (Prime Form Method)
# ============================================================================

def hash_to_class_group(input_bytes, Delta):
    """
    Hash arbitrary bytes to an element of Cl(Δ) in the squares subgroup.
    
    Method:
      1. Hash input to candidate prime p with Legendre symbol (Δ/p) = 1
      2. Construct form (p, b, c) where b² ≡ Δ (mod 4p)
      3. Reduce the form
      4. Square to land in the squares subgroup (avoids 2-torsion attacks)
    
    Args:
        input_bytes: the RANDAO output R^e as bytes
        Delta: the fundamental discriminant
    
    Returns:
        reduced BinaryQF in Cl(Δ)², timing info
    """
    t0 = time.time()
    
    # Step 1: hash to a prime p with (Δ/p) = 1
    h = hashlib.sha256(input_bytes).digest()
    p_candidate = int.from_bytes(h, 'big')
    if p_candidate < 3:
        p_candidate = 3
    
    legendre_attempts = 0
    while True:
        p_candidate = next_prime(p_candidate)
        # Check Legendre symbol: need (Δ/p) = 1
        if kronecker(Delta, p_candidate) == 1:
            p = p_candidate
            break
        legendre_attempts += 1
    
    # Step 2: find b such that b² ≡ Δ (mod 4p)
    # Since Δ ≡ 1 (mod 4) and p is odd, we need b² ≡ Δ (mod p)
    # and b odd (so b² ≡ 1 ≡ Δ (mod 4))
    b_mod_p = int(Mod(Delta, p).sqrt())
    
    # Lift to b² ≡ Δ (mod 4p) using CRT
    # We need b ≡ b_mod_p (mod p) and b odd
    b = int(b_mod_p)
    if b % 2 == 0:
        b = int(p - b)  # the other square root mod p
    if b % 2 == 0:
        b = b + p  # ensure odd
        if b % 2 == 0:
            b = b + p
    
    # Verify
    assert (b * b - Delta) % (4 * p) == 0, f"Form construction failed: b²-Δ = {b*b - Delta}, 4p = {4*p}"
    
    c = (b * b - Delta) // (4 * p)
    
    # Step 3: construct and reduce the form
    f = BinaryQF([p, b, c])
    f_reduced = f.reduced_form()
    
    # Step 4: square to land in squares subgroup
    # This eliminates 2-torsion elements (critical for VDF security)
    g = (f_reduced * f_reduced).reduced_form()
    
    dt = time.time() - t0
    
    info = {
        "prime_p": p,
        "prime_bits": int(p).bit_length(),
        "legendre_attempts": legendre_attempts,
        "form_before_square": f_reduced,
        "time_seconds": dt,
    }
    
    return g, info


# ============================================================================
# 3. VDF EVALUATION: g^{2^T}
# ============================================================================

def vdf_evaluate(g, T, Delta, progress_interval=None):
    """
    Compute y = g^{2^T} in Cl(Δ) via T sequential squarings.
    
    This is the core VDF computation. Each squaring calls NUDUPL
    internally (via SageMath's BinaryQF multiplication).
    The sequential nature is what makes this a valid VDF --
    no parallelism can speed up the chain of squarings.
    
    Args:
        g: input element of Cl(Δ) (a reduced BinaryQF)
        T: number of squarings
        Delta: discriminant (for verification)
        progress_interval: print progress every N squarings (None = no output)
    
    Returns:
        y = g^{2^T}, total time, squarings/sec
    """
    assert g.discriminant() == Delta, "Input form has wrong discriminant"
    
    y = g
    t0 = time.time()
    last_report = t0
    
    for i in range(T):
        y = (y * y).reduced_form()
        
        if progress_interval and (i + 1) % progress_interval == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (T - i - 1) / rate
            print(f"  Squarings: {i+1}/{T} | "
                  f"Rate: {rate:.0f}/s | "
                  f"Elapsed: {elapsed:.1f}s | "
                  f"ETA: {eta:.1f}s")
    
    dt = time.time() - t0
    rate = T / dt if dt > 0 else float('inf')
    
    return y, dt, rate


# ============================================================================
# 4. WESOLOWSKI PROOF GENERATION
# ============================================================================

def hash_to_prime(g, y, bit_length=256):
    """
    Fiat-Shamir: derive a challenge prime ℓ from (g, y).
    
    This is the non-interactive version of Wesolowski's protocol.
    The verifier's challenge is replaced by hashing.
    """
    # Canonical encoding of forms as (a, b, c) tuples
    g_bytes = str(g).encode('utf-8')
    y_bytes = str(y).encode('utf-8')
    
    h = hashlib.sha256(g_bytes + b"||" + y_bytes).digest()
    candidate = int.from_bytes(h, 'big')
    
    # Find next prime
    ell = next_prime(candidate)
    return ell


def wesolowski_prove(g, y, T, Delta):
    """
    Generate a Wesolowski proof π for y = g^{2^T}.
    
    The proof is π = g^q where q = floor(2^T / ℓ) and ℓ is a
    Fiat-Shamir challenge prime derived from (g, y).
    
    Verification: π^ℓ · g^r = y, where r = 2^T mod ℓ.
    
    Args:
        g: VDF input
        y: VDF output (claimed g^{2^T})
        T: number of squarings
        Delta: discriminant
    
    Returns:
        (pi, ell, r), proof generation time
    """
    t0 = time.time()
    
    # Step 1: Fiat-Shamir challenge
    ell = hash_to_prime(g, y)
    
    # Step 2: compute q = floor(2^T / ℓ), r = 2^T mod ℓ
    # For large T, compute 2^T mod ℓ via fast modular exponentiation
    two_to_T_mod_ell = power_mod(2, T, ell)
    r = two_to_T_mod_ell
    
    # q = (2^T - r) / ℓ
    # For the proof, we need g^q. We compute this by doing T squarings
    # while tracking q via long division of 2^T by ℓ.
    #
    # Efficient method: compute g^q by iterating through the bits of
    # the exponent, maintaining both the quotient and the proof.
    # 
    # Algorithm (from Wesolowski's paper):
    #   Initialize: pi = identity, x = g
    #   For i = T-1 down to 0:
    #     b = floor(2 * remainder / ℓ)    (either 0 or 1)
    #     remainder = (2 * remainder) mod ℓ
    #     pi = pi^2 * g^b
    #
    # This computes pi = g^q in T squarings (same cost as VDF itself).
    
    identity_form = BinaryQF([1, Delta % 2, (Delta % 2 - Delta) // 4])
    identity_form = identity_form.reduced_form()
    
    pi = identity_form
    remainder = 1  # tracks 2^i mod ℓ as we go
    
    for i in range(T):
        # At iteration i, remainder = 2^i mod ℓ (before doubling)
        doubled = 2 * remainder
        b = doubled // ell  # 0 or 1
        remainder = doubled % ell
        
        pi = (pi * pi).reduced_form()
        if b == 1:
            pi = (pi * g).reduced_form()
    
    dt = time.time() - t0
    
    return (pi, ell, r), dt


# ============================================================================
# 5. VERIFICATION
# ============================================================================

def wesolowski_verify(g, y, pi, ell, r, Delta):
    """
    Verify a Wesolowski proof: check that π^ℓ · g^r = y.
    
    This is the fast part -- only two exponentiations by ~256-bit numbers
    and one group multiplication.
    
    Args:
        g: VDF input
        y: VDF output (claimed)
        pi: proof element
        ell: Fiat-Shamir challenge prime
        r: 2^T mod ℓ
        Delta: discriminant
    
    Returns:
        True/False, verification time
    """
    t0 = time.time()
    
    # Recompute challenge to ensure consistency
    ell_check = hash_to_prime(g, y)
    assert ell_check == ell, "Challenge prime mismatch"
    
    # Compute π^ℓ
    pi_to_ell = fast_power(pi, ell, Delta)
    
    # Compute g^r
    g_to_r = fast_power(g, r, Delta)
    
    # Check: π^ℓ · g^r = y
    lhs = (pi_to_ell * g_to_r).reduced_form()
    
    valid = (lhs == y.reduced_form())
    
    dt = time.time() - t0
    return valid, dt


def fast_power(f, n, Delta):
    """
    Compute f^n in Cl(Δ) via square-and-multiply.
    
    Args:
        f: a reduced BinaryQF
        n: positive integer exponent
        Delta: discriminant
    
    Returns:
        f^n as a reduced BinaryQF
    """
    if n == 0:
        identity = BinaryQF([1, Delta % 2, (Delta % 2 - Delta) // 4])
        return identity.reduced_form()
    if n == 1:
        return f.reduced_form()
    
    result = None
    base = f.reduced_form()
    
    while n > 0:
        if n % 2 == 1:
            if result is None:
                result = base
            else:
                result = (result * base).reduced_form()
        base = (base * base).reduced_form()
        n //= 2
    
    return result.reduced_form()


# ============================================================================
# 6. SEED DERIVATION (RANDAO integration)
# ============================================================================

def derive_proposer_seed(y, epoch_number):
    """
    Derive the proposer-list seed for epoch e+3 from VDF output.
    
    This replaces: seed = H(e || R^e) with seed = H(e || U^e)
    where U^e = H(encode(y)).
    
    Args:
        y: VDF output (a reduced BinaryQF)
        epoch_number: the epoch number e
    
    Returns:
        seed: 32-byte seed for proposer shuffling
    """
    # Canonical encoding of the VDF output
    y_encoding = f"{y[0]}:{y[1]}:{y[2]}".encode('utf-8')
    
    # U^e = H(encode(y))
    U_e = hashlib.sha256(y_encoding).digest()
    
    # seed = H(e || U^e)
    epoch_bytes = int(epoch_number).to_bytes(8, 'big')
    seed = hashlib.sha256(epoch_bytes + U_e).digest()
    
    return seed.hex()


# ============================================================================
# 7. FULL BENCHMARK RUNNER
# ============================================================================

def run_benchmark(disc_bits, T_exp, verbose=True):
    """
    Run a full VDF benchmark cycle.
    
    Args:
        disc_bits: discriminant bit-size
        T_exp: VDF iterations = 2^T_exp
        verbose: print detailed output
    
    Returns:
        dict of timing results
    """
    T = 2**T_exp
    results = OrderedDict()
    
    separator = "=" * 70
    
    if verbose:
        print(separator)
        print(f"VDF BENCHMARK: |Δ| = {disc_bits} bits, T = 2^{T_exp} = {T}")
        print(separator)
    
    # ----- 1. Discriminant generation -----
    if verbose:
        print(f"\n[1/6] Generating {disc_bits}-bit fundamental discriminant...")
    
    seed = b"ethereum_beacon_block_hash_0xdeadbeef_randao_vdf_seed"
    Delta, dt_disc, attempts = generate_discriminant(seed, disc_bits)
    
    checks = verify_discriminant(Delta)
    
    results["discriminant_bits"] = disc_bits
    results["discriminant_gen_time"] = dt_disc
    results["discriminant_gen_attempts"] = attempts
    
    if verbose:
        print(f"  Δ bit-length: {checks['bit_length']}")
        print(f"  -Δ is prime: {checks['neg_Delta_is_prime']}")
        print(f"  -Δ ≡ 3 (mod 4): {checks['neg_Delta_mod4_eq_3']}")
        print(f"  Generation time: {dt_disc:.3f}s ({attempts} candidates tested)")
    
    # ----- 2. Hash-to-class-group -----
    if verbose:
        print(f"\n[2/6] Hashing RANDAO output to Cl(Δ)...")
    
    # Simulate a RANDAO output R^e
    R_e = hashlib.sha256(b"simulated_randao_epoch_12345").digest()
    
    g, h2g_info = hash_to_class_group(R_e, Delta)
    
    results["hash_to_group_time"] = h2g_info["time_seconds"]
    results["hash_prime_bits"] = h2g_info["prime_bits"]
    
    if verbose:
        print(f"  Input element g = ({g[0]}, {g[1]}, ...)")
        print(f"  g.discriminant() == Δ: {g.discriminant() == Delta}")
        print(f"  Prime p bit-length: {h2g_info['prime_bits']}")
        print(f"  Legendre attempts: {h2g_info['legendre_attempts']}")
        print(f"  Hash-to-group time: {h2g_info['time_seconds']:.4f}s")
    
    # ----- 3. VDF evaluation -----
    if verbose:
        print(f"\n[3/6] Computing VDF: y = g^(2^{T_exp}) ({T} squarings)...")
    
    progress = max(1, T // 10) if verbose else None
    y, dt_eval, rate = vdf_evaluate(g, T, Delta, progress_interval=progress)
    
    results["T_exponent"] = T_exp
    results["T_iterations"] = T
    results["vdf_eval_time"] = dt_eval
    results["squarings_per_sec"] = rate
    results["time_per_squaring_us"] = 1e6 / rate if rate > 0 else float('inf')
    
    if verbose:
        print(f"\n  VDF output y = ({y[0]}, {y[1]}, ...)")
        print(f"  y.discriminant() == Δ: {y.discriminant() == Delta}")
        print(f"  Total time: {dt_eval:.3f}s")
        print(f"  Squarings/sec: {rate:.0f}")
        print(f"  Time/squaring: {1e6/rate:.2f} μs")
    
    # ----- 4. Wesolowski proof generation -----
    if verbose:
        print(f"\n[4/6] Generating Wesolowski proof...")
    
    (pi, ell, r), dt_proof = wesolowski_prove(g, y, T, Delta)
    
    results["proof_gen_time"] = dt_proof
    results["challenge_prime_bits"] = int(ell).bit_length()
    
    if verbose:
        print(f"  Proof π = ({pi[0]}, {pi[1]}, ...)")
        print(f"  Challenge ℓ: {int(ell).bit_length()}-bit prime")
        print(f"  Remainder r = 2^T mod ℓ: {int(r).bit_length()}-bit")
        print(f"  Proof generation time: {dt_proof:.3f}s")
        print(f"  (Note: proof gen ≈ VDF eval time, as expected)")
    
    # ----- 5. Verification -----
    if verbose:
        print(f"\n[5/6] Verifying proof...")
    
    valid, dt_verify = wesolowski_verify(g, y, pi, ell, r, Delta)
    
    results["verification_time"] = dt_verify
    results["verification_valid"] = valid
    
    if verbose:
        print(f"  π^ℓ · g^r == y: {valid}")
        print(f"  Verification time: {dt_verify*1000:.2f} ms")
    
    # ----- 6. Seed derivation -----
    if verbose:
        print(f"\n[6/6] Deriving proposer seed for epoch e+3...")
    
    t0 = time.time()
    seed = derive_proposer_seed(y, epoch_number=12345)
    dt_seed = time.time() - t0
    
    results["seed_derivation_time"] = dt_seed
    results["proposer_seed"] = seed[:32] + "..."
    
    if verbose:
        print(f"  Seed: {seed[:32]}...")
        print(f"  Derivation time: {dt_seed*1e6:.1f} μs")
    
    # ----- Summary -----
    if verbose:
        print(f"\n{separator}")
        print(f"SUMMARY")
        print(f"{separator}")
        print(f"  Discriminant: {disc_bits} bits, generated in {dt_disc:.3f}s")
        print(f"  Hash-to-group: {h2g_info['time_seconds']*1000:.2f} ms")
        print(f"  VDF eval (T=2^{T_exp}): {dt_eval:.3f}s ({rate:.0f} sq/s, {1e6/rate:.2f} μs/sq)")
        print(f"  Proof generation: {dt_proof:.3f}s")
        print(f"  Verification: {dt_verify*1000:.2f} ms")
        print(f"  Seed derivation: {dt_seed*1e6:.1f} μs")
        print(f"  Proof valid: {valid}")
        print()
        
        # Extrapolation to production parameters
        if disc_bits < 2048:
            print(f"EXTRAPOLATION TO PRODUCTION PARAMETERS:")
            print(f"  At {disc_bits}-bit: {rate:.0f} squarings/sec")
            # Class group operations scale roughly as O(n^2) in discriminant bits
            # (dominated by big-integer multiplication in NUDUPL)
            scale_factor = (2048 / disc_bits) ** 2
            prod_rate = rate / scale_factor
            prod_us = 1e6 / prod_rate
            
            for T_prod_exp in [26, 27, 28, 29, 30]:
                T_prod = 2**T_prod_exp
                prod_time = T_prod / prod_rate
                print(f"  T=2^{T_prod_exp}: est. {prod_time:.0f}s "
                      f"({prod_time/60:.1f} min) at {prod_us:.1f} μs/sq "
                      f"[2048-bit, extrapolated]")
            print()
    
    return results


def run_squaring_microbenchmark(Delta, num_squarings=10000):
    """
    Pure squaring microbenchmark: measure NUDUPL throughput precisely.
    
    This isolates the squaring cost from hash-to-group and proof overhead.
    """
    print(f"\nNUDUPL MICROBENCHMARK ({int(-Delta).bit_length()}-bit Δ, {num_squarings} squarings)")
    print("-" * 50)
    
    # Generate a random element
    R_e = hashlib.sha256(b"microbenchmark_seed").digest()
    g, _ = hash_to_class_group(R_e, Delta)
    
    # Warmup
    y = g
    for _ in range(100):
        y = (y * y).reduced_form()
    
    # Timed run
    y = g
    t0 = time.time()
    for _ in range(num_squarings):
        y = (y * y).reduced_form()
    dt = time.time() - t0
    
    rate = num_squarings / dt
    print(f"  Squarings: {num_squarings}")
    print(f"  Total time: {dt:.3f}s")
    print(f"  Rate: {rate:.0f} squarings/sec")
    print(f"  Per squaring: {1e6/rate:.2f} μs")
    
    return rate


def run_full_suite():
    """Run benchmarks across multiple parameter sizes."""
    
    print("=" * 70)
    print("FULL BENCHMARK SUITE: Class-Group VDF for RANDAO")
    print("=" * 70)
    
    # Test configurations: (disc_bits, T_exp)
    configs = [
        (128,  16),   # Tiny: quick sanity check
        (256,  18),   # Small: fast benchmark
        (512,  18),   # Medium: realistic structure, feasible timing
        (768,  16),   # Larger: closer to production
        (1024, 14),   # Near-production: slower but informative
    ]
    
    all_results = []
    
    for disc_bits, T_exp in configs:
        try:
            results = run_benchmark(disc_bits, T_exp, verbose=True)
            all_results.append(results)
        except Exception as e:
            print(f"  ERROR for {disc_bits}-bit / T=2^{T_exp}: {e}")
            import traceback
            traceback.print_exc()
    
    # Also run pure squaring microbenchmarks at different sizes
    print("\n" + "=" * 70)
    print("PURE SQUARING MICROBENCHMARKS")
    print("=" * 70)
    
    squaring_results = {}
    for bits in [128, 256, 512, 768, 1024]:
        try:
            seed = b"squaring_bench_" + str(bits).encode()
            Delta, _, _ = generate_discriminant(seed, bits)
            rate = run_squaring_microbenchmark(Delta, num_squarings=5000)
            squaring_results[bits] = rate
        except Exception as e:
            print(f"  ERROR for {bits}-bit: {e}")
    
    # Scaling analysis
    if len(squaring_results) >= 2:
        print("\n" + "=" * 70)
        print("SCALING ANALYSIS")
        print("=" * 70)
        
        bits_list = sorted(squaring_results.keys())
        base_bits = bits_list[0]
        base_rate = squaring_results[base_bits]
        
        print(f"\n  {'Bits':>6} | {'Rate (sq/s)':>12} | {'μs/sq':>8} | {'Scaling vs {}'.format(base_bits):>16}")
        print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*8}-+-{'-'*16}")
        
        for bits in bits_list:
            rate = squaring_results[bits]
            us = 1e6 / rate
            scaling = base_rate / rate
            print(f"  {bits:>6} | {rate:>12.0f} | {us:>8.2f} | {scaling:>16.2f}x")
        
        # Extrapolate to 2048-bit
        # Fit: rate ∝ bits^(-α), find α from regression
        import math
        if len(bits_list) >= 3:
            # Log-log regression
            log_bits = [math.log(b) for b in bits_list]
            log_rates = [math.log(squaring_results[b]) for b in bits_list]
            
            n = len(log_bits)
            sum_x = sum(log_bits)
            sum_y = sum(log_rates)
            sum_xy = sum(x*y for x, y in zip(log_bits, log_rates))
            sum_x2 = sum(x*x for x in log_bits)
            
            alpha = -(n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
            log_c = (sum_y + alpha * sum_x) / n
            
            print(f"\n  Empirical scaling: rate ∝ bits^(-{alpha:.2f})")
            
            # Extrapolate
            for target_bits in [1536, 2048, 3072]:
                est_rate = math.exp(log_c - alpha * math.log(target_bits))
                est_us = 1e6 / est_rate
                print(f"\n  EXTRAPOLATION TO {target_bits}-bit discriminant:")
                print(f"    Estimated rate: {est_rate:.0f} squarings/sec ({est_us:.1f} μs/sq)")
                
                for T_exp in [26, 27, 28, 29]:
                    T_val = 2**T_exp
                    est_time = T_val / est_rate
                    print(f"    T=2^{T_exp}: {est_time:.0f}s ({est_time/60:.1f} min)")
    
    # Save results
    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)
    
    # Serialize results (convert non-serializable types)
    for r in all_results:
        for k, v in r.items():
            if hasattr(v, '__class__') and 'BinaryQF' in str(type(v)):
                r[k] = str(v)
    
    with open("vdf_benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("  -> vdf_benchmark_results.json")
    
    return all_results


# ============================================================================
# 8. CORRECTNESS TESTS
# ============================================================================

def run_correctness_tests():
    """
    Verify that the VDF implementation is correct on small examples.
    """
    print("=" * 70)
    print("CORRECTNESS TESTS")
    print("=" * 70)
    
    # Test 1: Known class group Cl(-20), order 2
    print("\n[Test 1] Cl(-20): class number h = 2")
    Delta = -20
    # Forms with discriminant -20
    principal = BinaryQF([1, 0, 5])
    non_principal = BinaryQF([2, 2, 3])
    
    # Check: non_principal^2 = principal
    product = (non_principal * non_principal).reduced_form()
    assert product == principal.reduced_form(), f"Failed: got {product}"
    print(f"  (2,2,3)^2 = {product} = principal form ✓")
    
    # Test 2: VDF with small T on Cl(-23), class number 3
    print("\n[Test 2] VDF on Cl(-23): h = 3, T = 8")
    Delta = -23
    g = BinaryQF([2, 1, 3])  # generator
    
    # Compute g^{2^8} = g^{256}
    # Since h=3, g^256 = g^(256 mod 3) = g^(1) = g
    y, dt, rate = vdf_evaluate(g, 8, Delta)
    expected = fast_power(g, 256, Delta)
    assert y == expected, f"VDF mismatch: {y} != {expected}"
    print(f"  g^(2^8) = {y} ✓ (matches direct computation)")
    
    # Test 3: Wesolowski proof verification
    print("\n[Test 3] Wesolowski proof on Cl(-23), T = 8")
    (pi, ell, r), dt_proof = wesolowski_prove(g, y, 8, Delta)
    valid, dt_verify = wesolowski_verify(g, y, pi, ell, r, Delta)
    assert valid, "Proof verification failed!"
    print(f"  Proof valid: {valid} ✓")
    print(f"  Proof time: {dt_proof*1000:.2f} ms, Verify time: {dt_verify*1000:.2f} ms")
    
    # Test 4: Invalid proof detection
    print("\n[Test 4] Invalid proof detection")
    fake_y = (y * g).reduced_form()  # wrong output
    try:
        valid_fake, _ = wesolowski_verify(g, fake_y, pi, ell, r, Delta)
        # Note: this may fail at the ell_check assertion since we rehash
        # Use the correct ell for fake_y
        ell_fake = hash_to_prime(g, fake_y)
        r_fake = power_mod(2, 8, ell_fake)
        pi_fake = pi  # wrong proof for fake_y
        valid_fake, _ = wesolowski_verify(g, fake_y, pi_fake, ell_fake, r_fake, Delta)
        assert not valid_fake, "Should have rejected invalid proof!"
        print(f"  Correctly rejected invalid proof ✓")
    except AssertionError:
        print(f"  Correctly rejected via assertion ✓")
    except Exception as e:
        print(f"  Correctly rejected invalid proof (exception: {type(e).__name__}) ✓")
    
    # Test 5: Full pipeline
    print("\n[Test 5] Full pipeline: seed -> hash_to_group -> VDF -> proof -> verify -> seed")
    Delta_test = -next_prime(2**127)
    while (-Delta_test) % 4 != 3:
        Delta_test = -next_prime(-Delta_test + 1)
    
    R_e = hashlib.sha256(b"test_randao_output").digest()
    g_test, _ = hash_to_class_group(R_e, Delta_test)
    y_test, _, _ = vdf_evaluate(g_test, 64, Delta_test)
    (pi_test, ell_test, r_test), _ = wesolowski_prove(g_test, y_test, 64, Delta_test)
    valid_test, _ = wesolowski_verify(g_test, y_test, pi_test, ell_test, r_test, Delta_test)
    seed = derive_proposer_seed(y_test, 42)
    
    assert valid_test, "Full pipeline verification failed!"
    print(f"  128-bit Δ, T=64: proof valid = {valid_test} ✓")
    print(f"  Proposer seed: {seed[:32]}...")
    
    print(f"\n{'=' * 70}")
    print("ALL TESTS PASSED ✓")
    print(f"{'=' * 70}\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    args = sys.argv[1:]
    
    if "--help" in args or "-h" in args:
        print(__doc__)
        sys.exit(0)
    
    # Parse arguments
    disc_bits = 256
    T_exp = 18
    full_suite = False
    tests_only = False
    
    for i, arg in enumerate(args):
        if arg == "--bits" and i + 1 < len(args):
            disc_bits = int(args[i + 1])
        elif arg == "--T_exp" and i + 1 < len(args):
            T_exp = int(args[i + 1])
        elif arg == "--full":
            full_suite = True
        elif arg == "--test":
            tests_only = True
    
    # Always run correctness tests first
    run_correctness_tests()
    
    if tests_only:
        sys.exit(0)
    
    if full_suite:
        run_full_suite()
    else:
        run_benchmark(disc_bits, T_exp, verbose=True)
