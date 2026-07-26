# Google Pixel 8 Pro acceleration results

Physical device measured on 2026-07-25:

- Pixel 8 Pro (`husky`), Google Tensor G3, `arm64-v8a`
- Android 17, nine online CPU cores
- Mali-G715, Vulkan 1.4.343
- public deterministic `MWMTV1` corpus fingerprint
  `0x710825165cb9eac9`
- exact workload: Monero `generate_key_derivation`, `D = 8 * a * R`

No real wallet key was used. Every accepted CPU run checked the result against
the same Dalek oracle and checked the invalid-point contract before timing.

## Formal CPU result

Each formal run used 8,192 derivations per round, 20 timed rounds and three
warm-up rounds. The table reports the median of three complete physical-device
runs.

| Implementation | Workers | Ableitungen/s | Relative to C/Ref10 |
| --- | ---: | ---: | ---: |
| Historical Monero C/Ref10 | 1 | 9,961.612 | 1.000x |
| Rust curve25519-dalek direct | 1 | 19,351.360 | 1.943x |
| Current Rust wallet adapter | 1 | 19,382.930 | 1.946x |
| Rust curve25519-dalek direct | 9 | 79,705.756 | 8.001x |
| Current Rust wallet adapter | 9 | 77,962.845 | 7.826x |

The three current-wallet nine-worker results were 79,645.241, 77,079.616 and
77,962.845 Ableitungen/s. Android thermal status remained `0` before and after
all three formal runs, so none was recorded as thermally throttled.

A separate short worker sweep tested every count from one through nine. Nine
workers was the best tested setting and reached 81,035.618 Ableitungen/s in
that diagnostic sweep. This peak is not substituted for the formal median.

## Pixel-specific CPU diagnostics

The heterogeneous CPU layout is four efficiency cores, four Cortex-A715-class
performance cores and one Cortex-X3-class prime core. Explicit affinity tests
did not beat using all nine cores:

| Diagnostic CPU configuration | Ableitungen/s |
| --- | ---: |
| All nine cores, explicit affinity | 80,724.178 |
| Eight cores | 74,760.746 |
| Performance plus prime cores, five workers | 68,658.924 |
| Four performance cores | 61,261.888 |
| Prime core only | 19,234.549 |

A separate `target-cpu=cortex-x3` build reached a three-run median of
81,133.453 Ableitungen/s versus 79,222.426 for a paired generic build, a
2.41% diagnostic improvement. Enabling SVE2 reduced the rate to 81,392.372
versus 82,690.224 for the otherwise identical X3 build and was rejected.
The X3 binary is not a safe universal Android binary; product use would require
runtime device selection plus a generic fallback.

## Vulkan capability result

The Pixel exposes a usable Vulkan compute device:

| Capability | Pixel 8 Pro |
| --- | --- |
| GPU | Mali-G715 |
| Compute queues | 2 |
| Dedicated compute-only queue | 0 |
| Subgroup size | 16 |
| 64-bit shader integers | yes |
| Maximum workgroup invocations | 1,024 |
| Shared memory per workgroup | 32,768 bytes |
| GPU timestamps | yes, 40.6901 ns period |

This proves that the phone has a compute-capable GPU. It does not prove that a
particular cryptographic shader is accepted, correct or faster.

## Exact cryptographic Vulkan result

The original monolithic M13 shader first exceeded the Mali compiler's practical
limits. Splitting decode, multiplication and compression into ten small
pipelines made it compile and pass exact validation. That portable pure-u32
version reached only 505.604 Ableitungen/s at 1,024 points.

The faster Metal M12 field representation was then ported to Vulkan using
radix-25/26 limbs and shader 64-bit integers. All ten pipelines compile on the
Mali-G715. Every measured run validates all output bytes against Dalek and also
validates the rejected-point behavior.

The selected configuration is 8,192 points, workgroup size 32, two point
doublings per dispatch, 10 timed rounds and two warm-up rounds:

| Formal run | Ableitungen/s | Validation | Thermal before/after |
| --- | ---: | --- | --- |
| `pixel8pro-vulkan-m12-d2-r1-20260725-001` | 8,522.704 | pass | 0 / 0 |
| `pixel8pro-vulkan-m12-d2-r2-20260725-001` | 8,531.040 | pass | 0 / 0 |
| `pixel8pro-vulkan-m12-d2-r3-20260725-001` | 8,453.612 | pass | 0 / 0 |
| **Median** | **8,522.704** | **pass** | **0 / 0** |

The median Vulkan result is 0.856x historical one-worker C/Ref10 and 0.109x
the current nine-worker Rust wallet path. In other words, the Rust CPU path is
9.15x faster on this phone.

The GPU timer splits the formal median run as follows:

| GPU phase | Time over 10 rounds | Share |
| --- | ---: | ---: |
| Decode | 0.369 s | 3.9% |
| Scalar multiplication | 9.080 s | 94.8% |
| Compress | 0.128 s | 1.3% |

Frequency sampling during a long run observed 890 MHz in 33 of 40 busy
samples; the GPU returned to 150 MHz after completion. This proves that the
frequency governor raised the GPU clock and that the arithmetic phase dominates.
It does not prove 100% execution-unit occupancy because that hardware counter
is not exposed to the unprivileged test process.

Several exact alternatives were tested and rejected because they were slower:

| Vulkan diagnostic | Ableitungen/s |
| --- | ---: |
| Selected separate double/add, two doubles per dispatch | 8,522.704 formal median |
| One double per dispatch | 8,353.559 |
| Four doubles per dispatch | about 8,188 |
| Batch inversion, chunks of 16 | 7,528.656 |
| Combined four-double-plus-add shader | 7,719.124 |
| Shared-memory four-digit chunks | 3,651.894 |

The workgroup sweep covered 16, 32, 64, 128 and 256 invocations. Differences
were small; size 32 produced the best repeatable long-run configuration. The
remaining bottleneck is serial dependency inside each individual Edwards25519
scalar multiplication, not host transfer or GPU clock selection.

## Second Pixel and selective SPIR-V optimization

A second physical Pixel 8 Pro was measured separately on 2026-07-25:

- device `husky`, Google Tensor G3, Android 16;
- Mali-G715, Vulkan 1.4.305;
- the same 8,192-point `MWMTV1` corpus and fingerprint
  `0x710825165cb9eac9`.

The generator previously kept every shader loop runtime-bounded so that the
large shaders would remain acceptable to the Mali driver. That conservative
workaround also prevented effective optimization in the two arithmetic-heavy
stages. The revised policy exposes only the fixed-size loops in
`multiply_double` and `multiply_add`, then runs `spirv-opt -O` on those two
stages. All other stages keep the conservative dead-function-only pass.

A same-device control using exposed loops but only dead-function pruning
reached 10,469.529 Ableitungen/s. The selectively fully optimized build then
produced these three complete from-source runs:

| Formal run | Ableitungen/s | Validation | Thermal before/after |
| --- | ---: | --- | --- |
| `pixel8pro2-vulkan-m12-spirvO-r1-20260725-001` | 20,744.386 | pass | 0 / 0 |
| `pixel8pro2-vulkan-m12-spirvO-r2-20260725-001` | 20,673.241 | pass | 0 / 0 |
| `pixel8pro2-vulkan-m12-spirvO-r3-20260725-001` | 20,693.944 | pass | 0 / 0 |
| **Median** | **20,693.944** | **pass** | **0 / 0** |

The new median is 97.66% above the same-device control. The shaders, generator
and vector corpus have identical hashes in all three formal runs. The median
GPU phase times over 10 rounds are 0.365 seconds for decode, 3.442 seconds for
scalar multiplication and 0.128 seconds for compression. The arithmetic phase
therefore remains the dominant target, but its time was nearly halved by
letting the offline SPIR-V optimizer inline and scalarize the fixed-size
arithmetic before the mobile driver sees it.

The workload-identical CPU references were also repeated three times on the
second phone:

| Implementation | Workers | Median Ableitungen/s | Relative to C/Ref10 |
| --- | ---: | ---: | ---: |
| Historical Monero C/Ref10 | 1 | 8,325.397 | 1.000x |
| Rust curve25519-dalek direct | 1 | 16,962.841 | 2.037x |
| Current Rust wallet adapter | 1 | 17,061.050 | 2.049x |
| Rust curve25519-dalek direct | 9 | 79,429.513 | 9.541x |
| Current Rust wallet adapter | 9 | 75,586.545 | 9.079x |
| Selectively optimized Vulkan | GPU | 20,693.944 | 2.486x |

On this second phone Vulkan is 1.213x the one-worker Rust wallet rate, but only
0.274x the nine-worker Rust wallet rate. Put another way, the full CPU path is
still 3.65x faster. A long same-device control measured 76,590.360 CPU and
20,884.642 GPU Ableitungen/s separately. The subsequent optimized concurrent
run was interrupted by a physical ADB device swap before a result was returned,
so no combined rate is claimed for that second phone.

The optimized shader was then rebuilt and measured three times on the first
Pixel and its different Android/Vulkan driver:

| Formal run | Ableitungen/s | Validation | Thermal before/after |
| --- | ---: | --- | --- |
| `pixel8pro1-vulkan-m12-spirvO-r1-20260725-001` | 20,677.429 | pass | 0 / 0 |
| `pixel8pro1-vulkan-m12-spirvO-r2-20260725-001` | 20,992.410 | pass | 0 / 0 |
| `pixel8pro1-vulkan-m12-spirvO-r3-20260725-001` | 20,299.998 | pass | 0 / 0 |
| **Median** | **20,677.429** | **pass** | **0 / 0** |

This is 2.43x the old 8,522.704 Vulkan median on the same phone, but the
77,962.845 nine-worker Rust wallet median remains 3.77x faster.

Two final diagnostics did not change that decision. Applying `spirv-opt -O` to
every stage reduced the rate to 19,455.518 Ableitungen/s and was rejected.
Optimizing only `multiply_init` and `multiply_finish` in addition to the two
selected stages reached 21,484.260 Ableitungen/s in a short validated run. That
3.90% diagnostic gain is not a formal result and would still leave the CPU
about 3.63x faster.

## Concurrent CPU and GPU diagnostics

The original shader plus an X3-tuned nine-worker CPU build had previously
produced only a noise-sized 0.07% combined gain. Concurrency was retested after
the large SPIR-V improvement.

The optimized Vulkan and current nine-worker Rust wallet workloads were started
with their timed windows aligned. CPU-only and GPU-only controls reached
78,336.722 and 20,849.415 Ableitungen/s respectively. Three sustained balanced
CPU+GPU runs produced:

| Balanced run | CPU Ableitungen/s | GPU Ableitungen/s | Combined over the longer timed window |
| --- | ---: | ---: | ---: |
| 1 | 56,776.852 | 21,084.331 | 76,080.981 |
| 2 | 56,268.285 | 21,002.672 | 75,399.502 |
| 3 | 46,267.961 | 20,924.188 | 61,999.068 |
| **Median** |  |  | **75,399.502** |

The combined median is 3.75% below the same-session CPU-only control. The GPU
held its rate, but sustained shared load reduced CPU throughput and raised the
Android thermal status from 0 to 1 by the third run. Eight CPU workers plus GPU
was also slower than nine workers plus GPU. Concurrent CPU+GPU execution is
therefore rejected for the Pixel 8 Pro.

## Product decision

The selected Pixel 8 Pro product policy is:

1. use the current Rust wallet adapter;
2. use nine CPU workers on the measured Tensor G3/Pixel 8 Pro;
3. do not dispatch the Vulkan derivation path;
4. do not run CPU and GPU derivations concurrently.

Nine workers is a measured Pixel 8 Pro choice, not a universal Android
constant. Other devices must select a safe worker budget from their available
CPU topology and retain a lower-worker fallback.

The CPU policy is about 7.83x historical one-worker C/Ref10 on the first phone
and 9.08x on the second, and passed all exact checks. The selectively optimized
Vulkan port remains correct, reproducible and useful as research evidence, but
stays disabled because it is substantially slower and worsens sustained
combined throughput. Reconsider it only if a future GPU path beats the Rust
CPU end-to-end or demonstrates a measured energy/performance benefit across a
physical-device matrix.

All Vulkan and target-specific CPU work remains in this isolated testbench. No
wallet product code was changed.
