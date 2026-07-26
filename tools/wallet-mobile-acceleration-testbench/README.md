# Mobile Wallet Acceleration Testbench

This directory prepares isolated iOS/Metal and Android CPU/Vulkan measurements
for the exact Monero key-derivation operation:

```text
D = 8 * a * R
```

It is not product integration. Every test uses only the deterministic public
`MWMTV1` corpus produced by `tools/wallet-crypto-testbench`. A real wallet view
key or spend key must never be supplied to these tools.

## Required comparison

Every physical-device result must report the same classes separately:

1. historical Monero C/Ref10 with one worker;
2. the current Rust wallet adapter with one worker;
3. the current Rust wallet adapter with the selected device worker budget;
4. Metal or Vulkan with byte-for-byte Dalek validation;
5. temperature or throttling state before and after the measured rounds.

Rates are always `derivations_per_second`. GPU dispatch-only measurements are
diagnostics and must not be presented as cryptographic derivation rates.

## iOS preparation

`ios/WalletMetalMobileTestbench.swift` is an app-independent runner for the
existing three-pass M12/M16 Metal kernel. It parses the same `MWMTV1` corpus,
checks every result and the invalid-point contract, reports timings and clears
all shared Metal buffers before returning.

The source can be compiled for both iPhoneOS and iPhoneSimulator without
touching the React Native product:

```sh
bash tools/wallet-mobile-acceleration-testbench/ios/build-ios-testbench.sh
```

The mobile runner itself can be exercised byte-for-byte on the Mac GPU before
an iPhone is available:

```sh
bash tools/wallet-mobile-acceleration-testbench/ios/run-macos-smoke.sh
```

An iPhone Simulator build checks Swift/Metal API compatibility, but runs on the
Mac GPU and is not an A-series performance, battery or thermal measurement.
Final enablement still requires a physical iPhone.

## Android preparation

First build the Vulkan capability probe without a connected phone:

```sh
bash tools/wallet-mobile-acceleration-testbench/android/run-vulkan-probe.sh \
  --build-only
```

When the Pixel is connected with USB debugging enabled, omit `--build-only`.
The probe records the Vulkan device, compute queues, subgroup size, workgroup
limits and `shaderInt64`. These values decide whether the 25/26-bit kernel or a
portable 32-bit kernel is appropriate.

The CPU reference script builds the existing Rust/Dalek adapter and historical
C/Ref10 benchmark for `arm64-v8a`, pushes only the binaries plus public corpus
to `/data/local/tmp`, and records one-worker and device-worker results:

```sh
bash tools/wallet-mobile-acceleration-testbench/android/run-cpu-references.sh
```

The validated Vulkan M12 port is deliberately isolated and split into ten
pipelines to limit mobile driver compiler pressure. It uses radix-25/26 field
limbs and requires shader 64-bit integers. Build the complete benchmark without
a phone:

```sh
bash tools/wallet-mobile-acceleration-testbench/android/run-vulkan-radix2625-benchmark.sh \
  --build-only
```

Omit `--build-only` for a connected physical device. The defaults reproduce the
selected Pixel configuration: 8,192 points, 10 timed rounds, two warm-up rounds,
workgroup size 32 and two point doublings per dispatch. The runner generates the
same public corpus and fully optimizes the two arithmetic-heavy SPIR-V stages
after exposing only their fixed-size loops to the optimizer. The other stages
retain the conservative dead-function-only pass to minimize driver-compiler
risk. The runner records both policies and the thermal state, and accepts a
rate only after every result and the invalid point match the Dalek oracle.
Set `WALLET_ANDROID_GPU_SPIRV_OPTIMIZATION=dead-functions` to reproduce the
pre-optimization control build.

The pure-u32 M13 port and its runner remain experimental fallback research. The
Pixel 8 Pro measurements, optimization rejections and product decision are
documented in `PIXEL_8_PRO_RESULTS.md`. A Vulkan kernel must not be integrated
into the product unless it is both byte-correct and meaningfully faster than
the Rust path on the same physical device.

The current measured Pixel 8 Pro decision is CPU-only: use the Rust wallet
adapter with nine workers and leave Vulkan plus concurrent CPU/GPU derivation
disabled. Nine is specific to the tested Tensor G3 topology; other Android
devices still require a dynamically selected, device-appropriate worker budget.
