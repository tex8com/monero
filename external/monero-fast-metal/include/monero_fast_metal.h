/*
 * Native macOS Metal backend for the wallet scan's shared-scalar key
 * derivation batch. This API is intentionally below wallet2: view keys never
 * cross into Tauri/JavaScript.
 */
#ifndef MONERO_FAST_METAL_H
#define MONERO_FAST_METAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum
{
  MONERO_FAST_METAL_UNAVAILABLE = -1,
  MONERO_FAST_METAL_INVALID_ARGUMENT = -2,
  MONERO_FAST_METAL_EXECUTION_ERROR = -3
};

/*
 * Returns 1 when the default Metal device, packaged metallib, and all three
 * compute pipelines are usable. Returns 0 otherwise.
 */
int fast_metal_derivation_available(void);

/*
 * Computes D[i] = 8 * Scalar::from_bytes_mod_order(scalar) * points[i].
 *
 * results: output, count * 32 bytes
 * scalar:  one shared 32-byte wallet view scalar
 * points:  input, count * 32-byte compressed Edwards points
 * valid:   output, count bytes; 1 for a valid point, 0 otherwise
 *
 * A non-negative result is the number of valid derivations. A negative result
 * is one of MONERO_FAST_METAL_* above and requires a CPU fallback. The call is
 * synchronous; all GPU work is complete and secret-dependent shared buffers
 * have been overwritten before it returns.
 */
int64_t fast_metal_generate_key_derivation_batch_same_scalar(
    uint8_t *results,
    const uint8_t *scalar,
    const uint8_t *points,
    uint8_t *valid,
    size_t count);

/*
 * Stable process-lifetime diagnostic for the first initialization or dispatch
 * failure. The returned string is never a wallet key or point.
 */
const char *fast_metal_derivation_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* MONERO_FAST_METAL_H */
