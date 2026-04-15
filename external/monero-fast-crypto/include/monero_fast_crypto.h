/*
 * monero_fast_crypto.h
 *
 * Drop-in replacement for Monero's ge_scalarmult using curve25519-dalek.
 * ~3.5x faster than ref10 on Apple Silicon (ARM NEON).
 *
 * Link with: libmonero_fast_crypto.a
 */

#ifndef MONERO_FAST_CRYPTO_H
#define MONERO_FAST_CRYPTO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Fast variable-base scalar multiplication on Ed25519.
 * Replaces: ge_scalarmult() in crypto-ops.c
 *
 * result: output, 32 bytes (compressed point)
 * scalar: input, 32 bytes
 * point:  input, 32 bytes (compressed point)
 *
 * Returns 0 on success, -1 on invalid point.
 */
int fast_ge_scalarmult(uint8_t *result, const uint8_t *scalar, const uint8_t *point);

/*
 * Fast fixed-base scalar multiplication (basepoint G).
 * Replaces: ge_scalarmult_base() in crypto-ops.c
 */
int fast_ge_scalarmult_base(uint8_t *result, const uint8_t *scalar);

#ifdef __cplusplus
}
#endif

#endif /* MONERO_FAST_CRYPTO_H */
