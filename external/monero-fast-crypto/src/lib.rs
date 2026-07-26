use curve25519_dalek::edwards::CompressedEdwardsY;
use curve25519_dalek::scalar::Scalar;
use rayon::prelude::*;
use std::sync::OnceLock;

// The wallet owns concurrency at one layer only. The C++ caller selects a
// worker budget once per process, then this dedicated pool is reused by every
// scan chunk. This avoids nesting Rayon underneath the C++ compute pool.
static WALLET_DERIVATION_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

fn wallet_derivation_pool(workers: usize) -> &'static rayon::ThreadPool {
    WALLET_DERIVATION_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers.max(1))
            .thread_name(|index| format!("monero-derivation-{}", index))
            .build()
            .expect("failed to create wallet derivation worker pool")
    })
}

/// Complete generate_key_derivation in Rust — no round-trip back to C.
///
/// Computes: result = 8 * scalar * point (ECDH + cofactor, full Monero key derivation)
///
/// This replaces BOTH ge_scalarmult AND ge_mul8 from Monero's crypto.cpp,
/// avoiding the expensive compress→decompress round-trip through C FFI.
///
/// Returns: 0 on success, -1 on invalid point
#[no_mangle]
pub extern "C" fn fast_generate_key_derivation(
    result_bytes: *mut u8,
    scalar_bytes: *const u8,
    point_bytes: *const u8,
) -> i32 {
    let scalar_slice = unsafe { std::slice::from_raw_parts(scalar_bytes, 32) };
    let point_slice = unsafe { std::slice::from_raw_parts(point_bytes, 32) };

    // Decode scalar
    let mut scalar_arr = [0u8; 32];
    scalar_arr.copy_from_slice(scalar_slice);
    let scalar = Scalar::from_bytes_mod_order(scalar_arr);

    // Decode point
    let mut point_arr = [0u8; 32];
    point_arr.copy_from_slice(point_slice);
    let compressed = CompressedEdwardsY(point_arr);

    let Some(point) = compressed.decompress() else {
        return -1;
    };

    // Scalar multiplication + cofactor ×8, all in Rust — no round-trip
    let cofactor = Scalar::from(8u64);
    let result = (cofactor * scalar) * point;

    // Encode result
    let result_compressed = result.compress();
    unsafe {
        std::ptr::copy_nonoverlapping(result_compressed.as_bytes().as_ptr(), result_bytes, 32);
    }

    0
}

/// Legacy API kept for compatibility
#[no_mangle]
pub extern "C" fn fast_ge_scalarmult(
    result_bytes: *mut u8,
    scalar_bytes: *const u8,
    point_bytes: *const u8,
) -> i32 {
    let scalar_slice = unsafe { std::slice::from_raw_parts(scalar_bytes, 32) };
    let point_slice = unsafe { std::slice::from_raw_parts(point_bytes, 32) };

    let mut scalar_arr = [0u8; 32];
    scalar_arr.copy_from_slice(scalar_slice);
    let scalar = Scalar::from_bytes_mod_order(scalar_arr);

    let mut point_arr = [0u8; 32];
    point_arr.copy_from_slice(point_slice);
    let compressed = CompressedEdwardsY(point_arr);

    let Some(point) = compressed.decompress() else {
        return -1;
    };

    let result = scalar * point;

    let result_compressed = result.compress();
    unsafe {
        std::ptr::copy_nonoverlapping(result_compressed.as_bytes().as_ptr(), result_bytes, 32);
    }

    0
}

/// Batch generate_key_derivation — process N derivations in parallel using rayon.
///
/// scalars: N×32 bytes (view keys or per-tx scalars)
/// points:  N×32 bytes (tx public keys)
/// results: N×32 bytes (output derivations)
/// count:   number of operations
///
/// Returns: number of successful operations (should equal count)
#[no_mangle]
pub extern "C" fn fast_generate_key_derivation_batch(
    results: *mut u8,
    scalars: *const u8,
    points: *const u8,
    count: usize,
) -> usize {
    let scalars_slice = unsafe { std::slice::from_raw_parts(scalars, count * 32) };
    let points_slice = unsafe { std::slice::from_raw_parts(points, count * 32) };
    let results_slice = unsafe { std::slice::from_raw_parts_mut(results, count * 32) };

    let cofactor = Scalar::from(8u64);

    let success_count: usize = results_slice
        .par_chunks_mut(32)
        .enumerate()
        .map(|(i, result_chunk)| {
            let mut scalar_arr = [0u8; 32];
            scalar_arr.copy_from_slice(&scalars_slice[i * 32..(i + 1) * 32]);
            let scalar = Scalar::from_bytes_mod_order(scalar_arr);

            let mut point_arr = [0u8; 32];
            point_arr.copy_from_slice(&points_slice[i * 32..(i + 1) * 32]);
            let compressed = CompressedEdwardsY(point_arr);

            let Some(point) = compressed.decompress() else {
                return 0;
            };

            let result = (cofactor * scalar) * point;
            let result_compressed = result.compress();
            result_chunk.copy_from_slice(result_compressed.as_bytes());
            1
        })
        .sum();

    success_count
}

/// Wallet scan batch for the common case where every transaction public key
/// is multiplied by the same view scalar. Besides avoiding one FFI boundary
/// per item, the scalar is decoded once and Rayon is owned by one fixed pool.
/// `valid` preserves the individual invalid-point outcome required by the
/// C++ fallback path.
#[no_mangle]
pub extern "C" fn fast_generate_key_derivation_batch_same_scalar(
    results: *mut u8,
    scalar: *const u8,
    points: *const u8,
    valid: *mut u8,
    count: usize,
    workers: usize,
) -> usize {
    if count == 0 {
        return 0;
    }

    let scalar_slice = unsafe { std::slice::from_raw_parts(scalar, 32) };
    let points_slice = unsafe { std::slice::from_raw_parts(points, count * 32) };
    let results_slice = unsafe { std::slice::from_raw_parts_mut(results, count * 32) };
    let valid_slice = unsafe { std::slice::from_raw_parts_mut(valid, count) };

    let mut scalar_arr = [0u8; 32];
    scalar_arr.copy_from_slice(scalar_slice);
    // Fold the Monero cofactor into the scalar once. Scalar multiplication is
    // associative, so this is bit-identical to `(8 * scalar) * point` used by
    // the scalar API above.
    let scalar = Scalar::from(8u64) * Scalar::from_bytes_mod_order(scalar_arr);

    wallet_derivation_pool(workers).install(|| {
        results_slice
            .par_chunks_mut(32)
            .zip(points_slice.par_chunks(32))
            .zip(valid_slice.par_iter_mut())
            .map(|((result_chunk, point_chunk), valid_entry)| {
                let mut point_arr = [0u8; 32];
                point_arr.copy_from_slice(point_chunk);
                let compressed = CompressedEdwardsY(point_arr);

                let Some(point) = compressed.decompress() else {
                    *valid_entry = 0;
                    return 0usize;
                };

                let result = scalar * point;
                result_chunk.copy_from_slice(result.compress().as_bytes());
                *valid_entry = 1;
                1usize
            })
            .sum()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use curve25519_dalek::constants::ED25519_BASEPOINT_POINT;

    #[test]
    fn same_scalar_batch_matches_scalar_api_and_reports_invalid_points() {
        let scalar = Scalar::from(17u64).to_bytes();
        let invalid = (0u16..=u16::MAX)
            .find_map(|candidate| {
                let mut bytes = [0u8; 32];
                bytes[0] = candidate as u8;
                bytes[1] = (candidate >> 8) as u8;
                CompressedEdwardsY(bytes)
                    .decompress()
                    .is_none()
                    .then_some(bytes)
            })
            .expect("expected an invalid compressed Edwards point");
        let points = vec![
            ED25519_BASEPOINT_POINT.compress().to_bytes(),
            (Scalar::from(3u64) * ED25519_BASEPOINT_POINT)
                .compress()
                .to_bytes(),
            invalid,
        ];
        let mut flat_points = Vec::new();
        for point in &points {
            flat_points.extend_from_slice(point);
        }

        let mut scalar_results = vec![0u8; points.len() * 32];
        let mut scalar_status = Vec::new();
        for (index, point) in points.iter().enumerate() {
            let status = fast_generate_key_derivation(
                scalar_results[index * 32..(index + 1) * 32].as_mut_ptr(),
                scalar.as_ptr(),
                point.as_ptr(),
            );
            scalar_status.push(status == 0);
        }

        let mut batch_results = vec![0u8; points.len() * 32];
        let mut valid = vec![0u8; points.len()];
        let successes = fast_generate_key_derivation_batch_same_scalar(
            batch_results.as_mut_ptr(),
            scalar.as_ptr(),
            flat_points.as_ptr(),
            valid.as_mut_ptr(),
            points.len(),
            3,
        );

        assert_eq!(successes, 2);
        assert_eq!(valid, vec![1, 1, 0]);
        for index in 0..points.len() {
            assert_eq!(valid[index] == 1, scalar_status[index]);
            if valid[index] == 1 {
                assert_eq!(
                    &batch_results[index * 32..(index + 1) * 32],
                    &scalar_results[index * 32..(index + 1) * 32]
                );
            }
        }
    }
}

/// Fast ge_scalarmult_base using curve25519-dalek (precomputed table).
/// Computes: result = scalar * G (fixed-base, basepoint)
#[no_mangle]
pub extern "C" fn fast_ge_scalarmult_base(result_bytes: *mut u8, scalar_bytes: *const u8) -> i32 {
    use curve25519_dalek::constants::ED25519_BASEPOINT_TABLE;

    let scalar_slice = unsafe { std::slice::from_raw_parts(scalar_bytes, 32) };
    let mut scalar_arr = [0u8; 32];
    scalar_arr.copy_from_slice(scalar_slice);
    let scalar = Scalar::from_bytes_mod_order(scalar_arr);

    let result = &scalar * ED25519_BASEPOINT_TABLE;
    let result_compressed = result.compress();
    unsafe {
        std::ptr::copy_nonoverlapping(result_compressed.as_bytes().as_ptr(), result_bytes, 32);
    }

    0
}
