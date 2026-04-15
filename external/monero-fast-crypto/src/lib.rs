use curve25519_dalek::edwards::CompressedEdwardsY;
use curve25519_dalek::scalar::Scalar;
use rayon::prelude::*;

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

/// Fast ge_scalarmult_base using curve25519-dalek (precomputed table).
/// Computes: result = scalar * G (fixed-base, basepoint)
#[no_mangle]
pub extern "C" fn fast_ge_scalarmult_base(
    result_bytes: *mut u8,
    scalar_bytes: *const u8,
) -> i32 {
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
