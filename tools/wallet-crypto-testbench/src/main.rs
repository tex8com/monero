//! Isolated benchmark for Monero's `generate_key_derivation` operation.
//!
//! The timed operation is exactly `8 * a * R`, where `a` is one common view
//! scalar and every `R` is a different, valid compressed Edwards25519 point.
//! Corpus generation and byte-for-byte equivalence checks deliberately happen
//! before the timer starts.

use curve25519_dalek::{
    constants::ED25519_BASEPOINT_POINT, edwards::CompressedEdwardsY, scalar::Scalar,
};
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use std::{
    env, fs,
    hint::black_box,
    path::PathBuf,
    process,
    time::{Duration, Instant},
};

// Import the exact wallet adapter, including the `extern "C"` entry points
// used by crypto.cpp. The testbench has the same Dalek and Rayon dependencies
// as the adapter, so it compiles the product code rather than a copy of it.
#[path = "../../../external/monero-fast-crypto/src/lib.rs"]
mod wallet_fast_crypto;

const DEFAULT_POINTS: usize = 131_072;
const DEFAULT_ROUNDS: usize = 100;
const DEFAULT_WARMUP_ROUNDS: usize = 2;
const CORPUS_SEED: u64 = 0x4d4f_4e45_524f_3852;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Variant {
    DalekDirect,
    WalletScalar,
}

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Self::DalekDirect => "dalek_direct_parallel",
            Self::WalletScalar => "wallet_scalar_ffi_parallel",
        }
    }
}

#[derive(Debug)]
struct Config {
    points: usize,
    rounds: usize,
    warmup_rounds: usize,
    workers: usize,
    variants: Vec<Variant>,
    export_metal_vectors: Option<PathBuf>,
}

struct Corpus {
    scalar: [u8; 32],
    points: Vec<[u8; 32]>,
    fingerprint: u64,
}

fn usage() -> ! {
    eprintln!(
        "Usage: monero-wallet-crypto-testbench [options]\n\n\
         --variant <all|dalek|wallet-scalar>\n\
         --points <N>          valid transaction public keys (default {DEFAULT_POINTS})\n\
         --rounds <N>          timed corpus passes (default {DEFAULT_ROUNDS})\n\
         --warmup-rounds <N>   unreported corpus passes (default {DEFAULT_WARMUP_ROUNDS})\n\
         --workers <N>         fixed CPU worker budget (default: hardware parallelism)\n\
         --export-metal-vectors <PATH>  write Dalek-checked M1 vector corpus and exit\n\
         --help"
    );
    process::exit(2);
}

fn parse_positive(value: Option<String>, option: &str) -> usize {
    value
        .and_then(|text| text.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            eprintln!("{option} requires a positive integer");
            usage();
        })
}

fn parse_variants(value: String) -> Vec<Variant> {
    match value.as_str() {
        "all" => vec![Variant::DalekDirect, Variant::WalletScalar],
        "dalek" => vec![Variant::DalekDirect],
        "wallet-scalar" => vec![Variant::WalletScalar],
        _ => {
            eprintln!("unknown --variant {value}");
            usage();
        }
    }
}

fn parse_args() -> Config {
    let mut points = DEFAULT_POINTS;
    let mut rounds = DEFAULT_ROUNDS;
    let mut warmup_rounds = DEFAULT_WARMUP_ROUNDS;
    let mut workers = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    let mut variants = parse_variants("all".to_owned());
    let mut export_metal_vectors = None;
    let mut args = env::args().skip(1);

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--points" => points = parse_positive(args.next(), "--points"),
            "--rounds" => rounds = parse_positive(args.next(), "--rounds"),
            "--warmup-rounds" => warmup_rounds = parse_positive(args.next(), "--warmup-rounds"),
            "--workers" => workers = parse_positive(args.next(), "--workers"),
            "--variant" => variants = args.next().map(parse_variants).unwrap_or_else(|| usage()),
            "--export-metal-vectors" => {
                export_metal_vectors = args.next().map(PathBuf::from).or_else(|| usage())
            }
            "--help" | "-h" => usage(),
            _ => {
                eprintln!("unknown argument {argument}");
                usage();
            }
        }
    }

    Config {
        points,
        rounds,
        warmup_rounds,
        workers,
        variants,
        export_metal_vectors,
    }
}

// SplitMix64 is used only to make public test inputs reproducible; it is not
// used for wallet key material or any product cryptography.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn scalar_from_state(state: &mut u64) -> Scalar {
    let mut bytes = [0u8; 32];
    for word in bytes.chunks_exact_mut(8) {
        word.copy_from_slice(&splitmix64(state).to_le_bytes());
    }
    let scalar = Scalar::from_bytes_mod_order(bytes);
    if scalar == Scalar::ZERO {
        Scalar::ONE
    } else {
        scalar
    }
}

fn fingerprint_bytes(mut fingerprint: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        fingerprint ^= u64::from(*byte);
        fingerprint = fingerprint.wrapping_mul(0x1000_0000_01b3);
    }
    fingerprint
}

fn make_corpus(point_count: usize) -> Corpus {
    let mut state = CORPUS_SEED;
    let scalar = scalar_from_state(&mut state).to_bytes();
    let mut points = Vec::with_capacity(point_count);
    let mut fingerprint = 0xcbf2_9ce4_8422_2325;
    fingerprint = fingerprint_bytes(fingerprint, &scalar);

    for _ in 0..point_count {
        let point_scalar = scalar_from_state(&mut state);
        let encoded = (point_scalar * ED25519_BASEPOINT_POINT)
            .compress()
            .to_bytes();
        fingerprint = fingerprint_bytes(fingerprint, &encoded);
        points.push(encoded);
    }

    Corpus {
        scalar,
        points,
        fingerprint,
    }
}

fn dalek_direct_one(scalar_bytes: &[u8; 32], point_bytes: &[u8; 32]) -> [u8; 32] {
    let scalar = Scalar::from_bytes_mod_order(*scalar_bytes);
    let point = CompressedEdwardsY(*point_bytes)
        .decompress()
        .expect("corpus must contain valid compressed Edwards25519 points");
    (Scalar::from(8u64) * scalar * point).compress().to_bytes()
}

fn wallet_scalar_one(scalar_bytes: &[u8; 32], point_bytes: &[u8; 32]) -> [u8; 32] {
    let mut result = [0u8; 32];
    let status = wallet_fast_crypto::fast_generate_key_derivation(
        result.as_mut_ptr(),
        scalar_bytes.as_ptr(),
        point_bytes.as_ptr(),
    );
    assert_eq!(
        status, 0,
        "valid corpus point was rejected by wallet adapter"
    );
    result
}

fn result_word(bytes: &[u8; 32]) -> u64 {
    u64::from_le_bytes(bytes[..8].try_into().expect("fixed-size slice"))
}

const METAL_VECTOR_MAGIC: &[u8; 8] = b"MWMTV1\0\0";
const METAL_VECTOR_VERSION: u32 = 1;
const METAL_VECTOR_HEADER_BYTES: usize = 88;

/// Export public, deterministic inputs plus exact Dalek adapter results for
/// the Metal M1 correctness harness. This contains no wallet material: the
/// scalar and points originate from `CORPUS_SEED` above. The invalid input is
/// included so the GPU kernel must preserve the adapter's rejection contract.
fn export_metal_vectors(corpus: &Corpus, path: &PathBuf) {
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
        .expect("an invalid compressed Edwards25519 point must exist");

    let capacity = METAL_VECTOR_HEADER_BYTES
        .checked_add(
            corpus
                .points
                .len()
                .checked_mul(64)
                .expect("metal vector size overflow"),
        )
        .expect("metal vector header overflow");
    let mut output = Vec::with_capacity(capacity);
    output.extend_from_slice(METAL_VECTOR_MAGIC);
    output.extend_from_slice(&METAL_VECTOR_VERSION.to_le_bytes());
    output.extend_from_slice(
        &u32::try_from(corpus.points.len())
            .expect("metal vector corpus exceeds u32")
            .to_le_bytes(),
    );
    output.extend_from_slice(&corpus.scalar);
    output.extend_from_slice(&corpus.fingerprint.to_le_bytes());
    output.extend_from_slice(&invalid);
    assert_eq!(output.len(), METAL_VECTOR_HEADER_BYTES);

    for point in &corpus.points {
        output.extend_from_slice(point);
        output.extend_from_slice(&wallet_scalar_one(&corpus.scalar, point));
    }
    assert_eq!(output.len(), capacity);
    fs::write(path, output).unwrap_or_else(|error| {
        eprintln!(
            "failed to write Metal vector file {}: {error}",
            path.display()
        );
        process::exit(1);
    });
    println!("metal_vector_export=pass");
    println!("metal_vector_format=MWMTV1");
    println!("metal_vector_version={METAL_VECTOR_VERSION}");
    println!("metal_vector_header_bytes={METAL_VECTOR_HEADER_BYTES}");
    println!("metal_vector_records={}", corpus.points.len());
    println!(
        "metal_vector_corpus_fingerprint_fnv1a64=0x{:016x}",
        corpus.fingerprint
    );
    println!("metal_vector_path={}", path.display());
}

fn checksum_parallel<F>(pool: &ThreadPool, points: &[[u8; 32]], operation: F) -> u64
where
    F: Fn(&[u8; 32]) -> [u8; 32] + Sync + Send,
{
    pool.install(|| {
        points
            .par_iter()
            .map(|point| result_word(&operation(point)))
            .reduce(|| 0, u64::wrapping_add)
    })
}

fn verify_equivalence(corpus: &Corpus, workers: usize) {
    let pool = ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|index| format!("crypto-bench-verify-{index}"))
        .build()
        .expect("verification thread pool");

    let direct_checksum = checksum_parallel(&pool, &corpus.points, |point| {
        dalek_direct_one(&corpus.scalar, point)
    });
    let wallet_checksum = checksum_parallel(&pool, &corpus.points, |point| {
        wallet_scalar_one(&corpus.scalar, point)
    });
    assert_eq!(
        direct_checksum, wallet_checksum,
        "Dalek direct and wallet scalar paths differ"
    );

    // The product contract retains invalid-point reporting. Check it here once
    // so an optimized kernel cannot silently turn an invalid input into data.
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
        .expect("an invalid compressed Edwards25519 point must exist");
    let mut output = [0u8; 32];
    assert_eq!(
        wallet_fast_crypto::fast_generate_key_derivation(
            output.as_mut_ptr(),
            corpus.scalar.as_ptr(),
            invalid.as_ptr(),
        ),
        -1
    );

    println!("preflight=pass");
    println!("preflight_direct_checksum={direct_checksum}");
    println!("preflight_wallet_checksum={wallet_checksum}");
}

fn run_parallel_variant(
    pool: &ThreadPool,
    corpus: &Corpus,
    rounds: usize,
    variant: Variant,
) -> u64 {
    let mut checksum = 0u64;
    for _ in 0..rounds {
        let round_checksum = match variant {
            Variant::DalekDirect => checksum_parallel(pool, &corpus.points, |point| {
                dalek_direct_one(&corpus.scalar, point)
            }),
            Variant::WalletScalar => checksum_parallel(pool, &corpus.points, |point| {
                wallet_scalar_one(&corpus.scalar, point)
            }),
        };
        checksum = checksum.wrapping_add(round_checksum);
    }
    checksum
}

fn run_variant(config: &Config, corpus: &Corpus, variant: Variant) {
    let pool = ThreadPoolBuilder::new()
        .num_threads(config.workers)
        .thread_name(move |index| format!("crypto-bench-{}-{index}", variant.name()))
        .build()
        .expect("benchmark thread pool");

    let checksum = run_parallel_variant(&pool, corpus, config.warmup_rounds, variant);
    black_box(checksum);

    let started = Instant::now();
    let checksum = run_parallel_variant(&pool, corpus, config.rounds, variant);
    let elapsed = started.elapsed();
    black_box(checksum);

    let operations = corpus
        .points
        .len()
        .checked_mul(config.rounds)
        .expect("operation count overflow");
    let ops_per_second = operations as f64 / elapsed.as_secs_f64();
    print_result(
        config,
        corpus,
        variant,
        operations,
        elapsed,
        ops_per_second,
        checksum,
    );
}

fn print_result(
    config: &Config,
    corpus: &Corpus,
    variant: Variant,
    operations: usize,
    elapsed: Duration,
    ops_per_second: f64,
    checksum: u64,
) {
    println!("result_begin");
    println!("algorithm=monero_generate_key_derivation_8_times_a_times_r");
    println!("variant={}", variant.name());
    println!("point_encoding=compressed_edwards25519_32_bytes");
    println!("scalar_contract=one_common_32_byte_view_scalar");
    println!("corpus_kind=deterministic_valid_transaction_public_key_simulation");
    println!("corpus_seed=0x{CORPUS_SEED:016x}");
    println!("corpus_fingerprint_fnv1a64=0x{:016x}", corpus.fingerprint);
    println!("points_per_round={}", corpus.points.len());
    println!("timed_rounds={}", config.rounds);
    println!("warmup_rounds={}", config.warmup_rounds);
    println!("worker_budget={}", config.workers);
    println!("operations={operations}");
    println!("elapsed_ns={}", elapsed.as_nanos());
    println!("elapsed_seconds={:.9}", elapsed.as_secs_f64());
    println!("derivations_per_second={ops_per_second:.3}");
    println!(
        "million_derivations_per_second={:.6}",
        ops_per_second / 1_000_000.0
    );
    println!("result_checksum_u64=0x{checksum:016x}");
    println!("result_end");
}

fn main() {
    let config = parse_args();
    let corpus = make_corpus(config.points);

    println!("testbench=monero_wallet_crypto_testbench_v1");
    println!("curve25519_dalek_version=4.1.3");
    println!("corpus_points={}", corpus.points.len());
    println!("configured_workers={}", config.workers);
    verify_equivalence(&corpus, config.workers);
    if let Some(path) = &config.export_metal_vectors {
        export_metal_vectors(&corpus, path);
        return;
    }
    for variant in config.variants.iter().copied() {
        run_variant(&config, &corpus, variant);
    }
}
