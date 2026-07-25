// Isolated benchmark for Monero's historical Ref10 key-derivation path.
//
// This intentionally copies the exact body present before commit e0a84c6e8:
// ge_frombytes_vartime -> ge_scalarmult -> ge_mul8 -> ge_tobytes.  It never
// handles a wallet key: MWMTV1 contains deterministic public test data and
// expected results generated and checked by the Rust/Dalek testbench.

#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "crypto/crypto-ops.h"

enum { header_bytes = 88, record_bytes = 64, scalar_offset = 16 };

typedef struct {
  const unsigned char *raw;
  size_t length;
  size_t count;
  const unsigned char *scalar;
  const unsigned char *invalid;
} corpus;

typedef struct {
  const corpus *vectors;
  size_t workers;
  pthread_mutex_t mutex;
  pthread_cond_t work_ready;
  pthread_cond_t work_done;
  unsigned long epoch;
  size_t completed;
  int stopping;
  int verify;
  atomic_int failed;
  uint64_t *checksums;
} worker_pool;

typedef struct {
  worker_pool *pool;
  size_t index;
} worker_arg;

static void fail(const char *message) {
  fprintf(stderr, "original Ref10 testbench error: %s\n", message);
  exit(2);
}

static uint32_t read_le32(const unsigned char *input) {
  return (uint32_t)input[0] | ((uint32_t)input[1] << 8) |
      ((uint32_t)input[2] << 16) | ((uint32_t)input[3] << 24);
}

static int original_generate_key_derivation(
    const unsigned char key1[32],
    const unsigned char key2[32],
    unsigned char derivation[32]) {
  ge_p3 point;
  ge_p2 point2;
  ge_p1p1 point3;
  if (sc_check(key2) != 0)
    return 0;
  if (ge_frombytes_vartime(&point, key1) != 0)
    return 0;
  ge_scalarmult(&point2, key2, &point);
  ge_mul8(&point3, &point2);
  ge_p1p1_to_p2(&point2, &point3);
  ge_tobytes(derivation, &point2);
  return 1;
}

static const unsigned char *point_at(const corpus *vectors, size_t index) {
  return vectors->raw + header_bytes + index * record_bytes;
}

static const unsigned char *expected_at(const corpus *vectors, size_t index) {
  return point_at(vectors, index) + 32;
}

static void *worker_main(void *opaque) {
  worker_arg *argument = (worker_arg *)opaque;
  worker_pool *pool = argument->pool;
  unsigned long observed_epoch = 0;

  for (;;) {
    pthread_mutex_lock(&pool->mutex);
    while (!pool->stopping && observed_epoch == pool->epoch)
      pthread_cond_wait(&pool->work_ready, &pool->mutex);
    if (pool->stopping) {
      pthread_mutex_unlock(&pool->mutex);
      break;
    }
    observed_epoch = pool->epoch;
    const int verify = pool->verify;
    pthread_mutex_unlock(&pool->mutex);

    uint64_t checksum = 0;
    for (size_t index = argument->index; index < pool->vectors->count; index += pool->workers) {
      unsigned char derived[32];
      const int success = original_generate_key_derivation(
          point_at(pool->vectors, index), pool->vectors->scalar, derived);
      if (!success || (verify && memcmp(derived, expected_at(pool->vectors, index), 32) != 0))
        atomic_store_explicit(&pool->failed, 1, memory_order_relaxed);
      checksum += derived[0];
    }
    pool->checksums[argument->index] = checksum;

    pthread_mutex_lock(&pool->mutex);
    ++pool->completed;
    if (pool->completed == pool->workers)
      pthread_cond_signal(&pool->work_done);
    pthread_mutex_unlock(&pool->mutex);
  }
  return NULL;
}

static uint64_t pool_run(worker_pool *pool, int verify) {
  pthread_mutex_lock(&pool->mutex);
  pool->verify = verify;
  pool->completed = 0;
  atomic_store_explicit(&pool->failed, 0, memory_order_relaxed);
  ++pool->epoch;
  pthread_cond_broadcast(&pool->work_ready);
  while (pool->completed != pool->workers)
    pthread_cond_wait(&pool->work_done, &pool->mutex);
  pthread_mutex_unlock(&pool->mutex);

  if (atomic_load_explicit(&pool->failed, memory_order_relaxed))
    fail("historical Ref10 result differs from Dalek or a valid point was rejected");
  uint64_t checksum = 0;
  for (size_t worker = 0; worker < pool->workers; ++worker)
    checksum += pool->checksums[worker];
  return checksum;
}

static double monotonic_seconds(void) {
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
    fail("clock_gettime(CLOCK_MONOTONIC) failed");
  return (double)now.tv_sec + (double)now.tv_nsec / 1000000000.0;
}

static size_t parse_positive(const char *text, const char *option) {
  char *end = NULL;
  errno = 0;
  const unsigned long long parsed = strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed == 0 || parsed > SIZE_MAX) {
    fprintf(stderr, "%s requires a positive integer\n", option);
    exit(2);
  }
  return (size_t)parsed;
}

static corpus read_vectors(const char *path) {
  FILE *file = fopen(path, "rb");
  if (file == NULL) fail("cannot open MWMTV1 vector file");
  if (fseek(file, 0, SEEK_END) != 0) fail("cannot seek vector file");
  const long end = ftell(file);
  if (end < 0) fail("cannot size vector file");
  rewind(file);
  unsigned char *raw = malloc((size_t)end);
  if (raw == NULL) fail("cannot allocate vector file buffer");
  if (fread(raw, 1, (size_t)end, file) != (size_t)end || fclose(file) != 0)
    fail("cannot read vector file");
  if ((size_t)end < header_bytes || memcmp(raw, "MWMTV1\0\0", 8) != 0 ||
      read_le32(raw + 8) != 1)
    fail("unsupported MWMTV1 vector file");
  const size_t count = read_le32(raw + 12);
  if (count == 0 || count > (SIZE_MAX - header_bytes) / record_bytes ||
      header_bytes + count * record_bytes != (size_t)end)
    fail("invalid MWMTV1 record count or file length");
  corpus result = {
      .raw = raw,
      .length = (size_t)end,
      .count = count,
      .scalar = raw + scalar_offset,
      .invalid = raw + 56,
  };
  return result;
}

int main(int argc, char **argv) {
  const char *vector_path = NULL;
  size_t rounds = 100;
  size_t warmup_rounds = 2;
  size_t workers = 1;
  for (int index = 1; index < argc; ++index) {
    if (strcmp(argv[index], "--vectors") == 0 && index + 1 < argc)
      vector_path = argv[++index];
    else if (strcmp(argv[index], "--rounds") == 0 && index + 1 < argc)
      rounds = parse_positive(argv[++index], "--rounds");
    else if (strcmp(argv[index], "--warmup-rounds") == 0 && index + 1 < argc)
      warmup_rounds = parse_positive(argv[++index], "--warmup-rounds");
    else if (strcmp(argv[index], "--workers") == 0 && index + 1 < argc)
      workers = parse_positive(argv[++index], "--workers");
    else {
      fprintf(stderr,
          "Usage: %s --vectors PATH [--rounds N] [--warmup-rounds N] [--workers N]\n",
          argv[0]);
      return 2;
    }
  }
  if (vector_path == NULL) fail("--vectors is required");

  corpus vectors = read_vectors(vector_path);
  if (workers > vectors.count) fail("--workers exceeds the vector count");
  if (sc_check(vectors.scalar) != 0) fail("MWMTV1 scalar is not a canonical Monero secret scalar");

  unsigned char invalid_out[32] = {0};
  if (original_generate_key_derivation(vectors.invalid, vectors.scalar, invalid_out) != 0)
    fail("historical Ref10 accepted the Dalek-rejected invalid point");

  worker_pool pool = {
      .vectors = &vectors,
      .workers = workers,
      .epoch = 0,
      .completed = 0,
      .stopping = 0,
      .verify = 0,
      .failed = ATOMIC_VAR_INIT(0),
      .checksums = calloc(workers, sizeof(uint64_t)),
  };
  if (pool.checksums == NULL || pthread_mutex_init(&pool.mutex, NULL) != 0 ||
      pthread_cond_init(&pool.work_ready, NULL) != 0 || pthread_cond_init(&pool.work_done, NULL) != 0)
    fail("cannot initialize worker pool");
  pthread_t *threads = calloc(workers, sizeof(pthread_t));
  worker_arg *arguments = calloc(workers, sizeof(worker_arg));
  if (threads == NULL || arguments == NULL) fail("cannot allocate worker pool");
  for (size_t worker = 0; worker < workers; ++worker) {
    arguments[worker] = (worker_arg){ .pool = &pool, .index = worker };
    if (pthread_create(&threads[worker], NULL, worker_main, &arguments[worker]) != 0)
      fail("cannot create worker thread");
  }

  const uint64_t preflight_checksum = pool_run(&pool, 1);
  uint64_t checksum = preflight_checksum;
  for (size_t round = 0; round < warmup_rounds; ++round)
    checksum += pool_run(&pool, 0);

  const double started = monotonic_seconds();
  for (size_t round = 0; round < rounds; ++round)
    checksum += pool_run(&pool, 0);
  const double elapsed = monotonic_seconds() - started;

  pthread_mutex_lock(&pool.mutex);
  pool.stopping = 1;
  pthread_cond_broadcast(&pool.work_ready);
  pthread_mutex_unlock(&pool.mutex);
  for (size_t worker = 0; worker < workers; ++worker)
    pthread_join(threads[worker], NULL);

  const size_t operations = vectors.count * rounds;
  printf("testbench=monero_original_ref10_derivation\n");
  printf("implementation=historical_crypto_ops_generate_key_derivation_pre_e0a84c6e8\n");
  printf("algorithm=ge_frombytes_vartime_then_ge_scalarmult_then_ge_mul8_then_ge_tobytes\n");
  printf("input_contract=deterministic_public_MWMTV1_scalar_and_compressed_points\n");
  printf("scalar_contract=canonical_32_byte_scalar_required_by_historical_sc_check\n");
  printf("points_per_round=%zu\n", vectors.count);
  printf("timed_rounds=%zu\n", rounds);
  printf("warmup_rounds=%zu\n", warmup_rounds);
  printf("workers=%zu\n", workers);
  printf("operations=%zu\n", operations);
  printf("elapsed_seconds=%.9f\n", elapsed);
  printf("original_ref10_derivations_per_second=%.3f\n", (double)operations / elapsed);
  printf("preflight=pass\n");
  printf("invalid_point_contract=historical_ref10_rejects_Dalek_rejected_input\n");
  printf("checksum=%llu\n", (unsigned long long)checksum);

  free((void *)vectors.raw);
  free(threads);
  free(arguments);
  free(pool.checksums);
  return 0;
}
