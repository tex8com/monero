#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define MONERO_CUDA2625_DEVICE __device__ __forceinline__
#define MONERO_CUDA2625_CONSTANT static __device__ __constant__
#define MONERO_CUDA2625_BARRIER() __syncthreads()
#else
#define MONERO_CUDA2625_DEVICE inline
#define MONERO_CUDA2625_CONSTANT inline constexpr
#define MONERO_CUDA2625_BARRIER() ((void)0)
#endif

namespace monero_cuda2625 {

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

MONERO_CUDA2625_DEVICE u32 cuda_min_u32(u32 left, u32 right) {
  return left < right ? left : right;
}

// CUDA C3 candidate for the exact wallet operation D = 8 * a * R.
//
// The point and scalar algorithms remain the byte-checked M5 implementation,
// while field elements use curve25519-dalek's unsigned 10-limb radix-2^25.5
// representation.  Each field element is 40 bytes instead of M5's 128 bytes,
// and multiplication uses ten reduced coefficients instead of a 31-entry
// product.  Additions retain Dalek's documented limb headroom instead of
// immediately running a 64-bit carry chain.  Canonical field comparisons work
// on ten limbs instead of repeatedly packing temporary 32-byte encodings.
// The common view scalar is recoded once per into 64 signed bytes,
// avoiding a private 64-int digit array in every GPU thread.  Keep M9 as the
// frozen baseline.  M11 additionally exposes a three-pass pipeline that
// replaces one projective-Z inversion per record with Montgomery's batch
// inversion: prefix products, one inversion, and reverse products.

struct Parameters { u32 count; u32 batch_chunk_size; };
struct Fe { u32 v[10]; };
struct Point { Fe x; Fe y; Fe z; Fe t; };
// The three representations below mirror the ones used by curve25519-dalek
// for variable-base scalar multiplication.  `Point` is its extended P3
// coordinate form; P2 makes consecutive doublings cheaper, P1P1 is the
// completed intermediate form, and Niels is the cached P3 form for P3 + P.
struct PointP2 { Fe x; Fe y; Fe z; };
struct PointP1P1 { Fe x; Fe y; Fe z; Fe t; };
struct ProjectiveNiels { Fe y_plus_x; Fe y_minus_x; Fe z; Fe t2d; };

MONERO_CUDA2625_CONSTANT u32 FIELD_D_LIMBS[10] = {
  0x35978a3u, 0x0d37284u, 0x3156ebdu, 0x06a0a0eu, 0x001c029u,
  0x179e898u, 0x3a03cbbu, 0x1ce7198u, 0x2e2b6ffu, 0x1480db3u
};

MONERO_CUDA2625_CONSTANT u32 SQRT_M1_LIMBS[10] = {
  0x20ea0b0u, 0x186c9d2u, 0x08f189du, 0x035697fu, 0x0bd0c60u,
  0x1fbd7a7u, 0x2804c9eu, 0x1e16569u, 0x004fc1du, 0x0ae0c92u
};

MONERO_CUDA2625_CONSTANT u8 FIELD_D[32] = {
  0xa3, 0x78, 0x59, 0x13, 0xca, 0x4d, 0xeb, 0x75,
  0xab, 0xd8, 0x41, 0x41, 0x4d, 0x0a, 0x70, 0x00,
  0x98, 0xe8, 0x79, 0x77, 0x79, 0x40, 0xc7, 0x8c,
  0x73, 0xfe, 0x6f, 0x2b, 0xee, 0x6c, 0x03, 0x52
};

MONERO_CUDA2625_CONSTANT u8 SQRT_M1[32] = {
  0xb0, 0xa0, 0x0e, 0x4a, 0x27, 0x1b, 0xee, 0xc4,
  0x78, 0xe4, 0x2f, 0xad, 0x06, 0x18, 0x43, 0x2f,
  0xa7, 0xd7, 0xfb, 0x3d, 0x99, 0x00, 0x4d, 0x2b,
  0x0b, 0xdf, 0xc1, 0x4f, 0x80, 0x24, 0x83, 0x2b
};

// p - 2 and (p + 3) / 8, both little endian. The exponents are public and
// fixed; their loops therefore do not depend on a wallet secret.
MONERO_CUDA2625_CONSTANT u8 INVERSE_EXPONENT[32] = {
  0xeb, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f
};

MONERO_CUDA2625_CONSTANT u8 SQRT_EXPONENT[32] = {
  0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x0f
};

// l, the Edwards25519 base-point order, little endian.
MONERO_CUDA2625_CONSTANT u8 SCALAR_L[32] = {
  0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58,
  0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10
};

MONERO_CUDA2625_DEVICE void fe_zero(Fe &out) {
  for (u32 i = 0; i < 10; ++i) out.v[i] = 0;
}

MONERO_CUDA2625_DEVICE void fe_one(Fe &out) {
  fe_zero(out);
  out.v[0] = 1;
}

MONERO_CUDA2625_DEVICE void fe_copy(Fe &out, const Fe &input) {
  for (u32 i = 0; i < 10; ++i) out.v[i] = input.v[i];
}

MONERO_CUDA2625_DEVICE void fe_from_field_d(Fe &out) {
  for (u32 i = 0; i < 10; ++i) out.v[i] = FIELD_D_LIMBS[i];
}

MONERO_CUDA2625_DEVICE void fe_from_sqrt_m1(Fe &out) {
  for (u32 i = 0; i < 10; ++i) out.v[i] = SQRT_M1_LIMBS[i];
}

// Port of curve25519-dalek 4.1.3 FieldElement2625::reduce.  Even limbs
// contain 26 bits and odd limbs 25 bits.  The fixed carry schedule is
// independent of wallet secrets.
MONERO_CUDA2625_DEVICE void fe_reduce_coefficients(Fe &out, u64 *z) {
  constexpr u64 LOW_25 = (1ul << 25) - 1ul;
  constexpr u64 LOW_26 = (1ul << 26) - 1ul;

  z[1] += z[0] >> 26; z[0] &= LOW_26;
  z[5] += z[4] >> 26; z[4] &= LOW_26;
  z[2] += z[1] >> 25; z[1] &= LOW_25;
  z[6] += z[5] >> 25; z[5] &= LOW_25;
  z[3] += z[2] >> 26; z[2] &= LOW_26;
  z[7] += z[6] >> 26; z[6] &= LOW_26;
  z[4] += z[3] >> 25; z[3] &= LOW_25;
  z[8] += z[7] >> 25; z[7] &= LOW_25;
  z[5] += z[4] >> 26; z[4] &= LOW_26;
  z[9] += z[8] >> 26; z[8] &= LOW_26;
  z[0] += 19ul * (z[9] >> 25); z[9] &= LOW_25;
  z[1] += z[0] >> 26; z[0] &= LOW_26;

  for (u32 i = 0; i < 10; ++i) out.v[i] = u32(z[i]);
}

MONERO_CUDA2625_DEVICE void fe_reduce(Fe &value) {
  u64 z[10];
  for (u32 i = 0; i < 10; ++i) z[i] = u64(value.v[i]);
  fe_reduce_coefficients(value, z);
}

MONERO_CUDA2625_DEVICE void fe_add(Fe &out, const Fe &left, const Fe &right) {
  // FieldElement2625::add deliberately does not reduce: a single addition
  // remains within the b < 1.75 headroom accepted by mul/square.  Every point
  // formula consumes this result in a mul/square before another addition.
  for (u32 i = 0; i < 10; ++i) out.v[i] = left.v[i] + right.v[i];
}

MONERO_CUDA2625_DEVICE void fe_sub(Fe &out, const Fe &left, const Fe &right) {
  // Compute left - right as left + 16p - right, matching Dalek's unsigned
  // backend and avoiding all unsigned underflow.
  u64 z[10] = {
    u64(left.v[0]) + u64(0x3ffffedu << 4) - u64(right.v[0]),
    u64(left.v[1]) + u64(0x1ffffffu << 4) - u64(right.v[1]),
    u64(left.v[2]) + u64(0x3ffffffu << 4) - u64(right.v[2]),
    u64(left.v[3]) + u64(0x1ffffffu << 4) - u64(right.v[3]),
    u64(left.v[4]) + u64(0x3ffffffu << 4) - u64(right.v[4]),
    u64(left.v[5]) + u64(0x1ffffffu << 4) - u64(right.v[5]),
    u64(left.v[6]) + u64(0x3ffffffu << 4) - u64(right.v[6]),
    u64(left.v[7]) + u64(0x1ffffffu << 4) - u64(right.v[7]),
    u64(left.v[8]) + u64(0x3ffffffu << 4) - u64(right.v[8]),
    u64(left.v[9]) + u64(0x1ffffffu << 4) - u64(right.v[9])
  };
  fe_reduce_coefficients(out, z);
}

MONERO_CUDA2625_DEVICE void fe_neg(Fe &out, const Fe &input) {
  Fe zero;
  fe_zero(zero);
  fe_sub(out, zero, input);
}

MONERO_CUDA2625_DEVICE void fe_mul(Fe &out, const Fe &left, const Fe &right) {
  const u32 *x = left.v;
  const u32 *y = right.v;
  const u32 y1_19 = 19u * y[1], y2_19 = 19u * y[2];
  const u32 y3_19 = 19u * y[3], y4_19 = 19u * y[4];
  const u32 y5_19 = 19u * y[5], y6_19 = 19u * y[6];
  const u32 y7_19 = 19u * y[7], y8_19 = 19u * y[8];
  const u32 y9_19 = 19u * y[9];
  const u32 x1_2 = 2u * x[1], x3_2 = 2u * x[3];
  const u32 x5_2 = 2u * x[5], x7_2 = 2u * x[7];
  const u32 x9_2 = 2u * x[9];

#define M2625(a, b) (u64(a) * u64(b))
  u64 z[10];
  z[0] = M2625(x[0],y[0]) + M2625(x1_2,y9_19) + M2625(x[2],y8_19) + M2625(x3_2,y7_19) + M2625(x[4],y6_19) + M2625(x5_2,y5_19) + M2625(x[6],y4_19) + M2625(x7_2,y3_19) + M2625(x[8],y2_19) + M2625(x9_2,y1_19);
  z[1] = M2625(x[0],y[1]) + M2625(x[1],y[0]) + M2625(x[2],y9_19) + M2625(x[3],y8_19) + M2625(x[4],y7_19) + M2625(x[5],y6_19) + M2625(x[6],y5_19) + M2625(x[7],y4_19) + M2625(x[8],y3_19) + M2625(x[9],y2_19);
  z[2] = M2625(x[0],y[2]) + M2625(x1_2,y[1]) + M2625(x[2],y[0]) + M2625(x3_2,y9_19) + M2625(x[4],y8_19) + M2625(x5_2,y7_19) + M2625(x[6],y6_19) + M2625(x7_2,y5_19) + M2625(x[8],y4_19) + M2625(x9_2,y3_19);
  z[3] = M2625(x[0],y[3]) + M2625(x[1],y[2]) + M2625(x[2],y[1]) + M2625(x[3],y[0]) + M2625(x[4],y9_19) + M2625(x[5],y8_19) + M2625(x[6],y7_19) + M2625(x[7],y6_19) + M2625(x[8],y5_19) + M2625(x[9],y4_19);
  z[4] = M2625(x[0],y[4]) + M2625(x1_2,y[3]) + M2625(x[2],y[2]) + M2625(x3_2,y[1]) + M2625(x[4],y[0]) + M2625(x5_2,y9_19) + M2625(x[6],y8_19) + M2625(x7_2,y7_19) + M2625(x[8],y6_19) + M2625(x9_2,y5_19);
  z[5] = M2625(x[0],y[5]) + M2625(x[1],y[4]) + M2625(x[2],y[3]) + M2625(x[3],y[2]) + M2625(x[4],y[1]) + M2625(x[5],y[0]) + M2625(x[6],y9_19) + M2625(x[7],y8_19) + M2625(x[8],y7_19) + M2625(x[9],y6_19);
  z[6] = M2625(x[0],y[6]) + M2625(x1_2,y[5]) + M2625(x[2],y[4]) + M2625(x3_2,y[3]) + M2625(x[4],y[2]) + M2625(x5_2,y[1]) + M2625(x[6],y[0]) + M2625(x7_2,y9_19) + M2625(x[8],y8_19) + M2625(x9_2,y7_19);
  z[7] = M2625(x[0],y[7]) + M2625(x[1],y[6]) + M2625(x[2],y[5]) + M2625(x[3],y[4]) + M2625(x[4],y[3]) + M2625(x[5],y[2]) + M2625(x[6],y[1]) + M2625(x[7],y[0]) + M2625(x[8],y9_19) + M2625(x[9],y8_19);
  z[8] = M2625(x[0],y[8]) + M2625(x1_2,y[7]) + M2625(x[2],y[6]) + M2625(x3_2,y[5]) + M2625(x[4],y[4]) + M2625(x5_2,y[3]) + M2625(x[6],y[2]) + M2625(x7_2,y[1]) + M2625(x[8],y[0]) + M2625(x9_2,y9_19);
  z[9] = M2625(x[0],y[9]) + M2625(x[1],y[8]) + M2625(x[2],y[7]) + M2625(x[3],y[6]) + M2625(x[4],y[5]) + M2625(x[5],y[4]) + M2625(x[6],y[3]) + M2625(x[7],y[2]) + M2625(x[8],y[1]) + M2625(x[9],y[0]);
  fe_reduce_coefficients(out, z);
#undef M2625
}

MONERO_CUDA2625_DEVICE void fe_square(Fe &out, const Fe &input) {
  const u32 *x = input.v;
  const u32 x0_2=2u*x[0], x1_2=2u*x[1], x2_2=2u*x[2], x3_2=2u*x[3];
  const u32 x4_2=2u*x[4], x5_2=2u*x[5], x6_2=2u*x[6], x7_2=2u*x[7];
  const u32 x5_19=19u*x[5], x6_19=19u*x[6], x7_19=19u*x[7];
  const u32 x8_19=19u*x[8], x9_19=19u*x[9];
#define S2625(a, b) (u64(a) * u64(b))
  u64 z[10];
  z[0] = S2625(x[0],x[0]) + S2625(x2_2,x8_19) + S2625(x4_2,x6_19) + (S2625(x1_2,x9_19) + S2625(x3_2,x7_19) + S2625(x[5],x5_19))*2ul;
  z[1] = S2625(x0_2,x[1]) + S2625(x3_2,x8_19) + S2625(x5_2,x6_19) + (S2625(x[2],x9_19) + S2625(x[4],x7_19))*2ul;
  z[2] = S2625(x0_2,x[2]) + S2625(x1_2,x[1]) + S2625(x4_2,x8_19) + S2625(x[6],x6_19) + (S2625(x3_2,x9_19) + S2625(x5_2,x7_19))*2ul;
  z[3] = S2625(x0_2,x[3]) + S2625(x1_2,x[2]) + S2625(x5_2,x8_19) + (S2625(x[4],x9_19) + S2625(x[6],x7_19))*2ul;
  z[4] = S2625(x0_2,x[4]) + S2625(x1_2,x3_2) + S2625(x[2],x[2]) + S2625(x6_2,x8_19) + (S2625(x5_2,x9_19) + S2625(x[7],x7_19))*2ul;
  z[5] = S2625(x0_2,x[5]) + S2625(x1_2,x[4]) + S2625(x2_2,x[3]) + S2625(x7_2,x8_19) + S2625(x[6],x9_19)*2ul;
  z[6] = S2625(x0_2,x[6]) + S2625(x1_2,x5_2) + S2625(x2_2,x[4]) + S2625(x3_2,x[3]) + S2625(x[8],x8_19) + S2625(x7_2,x9_19)*2ul;
  z[7] = S2625(x0_2,x[7]) + S2625(x1_2,x[6]) + S2625(x2_2,x[5]) + S2625(x3_2,x[4]) + S2625(x[8],x9_19)*2ul;
  z[8] = S2625(x0_2,x[8]) + S2625(x1_2,x7_2) + S2625(x2_2,x[6]) + S2625(x3_2,x5_2) + S2625(x[4],x[4]) + S2625(x[9],x9_19)*2ul;
  z[9] = S2625(x0_2,x[9]) + S2625(x1_2,x[8]) + S2625(x2_2,x[7]) + S2625(x3_2,x[6]) + S2625(x4_2,x[5]);
  fe_reduce_coefficients(out, z);
#undef S2625
}

MONERO_CUDA2625_DEVICE void fe_freeze(Fe &out, const Fe &input) {
  fe_copy(out, input);
  fe_reduce(out);

  u32 q = (out.v[0] + 19u) >> 26;
  q = (out.v[1] + q) >> 25; q = (out.v[2] + q) >> 26;
  q = (out.v[3] + q) >> 25; q = (out.v[4] + q) >> 26;
  q = (out.v[5] + q) >> 25; q = (out.v[6] + q) >> 26;
  q = (out.v[7] + q) >> 25; q = (out.v[8] + q) >> 26;
  q = (out.v[9] + q) >> 25;

  out.v[0] += 19u * q;
  out.v[1] += out.v[0] >> 26; out.v[0] &= (1u << 26) - 1u;
  out.v[2] += out.v[1] >> 25; out.v[1] &= (1u << 25) - 1u;
  out.v[3] += out.v[2] >> 26; out.v[2] &= (1u << 26) - 1u;
  out.v[4] += out.v[3] >> 25; out.v[3] &= (1u << 25) - 1u;
  out.v[5] += out.v[4] >> 26; out.v[4] &= (1u << 26) - 1u;
  out.v[6] += out.v[5] >> 25; out.v[5] &= (1u << 25) - 1u;
  out.v[7] += out.v[6] >> 26; out.v[6] &= (1u << 26) - 1u;
  out.v[8] += out.v[7] >> 25; out.v[7] &= (1u << 25) - 1u;
  out.v[9] += out.v[8] >> 26; out.v[8] &= (1u << 26) - 1u;
  out.v[9] &= (1u << 25) - 1u;
}

MONERO_CUDA2625_DEVICE bool fe_equal(const Fe &left_input, const Fe &right_input) {
  Fe left, right;
  fe_freeze(left, left_input);
  fe_freeze(right, right_input);
  u32 difference = 0;
  for (u32 i = 0; i < 10; ++i) difference |= left.v[i] ^ right.v[i];
  return difference == 0u;
}

MONERO_CUDA2625_DEVICE bool fe_is_zero(const Fe &input) {
  Fe value;
  fe_freeze(value, input);
  u32 aggregate = 0;
  for (u32 i = 0; i < 10; ++i) aggregate |= value.v[i];
  return aggregate == 0u;
}

MONERO_CUDA2625_DEVICE bool fe_is_negative(const Fe &input) {
  Fe value;
  fe_freeze(value, input);
  return (value.v[0] & 1u) != 0u;
}

MONERO_CUDA2625_DEVICE void fe_from_bytes(Fe &out, const u8 *bytes) {
#define LOAD3_T(i) (u64(bytes[i]) | (u64(bytes[(i)+1]) << 8) | (u64(bytes[(i)+2]) << 16))
#define LOAD4_T(i) (LOAD3_T(i) | (u64(bytes[(i)+3]) << 24))
  u64 z[10] = {
    LOAD4_T(0), LOAD3_T(4) << 6, LOAD3_T(7) << 5,
    LOAD3_T(10) << 3, LOAD3_T(13) << 2, LOAD4_T(16),
    LOAD3_T(20) << 7, LOAD3_T(23) << 5, LOAD3_T(26) << 4,
    (LOAD3_T(29) & ((1ul << 23) - 1ul)) << 2
  };
  fe_reduce_coefficients(out, z);
#undef LOAD3_T
#undef LOAD4_T
}

MONERO_CUDA2625_DEVICE void fe_from_constant(Fe &out, const u8 *bytes) {
#define LOAD3_C(i) (u64(bytes[i]) | (u64(bytes[(i)+1]) << 8) | (u64(bytes[(i)+2]) << 16))
#define LOAD4_C(i) (LOAD3_C(i) | (u64(bytes[(i)+3]) << 24))
  u64 z[10] = {
    LOAD4_C(0), LOAD3_C(4) << 6, LOAD3_C(7) << 5,
    LOAD3_C(10) << 3, LOAD3_C(13) << 2, LOAD4_C(16),
    LOAD3_C(20) << 7, LOAD3_C(23) << 5, LOAD3_C(26) << 4,
    (LOAD3_C(29) & ((1ul << 23) - 1ul)) << 2
  };
  fe_reduce_coefficients(out, z);
#undef LOAD3_C
#undef LOAD4_C
}

MONERO_CUDA2625_DEVICE void fe_to_bytes(u8 *out, const Fe &input) {
  Fe h;
  fe_freeze(h, input);
  out[0]=u8(h.v[0]); out[1]=u8(h.v[0]>>8); out[2]=u8(h.v[0]>>16);
  out[3]=u8((h.v[0]>>24)|(h.v[1]<<2)); out[4]=u8(h.v[1]>>6);
  out[5]=u8(h.v[1]>>14); out[6]=u8((h.v[1]>>22)|(h.v[2]<<3));
  out[7]=u8(h.v[2]>>5); out[8]=u8(h.v[2]>>13);
  out[9]=u8((h.v[2]>>21)|(h.v[3]<<5)); out[10]=u8(h.v[3]>>3);
  out[11]=u8(h.v[3]>>11); out[12]=u8((h.v[3]>>19)|(h.v[4]<<6));
  out[13]=u8(h.v[4]>>2); out[14]=u8(h.v[4]>>10); out[15]=u8(h.v[4]>>18);
  out[16]=u8(h.v[5]); out[17]=u8(h.v[5]>>8); out[18]=u8(h.v[5]>>16);
  out[19]=u8((h.v[5]>>24)|(h.v[6]<<1)); out[20]=u8(h.v[6]>>7);
  out[21]=u8(h.v[6]>>15); out[22]=u8((h.v[6]>>23)|(h.v[7]<<3));
  out[23]=u8(h.v[7]>>5); out[24]=u8(h.v[7]>>13);
  out[25]=u8((h.v[7]>>21)|(h.v[8]<<4)); out[26]=u8(h.v[8]>>4);
  out[27]=u8(h.v[8]>>12); out[28]=u8((h.v[8]>>20)|(h.v[9]<<6));
  out[29]=u8(h.v[9]>>2); out[30]=u8(h.v[9]>>10); out[31]=u8(h.v[9]>>18);
}

MONERO_CUDA2625_DEVICE void fe_pow(
    Fe &out,
    const Fe &input,
    const u8 *exponent,
    int highest_bit) {
  Fe accumulator, square;
  fe_one(accumulator);
  for (int bit = highest_bit; bit >= 0; --bit) {
    fe_square(square, accumulator);
    fe_copy(accumulator, square);
    if (((exponent[u32(bit) >> 3] >> (u32(bit) & 7u)) & 1u) != 0u) {
      fe_mul(square, accumulator, input);
      fe_copy(accumulator, square);
    }
  }
  fe_copy(out, accumulator);
}

MONERO_CUDA2625_DEVICE void fe_inverse(Fe &out, const Fe &input) {
  fe_pow(out, input, INVERSE_EXPONENT, 254);
}

MONERO_CUDA2625_DEVICE void fe_square_root(Fe &out, const Fe &input) {
  fe_pow(out, input, SQRT_EXPONENT, 251);
}

MONERO_CUDA2625_DEVICE bool encoded_y_is_canonical(const u8 *bytes) {
  const u8 top = bytes[31] & 127u;
  if (top != 127u) return top < 127u;
  for (int i = 30; i >= 1; --i)
    if (bytes[i] != 255u) return bytes[i] < 255u;
  return bytes[0] < 237u;
}

MONERO_CUDA2625_DEVICE void point_identity(Point &out) {
  fe_zero(out.x);
  fe_one(out.y);
  fe_one(out.z);
  fe_zero(out.t);
}

MONERO_CUDA2625_DEVICE void point_copy(Point &out, const Point &input) {
  fe_copy(out.x, input.x);
  fe_copy(out.y, input.y);
  fe_copy(out.z, input.z);
  fe_copy(out.t, input.t);
}

// Complete extended-Edwards addition formula for a = -1.
MONERO_CUDA2625_DEVICE void point_add(Point &out, const Point &left, const Point &right) {
  Fe a, b, c, d, e, f, g, h, tmp, curve_d;
  fe_sub(tmp, left.y, left.x);
  fe_sub(a, right.y, right.x);
  fe_mul(a, tmp, a);
  fe_add(tmp, left.y, left.x);
  fe_add(b, right.y, right.x);
  fe_mul(b, tmp, b);
  fe_from_field_d(curve_d);
  fe_mul(c, left.t, right.t);
  fe_mul(c, c, curve_d);
  fe_add(c, c, c);
  fe_mul(d, left.z, right.z);
  fe_add(d, d, d);
  fe_sub(e, b, a);
  fe_sub(f, d, c);
  fe_add(g, d, c);
  fe_add(h, b, a);
  fe_mul(out.x, e, f);
  fe_mul(out.y, g, h);
  fe_mul(out.t, e, h);
  fe_mul(out.z, f, g);
}

MONERO_CUDA2625_DEVICE void point_double(Point &out, const Point &input) {
  Fe a, b, c, d, e, f, g, h, tmp;
  fe_square(a, input.x);
  fe_square(b, input.y);
  fe_square(c, input.z);
  fe_add(c, c, c);
  fe_neg(d, a);
  fe_add(tmp, input.x, input.y);
  fe_square(e, tmp);
  fe_sub(e, e, a);
  fe_sub(e, e, b);
  fe_add(g, d, b);
  fe_sub(f, g, c);
  fe_sub(h, d, b);
  fe_mul(out.x, e, f);
  fe_mul(out.y, g, h);
  fe_mul(out.t, e, h);
  fe_mul(out.z, f, g);
}

MONERO_CUDA2625_DEVICE void point_cmov(Point &target, const Point &source, u32 choose_source) {
  const u64 mask = 0ul - u64(choose_source & 1u);
  for (u32 i = 0; i < 10; ++i) {
    target.x.v[i] = (target.x.v[i] & ~mask) | (source.x.v[i] & mask);
    target.y.v[i] = (target.y.v[i] & ~mask) | (source.y.v[i] & mask);
    target.z.v[i] = (target.z.v[i] & ~mask) | (source.z.v[i] & mask);
    target.t.v[i] = (target.t.v[i] & ~mask) | (source.t.v[i] & mask);
  }
}

MONERO_CUDA2625_DEVICE bool point_from_compressed(Point &out, const u8 *bytes) {
  if (!encoded_y_is_canonical(bytes)) return false;
  Fe y_squared, numerator, denominator, inverse, x_squared, x, check, sqrt_m1;
  fe_from_bytes(out.y, bytes);
  fe_one(out.z);
  fe_square(y_squared, out.y);
  fe_sub(numerator, y_squared, out.z);
  fe_from_field_d(sqrt_m1);
  fe_mul(denominator, y_squared, sqrt_m1);
  fe_add(denominator, denominator, out.z);
  fe_inverse(inverse, denominator);
  fe_mul(x_squared, numerator, inverse);
  fe_square_root(x, x_squared);
  fe_square(check, x);
  if (!fe_equal(check, x_squared)) {
    fe_from_sqrt_m1(sqrt_m1);
    fe_mul(x, x, sqrt_m1);
    fe_square(check, x);
    if (!fe_equal(check, x_squared)) return false;
  }
  const bool requested_negative = (bytes[31] >> 7) != 0u;
  if (fe_is_negative(x) != requested_negative) {
    if (fe_is_zero(x)) return false;
    fe_neg(x, x);
  }
  fe_copy(out.x, x);
  fe_mul(out.t, out.x, out.y);
  return true;
}

MONERO_CUDA2625_DEVICE void point_to_compressed(u8 *out, const Point &input) {
  Fe inverse, x, y;
  fe_inverse(inverse, input.z);
  fe_mul(x, input.x, inverse);
  fe_mul(y, input.y, inverse);
  fe_to_bytes(out, y);
  out[31] ^= u8(fe_is_negative(x) ? 128u : 0u);
}

MONERO_CUDA2625_DEVICE void scalar_conditional_subtract_l(u8 *scalar) {
  u32 difference[32];
  u32 borrow = 0;
  for (u32 i = 0; i < 32; ++i) {
    const u32 subtrahend = u32(SCALAR_L[i]) + borrow;
    const u32 value = u32(scalar[i]);
    difference[i] = (value - subtrahend) & 255u;
    borrow = u32(value < subtrahend);
  }
  const u32 mask = 0u - (borrow ^ 1u);
  for (u32 i = 0; i < 32; ++i)
    scalar[i] = u8((u32(scalar[i]) & ~mask) | (difference[i] & mask));
}

// Exactly Scalar::from_bytes_mod_order for a 32-byte little-endian input.
MONERO_CUDA2625_DEVICE void scalar_reduce_mod_l(u8 *out, const u8 *input) {
  for (u32 i = 0; i < 32; ++i) out[i] = 0;
  for (int bit = 255; bit >= 0; --bit) {
    u32 carry = u32((input[u32(bit) >> 3] >> (u32(bit) & 7u)) & 1u);
    for (u32 i = 0; i < 32; ++i) {
      const u32 value = (u32(out[i]) << 1) | carry;
      out[i] = u8(value & 255u);
      carry = value >> 8;
    }
    scalar_conditional_subtract_l(out);
  }
}

MONERO_CUDA2625_DEVICE void scalar_multiply_by_eight_mod_l(u8 *scalar) {
  u32 carry = 0;
  for (u32 i = 0; i < 32; ++i) {
    const u32 value = (u32(scalar[i]) << 3) | carry;
    scalar[i] = u8(value & 255u);
    carry = value >> 8;
  }
  // The unreduced value is below 8l, hence at most seven subtractions.
  for (u32 pass = 0; pass < 8; ++pass) scalar_conditional_subtract_l(scalar);
}

MONERO_CUDA2625_DEVICE void point_scalar_multiply(Point &out, const Point &point, const u8 *scalar) {
  Point accumulator, doubled, added;
  point_identity(accumulator);
  // Fixed 256 iterations. `point_cmov` avoids a secret-dependent branch.
  for (int bit = 255; bit >= 0; --bit) {
    point_double(doubled, accumulator);
    point_copy(accumulator, doubled);
    point_add(added, accumulator, point);
    const u32 select_added = u32((scalar[u32(bit) >> 3] >> (u32(bit) & 7u)) & 1u);
    point_cmov(accumulator, added, select_added);
  }
  point_copy(out, accumulator);
}

MONERO_CUDA2625_DEVICE void derivation_m1_reference(
    const u8 *view_scalar,
    const u8 *points,
    u8 *results,
    u8 *valid,
    Parameters &parameters,
    u32 id) {
  if (id >= parameters.count) return;
  const u32 offset = id * 32u;
  u8 encoded_point[32];
  u8 result[32];
  for (u32 i = 0; i < 32; ++i) encoded_point[i] = points[offset + i];
  Point point, derived;
  if (!point_from_compressed(point, encoded_point)) {
    for (u32 i = 0; i < 32; ++i) results[offset + i] = 0;
    valid[id] = 0;
    return;
  }
  u8 scalar[32];
  scalar_reduce_mod_l(scalar, view_scalar);
  scalar_multiply_by_eight_mod_l(scalar);
  point_scalar_multiply(derived, point, scalar);
  point_to_compressed(result, derived);
  for (u32 i = 0; i < 32; ++i) results[offset + i] = result[i];
  valid[id] = 1;
}

// M2 keeps the exact M1 math but eliminates redundant work: the view scalar
// is common to the whole wallet scan, so one lane reduces and folds it once
// per threadgroup. Every lane then copies the same 32-byte folded scalar into
// private memory for the fixed-length scalar multiplication.
MONERO_CUDA2625_DEVICE void derivation_m2_group_scalar(
    const u8 *view_scalar,
    const u8 *points,
    u8 *results,
    u8 *valid,
    Parameters &parameters,
    u32 id,
    u32 lane) {
  u8 folded_scalar[32];
  if (lane == 0u) {
    u8 local_scalar[32];
    scalar_reduce_mod_l(local_scalar, view_scalar);
    scalar_multiply_by_eight_mod_l(local_scalar);
    for (u32 i = 0; i < 32; ++i) folded_scalar[i] = local_scalar[i];
  }
  MONERO_CUDA2625_BARRIER();
  if (id >= parameters.count) return;
  const u32 offset = id * 32u;
  u8 encoded_point[32];
  u8 result[32];
  u8 scalar[32];
  for (u32 i = 0; i < 32; ++i) {
    encoded_point[i] = points[offset + i];
    scalar[i] = folded_scalar[i];
  }
  Point point, derived;
  if (!point_from_compressed(point, encoded_point)) {
    for (u32 i = 0; i < 32; ++i) results[offset + i] = 0;
    valid[id] = 0;
    return;
  }
  point_scalar_multiply(derived, point, scalar);
  point_to_compressed(result, derived);
  for (u32 i = 0; i < 32; ++i) results[offset + i] = result[i];
  valid[id] = 1;
}

// M4 ports the structure of curve25519-dalek's variable-base scalar
// multiplication.  It is deliberately separate from M1/M2 so those reference
// kernels remain available for direct byte-for-byte comparison.  The scalar
// has already been reduced and multiplied by eight, so it is < 2^255 and can
// be represented by 64 signed radix-16 digits in [-8, 8].
//
// This removes 192 of the 256 point additions in M1/M2.  It also uses P2 for
// consecutive doublings and Projective-Niels coordinates for the window-table
// additions.  Its mathematical control flow follows Dalek 4.1.3
// backend/serial/scalar_mul/variable_base.rs.

MONERO_CUDA2625_DEVICE void p2_from_p1p1(PointP2 &out, const PointP1P1 &input) {
  fe_mul(out.x, input.x, input.t);
  fe_mul(out.y, input.y, input.z);
  fe_mul(out.z, input.z, input.t);
}

MONERO_CUDA2625_DEVICE void p3_from_p1p1(Point &out, const PointP1P1 &input) {
  fe_mul(out.x, input.x, input.t);
  fe_mul(out.y, input.y, input.z);
  fe_mul(out.z, input.z, input.t);
  fe_mul(out.t, input.x, input.y);
}

MONERO_CUDA2625_DEVICE void p3_from_p2(Point &out, const PointP2 &input) {
  fe_copy(out.x, input.x);
  fe_copy(out.y, input.y);
  fe_copy(out.z, input.z);
  fe_mul(out.t, input.x, input.y);
}

MONERO_CUDA2625_DEVICE void p2_double(PointP1P1 &out, const PointP2 &input) {
  Fe xx, yy, zz2, x_plus_y, x_plus_y_sq, yy_plus_xx, yy_minus_xx;
  fe_square(xx, input.x);
  fe_square(yy, input.y);
  fe_square(zz2, input.z);
  fe_add(zz2, zz2, zz2);
  fe_add(x_plus_y, input.x, input.y);
  fe_square(x_plus_y_sq, x_plus_y);
  fe_add(yy_plus_xx, yy, xx);
  fe_sub(yy_minus_xx, yy, xx);
  fe_sub(out.x, x_plus_y_sq, yy_plus_xx);
  fe_copy(out.y, yy_plus_xx);
  fe_copy(out.z, yy_minus_xx);
  fe_sub(out.t, zz2, yy_minus_xx);
}

MONERO_CUDA2625_DEVICE void niels_identity(ProjectiveNiels &out) {
  fe_one(out.y_plus_x);
  fe_one(out.y_minus_x);
  fe_one(out.z);
  fe_zero(out.t2d);
}

MONERO_CUDA2625_DEVICE void niels_from_p3(ProjectiveNiels &out, const Point &input) {
  Fe curve_d;
  fe_add(out.y_plus_x, input.y, input.x);
  fe_sub(out.y_minus_x, input.y, input.x);
  fe_copy(out.z, input.z);
  fe_from_field_d(curve_d);
  fe_mul(out.t2d, input.t, curve_d);
  fe_add(out.t2d, out.t2d, out.t2d);
}

MONERO_CUDA2625_DEVICE void niels_cmov(
    ProjectiveNiels &target,
    const ProjectiveNiels &source,
    u32 choose_source) {
  const u32 mask = 0u - (choose_source & 1u);
  for (u32 i = 0; i < 10; ++i) {
    target.y_plus_x.v[i] = (target.y_plus_x.v[i] & ~mask) | (source.y_plus_x.v[i] & mask);
    target.y_minus_x.v[i] = (target.y_minus_x.v[i] & ~mask) | (source.y_minus_x.v[i] & mask);
    target.z.v[i] = (target.z.v[i] & ~mask) | (source.z.v[i] & mask);
    target.t2d.v[i] = (target.t2d.v[i] & ~mask) | (source.t2d.v[i] & mask);
  }
}

MONERO_CUDA2625_DEVICE u32 ct_equal_u32(u32 left, u32 right) {
  const u32 difference = left ^ right;
  return ((difference | (0u - difference)) >> 31u) ^ 1u;
}

MONERO_CUDA2625_DEVICE void niels_select_signed(
    ProjectiveNiels &out,
    const ProjectiveNiels *table,
    int digit) {
  // `digit` is in [-8, 8].  This is the same constant-time selection shape
  // as Dalek's LookupTable::select: select |digit| then conditionally negate.
  const int sign_mask = digit >> 31;
  const u32 absolute = u32((digit + sign_mask) ^ sign_mask);
  niels_identity(out);
  for (u32 value = 1; value <= 8; ++value)
    niels_cmov(out, table[value - 1u], ct_equal_u32(absolute, value));

  ProjectiveNiels negated;
  fe_copy(negated.y_plus_x, out.y_minus_x);
  fe_copy(negated.y_minus_x, out.y_plus_x);
  fe_copy(negated.z, out.z);
  fe_neg(negated.t2d, out.t2d);
  niels_cmov(out, negated, u32(sign_mask) & 1u);
}

MONERO_CUDA2625_DEVICE void p3_add_niels(
    PointP1P1 &out,
    const Point &left,
    const ProjectiveNiels &right) {
  Fe y_plus_x, y_minus_x, pp, mm, tt2d, zz, zz2;
  fe_add(y_plus_x, left.y, left.x);
  fe_sub(y_minus_x, left.y, left.x);
  fe_mul(pp, y_plus_x, right.y_plus_x);
  fe_mul(mm, y_minus_x, right.y_minus_x);
  fe_mul(tt2d, left.t, right.t2d);
  fe_mul(zz, left.z, right.z);
  fe_add(zz2, zz, zz);
  fe_sub(out.x, pp, mm);
  fe_add(out.y, pp, mm);
  fe_add(out.z, zz2, tt2d);
  fe_sub(out.t, zz2, tt2d);
}

MONERO_CUDA2625_DEVICE void scalar_as_radix_16(int *digits, const u8 *scalar) {
  for (u32 i = 0; i < 32; ++i) {
    digits[2u * i] = int(scalar[i] & 15u);
    digits[2u * i + 1u] = int(scalar[i] >> 4u);
  }
  for (u32 i = 0; i < 63; ++i) {
    const int carry = (digits[i] + 8) >> 4;
    digits[i] -= carry << 4;
    digits[i + 1u] += carry;
  }
}

MONERO_CUDA2625_DEVICE void point_scalar_multiply_radix16(
    Point &out,
    const Point &point,
    const u8 *scalar) {
  ProjectiveNiels table[8];
  niels_from_p3(table[0], point);
  for (u32 i = 0; i < 7; ++i) {
    PointP1P1 sum;
    Point next;
    p3_add_niels(sum, point, table[i]);
    p3_from_p1p1(next, sum);
    niels_from_p3(table[i + 1u], next);
  }

  int digits[64];
  scalar_as_radix_16(digits, scalar);

  Point accumulator;
  PointP1P1 completed;
  ProjectiveNiels selected;
  point_identity(accumulator);
  niels_select_signed(selected, table, digits[63]);
  p3_add_niels(completed, accumulator, selected);

  for (int i = 62; i >= 0; --i) {
    PointP2 projective;
    p2_from_p1p1(projective, completed);
    p2_double(completed, projective);
    p2_from_p1p1(projective, completed);
    p2_double(completed, projective);
    p2_from_p1p1(projective, completed);
    p2_double(completed, projective);
    p2_from_p1p1(projective, completed);
    p2_double(completed, projective);
    p3_from_p1p1(accumulator, completed);
    niels_select_signed(selected, table, digits[i]);
    p3_add_niels(completed, accumulator, selected);
  }
  p3_from_p1p1(out, completed);
}

MONERO_CUDA2625_DEVICE void scalar_as_radix_16_group(
    char *digits,
    const u8 *scalar) {
  for (u32 i = 0; i < 32; ++i) {
    digits[2u * i] = char(scalar[i] & 15u);
    digits[2u * i + 1u] = char(scalar[i] >> 4u);
  }
  for (u32 i = 0; i < 63; ++i) {
    const int digit = int(digits[i]);
    const int carry = (digit + 8) >> 4;
    digits[i] = char(digit - (carry << 4));
    digits[i + 1u] = char(int(digits[i + 1u]) + carry);
  }
}

MONERO_CUDA2625_DEVICE void point_scalar_multiply_radix16_group(
    Point &out,
    const Point &point,
    const char *digits) {
  ProjectiveNiels table[8];
  niels_from_p3(table[0], point);
  for (u32 i = 0; i < 7; ++i) {
    PointP1P1 sum;
    Point next;
    p3_add_niels(sum, point, table[i]);
    p3_from_p1p1(next, sum);
    niels_from_p3(table[i + 1u], next);
  }

  Point accumulator;
  PointP1P1 completed;
  ProjectiveNiels selected;
  point_identity(accumulator);
  niels_select_signed(selected, table, int(digits[63]));
  p3_add_niels(completed, accumulator, selected);

  for (int i = 62; i >= 0; --i) {
    PointP2 projective;
    p2_from_p1p1(projective, completed);
    p2_double(completed, projective);
    p2_from_p1p1(projective, completed);
    p2_double(completed, projective);
    p2_from_p1p1(projective, completed);
    p2_double(completed, projective);
    p2_from_p1p1(projective, completed);
    p2_double(completed, projective);
    p3_from_p1p1(accumulator, completed);
    niels_select_signed(selected, table, int(digits[i]));
    p3_add_niels(completed, accumulator, selected);
  }
  p3_from_p1p1(out, completed);
}

MONERO_CUDA2625_DEVICE void scalar_as_radix_8_group(
    char *digits, const u8 *scalar) {
  for (u32 index = 0; index < 86; ++index) {
    const u32 bit = index * 3;
    const u32 byte = bit >> 3;
    const u32 shift = bit & 7u;
    u32 value = u32(scalar[byte]) >> shift;
    if (shift > 5u && byte + 1u < 32u)
      value |= u32(scalar[byte + 1u]) << (8u - shift);
    digits[index] = char(value & 7u);
  }
  for (u32 index = 0; index < 85; ++index) {
    const int digit = int(digits[index]);
    const int carry = (digit + 4) >> 3;
    digits[index] = char(digit - (carry << 3));
    digits[index + 1u] =
        char(int(digits[index + 1u]) + carry);
  }
}

MONERO_CUDA2625_DEVICE void scalar_as_radix_4_group(
    char *digits, const u8 *scalar) {
  for (u32 index = 0; index < 128; ++index) {
    const u32 bit = index * 2u;
    digits[index] =
        char((u32(scalar[bit >> 3u]) >> (bit & 7u)) & 3u);
  }
  for (u32 index = 0; index < 127; ++index) {
    const int digit = int(digits[index]);
    const int carry = (digit + 2) >> 2;
    digits[index] = char(digit - (carry << 2));
    digits[index + 1u] =
        char(int(digits[index + 1u]) + carry);
  }
}

MONERO_CUDA2625_DEVICE void niels_select_signed_radix8(
    ProjectiveNiels &out, const ProjectiveNiels *table, int digit) {
  const int sign_mask = digit >> 31;
  const u32 absolute = u32((digit + sign_mask) ^ sign_mask);
  niels_identity(out);
  for (u32 value = 1; value <= 4; ++value)
    niels_cmov(out, table[value - 1u],
               ct_equal_u32(absolute, value));

  ProjectiveNiels negated;
  fe_copy(negated.y_plus_x, out.y_minus_x);
  fe_copy(negated.y_minus_x, out.y_plus_x);
  fe_copy(negated.z, out.z);
  fe_neg(negated.t2d, out.t2d);
  niels_cmov(out, negated, u32(sign_mask) & 1u);
}

MONERO_CUDA2625_DEVICE void niels_select_signed_radix4(
    ProjectiveNiels &out, const ProjectiveNiels *table, int digit) {
  const int sign_mask = digit >> 31;
  const u32 absolute = u32((digit + sign_mask) ^ sign_mask);
  niels_identity(out);
  for (u32 value = 1; value <= 2; ++value)
    niels_cmov(out, table[value - 1u],
               ct_equal_u32(absolute, value));

  ProjectiveNiels negated;
  fe_copy(negated.y_plus_x, out.y_minus_x);
  fe_copy(negated.y_minus_x, out.y_plus_x);
  fe_copy(negated.z, out.z);
  fe_neg(negated.t2d, out.t2d);
  niels_cmov(out, negated, u32(sign_mask) & 1u);
}

MONERO_CUDA2625_DEVICE void point_scalar_multiply_radix8_group(
    Point &out, const Point &point, const char *digits) {
  ProjectiveNiels table[4];
  niels_from_p3(table[0], point);
  for (u32 index = 0; index < 3; ++index) {
    PointP1P1 sum;
    Point next;
    p3_add_niels(sum, point, table[index]);
    p3_from_p1p1(next, sum);
    niels_from_p3(table[index + 1u], next);
  }

  Point accumulator;
  PointP1P1 completed;
  ProjectiveNiels selected;
  point_identity(accumulator);
  niels_select_signed_radix8(selected, table, int(digits[85]));
  p3_add_niels(completed, accumulator, selected);

  for (int index = 84; index >= 0; --index) {
    PointP2 projective;
    p2_from_p1p1(projective, completed);
    p2_double(completed, projective);
    p2_from_p1p1(projective, completed);
    p2_double(completed, projective);
    p2_from_p1p1(projective, completed);
    p2_double(completed, projective);
    p3_from_p1p1(accumulator, completed);
    niels_select_signed_radix8(
        selected, table, int(digits[index]));
    p3_add_niels(completed, accumulator, selected);
  }
  p3_from_p1p1(out, completed);
}

MONERO_CUDA2625_DEVICE void point_scalar_multiply_radix4_group(
    Point &out, const Point &point, const char *digits) {
  ProjectiveNiels table[2];
  niels_from_p3(table[0], point);
  PointP1P1 sum;
  Point twice;
  p3_add_niels(sum, point, table[0]);
  p3_from_p1p1(twice, sum);
  niels_from_p3(table[1], twice);

  Point accumulator;
  PointP1P1 completed;
  ProjectiveNiels selected;
  point_identity(accumulator);
  niels_select_signed_radix4(selected, table, int(digits[127]));
  p3_add_niels(completed, accumulator, selected);

  for (int index = 126; index >= 0; --index) {
    PointP2 projective;
    p2_from_p1p1(projective, completed);
    p2_double(completed, projective);
    p2_from_p1p1(projective, completed);
    p2_double(completed, projective);
    p3_from_p1p1(accumulator, completed);
    niels_select_signed_radix4(
        selected, table, int(digits[index]));
    p3_add_niels(completed, accumulator, selected);
  }
  p3_from_p1p1(out, completed);
}

MONERO_CUDA2625_DEVICE void derivation_m4_radix16_niels(
    const u8 *view_scalar,
    const u8 *points,
    u8 *results,
    u8 *valid,
    Parameters &parameters,
    u32 id,
    u32 lane) {
  // Preserve M2's valid reduction: the view scalar is common to the scan
  // batch, so only one lane per performs the public setup work.
  u8 folded_scalar[32];
  if (lane == 0u) {
    u8 local_scalar[32];
    scalar_reduce_mod_l(local_scalar, view_scalar);
    scalar_multiply_by_eight_mod_l(local_scalar);
    for (u32 i = 0; i < 32; ++i) folded_scalar[i] = local_scalar[i];
  }
  MONERO_CUDA2625_BARRIER();
  if (id >= parameters.count) return;

  const u32 offset = id * 32u;
  u8 encoded_point[32];
  u8 scalar[32];
  u8 result[32];
  for (u32 i = 0; i < 32; ++i) {
    encoded_point[i] = points[offset + i];
    scalar[i] = folded_scalar[i];
  }
  Point point, derived;
  if (!point_from_compressed(point, encoded_point)) {
    for (u32 i = 0; i < 32; ++i) results[offset + i] = 0;
    valid[id] = 0;
    return;
  }
  point_scalar_multiply_radix16(derived, point, scalar);
  point_to_compressed(result, derived);
  for (u32 i = 0; i < 32; ++i) results[offset + i] = result[i];
  valid[id] = 1;
}

// M5 keeps M4's proven scalar-multiplication path but replaces the generic
// binary exponentiation in point decoding/encoding with Dalek's addition
// chains.  In the reference path, a field inverse costs 254 squarings plus
// about 253 full multiplications, and the square-root exponentiation costs
// 252 squarings plus about 251 full multiplications.  The chains below use the
// same exponent identities with 11 and 11 full multiplications respectively.
// `fe_square` remains the already byte-checked reference square operation;
// this is an algorithmic exponentiation change, not a new field reduction.

MONERO_CUDA2625_DEVICE void fe_square_n(Fe &out, const Fe &input, u32 count) {
  Fe accumulator, squared;
  fe_copy(accumulator, input);
  for (u32 i = 0; i < count; ++i) {
    fe_square(squared, accumulator);
    fe_copy(accumulator, squared);
  }
  fe_copy(out, accumulator);
}

// Return x^(2^250 - 1) and x^11.  This is the exact exponentiation chain
// used by Dalek's FieldElement::pow22501().
MONERO_CUDA2625_DEVICE void fe_pow22501(Fe &power_22501, Fe &power_11, const Fe &input) {
  Fe t0, t1, t2, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18;
  fe_square(t0, input);                         // x^2
  fe_square_n(t1, t0, 2u);                      // x^8
  fe_mul(t2, input, t1);                        // x^9
  fe_mul(power_11, t0, t2);                     // x^11
  fe_square(t4, power_11);                      // x^22
  fe_mul(t5, t2, t4);                           // x^(2^5 - 1)
  fe_square_n(t6, t5, 5u);
  fe_mul(t7, t6, t5);                           // x^(2^10 - 1)
  fe_square_n(t8, t7, 10u);
  fe_mul(t9, t8, t7);                           // x^(2^20 - 1)
  fe_square_n(t10, t9, 20u);
  fe_mul(t11, t10, t9);                         // x^(2^40 - 1)
  fe_square_n(t12, t11, 10u);
  fe_mul(t13, t12, t7);                         // x^(2^50 - 1)
  fe_square_n(t14, t13, 50u);
  fe_mul(t15, t14, t13);                        // x^(2^100 - 1)
  fe_square_n(t16, t15, 100u);
  fe_mul(t17, t16, t15);                        // x^(2^200 - 1)
  fe_square_n(t18, t17, 50u);
  fe_mul(power_22501, t18, t13);                // x^(2^250 - 1)
}

MONERO_CUDA2625_DEVICE void fe_inverse_m5(Fe &out, const Fe &input) {
  Fe t19, t3, t20;
  fe_pow22501(t19, t3, input);
  fe_square_n(t20, t19, 5u);
  fe_mul(out, t20, t3);                         // x^(p - 2)
}

MONERO_CUDA2625_DEVICE void fe_square_root_m5(Fe &out, const Fe &input) {
  Fe t19, unused_t3, t20, input_squared;
  fe_pow22501(t19, unused_t3, input);
  fe_square_n(t20, t19, 2u);
  fe_square(input_squared, input);
  fe_mul(out, t20, input_squared);              // x^((p + 3) / 8)
}

MONERO_CUDA2625_DEVICE void fe_cmov(Fe &target, const Fe &source,
                                    u32 choose_source) {
  const u32 mask = 0u - (choose_source & 1u);
  for (u32 limb = 0; limb < 10; ++limb)
    target.v[limb] =
        (target.v[limb] & ~mask) | (source.v[limb] & mask);
}

// Raise x to (p - 5) / 8 = 2^252 - 3. This is Dalek 4.1.3's
// FieldElement::pow_p58 chain.
MONERO_CUDA2625_DEVICE void fe_pow_p58_m6(Fe &out, const Fe &input) {
  Fe power_22501, unused_power_11, squared;
  fe_pow22501(power_22501, unused_power_11, input);
  fe_square_n(squared, power_22501, 2u);
  fe_mul(out, input, squared);
}

// Compute sqrt(u / v) with one exponentiation, merging the inversion, square
// root, and square test exactly as FieldElement::sqrt_ratio_i does in
// curve25519-dalek 4.1.3. The returned root is always nonnegative.
MONERO_CUDA2625_DEVICE bool fe_sqrt_ratio_i_m6(Fe &out, const Fe &u,
                                               const Fe &v) {
  Fe v_squared, v3, v3_squared, v7;
  fe_square(v_squared, v);
  fe_mul(v3, v_squared, v);
  fe_square(v3_squared, v3);
  fe_mul(v7, v3_squared, v);

  Fe u_v3, u_v7, power, root;
  fe_mul(u_v3, u, v3);
  fe_mul(u_v7, u, v7);
  fe_pow_p58_m6(power, u_v7);
  fe_mul(root, u_v3, power);

  Fe root_squared, check, negative_u, sqrt_m1, negative_u_i;
  fe_square(root_squared, root);
  fe_mul(check, v, root_squared);
  fe_neg(negative_u, u);
  fe_from_sqrt_m1(sqrt_m1);
  fe_mul(negative_u_i, negative_u, sqrt_m1);

  const u32 correct_sign_sqrt = fe_equal(check, u) ? 1u : 0u;
  const u32 flipped_sign_sqrt = fe_equal(check, negative_u) ? 1u : 0u;
  const u32 flipped_sign_sqrt_i =
      fe_equal(check, negative_u_i) ? 1u : 0u;

  Fe root_prime;
  fe_mul(root_prime, sqrt_m1, root);
  fe_cmov(root, root_prime,
          flipped_sign_sqrt | flipped_sign_sqrt_i);

  Fe negative_root;
  fe_neg(negative_root, root);
  fe_cmov(root, negative_root, fe_is_negative(root) ? 1u : 0u);
  fe_copy(out, root);
  return (correct_sign_sqrt | flipped_sign_sqrt) != 0u;
}

MONERO_CUDA2625_DEVICE bool point_from_compressed_m6(Point &out,
                                                     const u8 *bytes) {
  if (!encoded_y_is_canonical(bytes)) return false;
  Fe y_squared, numerator, denominator, curve_d, x;
  fe_from_bytes(out.y, bytes);
  fe_one(out.z);
  fe_square(y_squared, out.y);
  fe_sub(numerator, y_squared, out.z);
  fe_from_field_d(curve_d);
  fe_mul(denominator, y_squared, curve_d);
  fe_add(denominator, denominator, out.z);
  if (!fe_sqrt_ratio_i_m6(x, numerator, denominator)) return false;

  const bool requested_negative = (bytes[31] >> 7) != 0u;
  if (fe_is_negative(x) != requested_negative) {
    if (fe_is_zero(x)) return false;
    fe_neg(x, x);
  }
  fe_copy(out.x, x);
  fe_mul(out.t, out.x, out.y);
  return true;
}

MONERO_CUDA2625_DEVICE bool point_from_compressed_m5(Point &out, const u8 *bytes) {
  if (!encoded_y_is_canonical(bytes)) return false;
  Fe y_squared, numerator, denominator, inverse, x_squared, x, check, sqrt_m1;
  fe_from_bytes(out.y, bytes);
  fe_one(out.z);
  fe_square(y_squared, out.y);
  fe_sub(numerator, y_squared, out.z);
  fe_from_field_d(sqrt_m1);
  fe_mul(denominator, y_squared, sqrt_m1);
  fe_add(denominator, denominator, out.z);
  fe_inverse_m5(inverse, denominator);
  fe_mul(x_squared, numerator, inverse);
  fe_square_root_m5(x, x_squared);
  fe_square(check, x);
  if (!fe_equal(check, x_squared)) {
    fe_from_sqrt_m1(sqrt_m1);
    fe_mul(x, x, sqrt_m1);
    fe_square(check, x);
    if (!fe_equal(check, x_squared)) return false;
  }
  const bool requested_negative = (bytes[31] >> 7) != 0u;
  if (fe_is_negative(x) != requested_negative) {
    if (fe_is_zero(x)) return false;
    fe_neg(x, x);
  }
  fe_copy(out.x, x);
  fe_mul(out.t, out.x, out.y);
  return true;
}

MONERO_CUDA2625_DEVICE void point_to_compressed_m5(u8 *out, const Point &input) {
  Fe inverse, x, y;
  fe_inverse_m5(inverse, input.z);
  fe_mul(x, input.x, inverse);
  fe_mul(y, input.y, inverse);
  fe_to_bytes(out, y);
  out[31] ^= u8(fe_is_negative(x) ? 128u : 0u);
}

MONERO_CUDA2625_DEVICE void derivation_m10_radix2625_groupdigits_niels(
    const u8 *view_scalar,
    const u8 *points,
    u8 *results,
    u8 *valid,
    Parameters &parameters,
    u32 id,
    u32 lane) {
  char scalar_digits[64];
  if (lane == 0u) {
    u8 local_scalar[32];
    scalar_reduce_mod_l(local_scalar, view_scalar);
    scalar_multiply_by_eight_mod_l(local_scalar);
    scalar_as_radix_16_group(scalar_digits, local_scalar);
  }
  MONERO_CUDA2625_BARRIER();
  if (id >= parameters.count) return;

  const u32 offset = id * 32u;
  u8 encoded_point[32];
  u8 result[32];
  for (u32 i = 0; i < 32; ++i) encoded_point[i] = points[offset + i];
  Point point, derived;
  if (!point_from_compressed_m5(point, encoded_point)) {
    for (u32 i = 0; i < 32; ++i) results[offset + i] = 0;
    valid[id] = 0;
    return;
  }
  point_scalar_multiply_radix16_group(derived, point, scalar_digits);
  point_to_compressed_m5(result, derived);
  for (u32 i = 0; i < 32; ++i) results[offset + i] = result[i];
  valid[id] = 1;
}

MONERO_CUDA2625_DEVICE void fe_store_device(u32 *destination, const Fe &value) {
  for (u32 i = 0; i < 10; ++i) destination[i] = value.v[i];
}

MONERO_CUDA2625_DEVICE void fe_load_device(Fe &value, const u32 *source) {
  for (u32 i = 0; i < 10; ++i) value.v[i] = source[i];
}

// Pass 1: point decoding and variable-base multiplication remain massively
// parallel.  X/Y/Z are retained in projective form for a shared inversion.
MONERO_CUDA2625_DEVICE void derivation_m12_projective_chunkinvert(
    const u8 *view_scalar,
    const u8 *points,
    u32 *projective,
    u8 *valid,
    Parameters &parameters,
    u32 id,
    u32 lane) {
  char scalar_digits[64];
  if (lane == 0u) {
    u8 local_scalar[32];
    scalar_reduce_mod_l(local_scalar, view_scalar);
    scalar_multiply_by_eight_mod_l(local_scalar);
    scalar_as_radix_16_group(scalar_digits, local_scalar);
  }
  MONERO_CUDA2625_BARRIER();
  if (id >= parameters.count) return;

  const u32 byte_offset = id * 32u;
  const u32 field_offset = id * 30u;
  u8 encoded_point[32];
  for (u32 i = 0; i < 32; ++i) encoded_point[i] = points[byte_offset + i];

  Point point, derived;
  if (!point_from_compressed_m5(point, encoded_point)) {
    for (u32 i = 0; i < 30; ++i) projective[field_offset + i] = 0u;
    valid[id] = 0u;
    return;
  }
  point_scalar_multiply_radix16_group(derived, point, scalar_digits);
  fe_store_device(projective + field_offset, derived.x);
  fe_store_device(projective + field_offset + 10u, derived.y);
  fe_store_device(projective + field_offset + 20u, derived.z);
  valid[id] = 1u;
}

// Pass 2: each GPU performs one independent Montgomery batch inversion
// over a small public-size chunk.  This retains hundreds of parallel threads
// while replacing chunk_size exponentiations with one.
MONERO_CUDA2625_DEVICE void derivation_m12_chunk_inverse(
    const u32 *projective,
    u32 *inverse_z,
    const u8 *valid,
    Parameters &parameters,
    u32 id) {
  const u32 chunk_size = parameters.batch_chunk_size;
  const u32 begin = id * chunk_size;
  if (begin >= parameters.count) return;
  const u32 end = cuda_min_u32(begin + chunk_size, parameters.count);

  Fe accumulator;
  fe_one(accumulator);
  for (u32 record = begin; record < end; ++record) {
    const u32 field_offset = record * 30u;
    const u32 inverse_offset = record * 10u;
    fe_store_device(inverse_z + inverse_offset, accumulator);
    Fe z, next;
    if (valid[record] != 0u)
      fe_load_device(z, projective + field_offset + 20u);
    else
      fe_one(z);
    fe_mul(next, accumulator, z);
    fe_copy(accumulator, next);
  }

  Fe inverse_product;
  fe_inverse_m5(inverse_product, accumulator);
  u32 remaining = end;
  while (remaining != begin) {
    const u32 record = --remaining;
    const u32 field_offset = record * 30u;
    const u32 inverse_offset = record * 10u;
    Fe prefix, z, inverse_record, next;
    fe_load_device(prefix, inverse_z + inverse_offset);
    if (valid[record] != 0u)
      fe_load_device(z, projective + field_offset + 20u);
    else
      fe_one(z);
    fe_mul(inverse_record, inverse_product, prefix);
    if (valid[record] != 0u)
      fe_store_device(inverse_z + inverse_offset, inverse_record);
    else {
      Fe zero;
      fe_zero(zero);
      fe_store_device(inverse_z + inverse_offset, zero);
    }
    fe_mul(next, inverse_product, z);
    fe_copy(inverse_product, next);
  }
}

// Pass 3: affine conversion and the normal canonical compressed-Edwards
// encoding are parallel again.
MONERO_CUDA2625_DEVICE void derivation_m12_compress_chunkinvert(
    const u32 *projective,
    const u32 *inverse_z,
    u8 *results,
    const u8 *valid,
    Parameters &parameters,
    u32 id) {
  if (id >= parameters.count) return;
  const u32 byte_offset = id * 32u;
  if (valid[id] == 0u) {
    for (u32 i = 0; i < 32; ++i) results[byte_offset + i] = 0u;
    return;
  }

  const u32 field_offset = id * 30u;
  const u32 inverse_offset = id * 10u;
  Fe projective_x, projective_y, inverse, x, y;
  fe_load_device(projective_x, projective + field_offset);
  fe_load_device(projective_y, projective + field_offset + 10u);
  fe_load_device(inverse, inverse_z + inverse_offset);
  fe_mul(x, projective_x, inverse);
  fe_mul(y, projective_y, inverse);
  u8 encoded[32];
  fe_to_bytes(encoded, y);
  encoded[31] ^= u8(fe_is_negative(x) ? 128u : 0u);
  for (u32 i = 0; i < 32; ++i) results[byte_offset + i] = encoded[i];
}

}  // namespace monero_cuda2625

#undef MONERO_CUDA2625_DEVICE
#undef MONERO_CUDA2625_CONSTANT
#undef MONERO_CUDA2625_BARRIER
