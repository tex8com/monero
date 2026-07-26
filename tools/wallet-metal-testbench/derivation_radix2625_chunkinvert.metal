// M12 candidate for the exact wallet operation D = 8 * a * R.
//
// The point and scalar algorithms remain the byte-checked M5 implementation,
// while field elements use curve25519-dalek's unsigned 10-limb radix-2^25.5
// representation.  Each field element is 40 bytes instead of M5's 128 bytes,
// and multiplication uses ten reduced coefficients instead of a 31-entry
// product.  Additions retain Dalek's documented limb headroom instead of
// immediately running a 64-bit carry chain.  Canonical field comparisons work
// on ten limbs instead of repeatedly packing temporary 32-byte encodings.
// The common view scalar is recoded once per threadgroup into 64 signed bytes,
// avoiding a private 64-int digit array in every GPU thread.  Keep M9 as the
// frozen baseline.  M11 additionally exposes a three-pass pipeline that
// replaces one projective-Z inversion per record with Montgomery's batch
// inversion: prefix products, one inversion, and reverse products.
#include <metal_stdlib>
using namespace metal;

struct Parameters { uint count; uint batch_chunk_size; };
struct Fe { uint v[10]; };
struct Point { Fe x; Fe y; Fe z; Fe t; };
// The three representations below mirror the ones used by curve25519-dalek
// for variable-base scalar multiplication.  `Point` is its extended P3
// coordinate form; P2 makes consecutive doublings cheaper, P1P1 is the
// completed intermediate form, and Niels is the cached P3 form for P3 + P.
struct PointP2 { Fe x; Fe y; Fe z; };
struct PointP1P1 { Fe x; Fe y; Fe z; Fe t; };
struct ProjectiveNiels { Fe y_plus_x; Fe y_minus_x; Fe z; Fe t2d; };

constant uchar FIELD_D[32] = {
  0xa3, 0x78, 0x59, 0x13, 0xca, 0x4d, 0xeb, 0x75,
  0xab, 0xd8, 0x41, 0x41, 0x4d, 0x0a, 0x70, 0x00,
  0x98, 0xe8, 0x79, 0x77, 0x79, 0x40, 0xc7, 0x8c,
  0x73, 0xfe, 0x6f, 0x2b, 0xee, 0x6c, 0x03, 0x52
};

constant uchar SQRT_M1[32] = {
  0xb0, 0xa0, 0x0e, 0x4a, 0x27, 0x1b, 0xee, 0xc4,
  0x78, 0xe4, 0x2f, 0xad, 0x06, 0x18, 0x43, 0x2f,
  0xa7, 0xd7, 0xfb, 0x3d, 0x99, 0x00, 0x4d, 0x2b,
  0x0b, 0xdf, 0xc1, 0x4f, 0x80, 0x24, 0x83, 0x2b
};

// p - 2 and (p + 3) / 8, both little endian. The exponents are public and
// fixed; their loops therefore do not depend on a wallet secret.
constant uchar INVERSE_EXPONENT[32] = {
  0xeb, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f
};

constant uchar SQRT_EXPONENT[32] = {
  0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x0f
};

// l, the Edwards25519 base-point order, little endian.
constant uchar SCALAR_L[32] = {
  0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58,
  0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10
};

inline void fe_zero(thread Fe &out) {
  for (uint i = 0; i < 10; ++i) out.v[i] = 0;
}

inline void fe_one(thread Fe &out) {
  fe_zero(out);
  out.v[0] = 1;
}

inline void fe_copy(thread Fe &out, thread const Fe &input) {
  for (uint i = 0; i < 10; ++i) out.v[i] = input.v[i];
}

// Port of curve25519-dalek 4.1.3 FieldElement2625::reduce.  Even limbs
// contain 26 bits and odd limbs 25 bits.  The fixed carry schedule is
// independent of wallet secrets.
inline void fe_reduce_coefficients(thread Fe &out, thread ulong *z) {
  constexpr ulong LOW_25 = (1ul << 25) - 1ul;
  constexpr ulong LOW_26 = (1ul << 26) - 1ul;

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

  for (uint i = 0; i < 10; ++i) out.v[i] = uint(z[i]);
}

inline void fe_reduce(thread Fe &value) {
  ulong z[10];
  for (uint i = 0; i < 10; ++i) z[i] = ulong(value.v[i]);
  fe_reduce_coefficients(value, z);
}

inline void fe_add(thread Fe &out, thread const Fe &left, thread const Fe &right) {
  // FieldElement2625::add deliberately does not reduce: a single addition
  // remains within the b < 1.75 headroom accepted by mul/square.  Every point
  // formula consumes this result in a mul/square before another addition.
  for (uint i = 0; i < 10; ++i) out.v[i] = left.v[i] + right.v[i];
}

inline void fe_sub(thread Fe &out, thread const Fe &left, thread const Fe &right) {
  // Compute left - right as left + 16p - right, matching Dalek's unsigned
  // backend and avoiding all unsigned underflow.
  ulong z[10] = {
    ulong(left.v[0]) + ulong(0x3ffffedu << 4) - ulong(right.v[0]),
    ulong(left.v[1]) + ulong(0x1ffffffu << 4) - ulong(right.v[1]),
    ulong(left.v[2]) + ulong(0x3ffffffu << 4) - ulong(right.v[2]),
    ulong(left.v[3]) + ulong(0x1ffffffu << 4) - ulong(right.v[3]),
    ulong(left.v[4]) + ulong(0x3ffffffu << 4) - ulong(right.v[4]),
    ulong(left.v[5]) + ulong(0x1ffffffu << 4) - ulong(right.v[5]),
    ulong(left.v[6]) + ulong(0x3ffffffu << 4) - ulong(right.v[6]),
    ulong(left.v[7]) + ulong(0x1ffffffu << 4) - ulong(right.v[7]),
    ulong(left.v[8]) + ulong(0x3ffffffu << 4) - ulong(right.v[8]),
    ulong(left.v[9]) + ulong(0x1ffffffu << 4) - ulong(right.v[9])
  };
  fe_reduce_coefficients(out, z);
}

inline void fe_neg(thread Fe &out, thread const Fe &input) {
  Fe zero;
  fe_zero(zero);
  fe_sub(out, zero, input);
}

inline void fe_mul(thread Fe &out, thread const Fe &left, thread const Fe &right) {
  thread const uint *x = left.v;
  thread const uint *y = right.v;
  const uint y1_19 = 19u * y[1], y2_19 = 19u * y[2];
  const uint y3_19 = 19u * y[3], y4_19 = 19u * y[4];
  const uint y5_19 = 19u * y[5], y6_19 = 19u * y[6];
  const uint y7_19 = 19u * y[7], y8_19 = 19u * y[8];
  const uint y9_19 = 19u * y[9];
  const uint x1_2 = 2u * x[1], x3_2 = 2u * x[3];
  const uint x5_2 = 2u * x[5], x7_2 = 2u * x[7];
  const uint x9_2 = 2u * x[9];

#define M2625(a, b) (ulong(a) * ulong(b))
  ulong z[10];
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

inline void fe_square(thread Fe &out, thread const Fe &input) {
  thread const uint *x = input.v;
  const uint x0_2=2u*x[0], x1_2=2u*x[1], x2_2=2u*x[2], x3_2=2u*x[3];
  const uint x4_2=2u*x[4], x5_2=2u*x[5], x6_2=2u*x[6], x7_2=2u*x[7];
  const uint x5_19=19u*x[5], x6_19=19u*x[6], x7_19=19u*x[7];
  const uint x8_19=19u*x[8], x9_19=19u*x[9];
#define S2625(a, b) (ulong(a) * ulong(b))
  ulong z[10];
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

inline void fe_freeze(thread Fe &out, thread const Fe &input) {
  fe_copy(out, input);
  fe_reduce(out);

  uint q = (out.v[0] + 19u) >> 26;
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

inline bool fe_equal(thread const Fe &left_input, thread const Fe &right_input) {
  Fe left, right;
  fe_freeze(left, left_input);
  fe_freeze(right, right_input);
  uint difference = 0;
  for (uint i = 0; i < 10; ++i) difference |= left.v[i] ^ right.v[i];
  return difference == 0u;
}

inline bool fe_is_zero(thread const Fe &input) {
  Fe value;
  fe_freeze(value, input);
  uint aggregate = 0;
  for (uint i = 0; i < 10; ++i) aggregate |= value.v[i];
  return aggregate == 0u;
}

inline bool fe_is_negative(thread const Fe &input) {
  Fe value;
  fe_freeze(value, input);
  return (value.v[0] & 1u) != 0u;
}

inline void fe_from_bytes(thread Fe &out, thread const uchar *bytes) {
#define LOAD3_T(i) (ulong(bytes[i]) | (ulong(bytes[(i)+1]) << 8) | (ulong(bytes[(i)+2]) << 16))
#define LOAD4_T(i) (LOAD3_T(i) | (ulong(bytes[(i)+3]) << 24))
  ulong z[10] = {
    LOAD4_T(0), LOAD3_T(4) << 6, LOAD3_T(7) << 5,
    LOAD3_T(10) << 3, LOAD3_T(13) << 2, LOAD4_T(16),
    LOAD3_T(20) << 7, LOAD3_T(23) << 5, LOAD3_T(26) << 4,
    (LOAD3_T(29) & ((1ul << 23) - 1ul)) << 2
  };
  fe_reduce_coefficients(out, z);
#undef LOAD3_T
#undef LOAD4_T
}

inline void fe_from_constant(thread Fe &out, constant const uchar *bytes) {
#define LOAD3_C(i) (ulong(bytes[i]) | (ulong(bytes[(i)+1]) << 8) | (ulong(bytes[(i)+2]) << 16))
#define LOAD4_C(i) (LOAD3_C(i) | (ulong(bytes[(i)+3]) << 24))
  ulong z[10] = {
    LOAD4_C(0), LOAD3_C(4) << 6, LOAD3_C(7) << 5,
    LOAD3_C(10) << 3, LOAD3_C(13) << 2, LOAD4_C(16),
    LOAD3_C(20) << 7, LOAD3_C(23) << 5, LOAD3_C(26) << 4,
    (LOAD3_C(29) & ((1ul << 23) - 1ul)) << 2
  };
  fe_reduce_coefficients(out, z);
#undef LOAD3_C
#undef LOAD4_C
}

inline void fe_to_bytes(thread uchar *out, thread const Fe &input) {
  Fe h;
  fe_freeze(h, input);
  out[0]=uchar(h.v[0]); out[1]=uchar(h.v[0]>>8); out[2]=uchar(h.v[0]>>16);
  out[3]=uchar((h.v[0]>>24)|(h.v[1]<<2)); out[4]=uchar(h.v[1]>>6);
  out[5]=uchar(h.v[1]>>14); out[6]=uchar((h.v[1]>>22)|(h.v[2]<<3));
  out[7]=uchar(h.v[2]>>5); out[8]=uchar(h.v[2]>>13);
  out[9]=uchar((h.v[2]>>21)|(h.v[3]<<5)); out[10]=uchar(h.v[3]>>3);
  out[11]=uchar(h.v[3]>>11); out[12]=uchar((h.v[3]>>19)|(h.v[4]<<6));
  out[13]=uchar(h.v[4]>>2); out[14]=uchar(h.v[4]>>10); out[15]=uchar(h.v[4]>>18);
  out[16]=uchar(h.v[5]); out[17]=uchar(h.v[5]>>8); out[18]=uchar(h.v[5]>>16);
  out[19]=uchar((h.v[5]>>24)|(h.v[6]<<1)); out[20]=uchar(h.v[6]>>7);
  out[21]=uchar(h.v[6]>>15); out[22]=uchar((h.v[6]>>23)|(h.v[7]<<3));
  out[23]=uchar(h.v[7]>>5); out[24]=uchar(h.v[7]>>13);
  out[25]=uchar((h.v[7]>>21)|(h.v[8]<<4)); out[26]=uchar(h.v[8]>>4);
  out[27]=uchar(h.v[8]>>12); out[28]=uchar((h.v[8]>>20)|(h.v[9]<<6));
  out[29]=uchar(h.v[9]>>2); out[30]=uchar(h.v[9]>>10); out[31]=uchar(h.v[9]>>18);
}

inline void fe_pow(
    thread Fe &out,
    thread const Fe &input,
    constant const uchar *exponent,
    int highest_bit) {
  Fe accumulator, square;
  fe_one(accumulator);
  for (int bit = highest_bit; bit >= 0; --bit) {
    fe_square(square, accumulator);
    fe_copy(accumulator, square);
    if (((exponent[uint(bit) >> 3] >> (uint(bit) & 7u)) & 1u) != 0u) {
      fe_mul(square, accumulator, input);
      fe_copy(accumulator, square);
    }
  }
  fe_copy(out, accumulator);
}

inline void fe_inverse(thread Fe &out, thread const Fe &input) {
  fe_pow(out, input, INVERSE_EXPONENT, 254);
}

inline void fe_square_root(thread Fe &out, thread const Fe &input) {
  fe_pow(out, input, SQRT_EXPONENT, 251);
}

inline bool encoded_y_is_canonical(thread const uchar *bytes) {
  const uchar top = bytes[31] & 127u;
  if (top != 127u) return top < 127u;
  for (int i = 30; i >= 1; --i)
    if (bytes[i] != 255u) return bytes[i] < 255u;
  return bytes[0] < 237u;
}

inline void point_identity(thread Point &out) {
  fe_zero(out.x);
  fe_one(out.y);
  fe_one(out.z);
  fe_zero(out.t);
}

inline void point_copy(thread Point &out, thread const Point &input) {
  fe_copy(out.x, input.x);
  fe_copy(out.y, input.y);
  fe_copy(out.z, input.z);
  fe_copy(out.t, input.t);
}

// Complete extended-Edwards addition formula for a = -1.
inline void point_add(thread Point &out, thread const Point &left, thread const Point &right) {
  Fe a, b, c, d, e, f, g, h, tmp, curve_d;
  fe_sub(tmp, left.y, left.x);
  fe_sub(a, right.y, right.x);
  fe_mul(a, tmp, a);
  fe_add(tmp, left.y, left.x);
  fe_add(b, right.y, right.x);
  fe_mul(b, tmp, b);
  fe_from_constant(curve_d, FIELD_D);
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

inline void point_double(thread Point &out, thread const Point &input) {
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

inline void point_cmov(thread Point &target, thread const Point &source, uint choose_source) {
  const ulong mask = 0ul - ulong(choose_source & 1u);
  for (uint i = 0; i < 10; ++i) {
    target.x.v[i] = (target.x.v[i] & ~mask) | (source.x.v[i] & mask);
    target.y.v[i] = (target.y.v[i] & ~mask) | (source.y.v[i] & mask);
    target.z.v[i] = (target.z.v[i] & ~mask) | (source.z.v[i] & mask);
    target.t.v[i] = (target.t.v[i] & ~mask) | (source.t.v[i] & mask);
  }
}

inline bool point_from_compressed(thread Point &out, thread const uchar *bytes) {
  if (!encoded_y_is_canonical(bytes)) return false;
  Fe y_squared, numerator, denominator, inverse, x_squared, x, check, sqrt_m1;
  fe_from_bytes(out.y, bytes);
  fe_one(out.z);
  fe_square(y_squared, out.y);
  fe_sub(numerator, y_squared, out.z);
  fe_from_constant(sqrt_m1, FIELD_D);
  fe_mul(denominator, y_squared, sqrt_m1);
  fe_add(denominator, denominator, out.z);
  fe_inverse(inverse, denominator);
  fe_mul(x_squared, numerator, inverse);
  fe_square_root(x, x_squared);
  fe_square(check, x);
  if (!fe_equal(check, x_squared)) {
    fe_from_constant(sqrt_m1, SQRT_M1);
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

inline void point_to_compressed(thread uchar *out, thread const Point &input) {
  Fe inverse, x, y;
  fe_inverse(inverse, input.z);
  fe_mul(x, input.x, inverse);
  fe_mul(y, input.y, inverse);
  fe_to_bytes(out, y);
  out[31] ^= uchar(fe_is_negative(x) ? 128u : 0u);
}

inline void scalar_conditional_subtract_l(thread uchar *scalar) {
  uint difference[32];
  uint borrow = 0;
  for (uint i = 0; i < 32; ++i) {
    const uint subtrahend = uint(SCALAR_L[i]) + borrow;
    const uint value = uint(scalar[i]);
    difference[i] = (value - subtrahend) & 255u;
    borrow = uint(value < subtrahend);
  }
  const uint mask = 0u - (borrow ^ 1u);
  for (uint i = 0; i < 32; ++i)
    scalar[i] = uchar((uint(scalar[i]) & ~mask) | (difference[i] & mask));
}

// Exactly Scalar::from_bytes_mod_order for a 32-byte little-endian input.
inline void scalar_reduce_mod_l(thread uchar *out, device const uchar *input) {
  for (uint i = 0; i < 32; ++i) out[i] = 0;
  for (int bit = 255; bit >= 0; --bit) {
    uint carry = uint((input[uint(bit) >> 3] >> (uint(bit) & 7u)) & 1u);
    for (uint i = 0; i < 32; ++i) {
      const uint value = (uint(out[i]) << 1) | carry;
      out[i] = uchar(value & 255u);
      carry = value >> 8;
    }
    scalar_conditional_subtract_l(out);
  }
}

inline void scalar_multiply_by_eight_mod_l(thread uchar *scalar) {
  uint carry = 0;
  for (uint i = 0; i < 32; ++i) {
    const uint value = (uint(scalar[i]) << 3) | carry;
    scalar[i] = uchar(value & 255u);
    carry = value >> 8;
  }
  // The unreduced value is below 8l, hence at most seven subtractions.
  for (uint pass = 0; pass < 8; ++pass) scalar_conditional_subtract_l(scalar);
}

inline void point_scalar_multiply(thread Point &out, thread const Point &point, thread const uchar *scalar) {
  Point accumulator, doubled, added;
  point_identity(accumulator);
  // Fixed 256 iterations. `point_cmov` avoids a secret-dependent branch.
  for (int bit = 255; bit >= 0; --bit) {
    point_double(doubled, accumulator);
    point_copy(accumulator, doubled);
    point_add(added, accumulator, point);
    const uint select_added = uint((scalar[uint(bit) >> 3] >> (uint(bit) & 7u)) & 1u);
    point_cmov(accumulator, added, select_added);
  }
  point_copy(out, accumulator);
}

kernel void derivation_m1_reference(
    device const uchar *view_scalar [[buffer(0)]],
    device const uchar *points [[buffer(1)]],
    device uchar *results [[buffer(2)]],
    device uchar *valid [[buffer(3)]],
    constant Parameters &parameters [[buffer(4)]],
    uint id [[thread_position_in_grid]]) {
  if (id >= parameters.count) return;
  const uint offset = id * 32u;
  uchar encoded_point[32];
  uchar result[32];
  for (uint i = 0; i < 32; ++i) encoded_point[i] = points[offset + i];
  Point point, derived;
  if (!point_from_compressed(point, encoded_point)) {
    for (uint i = 0; i < 32; ++i) results[offset + i] = 0;
    valid[id] = 0;
    return;
  }
  uchar scalar[32];
  scalar_reduce_mod_l(scalar, view_scalar);
  scalar_multiply_by_eight_mod_l(scalar);
  point_scalar_multiply(derived, point, scalar);
  point_to_compressed(result, derived);
  for (uint i = 0; i < 32; ++i) results[offset + i] = result[i];
  valid[id] = 1;
}

// M2 keeps the exact M1 math but eliminates redundant work: the view scalar
// is common to the whole wallet scan, so one lane reduces and folds it once
// per threadgroup. Every lane then copies the same 32-byte folded scalar into
// private memory for the fixed-length scalar multiplication.
kernel void derivation_m2_group_scalar(
    device const uchar *view_scalar [[buffer(0)]],
    device const uchar *points [[buffer(1)]],
    device uchar *results [[buffer(2)]],
    device uchar *valid [[buffer(3)]],
    constant Parameters &parameters [[buffer(4)]],
    uint id [[thread_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]) {
  threadgroup uchar folded_scalar[32];
  if (lane == 0u) {
    uchar local_scalar[32];
    scalar_reduce_mod_l(local_scalar, view_scalar);
    scalar_multiply_by_eight_mod_l(local_scalar);
    for (uint i = 0; i < 32; ++i) folded_scalar[i] = local_scalar[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (id >= parameters.count) return;
  const uint offset = id * 32u;
  uchar encoded_point[32];
  uchar result[32];
  uchar scalar[32];
  for (uint i = 0; i < 32; ++i) {
    encoded_point[i] = points[offset + i];
    scalar[i] = folded_scalar[i];
  }
  Point point, derived;
  if (!point_from_compressed(point, encoded_point)) {
    for (uint i = 0; i < 32; ++i) results[offset + i] = 0;
    valid[id] = 0;
    return;
  }
  point_scalar_multiply(derived, point, scalar);
  point_to_compressed(result, derived);
  for (uint i = 0; i < 32; ++i) results[offset + i] = result[i];
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

inline void p2_from_p1p1(thread PointP2 &out, thread const PointP1P1 &input) {
  fe_mul(out.x, input.x, input.t);
  fe_mul(out.y, input.y, input.z);
  fe_mul(out.z, input.z, input.t);
}

inline void p3_from_p1p1(thread Point &out, thread const PointP1P1 &input) {
  fe_mul(out.x, input.x, input.t);
  fe_mul(out.y, input.y, input.z);
  fe_mul(out.z, input.z, input.t);
  fe_mul(out.t, input.x, input.y);
}

inline void p3_from_p2(thread Point &out, thread const PointP2 &input) {
  fe_copy(out.x, input.x);
  fe_copy(out.y, input.y);
  fe_copy(out.z, input.z);
  fe_mul(out.t, input.x, input.y);
}

inline void p2_double(thread PointP1P1 &out, thread const PointP2 &input) {
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

inline void niels_identity(thread ProjectiveNiels &out) {
  fe_one(out.y_plus_x);
  fe_one(out.y_minus_x);
  fe_one(out.z);
  fe_zero(out.t2d);
}

inline void niels_from_p3(thread ProjectiveNiels &out, thread const Point &input) {
  Fe curve_d;
  fe_add(out.y_plus_x, input.y, input.x);
  fe_sub(out.y_minus_x, input.y, input.x);
  fe_copy(out.z, input.z);
  fe_from_constant(curve_d, FIELD_D);
  fe_mul(out.t2d, input.t, curve_d);
  fe_add(out.t2d, out.t2d, out.t2d);
}

inline void niels_cmov(
    thread ProjectiveNiels &target,
    thread const ProjectiveNiels &source,
    uint choose_source) {
  const ulong mask = 0ul - ulong(choose_source & 1u);
  for (uint i = 0; i < 10; ++i) {
    target.y_plus_x.v[i] = (target.y_plus_x.v[i] & ~mask) | (source.y_plus_x.v[i] & mask);
    target.y_minus_x.v[i] = (target.y_minus_x.v[i] & ~mask) | (source.y_minus_x.v[i] & mask);
    target.z.v[i] = (target.z.v[i] & ~mask) | (source.z.v[i] & mask);
    target.t2d.v[i] = (target.t2d.v[i] & ~mask) | (source.t2d.v[i] & mask);
  }
}

inline uint ct_equal_u32(uint left, uint right) {
  const uint difference = left ^ right;
  return ((difference | (0u - difference)) >> 31u) ^ 1u;
}

inline void niels_select_signed(
    thread ProjectiveNiels &out,
    thread const ProjectiveNiels *table,
    int digit) {
  // `digit` is in [-8, 8].  This is the same constant-time selection shape
  // as Dalek's LookupTable::select: select |digit| then conditionally negate.
  const int sign_mask = digit >> 31;
  const uint absolute = uint((digit + sign_mask) ^ sign_mask);
  niels_identity(out);
  for (uint value = 1; value <= 8; ++value)
    niels_cmov(out, table[value - 1u], ct_equal_u32(absolute, value));

  ProjectiveNiels negated;
  fe_copy(negated.y_plus_x, out.y_minus_x);
  fe_copy(negated.y_minus_x, out.y_plus_x);
  fe_copy(negated.z, out.z);
  fe_neg(negated.t2d, out.t2d);
  niels_cmov(out, negated, uint(sign_mask) & 1u);
}

inline void p3_add_niels(
    thread PointP1P1 &out,
    thread const Point &left,
    thread const ProjectiveNiels &right) {
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

inline void scalar_as_radix_16(thread int *digits, thread const uchar *scalar) {
  for (uint i = 0; i < 32; ++i) {
    digits[2u * i] = int(scalar[i] & 15u);
    digits[2u * i + 1u] = int(scalar[i] >> 4u);
  }
  for (uint i = 0; i < 63; ++i) {
    const int carry = (digits[i] + 8) >> 4;
    digits[i] -= carry << 4;
    digits[i + 1u] += carry;
  }
}

inline void point_scalar_multiply_radix16(
    thread Point &out,
    thread const Point &point,
    thread const uchar *scalar) {
  ProjectiveNiels table[8];
  niels_from_p3(table[0], point);
  for (uint i = 0; i < 7; ++i) {
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

inline void scalar_as_radix_16_group(
    threadgroup char *digits,
    thread const uchar *scalar) {
  for (uint i = 0; i < 32; ++i) {
    digits[2u * i] = char(scalar[i] & 15u);
    digits[2u * i + 1u] = char(scalar[i] >> 4u);
  }
  for (uint i = 0; i < 63; ++i) {
    const int digit = int(digits[i]);
    const int carry = (digit + 8) >> 4;
    digits[i] = char(digit - (carry << 4));
    digits[i + 1u] = char(int(digits[i + 1u]) + carry);
  }
}

inline void point_scalar_multiply_radix16_group(
    thread Point &out,
    thread const Point &point,
    threadgroup const char *digits) {
  ProjectiveNiels table[8];
  niels_from_p3(table[0], point);
  for (uint i = 0; i < 7; ++i) {
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

kernel void derivation_m4_radix16_niels(
    device const uchar *view_scalar [[buffer(0)]],
    device const uchar *points [[buffer(1)]],
    device uchar *results [[buffer(2)]],
    device uchar *valid [[buffer(3)]],
    constant Parameters &parameters [[buffer(4)]],
    uint id [[thread_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]) {
  // Preserve M2's valid reduction: the view scalar is common to the scan
  // batch, so only one lane per threadgroup performs the public setup work.
  threadgroup uchar folded_scalar[32];
  if (lane == 0u) {
    uchar local_scalar[32];
    scalar_reduce_mod_l(local_scalar, view_scalar);
    scalar_multiply_by_eight_mod_l(local_scalar);
    for (uint i = 0; i < 32; ++i) folded_scalar[i] = local_scalar[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (id >= parameters.count) return;

  const uint offset = id * 32u;
  uchar encoded_point[32];
  uchar scalar[32];
  uchar result[32];
  for (uint i = 0; i < 32; ++i) {
    encoded_point[i] = points[offset + i];
    scalar[i] = folded_scalar[i];
  }
  Point point, derived;
  if (!point_from_compressed(point, encoded_point)) {
    for (uint i = 0; i < 32; ++i) results[offset + i] = 0;
    valid[id] = 0;
    return;
  }
  point_scalar_multiply_radix16(derived, point, scalar);
  point_to_compressed(result, derived);
  for (uint i = 0; i < 32; ++i) results[offset + i] = result[i];
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

inline void fe_square_n(thread Fe &out, thread const Fe &input, uint count) {
  Fe accumulator, squared;
  fe_copy(accumulator, input);
  for (uint i = 0; i < count; ++i) {
    fe_square(squared, accumulator);
    fe_copy(accumulator, squared);
  }
  fe_copy(out, accumulator);
}

// Return x^(2^250 - 1) and x^11.  This is the exact exponentiation chain
// used by Dalek's FieldElement::pow22501().
inline void fe_pow22501(thread Fe &power_22501, thread Fe &power_11, thread const Fe &input) {
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

inline void fe_inverse_m5(thread Fe &out, thread const Fe &input) {
  Fe t19, t3, t20;
  fe_pow22501(t19, t3, input);
  fe_square_n(t20, t19, 5u);
  fe_mul(out, t20, t3);                         // x^(p - 2)
}

inline void fe_square_root_m5(thread Fe &out, thread const Fe &input) {
  Fe t19, unused_t3, t20, input_squared;
  fe_pow22501(t19, unused_t3, input);
  fe_square_n(t20, t19, 2u);
  fe_square(input_squared, input);
  fe_mul(out, t20, input_squared);              // x^((p + 3) / 8)
}

inline bool point_from_compressed_m5(thread Point &out, thread const uchar *bytes) {
  if (!encoded_y_is_canonical(bytes)) return false;
  Fe y_squared, numerator, denominator, inverse, x_squared, x, check, sqrt_m1;
  fe_from_bytes(out.y, bytes);
  fe_one(out.z);
  fe_square(y_squared, out.y);
  fe_sub(numerator, y_squared, out.z);
  fe_from_constant(sqrt_m1, FIELD_D);
  fe_mul(denominator, y_squared, sqrt_m1);
  fe_add(denominator, denominator, out.z);
  fe_inverse_m5(inverse, denominator);
  fe_mul(x_squared, numerator, inverse);
  fe_square_root_m5(x, x_squared);
  fe_square(check, x);
  if (!fe_equal(check, x_squared)) {
    fe_from_constant(sqrt_m1, SQRT_M1);
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

inline void point_to_compressed_m5(thread uchar *out, thread const Point &input) {
  Fe inverse, x, y;
  fe_inverse_m5(inverse, input.z);
  fe_mul(x, input.x, inverse);
  fe_mul(y, input.y, inverse);
  fe_to_bytes(out, y);
  out[31] ^= uchar(fe_is_negative(x) ? 128u : 0u);
}

kernel void derivation_m10_radix2625_groupdigits_niels(
    device const uchar *view_scalar [[buffer(0)]],
    device const uchar *points [[buffer(1)]],
    device uchar *results [[buffer(2)]],
    device uchar *valid [[buffer(3)]],
    constant Parameters &parameters [[buffer(4)]],
    uint id [[thread_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]) {
  threadgroup char scalar_digits[64];
  if (lane == 0u) {
    uchar local_scalar[32];
    scalar_reduce_mod_l(local_scalar, view_scalar);
    scalar_multiply_by_eight_mod_l(local_scalar);
    scalar_as_radix_16_group(scalar_digits, local_scalar);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (id >= parameters.count) return;

  const uint offset = id * 32u;
  uchar encoded_point[32];
  uchar result[32];
  for (uint i = 0; i < 32; ++i) encoded_point[i] = points[offset + i];
  Point point, derived;
  if (!point_from_compressed_m5(point, encoded_point)) {
    for (uint i = 0; i < 32; ++i) results[offset + i] = 0;
    valid[id] = 0;
    return;
  }
  point_scalar_multiply_radix16_group(derived, point, scalar_digits);
  point_to_compressed_m5(result, derived);
  for (uint i = 0; i < 32; ++i) results[offset + i] = result[i];
  valid[id] = 1;
}

inline void fe_store_device(device uint *destination, thread const Fe &value) {
  for (uint i = 0; i < 10; ++i) destination[i] = value.v[i];
}

inline void fe_load_device(thread Fe &value, device const uint *source) {
  for (uint i = 0; i < 10; ++i) value.v[i] = source[i];
}

// Pass 1: point decoding and variable-base multiplication remain massively
// parallel.  X/Y/Z are retained in projective form for a shared inversion.
kernel void derivation_m12_projective_chunkinvert(
    device const uchar *view_scalar [[buffer(0)]],
    device const uchar *points [[buffer(1)]],
    device uint *projective [[buffer(2)]],
    device uchar *valid [[buffer(3)]],
    constant Parameters &parameters [[buffer(4)]],
    uint id [[thread_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]) {
  threadgroup char scalar_digits[64];
  if (lane == 0u) {
    uchar local_scalar[32];
    scalar_reduce_mod_l(local_scalar, view_scalar);
    scalar_multiply_by_eight_mod_l(local_scalar);
    scalar_as_radix_16_group(scalar_digits, local_scalar);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (id >= parameters.count) return;

  const uint byte_offset = id * 32u;
  const uint field_offset = id * 30u;
  uchar encoded_point[32];
  for (uint i = 0; i < 32; ++i) encoded_point[i] = points[byte_offset + i];

  Point point, derived;
  if (!point_from_compressed_m5(point, encoded_point)) {
    for (uint i = 0; i < 30; ++i) projective[field_offset + i] = 0u;
    valid[id] = 0u;
    return;
  }
  point_scalar_multiply_radix16_group(derived, point, scalar_digits);
  fe_store_device(projective + field_offset, derived.x);
  fe_store_device(projective + field_offset + 10u, derived.y);
  fe_store_device(projective + field_offset + 20u, derived.z);
  valid[id] = 1u;
}

// Pass 2: each GPU thread performs one independent Montgomery batch inversion
// over a small public-size chunk.  This retains hundreds of parallel threads
// while replacing chunk_size exponentiations with one.
kernel void derivation_m12_chunk_inverse(
    device const uint *projective [[buffer(0)]],
    device uint *inverse_z [[buffer(1)]],
    device const uchar *valid [[buffer(2)]],
    constant Parameters &parameters [[buffer(3)]],
    uint id [[thread_position_in_grid]]) {
  const uint chunk_size = parameters.batch_chunk_size;
  const uint begin = id * chunk_size;
  if (begin >= parameters.count) return;
  const uint end = min(begin + chunk_size, parameters.count);

  Fe accumulator;
  fe_one(accumulator);
  for (uint record = begin; record < end; ++record) {
    const uint field_offset = record * 30u;
    const uint inverse_offset = record * 10u;
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
  uint remaining = end;
  while (remaining != begin) {
    const uint record = --remaining;
    const uint field_offset = record * 30u;
    const uint inverse_offset = record * 10u;
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
kernel void derivation_m12_compress_chunkinvert(
    device const uint *projective [[buffer(0)]],
    device const uint *inverse_z [[buffer(1)]],
    device uchar *results [[buffer(2)]],
    device const uchar *valid [[buffer(3)]],
    constant Parameters &parameters [[buffer(4)]],
    uint id [[thread_position_in_grid]]) {
  if (id >= parameters.count) return;
  const uint byte_offset = id * 32u;
  if (valid[id] == 0u) {
    for (uint i = 0; i < 32; ++i) results[byte_offset + i] = 0u;
    return;
  }

  const uint field_offset = id * 30u;
  const uint inverse_offset = id * 10u;
  Fe projective_x, projective_y, inverse, x, y;
  fe_load_device(projective_x, projective + field_offset);
  fe_load_device(projective_y, projective + field_offset + 10u);
  fe_load_device(inverse, inverse_z + inverse_offset);
  fe_mul(x, projective_x, inverse);
  fe_mul(y, projective_y, inverse);
  uchar encoded[32];
  fe_to_bytes(encoded, y);
  encoded[31] ^= uchar(fe_is_negative(x) ? 128u : 0u);
  for (uint i = 0; i < 32; ++i) results[byte_offset + i] = encoded[i];
}
