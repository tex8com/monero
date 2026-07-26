// M1 reference kernel for the exact wallet operation D = 8 * a * R.
//
// This is deliberately a correctness-first implementation for Apple Metal:
// 16 radix-2^16 limbs, complete extended-Edwards formulas, fixed-length
// scalar handling, and a final compressed-Edwards encoding. It is independent
// from wallet code and must be byte-for-byte compared with Dalek before any
// optimization or product integration is considered.
#include <metal_stdlib>
using namespace metal;

struct Parameters { uint count; };
struct Fe { ulong v[16]; };
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

inline ulong modulus_limb(uint index) {
  return index == 0 ? 65517ul : (index == 15 ? 32767ul : 65535ul);
}

inline void fe_zero(thread Fe &out) {
  for (uint i = 0; i < 16; ++i) out.v[i] = 0;
}

inline void fe_one(thread Fe &out) {
  fe_zero(out);
  out.v[0] = 1;
}

inline void fe_copy(thread Fe &out, thread const Fe &input) {
  for (uint i = 0; i < 16; ++i) out.v[i] = input.v[i];
}

// Fold radix-2^16 carries, with the top limb having only 15 bits because
// 2^255 == 19 modulo p. Three passes keep every operation in the canonical
// bounded representation before an optional final p subtraction.
inline void fe_carry(thread Fe &value) {
  for (uint pass = 0; pass < 3; ++pass) {
    for (uint i = 0; i < 15; ++i) {
      const ulong carry = value.v[i] >> 16;
      value.v[i] &= 65535ul;
      value.v[i + 1] += carry;
    }
    const ulong carry = value.v[15] >> 15;
    value.v[15] &= 32767ul;
    value.v[0] += carry * 19ul;
  }
}

inline void fe_reduce(thread Fe &value) {
  fe_carry(value);
  Fe difference;
  ulong borrow = 0;
  for (uint i = 0; i < 16; ++i) {
    const ulong subtrahend = modulus_limb(i) + borrow;
    difference.v[i] = value.v[i] - subtrahend;
    borrow = ulong(value.v[i] < subtrahend);
  }
  const ulong use_difference = borrow ^ 1ul;
  const ulong mask = 0ul - use_difference;
  for (uint i = 0; i < 16; ++i)
    value.v[i] = (value.v[i] & ~mask) | (difference.v[i] & mask);
}

inline void fe_add(thread Fe &out, thread const Fe &left, thread const Fe &right) {
  for (uint i = 0; i < 16; ++i) out.v[i] = left.v[i] + right.v[i];
  fe_carry(out);
}

inline void fe_sub(thread Fe &out, thread const Fe &left, thread const Fe &right) {
  // Add 2p first so unsigned subtraction cannot underflow.
  for (uint i = 0; i < 16; ++i)
    out.v[i] = left.v[i] + 2ul * modulus_limb(i) - right.v[i];
  fe_carry(out);
}

inline void fe_neg(thread Fe &out, thread const Fe &input) {
  Fe zero;
  fe_zero(zero);
  fe_sub(out, zero, input);
}

inline void fe_mul(thread Fe &out, thread const Fe &left, thread const Fe &right) {
  ulong product[31] = {0};
  for (uint i = 0; i < 16; ++i)
    for (uint j = 0; j < 16; ++j)
      product[i + j] += left.v[i] * right.v[j];
  // (2^16)^16 = 2^256 = 38 (mod 2^255 - 19).
  for (int i = 30; i >= 16; --i)
    product[i - 16] += product[i] * 38ul;
  for (uint i = 0; i < 16; ++i) out.v[i] = product[i];
  fe_carry(out);
}

inline void fe_square(thread Fe &out, thread const Fe &input) {
  fe_mul(out, input, input);
}

inline bool fe_equal(thread const Fe &left_input, thread const Fe &right_input) {
  Fe left, right;
  fe_copy(left, left_input);
  fe_copy(right, right_input);
  fe_reduce(left);
  fe_reduce(right);
  ulong difference = 0;
  for (uint i = 0; i < 16; ++i) difference |= left.v[i] ^ right.v[i];
  return difference == 0;
}

inline bool fe_is_zero(thread const Fe &input) {
  Fe value;
  fe_copy(value, input);
  fe_reduce(value);
  ulong aggregate = 0;
  for (uint i = 0; i < 16; ++i) aggregate |= value.v[i];
  return aggregate == 0;
}

inline bool fe_is_negative(thread const Fe &input) {
  Fe value;
  fe_copy(value, input);
  fe_reduce(value);
  return (value.v[0] & 1ul) != 0;
}

inline void fe_from_bytes(thread Fe &out, thread const uchar *bytes) {
  for (uint i = 0; i < 15; ++i)
    out.v[i] = ulong(bytes[2 * i]) | (ulong(bytes[2 * i + 1]) << 8);
  out.v[15] = (ulong(bytes[30]) | (ulong(bytes[31] & 127u) << 8)) & 32767ul;
}

inline void fe_from_constant(thread Fe &out, constant const uchar *bytes) {
  for (uint i = 0; i < 15; ++i)
    out.v[i] = ulong(bytes[2 * i]) | (ulong(bytes[2 * i + 1]) << 8);
  out.v[15] = (ulong(bytes[30]) | (ulong(bytes[31] & 127u) << 8)) & 32767ul;
}

inline void fe_to_bytes(thread uchar *out, thread const Fe &input) {
  Fe value;
  fe_copy(value, input);
  fe_reduce(value);
  for (uint i = 0; i < 15; ++i) {
    out[2 * i] = uchar(value.v[i] & 255ul);
    out[2 * i + 1] = uchar(value.v[i] >> 8);
  }
  out[30] = uchar(value.v[15] & 255ul);
  out[31] = uchar(value.v[15] >> 8);
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
  for (uint i = 0; i < 16; ++i) {
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
  for (uint i = 0; i < 16; ++i) {
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

kernel void derivation_m5_radix16_niels_addition_chain(
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
  uchar scalar[32];
  uchar result[32];
  for (uint i = 0; i < 32; ++i) {
    encoded_point[i] = points[offset + i];
    scalar[i] = folded_scalar[i];
  }
  Point point, derived;
  if (!point_from_compressed_m5(point, encoded_point)) {
    for (uint i = 0; i < 32; ++i) results[offset + i] = 0;
    valid[id] = 0;
    return;
  }
  point_scalar_multiply_radix16(derived, point, scalar);
  point_to_compressed_m5(result, derived);
  for (uint i = 0; i < 32; ++i) results[offset + i] = result[i];
  valid[id] = 1;
}
