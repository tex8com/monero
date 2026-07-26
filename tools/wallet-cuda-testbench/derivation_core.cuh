#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define MONERO_CUDA_DEVICE __device__ __forceinline__
#define MONERO_CUDA_CONSTANT static __device__ __constant__
#else
#define MONERO_CUDA_DEVICE inline
#define MONERO_CUDA_CONSTANT inline constexpr
#endif

namespace monero_cuda {

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

struct Fe {
  u64 v[16];
};

struct Point {
  Fe x;
  Fe y;
  Fe z;
  Fe t;
};

struct PointP2 {
  Fe x;
  Fe y;
  Fe z;
};

struct PointP1P1 {
  Fe x;
  Fe y;
  Fe z;
  Fe t;
};

struct ProjectiveNiels {
  Fe y_plus_x;
  Fe y_minus_x;
  Fe z;
  Fe t2d;
};

MONERO_CUDA_CONSTANT u8 FIELD_D[32] = {
    0xa3, 0x78, 0x59, 0x13, 0xca, 0x4d, 0xeb, 0x75,
    0xab, 0xd8, 0x41, 0x41, 0x4d, 0x0a, 0x70, 0x00,
    0x98, 0xe8, 0x79, 0x77, 0x79, 0x40, 0xc7, 0x8c,
    0x73, 0xfe, 0x6f, 0x2b, 0xee, 0x6c, 0x03, 0x52,
};

MONERO_CUDA_CONSTANT u8 SQRT_M1[32] = {
    0xb0, 0xa0, 0x0e, 0x4a, 0x27, 0x1b, 0xee, 0xc4,
    0x78, 0xe4, 0x2f, 0xad, 0x06, 0x18, 0x43, 0x2f,
    0xa7, 0xd7, 0xfb, 0x3d, 0x99, 0x00, 0x4d, 0x2b,
    0x0b, 0xdf, 0xc1, 0x4f, 0x80, 0x24, 0x83, 0x2b,
};

MONERO_CUDA_CONSTANT u8 SCALAR_L[32] = {
    0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58,
    0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10,
};

MONERO_CUDA_DEVICE u64 modulus_limb(u32 index) {
  return index == 0 ? 65517ull : (index == 15 ? 32767ull : 65535ull);
}

MONERO_CUDA_DEVICE void fe_zero(Fe &out) {
  for (u32 i = 0; i < 16; ++i) out.v[i] = 0;
}

MONERO_CUDA_DEVICE void fe_one(Fe &out) {
  fe_zero(out);
  out.v[0] = 1;
}

MONERO_CUDA_DEVICE void fe_copy(Fe &out, const Fe &input) {
  for (u32 i = 0; i < 16; ++i) out.v[i] = input.v[i];
}

MONERO_CUDA_DEVICE void fe_carry(Fe &value) {
  for (u32 pass = 0; pass < 3; ++pass) {
    for (u32 i = 0; i < 15; ++i) {
      const u64 carry = value.v[i] >> 16;
      value.v[i] &= 65535ull;
      value.v[i + 1] += carry;
    }
    const u64 carry = value.v[15] >> 15;
    value.v[15] &= 32767ull;
    value.v[0] += carry * 19ull;
  }
}

MONERO_CUDA_DEVICE void fe_reduce(Fe &value) {
  fe_carry(value);
  Fe difference;
  u64 borrow = 0;
  for (u32 i = 0; i < 16; ++i) {
    const u64 subtrahend = modulus_limb(i) + borrow;
    difference.v[i] = value.v[i] - subtrahend;
    borrow = static_cast<u64>(value.v[i] < subtrahend);
  }
  const u64 mask = 0ull - (borrow ^ 1ull);
  for (u32 i = 0; i < 16; ++i)
    value.v[i] = (value.v[i] & ~mask) | (difference.v[i] & mask);
}

MONERO_CUDA_DEVICE void fe_add(Fe &out, const Fe &left, const Fe &right) {
  for (u32 i = 0; i < 16; ++i) out.v[i] = left.v[i] + right.v[i];
  fe_carry(out);
}

MONERO_CUDA_DEVICE void fe_sub(Fe &out, const Fe &left, const Fe &right) {
  for (u32 i = 0; i < 16; ++i)
    out.v[i] = left.v[i] + 2ull * modulus_limb(i) - right.v[i];
  fe_carry(out);
}

MONERO_CUDA_DEVICE void fe_neg(Fe &out, const Fe &input) {
  Fe zero;
  fe_zero(zero);
  fe_sub(out, zero, input);
}

MONERO_CUDA_DEVICE void fe_mul(Fe &out, const Fe &left, const Fe &right) {
  u64 product[31] = {};
  for (u32 i = 0; i < 16; ++i)
    for (u32 j = 0; j < 16; ++j)
      product[i + j] += left.v[i] * right.v[j];
  for (int i = 30; i >= 16; --i)
    product[i - 16] += product[i] * 38ull;
  for (u32 i = 0; i < 16; ++i) out.v[i] = product[i];
  fe_carry(out);
}

MONERO_CUDA_DEVICE void fe_square(Fe &out, const Fe &input) {
  fe_mul(out, input, input);
}

MONERO_CUDA_DEVICE bool fe_equal(const Fe &left_input, const Fe &right_input) {
  Fe left, right;
  fe_copy(left, left_input);
  fe_copy(right, right_input);
  fe_reduce(left);
  fe_reduce(right);
  u64 difference = 0;
  for (u32 i = 0; i < 16; ++i) difference |= left.v[i] ^ right.v[i];
  return difference == 0;
}

MONERO_CUDA_DEVICE bool fe_is_zero(const Fe &input) {
  Fe value;
  fe_copy(value, input);
  fe_reduce(value);
  u64 aggregate = 0;
  for (u32 i = 0; i < 16; ++i) aggregate |= value.v[i];
  return aggregate == 0;
}

MONERO_CUDA_DEVICE bool fe_is_negative(const Fe &input) {
  Fe value;
  fe_copy(value, input);
  fe_reduce(value);
  return (value.v[0] & 1ull) != 0;
}

MONERO_CUDA_DEVICE void fe_from_bytes(Fe &out, const u8 *bytes) {
  for (u32 i = 0; i < 15; ++i)
    out.v[i] = static_cast<u64>(bytes[2 * i]) |
               (static_cast<u64>(bytes[2 * i + 1]) << 8);
  out.v[15] =
      (static_cast<u64>(bytes[30]) |
       (static_cast<u64>(bytes[31] & 127u) << 8)) &
      32767ull;
}

MONERO_CUDA_DEVICE void fe_from_constant(Fe &out, const u8 *bytes) {
  fe_from_bytes(out, bytes);
}

MONERO_CUDA_DEVICE void fe_to_bytes(u8 *out, const Fe &input) {
  Fe value;
  fe_copy(value, input);
  fe_reduce(value);
  for (u32 i = 0; i < 15; ++i) {
    out[2 * i] = static_cast<u8>(value.v[i] & 255ull);
    out[2 * i + 1] = static_cast<u8>(value.v[i] >> 8);
  }
  out[30] = static_cast<u8>(value.v[15] & 255ull);
  out[31] = static_cast<u8>(value.v[15] >> 8);
}

MONERO_CUDA_DEVICE void fe_square_n(Fe &out, const Fe &input, u32 count) {
  Fe accumulator, squared;
  fe_copy(accumulator, input);
  for (u32 i = 0; i < count; ++i) {
    fe_square(squared, accumulator);
    fe_copy(accumulator, squared);
  }
  fe_copy(out, accumulator);
}

MONERO_CUDA_DEVICE void fe_pow22501(Fe &power_22501, Fe &power_11,
                                    const Fe &input) {
  Fe t0, t1, t2, t4, t5, t6, t7, t8, t9;
  Fe t10, t11, t12, t13, t14, t15, t16, t17, t18;
  fe_square(t0, input);
  fe_square_n(t1, t0, 2u);
  fe_mul(t2, input, t1);
  fe_mul(power_11, t0, t2);
  fe_square(t4, power_11);
  fe_mul(t5, t2, t4);
  fe_square_n(t6, t5, 5u);
  fe_mul(t7, t6, t5);
  fe_square_n(t8, t7, 10u);
  fe_mul(t9, t8, t7);
  fe_square_n(t10, t9, 20u);
  fe_mul(t11, t10, t9);
  fe_square_n(t12, t11, 10u);
  fe_mul(t13, t12, t7);
  fe_square_n(t14, t13, 50u);
  fe_mul(t15, t14, t13);
  fe_square_n(t16, t15, 100u);
  fe_mul(t17, t16, t15);
  fe_square_n(t18, t17, 50u);
  fe_mul(power_22501, t18, t13);
}

MONERO_CUDA_DEVICE void fe_inverse(Fe &out, const Fe &input) {
  Fe power_22501, power_11, squared;
  fe_pow22501(power_22501, power_11, input);
  fe_square_n(squared, power_22501, 5u);
  fe_mul(out, squared, power_11);
}

MONERO_CUDA_DEVICE void fe_square_root(Fe &out, const Fe &input) {
  Fe power_22501, unused_power_11, squared, input_squared;
  fe_pow22501(power_22501, unused_power_11, input);
  fe_square_n(squared, power_22501, 2u);
  fe_square(input_squared, input);
  fe_mul(out, squared, input_squared);
}

MONERO_CUDA_DEVICE bool encoded_y_is_canonical(const u8 *bytes) {
  const u8 top = bytes[31] & 127u;
  if (top != 127u) return top < 127u;
  for (int i = 30; i >= 1; --i)
    if (bytes[i] != 255u) return bytes[i] < 255u;
  return bytes[0] < 237u;
}

MONERO_CUDA_DEVICE void point_identity(Point &out) {
  fe_zero(out.x);
  fe_one(out.y);
  fe_one(out.z);
  fe_zero(out.t);
}

MONERO_CUDA_DEVICE void point_copy(Point &out, const Point &input) {
  fe_copy(out.x, input.x);
  fe_copy(out.y, input.y);
  fe_copy(out.z, input.z);
  fe_copy(out.t, input.t);
}

MONERO_CUDA_DEVICE void point_add(Point &out, const Point &left,
                                  const Point &right) {
  Fe a, b, c, d, e, f, g, h, temporary, curve_d;
  fe_sub(temporary, left.y, left.x);
  fe_sub(a, right.y, right.x);
  fe_mul(a, temporary, a);
  fe_add(temporary, left.y, left.x);
  fe_add(b, right.y, right.x);
  fe_mul(b, temporary, b);
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

MONERO_CUDA_DEVICE void point_double(Point &out, const Point &input) {
  Fe a, b, c, d, e, f, g, h, temporary;
  fe_square(a, input.x);
  fe_square(b, input.y);
  fe_square(c, input.z);
  fe_add(c, c, c);
  fe_neg(d, a);
  fe_add(temporary, input.x, input.y);
  fe_square(e, temporary);
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

MONERO_CUDA_DEVICE void point_cmov(Point &target, const Point &source,
                                   u32 choose_source) {
  const u64 mask = 0ull - static_cast<u64>(choose_source & 1u);
  for (u32 i = 0; i < 16; ++i) {
    target.x.v[i] = (target.x.v[i] & ~mask) | (source.x.v[i] & mask);
    target.y.v[i] = (target.y.v[i] & ~mask) | (source.y.v[i] & mask);
    target.z.v[i] = (target.z.v[i] & ~mask) | (source.z.v[i] & mask);
    target.t.v[i] = (target.t.v[i] & ~mask) | (source.t.v[i] & mask);
  }
}

MONERO_CUDA_DEVICE bool point_from_compressed(Point &out, const u8 *bytes) {
  if (!encoded_y_is_canonical(bytes)) return false;
  Fe y_squared, numerator, denominator, inverse;
  Fe x_squared, x, check, constant;
  fe_from_bytes(out.y, bytes);
  fe_one(out.z);
  fe_square(y_squared, out.y);
  fe_sub(numerator, y_squared, out.z);
  fe_from_constant(constant, FIELD_D);
  fe_mul(denominator, y_squared, constant);
  fe_add(denominator, denominator, out.z);
  fe_inverse(inverse, denominator);
  fe_mul(x_squared, numerator, inverse);
  fe_square_root(x, x_squared);
  fe_square(check, x);
  if (!fe_equal(check, x_squared)) {
    fe_from_constant(constant, SQRT_M1);
    fe_mul(x, x, constant);
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

MONERO_CUDA_DEVICE void point_to_compressed(u8 *out, const Point &input) {
  Fe inverse, x, y;
  fe_inverse(inverse, input.z);
  fe_mul(x, input.x, inverse);
  fe_mul(y, input.y, inverse);
  fe_to_bytes(out, y);
  out[31] ^= static_cast<u8>(fe_is_negative(x) ? 128u : 0u);
}

MONERO_CUDA_DEVICE void scalar_conditional_subtract_l(u8 *scalar) {
  u32 difference[32];
  u32 borrow = 0;
  for (u32 i = 0; i < 32; ++i) {
    const u32 subtrahend = static_cast<u32>(SCALAR_L[i]) + borrow;
    const u32 value = static_cast<u32>(scalar[i]);
    difference[i] = (value - subtrahend) & 255u;
    borrow = static_cast<u32>(value < subtrahend);
  }
  const u32 mask = 0u - (borrow ^ 1u);
  for (u32 i = 0; i < 32; ++i)
    scalar[i] =
        static_cast<u8>((static_cast<u32>(scalar[i]) & ~mask) |
                        (difference[i] & mask));
}

MONERO_CUDA_DEVICE void scalar_reduce_mod_l(u8 *out, const u8 *input) {
  for (u32 i = 0; i < 32; ++i) out[i] = 0;
  for (int bit = 255; bit >= 0; --bit) {
    u32 carry = static_cast<u32>(
        (input[static_cast<u32>(bit) >> 3] >>
         (static_cast<u32>(bit) & 7u)) &
        1u);
    for (u32 i = 0; i < 32; ++i) {
      const u32 value = (static_cast<u32>(out[i]) << 1) | carry;
      out[i] = static_cast<u8>(value & 255u);
      carry = value >> 8;
    }
    scalar_conditional_subtract_l(out);
  }
}

MONERO_CUDA_DEVICE void scalar_multiply_by_eight_mod_l(u8 *scalar) {
  u32 carry = 0;
  for (u32 i = 0; i < 32; ++i) {
    const u32 value = (static_cast<u32>(scalar[i]) << 3) | carry;
    scalar[i] = static_cast<u8>(value & 255u);
    carry = value >> 8;
  }
  for (u32 pass = 0; pass < 8; ++pass) scalar_conditional_subtract_l(scalar);
}

MONERO_CUDA_DEVICE void fold_scalar(u8 *out, const u8 *input) {
  scalar_reduce_mod_l(out, input);
  scalar_multiply_by_eight_mod_l(out);
}

MONERO_CUDA_DEVICE void point_scalar_multiply_ladder(
    Point &out, const Point &point, const u8 *scalar) {
  Point accumulator, doubled, added;
  point_identity(accumulator);
  for (int bit = 255; bit >= 0; --bit) {
    point_double(doubled, accumulator);
    point_copy(accumulator, doubled);
    point_add(added, accumulator, point);
    const u32 selected =
        static_cast<u32>((scalar[static_cast<u32>(bit) >> 3] >>
                          (static_cast<u32>(bit) & 7u)) &
                         1u);
    point_cmov(accumulator, added, selected);
  }
  point_copy(out, accumulator);
}

MONERO_CUDA_DEVICE void p2_from_p1p1(PointP2 &out,
                                     const PointP1P1 &input) {
  fe_mul(out.x, input.x, input.t);
  fe_mul(out.y, input.y, input.z);
  fe_mul(out.z, input.z, input.t);
}

MONERO_CUDA_DEVICE void p3_from_p1p1(Point &out,
                                     const PointP1P1 &input) {
  fe_mul(out.x, input.x, input.t);
  fe_mul(out.y, input.y, input.z);
  fe_mul(out.z, input.z, input.t);
  fe_mul(out.t, input.x, input.y);
}

MONERO_CUDA_DEVICE void p2_double(PointP1P1 &out, const PointP2 &input) {
  Fe xx, yy, zz2, x_plus_y, x_plus_y_squared, yy_plus_xx, yy_minus_xx;
  fe_square(xx, input.x);
  fe_square(yy, input.y);
  fe_square(zz2, input.z);
  fe_add(zz2, zz2, zz2);
  fe_add(x_plus_y, input.x, input.y);
  fe_square(x_plus_y_squared, x_plus_y);
  fe_add(yy_plus_xx, yy, xx);
  fe_sub(yy_minus_xx, yy, xx);
  fe_sub(out.x, x_plus_y_squared, yy_plus_xx);
  fe_copy(out.y, yy_plus_xx);
  fe_copy(out.z, yy_minus_xx);
  fe_sub(out.t, zz2, yy_minus_xx);
}

MONERO_CUDA_DEVICE void niels_identity(ProjectiveNiels &out) {
  fe_one(out.y_plus_x);
  fe_one(out.y_minus_x);
  fe_one(out.z);
  fe_zero(out.t2d);
}

MONERO_CUDA_DEVICE void niels_from_p3(ProjectiveNiels &out,
                                      const Point &input) {
  Fe curve_d;
  fe_add(out.y_plus_x, input.y, input.x);
  fe_sub(out.y_minus_x, input.y, input.x);
  fe_copy(out.z, input.z);
  fe_from_constant(curve_d, FIELD_D);
  fe_mul(out.t2d, input.t, curve_d);
  fe_add(out.t2d, out.t2d, out.t2d);
}

MONERO_CUDA_DEVICE void niels_cmov(ProjectiveNiels &target,
                                   const ProjectiveNiels &source,
                                   u32 choose_source) {
  const u64 mask = 0ull - static_cast<u64>(choose_source & 1u);
  for (u32 i = 0; i < 16; ++i) {
    target.y_plus_x.v[i] =
        (target.y_plus_x.v[i] & ~mask) | (source.y_plus_x.v[i] & mask);
    target.y_minus_x.v[i] =
        (target.y_minus_x.v[i] & ~mask) | (source.y_minus_x.v[i] & mask);
    target.z.v[i] = (target.z.v[i] & ~mask) | (source.z.v[i] & mask);
    target.t2d.v[i] =
        (target.t2d.v[i] & ~mask) | (source.t2d.v[i] & mask);
  }
}

MONERO_CUDA_DEVICE u32 ct_equal_u32(u32 left, u32 right) {
  const u32 difference = left ^ right;
  return ((difference | (0u - difference)) >> 31u) ^ 1u;
}

MONERO_CUDA_DEVICE void niels_select_signed(
    ProjectiveNiels &out, const ProjectiveNiels *table, int digit) {
  const u32 raw = static_cast<u32>(digit);
  const u32 negative = raw >> 31u;
  const u32 sign_mask = 0u - negative;
  const u32 absolute = (raw ^ sign_mask) + negative;
  niels_identity(out);
  for (u32 value = 1; value <= 8; ++value)
    niels_cmov(out, table[value - 1u], ct_equal_u32(absolute, value));
  ProjectiveNiels negated;
  fe_copy(negated.y_plus_x, out.y_minus_x);
  fe_copy(negated.y_minus_x, out.y_plus_x);
  fe_copy(negated.z, out.z);
  fe_neg(negated.t2d, out.t2d);
  niels_cmov(out, negated, negative);
}

MONERO_CUDA_DEVICE void p3_add_niels(PointP1P1 &out, const Point &left,
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

MONERO_CUDA_DEVICE void scalar_as_radix_16(int *digits,
                                           const u8 *scalar) {
  for (u32 i = 0; i < 32; ++i) {
    digits[2u * i] = static_cast<int>(scalar[i] & 15u);
    digits[2u * i + 1u] = static_cast<int>(scalar[i] >> 4u);
  }
  for (u32 i = 0; i < 63; ++i) {
    const int carry = (digits[i] + 8) >> 4;
    digits[i] -= carry << 4;
    digits[i + 1u] += carry;
  }
}

MONERO_CUDA_DEVICE void point_scalar_multiply_radix16(
    Point &out, const Point &point, const u8 *scalar) {
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

MONERO_CUDA_DEVICE bool derive_ladder(u8 *result, const u8 *folded_scalar,
                                      const u8 *encoded_point) {
  Point point, derived;
  if (!point_from_compressed(point, encoded_point)) {
    for (u32 i = 0; i < 32; ++i) result[i] = 0;
    return false;
  }
  point_scalar_multiply_ladder(derived, point, folded_scalar);
  point_to_compressed(result, derived);
  return true;
}

MONERO_CUDA_DEVICE bool derive_radix16(u8 *result,
                                       const u8 *folded_scalar,
                                       const u8 *encoded_point) {
  Point point, derived;
  if (!point_from_compressed(point, encoded_point)) {
    for (u32 i = 0; i < 32; ++i) result[i] = 0;
    return false;
  }
  point_scalar_multiply_radix16(derived, point, folded_scalar);
  point_to_compressed(result, derived);
  return true;
}

}  // namespace monero_cuda

#undef MONERO_CUDA_DEVICE
#undef MONERO_CUDA_CONSTANT
