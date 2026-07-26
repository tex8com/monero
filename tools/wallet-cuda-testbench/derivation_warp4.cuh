#pragma once

#include "derivation_radix2625.cuh"

#if defined(__CUDACC__)

// Experimental four-lane cooperative backend for sm_86.
//
// A conventional thread keeps every limb of every live field element in its
// own register file.  The C6/C7 kernel consequently reaches CUDA's
// 255-register ceiling and can keep only eight warps resident per Ampere SM.
// This backend distributes each ten-limb radix-2^25.5 field element over a
// four-lane tile:
//
//   lane 0: limbs 0, 4, 8
//   lane 1: limbs 1, 5, 9
//   lane 2: limbs 2, 6
//   lane 3: limbs 3, 7
//
// Eight independent points therefore share one hardware warp.  Field
// multiplication and squaring use tile shuffles to obtain remote limbs while
// each lane accumulates only the coefficients it owns.  The mathematical
// formulas, carry schedule, point formulas, sqrt_ratio_i chain, and signed
// radix-8 scalar multiplication are the same ones used by the byte-checked
// scalar C6/C7 backend.

namespace monero_cuda_warp4 {

using monero_cuda2625::u8;
using monero_cuda2625::u32;
using monero_cuda2625::u64;

constexpr u32 TILE_WIDTH = 4u;
constexpr u32 LIMB_COUNT = 10u;

#define MONERO_WARP4_DEVICE __device__ __forceinline__

struct Fe {
  u32 v0;
  u32 v1;
  u32 v2;
};

struct Coefficients {
  u64 v0;
  u64 v1;
  u64 v2;
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

struct NielsTable4 {
  ProjectiveNiels p1;
  ProjectiveNiels p2;
  ProjectiveNiels p3;
  ProjectiveNiels p4;
};

MONERO_WARP4_DEVICE u32 lane() {
  return threadIdx.x & (TILE_WIDTH - 1u);
}

MONERO_WARP4_DEVICE u32 limb_index(u32 slot) {
  return lane() + TILE_WIDTH * slot;
}

MONERO_WARP4_DEVICE u32 local_slot(const Fe &value, u32 slot) {
  return slot == 0u ? value.v0 : (slot == 1u ? value.v1 : value.v2);
}

MONERO_WARP4_DEVICE u64 local_slot(const Coefficients &value, u32 slot) {
  return slot == 0u ? value.v0 : (slot == 1u ? value.v1 : value.v2);
}

MONERO_WARP4_DEVICE void set_local_slot(Fe &value, u32 slot, u32 limb) {
  if (slot == 0u)
    value.v0 = limb;
  else if (slot == 1u)
    value.v1 = limb;
  else
    value.v2 = limb;
}

MONERO_WARP4_DEVICE void set_local_slot(
    Coefficients &value, u32 slot, u64 coefficient) {
  if (slot == 0u)
    value.v0 = coefficient;
  else if (slot == 1u)
    value.v1 = coefficient;
  else
    value.v2 = coefficient;
}

MONERO_WARP4_DEVICE u32 fe_get(const Fe &value, u32 index) {
  const int source_lane = int(index & 3u);
  const u32 slot0 =
      __shfl_sync(__activemask(), value.v0, source_lane, 4);
  const u32 slot1 =
      __shfl_sync(__activemask(), value.v1, source_lane, 4);
  const u32 slot2 =
      __shfl_sync(__activemask(), value.v2, source_lane, 4);
  const u32 slot = index >> 2u;
  return slot == 0u ? slot0 : (slot == 1u ? slot1 : slot2);
}

MONERO_WARP4_DEVICE u64 coefficients_get(
    const Coefficients &value, u32 index) {
  const int source_lane = int(index & 3u);
  const u64 slot0 =
      __shfl_sync(__activemask(), value.v0, source_lane, 4);
  const u64 slot1 =
      __shfl_sync(__activemask(), value.v1, source_lane, 4);
  const u64 slot2 =
      __shfl_sync(__activemask(), value.v2, source_lane, 4);
  const u32 slot = index >> 2u;
  return slot == 0u ? slot0 : (slot == 1u ? slot1 : slot2);
}

MONERO_WARP4_DEVICE void coefficients_add(
    Coefficients &value, u32 index, u64 addend) {
  if (lane() != (index & 3u)) return;
  const u32 slot = index >> 2u;
  set_local_slot(value, slot, local_slot(value, slot) + addend);
}

MONERO_WARP4_DEVICE void coefficients_mask(
    Coefficients &value, u32 index, u64 mask) {
  if (lane() != (index & 3u)) return;
  const u32 slot = index >> 2u;
  set_local_slot(value, slot, local_slot(value, slot) & mask);
}

MONERO_WARP4_DEVICE void carry(
    Coefficients &value, u32 from, u32 to, u32 bits,
    u64 multiplier = 1u) {
  const u64 source = coefficients_get(value, from);
  coefficients_mask(value, from, (u64(1) << bits) - 1u);
  coefficients_add(value, to, multiplier * (source >> bits));
}

MONERO_WARP4_DEVICE void fe_reduce_coefficients(
    Fe &out, Coefficients value) {
  carry(value, 0u, 1u, 26u);
  carry(value, 4u, 5u, 26u);
  carry(value, 1u, 2u, 25u);
  carry(value, 5u, 6u, 25u);
  carry(value, 2u, 3u, 26u);
  carry(value, 6u, 7u, 26u);
  carry(value, 3u, 4u, 25u);
  carry(value, 7u, 8u, 25u);
  carry(value, 4u, 5u, 26u);
  carry(value, 8u, 9u, 26u);
  carry(value, 9u, 0u, 25u, 19u);
  carry(value, 0u, 1u, 26u);

  out.v0 = u32(value.v0);
  out.v1 = u32(value.v1);
  out.v2 = limb_index(2u) < LIMB_COUNT ? u32(value.v2) : 0u;
}

MONERO_WARP4_DEVICE void fe_zero(Fe &out) {
  out.v0 = 0u;
  out.v1 = 0u;
  out.v2 = 0u;
}

MONERO_WARP4_DEVICE void fe_one(Fe &out) {
  fe_zero(out);
  if (lane() == 0u) out.v0 = 1u;
}

MONERO_WARP4_DEVICE void fe_copy(Fe &out, const Fe &input) {
  out = input;
}

MONERO_WARP4_DEVICE void fe_from_field_d(Fe &out) {
  const u32 l = lane();
  out.v0 = monero_cuda2625::FIELD_D_LIMBS[l];
  out.v1 = monero_cuda2625::FIELD_D_LIMBS[l + 4u];
  out.v2 =
      l < 2u ? monero_cuda2625::FIELD_D_LIMBS[l + 8u] : 0u;
}

MONERO_WARP4_DEVICE void fe_from_sqrt_m1(Fe &out) {
  const u32 l = lane();
  out.v0 = monero_cuda2625::SQRT_M1_LIMBS[l];
  out.v1 = monero_cuda2625::SQRT_M1_LIMBS[l + 4u];
  out.v2 =
      l < 2u ? monero_cuda2625::SQRT_M1_LIMBS[l + 8u] : 0u;
}

MONERO_WARP4_DEVICE u64 load3(const u8 *bytes, u32 index) {
  return u64(bytes[index]) |
         (u64(bytes[index + 1u]) << 8u) |
         (u64(bytes[index + 2u]) << 16u);
}

MONERO_WARP4_DEVICE u64 load4(const u8 *bytes, u32 index) {
  return load3(bytes, index) |
         (u64(bytes[index + 3u]) << 24u);
}

MONERO_WARP4_DEVICE u64 encoded_coefficient(
    const u8 *bytes, u32 index) {
  switch (index) {
    case 0u:
      return load4(bytes, 0u);
    case 1u:
      return load3(bytes, 4u) << 6u;
    case 2u:
      return load3(bytes, 7u) << 5u;
    case 3u:
      return load3(bytes, 10u) << 3u;
    case 4u:
      return load3(bytes, 13u) << 2u;
    case 5u:
      return load4(bytes, 16u);
    case 6u:
      return load3(bytes, 20u) << 7u;
    case 7u:
      return load3(bytes, 23u) << 5u;
    case 8u:
      return load3(bytes, 26u) << 4u;
    case 9u:
      return (load3(bytes, 29u) & ((u64(1) << 23u) - 1u))
             << 2u;
    default:
      return 0u;
  }
}

MONERO_WARP4_DEVICE void fe_from_bytes(Fe &out, const u8 *bytes) {
  Coefficients value;
  value.v0 = encoded_coefficient(bytes, limb_index(0u));
  value.v1 = encoded_coefficient(bytes, limb_index(1u));
  value.v2 = encoded_coefficient(bytes, limb_index(2u));
  fe_reduce_coefficients(out, value);
}

MONERO_WARP4_DEVICE void fe_add(
    Fe &out, const Fe &left, const Fe &right) {
  out.v0 = left.v0 + right.v0;
  out.v1 = left.v1 + right.v1;
  out.v2 = left.v2 + right.v2;
}

MONERO_WARP4_DEVICE u64 subtraction_bias(u32 index) {
  if (index >= LIMB_COUNT) return 0u;
  const u32 limb_bias =
      index == 0u ? 0x3ffffedu :
      ((index & 1u) == 0u ? 0x3ffffffu : 0x1ffffffu);
  return u64(limb_bias) << 4u;
}

MONERO_WARP4_DEVICE void fe_sub(
    Fe &out, const Fe &left, const Fe &right) {
  Coefficients value;
  const u32 k0 = limb_index(0u);
  const u32 k1 = limb_index(1u);
  const u32 k2 = limb_index(2u);
  value.v0 =
      u64(left.v0) + subtraction_bias(k0) - u64(right.v0);
  value.v1 =
      u64(left.v1) + subtraction_bias(k1) - u64(right.v1);
  value.v2 = k2 < LIMB_COUNT
                 ? u64(left.v2) + subtraction_bias(k2) -
                       u64(right.v2)
                 : 0u;
  fe_reduce_coefficients(out, value);
}

MONERO_WARP4_DEVICE void fe_neg(Fe &out, const Fe &input) {
  Fe zero;
  fe_zero(zero);
  fe_sub(out, zero, input);
}

MONERO_WARP4_DEVICE u64 multiplication_coefficient(
    const Fe &left, const Fe &right, u32 k) {
  const bool valid_limb = k < LIMB_COUNT;
  const u32 effective_k = valid_limb ? k : 0u;
  u64 accumulator = 0u;
  for (u32 i = 0u; i < LIMB_COUNT; ++i) {
    int raw_j = int(effective_k) - int(i);
    const bool wrapped = raw_j < 0;
    if (wrapped) raw_j += int(LIMB_COUNT);
    u32 factor = wrapped ? 19u : 1u;
    if ((effective_k & 1u) == 0u && (i & 1u) != 0u)
      factor *= 2u;
    accumulator +=
        u64(fe_get(left, i)) * u64(fe_get(right, u32(raw_j))) *
        u64(factor);
  }
  return valid_limb ? accumulator : 0u;
}

MONERO_WARP4_DEVICE void fe_mul(
    Fe &out, const Fe &left, const Fe &right) {
  Coefficients value;
  value.v0 = multiplication_coefficient(
      left, right, limb_index(0u));
  value.v1 = multiplication_coefficient(
      left, right, limb_index(1u));
  value.v2 = multiplication_coefficient(
      left, right, limb_index(2u));
  fe_reduce_coefficients(out, value);
}

MONERO_WARP4_DEVICE u64 square_coefficient(
    const Fe &input, u32 k) {
  const bool valid_limb = k < LIMB_COUNT;
  const u32 effective_k = valid_limb ? k : 0u;
  u64 accumulator = 0u;
  for (u32 i = 0u; i < LIMB_COUNT; ++i) {
    const u32 j =
        (effective_k + LIMB_COUNT - i) % LIMB_COUNT;
    const bool include = i <= j;

    // All four lanes must participate in every shuffle.  The coefficient
    // predicate differs between lanes, so apply it only after both reads.
    const u32 left = fe_get(input, i);
    const u32 right = fe_get(input, j);
    const u32 sum = i + j;
    const u32 reduction = sum >= LIMB_COUNT ? 19u : 1u;
    u32 factor = reduction;
    if (i == j) {
      if ((i & 1u) != 0u) factor *= 2u;
    } else if ((effective_k & 1u) == 0u &&
               (i & 1u) != 0u) {
      factor *= 4u;
    } else {
      factor *= 2u;
    }
    if (include)
      accumulator += u64(left) * u64(right) * u64(factor);
  }
  return valid_limb ? accumulator : 0u;
}

MONERO_WARP4_DEVICE void fe_square(Fe &out, const Fe &input) {
  Coefficients value;
  value.v0 = square_coefficient(input, limb_index(0u));
  value.v1 = square_coefficient(input, limb_index(1u));
  value.v2 = square_coefficient(input, limb_index(2u));
  fe_reduce_coefficients(out, value);
}

MONERO_WARP4_DEVICE void fe_reduce(Fe &value) {
  Coefficients coefficients{
      u64(value.v0), u64(value.v1), u64(value.v2)};
  fe_reduce_coefficients(value, coefficients);
}

MONERO_WARP4_DEVICE void fe_freeze(Fe &out, const Fe &input) {
  fe_copy(out, input);
  fe_reduce(out);

  u32 q = (fe_get(out, 0u) + 19u) >> 26u;
  for (u32 index = 1u; index < LIMB_COUNT; ++index) {
    const u32 bits = (index & 1u) == 0u ? 26u : 25u;
    q = (fe_get(out, index) + q) >> bits;
  }

  Coefficients value{u64(out.v0), u64(out.v1), u64(out.v2)};
  coefficients_add(value, 0u, 19u * u64(q));
  for (u32 index = 0u; index < 9u; ++index) {
    const u32 bits = (index & 1u) == 0u ? 26u : 25u;
    carry(value, index, index + 1u, bits);
  }
  coefficients_mask(value, 9u, (u64(1) << 25u) - 1u);
  out.v0 = u32(value.v0);
  out.v1 = u32(value.v1);
  out.v2 = limb_index(2u) < LIMB_COUNT ? u32(value.v2) : 0u;
}

MONERO_WARP4_DEVICE u32 tile_or(u32 value) {
  value |= __shfl_xor_sync(__activemask(), value, 1, 4);
  value |= __shfl_xor_sync(__activemask(), value, 2, 4);
  return value;
}

MONERO_WARP4_DEVICE bool fe_equal(
    const Fe &left_input, const Fe &right_input) {
  Fe left, right;
  fe_freeze(left, left_input);
  fe_freeze(right, right_input);
  u32 difference =
      (left.v0 ^ right.v0) | (left.v1 ^ right.v1);
  if (limb_index(2u) < LIMB_COUNT)
    difference |= left.v2 ^ right.v2;
  return tile_or(difference) == 0u;
}

MONERO_WARP4_DEVICE bool fe_is_zero(const Fe &input) {
  Fe value;
  fe_freeze(value, input);
  u32 aggregate = value.v0 | value.v1;
  if (limb_index(2u) < LIMB_COUNT) aggregate |= value.v2;
  return tile_or(aggregate) == 0u;
}

MONERO_WARP4_DEVICE bool fe_is_negative(const Fe &input) {
  Fe value;
  fe_freeze(value, input);
  return (fe_get(value, 0u) & 1u) != 0u;
}

MONERO_WARP4_DEVICE void fe_cmov(
    Fe &target, const Fe &source, u32 choose_source) {
  const u32 mask = 0u - (choose_source & 1u);
  target.v0 = (target.v0 & ~mask) | (source.v0 & mask);
  target.v1 = (target.v1 & ~mask) | (source.v1 & mask);
  target.v2 = (target.v2 & ~mask) | (source.v2 & mask);
}

MONERO_WARP4_DEVICE void fe_square_n(
    Fe &out, const Fe &input, u32 count) {
  Fe accumulator, squared;
  fe_copy(accumulator, input);
  for (u32 index = 0u; index < count; ++index) {
    fe_square(squared, accumulator);
    fe_copy(accumulator, squared);
  }
  fe_copy(out, accumulator);
}

MONERO_WARP4_DEVICE void fe_pow22501(
    Fe &power_22501, Fe &power_11, const Fe &input) {
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

MONERO_WARP4_DEVICE void fe_pow_p58(Fe &out, const Fe &input) {
  Fe power_22501, unused_power_11, squared;
  fe_pow22501(power_22501, unused_power_11, input);
  fe_square_n(squared, power_22501, 2u);
  fe_mul(out, input, squared);
}

MONERO_WARP4_DEVICE bool fe_sqrt_ratio_i(
    Fe &out, const Fe &u, const Fe &v) {
  Fe v_squared, v3, v3_squared, v7;
  fe_square(v_squared, v);
  fe_mul(v3, v_squared, v);
  fe_square(v3_squared, v3);
  fe_mul(v7, v3_squared, v);

  Fe u_v3, u_v7, power, root;
  fe_mul(u_v3, u, v3);
  fe_mul(u_v7, u, v7);
  fe_pow_p58(power, u_v7);
  fe_mul(root, u_v3, power);

  Fe root_squared, check, negative_u, sqrt_m1, negative_u_i;
  fe_square(root_squared, root);
  fe_mul(check, v, root_squared);
  fe_neg(negative_u, u);
  fe_from_sqrt_m1(sqrt_m1);
  fe_mul(negative_u_i, negative_u, sqrt_m1);

  const u32 correct_sign_sqrt = fe_equal(check, u) ? 1u : 0u;
  const u32 flipped_sign_sqrt =
      fe_equal(check, negative_u) ? 1u : 0u;
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

MONERO_WARP4_DEVICE bool encoded_y_is_canonical(const u8 *bytes) {
  u32 canonical = 0u;
  if (lane() == 0u) {
    const u8 top = bytes[31] & 127u;
    if (top != 127u) {
      canonical = top < 127u ? 1u : 0u;
    } else {
      canonical = 1u;
      for (int index = 30; index >= 1; --index) {
        if (bytes[index] != 255u) {
          canonical = bytes[index] < 255u ? 1u : 0u;
          break;
        }
        if (index == 1) canonical = bytes[0] < 237u ? 1u : 0u;
      }
    }
  }
  canonical =
      __shfl_sync(__activemask(), canonical, 0, 4);
  return canonical != 0u;
}

MONERO_WARP4_DEVICE bool point_from_compressed(
    Point &out, const u8 *bytes) {
  if (!encoded_y_is_canonical(bytes)) return false;
  Fe y_squared, numerator, denominator, curve_d, x;
  fe_from_bytes(out.y, bytes);
  fe_one(out.z);
  fe_square(y_squared, out.y);
  fe_sub(numerator, y_squared, out.z);
  fe_from_field_d(curve_d);
  fe_mul(denominator, y_squared, curve_d);
  fe_add(denominator, denominator, out.z);
  if (!fe_sqrt_ratio_i(x, numerator, denominator)) return false;

  const bool requested_negative = (bytes[31] >> 7u) != 0u;
  if (fe_is_negative(x) != requested_negative) {
    if (fe_is_zero(x)) return false;
    fe_neg(x, x);
  }
  fe_copy(out.x, x);
  fe_mul(out.t, out.x, out.y);
  return true;
}

MONERO_WARP4_DEVICE void point_identity(Point &out) {
  fe_zero(out.x);
  fe_one(out.y);
  fe_one(out.z);
  fe_zero(out.t);
}

MONERO_WARP4_DEVICE void p2_from_p1p1(
    PointP2 &out, const PointP1P1 &input) {
  fe_mul(out.x, input.x, input.t);
  fe_mul(out.y, input.y, input.z);
  fe_mul(out.z, input.z, input.t);
}

MONERO_WARP4_DEVICE void p3_from_p1p1(
    Point &out, const PointP1P1 &input) {
  fe_mul(out.x, input.x, input.t);
  fe_mul(out.y, input.y, input.z);
  fe_mul(out.z, input.z, input.t);
  fe_mul(out.t, input.x, input.y);
}

MONERO_WARP4_DEVICE void p2_double(
    PointP1P1 &out, const PointP2 &input) {
  Fe xx, yy, zz2, x_plus_y, x_plus_y_sq;
  Fe yy_plus_xx, yy_minus_xx;
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

MONERO_WARP4_DEVICE void niels_identity(ProjectiveNiels &out) {
  fe_one(out.y_plus_x);
  fe_one(out.y_minus_x);
  fe_one(out.z);
  fe_zero(out.t2d);
}

MONERO_WARP4_DEVICE void niels_from_p3(
    ProjectiveNiels &out, const Point &input) {
  Fe curve_d;
  fe_add(out.y_plus_x, input.y, input.x);
  fe_sub(out.y_minus_x, input.y, input.x);
  fe_copy(out.z, input.z);
  fe_from_field_d(curve_d);
  fe_mul(out.t2d, input.t, curve_d);
  fe_add(out.t2d, out.t2d, out.t2d);
}

MONERO_WARP4_DEVICE void niels_cmov(
    ProjectiveNiels &target, const ProjectiveNiels &source,
    u32 choose_source) {
  fe_cmov(target.y_plus_x, source.y_plus_x, choose_source);
  fe_cmov(target.y_minus_x, source.y_minus_x, choose_source);
  fe_cmov(target.z, source.z, choose_source);
  fe_cmov(target.t2d, source.t2d, choose_source);
}

MONERO_WARP4_DEVICE u32 ct_equal_u32(u32 left, u32 right) {
  const u32 difference = left ^ right;
  return ((difference | (0u - difference)) >> 31u) ^ 1u;
}

MONERO_WARP4_DEVICE void niels_select_signed(
    ProjectiveNiels &out, const NielsTable4 &table, int digit) {
  const int sign_mask = digit >> 31;
  const u32 absolute = u32((digit + sign_mask) ^ sign_mask);
  niels_identity(out);
  niels_cmov(out, table.p1, ct_equal_u32(absolute, 1u));
  niels_cmov(out, table.p2, ct_equal_u32(absolute, 2u));
  niels_cmov(out, table.p3, ct_equal_u32(absolute, 3u));
  niels_cmov(out, table.p4, ct_equal_u32(absolute, 4u));

  ProjectiveNiels negated;
  fe_copy(negated.y_plus_x, out.y_minus_x);
  fe_copy(negated.y_minus_x, out.y_plus_x);
  fe_copy(negated.z, out.z);
  fe_neg(negated.t2d, out.t2d);
  niels_cmov(out, negated, u32(sign_mask) & 1u);
}

MONERO_WARP4_DEVICE void p3_add_niels(
    PointP1P1 &out, const Point &left,
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

MONERO_WARP4_DEVICE void point_scalar_multiply_radix8(
    Point &out, const Point &point, const char *digits) {
  NielsTable4 table;
  niels_from_p3(table.p1, point);

  PointP1P1 sum;
  Point next;
  p3_add_niels(sum, point, table.p1);
  p3_from_p1p1(next, sum);
  niels_from_p3(table.p2, next);
  p3_add_niels(sum, point, table.p2);
  p3_from_p1p1(next, sum);
  niels_from_p3(table.p3, next);
  p3_add_niels(sum, point, table.p3);
  p3_from_p1p1(next, sum);
  niels_from_p3(table.p4, next);

  Point accumulator;
  PointP1P1 completed;
  ProjectiveNiels selected;
  point_identity(accumulator);
  niels_select_signed(selected, table, int(digits[85]));
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
    niels_select_signed(selected, table, int(digits[index]));
    p3_add_niels(completed, accumulator, selected);
  }
  p3_from_p1p1(out, completed);
}

MONERO_WARP4_DEVICE void fe_store_device(u32 *out, const Fe &input) {
  const u32 l = lane();
  out[l] = input.v0;
  out[l + 4u] = input.v1;
  if (l < 2u) out[l + 8u] = input.v2;
}

#undef MONERO_WARP4_DEVICE

}  // namespace monero_cuda_warp4

#endif  // defined(__CUDACC__)
