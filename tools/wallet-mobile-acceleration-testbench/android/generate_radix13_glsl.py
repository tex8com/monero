#!/usr/bin/env python3
"""Generate a Vulkan GLSL correctness kernel from the byte-checked Metal M13.

The generated shader intentionally stores every byte in one uint SSBO element.
That layout is not a performance claim; it avoids optional 8-bit storage
features while the first physical-device Vulkan derivation path is validated.
"""

from __future__ import annotations

import argparse
import pathlib
import re


def remove_metal_kernels(source: str) -> str:
    lines = source.splitlines(keepends=True)
    output: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index].startswith("kernel void "):
            depth = 0
            saw_opening = False
            while index < len(lines):
                line = lines[index]
                depth += line.count("{")
                if "{" in line:
                    saw_opening = True
                depth -= line.count("}")
                index += 1
                if saw_opening and depth == 0:
                    break
            continue
        output.append(lines[index])
        index += 1
    return "".join(output)


def transform_helpers(source: str) -> str:
    start = source.index("struct Fe")
    source = source[start:]
    source = remove_metal_kernels(source)
    source = re.sub(r"^struct Parameters[^\n]*\n", "", source, flags=re.MULTILINE)

    # Avoid colliding with GLSL parameter/storage qualifiers.
    source = re.sub(r"\bout\b", "resultValue", source)
    source = re.sub(r"\binput\b", "inputValue", source)
    source = source.replace("inline ", "")
    source = source.replace("thread const ", "const ")
    source = source.replace("constant const ", "const ")
    source = source.replace("device const ", "const ")
    source = source.replace("thread ", "")
    source = source.replace("constant ", "const ")
    # SSBO bytes are widened to uint for portable Vulkan storage. Preserve
    # Metal uchar's truncating conversion where fe_to_bytes ORs shifted field
    # limbs into logical bytes.
    source = source.replace("uchar(word)", "(word & 255u)")
    source = source.replace("uchar(word >> 8)", "((word >> 8) & 255u)")
    source = source.replace("uchar(word >> 16)", "((word >> 16) & 255u)")
    source = source.replace("uchar", "uint")
    source = source.replace("ulong", "uint")
    source = re.sub(r"([0-9a-fA-Fx]+)ul\b", r"\1u", source)

    aggregate_types = r"(Fe|Point|PointP2|PointP1P1|ProjectiveNiels)"
    source = re.sub(
        rf"const {aggregate_types} &([A-Za-z_][A-Za-z0-9_]*)",
        r"const \1 \2",
        source,
    )
    source = re.sub(
        rf"{aggregate_types} &resultValue",
        r"out \1 resultValue",
        source,
    )
    source = re.sub(
        rf"{aggregate_types} &(power_22501|power_11)",
        r"out \1 \2",
        source,
    )
    source = re.sub(
        rf"{aggregate_types} &([A-Za-z_][A-Za-z0-9_]*)",
        r"inout \1 \2",
        source,
    )

    source = source.replace("uint *resultValue", "out uint resultValue[32]")
    source = source.replace("uint *z", "inout uint z[40]")
    source = source.replace("int *digits", "out int digits[64]")
    source = source.replace(
        "const ProjectiveNiels *table",
        "const ProjectiveNiels table[8]",
    )
    source = source.replace("const uint *bytes", "const uint bytes[32]")
    source = source.replace("const uint *exponent", "const uint exponent[32]")
    source = source.replace(
        "const uint *inputValue",
        "const uint inputValue[32]",
    )
    source = source.replace("const uint *scalar", "const uint scalar[32]")
    source = source.replace("uint *scalar", "inout uint scalar[32]")

    source = re.sub(
        r"const uint ([A-Z0-9_]+)\[32\] = \{(.*?)\};",
        lambda match: (
            f"const uint {match.group(1)}[32] = uint[32]("
            f"{match.group(2)});"
        ),
        source,
        flags=re.DOTALL,
    )
    zero_array = (
        "uint {name}[40];\n"
        "  for (uint zeroIndex = 0u; zeroIndex < 40u; ++zeroIndex) "
        "{name}[zeroIndex] = 0u;"
    )
    source = source.replace(
        "uint z[40] = {0};",
        zero_array.format(name="z"),
    )
    source = source.replace(
        "uint product[40] = {0};",
        zero_array.format(name="product"),
    )

    return source


GLSL_HEADER = """#version 450
#extension GL_EXT_control_flow_attributes : require

// Generated from the byte-checked pure-u32 Metal M13 field and point helpers.
// One uint stores one logical byte so no optional 8-bit storage is required.

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer ScalarBuffer {
  uint viewScalar[];
};
layout(set = 0, binding = 1, std430) readonly buffer PointBuffer {
  uint compressedPoints[];
};
layout(set = 0, binding = 2, std430) buffer ScratchPointBuffer {
  uint scratchPoints[];
};
layout(set = 0, binding = 3, std430) buffer MultiplyStateBuffer {
  uint multiplyState[];
};
layout(set = 0, binding = 4, std430) writeonly buffer ResultBuffer {
  uint derivedResults[];
};
layout(set = 0, binding = 5, std430) buffer ValidBuffer {
  uint derivationValid[];
};
layout(set = 0, binding = 6, std430) buffer ScalarDigitBuffer {
  uint scalarDigits[];
};
layout(set = 0, binding = 7, std430) buffer NielsTableBuffer {
  uint nielsTables[];
};
layout(push_constant) uniform Parameters {
  uint count;
  uint loopBias;
  uint stepIndex;
} parameters;

shared uint foldedScalar[32];

"""


GLSL_MAIN = """
void main() {
  uint id = gl_GlobalInvocationID.x;
  uint lane = gl_LocalInvocationID.x;
  if (lane == 0u) {
    uint scalarInput[32];
    uint localScalar[32];
    for (uint i = 0u; i < 32u; ++i) scalarInput[i] = viewScalar[i];
    scalar_reduce_mod_l(localScalar, scalarInput);
    scalar_multiply_by_eight_mod_l(localScalar);
    for (uint i = 0u; i < 32u; ++i) foldedScalar[i] = localScalar[i];
  }
  barrier();
  if (id >= parameters.count) return;

  uint offset = id * 32u;
  uint encodedPoint[32];
  uint scalar[32];
  uint resultBytes[32];
  for (uint i = 0u; i < 32u; ++i) {
    encodedPoint[i] = compressedPoints[offset + i];
    scalar[i] = foldedScalar[i];
  }

  Point point;
  Point derived;
  if (!point_from_compressed_m5(point, encodedPoint)) {
    for (uint i = 0u; i < 32u; ++i) derivedResults[offset + i] = 0u;
    derivationValid[id] = 0u;
    return;
  }
  point_scalar_multiply_radix16(derived, point, scalar);
  point_to_compressed_m5(resultBytes, derived);
  for (uint i = 0u; i < 32u; ++i) {
    derivedResults[offset + i] = resultBytes[i];
  }
  derivationValid[id] = 1u;
}
"""


DECODE_MAIN = """
void store_fe20(uint base, const Fe value) {
  for (uint i = 0u; i < 20u; ++i) decodedPoints[base + i] = value.v[i];
}

void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= parameters.count) return;
  uint byteOffset = id * 32u;
  uint pointOffset = id * 80u;
  uint encodedPoint[32];
  for (uint i = 0u; i < 32u; ++i)
    encodedPoint[i] = compressedPoints[byteOffset + i];

  Point point;
  if (!point_from_compressed_m5(point, encodedPoint)) {
    for (uint i = 0u; i < 80u; ++i) decodedPoints[pointOffset + i] = 0u;
    derivationValid[id] = 0u;
    return;
  }
  store_fe20(pointOffset, point.x);
  store_fe20(pointOffset + 20u, point.y);
  store_fe20(pointOffset + 40u, point.z);
  store_fe20(pointOffset + 60u, point.t);
  derivationValid[id] = 1u;
}
"""


MULTIPLY_MAIN = """
void load_fe20(out Fe value, uint base) {
  for (uint i = 0u; i < 20u; ++i) value.v[i] = decodedPoints[base + i];
}

void store_projective_fe20(uint base, const Fe value) {
  for (uint i = 0u; i < 20u; ++i) projectivePoints[base + i] = value.v[i];
}

void main() {
  uint id = gl_GlobalInvocationID.x;
  uint lane = gl_LocalInvocationID.x;
  if (lane == 0u) {
    uint scalarInput[32];
    uint localScalar[32];
    for (uint i = 0u; i < 32u; ++i) scalarInput[i] = viewScalar[i];
    scalar_reduce_mod_l(localScalar, scalarInput);
    scalar_multiply_by_eight_mod_l(localScalar);
    for (uint i = 0u; i < 32u; ++i) foldedScalar[i] = localScalar[i];
  }
  barrier();
  if (id >= parameters.count) return;

  uint pointOffset = id * 80u;
  uint projectiveOffset = id * 60u;
  if (derivationValid[id] == 0u) {
    for (uint i = 0u; i < 60u; ++i)
      projectivePoints[projectiveOffset + i] = 0u;
    return;
  }
  Point point;
  load_fe20(point.x, pointOffset);
  load_fe20(point.y, pointOffset + 20u);
  load_fe20(point.z, pointOffset + 40u);
  load_fe20(point.t, pointOffset + 60u);
  uint scalar[32];
  for (uint i = 0u; i < 32u; ++i) scalar[i] = foldedScalar[i];

  Point derived;
  point_scalar_multiply_radix16(derived, point, scalar);
  store_projective_fe20(projectiveOffset, derived.x);
  store_projective_fe20(projectiveOffset + 20u, derived.y);
  store_projective_fe20(projectiveOffset + 40u, derived.z);
}
"""


COMPRESS_MAIN = """
void load_projective_fe20(out Fe value, uint base) {
  for (uint i = 0u; i < 20u; ++i) value.v[i] = projectivePoints[base + i];
}

void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= parameters.count) return;
  uint byteOffset = id * 32u;
  if (derivationValid[id] == 0u) {
    for (uint i = 0u; i < 32u; ++i) derivedResults[byteOffset + i] = 0u;
    return;
  }

  uint projectiveOffset = id * 60u;
  Point derived;
  load_projective_fe20(derived.x, projectiveOffset);
  load_projective_fe20(derived.y, projectiveOffset + 20u);
  load_projective_fe20(derived.z, projectiveOffset + 40u);
  fe_zero(derived.t);
  uint resultBytes[32];
  point_to_compressed_m5(resultBytes, derived);
  for (uint i = 0u; i < 32u; ++i)
    derivedResults[byteOffset + i] = resultBytes[i];
}
"""

SCALAR_DIGITS_MAIN = """
void main() {
  if (gl_GlobalInvocationID.x != 0u) return;
  uint scalarInput[32];
  uint scalar[32];
  int digits[64];
  for (uint i = 0u; i < 32u; ++i) scalarInput[i] = viewScalar[i];
  scalar_reduce_mod_l(scalar, scalarInput);
  scalar_multiply_by_eight_mod_l(scalar);
  scalar_as_radix_16(digits, scalar);
  for (uint i = 0u; i < 64u; ++i) scalarDigits[i] = uint(digits[i]);
}
"""


DECODE_PREPARE_MAIN = """
void store_scratch_fe(uint base, const Fe value) {
  for (uint i = 0u; i < 20u; ++i) scratchPoints[base + i] = value.v[i];
}

void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= parameters.count) return;
  uint byteOffset = id * 32u;
  uint scratchOffset = id * 200u;
  uint encoded[32];
  for (uint i = 0u; i < 32u; ++i)
    encoded[i] = compressedPoints[byteOffset + i];
  if (!encoded_y_is_canonical(encoded)) {
    derivationValid[id] = 0u;
    return;
  }

  Fe y;
  Fe ySquared;
  Fe numerator;
  Fe denominator;
  Fe one;
  Fe curveD;
  fe_from_bytes(y, encoded);
  fe_one(one);
  fe_square(ySquared, y);
  fe_sub(numerator, ySquared, one);
  fe_from_constant(curveD, FIELD_D);
  fe_mul(denominator, ySquared, curveD);
  fe_add(denominator, denominator, one);
  store_scratch_fe(scratchOffset, y);
  store_scratch_fe(scratchOffset + 20u, numerator);
  store_scratch_fe(scratchOffset + 40u, denominator);
  derivationValid[id] = 1u;
}
"""


DECODE_INVERSE_MAIN = """
void load_scratch_fe(out Fe value, uint base) {
  for (uint i = 0u; i < 20u; ++i) value.v[i] = scratchPoints[base + i];
}
void store_scratch_fe(uint base, const Fe value) {
  for (uint i = 0u; i < 20u; ++i) scratchPoints[base + i] = value.v[i];
}

void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= parameters.count || derivationValid[id] == 0u) return;
  uint scratchOffset = id * 200u;
  Fe denominator;
  Fe inverse;
  load_scratch_fe(denominator, scratchOffset + 40u);
  fe_inverse_m5(inverse, denominator);
  store_scratch_fe(scratchOffset + 60u, inverse);
}
"""


DECODE_SQRT_MAIN = """
void load_scratch_fe(out Fe value, uint base) {
  for (uint i = 0u; i < 20u; ++i) value.v[i] = scratchPoints[base + i];
}
void store_scratch_fe(uint base, const Fe value) {
  for (uint i = 0u; i < 20u; ++i) scratchPoints[base + i] = value.v[i];
}

void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= parameters.count || derivationValid[id] == 0u) return;
  uint scratchOffset = id * 200u;
  Fe numerator;
  Fe inverse;
  Fe xSquared;
  Fe x;
  load_scratch_fe(numerator, scratchOffset + 20u);
  load_scratch_fe(inverse, scratchOffset + 60u);
  fe_mul(xSquared, numerator, inverse);
  fe_square_root_m5(x, xSquared);
  store_scratch_fe(scratchOffset + 80u, xSquared);
  store_scratch_fe(scratchOffset + 100u, x);
}
"""


DECODE_FINISH_MAIN = """
void load_scratch_fe(out Fe value, uint base) {
  for (uint i = 0u; i < 20u; ++i) value.v[i] = scratchPoints[base + i];
}
void store_scratch_fe(uint base, const Fe value) {
  for (uint i = 0u; i < 20u; ++i) scratchPoints[base + i] = value.v[i];
}

void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= parameters.count || derivationValid[id] == 0u) return;
  uint byteOffset = id * 32u;
  uint scratchOffset = id * 200u;
  Fe y;
  Fe xSquared;
  Fe x;
  Fe check;
  load_scratch_fe(y, scratchOffset);
  load_scratch_fe(xSquared, scratchOffset + 80u);
  load_scratch_fe(x, scratchOffset + 100u);
  fe_square(check, x);
  if (!fe_equal(check, xSquared)) {
    Fe sqrtM1;
    fe_from_constant(sqrtM1, SQRT_M1);
    fe_mul(x, x, sqrtM1);
    fe_square(check, x);
    if (!fe_equal(check, xSquared)) {
      derivationValid[id] = 0u;
      return;
    }
  }
  bool requestedNegative =
      (compressedPoints[byteOffset + 31u] >> 7u) != 0u;
  if (fe_is_negative(x) != requestedNegative) {
    if (fe_is_zero(x)) {
      derivationValid[id] = 0u;
      return;
    }
    Fe negated;
    fe_neg(negated, x);
    fe_copy(x, negated);
  }
  Fe z;
  Fe t;
  fe_one(z);
  fe_mul(t, x, y);
  store_scratch_fe(scratchOffset + 120u, x);
  store_scratch_fe(scratchOffset + 140u, y);
  store_scratch_fe(scratchOffset + 160u, z);
  store_scratch_fe(scratchOffset + 180u, t);
}
"""


MULTIPLY_INIT_MAIN = """
void load_scratch_fe(out Fe value, uint base) {
  for (uint i = 0u; i < 20u; ++i) value.v[i] = scratchPoints[base + i];
}
void store_state_fe(uint base, const Fe value) {
  for (uint i = 0u; i < 20u; ++i) multiplyState[base + i] = value.v[i];
}
void store_niels(uint base, const ProjectiveNiels value) {
  for (uint i = 0u; i < 20u; ++i) {
    nielsTables[base + i] = value.y_plus_x.v[i];
    nielsTables[base + 20u + i] = value.y_minus_x.v[i];
    nielsTables[base + 40u + i] = value.z.v[i];
    nielsTables[base + 60u + i] = value.t2d.v[i];
  }
}
void load_niels(out ProjectiveNiels value, uint base) {
  for (uint i = 0u; i < 20u; ++i) {
    value.y_plus_x.v[i] = nielsTables[base + i];
    value.y_minus_x.v[i] = nielsTables[base + 20u + i];
    value.z.v[i] = nielsTables[base + 40u + i];
    value.t2d.v[i] = nielsTables[base + 60u + i];
  }
}
void select_niels(out ProjectiveNiels selected, uint tableBase, int digit) {
  int signMask = digit >> 31;
  uint absolute = uint((digit + signMask) ^ signMask);
  niels_identity(selected);
  for (uint value = 1u; value <= 8u; ++value) {
    ProjectiveNiels candidate;
    load_niels(candidate, tableBase + (value - 1u) * 80u);
    niels_cmov(selected, candidate, ct_equal_u32(absolute, value));
  }
  ProjectiveNiels negated;
  fe_copy(negated.y_plus_x, selected.y_minus_x);
  fe_copy(negated.y_minus_x, selected.y_plus_x);
  fe_copy(negated.z, selected.z);
  fe_neg(negated.t2d, selected.t2d);
  niels_cmov(selected, negated, uint(signMask) & 1u);
}

void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= parameters.count || derivationValid[id] == 0u) return;
  uint scratchOffset = id * 200u;
  uint stateOffset = id * 80u;
  uint tableBase = id * 640u;
  Point point;
  load_scratch_fe(point.x, scratchOffset + 120u);
  load_scratch_fe(point.y, scratchOffset + 140u);
  load_scratch_fe(point.z, scratchOffset + 160u);
  load_scratch_fe(point.t, scratchOffset + 180u);

  ProjectiveNiels current;
  niels_from_p3(current, point);
  store_niels(tableBase, current);
  for (uint i = 0u; i < 7u; ++i) {
    PointP1P1 sum;
    Point next;
    p3_add_niels(sum, point, current);
    p3_from_p1p1(next, sum);
    niels_from_p3(current, next);
    store_niels(tableBase + (i + 1u) * 80u, current);
  }
  memoryBarrierBuffer();

  Point accumulator;
  PointP1P1 completed;
  ProjectiveNiels selected;
  point_identity(accumulator);
  select_niels(selected, tableBase, int(scalarDigits[63]));
  p3_add_niels(completed, accumulator, selected);
  store_state_fe(stateOffset, completed.x);
  store_state_fe(stateOffset + 20u, completed.y);
  store_state_fe(stateOffset + 40u, completed.z);
  store_state_fe(stateOffset + 60u, completed.t);
}
"""


MULTIPLY_STEP_MAIN = """
void load_state_fe(out Fe value, uint base) {
  for (uint i = 0u; i < 20u; ++i) value.v[i] = multiplyState[base + i];
}
void store_state_fe(uint base, const Fe value) {
  for (uint i = 0u; i < 20u; ++i) multiplyState[base + i] = value.v[i];
}
void load_niels(out ProjectiveNiels value, uint base) {
  for (uint i = 0u; i < 20u; ++i) {
    value.y_plus_x.v[i] = nielsTables[base + i];
    value.y_minus_x.v[i] = nielsTables[base + 20u + i];
    value.z.v[i] = nielsTables[base + 40u + i];
    value.t2d.v[i] = nielsTables[base + 60u + i];
  }
}
void select_niels(out ProjectiveNiels selected, uint tableBase, int digit) {
  int signMask = digit >> 31;
  uint absolute = uint((digit + signMask) ^ signMask);
  niels_identity(selected);
  for (uint value = 1u; value <= 8u; ++value) {
    ProjectiveNiels candidate;
    load_niels(candidate, tableBase + (value - 1u) * 80u);
    niels_cmov(selected, candidate, ct_equal_u32(absolute, value));
  }
  ProjectiveNiels negated;
  fe_copy(negated.y_plus_x, selected.y_minus_x);
  fe_copy(negated.y_minus_x, selected.y_plus_x);
  fe_copy(negated.z, selected.z);
  fe_neg(negated.t2d, selected.t2d);
  niels_cmov(selected, negated, uint(signMask) & 1u);
}

void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= parameters.count || derivationValid[id] == 0u) return;
  uint stateOffset = id * 80u;
  uint tableBase = id * 640u;
  PointP1P1 completed;
  load_state_fe(completed.x, stateOffset);
  load_state_fe(completed.y, stateOffset + 20u);
  load_state_fe(completed.z, stateOffset + 40u);
  load_state_fe(completed.t, stateOffset + 60u);

  PointP2 projective;
  p2_from_p1p1(projective, completed);
  p2_double(completed, projective);
  p2_from_p1p1(projective, completed);
  p2_double(completed, projective);
  p2_from_p1p1(projective, completed);
  p2_double(completed, projective);
  p2_from_p1p1(projective, completed);
  p2_double(completed, projective);

  Point accumulator;
  p3_from_p1p1(accumulator, completed);
  ProjectiveNiels selected;
  select_niels(selected, tableBase, int(scalarDigits[parameters.stepIndex]));
  p3_add_niels(completed, accumulator, selected);
  store_state_fe(stateOffset, completed.x);
  store_state_fe(stateOffset + 20u, completed.y);
  store_state_fe(stateOffset + 40u, completed.z);
  store_state_fe(stateOffset + 60u, completed.t);
}
"""

MULTIPLY_DOUBLE_MAIN = """
void load_state_fe(out Fe value, uint base) {
  for (uint i = 0u; i < 20u; ++i) value.v[i] = multiplyState[base + i];
}
void store_state_fe(uint base, const Fe value) {
  for (uint i = 0u; i < 20u; ++i) multiplyState[base + i] = value.v[i];
}

void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= parameters.count || derivationValid[id] == 0u) return;
  uint stateOffset = id * 80u;
  PointP1P1 completed;
  load_state_fe(completed.x, stateOffset);
  load_state_fe(completed.y, stateOffset + 20u);
  load_state_fe(completed.z, stateOffset + 40u);
  load_state_fe(completed.t, stateOffset + 60u);
  PointP2 projective;
  p2_from_p1p1(projective, completed);
  p2_double(completed, projective);
  p2_from_p1p1(projective, completed);
  p2_double(completed, projective);
  p2_from_p1p1(projective, completed);
  p2_double(completed, projective);
  p2_from_p1p1(projective, completed);
  p2_double(completed, projective);
  store_state_fe(stateOffset, completed.x);
  store_state_fe(stateOffset + 20u, completed.y);
  store_state_fe(stateOffset + 40u, completed.z);
  store_state_fe(stateOffset + 60u, completed.t);
}
"""


MULTIPLY_ADD_MAIN = """
void load_state_fe(out Fe value, uint base) {
  for (uint i = 0u; i < 20u; ++i) value.v[i] = multiplyState[base + i];
}
void store_state_fe(uint base, const Fe value) {
  for (uint i = 0u; i < 20u; ++i) multiplyState[base + i] = value.v[i];
}
void load_niels(out ProjectiveNiels value, uint base) {
  for (uint i = 0u; i < 20u; ++i) {
    value.y_plus_x.v[i] = nielsTables[base + i];
    value.y_minus_x.v[i] = nielsTables[base + 20u + i];
    value.z.v[i] = nielsTables[base + 40u + i];
    value.t2d.v[i] = nielsTables[base + 60u + i];
  }
}
void select_niels(out ProjectiveNiels selected, uint tableBase, int digit) {
  int signMask = digit >> 31;
  uint absolute = uint((digit + signMask) ^ signMask);
  niels_identity(selected);
  for (uint value = 1u; value <= 8u; ++value) {
    ProjectiveNiels candidate;
    load_niels(candidate, tableBase + (value - 1u) * 80u);
    niels_cmov(selected, candidate, ct_equal_u32(absolute, value));
  }
  ProjectiveNiels negated;
  fe_copy(negated.y_plus_x, selected.y_minus_x);
  fe_copy(negated.y_minus_x, selected.y_plus_x);
  fe_copy(negated.z, selected.z);
  fe_neg(negated.t2d, selected.t2d);
  niels_cmov(selected, negated, uint(signMask) & 1u);
}

void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= parameters.count || derivationValid[id] == 0u) return;
  uint stateOffset = id * 80u;
  uint tableBase = id * 640u;
  PointP1P1 completed;
  load_state_fe(completed.x, stateOffset);
  load_state_fe(completed.y, stateOffset + 20u);
  load_state_fe(completed.z, stateOffset + 40u);
  load_state_fe(completed.t, stateOffset + 60u);
  Point accumulator;
  p3_from_p1p1(accumulator, completed);
  ProjectiveNiels selected;
  select_niels(selected, tableBase, int(scalarDigits[parameters.stepIndex]));
  p3_add_niels(completed, accumulator, selected);
  store_state_fe(stateOffset, completed.x);
  store_state_fe(stateOffset + 20u, completed.y);
  store_state_fe(stateOffset + 40u, completed.z);
  store_state_fe(stateOffset + 60u, completed.t);
}
"""


MULTIPLY_FINISH_MAIN = """
void load_state_fe(out Fe value, uint base) {
  for (uint i = 0u; i < 20u; ++i) value.v[i] = multiplyState[base + i];
}
void store_state_fe(uint base, const Fe value) {
  for (uint i = 0u; i < 20u; ++i) multiplyState[base + i] = value.v[i];
}

void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= parameters.count || derivationValid[id] == 0u) return;
  uint stateOffset = id * 80u;
  PointP1P1 completed;
  load_state_fe(completed.x, stateOffset);
  load_state_fe(completed.y, stateOffset + 20u);
  load_state_fe(completed.z, stateOffset + 40u);
  load_state_fe(completed.t, stateOffset + 60u);
  Point resultPoint;
  p3_from_p1p1(resultPoint, completed);
  store_state_fe(stateOffset, resultPoint.x);
  store_state_fe(stateOffset + 20u, resultPoint.y);
  store_state_fe(stateOffset + 40u, resultPoint.z);
  store_state_fe(stateOffset + 60u, resultPoint.t);
}
"""


CHUNKED_COMPRESS_MAIN = """
void load_state_fe(out Fe value, uint base) {
  for (uint i = 0u; i < 20u; ++i) value.v[i] = multiplyState[base + i];
}

void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= parameters.count) return;
  uint byteOffset = id * 32u;
  if (derivationValid[id] == 0u) {
    for (uint i = 0u; i < 32u; ++i) derivedResults[byteOffset + i] = 0u;
    return;
  }
  uint stateOffset = id * 80u;
  Point point;
  load_state_fe(point.x, stateOffset);
  load_state_fe(point.y, stateOffset + 20u);
  load_state_fe(point.z, stateOffset + 40u);
  load_state_fe(point.t, stateOffset + 60u);
  uint encoded[32];
  point_to_compressed_m5(encoded, point);
  for (uint i = 0u; i < 32u; ++i)
    derivedResults[byteOffset + i] = encoded[i];
}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metal-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "scalar_digits",
            "decode_prepare",
            "decode_inverse",
            "decode_sqrt",
            "decode_finish",
            "multiply_init",
            "multiply_double",
            "multiply_add",
            "multiply_finish",
            "compress",
        ),
        required=True,
    )
    arguments = parser.parse_args()

    metal_source = pathlib.Path(arguments.metal_source)
    output = pathlib.Path(arguments.output)
    transformed = transform_helpers(metal_source.read_text(encoding="utf-8"))
    stage_main = {
        "scalar_digits": SCALAR_DIGITS_MAIN,
        "decode_prepare": DECODE_PREPARE_MAIN,
        "decode_inverse": DECODE_INVERSE_MAIN,
        "decode_sqrt": DECODE_SQRT_MAIN,
        "decode_finish": DECODE_FINISH_MAIN,
        "multiply_init": MULTIPLY_INIT_MAIN,
        "multiply_double": MULTIPLY_DOUBLE_MAIN,
        "multiply_add": MULTIPLY_ADD_MAIN,
        "multiply_finish": MULTIPLY_FINISH_MAIN,
        "compress": CHUNKED_COMPRESS_MAIN,
    }[arguments.stage]
    generated = GLSL_HEADER + transformed + stage_main
    # Keep literal loop limits opaque to the physical-device compiler. The
    # bias is always pushed as zero, but prevents the Mali optimizer from
    # treating hundreds of nested arithmetic iterations as one giant
    # compile-time-unrolled expression.
    def make_loop_limit_dynamic(match: re.Match[str]) -> str:
        loop = match.group(0)
        loop = re.sub(
            r"(<|<=)\s*([0-9]+)u?(?=\s*;)",
            r"\1 (\2u + parameters.loopBias)",
            loop,
        )
        loop = re.sub(
            r">=\s*([0-9]+)(?=\s*;)",
            r">= int(\1u + parameters.loopBias)",
            loop,
        )
        return loop

    generated = re.sub(
        r"(?m)^\s*for \([^\n]+",
        make_loop_limit_dynamic,
        generated,
    )
    # The Pixel 8 Pro Mali compiler otherwise attempts to fully unroll the
    # large, constant-bound arithmetic loops and exhausts its compiler memory
    # while creating the compute pipeline. Keep the portable loop structure
    # explicitly in SPIR-V instead.
    generated = re.sub(
        r"(?m)^(\s*)for \(",
        r"\1[[dont_unroll]] for (",
        generated,
    )
    output.write_text(generated, encoding="utf-8")


if __name__ == "__main__":
    main()
