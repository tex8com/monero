#!/usr/bin/env python3
"""Generate staged Vulkan GLSL from the byte-checked Metal M12 field kernel."""

from __future__ import annotations

import argparse
import pathlib
import re

import generate_radix13_glsl as radix13


def replace_uchar_casts(source: str) -> str:
    marker = "uchar("
    search_from = 0
    while True:
        start = source.find(marker, search_from)
        if start < 0:
            return source
        expression_start = start + len(marker)
        depth = 1
        end = expression_start
        while end < len(source) and depth != 0:
            if source[end] == "(":
                depth += 1
            elif source[end] == ")":
                depth -= 1
            end += 1
        if depth != 0:
            raise ValueError("unbalanced uchar cast")
        expression = source[expression_start : end - 1]
        replacement = f"uint(({expression}) & 255u)"
        source = source[:start] + replacement + source[end:]
        search_from = start + len(replacement)


def replace_function_body_tokens(
    source: str, start_name: str, end_name: str, replacements: dict[str, str]
) -> str:
    start = source.index(start_name)
    end = source.index(end_name, start)
    body = source[start:end]
    for old, new in replacements.items():
        body = body.replace(old, new)
    return source[:start] + body + source[end:]


def transform_helpers(source: str) -> str:
    source = source[source.index("struct Fe") :]
    source = radix13.remove_metal_kernels(source)
    source = re.sub(r"^struct Parameters[^\n]*\n", "", source, flags=re.MULTILINE)
    source = re.sub(
        r"inline void fe_store_device\(.*?\n}\n\n"
        r"inline void fe_load_device\(.*?\n}\n",
        "",
        source,
        flags=re.DOTALL,
    )

    source = re.sub(r"\bout\b", "resultValue", source)
    source = re.sub(r"\binput\b", "inputValue", source)
    source = source.replace("inline ", "")
    source = source.replace("thread const ", "const ")
    source = source.replace("constant const ", "const ")
    source = source.replace("device const ", "const ")
    source = source.replace("threadgroup const ", "const ")
    source = source.replace("threadgroup ", "")
    source = source.replace("thread ", "")
    source = source.replace("constant ", "const ")
    source = source.replace("device ", "")
    source = source.replace("constexpr ", "const ")

    source = source.replace(
        "const ulong mask = 0ul - ulong(choose_source & 1u);",
        "const uint mask = 0u - uint(choose_source & 1u);",
    )
    source = replace_uchar_casts(source)
    source = source.replace("uchar", "uint")
    source = re.sub(r"\bchar\b", "int", source)
    source = source.replace("ulong", "uint64_t")

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
    source = source.replace("uint64_t *z", "inout uint64_t z[10]")
    source = source.replace("const int *digits", "const int digits[64]")
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
    source = re.sub(
        r"uint64_t z\[10\] = \{(.*?)\};",
        lambda match: f"uint64_t z[10] = uint64_t[10]({match.group(1)});",
        source,
        flags=re.DOTALL,
    )

    source = source.replace("  const uint *x = left.v;\n", "")
    source = source.replace("  const uint *y = right.v;\n", "")
    source = replace_function_body_tokens(
        source,
        "void fe_mul",
        "void fe_square",
        {"x[": "left.v[", "y[": "right.v["},
    )
    source = source.replace("  const uint *x = inputValue.v;\n", "")
    source = replace_function_body_tokens(
        source,
        "void fe_square",
        "void fe_freeze",
        {"x[": "inputValue.v["},
    )
    return source


GLSL_HEADER = """#version 450
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

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

"""

BATCH_INVERSE_MAIN = """
void load_state_fe(out Fe value, uint base) {
  for (uint i = 0u; i < 10u; ++i) value.v[i] = multiplyState[base + i];
}
void load_inverse_fe(out Fe value, uint base) {
  for (uint i = 0u; i < 10u; ++i) value.v[i] = scratchPoints[base + i];
}
void store_inverse_fe(uint base, const Fe value) {
  for (uint i = 0u; i < 10u; ++i) scratchPoints[base + i] = value.v[i];
}

void main() {
  uint chunkSize = parameters.stepIndex;
  uint begin = gl_GlobalInvocationID.x * chunkSize;
  if (begin >= parameters.count) return;
  uint end = min(begin + chunkSize, parameters.count);

  Fe accumulator;
  fe_one(accumulator);
  for (uint record = begin; record < end; ++record) {
    store_inverse_fe(record * 10u, accumulator);
    Fe z;
    Fe next;
    if (derivationValid[record] != 0u)
      load_state_fe(z, record * 40u + 20u);
    else
      fe_one(z);
    fe_mul(next, accumulator, z);
    fe_copy(accumulator, next);
  }

  Fe inverseProduct;
  fe_inverse_m5(inverseProduct, accumulator);
  uint remaining = end;
  while (remaining != begin) {
    uint record = --remaining;
    Fe prefix;
    Fe z;
    Fe inverseRecord;
    Fe next;
    load_inverse_fe(prefix, record * 10u);
    if (derivationValid[record] != 0u)
      load_state_fe(z, record * 40u + 20u);
    else
      fe_one(z);
    fe_mul(inverseRecord, inverseProduct, prefix);
    if (derivationValid[record] != 0u)
      store_inverse_fe(record * 10u, inverseRecord);
    else {
      Fe zero;
      fe_zero(zero);
      store_inverse_fe(record * 10u, zero);
    }
    fe_mul(next, inverseProduct, z);
    fe_copy(inverseProduct, next);
  }
}
"""


BATCH_COMPRESS_MAIN = """
void load_state_fe(out Fe value, uint base) {
  for (uint i = 0u; i < 10u; ++i) value.v[i] = multiplyState[base + i];
}
void load_inverse_fe(out Fe value, uint base) {
  for (uint i = 0u; i < 10u; ++i) value.v[i] = scratchPoints[base + i];
}

void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= parameters.count) return;
  uint byteOffset = id * 32u;
  if (derivationValid[id] == 0u) {
    for (uint i = 0u; i < 32u; ++i) derivedResults[byteOffset + i] = 0u;
    return;
  }
  uint stateOffset = id * 40u;
  Fe projectiveX;
  Fe projectiveY;
  Fe inverse;
  Fe x;
  Fe y;
  load_state_fe(projectiveX, stateOffset);
  load_state_fe(projectiveY, stateOffset + 10u);
  load_inverse_fe(inverse, id * 10u);
  fe_mul(x, projectiveX, inverse);
  fe_mul(y, projectiveY, inverse);
  uint encoded[32];
  fe_to_bytes(encoded, y);
  encoded[31] ^= uint(fe_is_negative(x) ? 128u : 0u);
  for (uint i = 0u; i < 32u; ++i)
    derivedResults[byteOffset + i] = encoded[i];
}
"""

MULTIPLY_NOOP_MAIN = """
void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= parameters.count) return;
}
"""

MULTIPLY_CHUNK4_SHARED_MAIN = """
shared uint sharedNielsTables[5120];

void load_state_fe(out Fe value, uint base) {
  for (uint i = 0u; i < 10u; ++i) value.v[i] = multiplyState[base + i];
}
void store_state_fe(uint base, const Fe value) {
  for (uint i = 0u; i < 10u; ++i) multiplyState[base + i] = value.v[i];
}
void load_shared_niels(out ProjectiveNiels value, uint base) {
  for (uint i = 0u; i < 10u; ++i) {
    value.y_plus_x.v[i] = sharedNielsTables[base + i];
    value.y_minus_x.v[i] = sharedNielsTables[base + 10u + i];
    value.z.v[i] = sharedNielsTables[base + 20u + i];
    value.t2d.v[i] = sharedNielsTables[base + 30u + i];
  }
}
void select_shared_niels(
    out ProjectiveNiels selected, uint laneTableBase, int digit) {
  int signMask = digit >> 31;
  uint absolute = uint((digit + signMask) ^ signMask);
  niels_identity(selected);
  for (uint value = 1u; value <= 8u; ++value) {
    ProjectiveNiels candidate;
    load_shared_niels(candidate, laneTableBase + (value - 1u) * 40u);
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
  uint lane = gl_LocalInvocationID.x;
  uint laneTableBase = lane * 320u;
  if (id < parameters.count) {
    uint deviceTableBase = id * 320u;
    for (uint i = 0u; i < 320u; ++i)
      sharedNielsTables[laneTableBase + i] =
          nielsTables[deviceTableBase + i];
  }
  memoryBarrierShared();
  barrier();
  if (id >= parameters.count || derivationValid[id] == 0u) return;

  uint stateOffset = id * 40u;
  PointP1P1 completed;
  load_state_fe(completed.x, stateOffset);
  load_state_fe(completed.y, stateOffset + 10u);
  load_state_fe(completed.z, stateOffset + 20u);
  load_state_fe(completed.t, stateOffset + 30u);

  uint digitCount = min(4u, parameters.stepIndex + 1u);
  for (uint processed = 0u; processed < digitCount; ++processed) {
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
    uint digitIndex = parameters.stepIndex - processed;
    select_shared_niels(
        selected, laneTableBase, int(scalarDigits[digitIndex]));
    p3_add_niels(completed, accumulator, selected);
  }
  store_state_fe(stateOffset, completed.x);
  store_state_fe(stateOffset + 10u, completed.y);
  store_state_fe(stateOffset + 20u, completed.z);
  store_state_fe(stateOffset + 30u, completed.t);
}
"""


def adapt_stage(stage: str) -> str:
    replacements = {
        "640": "320",
        "200": "100",
        "180": "90",
        "160": "80",
        "140": "70",
        "120": "60",
        "100": "50",
        "80": "40",
        "60": "30",
        "40": "20",
        "20": "10",
    }
    return re.sub(
        r"\b(640|200|180|160|140|120|100|80|60|40|20)u\b",
        lambda match: replacements[match.group(1)] + "u",
        stage,
    )

def multiply_double_stage(repetitions: int) -> str:
    source = radix13.MULTIPLY_DOUBLE_MAIN
    start = source.index("  PointP2 projective;")
    end = source.index("  store_state_fe(stateOffset, completed.x);", start)
    operations = "  PointP2 projective;\n" + "".join(
        "  p2_from_p1p1(projective, completed);\n"
        "  p2_double(completed, projective);\n"
        for _ in range(repetitions)
    )
    return source[:start] + operations + source[end:]


COMPRESS_MAIN = """
void load_state_fe(out Fe value, uint base) {
  for (uint i = 0u; i < 20u; ++i) value.v[i] = multiplyState[base + i];
}

void main() {
  uint id = gl_GlobalInvocationID.x;
  if (id >= parameters.count) return;
  uint byteOffset = id * 32u;
  if (derivationValid[id] == 0u) {
    for (uint i = 0u; i < 32u; ++i)
      derivedResults[byteOffset + i] = 0u;
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


STAGES = {
    "scalar_digits": radix13.SCALAR_DIGITS_MAIN,
    "decode_prepare": radix13.DECODE_PREPARE_MAIN,
    "decode_inverse": radix13.DECODE_INVERSE_MAIN,
    "decode_sqrt": radix13.DECODE_SQRT_MAIN,
    "decode_finish": radix13.DECODE_FINISH_MAIN,
    "multiply_init": radix13.MULTIPLY_INIT_MAIN,
    "multiply_double": radix13.MULTIPLY_DOUBLE_MAIN,
    "multiply_add": radix13.MULTIPLY_ADD_MAIN,
    "multiply_digit": radix13.MULTIPLY_STEP_MAIN,
    "multiply_noop": MULTIPLY_NOOP_MAIN,
    "multiply_chunk4_shared": MULTIPLY_CHUNK4_SHARED_MAIN,
    "multiply_finish": radix13.MULTIPLY_FINISH_MAIN,
    "compress": COMPRESS_MAIN,
    "batch_inverse": BATCH_INVERSE_MAIN,
    "compress_batch": BATCH_COMPRESS_MAIN,
}


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metal-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument(
        "--workgroup-size",
        type=int,
        choices=(16, 32, 64, 128, 256),
        default=64,
    )
    parser.add_argument(
        "--doublings-per-dispatch",
        type=int,
        choices=(1, 2, 4),
        default=4,
    )
    parser.add_argument(
        "--loop-policy",
        choices=("dynamic", "selective-unroll"),
        default="dynamic",
        help=(
            "Keep every loop runtime-bounded for conservative driver "
            "compatibility, or expose only the measured fixed-size multiply "
            "loops to the driver unroller."
        ),
    )
    arguments = parser.parse_args()

    helpers = transform_helpers(
        pathlib.Path(arguments.metal_source).read_text(encoding="utf-8")
    )
    header = GLSL_HEADER.replace(
        "local_size_x = 64",
        f"local_size_x = {arguments.workgroup_size}",
    )
    stage_source = STAGES[arguments.stage]
    if arguments.stage == "multiply_double":
        stage_source = multiply_double_stage(arguments.doublings_per_dispatch)
    if arguments.stage not in (
        "batch_inverse",
        "compress_batch",
        "multiply_chunk4_shared",
        "multiply_noop",
    ):
        stage_source = adapt_stage(stage_source)
    generated = header + helpers + stage_source
    selectively_unrolled_stages = ("multiply_double", "multiply_add")
    if (
        arguments.loop_policy == "selective-unroll"
        and arguments.stage in selectively_unrolled_stages
    ):
        generated = re.sub(
            r"(?m)^(\s*)for \(",
            r"\1[[unroll]] for (",
            generated,
        )
    else:
        generated = re.sub(
            r"(?m)^\s*for \([^\n]+", make_loop_limit_dynamic, generated
        )
        generated = re.sub(
            r"(?m)^(\s*)for \(", r"\1[[dont_unroll]] for (", generated
        )
    pathlib.Path(arguments.output).write_text(generated, encoding="utf-8")


if __name__ == "__main__":
    main()
