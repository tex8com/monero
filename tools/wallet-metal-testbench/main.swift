import Foundation
import Metal

// M1 host harness. The vector file is emitted by the Rust testbench after it
// has checked the exact wallet adapter against Dalek. It uses a deterministic
// public scalar and points, never a wallet view key.
struct Parameters { var count: UInt32; var batchChunkSize: UInt32 }

struct Config {
    var vectorPath: String?
    var kernelPath: String?
    var kernelName = "derivation_m1_reference"
    var rounds = 1
    var warmupRounds = 1
    var threadsPerGroup = 32
    var projectiveThreadsPerGroup: Int?
    var inverseThreadsPerGroup = 32
    var compressThreadsPerGroup: Int?
    var batchInversionChunkSize = 16
    // M6 scheduling experiment: this many independent output buffers may be
    // committed before waiting. Inputs are immutable and each slot is checked
    // byte-for-byte, so no command shares writable memory with another slot.
    var inflightCommands = 1
}

struct VectorCorpus {
    let scalar: [UInt8]
    let points: [UInt8]
    let expected: [UInt8]
    let invalid: [UInt8]
    let fingerprint: UInt64
    let count: Int
}

let vectorMagic: [UInt8] = [0x4d, 0x57, 0x4d, 0x54, 0x56, 0x31, 0x00, 0x00]
let vectorHeaderBytes = 88

func fail(_ message: String, code: Int32 = 1) -> Never {
    FileHandle.standardError.write(Data("Metal M1 testbench error: \(message)\n".utf8))
    exit(code)
}

func positive(_ value: String?, _ option: String) -> Int {
    guard let value, let parsed = Int(value), parsed > 0 else {
        fail("\(option) requires a positive integer", code: 2)
    }
    return parsed
}

func parseArgs() -> Config {
    var config = Config()
    var args = CommandLine.arguments.dropFirst().makeIterator()
    while let argument = args.next() {
        switch argument {
        case "--vectors": config.vectorPath = args.next()
        case "--kernel-source": config.kernelPath = args.next()
        case "--kernel":
            guard let name = args.next() else { fail("--kernel requires a name", code: 2) }
            config.kernelName = name
        case "--rounds": config.rounds = positive(args.next(), argument)
        case "--warmup-rounds": config.warmupRounds = positive(args.next(), argument)
        case "--threads-per-group": config.threadsPerGroup = positive(args.next(), argument)
        case "--projective-threads-per-group":
            config.projectiveThreadsPerGroup = positive(args.next(), argument)
        case "--inverse-threads-per-group":
            config.inverseThreadsPerGroup = positive(args.next(), argument)
        case "--compress-threads-per-group":
            config.compressThreadsPerGroup = positive(args.next(), argument)
        case "--batch-inversion-chunk-size": config.batchInversionChunkSize = positive(args.next(), argument)
        case "--inflight-commands": config.inflightCommands = positive(args.next(), argument)
        case "--help", "-h":
            print("Usage: wallet-metal-derivation-testbench --vectors PATH --kernel-source PATH [--kernel NAME] [--rounds N] [--warmup-rounds N] [--threads-per-group N] [--projective-threads-per-group N] [--inverse-threads-per-group N] [--compress-threads-per-group N] [--inflight-commands N] [--batch-inversion-chunk-size N]")
            exit(0)
        default: fail("unknown argument \(argument)", code: 2)
        }
    }
    guard config.vectorPath != nil else { fail("--vectors is required", code: 2) }
    guard config.kernelPath != nil else { fail("--kernel-source is required", code: 2) }
    return config
}

func le32(_ bytes: [UInt8], _ offset: Int) -> UInt32 {
    UInt32(bytes[offset])
        | (UInt32(bytes[offset + 1]) << 8)
        | (UInt32(bytes[offset + 2]) << 16)
        | (UInt32(bytes[offset + 3]) << 24)
}

func le64(_ bytes: [UInt8], _ offset: Int) -> UInt64 {
    var value: UInt64 = 0
    for index in 0 ..< 8 { value |= UInt64(bytes[offset + index]) << UInt64(index * 8) }
    return value
}

func readVectorCorpus(_ path: String) -> VectorCorpus {
    let raw: [UInt8]
    do { raw = Array(try Data(contentsOf: URL(fileURLWithPath: path))) }
    catch { fail("cannot read vector file \(path): \(error.localizedDescription)") }
    guard raw.count >= vectorHeaderBytes else { fail("vector file is shorter than the M1 header") }
    guard Array(raw[0 ..< 8]) == vectorMagic else { fail("vector file magic is not MWMTV1") }
    guard le32(raw, 8) == 1 else { fail("unsupported vector format version \(le32(raw, 8))") }
    let count = Int(le32(raw, 12))
    guard count > 0 else { fail("vector file has zero records") }
    let recordBytes = count.multipliedReportingOverflow(by: 64)
    guard !recordBytes.overflow, vectorHeaderBytes + recordBytes.partialValue == raw.count else {
        fail("vector file size does not match its record count")
    }
    var points = [UInt8](); points.reserveCapacity(count * 32)
    var expected = [UInt8](); expected.reserveCapacity(count * 32)
    for record in 0 ..< count {
        let offset = vectorHeaderBytes + record * 64
        points += raw[offset ..< offset + 32]
        expected += raw[offset + 32 ..< offset + 64]
    }
    return VectorCorpus(
        scalar: Array(raw[16 ..< 48]),
        points: points,
        expected: expected,
        invalid: Array(raw[56 ..< 88]),
        fingerprint: le64(raw, 48),
        count: count
    )
}

func sharedBuffer(_ device: MTLDevice, _ bytes: [UInt8], _ label: String) -> MTLBuffer {
    bytes.withUnsafeBytes { raw in
        guard let address = raw.baseAddress,
              let buffer = device.makeBuffer(bytes: address, length: raw.count, options: .storageModeShared)
        else { fail("failed to allocate \(label) buffer") }
        buffer.label = label
        return buffer
    }
}

func checkResults(_ resultBuffer: MTLBuffer, _ validBuffer: MTLBuffer, corpus: VectorCorpus) {
    let results = resultBuffer.contents().assumingMemoryBound(to: UInt8.self)
    let valid = validBuffer.contents().assumingMemoryBound(to: UInt8.self)
    for index in 0 ..< corpus.count {
        guard valid[index] == 1 else { fail("valid point \(index) was rejected") }
        let base = index * 32
        for byte in 0 ..< 32 where results[base + byte] != corpus.expected[base + byte] {
            fail("Dalek mismatch: record \(index), byte \(byte)")
        }
    }
    let invalidIndex = corpus.count
    guard valid[invalidIndex] == 0 else { fail("invalid compressed point was accepted") }
    let invalidBase = invalidIndex * 32
    for byte in 0 ..< 32 where results[invalidBase + byte] != 0 {
        fail("invalid compressed point left nonzero output byte \(byte)")
    }
}

let config = parseArgs()
let corpus = readVectorCorpus(config.vectorPath!)
guard corpus.count <= Int(UInt32.max) - 1 else { fail("vector count exceeds UInt32") }
guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    fail("no usable Metal device")
}
let kernelSource: String
do { kernelSource = try String(contentsOfFile: config.kernelPath!, encoding: .utf8) }
catch { fail("cannot read Metal kernel source: \(error.localizedDescription)") }
let library: MTLLibrary
do { library = try device.makeLibrary(source: kernelSource, options: nil) }
catch { fail("kernel compilation failed: \(error.localizedDescription)") }
func makePipeline(_ name: String) -> MTLComputePipelineState {
    guard let function = library.makeFunction(name: name) else { fail("Metal kernel \(name) missing") }
    do { return try device.makeComputePipelineState(function: function) }
    catch { fail("pipeline creation failed for \(name): \(error.localizedDescription)") }
}
let usesChunkInversion = config.kernelName == "derivation_m12_projective_chunkinvert"
let usesBatchInversion = config.kernelName == "derivation_m11_projective_batchinvert" || usesChunkInversion
let pipeline = makePipeline(config.kernelName)
let inversePipeline = usesBatchInversion
    ? makePipeline(usesChunkInversion ? "derivation_m12_chunk_inverse" : "derivation_m11_batch_inverse")
    : nil
let compressPipeline = usesBatchInversion
    ? makePipeline(usesChunkInversion ? "derivation_m12_compress_chunkinvert" : "derivation_m11_compress_batchinvert")
    : nil
let projectiveThreadCount = config.projectiveThreadsPerGroup ?? config.threadsPerGroup
let compressThreadCount = config.compressThreadsPerGroup ?? config.threadsPerGroup
guard projectiveThreadCount <= pipeline.maxTotalThreadsPerThreadgroup else {
    fail("requested \(projectiveThreadCount) projective threads/group exceeds kernel limit \(pipeline.maxTotalThreadsPerThreadgroup)")
}
if let inversePipeline, config.inverseThreadsPerGroup > inversePipeline.maxTotalThreadsPerThreadgroup {
    fail("requested \(config.inverseThreadsPerGroup) inverse threads/group exceeds inversion-kernel limit \(inversePipeline.maxTotalThreadsPerThreadgroup)")
}
if let compressPipeline, compressThreadCount > compressPipeline.maxTotalThreadsPerThreadgroup {
    fail("requested \(compressThreadCount) compression threads/group exceeds compression-kernel limit \(compressPipeline.maxTotalThreadsPerThreadgroup)")
}

var dispatchPoints = corpus.points
dispatchPoints += corpus.invalid
let scalarBuffer = sharedBuffer(device, corpus.scalar, "view_scalar_public_test_only")
let pointBuffer = sharedBuffer(device, dispatchPoints, "compressed_points_plus_invalid")
var resultBuffers = [MTLBuffer]()
var validBuffers = [MTLBuffer]()
var projectiveBuffers = [MTLBuffer]()
var inverseBuffers = [MTLBuffer]()
resultBuffers.reserveCapacity(config.inflightCommands)
validBuffers.reserveCapacity(config.inflightCommands)
projectiveBuffers.reserveCapacity(config.inflightCommands)
inverseBuffers.reserveCapacity(config.inflightCommands)
for slot in 0 ..< config.inflightCommands {
    guard let resultBuffer = device.makeBuffer(length: dispatchPoints.count, options: .storageModeShared),
          let validBuffer = device.makeBuffer(length: corpus.count + 1, options: .storageModeShared)
    else { fail("failed to allocate output buffers for in-flight slot \(slot)") }
    resultBuffer.label = "derivation_results_slot_\(slot)"
    validBuffer.label = "derivation_validity_slot_\(slot)"
    resultBuffers.append(resultBuffer)
    validBuffers.append(validBuffer)
    if usesBatchInversion {
        let recordCount = corpus.count + 1
        guard let projectiveBuffer = device.makeBuffer(
                  length: recordCount * 30 * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let inverseBuffer = device.makeBuffer(
                  length: recordCount * 10 * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared)
        else { fail("failed to allocate M11 intermediate buffers for in-flight slot \(slot)") }
        projectiveBuffer.label = "derivation_projective_xyz_slot_\(slot)"
        inverseBuffer.label = "derivation_inverse_z_slot_\(slot)"
        projectiveBuffers.append(projectiveBuffer)
        inverseBuffers.append(inverseBuffer)
    }
}
guard config.batchInversionChunkSize <= Int(UInt32.max) else {
    fail("--batch-inversion-chunk-size exceeds UInt32")
}
var parameters = Parameters(
    count: UInt32(corpus.count + 1),
    batchChunkSize: UInt32(config.batchInversionChunkSize))
guard let parameterBuffer = device.makeBuffer(bytes: &parameters, length: MemoryLayout<Parameters>.stride, options: .storageModeShared)
else { fail("failed to allocate parameter buffer") }
let projectiveGroup = MTLSize(width: projectiveThreadCount, height: 1, depth: 1)
let inverseGroup = MTLSize(width: config.inverseThreadsPerGroup, height: 1, depth: 1)
let compressGroup = MTLSize(width: compressThreadCount, height: 1, depth: 1)
let grid = MTLSize(width: corpus.count + 1, height: 1, depth: 1)

func dispatchBatch(_ count: Int) -> (UInt64, Double) {
    let started = DispatchTime.now().uptimeNanoseconds
    var commands = [MTLCommandBuffer]()
    commands.reserveCapacity(count)
    for slot in 0 ..< count {
        guard let command = queue.makeCommandBuffer() else {
            fail("failed to allocate command buffer")
        }
        if usesBatchInversion {
            guard let projectiveEncoder = command.makeComputeCommandEncoder(),
                  let inversePipeline,
                  let compressPipeline
            else { fail("failed to allocate M11 projective encoder or pipeline") }
            projectiveEncoder.setComputePipelineState(pipeline)
            projectiveEncoder.setBuffer(scalarBuffer, offset: 0, index: 0)
            projectiveEncoder.setBuffer(pointBuffer, offset: 0, index: 1)
            projectiveEncoder.setBuffer(projectiveBuffers[slot], offset: 0, index: 2)
            projectiveEncoder.setBuffer(validBuffers[slot], offset: 0, index: 3)
            projectiveEncoder.setBuffer(parameterBuffer, offset: 0, index: 4)
            projectiveEncoder.dispatchThreads(grid, threadsPerThreadgroup: projectiveGroup)
            projectiveEncoder.endEncoding()

            guard let inverseEncoder = command.makeComputeCommandEncoder() else {
                fail("failed to allocate M11 inversion encoder")
            }
            inverseEncoder.setComputePipelineState(inversePipeline)
            inverseEncoder.setBuffer(projectiveBuffers[slot], offset: 0, index: 0)
            inverseEncoder.setBuffer(inverseBuffers[slot], offset: 0, index: 1)
            inverseEncoder.setBuffer(validBuffers[slot], offset: 0, index: 2)
            inverseEncoder.setBuffer(parameterBuffer, offset: 0, index: 3)
            let inverseThreads = usesChunkInversion
                ? (corpus.count + 1 + config.batchInversionChunkSize - 1) / config.batchInversionChunkSize
                : 1
            inverseEncoder.dispatchThreads(
                MTLSize(width: inverseThreads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(
                    width: min(inverseGroup.width, inverseThreads),
                    height: 1,
                    depth: 1))
            inverseEncoder.endEncoding()

            guard let compressEncoder = command.makeComputeCommandEncoder() else {
                fail("failed to allocate M11 compression encoder")
            }
            compressEncoder.setComputePipelineState(compressPipeline)
            compressEncoder.setBuffer(projectiveBuffers[slot], offset: 0, index: 0)
            compressEncoder.setBuffer(inverseBuffers[slot], offset: 0, index: 1)
            compressEncoder.setBuffer(resultBuffers[slot], offset: 0, index: 2)
            compressEncoder.setBuffer(validBuffers[slot], offset: 0, index: 3)
            compressEncoder.setBuffer(parameterBuffer, offset: 0, index: 4)
            compressEncoder.dispatchThreads(grid, threadsPerThreadgroup: compressGroup)
            compressEncoder.endEncoding()
        } else {
            guard let encoder = command.makeComputeCommandEncoder() else {
                fail("failed to allocate command encoder")
            }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(scalarBuffer, offset: 0, index: 0)
            encoder.setBuffer(pointBuffer, offset: 0, index: 1)
            encoder.setBuffer(resultBuffers[slot], offset: 0, index: 2)
            encoder.setBuffer(validBuffers[slot], offset: 0, index: 3)
            encoder.setBuffer(parameterBuffer, offset: 0, index: 4)
            encoder.dispatchThreads(grid, threadsPerThreadgroup: projectiveGroup)
            encoder.endEncoding()
        }
        command.commit()
        commands.append(command)
    }
    for command in commands {
        command.waitUntilCompleted()
        guard command.error == nil else { fail("GPU command failed") }
    }
    let hostNs = DispatchTime.now().uptimeNanoseconds - started
    let gpuSeconds = commands.reduce(0.0) { total, command in
        total + max(0, command.gpuEndTime - command.gpuStartTime)
    }
    return (hostNs, gpuSeconds)
}

func runRounds(_ rounds: Int, collectTiming: Bool) -> (UInt64, Double) {
    var remaining = rounds
    var hostNs: UInt64 = 0
    var gpuSeconds = 0.0
    while remaining > 0 {
        let batchCount = min(config.inflightCommands, remaining)
        let timing = dispatchBatch(batchCount)
        for slot in 0 ..< batchCount {
            checkResults(resultBuffers[slot], validBuffers[slot], corpus: corpus)
        }
        if collectTiming {
            hostNs &+= timing.0
            // With more than one command in flight this is intentionally a
            // sum of GPU intervals, not a wall-clock duration.
            gpuSeconds += timing.1
        }
        remaining -= batchCount
    }
    return (hostNs, gpuSeconds)
}

_ = runRounds(config.warmupRounds, collectTiming: false)
let timing = runRounds(config.rounds, collectTiming: true)
let hostNs = timing.0
let gpuSeconds = timing.1

let operations = corpus.count * config.rounds
let hostSeconds = Double(hostNs) / 1_000_000_000
let logicalPayload = (32 + corpus.count * 32 + corpus.count * 32 + corpus.count) * config.rounds
let stage: String
switch config.kernelName {
case "derivation_m1_reference": stage = "M1_reference_correctness_first"
case "derivation_m2_group_scalar": stage = "M2_group_scalar_reuse"
case "derivation_m4_radix16_niels": stage = "M4_radix16_projective_niels"
case "derivation_m5_radix16_niels_addition_chain": stage = "M5_radix16_niels_addition_chain"
case "derivation_m7_radix2625_niels_addition_chain": stage = "M7_radix2625_niels_addition_chain"
case "derivation_m8_radix2625_lazyadd_niels": stage = "M8_radix2625_lazyadd_niels"
case "derivation_m9_radix2625_freeze_niels": stage = "M9_radix2625_freeze_niels"
case "derivation_m10_radix2625_groupdigits_niels": stage = "M10_radix2625_groupdigits_niels"
case "derivation_m11_projective_batchinvert": stage = "M11_radix2625_batch_inversion"
case "derivation_m12_projective_chunkinvert": stage = "M12_radix2625_chunk_inversion"
case "derivation_m13_radix13_u32_niels": stage = "M13_radix13_u32_niels"
default: stage = "custom_kernel"
}
print("testbench=wallet_metal_derivation_testbench_m1")
print("stage=\(stage)")
print("execution_schedule=\(config.inflightCommands == 1 ? "sequential_command_queue" : "M6_multi_inflight_command_queue")")
print("algorithm=monero_generate_key_derivation_8_times_a_times_r")
print("algorithm_status=implemented_and_byte_checked_against_dalek")
print("metal_device=\(device.name)")
print("kernel=\(config.kernelName)")
print("field_representation=\(config.kernelName.hasPrefix("derivation_m13_") ? "radix_2_to_13_pure_u32" : (config.kernelName.hasPrefix("derivation_m7_") || config.kernelName.hasPrefix("derivation_m8_") || config.kernelName.hasPrefix("derivation_m9_") || config.kernelName.hasPrefix("derivation_m10_") || config.kernelName.hasPrefix("derivation_m11_") || config.kernelName.hasPrefix("derivation_m12_") ? "radix_2_to_25_5_dalek_u32" : "radix_2_to_16_reference"))")
print("affine_conversion=\(usesChunkInversion ? "three_pass_chunked_montgomery_batch_inversion" : (usesBatchInversion ? "three_pass_montgomery_batch_inversion" : "one_inversion_per_record"))")
if usesChunkInversion { print("batch_inversion_chunk_size=\(config.batchInversionChunkSize)") }
print("point_representation=extended_edwards25519_complete_formulas")
print("scalar_contract=scalar_from_bytes_mod_order_then_8_times_scalar_mod_l")
print("input_contract=deterministic_public_test_scalar_and_valid_compressed_edwards_points")
print("corpus_fingerprint_fnv1a64=0x\(String(corpus.fingerprint, radix: 16))")
print("points_per_round=\(corpus.count)")
print("invalid_point_contract=one_Dalek_rejected_point_must_return_valid_0_and_zero_output")
print("dispatch_records_per_round=\(corpus.count + 1)")
print("timed_rounds=\(config.rounds)")
print("warmup_rounds=\(config.warmupRounds)")
print("threads_per_threadgroup=\(projectiveGroup.width)")
print("projective_threads_per_threadgroup=\(projectiveGroup.width)")
if usesBatchInversion {
    print("inverse_threads_per_threadgroup=\(inverseGroup.width)")
    print("compress_threads_per_threadgroup=\(compressGroup.width)")
}
print("pipeline_thread_execution_width=\(pipeline.threadExecutionWidth)")
print("pipeline_max_total_threads_per_threadgroup=\(pipeline.maxTotalThreadsPerThreadgroup)")
print("inflight_commands=\(config.inflightCommands)")
print("operations=\(operations)")
print("host_submit_wait_ns=\(hostNs)")
print(String(format: "host_submit_wait_seconds=%.9f", hostSeconds))
print(String(format: "m1_derivations_per_second=%.3f", Double(operations) / hostSeconds))
print(String(format: "derivations_per_second=%.3f", Double(operations) / hostSeconds))
print("logical_payload_bytes=\(logicalPayload)")
print(String(format: "logical_payload_mib_per_second=%.6f", Double(logicalPayload) / hostSeconds / 1_048_576))
print(String(format: "gpu_execution_seconds_sum=%.9f", gpuSeconds))
print(String(format: "gpu_execution_seconds=%.9f", gpuSeconds))
print("validation=pass")
