import Dispatch
import Foundation
import Metal

// Isolated mobile testbench for deterministic public MWMTV1 vectors.
// This is deliberately not a product wallet-key API.

public enum WalletMetalMobileTestbenchError: Error, CustomStringConvertible {
    case invalidConfiguration(String)
    case invalidCorpus(String)
    case metal(String)
    case validation(String)
    case executionNotAllowed

    public var description: String {
        switch self {
        case .invalidConfiguration(let message): return "invalid configuration: \(message)"
        case .invalidCorpus(let message): return "invalid MWMTV1 corpus: \(message)"
        case .metal(let message): return "Metal error: \(message)"
        case .validation(let message): return "validation error: \(message)"
        case .executionNotAllowed:
            return "Metal execution is not allowed while the app is inactive or backgrounded"
        }
    }
}

public struct WalletMetalMobileConfiguration: Sendable {
    public var rounds: Int
    public var warmupRounds: Int
    public var projectiveThreadsPerGroup: Int
    public var inverseThreadsPerGroup: Int
    public var compressThreadsPerGroup: Int
    public var batchInversionChunkSize: Int

    public init(
        rounds: Int = 60,
        warmupRounds: Int = 3,
        projectiveThreadsPerGroup: Int = 256,
        inverseThreadsPerGroup: Int = 16,
        compressThreadsPerGroup: Int = 128,
        batchInversionChunkSize: Int = 16
    ) {
        self.rounds = rounds
        self.warmupRounds = warmupRounds
        self.projectiveThreadsPerGroup = projectiveThreadsPerGroup
        self.inverseThreadsPerGroup = inverseThreadsPerGroup
        self.compressThreadsPerGroup = compressThreadsPerGroup
        self.batchInversionChunkSize = batchInversionChunkSize
    }
}

public struct WalletMetalMobileReport: Sendable {
    public let deviceName: String
    public let pointsPerRound: Int
    public let timedRounds: Int
    public let operations: Int
    public let hostSeconds: Double
    public let gpuSeconds: Double
    public let derivationsPerSecond: Double
    public let projectiveThreadsPerGroup: Int
    public let inverseThreadsPerGroup: Int
    public let compressThreadsPerGroup: Int
    public let projectiveExecutionWidth: Int
    public let corpusFingerprint: UInt64
    public let validationPassed: Bool
}

private struct MetalParameters {
    var count: UInt32
    var batchChunkSize: UInt32
}

private struct MobileVectorCorpus {
    let scalar: [UInt8]
    let points: [UInt8]
    let expected: [UInt8]
    let invalid: [UInt8]
    let fingerprint: UInt64
    let count: Int
}

public enum WalletMetalMobileTestbench {
    private static let vectorMagic: [UInt8] =
        [0x4d, 0x57, 0x4d, 0x54, 0x56, 0x31, 0x00, 0x00]
    private static let vectorHeaderBytes = 88

    public static func run(
        vectorData: Data,
        kernelSource: String,
        configuration: WalletMetalMobileConfiguration = .init(),
        isExecutionAllowed: () -> Bool = { true }
    ) throws -> WalletMetalMobileReport {
        try validate(configuration)
        let corpus = try parseCorpus(vectorData)
        guard corpus.count <= Int(UInt32.max) - 1 else {
            throw WalletMetalMobileTestbenchError.invalidCorpus(
                "record count exceeds UInt32"
            )
        }
        guard isExecutionAllowed() else {
            throw WalletMetalMobileTestbenchError.executionNotAllowed
        }
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue()
        else {
            throw WalletMetalMobileTestbenchError.metal(
                "no usable default Metal device or command queue"
            )
        }

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: kernelSource, options: nil)
        } catch {
            throw WalletMetalMobileTestbenchError.metal(
                "M12 kernel compilation failed: \(error.localizedDescription)"
            )
        }

        let projectivePipeline = try makePipeline(
            library: library,
            device: device,
            name: "derivation_m12_projective_chunkinvert"
        )
        let inversePipeline = try makePipeline(
            library: library,
            device: device,
            name: "derivation_m12_chunk_inverse"
        )
        let compressPipeline = try makePipeline(
            library: library,
            device: device,
            name: "derivation_m12_compress_chunkinvert"
        )

        let projectiveThreads = alignedThreadCount(
            preferred: configuration.projectiveThreadsPerGroup,
            pipeline: projectivePipeline
        )
        let inverseThreads = min(
            configuration.inverseThreadsPerGroup,
            inversePipeline.maxTotalThreadsPerThreadgroup
        )
        let compressThreads = alignedThreadCount(
            preferred: configuration.compressThreadsPerGroup,
            pipeline: compressPipeline
        )

        var dispatchPoints = corpus.points
        dispatchPoints += corpus.invalid
        let scalarBuffer = try sharedBuffer(
            device: device,
            bytes: corpus.scalar,
            label: "public_test_scalar"
        )
        let pointBuffer = try sharedBuffer(
            device: device,
            bytes: dispatchPoints,
            label: "public_test_points_plus_invalid"
        )
        let recordCount = corpus.count + 1
        guard let projectiveBuffer = device.makeBuffer(
                  length: recordCount * 30 * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared
              ),
              let inverseBuffer = device.makeBuffer(
                  length: recordCount * 10 * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared
              ),
              let resultBuffer = device.makeBuffer(
                  length: recordCount * 32,
                  options: .storageModeShared
              ),
              let validBuffer = device.makeBuffer(
                  length: recordCount,
                  options: .storageModeShared
              )
        else {
            throw WalletMetalMobileTestbenchError.metal(
                "failed to allocate M12 shared buffers"
            )
        }
        projectiveBuffer.label = "public_test_projective_xyz"
        inverseBuffer.label = "public_test_inverse_z"
        resultBuffer.label = "public_test_derivation_results"
        validBuffer.label = "public_test_validity"

        var parameters = MetalParameters(
            count: UInt32(recordCount),
            batchChunkSize: UInt32(configuration.batchInversionChunkSize)
        )
        guard let parameterBuffer = device.makeBuffer(
                  bytes: &parameters,
                  length: MemoryLayout<MetalParameters>.stride,
                  options: .storageModeShared
              )
        else {
            throw WalletMetalMobileTestbenchError.metal(
                "failed to allocate parameter buffer"
            )
        }
        parameterBuffer.label = "public_test_parameters"

        let sensitiveBuffers = [
            scalarBuffer,
            pointBuffer,
            projectiveBuffer,
            inverseBuffer,
            resultBuffer,
            validBuffer,
            parameterBuffer,
        ]
        defer {
            for buffer in sensitiveBuffers {
                buffer.contents().initializeMemory(
                    as: UInt8.self,
                    repeating: 0,
                    count: buffer.length
                )
            }
        }

        let projectiveGroup = MTLSize(
            width: projectiveThreads,
            height: 1,
            depth: 1
        )
        let inverseGroup = MTLSize(
            width: inverseThreads,
            height: 1,
            depth: 1
        )
        let compressGroup = MTLSize(
            width: compressThreads,
            height: 1,
            depth: 1
        )
        let grid = MTLSize(width: recordCount, height: 1, depth: 1)

        func executeRound(collectTiming: Bool) throws -> (UInt64, Double) {
            guard isExecutionAllowed() else {
                throw WalletMetalMobileTestbenchError.executionNotAllowed
            }
            guard let command = queue.makeCommandBuffer() else {
                throw WalletMetalMobileTestbenchError.metal(
                    "failed to allocate command buffer"
                )
            }

            guard let projectiveEncoder = command.makeComputeCommandEncoder()
            else {
                throw WalletMetalMobileTestbenchError.metal(
                    "failed to allocate projective encoder"
                )
            }
            projectiveEncoder.setComputePipelineState(projectivePipeline)
            projectiveEncoder.setBuffer(scalarBuffer, offset: 0, index: 0)
            projectiveEncoder.setBuffer(pointBuffer, offset: 0, index: 1)
            projectiveEncoder.setBuffer(projectiveBuffer, offset: 0, index: 2)
            projectiveEncoder.setBuffer(validBuffer, offset: 0, index: 3)
            projectiveEncoder.setBuffer(parameterBuffer, offset: 0, index: 4)
            projectiveEncoder.dispatchThreads(
                grid,
                threadsPerThreadgroup: projectiveGroup
            )
            projectiveEncoder.endEncoding()

            guard let inverseEncoder = command.makeComputeCommandEncoder()
            else {
                throw WalletMetalMobileTestbenchError.metal(
                    "failed to allocate inversion encoder"
                )
            }
            inverseEncoder.setComputePipelineState(inversePipeline)
            inverseEncoder.setBuffer(projectiveBuffer, offset: 0, index: 0)
            inverseEncoder.setBuffer(inverseBuffer, offset: 0, index: 1)
            inverseEncoder.setBuffer(validBuffer, offset: 0, index: 2)
            inverseEncoder.setBuffer(parameterBuffer, offset: 0, index: 3)
            let inverseGridWidth =
                (recordCount + configuration.batchInversionChunkSize - 1) /
                configuration.batchInversionChunkSize
            inverseEncoder.dispatchThreads(
                MTLSize(width: inverseGridWidth, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(
                    width: min(inverseGroup.width, inverseGridWidth),
                    height: 1,
                    depth: 1
                )
            )
            inverseEncoder.endEncoding()

            guard let compressEncoder = command.makeComputeCommandEncoder()
            else {
                throw WalletMetalMobileTestbenchError.metal(
                    "failed to allocate compression encoder"
                )
            }
            compressEncoder.setComputePipelineState(compressPipeline)
            compressEncoder.setBuffer(projectiveBuffer, offset: 0, index: 0)
            compressEncoder.setBuffer(inverseBuffer, offset: 0, index: 1)
            compressEncoder.setBuffer(resultBuffer, offset: 0, index: 2)
            compressEncoder.setBuffer(validBuffer, offset: 0, index: 3)
            compressEncoder.setBuffer(parameterBuffer, offset: 0, index: 4)
            compressEncoder.dispatchThreads(
                grid,
                threadsPerThreadgroup: compressGroup
            )
            compressEncoder.endEncoding()

            let started = DispatchTime.now().uptimeNanoseconds
            command.commit()
            command.waitUntilCompleted()
            let hostNanoseconds =
                DispatchTime.now().uptimeNanoseconds - started
            if let error = command.error {
                throw WalletMetalMobileTestbenchError.metal(
                    "GPU command failed: \(error.localizedDescription)"
                )
            }
            try checkResults(
                resultBuffer: resultBuffer,
                validBuffer: validBuffer,
                corpus: corpus
            )
            let gpuSeconds = collectTiming
                ? max(0, command.gpuEndTime - command.gpuStartTime)
                : 0
            return (collectTiming ? hostNanoseconds : 0, gpuSeconds)
        }

        for _ in 0 ..< configuration.warmupRounds {
            _ = try executeRound(collectTiming: false)
        }
        var hostNanoseconds: UInt64 = 0
        var gpuSeconds = 0.0
        for _ in 0 ..< configuration.rounds {
            let timing = try executeRound(collectTiming: true)
            hostNanoseconds &+= timing.0
            gpuSeconds += timing.1
        }

        let operations = corpus.count * configuration.rounds
        let hostSeconds = Double(hostNanoseconds) / 1_000_000_000
        return WalletMetalMobileReport(
            deviceName: device.name,
            pointsPerRound: corpus.count,
            timedRounds: configuration.rounds,
            operations: operations,
            hostSeconds: hostSeconds,
            gpuSeconds: gpuSeconds,
            derivationsPerSecond: Double(operations) / hostSeconds,
            projectiveThreadsPerGroup: projectiveThreads,
            inverseThreadsPerGroup: inverseThreads,
            compressThreadsPerGroup: compressThreads,
            projectiveExecutionWidth:
                projectivePipeline.threadExecutionWidth,
            corpusFingerprint: corpus.fingerprint,
            validationPassed: true
        )
    }

    private static func validate(
        _ configuration: WalletMetalMobileConfiguration
    ) throws {
        let settings = [
            configuration.rounds,
            configuration.warmupRounds,
            configuration.projectiveThreadsPerGroup,
            configuration.inverseThreadsPerGroup,
            configuration.compressThreadsPerGroup,
            configuration.batchInversionChunkSize,
        ]
        guard settings.allSatisfy({ $0 > 0 }) else {
            throw WalletMetalMobileTestbenchError.invalidConfiguration(
                "all rounds, thread counts and chunk sizes must be positive"
            )
        }
        guard configuration.batchInversionChunkSize <= Int(UInt32.max) else {
            throw WalletMetalMobileTestbenchError.invalidConfiguration(
                "batch inversion chunk size exceeds UInt32"
            )
        }
    }

    private static func makePipeline(
        library: MTLLibrary,
        device: MTLDevice,
        name: String
    ) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else {
            throw WalletMetalMobileTestbenchError.metal(
                "kernel function is missing: \(name)"
            )
        }
        do {
            return try device.makeComputePipelineState(function: function)
        } catch {
            throw WalletMetalMobileTestbenchError.metal(
                "pipeline creation failed for \(name): " +
                error.localizedDescription
            )
        }
    }

    private static func alignedThreadCount(
        preferred: Int,
        pipeline: MTLComputePipelineState
    ) -> Int {
        let maximum = min(
            preferred,
            pipeline.maxTotalThreadsPerThreadgroup
        )
        let width = max(1, pipeline.threadExecutionWidth)
        if maximum < width {
            return maximum
        }
        return max(width, (maximum / width) * width)
    }

    private static func sharedBuffer(
        device: MTLDevice,
        bytes: [UInt8],
        label: String
    ) throws -> MTLBuffer {
        try bytes.withUnsafeBytes { raw in
            guard let address = raw.baseAddress,
                  let buffer = device.makeBuffer(
                      bytes: address,
                      length: raw.count,
                      options: .storageModeShared
                  )
            else {
                throw WalletMetalMobileTestbenchError.metal(
                    "failed to allocate \(label)"
                )
            }
            buffer.label = label
            return buffer
        }
    }

    private static func checkResults(
        resultBuffer: MTLBuffer,
        validBuffer: MTLBuffer,
        corpus: MobileVectorCorpus
    ) throws {
        let results = resultBuffer.contents().assumingMemoryBound(to: UInt8.self)
        let valid = validBuffer.contents().assumingMemoryBound(to: UInt8.self)
        for index in 0 ..< corpus.count {
            guard valid[index] == 1 else {
                throw WalletMetalMobileTestbenchError.validation(
                    "valid record \(index) was rejected"
                )
            }
            let base = index * 32
            for byte in 0 ..< 32
                where results[base + byte] != corpus.expected[base + byte]
            {
                throw WalletMetalMobileTestbenchError.validation(
                    "Dalek mismatch at record \(index), byte \(byte)"
                )
            }
        }
        let invalidIndex = corpus.count
        guard valid[invalidIndex] == 0 else {
            throw WalletMetalMobileTestbenchError.validation(
                "Dalek-rejected point was accepted"
            )
        }
        let invalidBase = invalidIndex * 32
        for byte in 0 ..< 32 where results[invalidBase + byte] != 0 {
            throw WalletMetalMobileTestbenchError.validation(
                "invalid point left nonzero output at byte \(byte)"
            )
        }
    }

    private static func parseCorpus(_ data: Data) throws -> MobileVectorCorpus {
        let raw = [UInt8](data)
        guard raw.count >= vectorHeaderBytes else {
            throw WalletMetalMobileTestbenchError.invalidCorpus(
                "file is shorter than the 88-byte header"
            )
        }
        guard Array(raw[0 ..< 8]) == vectorMagic else {
            throw WalletMetalMobileTestbenchError.invalidCorpus(
                "magic is not MWMTV1"
            )
        }
        guard readLE32(raw, 8) == 1 else {
            throw WalletMetalMobileTestbenchError.invalidCorpus(
                "unsupported format version \(readLE32(raw, 8))"
            )
        }
        let count = Int(readLE32(raw, 12))
        guard count > 0 else {
            throw WalletMetalMobileTestbenchError.invalidCorpus(
                "record count is zero"
            )
        }
        let recordBytes = count.multipliedReportingOverflow(by: 64)
        guard !recordBytes.overflow,
              vectorHeaderBytes + recordBytes.partialValue == raw.count
        else {
            throw WalletMetalMobileTestbenchError.invalidCorpus(
                "file size does not match record count"
            )
        }

        var points = [UInt8]()
        var expected = [UInt8]()
        points.reserveCapacity(count * 32)
        expected.reserveCapacity(count * 32)
        for record in 0 ..< count {
            let offset = vectorHeaderBytes + record * 64
            points += raw[offset ..< offset + 32]
            expected += raw[offset + 32 ..< offset + 64]
        }
        return MobileVectorCorpus(
            scalar: Array(raw[16 ..< 48]),
            points: points,
            expected: expected,
            invalid: Array(raw[56 ..< 88]),
            fingerprint: readLE64(raw, 48),
            count: count
        )
    }

    private static func readLE32(
        _ bytes: [UInt8],
        _ offset: Int
    ) -> UInt32 {
        UInt32(bytes[offset])
            | (UInt32(bytes[offset + 1]) << 8)
            | (UInt32(bytes[offset + 2]) << 16)
            | (UInt32(bytes[offset + 3]) << 24)
    }

    private static func readLE64(
        _ bytes: [UInt8],
        _ offset: Int
    ) -> UInt64 {
        var value: UInt64 = 0
        for index in 0 ..< 8 {
            value |= UInt64(bytes[offset + index]) << UInt64(index * 8)
        }
        return value
    }
}
