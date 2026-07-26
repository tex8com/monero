import Foundation

@main
enum MobileRunnerMacSmoke {
    static func main() throws {
        guard CommandLine.arguments.count == 3 else {
            FileHandle.standardError.write(
                Data("usage: mobile-runner-mac-smoke VECTORS KERNEL\n".utf8)
            )
            exit(2)
        }
        let vectorData = try Data(
            contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])
        )
        let kernelSource = try String(
            contentsOfFile: CommandLine.arguments[2],
            encoding: .utf8
        )
        let report = try WalletMetalMobileTestbench.run(
            vectorData: vectorData,
            kernelSource: kernelSource,
            configuration: WalletMetalMobileConfiguration(
                rounds: 1,
                warmupRounds: 1
            )
        )
        print("testbench=wallet_metal_mobile_runner_mac_smoke")
        print("metal_device=\(report.deviceName)")
        print("points_per_round=\(report.pointsPerRound)")
        print("operations=\(report.operations)")
        print(
            String(
                format: "host_submit_wait_seconds=%.9f",
                report.hostSeconds
            )
        )
        print(
            String(
                format: "gpu_execution_seconds=%.9f",
                report.gpuSeconds
            )
        )
        print(
            String(
                format: "derivations_per_second=%.3f",
                report.derivationsPerSecond
            )
        )
        print(
            "projective_threads_per_threadgroup=" +
            "\(report.projectiveThreadsPerGroup)"
        )
        print(
            "inverse_threads_per_threadgroup=" +
            "\(report.inverseThreadsPerGroup)"
        )
        print(
            "compress_threads_per_threadgroup=" +
            "\(report.compressThreadsPerGroup)"
        )
        print(
            "corpus_fingerprint_fnv1a64=0x" +
            String(report.corpusFingerprint, radix: 16)
        )
        print("validation=\(report.validationPassed ? "pass" : "fail")")
    }
}
