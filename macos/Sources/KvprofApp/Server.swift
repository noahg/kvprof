import AppKit
import Foundation

/// `GET /api/status` of a running `kvprof start` (see serve.py).
struct Status: Decodable, Equatable {
    var ready: Bool
    var upstream: String
    var upstream_ok: Bool
    var upstream_error: String?
    var tap_url: String?
    var gui_url: String?
    var trace_path: String?
    var model: String?
    var n_requests: Int?
}

/// The run file `kvprof start` writes next to the traces: where its dashboard is.
private struct RunFile: Decodable {
    var gui_url: String
}

enum Phase: Equatable {
    case starting
    case running(Status)
    case portBusy
    case failed(String)
}

/// What the user can change. Stored in UserDefaults, handed to `kvprof start` as flags.
struct AppSettings: Equatable {
    var upstream: String
    var tapPort: Int
    var guiPort: Int
    var serverLog: String

    static let defaults = AppSettings(upstream: "http://127.0.0.1:8080", tapPort: 8081, guiPort: 8084, serverLog: "")

    static func load(_ d: UserDefaults = .standard) -> AppSettings {
        var s = defaults
        if let v = d.string(forKey: "upstream"), !v.isEmpty { s.upstream = v }
        if d.integer(forKey: "tapPort") > 0 { s.tapPort = d.integer(forKey: "tapPort") }
        if d.integer(forKey: "guiPort") > 0 { s.guiPort = d.integer(forKey: "guiPort") }
        s.serverLog = d.string(forKey: "serverLog") ?? ""
        return s
    }

    func save(_ d: UserDefaults = .standard) {
        d.set(upstream, forKey: "upstream")
        d.set(tapPort, forKey: "tapPort")
        d.set(guiPort, forKey: "guiPort")
        d.set(serverLog, forKey: "serverLog")
    }
}

/// Keeps one `kvprof start` alive and mirrors its status. If one is already serving the tap port (started from a
/// terminal, say), it is adopted rather than fought: the run file says where it is.
@MainActor
final class Server: ObservableObject {
    static let shared = Server()

    @Published private(set) var phase: Phase = .starting
    @Published private(set) var settings = AppSettings.load()

    private var process: Process?
    private var timer: Timer?
    private var notBefore = Date.distantPast
    private var backoff: TimeInterval = 1
    private var quitting = false

    static let exitTapPort: Int32 = 3   // start.py EXIT_TAP_PORT

    nonisolated static var runtime: URL { Bundle.main.resourceURL!.appendingPathComponent("runtime/bin/kvprof") }
    nonisolated static var dataDir: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("kvprof")
    }
    nonisolated static var logFile: URL {
        FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask)[0].appendingPathComponent("Logs/kvprof/kvprof.log")
    }

    var status: Status? {
        if case .running(let s) = phase { return s }
        return nil
    }
    var tapURL: String { status?.tap_url ?? "http://127.0.0.1:\(settings.tapPort)" }

    func begin() {
        timer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in
            Task { @MainActor in await self?.tick() }
        }
        Task { await tick() }
    }

    /// Stop the child and wait for it, so the ports are free when the app is gone.
    func quit() {
        quitting = true
        timer?.invalidate()
        guard let p = process, p.isRunning else { return }
        p.terminate()
        let deadline = Date().addingTimeInterval(3)
        while p.isRunning && Date() < deadline { usleep(50_000) }
    }

    func apply(_ new: AppSettings) {
        guard new != settings else { return }
        new.save()
        settings = new
        phase = .starting
        backoff = 1
        notBefore = .distantPast
        process?.terminate()   // the next tick starts it again with the new flags
    }

    func openDashboard() {
        if let s = status?.gui_url, let url = URL(string: s) { NSWorkspace.shared.open(url) }
    }

    func revealTraces() {
        let path = status?.trace_path ?? Server.dataDir.appendingPathComponent("traces").path
        try? FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
        NSWorkspace.shared.open(URL(fileURLWithPath: path))
    }

    private func tick() async {
        if quitting { return }
        if let s = await fetchStatus() {
            phase = .running(s)
            backoff = 1
            if !UserDefaults.standard.bool(forKey: "openedDashboard") {
                UserDefaults.standard.set(true, forKey: "openedDashboard")   // first launch: show where things are
                openDashboard()
            }
            return
        }
        if process?.isRunning == true {
            if case .running = phase { phase = .starting }
            return
        }
        if Date() >= notBefore { launch() }
    }

    private func fetchStatus() async -> Status? {
        let run = Server.dataDir.appendingPathComponent("run-\(settings.tapPort).json")
        guard let data = try? Data(contentsOf: run),
              let file = try? JSONDecoder().decode(RunFile.self, from: data),
              let url = URL(string: file.gui_url + "api/status") else { return nil }
        var req = URLRequest(url: url)
        req.timeoutInterval = 2
        req.cachePolicy = .reloadIgnoringLocalCacheData
        guard let (body, _) = try? await URLSession.shared.data(for: req) else { return nil }
        return try? JSONDecoder().decode(Status.self, from: body)
    }

    private func launch() {
        let fm = FileManager.default
        try? fm.createDirectory(at: Server.dataDir, withIntermediateDirectories: true)
        try? fm.createDirectory(at: Server.logFile.deletingLastPathComponent(), withIntermediateDirectories: true)
        if let size = (try? fm.attributesOfItem(atPath: Server.logFile.path))?[.size] as? Int, size > 5_000_000 {
            try? fm.removeItem(at: Server.logFile)
        }
        if !fm.fileExists(atPath: Server.logFile.path) { fm.createFile(atPath: Server.logFile.path, contents: nil) }
        let log = try? FileHandle(forWritingTo: Server.logFile)
        _ = try? log?.seekToEnd()

        let p = Process()
        p.executableURL = Server.runtime
        p.arguments = ["start", "--no-open", "--exit-with-parent", "--quiet",
                       "--listen", "127.0.0.1:\(settings.tapPort)", "--port", "\(settings.guiPort)", "--upstream", settings.upstream]
            + (settings.serverLog.isEmpty ? [] : ["--server-log", settings.serverLog])
        var env = ProcessInfo.processInfo.environment
        env["KVPROF_DATA_DIR"] = Server.dataDir.path
        env["KVPROF_TRACE_DIR"] = nil
        p.environment = env
        p.currentDirectoryURL = Server.dataDir   // not a project directory: traces go to the per-user root
        p.standardOutput = log
        p.standardError = log
        p.terminationHandler = { [weak self] done in
            Task { @MainActor in self?.exited(done) }
        }
        do {
            try p.run()
            process = p
        } catch {
            phase = .failed("cannot run the bundled kvprof: \(error.localizedDescription)")
            notBefore = Date().addingTimeInterval(30)
        }
    }

    private func exited(_ p: Process) {
        guard p === process, !quitting else { return }
        process = nil
        switch p.terminationStatus {
        case 0, 15:   // stopped on purpose, or another instance already serves this tap port: the next tick finds out
            notBefore = Date().addingTimeInterval(1)
        case Server.exitTapPort:
            phase = .portBusy
            notBefore = Date().addingTimeInterval(10)
        default:
            phase = .failed("kvprof stopped (status \(p.terminationStatus)). The log is in ~/Library/Logs/kvprof.")
            backoff = min(backoff * 2, 30)
            notBefore = Date().addingTimeInterval(backoff)
        }
    }

    /// `kvprof doctor` against the configured server; its report, as text.
    nonisolated static func doctor(_ s: AppSettings) async -> String {
        await withCheckedContinuation { done in
            DispatchQueue.global().async {
                let p = Process()
                let pipe = Pipe()
                p.executableURL = runtime
                p.arguments = ["doctor", "--upstream", s.upstream] + (s.serverLog.isEmpty ? [] : ["--server-log", s.serverLog])
                p.standardOutput = pipe
                p.standardError = pipe
                do {
                    try p.run()
                    let out = pipe.fileHandleForReading.readDataToEndOfFile()
                    p.waitUntilExit()
                    done.resume(returning: String(decoding: out, as: UTF8.self))
                } catch {
                    done.resume(returning: "cannot run the bundled kvprof: \(error.localizedDescription)")
                }
            }
        }
    }
}
