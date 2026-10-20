import AppKit
import ServiceManagement
import SwiftUI

@main
struct KvprofApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var server = Server.shared

    var body: some Scene {
        MenuBarExtra {
            MenuContent(server: server)
        } label: {
            Image(systemName: symbol)
        }
    }

    private var symbol: String {
        switch server.phase {
        case .running(let s): return s.upstream_ok ? "speedometer" : "hourglass"
        case .starting: return "hourglass"
        case .portBusy, .failed: return "exclamationmark.triangle"
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        Server.shared.begin()
    }

    func applicationWillTerminate(_ notification: Notification) {
        Server.shared.quit()
    }
}

struct MenuContent: View {
    @ObservedObject var server: Server

    var body: some View {
        Text(headline)
        if let detail { Text(detail) }
        Divider()
        Button("Open Dashboard") { server.openDashboard() }
            .keyboardShortcut("d")
            .disabled(server.status == nil)
        Button("Copy Claude Code Command") { copy("ANTHROPIC_BASE_URL=\(server.tapURL) claude") }
        Button("Copy OpenAI Base URL") { copy("\(server.tapURL)/v1") }
        Button("Show Traces in Finder") { server.revealTraces() }
        Divider()
        Button("Check Setup…") {
            Windows.show("doctor", title: "kvprof: Check Setup", size: NSSize(width: 640, height: 420)) {
                DoctorView(settings: server.settings)
            }
        }
        Button("Settings…") {
            Windows.show("settings", title: "kvprof Settings", size: NSSize(width: 480, height: 300)) {
                SettingsView(server: server)
            }
        }
        .keyboardShortcut(",")
        Divider()
        Button("Quit kvprof") { NSApp.terminate(nil) }
            .keyboardShortcut("q")
    }

    private var headline: String {
        switch server.phase {
        case .starting: return "Starting…"
        case .portBusy: return "Port \(server.settings.tapPort) is in use by another program"
        case .failed(let why): return why
        case .running(let s):
            if !s.upstream_ok { return "Waiting for llama-server at \(s.upstream)" }
            return "Recording" + (s.model.map { " · \($0)" } ?? "")
        }
    }

    private var detail: String? {
        switch server.phase {
        case .portBusy: return "Change the tap port in Settings, or stop the other program"
        case .running(let s):
            guard let n = s.n_requests else { return nil }
            return n == 0 ? "No requests yet: point your client at \(server.tapURL)" : "\(n) request\(n == 1 ? "" : "s") for this model"
        default: return nil
        }
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}

/// Windows of a menu-bar app: created on demand, one per id, brought to the front.
@MainActor
enum Windows {
    private static var open: [String: NSWindow] = [:]

    static func show<V: View>(_ id: String, title: String, size: NSSize, @ViewBuilder content: () -> V) {
        open[id]?.close()
        let w = NSWindow(contentRect: NSRect(origin: .zero, size: size), styleMask: [.titled, .closable, .resizable],
                         backing: .buffered, defer: false)
        w.title = title
        w.isReleasedWhenClosed = false
        w.contentView = NSHostingView(rootView: content())
        w.center()
        open[id] = w
        NSApp.activate(ignoringOtherApps: true)
        w.makeKeyAndOrderFront(nil)
    }

    static func close(_ id: String) {
        open[id]?.close()
        open[id] = nil
    }
}

struct SettingsView: View {
    @ObservedObject var server: Server
    @State private var upstream = ""
    @State private var tapPort = ""
    @State private var guiPort = ""
    @State private var serverLog = ""
    @State private var atLogin = SMAppService.mainApp.status == .enabled
    @State private var problem: String?

    var body: some View {
        Form {
            TextField("llama-server URL", text: $upstream)
            TextField("Tap port (clients connect here)", text: $tapPort)
            TextField("Dashboard port", text: $guiPort)
            HStack {
                TextField("llama-server log file", text: $serverLog, prompt: Text("optional"))
                Button("Choose…") { chooseLog() }
            }
            Text("The log gives server-side prefill and decode times. Start llama-server with --log-file <path> and choose that file here.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Toggle("Start kvprof at login", isOn: $atLogin)
                .onChange(of: atLogin) { on in setAtLogin(on) }
            if let problem {
                Text(problem).font(.footnote).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Spacer()
                Button("Apply") { apply() }.keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .onAppear {
            let s = server.settings
            upstream = s.upstream
            tapPort = String(s.tapPort)
            guiPort = String(s.guiPort)
            serverLog = s.serverLog
        }
    }

    private func apply() {
        let url = upstream.trimmingCharacters(in: .whitespaces)
        guard let tap = Int(tapPort), let gui = Int(guiPort), (1...65535).contains(tap), (1...65535).contains(gui), tap != gui else {
            problem = "The ports must be two different numbers between 1 and 65535."
            return
        }
        guard let parsed = URL(string: url), parsed.scheme == "http" || parsed.scheme == "https", parsed.host != nil else {
            problem = "The llama-server URL should look like http://127.0.0.1:8080."
            return
        }
        problem = nil
        server.apply(AppSettings(upstream: url, tapPort: tap, guiPort: gui, serverLog: serverLog.trimmingCharacters(in: .whitespaces)))
        Windows.close("settings")
    }

    private func chooseLog() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.message = "The file llama-server writes with --log-file"
        if panel.runModal() == .OK, let url = panel.url { serverLog = url.path }
    }

    private func setAtLogin(_ on: Bool) {
        guard on != (SMAppService.mainApp.status == .enabled) else { return }
        do {
            if on { try SMAppService.mainApp.register() } else { try SMAppService.mainApp.unregister() }
            problem = nil
        } catch {
            problem = "Could not change the login item: \(error.localizedDescription)"
            atLogin = SMAppService.mainApp.status == .enabled
        }
    }
}

struct DoctorView: View {
    let settings: AppSettings
    @State private var report = "Checking \(Server.shared.settings.upstream)…"

    var body: some View {
        ScrollView {
            Text(report)
                .font(.system(.body, design: .monospaced))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(14)
        }
        .task { report = await Server.doctor(settings) }
    }
}
