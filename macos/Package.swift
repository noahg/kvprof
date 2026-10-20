// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "KvprofApp",
    platforms: [.macOS(.v13)],
    targets: [.executableTarget(name: "KvprofApp", path: "Sources/KvprofApp")]
)
