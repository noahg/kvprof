// Draws the app icon: three prompts as bars, grey for what the server reused and orange for what it computed
// fresh (the colours of the request list). Run from the repository root:  swift macos/make-icon.swift
import AppKit

func png(_ pixels: Int) -> Data {
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: pixels, pixelsHigh: pixels, bitsPerSample: 8, samplesPerPixel: 4,
                               hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    rep.size = NSSize(width: 1024, height: 1024)   // draw in a 1024-point space whatever the pixel size
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    let tile = NSBezierPath(roundedRect: NSRect(x: 100, y: 100, width: 824, height: 824), xRadius: 186, yRadius: 186)
    NSGradient(colors: [NSColor(srgbRed: 0.17, green: 0.20, blue: 0.27, alpha: 1), NSColor(srgbRed: 0.08, green: 0.09, blue: 0.13, alpha: 1)])!
        .draw(in: tile, angle: -90)
    let grey = NSColor(srgbRed: 0.79, green: 0.79, blue: 0.76, alpha: 1), orange = NSColor(srgbRed: 0.94, green: 0.64, blue: 0.23, alpha: 1)
    // (total width, reused width) of each bar, top to bottom: a cold start, then two cheap extensions
    for (row, bar) in [(300.0, 0.0), (420.0, 300.0), (560.0, 420.0)].enumerated() {
        let y = 620.0 - Double(row) * 160.0
        let whole = NSBezierPath(roundedRect: NSRect(x: 232, y: y, width: bar.0, height: 104), xRadius: 26, yRadius: 26)
        NSGraphicsContext.saveGraphicsState()
        whole.addClip()
        orange.setFill()
        NSRect(x: 232, y: y, width: bar.0, height: 104).fill()
        grey.setFill()
        NSRect(x: 232, y: y, width: bar.1, height: 104).fill()
        NSGraphicsContext.restoreGraphicsState()
    }
    NSGraphicsContext.restoreGraphicsState()
    return rep.representation(using: .png, properties: [:])!
}

let set = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("kvprof.iconset")
try? FileManager.default.removeItem(at: set)
try FileManager.default.createDirectory(at: set, withIntermediateDirectories: true)
for size in [16, 32, 128, 256, 512] {
    try png(size).write(to: set.appendingPathComponent("icon_\(size)x\(size).png"))
    try png(size * 2).write(to: set.appendingPathComponent("icon_\(size)x\(size)@2x.png"))
}
let tool = Process()
tool.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
tool.arguments = ["-c", "icns", set.path, "-o", "macos/AppIcon.icns"]
try tool.run()
tool.waitUntilExit()
exit(tool.terminationStatus)
