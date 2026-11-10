// Crop a Terminal window capture's title bar and scale it for the README.
// The terminal content is not altered, only trimmed at the top and resized.
//
//   target/crop-screenshot <in.png> <out.png> <top-pixels> <max-width>
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

let a = CommandLine.arguments
guard a.count == 5, let top = Int(a[3]), let maxWidth = Int(a[4]) else {
    FileHandle.standardError.write("usage: crop-screenshot <in.png> <out.png> <top-pixels> <max-width>\n".data(using: .utf8)!)
    exit(2)
}
guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: a[1]) as CFURL, nil),
      let image = CGImageSourceCreateImageAtIndex(src, 0, nil),
      let cropped = image.cropping(to: CGRect(x: 0, y: top, width: image.width, height: image.height - top)) else {
    FileHandle.standardError.write("cannot read or crop \(a[1])\n".data(using: .utf8)!)
    exit(1)
}
let scale = min(1.0, Double(maxWidth) / Double(cropped.width))
let w = Int((Double(cropped.width) * scale).rounded()), h = Int((Double(cropped.height) * scale).rounded())
guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                          space: CGColorSpace(name: CGColorSpace.sRGB)!,
                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { exit(1) }
ctx.interpolationQuality = .high
ctx.draw(cropped, in: CGRect(x: 0, y: 0, width: w, height: h))
guard let out = ctx.makeImage(),
      let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: a[2]) as CFURL, UTType.png.identifier as CFString, 1, nil) else { exit(1) }
CGImageDestinationAddImage(dest, out, nil)
guard CGImageDestinationFinalize(dest) else { exit(1) }
print("\(a[2]) \(w)×\(h)")
