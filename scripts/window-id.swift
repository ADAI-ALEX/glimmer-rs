// Print the CGWindowID of the on-screen Terminal window whose title contains the given marker.
import CoreGraphics
import Foundation

let marker = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "glimmer-shot"
let info = CGWindowListCopyWindowInfo([.optionOnScreenOnly], kCGNullWindowID) as? [[String: Any]] ?? []
for w in info {
    let owner = w[kCGWindowOwnerName as String] as? String ?? ""
    let name = w[kCGWindowName as String] as? String ?? ""
    if owner == "Terminal" && name.contains(marker), let id = w[kCGWindowNumber as String] as? Int {
        print(id)
        exit(0)
    }
}
exit(1)
