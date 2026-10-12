// Person mattes with Apple Vision (VNGeneratePersonSegmentationRequest, accurate). One stateful request per
// shot, frames in order, so the matte stays temporally stable inside a shot and restarts at every cut.
// usage: personseg IN_DIR OUT_DIR start:end[,start:end...]
import Foundation
import Vision
import CoreImage

let a = CommandLine.arguments
let inDir = a[1], outDir = a[2]
let ctx = CIContext()
for range in a[3].split(separator: ",") {
    let p = range.split(separator: ":"); let s = Int(p[0])!, e = Int(p[1])!
    let req = VNGeneratePersonSegmentationRequest()
    req.qualityLevel = .accurate
    req.outputPixelFormat = kCVPixelFormatType_OneComponent8
    let seq = VNSequenceRequestHandler()
    for j in s..<e {
        autoreleasepool {
            let name = String(format: "%06d", j)
            guard let img = CIImage(contentsOf: URL(fileURLWithPath: "\(inDir)/\(name).jpg")) else { print("missing \(name)"); return }
            do { try seq.perform([req], on: img) } catch { print("fail \(name): \(error)"); return }
            guard let obs = req.results?.first else { print("no matte \(name)"); return }
            let m = CIImage(cvPixelBuffer: obs.pixelBuffer)
            try? ctx.writePNGRepresentation(of: m, to: URL(fileURLWithPath: "\(outDir)/\(name).png"), format: .L8,
                                            colorSpace: CGColorSpaceCreateDeviceGray())
        }
    }
    print("shot \(s)-\(e) done")
}
