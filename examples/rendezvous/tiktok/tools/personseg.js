// Person mattes with Apple Vision via JXA (built into macOS; no compiler needed).
// One stateful VNGeneratePersonSegmentationRequest per shot, frames in order: the matte is temporally stable
// inside a shot and restarts at every cut.  usage: osascript -l JavaScript personseg.js IN_DIR OUT_DIR s:e[,s:e]
ObjC.import('Foundation'); ObjC.import('AppKit'); ObjC.import('Vision'); ObjC.import('CoreImage');
function run(argv) {
  const inDir = argv[0], outDir = argv[1];
  let n = 0;
  for (const range of argv[2].split(',')) {
    const [s, e] = range.split(':').map(Number);
    const req = $.VNGeneratePersonSegmentationRequest.alloc.init;
    req.qualityLevel = 0;                                  // accurate
    req.outputPixelFormat = 1278226488;                    // kCVPixelFormatType_OneComponent8
    const seq = $.VNSequenceRequestHandler.alloc.init;
    for (let j = s; j < e; j++) {
      const name = ('000000' + j).slice(-6);
      const img = $.CIImage.imageWithContentsOfURL($.NSURL.fileURLWithPath(inDir + '/' + name + '.jpg'));
      const ok = seq.performRequestsOnCIImageError($.NSArray.arrayWithObject(req), img, Ref());
      if (!ok || req.results.count == 0) { console.log('fail ' + name); continue; }
      const m = $.CIImage.imageWithCVPixelBuffer(req.results.objectAtIndex(0).pixelBuffer);
      const rep = $.NSBitmapImageRep.alloc.initWithCIImage(m);
      rep.representationUsingTypeProperties(4, $()).writeToFileAtomically(outDir + '/' + name + '.png', true);
      n++;
    }
    console.log('shot ' + s + '-' + e + ' done');
  }
  return 'wrote ' + n;
}
