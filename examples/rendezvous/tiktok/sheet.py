"""Contact sheet of preview stills: python sheet.py OUT.jpg files... (labels from the file names)."""
import sys, cv2, numpy as np
from pathlib import Path
files = sys.argv[2:]; tw, th = 300, 533; cols = min(len(files), 7)
tiles = []
for f in files:
    im = cv2.resize(cv2.imread(f), (tw, th), interpolation=cv2.INTER_AREA)
    cv2.putText(im, Path(f).stem.split("_")[-1], (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    tiles.append(im)
while len(tiles) % cols: tiles.append(np.zeros((th, tw, 3), np.uint8))
cv2.imwrite(sys.argv[1], np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
