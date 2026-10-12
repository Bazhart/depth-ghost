DEPTH GHOST — a Claude Code skill
Turns any photo or video into the glowing "ghost depth" 3D look.
Photos: JPG / PNG / iPhone HEIC.  Videos: MP4 / iPhone MOV (audio kept).

INSTALL (Mac, 1 minute)
  1. Unzip this. You get a folder called "depth-ghost".
  2. Move that folder into:  ~/.claude/skills/
     (In Finder: Cmd+Shift+G, paste ~/.claude/skills, press Enter. Create the
      "skills" folder if it doesn't exist.)
     Or in Terminal:
        mkdir -p ~/.claude/skills && mv ~/Downloads/depth-ghost ~/.claude/skills/
  3. Open a NEW Claude Code session (desktop app Code tab, the `claude` terminal,
     or the Claude Code extension in VS Code / Antigravity / Cursor).

USE
  Type /depth-ghost and attach a photo (iPhone HEIC works), e.g.
     /depth-ghost
     /depth-ghost show the folds, bring the background back
     /depth-ghost 1:1 crop, 1080
     /depth-ghost gamma 2.5, exposure 1.7, detail 1.5, lift 0.45
     /depth-ghost ~/Desktop/clip.mov        (video: Claude renders a 3-second
                                            preview first, then the full clip)
  Video takes a while: every frame goes through the AI model (roughly
  0.2-0.5 s per frame on Apple Silicon, 1.5-2.5 s on Intel Macs).
  The first time, Claude installs what it needs (~2 GB: Python env + AI depth
  model) into ~/.depth-ghost — takes a few minutes, only once.

NEEDS
  Claude Code (not the regular claude.ai chat — it needs to run code on your Mac).
  Python 3.10-3.12 or "uv" (Claude will tell you if neither is there).
  ffmpeg for video (brew install ffmpeg).
  Works on Apple Silicon and Intel Macs, and on Linux.
