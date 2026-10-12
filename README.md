# depth-ghost

Turn any photo or video into the glowing **ghost depth** look (near things glow white, far things fall into black, like a depth scan) with one command in **Claude Code**. This repo also has the full recipe behind the *RENDËZVOUS* video: the music-video edit and the TikTok clips with seamless loops.

> 🇫🇷 **En bref :** installe le skill, ouvre Claude Code, tape `/depth-ghost` et glisse une photo ou une vidéo. Les guides ci-dessous expliquent comment faire une image, une vidéo, un clip complet calé sur le son et des TikToks qui bouclent.

## What's in here

| Folder | What it is |
|---|---|
| [`skill/depth-ghost`](skill/depth-ghost) | The Claude Code skill: `/depth-ghost` for photos, videos, folders |
| [`guides/`](guides) | Step-by-step guides (below) |
| [`examples/rendezvous`](examples/rendezvous) | The exact scripts behind the RENDËZVOUS edit and TikToks: a reference to adapt, not a one-click tool |

## Install (Mac, a few minutes the first time)

```bash
git clone https://github.com/Bazhart/depth-ghost.git
cd depth-ghost
bash install.sh
```

Then open a **new** Claude Code session (desktop app, `claude` in a terminal, or the VS Code / Cursor extension) and type:

```
/depth-ghost
```

and attach a photo or a video. The first run installs a private Python environment in `~/.depth-ghost` (~2 GB with the depth model). You only do this once.

**Needs:** [Claude Code](https://claude.com/claude-code), Python 3.10–3.12 or [uv](https://docs.astral.sh/uv/), and ffmpeg (`brew install ffmpeg`). Apple Silicon is fastest; Intel Macs and Linux work too.

## Guides

1. **[Make a picture](guides/01-picture.md):** one photo to ghost depth, settings in plain words.
2. **[Make a video](guides/02-video.md):** smooth, flicker-free video, cut by cut.
3. **[Cut a music video around it](guides/03-music-video.md):** a camera keyed to the beat, colour punches, light trails, posterize time, speed ramps, layering, a flash-safety check.
4. **[TikTok clips & perfect loops](guides/04-tiktok.md):** vertical framing, clean person masks, 8-bar loops with no gap.
5. **[Prompts that worked](guides/prompts.md):** what to type in Claude Code.

## Credits & licenses

- **Depth model:** [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), downloaded at first run through Hugging Face `transformers`. **Small** is Apache-2.0; **Base and Large are CC BY-NC 4.0 (non-commercial)**. The skill uses Large for photos and Small for video by default, and the RENDËZVOUS renders used Base. For commercial work use `--model small`, or check the model licenses yourself.
- **Code in this repo:** MIT, see [LICENSE](LICENSE).
- Built with **Claude Code** ✻
