# Demo video publishing

- **Video file**: `docs/assets/demo_promo.mp4` (regenerate with `python docs/scripts/generate_media.py`).
- **GitHub**: attach the MP4 to the next release and link it in the README and release notes. Optionally add it to a pinned Discussion.
- **LinkedIn**: upload the MP4 directly (LinkedIn favors native uploads) and pair it with the caption below.

## Suggested LinkedIn caption

> 🚀 Meet **Modeling-GUI** — a no-code desktop app for real modeling work. Load a CSV, click **Smart Analyze**, and get metrics, plots, reports, and project files you can share. AutoML, forecasting, explainability, drift checks, and reproducibility are built in. Open source, runs locally.  
>
> 🔗 Repo: https://github.com/chandrasekarnarayana/Modeling-GUI  
> 💻 Docs: https://chandrasekarnarayana.github.io/Modeling-GUI/  
>
> #nocode #machinelearning #datascience #python #opensource #automl

## Notes

- Slides live in `docs/screenshots/demo_slide_*.png`; the sequence file is `docs/screenshots/demo_slides.txt`.
- Keep captions short; LinkedIn truncates after ~3 lines—put the hook first.
- If you want audio/VO, record separately and mux with `ffmpeg -i demo_promo.mp4 -i voiceover.wav -shortest -c:v copy -c:a aac demo_promo_vo.mp4`.
- For the long-form 6–8 minute walkthrough, follow `docs/marketing/demo_storyboard_longform.md` and place the recorded video at `docs/assets/demo_promo.mp4` before releasing.
