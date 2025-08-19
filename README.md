Sonify the Manga Space

This project is an end-to-end, fully open-source pipeline that converts two-speaker manga panels into a short audio drama. It performs: light OCR cleaning → speaker-aware TTS (XTTS-v2) → text-to-music BGM (MusicGen) → rule-based mixing (−16 LUFS, side-chain ducking, light EQ/pan) → objective intelligibility evaluation (WER/CER via Whisper).
Design goals: intelligibility first, reproducibility, and single-GPU practicality.

Features

Two-speaker dialogue synthesis with stable A/B voices (XTTS-v2), optional mild pitch shift and de-essing to create subtle timbre separation.
Steady lo-fi background music from MusicGen, then spectral notch (≈2–4 kHz) + −6 dB ducking keyed by speech to reduce masking.
Fixed loudness target (−16 LUFS, true-peak ≤ −1 dBTP) for listener comfort and comparability across runs.
Objective evaluation: WER/CER computed from Whisper transcripts with evaluation-time normalization (lowercasing, depunctuation, number & colloquial mapping).
Config-driven: all tunables (speaker IDs, pitch, EQ, duck depth/attack/release, loudness, sample rate, output paths) live in a human-readable YAML/JSON.

Quick Start (Windows + conda, single GPU)

Prerequisites:
Windows 11, NVIDIA GPU (tested on RTX 4060 Laptop)
CUDA Runtime compatible with PyTorch 2.1.0 + cu121
Anaconda / Miniconda

1)Create environment
conda create -n audi_manga39 python=3.9 -y
conda activate audi_manga39

Install PyTorch (CUDA 12.1 build):
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121

Core libs (pin to known-good versions from this project):
pip install TTS==0.22.0 --no-build-isolation --prefer-binary
pip install audiocraft==1.3.0
pip install openai-whisper==20231117 jiwer==3.0.4 pydub==0.25.1 soundfile==0.12.1 librosa==0.10.1 numpy pandas pyyaml rich tqdm

2) Prepare a tiny demo dataset
Place a small, cleaned set of two-speaker panels (PNG/JPG) in a folder, e.g.:
DATA\panels_clean\   (dozens of images is enough for the demo)

3) Generate dialogue (TTS)
python src\tts\tts_batch_demo_safe.py ^
  --input_json OUTPUTS\dialogs_demo_spk_deess.json ^
  --out_dir OUTPUTS\wav_demo ^
  --voice_A "Claribel Dervla" ^
  --voice_B "Daisy Studious" ^
  --pitch_B_semitones 2 ^
  --device cuda

4) Generate BGM (MusicGen)
python src\music\music_gen.py ^
  --prompt_file CONFIG\musicgen_prompt.txt ^
  --duration_sec 90 ^
  --out_wav OUTPUTS\bgm.wav ^
  --device cuda

5) Mix (−16 LUFS, duck −6 dB, pan/EQ)
python src\mix\mix_constant_bgm_with_panning.py ^
  --dialog_dir OUTPUTS\wav_demo ^
  --bgm OUTPUTS\bgm.wav ^
  --config CONFIG\config_demo.yaml ^
  --out_wav OUTPUTS\demo_mix.wav ^
  --out_srt OUTPUTS\demo_mix.srt

6) Evaluate intelligibility (WER/CER)
python src\eval\wer_eval_v3.py ^
  --audio OUTPUTS\demo_mix.wav ^
  --ref_json OUTPUTS\dialogs_demo_spk_deess.json ^
  --model small.en ^
  --device cuda

Building / Cleaning the Dialogue JSON

If you start from images + OCR (CSV/JSON), use the data-prep scripts to:

Filter for exactly two speakers and remove truncated bubbles or non-dialogue lines.

Lightly clean (unify quotes/dashes, collapse whitespace).

Assign speakers A/B within a panel (alternate; cross-panel continuity heuristic).

Export a JSON like:{
  "id": 14903,
  "page_no": 12,
  "panel_no": 3,
  "textbox_no": 1,
  "text": "Thanks, Captain Marvel!",
  "speaker": "A",
  "img_relpath": "DATA/panels_clean/2664_30_6.jpg"
}

Reproducibility Notes

Pin model versions and decoding params (XTTS-v2 variant, MusicGen decoding, Whisper small.en with temperature=0.0, beam=5, best_of=5).

Log each run (CUDA, PyTorch, LUFS, true peak, config hash).

Keep voice-only baseline to compare fairly against any mixed version.

Ethics & Licensing

Use public-domain or licensed comic pages for demos.

Do not impersonate real individuals with synthetic voices; disclose where synthetic audio is used.

XTTS-v2, MusicGen, and Whisper are third-party models with their own licenses—see the included notices and model cards.

Expected Outcomes (demo)

A voice-only WAV (demo_voice.wav) and a final mixed WAV (demo_mix.wav) with aligned SRT.

WER/CER JSON/printout for both conditions.

On our small subset, adding a steady lo-fi bed with ducking increased WER by ~+2.78 pp while keeping speech clear—your exact numbers depend on content and settings.
