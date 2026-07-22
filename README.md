# DeepSpeech-Lite — End-to-End Speech Recognition from Scratch

A from-scratch, PyTorch implementation of a **CNN + Bidirectional GRU + CTC** speech-to-text
model, trained on LibriSpeech. Includes a full training pipeline, greedy CTC decoding,
WER/CER evaluation, a Gradio web demo for live microphone transcription, and Docker
deployment. built to be portfolio-ready, not just a notebook experiment.

## Why this project

Most speech-to-text portfolio projects just call the OpenAI Whisper API. This one
implements the actual acoustic model, the CTC loss/decoding logic, the audio feature
pipeline, and the training loop from first principles, the same architecture family
(DeepSpeech2 / early wav2letter) that predates and underpins modern ASR systems.

## Architecture

```
Raw waveform (16kHz)
      │
      ▼
Log-Mel Spectrogram (80 mel bins)
      │
      ▼
┌─────────────────────────┐
│  2D CNN feature extractor │  (2 conv layers, batchnorm, ReLU, freq downsampling)
└─────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│  5-layer Bidirectional   │
│  GRU (512 hidden units)  │
└─────────────────────────┘
      │
      ▼
   Fully Connected
      │
      ▼
  CTC log-softmax over
  character vocabulary
      │
      ▼
 Greedy / Beam CTC Decoder
      │
      ▼
   Transcribed text
```

## Project Structure

```
asr-project/
├── README.md
├── requirements.txt
├── Dockerfile
├── .gitignore
├── LICENSE
├── configs/
│   └── config.yaml          # all hyperparameters in one place
├── src/
│   ├── __init__.py
│   ├── model.py              # DeepSpeech2-style model definition
│   ├── dataset.py            # LibriSpeech Dataset + feature extraction + tokenizer
│   ├── decoder.py            # Greedy CTC decoder + WER/CER metrics
│   ├── train.py              # training loop, checkpointing, logging
│   ├── inference.py          # transcribe a single .wav file
│   └── utils.py              # seeding, AverageMeter, logging helpers
├── app.py                    # Gradio web demo (mic or file upload -> transcript)
├── tests/
│   └── test_model.py         # shape/sanity unit tests
├── checkpoints/               # trained model weights land here
├── data/                      # LibriSpeech downloads land here
└── samples/                   # example audio for quick testing
```

## Setup

```bash
git clone <your-repo-url>
cd asr-project
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python -m src.train --config configs/config.yaml
```

This downloads `train-clean-100` / `dev-clean` from LibriSpeech via `torchaudio`
on first run, extracts log-mel features on the fly, and trains with CTC loss.
Checkpoints are saved to `checkpoints/`. Training curves (loss, WER) are logged
per epoch to stdout and `checkpoints/history.json`.

**Realistic expectation:** on a single GPU, `train-clean-100` (100 hrs) gets you to
roughly 20-30% WER after ~30-40 epochs with this architecture — enough to
demonstrate a working system and a real training curve, not a SOTA model. Mention
this honestly in interviews; it's the difference between "I understand ASR" and
"I copied a Whisper call."

## Run inference on a single file

```bash
python -m src.inference --checkpoint checkpoints/best.pt --audio samples/example.wav
```

## Launch the web demo

```bash
python app.py
```

Opens a Gradio UI at `http://localhost:7860` where you can upload audio or record
from your mic and see the live transcript plus a confidence/timing breakdown.

## Run tests

```bash
pytest tests/ -v
```

## Docker

```bash
docker build -t asr-demo .
docker run -p 7860:7860 asr-demo
```

## Evaluation

`src/decoder.py` implements Word Error Rate (WER) and Character Error Rate (CER)
via Levenshtein distance, matching the standard ASR evaluation convention.
Run:

```bash
python -m src.train --config configs/config.yaml --eval-only --checkpoint checkpoints/best.pt
```

## License

MIT — see `LICENSE`.

## Author

## Michael Iseoluwa Adedayo.