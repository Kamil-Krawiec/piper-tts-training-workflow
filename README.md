# Piper Voice Trainer

A local-first UI for preparing a known-text voice dataset, recording and reviewing samples, exporting portable Piper datasets, fine-tuning Piper, and testing exported ONNX voices.

## Start the local CPU UI

```bash
git clone <repo-url>
cd piper-tts-training-workflow
docker compose up --build
```

Open <http://localhost:7860>. The first image build installs Piper's pinned training environment and can take several minutes. The normal app binds to `127.0.0.1` on the host and does not require a GPU.

The container defaults to host UID/GID `1000:1000` so files in `./data` remain editable by a typical Linux user. If `id -u` or `id -g` differs, set `PIPER_UID` and `PIPER_GID` in `.env` before starting Compose.

## Main workflow

1. Create a project, or choose the existing project after a restart.
2. Paste Polish prose, upload a `.txt` file, or choose the built-in demo pack. Select **Prose / auto split** or **One prompt per line**, set an assumed words-per-minute rate, and preview the estimated duration.
3. Edit the prompt queue before recording. Queue edits keep stable prompt IDs. Existing recordings retain their original text snapshot if a prompt is changed later.
4. Record one prompt at a time in the browser. Play it back, accept it, save it for review, or reject it. Accepted samples are normalized to mono 22,050 Hz PCM WAV without denoising or compression. Warnings remain reviewable.
5. Create a 15-minute, 30-minute, 60-minute, custom-duration, or all-accepted dataset. A fixed seed chooses the samples; held-out validation and test recordings stay the same across target sizes. `metadata.csv` documents all samples, while Piper trains from `train_metadata.csv`, which excludes those held-out recordings.
6. Export a dataset ZIP and import that ZIP in the same UI on a GPU host. The Train tab shows the fixed split summary, then lets you select fine-tuning or full training explicitly before starting the run.
7. Export a finished run to an ONNX plus JSON pair, synthesize a local test, and optionally publish the pair to the included Piper OpenAI-compatible server.

## NVIDIA GPU workflow

Install Docker Compose and NVIDIA Container Toolkit on the host, then run:

```bash
docker compose -f compose.yml -f compose.gpu.yml up --build
```

This builds the same UI with CUDA-enabled PyTorch and exposes the host GPU to the trainer container. Choose **Auto** to use CUDA when available or choose **CUDA** explicitly. An explicit CUDA request is rejected if the container cannot see CUDA; it will not silently run on CPU.

To move the recording project to the GPU host, copy the exported dataset ZIP and import it from the Dataset tab. Recordings do not need to be copied. The curated Polish checkpoint is available in the Train tab as an on-demand download; it is cached in `data/checkpoints/` after download. Checkpoints are large, so the download is never automatic.

## Training modes

- **Fine-tune existing Piper checkpoint** is the default and requires a `.ckpt` path. The curated Polish medium checkpoint is `pl_PL-darkman-medium`, cached from a pinned Piper checkpoint dataset revision.
- **Full training from scratch** omits `--ckpt_path`. It can optionally use `--model.vocoder_warmstart_ckpt`; that warm-start is recorded separately from the training mode. Small datasets display an informational warning but are not blocked.

Every run stores `run-config.json`, the actual command, status, and `train.log` in the project's persistent run directory. The command uses Piper `v1.3.0`, PyTorch `2.6.0`, and pinned training dependency constraints with Python Lightning CPU/GPU accelerator selection. Runs do not send recordings or model files to a remote service.

## Optional OpenAI-compatible Piper API

Export an ONNX pair and click **Publish to Piper API shared directory**. Then start the optional service:

```bash
docker compose --profile inference up -d
```

The pinned `kamilkrawiec/piper-openai-tts:v1.0.2` container mounts `./data/piper-voices` at `/data` and listens on <http://localhost:5000>. If host port 5000 is already in use, set `PIPER_API_HOST_PORT=5001` in `.env`; the container still listens on port 5000. Test with:

```bash
curl -o output.wav \
  -H 'Content-Type: application/json' \
  -d '{"model":"piper","voice":"pl_PL-kamil-medium","input":"To jest test mojego własnego modelu głosu.","response_format":"wav"}' \
  http://localhost:5000/v1/audio/speech
```

The service may download a requested official voice on first use. A published custom voice is loaded from the shared `/data` folder.

## Persistent data

All generated content lives under `./data` on the host:

```text
data/
  trainer.sqlite3
  projects/<project-id>/
    source/original.txt
    prompts/parsed.jsonl
    prompts/current.jsonl
    recordings/raw/
    recordings/normalized/
    datasets/<dataset-id>/
    runs/<run-id>/
    models/<voice-name>/
  checkpoints/
  exports/
  piper-voices/
```

Back up this directory to retain projects, recordings, bundles, logs, checkpoints, and models. Deleting a project or the `data` directory removes its stored voice data. Uploaded text, audio, checkpoint, dataset, and model files stay local by default.

## Prompt handling and sample checks

Sentence splitting is deterministic and handles common Polish abbreviations and decimal/version numbers without paraphrasing. Line mode makes every non-empty line one prompt. The prompt text is the dataset label; no ASR is used. Prompt length and audio quality checks are recommendations for review, not automatic deletion rules.

The recording UI estimates duration from word count and the selected speaking rate. It is not a promise about accepted dataset duration. Dataset target selection uses actual normalized audio lengths.

## Limitations

- This is a single-speaker v1 workflow at Piper medium configuration and 22,050 Hz. Piper applies its seeded internal validation split to the training-only metadata; the fixed validation and test samples remain available for external listening and model-to-model comparison.
- Browser recording requires microphone permission and a secure browser context (localhost is treated as secure by modern browsers).
- Training quality depends on recording conditions, transcription fidelity, dataset size, and checkpoint compatibility. Training time is hardware-dependent and is not estimated here.
- The manager allows one active training job per project. A completed run can be exported from its saved checkpoint. If the container stops during training, status is recovered from the persisted process record and logs; interrupted jobs may need to be restarted from a suitable checkpoint.
- The UI is designed for local use. Do not expose port 7860 publicly without adding authentication and a protected reverse proxy.

## Development checks

Core logic tests use Python's standard library:

```bash
python3 -m unittest discover -s tests -v
```

Docker configuration can be inspected with:

```bash
docker compose config
docker compose -f compose.yml -f compose.gpu.yml config
docker compose --profile inference config
```

The app's Diagnostics tab reports ffmpeg, eSpeak NG, Piper training, CUDA, and optional API availability.

## Licensing

This repository is MIT-licensed. The Piper training package is GPL-3.0 and is installed as a separately pinned upstream dependency inside the image; the optional API image is MIT-licensed. The curated upstream checkpoint repository declares MIT licensing. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and model cards for sources and terms.
