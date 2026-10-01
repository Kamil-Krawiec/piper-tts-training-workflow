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

1. **Project:** Choose a saved project by name or create one. The app creates its ID for you.
2. **Text:** Paste prose, upload a `.txt` file, or load a built-in prompt pack. Preview the prompts and estimated duration. You can edit prompt text before recording; existing takes retain their original text snapshot.
3. **Record:** Read one displayed prompt, record it with the microphone, listen, and accept it or save it for review. Accepted samples are normalized to mono 22,050 Hz PCM WAV without denoising or compression. Choose a saved recording to listen or change its status.
4. **Dataset:** Build a 15-minute, 30-minute, 60-minute, custom-duration, or all-accepted dataset, or import a dataset ZIP. Dataset creation trims long leading and trailing silence from training copies, keeps about 0.25 seconds at each end of speech, and leaves the saved recordings unchanged. Duration targets use the trimmed copies. Export a ZIP here to move to a GPU host. Fixed validation and test recordings stay the same across target sizes. Piper trains from `train_metadata.csv`, which excludes those held-out recordings.
5. **Train:** Choose fine-tuning or full training, supply a checkpoint if fine-tuning, choose your device, review the run summary, and start. The saved epoch bar and loss chart refresh while the page is open. Recent logs remain available under the chart.
6. **Voice:** Export a saved checkpoint as an ONNX plus JSON pair, download the voice ZIP, and generate a local listening test. Listening comparisons and publishing to the optional Piper API are available in expandable sections.

Each step reads from top to bottom and ends with Back/Continue navigation. Later steps unlock when their prerequisites are saved. **Resume saved progress** in Step 1 jumps to the next stage for an existing project. **Import a dataset ZIP** goes directly to Step 4 for the two-machine workflow. Switching projects clears temporary outputs from the previous workspace.

## Docker Hub images

The publishing workflow builds Linux x86-64 (`linux/amd64`) images for the existing public repository `kamilkrawiec/piper-tts-training-workflow`:

After the first successful publish:

```bash
docker pull kamilkrawiec/piper-tts-training-workflow:cpu
docker pull kamilkrawiec/piper-tts-training-workflow:cuda
```

The CUDA image requires an NVIDIA-compatible host and container runtime for GPU training. Use the CUDA image for RunPod; pin a release such as `v0.1.0-cuda` after that release is published. Both variants use the same Dockerfile and application code. Local Compose commands keep their existing `build:` support and do not require prebuilt Docker Hub images.

### Publishing images (maintainers)

In GitHub repository **Settings → Secrets and variables → Actions → New repository secret**, add:

- `DOCKERHUB_USERNAME`: the Docker Hub account with push access to `kamilkrawiec/piper-tts-training-workflow`.
- `DOCKERHUB_TOKEN`: a Docker Hub access token with write permission, not the account password.

After the workflow is on the default branch, open **Actions → Publish Docker images → Run workflow** to publish the moving `cpu` and `cuda` tags. Pushing a version tag such as `v0.1.0` also publishes `v0.1.0-cpu` and `v0.1.0-cuda` and updates both moving tags. Manually running on a version tag produces the same release tags. Ordinary branch commits do not publish images; no `latest` tag is published.

The workflow builds each variant for `linux/amd64`, checks its PyTorch CUDA build and application/trainer imports without requiring a physical GPU, and runs unit tests before pushing. Authentication, build, or verification failures stop that variant's publish step. Actual GPU availability is checked later on the NVIDIA host.

## NVIDIA GPU workflow

Install Docker Compose and NVIDIA Container Toolkit on the host, then run:

```bash
docker compose -f compose.yml -f compose.gpu.yml up --build
```

This builds the same UI with CUDA-enabled PyTorch and exposes the host GPU to the trainer container. Choose **Auto** to use CUDA when available or choose **CUDA** explicitly. An explicit CUDA request is rejected if the container cannot see CUDA; it will not silently run on CPU.

To move the recording project to the GPU host, copy the exported dataset ZIP and import it from the Dataset tab. Recordings do not need to be copied. The curated Polish checkpoint is available in the Train tab as an on-demand download; it is cached in `data/checkpoints/` after download. Checkpoints are large, so the download is never automatic.

## Training modes

- **Fine-tune existing Piper checkpoint** is the default and requires a `.ckpt` path. The curated Polish medium checkpoint is `pl_PL-darkman-medium`, cached from a pinned Piper checkpoint dataset revision.
- **Full training from scratch** omits `--ckpt_path`. It can optionally use `--model.vocoder_warmstart_ckpt`; that warm-start is recorded separately from the training mode. Around one hour of speech may be insufficient for a high-quality voice, but the experiment is not blocked.

The recommended starting caps are **1000 epochs for fine-tuning** and **2000 for training from scratch**. Fine-tuning also offers 250 and 500 epoch experiments; scratch training offers a 500 epoch experiment. These are starting points, not universal quality optima. Automatic early stopping is disabled. Compare the same held-out listening sentences from interval checkpoints before deciding which voice sounds best.

## Automatic training settings and estimates

Batch size defaults to **Auto**. On CUDA, a separate short Piper training process tries progressively larger batches using real dataset samples and forward/backward work. Each trial records its outcome and peak VRAM. The chosen batch must leave configurable memory headroom; an out-of-memory trial cannot damage the subsequent training process. The run stays on one GPU. On CPU, Auto uses available RAM, the longest utterance, and dataset size to choose a conservative batch of 4, 8, or 16. You can manually choose a batch size in Advanced settings.

DataLoader workers and PyTorch CPU threads also default to **Auto**. The worker heuristic reserves CPU capacity for the trainer and operating system; CPU training gets fewer workers because its model computation already uses CPU cores. Actual resolved worker, thread, and batch settings are saved in `run-config.json`, along with the command, batch probe results, hardware information, and the effective learning-rate schedule. Advanced settings allow manual overrides. Docker Desktop may impose CPU and RAM VM limits; Compose does not set an artificial CPU limit.

The training panel shows current epoch, global optimizer step, estimated total optimizer steps, batches per epoch, throughput, elapsed time, and an **estimated** finish time. ETA appears after five warm-up batches and at least ten measured batches, then uses recent batch times. Validation, checkpoint writing, and changing input lengths can shift the finish time. An optional GPU hourly rate estimates compute cost only; storage and provider fees are excluded.

Each run saves CPU/RAM and GPU samples every five seconds in `hardware-metrics.jsonl`. GPU compute utilization, GPU memory utilization, and allocated VRAM are separate measurements. Low VRAM allocation alone is not a performance fault; training throughput is the goal. The diagnostics section reports the trainer's PyTorch/CUDA versions, visible GPU, available CPU cores, RAM, and automatic defaults. Possible bottlenecks are hints, not certain diagnoses.

The pinned Piper revision has fixed per-epoch generator/discriminator learning-rate decays of `0.999875`/`0.9999` by default. Changing `max_epochs` changes the **final learning-rate ratio**, but does not recalculate the per-epoch decay. A checkpoint can carry different initial rates or decays; each run records the effective model values after loading it. Therefore a 250-epoch run is not equivalent to the final result of a 1000-epoch run. For controlled 30-minute versus 60-minute comparisons, use the same base checkpoint, model configuration, seed, epoch cap, learning-rate policy, and listening sentences. The 60-minute dataset naturally has more batches and optimizer steps per epoch.

Training keeps a rolling latest checkpoint every 25 epochs, interval checkpoints every 250 epochs, and a final checkpoint. The Voice step can export any saved checkpoint for listening comparison. Avoiding a checkpoint every epoch limits disk use and pause time.

Every run stores `run-config.json`, the actual command, status, `train.log`, and CSV loss metrics in the project's persistent run directory. Training runs as a separate process, so closing the browser or losing its connection does not stop it. Reopen the project to see saved progress. Stopping the trainer container interrupts the process. The command uses Piper `v1.3.0`, PyTorch `2.6.0`, and pinned training dependency constraints with Python Lightning CPU/GPU accelerator selection. Runs do not send recordings or model files to a remote service.

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

The recording UI estimates duration from word count at 140 words per minute. It is not a promise about accepted dataset duration. Dataset target selection uses actual normalized audio lengths.

## Limitations

- This is a single-speaker v1 workflow at Piper medium configuration and 22,050 Hz. Piper applies its seeded internal validation split to the training-only metadata; the fixed validation and test samples remain available for external listening and model-to-model comparison.
- Browser recording requires microphone permission and a secure browser context (localhost is treated as secure by modern browsers).
- Training quality depends on recording conditions, transcription fidelity, dataset size, and checkpoint compatibility. The measured training ETA is approximate and becomes available only after warm-up batches.
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

UI callback tests also run when Gradio is installed (they are skipped by the core-only command). Use the app's Python environment to include them:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Licensing

This repository is MIT-licensed. The Piper training package is GPL-3.0 and is installed as a separately pinned upstream dependency inside the image; the optional API image is MIT-licensed. The curated upstream checkpoint repository declares MIT licensing. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and model cards for sources and terms.
