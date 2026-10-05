# Piper Voice Trainer

Record your voice, prepare a speech dataset, train a Piper voice, and export it for text-to-speech—all from a browser UI. Projects, recordings, training progress, and exported voices are saved locally. The training image uses Piper v1.8.0, pinned to an exact upstream commit.

You can record and review samples on a CPU machine, then transfer a dataset ZIP to an NVIDIA GPU machine for training. The app supports fine-tuning an existing Piper checkpoint or training a single-speaker voice from scratch. Exported voices contain an ONNX model and its JSON configuration, ready for Piper inference.

## Choose a Docker image

Both images contain the same web application, Piper training tools, and voice export tools. They differ in the installed PyTorch build:

| Image | Use it for | Requirements |
| --- | --- | --- |
| `kamilkrawiec/piper-tts-training-workflow:cpu` | Recording, dataset preparation, CPU training, export, and listening tests | Docker; no GPU required |
| `kamilkrawiec/piper-tts-training-workflow:cuda` | The same workflow with NVIDIA GPU training | Docker, a compatible NVIDIA GPU and driver, and NVIDIA Container Toolkit |

The images target Linux x86-64 (`linux/amd64`). Use the CUDA image on an NVIDIA GPU host, including RunPod. CPU training works but can take considerably longer.

The `cpu` and `cuda` tags track the current published images. For repeatable runs, use an available versioned tag such as `<version>-cpu` or `<version>-cuda` from [Docker Hub](https://hub.docker.com/r/kamilkrawiec/piper-tts-training-workflow/tags). There is no `latest` tag.

## Run a prebuilt image

You do not need to clone this repository or build an image. `docker pull` downloads it; `docker run` starts the application. The following commands are for a Linux/macOS shell. GPU access requires an NVIDIA-compatible host/runtime.

### CPU

```bash
docker pull kamilkrawiec/piper-tts-training-workflow:cpu
mkdir -p data
docker run -d --name piper-trainer \
  --user "$(id -u):$(id -g)" \
  --shm-size=2g \
  -p 127.0.0.1:7860:7860 \
  -v "$PWD/data:/data" \
  kamilkrawiec/piper-tts-training-workflow:cpu
```

### NVIDIA GPU

Install the NVIDIA driver and NVIDIA Container Toolkit on your GPU host, then run:

```bash
docker pull kamilkrawiec/piper-tts-training-workflow:cuda
mkdir -p data
docker run -d --name piper-trainer \
  --user "$(id -u):$(id -g)" \
  --gpus all \
  --shm-size=2g \
  -p 127.0.0.1:7860:7860 \
  -v "$PWD/data:/data" \
  kamilkrawiec/piper-tts-training-workflow:cuda
```

Choose one of these commands; both use the same container name and port. Open <http://localhost:7860> and allow microphone access when recording. On the GPU host, choose **Auto** or **CUDA** in the Train step. Explicit CUDA selection reports an error if the container cannot access the GPU.

The mounted `data` folder keeps your work when the container is removed. The user option keeps files owned by your host account. Shared memory is set to 2 GiB so DataLoader workers have room to exchange training batches; this is not a limit on total container RAM.

To inspect or restart the container:

```bash
docker logs -f piper-trainer
docker stop piper-trainer
docker start piper-trainer
```

Closing the browser does not stop training. Stopping the container interrupts it. To switch images, stop and remove the existing container with `docker rm piper-trainer`, then run the other image with the same data folder.

## Build and run from this repository

Use Compose if you want to build the image locally or work with the source code:

```bash
git clone https://github.com/Kamil-Krawiec/piper-tts-training-workflow.git
cd piper-tts-training-workflow
docker compose up --build
```

For an NVIDIA GPU build:

```bash
docker compose -f compose.yml -f compose.gpu.yml up --build
```

Open <http://localhost:7860>. The first build installs the training environment and can take several minutes. Both commands save your work in `./data` and bind the UI to localhost.

Compose defaults to user/group `1000:1000`. If your Linux account uses different IDs, copy `.env.example` to `.env` and set `PIPER_UID` and `PIPER_GID` to the output of `id -u` and `id -g`. You can also set `PIPER_HOST_DATA_DIR` to store data elsewhere.

## Main workflow

1. **Project:** Choose a saved project by name or create one. The app creates its ID for you.
2. **Text:** Paste prose, upload a `.txt` file, or load a built-in prompt pack. Preview the prompts and estimated duration. You can edit prompt text before recording; existing takes retain their original text snapshot.
3. **Record:** Read one displayed prompt, record it with the microphone, listen, and accept it or save it for review. Accepted samples are normalized to mono 22,050 Hz PCM WAV without denoising or compression. Switch to **Review recordings** to browse flagged takes or search all recordings, including imported dataset clips. Listen, then choose **Keep**, **Re-record**, or **Reject**. Expand **Comparison and review details** to compare a saved dataset copy with the original and see how much edge audio was removed. A large cut is a reason to listen, not proof of damaged speech. Imported bundles do not contain untrimmed originals. Build a new dataset to apply review decisions; existing datasets remain unchanged and saved split membership is preserved for replacements.
4. **Dataset:** Build a 15-minute, 30-minute, 60-minute, custom-duration, or all-accepted dataset, or import a dataset ZIP. Dataset creation trims long leading and trailing silence from training copies, keeps about 0.25 seconds at each end of speech, and leaves the saved recordings unchanged. Duration targets use the trimmed copies. Export a ZIP here to move to a GPU host. Fixed validation and test recordings stay the same across target sizes. Training uses the saved split membership directly: training recordings update the model, validation recordings provide validation loss, and test recordings remain held out. No second random split is applied. Existing exported dataset ZIPs remain compatible.
5. **Train:** Use **Setup** to choose a dataset, training mode, checkpoint and epoch cap, then review and start. **Metrics** shows saved epoch progress, training/validation loss, elapsed time, measured ETA, throughput and CPU/GPU usage. Timing, hardware details and logs stay in expandable panels. Metrics refresh every 10 seconds.
6. **Voice:** Select a run and checkpoint. Choose a held-out **Reference sentence** and click **Load reference & matching text** to compare the original recording with generated checkpoint speech on the same words. Click **Generate checkpoint sample** to export and hear the checkpoint, then download its ONNX plus JSON ZIP. If no reference exists, enter your own text. The two players sit side by side on desktop and stack on mobile; generated A/B comparison, training downloads and API publishing remain in expandable panels.

Each step reads from top to bottom and ends with Back/Continue navigation. Later steps unlock when their prerequisites are saved. **Resume saved progress** in Step 1 jumps to the next stage for an existing project. **Import a dataset ZIP** goes directly to Step 4 for the two-machine workflow. Switching projects clears temporary outputs from the previous workspace.

## Record locally, train on another machine

1. Record and accept your samples in the local UI.
2. Build a dataset in the Dataset step and download its ZIP.
3. Start the CUDA image on the GPU machine and import the ZIP in the Dataset step.
4. Select a checkpoint and training settings, then start training. The project summary includes imported recordings and audio duration; importing the same recordings again does not count them twice.
5. Generate a sample from a saved checkpoint in the Voice step and download its voice ZIP.

You only need the dataset ZIP for this transfer; the original recording project can stay on your recording machine. Microphone recording on a remote host requires a secure browser connection. Keep the UI private; the commands above expose it only on the host's localhost interface.

## Training and progress

**Fine-tuning** starts from an existing `.ckpt` checkpoint. The Train step offers an on-demand download of the curated Polish `pl_PL-darkman-medium` checkpoint, or you can supply your own compatible checkpoint. Downloads are cached in `data/checkpoints/`.

**Training from scratch** starts a new model and can optionally use a vocoder warm-start checkpoint. It usually needs more data and training time. A small dataset does not guarantee a good voice in either mode.

Start with **Auto** batch size, DataLoader workers, and CPU threads. CUDA batch selection tests the dataset against available GPU memory; CPU settings use available memory and cores. Advanced settings let you override them.

The Train step shows epoch progress, loss curves, throughput, hardware use, and an estimated finish time. Loss curves appear after logged batches; runtime estimates need enough measured batches and can change during training. Reopening the UI restores saved progress.

In **Advanced · device, performance and learning rates**, set **Save checkpoint every X epochs** (default: 250). Each milestone is retained, so short intervals require more disk space. Training also saves a rolling checkpoint every 25 epochs, the checkpoint with the lowest validation mel reconstruction loss (`val_mel`), and a final checkpoint. Use the Voice step to compare checkpoints on the same listening sentences. Validation loss is a guide; choose your voice by listening.

Piper's individual training and validation losses are saved in the metrics CSV. Optional MOS scoring is disabled so training does not need to download another scoring model. The run diagnostics record the installed Piper version and revision.

The generator and discriminator learning rates are configurable (defaults: `0.0002` and `0.0001`). Fine-tuning preserves the rates you select instead of inheriting them from the base checkpoint. Both rates decay once per completed epoch and the rates actually used appear as `lr_g` and `lr_d` in the run's metrics CSV. Current optimizer rates are available in training diagnostics. Lower rates are an experiment, not a guarantee of better voice quality.

## Use your exported voice

The Voice step exports the selected checkpoint automatically when generating a listening sample, then packages the ONNX model and matching JSON configuration in a downloadable ZIP. Keep both files together when loading the voice into Piper. Checkpoint choices refresh during training and your selection stays selected.

Open **Speech settings · variation and speed** to adjust `noise_scale` (voice variation), `noise_w` (phoneme timing variation), and `length_scale` (speech speed: above 1 is slower, below 1 is faster). Blank fields preserve the model's defaults. Overrides apply to checkpoint previews and both generated A/B voices; the original recording is unchanged. The download keeps the model's original JSON defaults, so pass the same overrides to Piper when using the exported voice elsewhere.

For a complete run archive, open **Download all training artifacts** after training finishes. It contains saved checkpoints, CSV metrics, hardware telemetry, logs and configuration. Keep the dataset ZIP from Step 4 separately. If an artifact changes during packaging, retry after training stops.

To test a base or imported checkpoint, open **Use a checkpoint file instead · e.g. Polish Darkman** in Voice. Enter its container path, such as `/data/checkpoints/pl_PL-darkman-medium/checkpoint.ckpt`, and click **Generate from checkpoint file**. The checkpoint must already exist inside the data directory. A project is required; a training run or dataset is not. The app uses `voice.onnx.json` or `config.json` beside the checkpoint, or the matching JSON path you provide. For Darkman, its pinned config is fetched automatically if absent. For a checkpoint copied from your training run, provide that run's `voice.onnx.json`. The generated voice appears in the same player, ZIP download and saved-model comparison choices.

In **Compare two generated voices · models or checkpoints**, choose **Voice A** and **Voice B**, then click **Generate A and B**. Each selector lists saved ONNX voice models and epoch checkpoints from all runs in the project. You can compare two models, two checkpoints, or one of each using the listening text above. The comparison selectors are independent of the single-checkpoint preview. Changing either selection or the text clears the previous comparison.

To preserve training weights, open **Download checkpoint for further training** and prepare the selected `.ckpt` download. This makes a stable copy, including when you choose the rolling checkpoint. Keep the dataset ZIP as well. An ONNX voice is for speech generation and cannot be converted back into a full training checkpoint. Upload the `.ckpt` in Step 5 to start a new fine-tuning session from its weights; this does not resume the old epoch or optimizer state.

### Optional text-to-speech API

The trainer is the browser app for recording and training. The separate `kamilkrawiec/piper-openai-tts` image serves exported voices through an OpenAI-compatible speech API.

When running from the repository, export a voice and open **Advanced: filenames and API publishing** and click **Publish selected voice to Piper API**, then start the API:

```bash
docker compose --profile inference up -d piper-api
```

It reads voices from `./data/piper-voices` and listens on <http://localhost:5000>. Set `PIPER_API_HOST_PORT` in `.env` if you need a different host port. Replace the voice name below with your exported voice name:

```bash
curl -o output.wav \
  -H 'Content-Type: application/json' \
  -d '{"model":"piper","voice":"pl_PL-kamil-medium","input":"To jest test mojego własnego modelu głosu.","response_format":"wav"}' \
  http://localhost:5000/v1/audio/speech
```

The API may download a requested official voice on first use. The trainer images do not start this API automatically.

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

## Code organization

All six pages share a fixed viewport frame and the same width, independent of their content. Longer content scrolls inside the page while the step tabs and Back/Next buttons stay in place.

The six workflow pages and their event wiring live in `app/pages.py`. Browser callbacks stay in `app/main.py`; metric presentation and theme styling live in `app/ui.py` and `app/ui.css`. Domain modules retain ownership of projects, datasets, training and export. See the [code review and module map](docs/code-review.md) for findings and the relevant files.

## Licensing

This repository is MIT-licensed. The Piper training package is GPL-3.0 and is installed as a separately pinned upstream dependency inside the image; the optional API image is MIT-licensed. The curated upstream checkpoint repository declares MIT licensing. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and model cards for sources and terms.
