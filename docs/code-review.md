# Code review: workflow and presentation

Reviewed the browser callbacks, project/recording persistence, dataset and ZIP validation, training supervision, telemetry, checkpoint selection, export, and synthesis. The existing domain modules are useful boundaries. The main readability problem was combining the entire page layout and event wiring with callbacks in `main.py`.

## Findings addressed

| Priority | Finding | Change |
| --- | --- | --- |
| Medium | Switching pages changed the document height and navigation forced a smooth scroll. | Use one viewport frame with scrollable page content and a fixed footer; remove forced navigation scrolling. |
| Medium | Training setup, monitoring, hardware readings and diagnostics competed for attention in one long page. | Separate Setup and Metrics views in Step 5; readable metric cards, loss history and expandable diagnostics. |
| Medium | The original recording was hidden inside the generated A/B comparison; unrelated text could remain beside the reference. | Step 6 pairs original speaker and checkpoint. Loading a reference updates the shared text; editing text clears the reference and generated samples. |
| Medium | Completed epochs relied on validation CSV rows. Datasets without validation could show no completed epochs despite saved batch timing. | Job status also derives completed epochs from saved completed batches and batches per epoch. |
| Low | Missing GPU/memory measurements appeared as zero; sub-minute epochs rounded to zero minutes. | Missing values are explicit, seconds remain readable, and finished runs with no timing say “Not recorded.” |
| Low | Layout and callback logic shared a 1,476-line entrypoint, while CSS repeated theme values. | Move layout/event wiring to `pages.py`, metric presentation to `ui.py`, and theme tokens to `ui.css`. Keep domain logic and saved formats in their existing modules. |

## Remaining lifecycle issue

**Medium — process recovery can overstate success.** In `TrainingJobs.status`, after the UI process has restarted, an exited child has no recoverable exit code. Any checkpoint found under the run directory is treated as evidence of completion, including probe checkpoints. A failed run with a saved checkpoint can therefore be labelled completed. This remains existing behaviour; a follow-up should persist the supervisor's terminal exit status and distinguish partial checkpoints from successful completion. The UI continues to expose saved logs for diagnosis.

## Where to make changes

| Module | Owns |
| --- | --- |
| `app/main.py` | Browser callbacks and application startup |
| `app/pages.py` | Six workflow pages and event wiring; receives the callbacks explicitly |
| `app/ui.py`, `app/ui.css` | Metric presentation, headings, theme and responsive layout |
| `app/workflow.py` | Page prerequisites and resume navigation |
| `app/projects.py` | SQLite projects, prompts, recording decisions and dataset registration |
| `app/text.py`, `app/audio.py`, `app/recording_review.py` | Prompt parsing, audio preparation/quality and recording review facts |
| `app/datasets.py`, `app/bundles.py` | Reproducible datasets, saved splits and validated ZIP transfer |
| `app/training.py`, `app/train_run.py` | Job lifecycle, training configuration, automatic settings and hardware sampling |
| `app/train_cli.py`, `app/train_data.py`, `app/training_metrics.py` | Piper integration, saved split loaders, learning rates, checkpoint roles and batch timings |
| `app/performance.py` | Hardware policies, measurements and runtime estimates |
| `app/checkpoints.py`, `app/export.py`, `app/inference.py` | Base checkpoint cache, stable exports/downloads and speech generation |

## Validation

Run `python -m unittest discover -s tests -v` in the trainer image for the complete suite. Host-only runs skip tests requiring Piper/Lightning.

Browser verification uses an isolated trainer container with copied dataset/metrics and a read-only saved checkpoint. Check desktop and 390 × 844 layouts in light/dark themes, restored loss plots, matching reference text, checkpoint synthesis, voice ZIP download and stale-audio clearing. No change to the running Compose service is required.

Check page stability with `agent-browser eval --stdin < scripts/check-ui-layout.js` after selecting a saved dataset that unlocks all six pages. Run at desktop and mobile viewport sizes; the check rejects page/footer movement and viewport overflow.
