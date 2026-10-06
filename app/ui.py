"""Presentation for the voice workflow; no project or job state."""

from datetime import datetime, timezone
from html import escape
from pathlib import Path

import gradio as gr
from app.workflow import STEP_COUNT


APP_CSS = Path(__file__).with_name("ui.css").read_text(encoding="utf-8")


def step_heading(number: int, title: str, description: str):
    gr.HTML(
        f'<div class="step-count">Step {number} of {STEP_COUNT}</div>'
        f"<h2>{title}</h2><p>{description}</p>",
        elem_classes="step-heading",
    )


def _duration(seconds, pending="Measuring"):
    if seconds is None:
        return pending
    seconds = max(0, round(seconds))
    if seconds < 60:
        return f"{seconds} sec"
    minutes = round(seconds / 60)
    return f"{minutes // 60} h {minutes % 60} min" if minutes >= 60 else f"{minutes} min"


def _value(value, suffix="", precision=None):
    if value is None:
        return "—"
    if precision is not None:
        value = f"{value:.{precision}f}"
    return escape(str(value)) + suffix


def _metric(label, value, detail=""):
    return (f'<div class="metric-card"><span>{escape(label)}</span><strong>{value}</strong>'
            f'<small>{escape(detail)}</small></div>')


def _table(title, rows):
    return (f'<section class="metric-table"><h4>{escape(title)}</h4><dl>'
            + "".join(f"<div><dt>{escape(label)}</dt><dd>{value}</dd></div>" for label, value in rows)
            + "</dl></section>")


def training_summary(state):
    """Render saved measurements, keeping unavailable values distinct from zero."""
    progress = state["progress"]
    config = state.get("config") or {}
    performance = state.get("performance") or {}
    hardware = state.get("hardware") or {}
    cpu, gpu = hardware.get("cpu") or {}, hardware.get("gpu") or {}
    estimate = performance.get("estimate") or {}
    maximum = progress["max_epochs"]
    completed = min(progress["completed_epochs"], maximum) if maximum else 0
    percentage = round(100 * completed / maximum) if maximum else 0
    resolving = config.get("resolution_status") == "pending"
    phase = "preparing" if state["status"] == "training" and resolving else state["status"]
    current = progress["current_epoch"]
    pending = "Measuring" if state["status"] in {"queued", "preparing", "training"} else "Not recorded"
    epoch = f"Epoch {current} of {maximum}" if current else "Waiting for first epoch"
    summary = (
        f'<div class="train-summary"><div class="train-summary-head"><strong>{escape(phase.replace("_", " ").title())}</strong>'
        f'<span>Run {escape(state["run_id"][:8])}</span></div><p>{epoch} · {percentage}% of epochs completed</p>'
        f'<div class="train-progress" role="progressbar" aria-label="Completed training epochs" '
        f'aria-valuenow="{completed}" aria-valuemin="0" aria-valuemax="{maximum or 1}">'
        f'<span style="width:{percentage}%"></span></div></div>'
    )
    summary += '<div class="metric-grid">' + "".join([
        _metric("Elapsed", _duration(performance.get("elapsed_seconds"), pending), "Saved training time"),
        _metric("Estimated remaining", _duration(estimate.get("remaining_seconds"), pending), "Measured after warm-up; estimate may change"),
        _metric("Average epoch", _duration(performance.get("average_epoch_seconds"), pending), "Recent completed epochs"),
        _metric("Throughput", _value(performance.get("samples_per_second"), " samples/s", 2), "Includes data loading"),
    ]) + "</div>"
    latest = {}
    for loss in progress["losses"]:
        latest[loss["series"]] = loss["loss"]
    summary += '<div class="metric-grid">' + "".join([
        _metric("Training loss", _value(latest.get("Training"), precision=4), "Last logged generator loss"),
        _metric("Validation loss", _value(latest.get("Validation"), precision=4), "Last logged held-out validation loss"),
        _metric("CPU usage", _value(cpu.get("total_percent"), "%"), "Across available logical cores"),
        _metric("GPU compute", _value(gpu.get("compute_percent"), "%"), str(gpu.get("name") or "No GPU measurement")),
    ]) + "</div>"
    finish = (datetime.fromtimestamp(estimate["estimated_finish_at"], timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
              if estimate.get("estimated_finish_at") else pending)
    gib = 1024 ** 3
    def memory(used, total):
        return (f"{used / gib:.1f} / {total / gib:.1f} GiB"
                if used is not None and total is not None else "—")
    steps = performance.get("steps_per_second")
    summary += '<details class="metric-details"><summary>Timing, hardware and run settings</summary><div class="metric-columns">'
    summary += _table("Timing & cost", [
        ("Estimated finish", finish),
        ("Global optimizer step", f'{_value(performance.get("global_step", progress.get("global_step")))} / {_value(config.get("total_optimizer_steps", progress.get("total_optimizer_steps")))}'),
        ("Batches per epoch", _value(config.get("steps_per_epoch", progress.get("steps_per_epoch")))),
        ("Batches per second", _value(steps, precision=2)),
        ("Seconds per batch", _value(1 / steps if steps else None, precision=2)),
        ("Remaining compute cost", _value(estimate.get("remaining_cost"), " USD", 2)),
        ("Total compute cost", _value(estimate.get("total_cost"), " USD", 2)),
    ])
    summary += _table("Hardware", [
        ("Device", _value(config.get("device"))),
        ("RAM", memory(cpu.get("ram_used_bytes"), cpu.get("ram_total_bytes"))),
        ("Process CPU", _value(cpu.get("process_percent"), "%")),
        ("Logical cores", _value(cpu.get("logical_cores"))),
        ("Per-core CPU (%)", _value(cpu.get("per_core_percent"))),
        ("VRAM", memory(gpu.get("memory_used_bytes"), gpu.get("memory_total_bytes"))),
        ("GPU memory usage", _value(gpu.get("memory_percent"), "%")),
        ("GPU power / limit", f'{_value(gpu.get("power_w"))} / {_value(gpu.get("power_limit_w"))} W'),
        ("GPU temperature", _value(gpu.get("temperature_c"), " °C")),
        ("SM / memory clocks", f'{_value(gpu.get("sm_clock_mhz"))} / {_value(gpu.get("memory_clock_mhz"))} MHz'),
    ])
    schedule = config.get("effective_lr_schedule") or {}
    summary += _table("Run settings", [
        ("Batch size", "Resolving" if resolving else _value(config.get("batch_size"))),
        ("DataLoader workers", "Resolving" if resolving else _value(config.get("num_workers"))),
        ("PyTorch threads", _value(config.get("torch_threads"))),
        ("Current generator / discriminator LR", _value(performance.get("learning_rates"))),
        ("Initial generator / discriminator LR", f'{_value(schedule.get("generator_initial_lr"))} / {_value(schedule.get("discriminator_initial_lr"))}'),
        ("Per-epoch decay", f'{_value(schedule.get("generator_per_epoch_decay"))} / {_value(schedule.get("discriminator_per_epoch_decay"))}'),
        ("Planned final LR ratio", f'{_value(schedule.get("generator_final_ratio"), precision=3)} / {_value(schedule.get("discriminator_final_ratio"), precision=3)}'),
    ]) + "</div></details>"
    average_gpu = state.get("gpu_compute_average")
    if config.get("device") == "cuda" and average_gpu is not None and average_gpu < 50:
        hint = ("CPU or data loading" if (cpu.get("total_percent") or 0) >= 75
                else "small batches or synchronization overhead")
        summary += f'<p class="metric-note">Possible bottleneck: {hint}.</p>'
    summary += '<p class="metric-note">Loss measures training behaviour. Listen to checkpoints to judge voice quality. Hardware values are the last saved readings.</p>'
    if not progress["losses"]:
        summary += '<p class="empty-state">Loss chart will appear after the first logged training batches. No loss metrics were saved for this run yet.</p>'
    return summary
