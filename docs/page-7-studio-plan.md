# Page 7: Generate & Mix

Design and implementation contract, 2026-10-06. Based on `master`; implemented on `feat/page-7-voice-studio`.

## Outcome

Choose an exported or imported Piper voice, generate speech, download the original WAV, hear all nine effects change during playback, and download a WAV rendered with the selected settings. The original recording and voice model remain intact.

## Page layout

Add `7 · Studio` using the existing page shell, theme, scrolling content, and fixed navigation footer.

1. **Choose your voice:** a dropdown of the selected project's exported and imported ONNX voices, Refresh, and a collapsed Import voice panel. Import accepts the application's voice ZIP or an `.onnx` file with its matching `.onnx.json`. Show voice name and language, distinguishing imports and exports where names repeat. Preserve a valid selection on refresh; automatically select a successful import.
2. **Generate audio:** text area, Generate audio button, status, original player, and Download original WAV. Keep optional Piper speech settings in a collapsed panel and reuse their existing validation. Mixing becomes available only after generation succeeds.
3. **Mix your audio:** one processed player with Play/Pause, seek, optional loop, and an Original/Mixed switch. Below it, three groups of three sliders. Add Restore defaults, Reset to neutral, and Download mixed WAV. Downloads render the whole clip with the current settings, not just the portion played.

Use the supplied Polish effect names and descriptions. Render actual small inline icons if useful; do not display Material icon identifiers such as `height` as text. Each slider has an accessible label, keyboard control, value, unit where appropriate, and concise help text. Three columns on desktop, one column on mobile.

| Group | Control | Initial value | Proposed range | Meaning |
| --- | --- | ---: | --- | --- |
| Barwa i Tonacja | Wysokość (Pitch) | 0 st | -12 to +12 st, step 1 | Shift pitch while preserving speech tempo |
| Barwa i Tonacja | Ciepło (Warmth) | 60 | 0–100 | Low-frequency body through a gentle low shelf |
| Barwa i Tonacja | Powietrze (Air) | 40 | 0–100 | High-frequency detail through a gentle high shelf |
| Tekstura i Dynamika | Saturacja (Saturation) | 30 | 0–100 | Soft saturation with compensated drive |
| Tekstura i Dynamika | Kompresja (Dynamics) | 50 | 0–100 | Progressively stronger level compression |
| Tekstura i Dynamika | De-Esser (Sibilance) | 40 | 0–100 | Dynamic attenuation of the sibilant frequency band |
| Przestrzeń i Poziom | Pogłos (Reverb) | 15 | 0–100 | Dry/wet blend of a short stereo studio room |
| Przestrzeń i Poziom | Szerokość (Width) | 50 | 0–100 | Stereo ambience width: 0 mono, 50 normal, 100 wider |
| Przestrzeń i Poziom | Głośność (Gain) | 0 dB | -18 to +12 dB, step 0.5 | Output level compensation |

The initial values are a starting mix, not neutral settings. Reset to neutral sets pitch/gain to zero, effects to zero, and width to 50. Original bypass always plays the untouched source. Width acts on stereo room ambience; with a dry mono Piper clip and Reverb at zero, it has no stereo content to widen. Explain this next to Width.

## Recommended architecture

Keep Piper synthesis in Python and run the mixer in the browser. Slider movement must not trigger Python callbacks, network requests, Piper synthesis, or playback restarts. Use native Web Audio filters, waveshaping, compression, convolution, channel operations, and gain wherever they cover the effect.

Use one additional, locally bundled and version-pinned dependency for independent pitch shifting: `@soundtouchjs/audio-worklet`. Its current upstream API supports live semitone control and offline rendering; integration and speech quality still require a real browser check. Keep its license with the bundled assets. No CDN at runtime. [Upstream documentation](https://github.com/cutterbl/SoundTouchJS/blob/master/packages/audio-worklet/README.md).

The shared graph is:

```text
Original decoded WAV
  → Pitch
  → Warmth / Air
  → Saturation
  → De-esser
  → Compression
  → Stereo room / Width
  → Gain / peak protection
  → Live playback OR offline WAV rendering
```

Implement a small graph builder accepting an audio context and settings. Live preview uses `AudioContext`; export uses `OfflineAudioContext` with the same graph builder, parameter mappings, room impulse, and output protection. Use a fixed room impulse so two renders do not randomly sound different. Include pitch buffering and the reverb tail in rendering, accounting for startup latency rather than cutting off words. [OfflineAudioContext documentation](https://developer.mozilla.org/en-US/docs/Web/API/OfflineAudioContext).

Pitch uses a time-stretching processor, not a playback-rate shortcut. Bypass it at neutral pitch. A split-band compressor can implement the de-esser with native nodes: compress the high-frequency band and recombine it with the lower band. Tune conservative ranges using Polish speech. Air remains EQ; de-essing responds dynamically to sharp consonants.

Encode the mixed result as 16-bit PCM WAV in the browser. Preserve the source sample rate and use stereo output for the room effect. Render on Download only; capture a settings snapshot and disable editing during that short render so the download corresponds to the displayed mix. Export selected final settings, not a recording of slider movements.

Other approaches considered: debounced server-side FFmpeg renders would reuse an installed tool but would not provide continuous live mixing; separate browser and FFmpeg implementations would duplicate DSP rules and risk different preview/export sound. The shared browser graph best meets this request.

## Integration and ownership

- **`app/pages.py`:** add the page, voice controls, generation wiring, and Next/Back navigation. Keep new page UI in a focused builder function if it makes the existing file easier to read.
- **New `app/voices.py`:** discover valid voice pairs, validate selected models, and safely import voices into the existing project `models` directory. Have existing `model_choices()` use this discovery logic rather than create a second voice registry.
- **`app/main.py` / `app/inference.py`:** reuse `synthesize_model()`, `synthesize()`, and synthesis-parameter validation. Add only the necessary import/refresh wrappers and pass the returned audio URL to the mixer explicitly through Gradio's event bridge.
- **New `app/static/studio.js`:** own effect definitions, defaults/ranges, DSP mappings, graph construction, playback state, offline rendering, and WAV encoding. Generate mixer controls from these definitions so UI values and DSP validation share one source of truth. Keep event binding separate from DSP functions inside this module.
- **`app/workflow.py`, `app/ui.py`, `app/ui.css`:** extend navigation and availability to seven pages. Put the workflow step count in one shared constant and derive headings/ranges from it. Keep page 7 available after selecting a project, without requiring a dataset, training run, or checkpoint; an import is sufficient to start.
- **Static assets and notices:** serve only the specific mixer asset directory through Gradio's supported file serving. Include the pinned pitch processor and its license. Docker already copies `app`; no new service is needed.
- **Tests and `scripts/check-ui-layout.js`:** cover the seventh page and the complete generation/mixing/download flow.

The repo pins Gradio 5.49.1. Use its `Blocks(js=..., head=...)` and JavaScript event callbacks; its HTML component renders static HTML and does not provide the newer `js_on_load` API. Bind only to our own page root and controls, avoiding Gradio's internal DOM. Pass the framework-provided media URL rather than construct a raw file route. [Pinned HTML source](https://github.com/gradio-app/gradio/blob/gradio%405.49.1/gradio/components/html.py).

## State, validation, and cleanup

- Keep one immutable decoded source buffer per browser session. Effects never overwrite it. Slider parameters remain per session, with no global audio state on the server.
- Clear the source/player/download when project, voice, text, or synthesis settings change. Stop playback when leaving the page. Preserve mixer settings for the next generation within the same project; reset them when switching projects.
- Tag asynchronous source loading and rendering with a generation identifier so an older completion cannot replace a newer clip. Only successful synthesis enables the mixer.
- Ramp parameter changes over short intervals and crossfade bypass transitions to avoid clicks. Pitch may have processing latency, but changing it must not reset playback or change tempo.
- Release audio nodes, contexts, listeners, and object URLs when replaced or disposed. Avoid duplicate initialization after tab changes.
- Validate uploads before making them available: one unambiguous model/config pair, valid Piper configuration, a loadable ONNX model, and project-contained resolved paths. Reject traversal, symlinks, duplicate archive paths, and oversized archives. Start with a 1 GiB upload/extracted-size cap and a small archive entry limit suitable for a two-file voice package. Copy only the voice pair, using a temporary directory and atomic final placement; clean up failures and never overwrite an existing voice.
- Apply client-side bounds and finite-number checks to mixer values. Show decode, pitch initialization, and export errors inline. Keep original WAV downloading available if mixing fails.
- AudioWorklet needs a secure context. Support localhost and HTTPS deployment; detect unsupported contexts/browsers before enabling pitch and mixed export, with an actionable explanation. Plain HTTP on a remote IP is not a supported full-mixer deployment. [AudioWorklet documentation](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet).
- Bound decoded audio memory; for the first version, set a five-minute clip limit for browser mixing and explain it inline. Longer synthesis can still be downloaded as original audio. This limit can be revisited from actual browser memory measurements.

## Delivery and acceptance

1. **Prove the path first:** select one exported voice, generate real Piper audio, bridge its URL to the page, play it through gain and pitch, and download a processed WAV. Verify pitch preserves duration and export retains the first and final words before adding the remaining effects.
2. **Add import and page integration:** safe ZIP/pair import, refresh/selection behavior, seven-page navigation, original download, and clear empty/error states.
3. **Complete the mixer:** all nine controls, exact supplied defaults/descriptions, original/mixed switch, resets, shared offline rendering, and tail/latency handling.
4. **Verify actual behavior:** use the existing Python test runner for import/inference/workflow contracts and one focused browser DSP check for neutral output, pitch frequency/duration, finite samples, de-esser response, stereo width, and export tail. Compare a fresh static live graph with the downloaded WAV within an appropriate numeric tolerance. Audition Polish speech with sibilants at defaults and extremes; numerical tests alone do not establish voice quality.
5. **Check the UI and runtime:** exported voice → generation → original download → slider changes during playback → mixed download → playback of downloaded WAV; repeat with an imported voice. Verify Chromium and Firefox, mobile layout at 390 px, desktop layout at 1280/1440/1920 px, keyboard operation, no overflow/console errors, two-session isolation, and stale-output handling. Run relevant existing tests and extend the layout script to include page 7.

Completion means all nine controls affect their documented signal, live playback continues during edits, pitch preserves speech tempo, the original remains downloadable, and the mixed download reproduces the chosen effects without missing speech or the room tail. Browser pitch integration remains an implementation risk until step 1 passes.

## Implementation verification

- Python suite: 104 tests pass in the Piper CPU container. Host suite passes with six missing-dependency skips.
- Chromium: real exported and ZIP-imported Piper voices generate successfully; original and mixed WAV downloads work. Refresh preserves the selected voice and generated audio. All nine controls update during playback; seeking, reset, and blocked-playback recovery pass.
- `scripts/check-studio-audio.js`: neutral fidelity, pitch frequency/duration and final samples, every effect's signal contract, stereo width, reverb tail, deterministic rendering, invalid settings, and WAV headers pass in Chromium and Firefox.
- `scripts/check-studio-preview.js`: 48,510 live samples match the default offline export with zero relative RMS error in Chromium. This verifies a static default mix; it does not record slider automation.
- `scripts/check-studio-ui.js`: live controls and error recovery pass in Chromium. Firefox can generate and export, but its headless realtime audio backend stays suspended even for a plain AudioContext, so Firefox live playback remains unverified in this environment. No subjective voice-quality claim is made.
- Insecure HTTP: generation and original WAV download work; the mixer explains its secure-context requirement.
- Layout checks pass across all seven pages at 390, 1280, 1440, and 1920 pixels, with stable page geometry and footer and no viewport overflow.
- Final code review findings were reproduced and fixed: staging outside voice discovery, monotonic request IDs for insecure HTTP, and upstream Piper configuration validation. Import checks also reject external ONNX tensor paths before runtime loading.
- Reproduce the bundled pitch asset with `python scripts/vendor-soundtouch.py`; the pinned archive's SHA-256 is verified. Run browser checks with `agent-browser eval --stdin < scripts/check-studio-audio.js` (and the preview/UI scripts) after opening Studio and generating a clip. The preview/UI scripts require a functioning realtime audio backend.
