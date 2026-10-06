// Run after generating real Piper audio in page 7. Leaves playback paused.
// agent-browser eval --stdin < scripts/check-studio-ui.js
(async () => {
  const root = document.querySelector('#studio-mixer');
  const assert = (condition, message) => { if (!condition) throw Error(message); };
  const wait = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
  assert(!root.querySelector('#studio-original').hidden, 'Generated original audio is missing');
  assert(!root.querySelector('fieldset').disabled, 'Mixer is unavailable after successful generation');
  assert(root.querySelectorAll('[data-effect]').length === 9, 'Expected nine effects');
  const seek = root.querySelector('#studio-seek');
  seek.value = Math.min(1, Number(seek.max) / 2);
  const target = Number(seek.value);
  seek.dispatchEvent(new Event('change', {bubbles: true}));
  await wait(50);
  assert(Math.abs(Number(seek.value) - target) < .05, 'Seeking loses the chosen position');
  const play = root.querySelector('[data-action="play"]');
  play.click();
  await wait(500);
  assert(play.textContent === 'Pause', 'Mix playback did not start');
  const before = Number(seek.value);
  root.querySelector('#studio-gain').value = 6;
  root.querySelector('#studio-gain').dispatchEvent(new Event('input', {bubbles: true}));
  await wait(200);
  assert(Number(seek.value) > before && play.textContent === 'Pause', 'Live effect change restarts/stops playback');
  assert(root.querySelector('#studio-value-gain').textContent === '6 dB', 'Effect value label is stale');
  for (const [id, value] of Object.entries({pitch: 3, warmth: 80, air: 65, saturation: 40, dynamics: 55, deesser: 55, reverb: 20, width: 75})) {
    const control = root.querySelector(`#studio-${id}`);
    control.value = value; control.dispatchEvent(new Event('input', {bubbles: true}));
    assert(play.textContent === 'Pause', `${id} interrupted playback`);
  }
  play.click();
  root.querySelector('[data-action="defaults"]').click();
  assert(root.querySelector('#studio-gain').value === '0', 'Restore defaults failed');
  const resume = AudioContext.prototype.resume;
  try {
    AudioContext.prototype.resume = () => new Promise(() => {});
    play.click(); await wait(5300);
    assert(root.querySelector('#studio-message').textContent.includes('Audio playback could not start'), 'Blocked playback hangs without an explanation');
    assert(!play.disabled && !root.querySelector('#studio-original').hidden, 'Playback failure disables retries or original download');
  } finally { AudioContext.prototype.resume = resume; }
  return {result: 'PASS', livePlayback: true, effects: 9, seekTarget: target, playbackFailureRecovery: true};
})()
