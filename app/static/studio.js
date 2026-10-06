// DSP and controls share these definitions. No server requests during mixing.
export const controls = [
  ['pitch', 'Wysokość (Pitch)', 'Przesunięcie półtonów bez zmiany tempa', -12, 12, 1, 0, ' st'],
  ['warmth', 'Ciepło (Warmth)', 'Nasycenie niskich tonów i analogowa głębia', 0, 100, 1, 60, ''],
  ['air', 'Powietrze (Air)', 'Krystaliczna góra i detale oddechu', 0, 100, 1, 40, ''],
  ['saturation', 'Saturacja (Saturation)', 'Lampowy charakter i gęstość brzmienia', 0, 100, 1, 30, ''],
  ['dynamics', 'Kompresja (Dynamics)', 'Wyrównanie poziomów i zwartość wokalu', 0, 100, 1, 50, ''],
  ['deesser', 'De-Esser (Sibilance)', 'Tłumienie ostrych syczeń (S, C, Z)', 0, 100, 1, 40, ''],
  ['reverb', 'Pogłos (Reverb)', 'Symulacja przestrzeni studyjnej', 0, 100, 1, 15, ''],
  ['width', 'Szerokość (Width)', 'Szerokość sceny stereo', 0, 100, 1, 50, ''],
  ['gain', 'Głośność (Gain)', 'Kompensacja poziomu wyjściowego', -18, 12, .5, 0, ' dB'],
];
export const defaults = () => Object.fromEntries(controls.map(([id, , , , , , value]) => [id, value]));
export const neutral = () => ({...Object.fromEntries(controls.map(([id]) => [id, 0])), width: 50});
const ROOM_SECONDS = 1.2;
const FLUSH_SECONDS = .35;
const processorUrl = new URL('./soundtouch-processor.js', import.meta.url).href;
const registered = new WeakMap();

function validated(settings) {
  const result = {...defaults(), ...settings};
  for (const [id, , , min, max] of controls) {
    if (!Number.isFinite(result[id]) || result[id] < min || result[id] > max) throw Error(`Invalid ${id} value`);
  }
  return result;
}

function roomImpulse(context) {
  const buffer = context.createBuffer(2, Math.ceil(ROOM_SECONDS * context.sampleRate), context.sampleRate);
  let seed = 123456;
  for (let channel = 0; channel < 2; channel++) {
    const data = buffer.getChannelData(channel);
    for (let i = 0; i < data.length; i++) {
      seed = (1664525 * seed + 1013904223) >>> 0;
      data[i] = (seed / 2 ** 32 * 2 - 1) * .035 * Math.exp(-6 * i / data.length);
    }
  }
  return buffer;
}

export async function createGraph(context, initialSettings) {
  validated(initialSettings);
  if (!context.audioWorklet) throw Error('Live mixing requires HTTPS or localhost and a browser with AudioWorklet. You can still download the original.');
  if (!registered.has(context)) registered.set(context, context.audioWorklet.addModule(processorUrl));
  await registered.get(context);
  const nodes = [];
  const node = value => { nodes.push(value); return value; };
  const gain = value => { const result = node(context.createGain()); result.gain.value = value; return result; };
  const input = gain(1);
  input.channelCount = 2; input.channelCountMode = 'explicit';
  let last = input;
  const param = (parameter, value, smooth) => {
    parameter.cancelScheduledValues(context.currentTime);
    if (smooth) parameter.setTargetAtTime(value, context.currentTime, .015);
    else parameter.setValueAtTime(value, context.currentTime);
  };
  const effect = (first, end = first) => {
    const dry = gain(1), wet = gain(0), output = gain(1);
    last.connect(dry).connect(output);
    last.connect(first);
    end.connect(wet).connect(output);
    last = output;
    return (active, smooth) => { param(dry.gain, active ? 0 : 1, smooth); param(wet.gain, active ? 1 : 0, smooth); };
  };
  const pitch = node(new AudioWorkletNode(context, 'soundtouch-processor', {outputChannelCount: [2]}));
  const pitchActive = effect(pitch);
  const filter = (type, frequency) => {
    const result = node(context.createBiquadFilter());
    result.type = type; result.frequency.value = frequency; result.Q.value = Math.SQRT1_2;
    result.channelCount = 2; result.channelCountMode = 'explicit';
    return result;
  };
  const warmth = filter('lowshelf', 180), air = filter('highshelf', 6500);
  last.connect(warmth).connect(air); last = air;
  const saturation = node(context.createWaveShaper());
  saturation.oversample = '2x';
  const saturationActive = effect(saturation);
  const splitInput = gain(1), splitOutput = gain(1);
  const low1 = filter('lowpass', 4800), low2 = filter('lowpass', 4800);
  const high1 = filter('highpass', 4800), high2 = filter('highpass', 4800);
  const sibilance = node(context.createDynamicsCompressor());
  sibilance.attack.value = .001; sibilance.release.value = .06; sibilance.knee.value = 0;
  const sibilanceCompensation = gain(1);
  // Compressor lookahead is 6 ms; delay the lower band to keep the crossover aligned.
  const lowDelay = node(context.createDelay(.01)); lowDelay.delayTime.value = .006;
  splitInput.connect(low1).connect(low2).connect(lowDelay).connect(splitOutput);
  splitInput.connect(high1).connect(high2).connect(sibilance).connect(sibilanceCompensation).connect(splitOutput);
  const deesserActive = effect(splitInput, splitOutput);
  const compressor = node(context.createDynamicsCompressor());
  compressor.attack.value = .01; compressor.release.value = .15; compressor.knee.value = 18;
  const makeup = gain(1); compressor.connect(makeup);
  const compressorActive = effect(compressor, makeup);
  const room = node(context.createConvolver()); room.normalize = false; room.buffer = roomImpulse(context);
  const roomWet = gain(0), roomDry = gain(1), roomOutput = gain(1);
  last.connect(roomDry).connect(roomOutput);
  const splitter = node(context.createChannelSplitter(2)), merger = node(context.createChannelMerger(2));
  last.connect(room).connect(splitter);
  const direct = [gain(1), gain(1)], cross = [gain(0), gain(0)];
  splitter.connect(direct[0], 0); direct[0].connect(merger, 0, 0);
  splitter.connect(direct[1], 1); direct[1].connect(merger, 0, 1);
  splitter.connect(cross[0], 1); cross[0].connect(merger, 0, 0);
  splitter.connect(cross[1], 0); cross[1].connect(merger, 0, 1);
  merger.connect(roomWet).connect(roomOutput);
  const outputGain = gain(1), ceiling = node(context.createWaveShaper());
  ceiling.curve = Float32Array.from({length: 2049}, (_, i) => Math.max(-.98, Math.min(.98, i / 1024 - 1)));
  roomOutput.connect(outputGain).connect(ceiling);
  ceiling.channelCount = 2; ceiling.channelCountMode = 'explicit';
  let previousSaturation;
  function set(raw, smooth = true) {
    const s = validated(raw);
    param(pitch.parameters.get('pitchSemitones'), s.pitch, smooth); pitchActive(s.pitch !== 0, smooth);
    param(warmth.gain, s.warmth * .06, smooth); param(air.gain, s.air * .05, smooth);
    if (previousSaturation !== s.saturation) {
      const drive = 1 + s.saturation * .04;
      saturation.curve = Float32Array.from({length: 2049}, (_, i) => Math.tanh((i / 1024 - 1) * drive) / drive);
      previousSaturation = s.saturation;
    }
    saturationActive(s.saturation > 0, smooth);
    const threshold = -12 - s.deesser * .28, ratio = 1 + s.deesser * .09;
    param(sibilance.threshold, threshold, smooth);
    param(sibilance.ratio, ratio, smooth);
    // With a hard knee, native makeup gain is -threshold * (1 - 1/ratio) * .6 dB.
    // Cancel it: a de-esser must attenuate the high band, never automatically boost it.
    param(sibilanceCompensation.gain, 10 ** (threshold * (1 - 1 / ratio) * .6 / 20), smooth);
    deesserActive(s.deesser > 0, smooth);
    param(compressor.threshold, -12 - s.dynamics * .24, smooth);
    param(compressor.ratio, 1 + s.dynamics * .07, smooth);
    param(makeup.gain, 10 ** (s.dynamics * .03 / 20), smooth); compressorActive(s.dynamics > 0, smooth);
    param(roomWet.gain, s.reverb * .004, smooth); param(roomDry.gain, 1 - s.reverb * .002, smooth);
    const width = s.width / 50;
    for (const value of direct) param(value.gain, (1 + width) / 2, smooth);
    for (const value of cross) param(value.gain, (1 - width) / 2, smooth);
    param(outputGain.gain, 10 ** (s.gain / 20), smooth);
  }
  set(initialSettings, false);
  return {input, output: ceiling, set, dispose() {
    for (const value of nodes) { value.disconnect(); if (value.port) value.port.close(); }
  }};
}

function firstSignal(buffer) {
  const data = buffer.getChannelData(0);
  const index = data.findIndex(value => Math.abs(value) > 1e-7);
  return index < 0 ? 0 : index;
}

export async function renderAudio(input, rawSettings) {
  const settings = validated(rawSettings);
  const tail = settings.reverb ? ROOM_SECONDS : 0;
  const length = input.length + Math.ceil((FLUSH_SECONDS + tail) * input.sampleRate);
  const context = new OfflineAudioContext(2, length, input.sampleRate);
  const graph = await createGraph(context, settings);
  const padded = context.createBuffer(input.numberOfChannels, length, input.sampleRate);
  for (let i = 0; i < input.numberOfChannels; i++) padded.copyToChannel(input.getChannelData(i), i);
  const source = context.createBufferSource(); source.buffer = padded;
  source.connect(graph.input); graph.output.connect(context.destination); source.start(0);
  try {
    const rendered = await context.startRendering();
    // Remove only added DSP startup latency, preserving the source's leading silence.
    const latency = Math.max(0, Math.min(Math.ceil(FLUSH_SECONDS * input.sampleRate), firstSignal(rendered) - firstSignal(input)));
    const result = context.createBuffer(2, input.length + Math.ceil(tail * input.sampleRate), input.sampleRate);
    for (let i = 0; i < 2; i++) result.copyToChannel(rendered.getChannelData(i).subarray(latency, latency + result.length), i);
    return result;
  } finally { source.disconnect(); graph.dispose(); }
}

export function encodeWav(buffer) {
  const channels = buffer.numberOfChannels, size = buffer.length * channels * 2;
  const bytes = new ArrayBuffer(44 + size), view = new DataView(bytes);
  const text = (offset, value) => [...value].forEach((char, i) => view.setUint8(offset + i, char.charCodeAt(0)));
  text(0, 'RIFF'); view.setUint32(4, 36 + size, true); text(8, 'WAVE'); text(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, channels, true);
  view.setUint32(24, buffer.sampleRate, true); view.setUint32(28, buffer.sampleRate * channels * 2, true);
  view.setUint16(32, channels * 2, true); view.setUint16(34, 16, true); text(36, 'data'); view.setUint32(40, size, true);
  const data = Array.from({length: channels}, (_, i) => buffer.getChannelData(i));
  for (let i = 0, offset = 44; i < buffer.length; i++) for (let channel = 0; channel < channels; channel++, offset += 2) {
    const value = Math.max(-1, Math.min(1, data[channel][i]));
    view.setInt16(offset, Math.round(value * (value < 0 ? 32768 : 32767)), true);
  }
  return new Blob([bytes], {type: 'audio/wav'});
}

let controller, requestCounter = 0;
export function mount() {
  const root = document.getElementById('studio-mixer');
  if (!root || controller?.root === root) return;
  controller?.dispose();
  root.innerHTML = `
    <p id="studio-message" role="status" aria-live="polite">Choose a voice and generate audio to start mixing.</p>
    <section id="studio-original" hidden>
      <label for="studio-original-player">Original audio</label>
      <audio id="studio-original-player" aria-label="Original audio" controls preload="metadata"></audio>
      <a id="studio-original-download" download="piper-original.wav">Download original WAV</a>
    </section>
    <h3>3. Mix your audio</h3>
    <fieldset id="studio-controls" disabled>
      <legend class="studio-sr-only">Live audio mixer</legend>
      <div class="studio-transport">
        <button type="button" data-action="play">Play mix</button>
        <button type="button" data-action="bypass" aria-pressed="false">Hear original</button>
        <label><input id="studio-loop" type="checkbox"> Loop</label>
        <output id="studio-time">0:00 / 0:00</output>
      </div>
      <label class="studio-sr-only" for="studio-seek">Playback position</label>
      <input id="studio-seek" type="range" min="0" max="1" step="0.01" value="0">
      <div class="studio-effects">${['Barwa i Tonacja', 'Tekstura i Dynamika', 'Przestrzeń i Poziom'].map((group, index) => `
        <section class="studio-effect-group"><h4>${group}</h4>${controls.slice(index * 3, index * 3 + 3).map(([id, title, help, min, max, step, value, unit]) => `
          <div class="studio-effect"><div><label for="studio-${id}">${title}</label><output id="studio-value-${id}" for="studio-${id}">${value}${unit}</output></div>
          <p id="studio-help-${id}">${help}</p><input id="studio-${id}" data-effect="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${value}" aria-valuetext="${value}${unit}" aria-describedby="studio-help-${id}"></div>
        `).join('')}</section>`).join('')}
      </div>
      <p class="studio-mixer-note">Szerokość działa na pogłos stereo. Przy pogłosie 0 głos pozostaje mono.</p>
      <div class="studio-transport">
        <button type="button" data-action="defaults">Restore defaults</button>
        <button type="button" data-action="neutral">Reset to neutral</button>
        <button type="button" data-action="download" class="studio-primary">Download mixed WAV</button>
      </div>
    </fieldset>`;
  const element = id => root.querySelector(`#studio-${id}`);
  const original = element('original-player'), fields = element('controls'), seek = element('seek');
  const playButton = root.querySelector('[data-action="play"]'), bypassButton = root.querySelector('[data-action="bypass"]');
  let settings = defaults(), buffer = null, context = null, graph = null, source = null;
  let rawGain = null, mixedGain = null, offset = 0, started = 0, frame = null;
  let request = '', bypass = false, busy = false, starting = false;
  const message = text => { element('message').textContent = text; };
  const clock = value => `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, '0')}`;
  const position = () => {
    if (!source) return offset;
    const elapsed = offset + context.currentTime - started;
    return source.loop ? elapsed % buffer.duration : Math.min(buffer.duration, elapsed);
  };
  const ramp = (parameter, value) => parameter.setTargetAtTime(value, context.currentTime, .015);
  function showSettings() {
    for (const [id, , , , , , , unit] of controls) {
      element(id).value = settings[id]; element(id).setAttribute('aria-valuetext', settings[id] + unit);
      element(`value-${id}`).textContent = settings[id] + unit;
    }
  }
  function tick() {
    let time = position();
    if (source && source.loop) time = time % buffer.duration;
    seek.value = time;
    element('time').textContent = `${clock(time)} / ${clock(buffer?.duration || 0)}`;
    frame = source ? requestAnimationFrame(tick) : null;
  }
  function pause(reset = false) {
    if (source) {
      offset = position(); source.onended = null; source.stop(); source.disconnect(); source = null;
      graph.dispose(); graph = null; rawGain.disconnect(); mixedGain.disconnect();
    }
    if (reset) offset = 0;
    cancelAnimationFrame(frame); frame = null;
    playButton.textContent = 'Play mix';
    if (buffer) tick();
  }
  async function closeAudio() {
    pause();
    const previous = context; context = null;
    if (previous && previous.state !== 'closed') await previous.close();
  }
  async function play() {
    if (!buffer || starting || busy) return;
    starting = true; playButton.disabled = true;
    const expected = request;
    try {
      original.pause();
      if (!context || context.state === 'closed') context = new AudioContext({sampleRate: buffer.sampleRate});
      const activeContext = context;
      let timeout;
      try {
        await Promise.race([activeContext.resume(), new Promise((_, reject) => {
          timeout = setTimeout(() => reject(Error('Audio playback could not start. Check browser sound permissions or your output device.')), 5000);
        })]);
      } finally { clearTimeout(timeout); }
      const newGraph = await createGraph(activeContext, settings);
      if (request !== expected || context !== activeContext) { newGraph.dispose(); return; }
      graph = newGraph;
      if (offset >= buffer.duration) offset = 0;
      const padded = context.createBuffer(buffer.numberOfChannels, buffer.length + Math.ceil((FLUSH_SECONDS + ROOM_SECONDS) * buffer.sampleRate), buffer.sampleRate);
      for (let i = 0; i < buffer.numberOfChannels; i++) padded.copyToChannel(buffer.getChannelData(i), i);
      source = context.createBufferSource(); source.buffer = padded;
      source.loop = element('loop').checked; source.loopEnd = buffer.duration;
      rawGain = context.createGain(); rawGain.gain.value = bypass ? 1 : 0;
      mixedGain = context.createGain(); mixedGain.gain.value = bypass ? 0 : 1;
      source.connect(graph.input); graph.output.connect(mixedGain).connect(context.destination);
      source.connect(rawGain).connect(context.destination);
      source.onended = () => pause(true);
      started = context.currentTime; source.start(0, offset);
      playButton.textContent = 'Pause'; tick();
    } catch (error) {
      if (request !== expected) return;
      await closeAudio();
      throw error;
    } finally { starting = false; playButton.disabled = false; }
  }
  function invalidate(resetSettings = false) {
    request = String(++requestCounter);
    closeAudio().catch(error => message(error.message));
    original.pause(); original.removeAttribute('src'); original.load();
    buffer = null; offset = 0; fields.disabled = true; element('original').hidden = true;
    element('original-download').removeAttribute('href'); seek.value = 0;
    bypass = false; bypassButton.setAttribute('aria-pressed', 'false'); bypassButton.textContent = 'Hear original';
    element('time').textContent = '0:00 / 0:00';
    if (resetSettings) { settings = defaults(); showSettings(); }
    message('Generate audio to start mixing.');
    return request;
  }
  const inputListener = event => {
    const id = event.target.dataset.effect;
    if (id) {
      settings = validated({...settings, [id]: Number(event.target.value)});
      showSettings(); graph?.set(settings);
    }
    if (event.target === element('loop') && source) source.loop = event.target.checked;
  };
  root.addEventListener('input', inputListener);
  const seekListener = async () => {
    const resume = !!source;
    const target = Number(seek.value);
    pause(); offset = target; tick();
    if (resume) try { await play(); } catch (error) { message(error.message); }
  };
  seek.addEventListener('change', seekListener);
  const originalPlay = () => pause(); original.addEventListener('play', originalPlay);
  const clickListener = async event => {
    const action = event.target.closest('[data-action]')?.dataset.action;
    if (!action || busy) return;
    try {
      if (action === 'play') { if (source) pause(); else await play(); }
      if (action === 'bypass') {
        bypass = !bypass; bypassButton.setAttribute('aria-pressed', String(bypass));
        bypassButton.textContent = bypass ? 'Hear mixed' : 'Hear original';
        if (source) { ramp(rawGain.gain, bypass ? 1 : 0); ramp(mixedGain.gain, bypass ? 0 : 1); }
      }
      if (action === 'defaults' || action === 'neutral') {
        settings = action === 'defaults' ? defaults() : neutral(); showSettings(); graph?.set(settings);
      }
      if (action === 'download') {
        const expected = request, clip = buffer, snapshot = {...settings};
        busy = true; fields.disabled = true; pause(); message('Rendering your full mix…');
        try {
          const result = await renderAudio(clip, snapshot);
          if (expected !== request) return;
          const url = URL.createObjectURL(encodeWav(result));
          const link = document.createElement('a'); link.href = url; link.download = 'piper-mixed.wav';
          document.body.append(link); link.click(); link.remove();
          // Give the browser time to start reading the download before releasing its URL.
          setTimeout(() => URL.revokeObjectURL(url), 10000);
          message('Mixed WAV downloaded. Includes the full clip and studio room tail.');
        } finally { busy = false; fields.disabled = !buffer; }
      }
    } catch (error) { if (source) pause(); message(error.message); }
  };
  root.addEventListener('click', clickListener);
  const resize = new ResizeObserver(entries => {
    if (!entries[0].contentRect.width) {
      original.pause();
      if (context) closeAudio().catch(error => message(error.message));
    }
  });
  resize.observe(root);
  controller = {root, invalidate, beginGeneration() { invalidate(); message('Generating Piper audio…'); return request; },
    async load(file, metadata) {
      if (metadata?.request !== request) return;
      if (!file?.url) { message(metadata?.message || 'Audio generation failed.'); return; }
      const expected = request;
      const url = new URL(file.url, document.baseURI);
      if (url.origin !== location.origin) { message('Audio must come from this application.'); return; }
      original.src = url.href; element('original-download').href = url.href; element('original').hidden = false;
      if (metadata.duration > 300) { message('Original ready. Browser mixing supports clips up to five minutes; shorten the text to mix it.'); return; }
      try {
        const response = await fetch(url);
        if (!response.ok) throw Error(`Could not load audio (${response.status})`);
        const decoder = new OfflineAudioContext(1, 1, metadata.sample_rate);
        const decoded = await decoder.decodeAudioData(await response.arrayBuffer());
        if (request !== expected) return;
        buffer = decoded; seek.max = buffer.duration; tick();
        fields.disabled = !window.isSecureContext || !('AudioWorkletNode' in window);
        message(fields.disabled ? 'Original ready. Live mixing requires HTTPS or localhost and AudioWorklet support.' : 'Audio ready. Press Play mix and move the sliders to hear changes live.');
      } catch (error) { if (expected === request) message(error.message); }
    },
    dispose() { invalidate(); resize.disconnect(); root.removeEventListener('input', inputListener); root.removeEventListener('click', clickListener); seek.removeEventListener('change', seekListener); original.removeEventListener('play', originalPlay); },
  };
}
export function invalidate(resetSettings = false) { mount(); controller?.invalidate(resetSettings); }
export function beginGeneration() { mount(); return controller.beginGeneration(); }
export async function load(file, metadata) { mount(); await controller.load(file, metadata); }
window.addEventListener('pagehide', () => { controller?.dispose(); controller = null; });
