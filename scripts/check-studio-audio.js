// Runnable in the Studio page: agent-browser eval --stdin < scripts/check-studio-audio.js
(async () => {
  const {renderAudio, defaults, neutral, encodeWav, createGraph} = window.PiperStudio;
  const assert = (condition, message) => { if (!condition) throw Error(message); };
  const rate = 22050;
  const context = new OfflineAudioContext(1, rate * 2, rate);
  const input = context.createBuffer(1, rate * 2, rate);
  const samples = input.getChannelData(0);
  for (let i = 0; i < samples.length; i++) samples[i] = .2 * Math.sin(2 * Math.PI * 440 * i / rate);
  const rms = (buffer, channel = 0) => {
    const data = buffer.getChannelData(channel);
    let sum = 0;
    for (const value of data) { assert(Number.isFinite(value), 'Non-finite audio'); sum += value * value; }
    return Math.sqrt(sum / data.length);
  };
  const dry = await renderAudio(input, neutral());
  assert(dry.length === input.length, 'Neutral changes duration');
  let error = 0;
  for (let i = 0; i < samples.length; i++) error = Math.max(error, Math.abs(samples[i] - dry.getChannelData(0)[i]));
  assert(error < 1e-6, `Neutral changes samples: ${error}`);
  const louder = await renderAudio(input, {...neutral(), gain: 6});
  assert(Math.abs(rms(louder) / rms(dry) - 10 ** .3) < .01, 'Gain dB mapping');
  const pitched = await renderAudio(input, {...neutral(), pitch: 12});
  let crossings = 0;
  const shifted = pitched.getChannelData(0);
  for (let i = rate / 2 | 0; i < rate * 1.5; i++) if (shifted[i - 1] < 0 && shifted[i] >= 0) crossings++;
  assert(Math.abs(crossings - 880) < 15, `Pitch octave incorrect: ${crossings}`);
  assert(pitched.length === input.length, 'Pitch changes speech duration');
  assert(rms(pitched) > .05, 'Pitch output missing');
  assert(shifted.slice(-rate / 10).some(value => Math.abs(value) > .05), 'Pitch cuts final audio');
  const tone = frequency => {
    const buffer = context.createBuffer(1, rate * 2, rate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = .3 * Math.sin(2 * Math.PI * frequency * i / rate);
    return buffer;
  };
  const low = tone(100), high = tone(8000);
  assert(rms(await renderAudio(low, {...neutral(), warmth: 100})) > rms(low) * 1.5, 'Warmth does not lift low frequencies');
  assert(rms(await renderAudio(high, {...neutral(), air: 100})) > rms(high) * 1.5, 'Air does not lift high frequencies');
  assert(rms(await renderAudio(input, {...neutral(), saturation: 100})) < rms(input) * .9, 'Saturation has no effect');
  const changing = tone(440);
  changing.getChannelData(0).forEach((value, i, data) => { data[i] = value * (i < rate ? .2 : 2); });
  const compressed = await renderAudio(changing, {...neutral(), dynamics: 100});
  const sectionRms = (data, start, end) => Math.sqrt(data.slice(start, end).reduce((sum, value) => sum + value * value, 0) / (end - start));
  const compressedData = compressed.getChannelData(0);
  assert(sectionRms(compressedData, rate * 1.5, rate * 1.9) / sectionRms(compressedData, rate * .5, rate * .9) < 3, 'Compression does not level quiet/loud sections');
  assert(rms(await renderAudio(high, {...neutral(), deesser: 100})) < rms(high) * .6, 'De-esser does not attenuate sibilant band');
  assert(Math.abs(rms(await renderAudio(low, {...neutral(), deesser: 100})) / rms(low) - 1) < .03, 'De-esser damages low frequencies');
  const room = await renderAudio(input, {...neutral(), reverb: 50, width: 100});
  assert(room.length > input.length, 'Missing room tail');
  assert(room.getChannelData(0).slice(input.length).some(value => Math.abs(value) > 1e-5), 'Empty room tail');
  let stereo = 0;
  for (let i = 0; i < room.length; i++) stereo += Math.abs(room.getChannelData(0)[i] - room.getChannelData(1)[i]);
  assert(stereo > 1, 'Width produces no stereo');
  const mono = await renderAudio(input, {...neutral(), reverb: 50, width: 0});
  assert(mono.getChannelData(0).every((value, i) => value === mono.getChannelData(1)[i]), 'Width zero is not mono');
  const mixed = await renderAudio(input, defaults());
  assert(rms(mixed) > .01, 'Default mix is silent');
  const repeated = await renderAudio(input, defaults());
  assert(mixed.getChannelData(0).every((value, i) => Math.abs(value - repeated.getChannelData(0)[i]) < 1e-6), 'Export is nondeterministic');
  const blob = encodeWav(mixed);
  const bytes = new DataView(await blob.arrayBuffer());
  assert(bytes.getUint32(24, true) === rate && bytes.getUint16(22, true) === 2, 'WAV format incorrect');
  assert(bytes.getUint32(40, true) === mixed.length * 4, 'WAV frame count incorrect');
  for (const invalid of [NaN, Infinity, 13]) {
    let rejected = false;
    try { await renderAudio(input, {...neutral(), pitch: invalid}); } catch { rejected = true; }
    assert(rejected, 'Invalid pitch accepted');
  }
  const live = new AudioContext({sampleRate: rate});
  const graph = await createGraph(live, neutral());
  graph.set({...defaults(), pitch: -3});
  graph.dispose();
  await live.close();
  return {result: 'PASS', neutralError: error, pitchHz: crossings, samples: mixed.length, wavBytes: blob.size};
})()
