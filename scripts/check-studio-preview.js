// Compare live graph samples with the exported graph. Requires functioning audio output.
// Run after clicking Generate in Studio: agent-browser eval --stdin < scripts/check-studio-preview.js
(async () => {
  const {createGraph, renderAudio, defaults} = window.PiperStudio;
  const context = new AudioContext({sampleRate: 22050});
  const url = URL.createObjectURL(new Blob([`
    registerProcessor('studio-capture', class extends AudioWorkletProcessor {
      process(inputs) { if (inputs[0]?.length) this.port.postMessage(inputs[0].map(channel => channel.slice())); return true; }
    });`], {type: 'text/javascript'}));
  let graph, capture, source;
  try {
    await context.resume();
    await context.audioWorklet.addModule(url);
    const input = context.createBuffer(1, context.sampleRate, context.sampleRate);
    const data = input.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = .2 * Math.sin(2 * Math.PI * 440 * i / context.sampleRate);
    const exported = await renderAudio(input, defaults());
    graph = await createGraph(context, defaults());
    capture = new AudioWorkletNode(context, 'studio-capture', {outputChannelCount: [2]});
    const chunks = [];
    capture.port.onmessage = event => chunks.push(event.data[0]);
    graph.output.connect(capture).connect(context.destination);
    source = context.createBufferSource();
    source.buffer = context.createBuffer(1, input.length + Math.ceil(1.55 * context.sampleRate), context.sampleRate);
    source.buffer.copyToChannel(data, 0); source.connect(graph.input);
    const finished = new Promise(resolve => {source.onended = resolve;});
    source.start(); await finished;
    await new Promise(resolve => setTimeout(resolve, 50));
    const live = new Float32Array(chunks.reduce((sum, chunk) => sum + chunk.length, 0));
    let position = 0;
    for (const chunk of chunks) { live.set(chunk, position); position += chunk.length; }
    const offset = Math.max(0, live.findIndex(value => Math.abs(value) > 1e-7) - 1);
    const expected = exported.getChannelData(0);
    if (live.length - offset < expected.length) throw Error('Live capture missed the tail');
    let error = 0, energy = 0;
    for (let i = 0; i < expected.length; i++) { error += (live[i + offset] - expected[i]) ** 2; energy += expected[i] ** 2; }
    const relativeError = Math.sqrt(error / energy);
    if (relativeError > .01) throw Error(`Live/export samples differ: relative RMS error ${relativeError}`);
    return {result: 'PASS', relativeError, framesCompared: expected.length};
  } finally {
    URL.revokeObjectURL(url); source?.disconnect(); graph?.dispose(); capture?.disconnect(); capture?.port.close();
    await context.close();
  }
})()
