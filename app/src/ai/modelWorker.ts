import * as ort from "onnxruntime-web";

// Single-threaded WASM — works on GitHub Pages (no COOP/COEP headers needed)
ort.env.wasm.numThreads = 1;

let session: ort.InferenceSession | null = null;
let loadingUrl: string | null = null;
let loadPromise: Promise<void> | null = null;

async function ensureLoaded(modelUrl: string): Promise<void> {
  if (session && loadingUrl === modelUrl) return;
  if (loadPromise && loadingUrl === modelUrl) return loadPromise;

  loadingUrl = modelUrl;
  loadPromise = ort.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"],
  }).then((s) => {
    session = s;
  });

  return loadPromise;
}

async function runInference(input: Float32Array, boardSize: number): Promise<Float32Array> {
  const N = boardSize + 2;
  const tensor = new ort.Tensor("float32", input, [1, 2, N, N]);
  const results = await session!.run({ [session!.inputNames[0]]: tensor });
  return results[session!.outputNames[0]].data as Float32Array;
}

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;

  if (msg.type === "INFER_PAIR") {
    try {
      await ensureLoaded(msg.modelUrl);
      // ORT WASM backend is single-threaded — run sequentially, not in parallel
      const out1 = await runInference(msg.input1, msg.boardSize);
      const out2 = await runInference(msg.input2, msg.boardSize);
      // Transfer buffers to avoid copying
      const out1Copy = new Float32Array(out1);
      const out2Copy = new Float32Array(out2);
      self.postMessage(
        { type: "RESULT_PAIR", out1: out1Copy, out2: out2Copy },
        { transfer: [out1Copy.buffer, out2Copy.buffer] }
      );
    } catch (err) {
      self.postMessage({ type: "ERROR", message: String(err) });
    }
  }
};
