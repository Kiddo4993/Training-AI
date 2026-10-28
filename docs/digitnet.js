// The digit recognizer from train.py, re-implemented in plain JavaScript.
// Same layers, same trained weights (exported by scripts/export_web_demo.py), no libraries.
//
//   1x28x28 → conv3x3(32) → ReLU → maxpool → conv3x3(64) → ReLU → maxpool → linear(128) → ReLU → linear(10)

const SHAPES = [
  [32, 1, 3, 3], [32],      // features.0: first conv layer
  [64, 32, 3, 3], [64],     // features.3: second conv layer
  [128, 3136], [128],       // classifier.1: hidden linear layer
  [10, 128], [10],          // classifier.4: one score per digit
];

export async function loadDigitNet(url) {
  const buffer = await (await fetch(url)).arrayBuffer();
  return fromBuffer(buffer);
}

export function fromBuffer(buffer) {
  const all = new Float32Array(buffer);
  const tensors = [];
  let offset = 0;
  for (const shape of SHAPES) {
    const size = shape.reduce((a, b) => a * b, 1);
    tensors.push(all.subarray(offset, offset + size));
    offset += size;
  }
  if (offset !== all.length) throw new Error(`expected ${offset} weights, got ${all.length}`);
  const [c1w, c1b, c2w, c2b, f1w, f1b, f2w, f2b] = tensors;

  // 3x3 convolution with padding 1 (output keeps the input's height and width), then ReLU.
  function convRelu(input, inC, size, weight, bias, outC) {
    const out = new Float32Array(outC * size * size);
    for (let o = 0; o < outC; o++) {
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          let sum = bias[o];
          for (let i = 0; i < inC; i++) {
            for (let ky = 0; ky < 3; ky++) {
              const iy = y + ky - 1;
              if (iy < 0 || iy >= size) continue;
              for (let kx = 0; kx < 3; kx++) {
                const ix = x + kx - 1;
                if (ix < 0 || ix >= size) continue;
                sum += input[(i * size + iy) * size + ix] * weight[((o * inC + i) * 3 + ky) * 3 + kx];
              }
            }
          }
          out[(o * size + y) * size + x] = sum > 0 ? sum : 0;
        }
      }
    }
    return out;
  }

  // 2x2 max pooling: keep the strongest signal in each 2x2 patch, halving height and width.
  function maxPool(input, channels, size) {
    const half = size / 2;
    const out = new Float32Array(channels * half * half);
    for (let c = 0; c < channels; c++) {
      for (let y = 0; y < half; y++) {
        for (let x = 0; x < half; x++) {
          const at = (yy, xx) => input[(c * size + yy) * size + xx];
          out[(c * half + y) * half + x] = Math.max(
            at(2 * y, 2 * x), at(2 * y, 2 * x + 1), at(2 * y + 1, 2 * x), at(2 * y + 1, 2 * x + 1));
        }
      }
    }
    return out;
  }

  function linear(input, weight, bias, outN, relu) {
    const inN = input.length;
    const out = new Float32Array(outN);
    for (let o = 0; o < outN; o++) {
      let sum = bias[o];
      for (let i = 0; i < inN; i++) sum += weight[o * inN + i] * input[i];
      out[o] = relu && sum < 0 ? 0 : sum;
    }
    return out;
  }

  /** pixels: Float32Array(784), already normalized like train.py. Returns 10 raw scores (logits). */
  function logits(pixels) {
    let x = convRelu(pixels, 1, 28, c1w, c1b, 32);
    x = maxPool(x, 32, 28);
    x = convRelu(x, 32, 14, c2w, c2b, 64);
    x = maxPool(x, 64, 14);                 // 64 x 7 x 7 = 3136, flattened in the same order as PyTorch
    x = linear(x, f1w, f1b, 128, true);
    return linear(x, f2w, f2b, 10, false);  // dropout only matters during training, so it's skipped here
  }

  /** Same as logits(), turned into probabilities that add up to 1 (softmax). */
  function predict(pixels) {
    const z = logits(pixels);
    const max = Math.max(...z);
    const exps = Array.from(z, (v) => Math.exp(v - max));
    const total = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / total);
  }

  return { logits, predict };
}

/**
 * Turn a drawing (white strokes on a transparent canvas) into the 784 numbers the network expects,
 * mirroring to_mnist_style() in finetune.py: crop to the ink, fit it in a 20x20 box, centre it by
 * centre of mass on a 28x28 canvas, then normalize with MNIST's mean and std.
 * Returns null for a blank canvas. Needs a browser (uses canvas).
 */
export function canvasToPixels(source) {
  const { width, height } = source;
  const ink = source.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, width, height).data;
  let x0 = width, y0 = height, x1 = -1, y1 = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (ink[(y * width + x) * 4 + 3] > 0) {
        if (x < x0) x0 = x; if (x > x1) x1 = x;
        if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
    }
  }
  if (x1 < 0) return null;

  const cropW = x1 - x0 + 1, cropH = y1 - y0 + 1;
  const scale = 20 / Math.max(cropW, cropH);
  const w = Math.max(1, Math.round(cropW * scale)), h = Math.max(1, Math.round(cropH * scale));
  const small = document.createElement("canvas");
  small.width = w; small.height = h;
  const sctx = small.getContext("2d", { willReadFrequently: true });
  sctx.imageSmoothingQuality = "high";
  sctx.drawImage(source, x0, y0, cropW, cropH, 0, 0, w, h);

  const alpha = sctx.getImageData(0, 0, w, h).data;
  let mass = 0, mx = 0, my = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const a = alpha[(y * w + x) * 4 + 3];
      mass += a; mx += a * x; my += a * y;
    }
  }
  const dx = Math.round(14 - mx / mass), dy = Math.round(14 - my / mass);

  const pixels = new Float32Array(784);
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const sx = x - dx, sy = y - dy;
      const a = sx >= 0 && sx < w && sy >= 0 && sy < h ? alpha[(sy * w + sx) * 4 + 3] / 255 : 0;
      pixels[y * 28 + x] = (a - 0.1307) / 0.3081;
    }
  }
  return pixels;
}
