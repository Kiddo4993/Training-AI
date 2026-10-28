// Called by test_web_demo.py: runs the browser demo's JS modules in Node and prints JSON results.
import { readFileSync } from "node:fs";
import { fromBuffer } from "../docs/digitnet.js";
import { SnakeGame, chooseAction } from "../docs/snake.js";

const [, , docs, inputsPath] = process.argv;
const bin = readFileSync(`${docs}/digitnet.bin`);
const net = fromBuffer(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const inputs = JSON.parse(readFileSync(inputsPath, "utf8"));
const logits = inputs.map((px) => Array.from(net.logits(Float32Array.from(px))));

// Seeded random so the snake games are repeatable.
let seed = 42;
const random = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
const q = JSON.parse(readFileSync(`${docs}/q_table.json`, "utf8"));
const play = (table, games) => {
  let total = 0;
  for (let g = 0; g < games; g++) {
    const game = new SnakeGame(random);
    let key = game.reset(), done = false;
    while (!done) { ({ done } = game.step(chooseAction(table, key, random))); key = game.state(); }
    total += game.score;
  }
  return total / games;
};
console.log(JSON.stringify({ logits, trained: play(q, 50), random: play(null, 50) }));
