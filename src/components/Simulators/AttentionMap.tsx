import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Sparkles, BookOpen, Type, Layers, Scale, Send } from 'lucide-react';

function hashString(str: string): number {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash) + str.charCodeAt(i);
    hash = hash & hash;
  }
  return hash >>> 0;
}

function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = s + 0x6D2B79F5 | 0;
    let t = Math.imul(s ^ s >>> 15, 1 | s);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function softmax(arr: number[]): number[] {
  const max = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

function interpolateColor(t: number): string {
  const r = Math.round(255 - t * (255 - 182));
  const g = Math.round(255 - t * (255 - 83));
  const b = Math.round(255 - t * (255 - 43));
  return `rgb(${r},${g},${b})`;
}

function generateVectors(tokens: string[], dim: number) {
  const genVec = (token: string, kind: string, d: number) => {
    const seed = hashString(token + "_" + kind);
    const rng = mulberry32(seed);
    return Array.from({ length: d }, () => (rng() - 0.5) * 2);
  };
  return {
    Q: tokens.map(t => genVec(t, "Q", dim)),
    K: tokens.map(t => genVec(t, "K", dim)),
    V: tokens.map(t => genVec(t, "V", dim))
  };
}

function computeAttentionMatrices(
  Q: number[][], K: number[][],
  numHeads: number, headDim: number, useScale: boolean
): number[][][] {
  const n = Q.length;
  const result: number[][][] = [];
  for (let h = 0; h < numHeads; h++) {
    const hStart = h * headDim;
    const matrix: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      for (let j = 0; j < n; j++) {
        let dot = 0;
        for (let d = 0; d < headDim; d++) {
          dot += Q[i][hStart + d] * K[j][hStart + d];
        }
        if (useScale) dot /= Math.sqrt(headDim);
        row.push(dot);
      }
      matrix.push(softmax(row));
    }
    result.push(matrix);
  }
  return result;
}

function computeOutputs(
  attentionMatrices: number[][][],
  V: number[][], numHeads: number, headDim: number, dim: number
): number[][] {
  const n = V.length;
  const output: number[][] = [];
  for (let i = 0; i < n; i++) {
    const vec = new Array(dim).fill(0);
    for (let h = 0; h < numHeads; h++) {
      const hStart = h * headDim;
      const attn = attentionMatrices[h];
      for (let j = 0; j < n; j++) {
        const w = attn[i][j];
        for (let d = 0; d < headDim; d++) {
          vec[hStart + d] += w * V[j][hStart + d];
        }
      }
    }
    output.push(vec);
  }
  return output;
}

interface CellInfo {
  head: number;
  row: number;
  col: number;
  value: number;
}

export const AttentionMap: React.FC = () => {
  const presets = [
    "The cat sat on the mat",
    "Attention is all you need",
    "I love machine learning",
    "The dog chased the ball"
  ];

  const [sentence, setSentence] = useState(presets[0]);
  const [customInput, setCustomInput] = useState("");
  const [numHeads, setNumHeads] = useState<1 | 4>(1);
  const [useScale, setUseScale] = useState(true);
  const [hoveredCell, setHoveredCell] = useState<CellInfo | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  const canvasRef = useRef<HTMLCanvasElement>(null);

  const dim = 8;

  const tokens = useMemo(() => {
    return sentence.trim().split(/\s+/).filter(Boolean);
  }, [sentence]);

  const headDim = dim / numHeads;

  const vectors = useMemo(() => generateVectors(tokens, dim), [tokens, dim]);

  const attentionMatrices = useMemo(
    () => computeAttentionMatrices(vectors.Q, vectors.K, numHeads, headDim, useScale),
    [vectors, numHeads, headDim, useScale]
  );

  const outputs = useMemo(
    () => computeOutputs(attentionMatrices, vectors.V, numHeads, headDim, dim),
    [attentionMatrices, vectors.V, numHeads, headDim, dim]
  );

  const isValidSentence = tokens.length >= 1 && tokens.length <= 10;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const cw = 400;
    const ch = 320;
    canvas.width = cw;
    canvas.height = ch;

    ctx.clearRect(0, 0, cw, ch);
    ctx.fillStyle = '#FAF6EE';
    ctx.fillRect(0, 0, cw, ch);

    const n = tokens.length;
    if (n === 0) return;

    const drawHeatmap = (
      matrix: number[][], tkns: string[],
      ox: number, oy: number, w: number, h: number,
      label: string
    ) => {
      const m = tkns.length;
      const lm = label ? 24 : 20;
      const tm = label ? 22 : 18;
      const gw = w - lm - 4;
      const gh = h - tm - 4;
      const cw = gw / m;
      const ch = gh / m;

      for (let i = 0; i < m; i++) {
        for (let j = 0; j < m; j++) {
          const val = matrix[i][j];
          const x = ox + lm + j * cw;
          const y = oy + tm + i * ch;
          ctx.fillStyle = interpolateColor(val);
          ctx.fillRect(x, y, cw, ch);
          ctx.strokeStyle = '#E5DDD0';
          ctx.lineWidth = 0.5;
          ctx.strokeRect(x, y, cw, ch);
        }
      }

      const fontSize = Math.max(6.5, Math.min(9, cw * 0.75));
      ctx.fillStyle = '#2E251E';
      ctx.font = `bold ${fontSize}px monospace`;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (let i = 0; i < m; i++) {
        ctx.fillText(tkns[i].slice(0, 4), ox + lm - 3, oy + tm + i * ch + ch / 2);
      }

      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      for (let j = 0; j < m; j++) {
        ctx.fillText(tkns[j].slice(0, 4), ox + lm + j * cw + cw / 2, oy + tm - 3);
      }

      if (label) {
        ctx.fillStyle = '#6E6257';
        ctx.font = 'bold 7px monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(label, ox + 3, oy + 3);
      }
    };

    if (numHeads === 1) {
      drawHeatmap(attentionMatrices[0], tokens, 0, 0, cw, ch, "");
    } else {
      const hw = cw / 2;
      const hh = ch / 2;
      for (let h = 0; h < 4; h++) {
        const col = h % 2;
        const row = Math.floor(h / 2);
        drawHeatmap(
          attentionMatrices[h], tokens,
          col * hw, row * hh, hw, hh,
          `H${h + 1}`
        );
      }
    }
  }, [attentionMatrices, tokens, numHeads]);

  const getCellFromPos = (cx: number, cy: number): CellInfo | null => {
    const n = tokens.length;
    if (n === 0) return null;

    const cw = 400;
    const ch = 320;

    if (numHeads === 1) {
      const lm = 20;
      const tm = 18;
      const gw = cw - lm - 4;
      const gh = ch - tm - 4;
      const cellW = gw / n;
      const cellH = gh / n;

      const j = Math.floor((cx - lm) / cellW);
      const i = Math.floor((cy - tm) / cellH);
      if (i < 0 || i >= n || j < 0 || j >= n) return null;
      return { head: 1, row: i, col: j, value: attentionMatrices[0][i][j] };
    }

    const hw = cw / 2;
    const hh = ch / 2;
    const hx = Math.floor(cx / hw);
    const hy = Math.floor(cy / hh);
    if (hx < 0 || hx > 1 || hy < 0 || hy > 1) return null;
    const headIdx = hy * 2 + hx;

    const ox = hx * hw;
    const oy = hy * hh;
    const lm = 24;
    const tm = 22;
    const gw = hw - lm - 4;
    const gh = hh - tm - 4;
    const cellW = gw / n;
    const cellH = gh / n;

    const j = Math.floor((cx - ox - lm) / cellW);
    const i = Math.floor((cy - oy - tm) / cellH);
    if (i < 0 || i >= n || j < 0 || j >= n) return null;
    return { head: headIdx + 1, row: i, col: j, value: attentionMatrices[headIdx][i][j] };
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width;
    const sy = canvas.height / rect.height;
    const cx = (e.clientX - rect.left) * sx;
    const cy = (e.clientY - rect.top) * sy;
    const cell = getCellFromPos(cx, cy);
    if (cell) {
      setHoveredCell(cell);
      setTooltipPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    } else {
      setHoveredCell(null);
    }
  };

  const handleCanvasMouseLeave = () => {
    setHoveredCell(null);
  };

  const handleCustomSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = customInput.trim();
    if (trimmed.length > 0) {
      const words = trimmed.split(/\s+/);
      if (words.length >= 1 && words.length <= 10) {
        setSentence(trimmed);
      }
    }
  };

  const handleCustomKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleCustomSubmit(e);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-6 h-full bg-[#FAF6EE] overflow-y-auto">
      {/* Left Panel: Controls */}
      <div className="lg:col-span-5 bg-[#F4EFE6] border border-[#E5DDD0] p-5 rounded-2xl flex flex-col justify-between">
        <div>
          <h4 className="text-[#2E251E] font-bold text-base mb-1 tracking-wide flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-[#B6532B]" /> Scaled Dot-Product Attention
          </h4>
          <p className="text-[#6E6257] text-xs mb-5">
            Real Q·K matching with softmax normalization.
          </p>

          <div className="space-y-4">
            {/* Preset sentences */}
            <div>
              <label className="text-[10px] text-[#6E6257] font-mono block mb-1.5 uppercase font-bold">
                <BookOpen className="w-3 h-3 inline mr-1" /> Preset Sentences
              </label>
              <div className="space-y-1.5">
                {presets.map((sent, idx) => (
                  <button
                    key={idx}
                    onClick={() => { setSentence(sent); setCustomInput(""); }}
                    className={`w-full text-left p-2.5 rounded-xl border text-xs font-mono transition-all leading-normal cursor-pointer ${
                      sentence === sent && customInput === ""
                        ? 'border-[#B6532B] bg-[#B6532B]/10 text-[#2E251E] font-bold'
                        : 'border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257] hover:text-[#2E251E] hover:bg-[#F4EFE6]'
                    }`}
                  >
                    {sent}
                  </button>
                ))}
              </div>
            </div>

            {/* Custom sentence input */}
            <div>
              <label className="text-[10px] text-[#6E6257] font-mono block mb-1.5 uppercase font-bold">
                <Type className="w-3 h-3 inline mr-1" /> Custom Sentence
              </label>
              <form onSubmit={handleCustomSubmit} className="flex gap-2">
                <input
                  type="text"
                  value={customInput}
                  onChange={e => setCustomInput(e.target.value)}
                  onKeyDown={handleCustomKeyDown}
                  placeholder="Type 1-10 words..."
                  className="flex-1 bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl px-3 py-2 text-xs font-mono text-[#2E251E] placeholder:text-[#CFC5B4] focus:outline-none focus:border-[#B6532B]"
                />
                <button
                  type="submit"
                  className="px-3 py-2 bg-[#B6532B] text-white rounded-xl hover:bg-[#9E4521] transition-colors cursor-pointer"
                >
                  <Send className="w-4 h-4" />
                </button>
              </form>
              {!isValidSentence && customInput.trim().length > 0 && (
                <p className="text-[#B6532B] text-[10px] font-mono mt-1">Enter 1 to 10 words</p>
              )}
            </div>

            {/* Head toggle */}
            <div>
              <label className="text-[10px] text-[#6E6257] font-mono block mb-1.5 uppercase font-bold">
                <Layers className="w-3 h-3 inline mr-1" /> Attention Heads
              </label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setNumHeads(1)}
                  className={`py-2 rounded-xl text-xs font-mono border transition-all cursor-pointer ${
                    numHeads === 1
                      ? 'border-[#B6532B] bg-[#B6532B]/15 text-[#B6532B] font-semibold'
                      : 'border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257]'
                  }`}
                >
                  Single Head
                </button>
                <button
                  onClick={() => setNumHeads(4)}
                  className={`py-2 rounded-xl text-xs font-mono border transition-all cursor-pointer ${
                    numHeads === 4
                      ? 'border-[#B6532B] bg-[#B6532B]/15 text-[#B6532B] font-semibold'
                      : 'border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257]'
                  }`}
                >
                  4 Heads
                </button>
              </div>
            </div>

            {/* Scale toggle */}
            <div>
              <label className="text-[10px] text-[#6E6257] font-mono block mb-1.5 uppercase font-bold">
                <Scale className="w-3 h-3 inline mr-1" /> Scaling
              </label>
              <label className="flex items-center gap-2.5 p-2.5 bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl cursor-pointer">
                <input
                  type="checkbox"
                  checked={useScale}
                  onChange={e => setUseScale(e.target.checked)}
                  className="accent-[#B6532B] w-4 h-4"
                />
                <span className="text-xs font-mono text-[#2E251E]">
                  Apply {"\u221A"}d<sub>k</sub> scaling
                </span>
              </label>
              <p className="text-[10px] text-[#6E6257] font-mono mt-1 ml-1">
                {useScale
                  ? "Dividing by \u221Ad\u2096 prevents softmax saturation"
                  : "Without scaling, logits grow large and softmax becomes peaked"
                }
              </p>
            </div>
          </div>
        </div>

        {/* Info box */}
        <div className="p-3 bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl flex gap-2.5 items-start text-xs font-mono leading-normal pt-4 mt-4 text-[#2E251E]">
          <Sparkles className="w-4 h-4 text-[#B6532B] shrink-0 mt-0.5" />
          <div>
            <span className="font-bold block mb-0.5">How it works:</span>
            <span className="text-[#6E6257] text-[11px]">
              Each token has random Q/K/V vectors (dim={dim}) seeded by a hash of the word. 
              Attention = softmax(Q·Kᵀ / {"\u221A"}d<sub>k</sub>). 
              Same sentence always produces the same matrix.
            </span>
          </div>
        </div>
      </div>

      {/* Right Panel: Canvas + Output */}
      <div className="lg:col-span-7 space-y-5 flex flex-col">
        {/* Canvas */}
        <div className="bg-[#F4EFE6] border border-[#E5DDD0] p-4 rounded-2xl flex flex-col items-center shadow-inner relative">
          <div className="relative inline-block">
            <canvas
              ref={canvasRef}
              className="rounded-xl border border-[#E5DDD0] bg-white"
              style={{ width: 400, height: 320, cursor: hoveredCell ? 'crosshair' : 'default' }}
              onMouseMove={handleCanvasMouseMove}
              onMouseLeave={handleCanvasMouseLeave}
            />
            {hoveredCell && (
              <div
                className="absolute pointer-events-none bg-[#2E251E] text-[#FAF6EE] text-[10px] font-mono px-2 py-1 rounded-lg border border-[#E5DDD0]/20 whitespace-nowrap z-10"
                style={{
                  left: tooltipPos.x + 12,
                  top: tooltipPos.y - 20,
                }}
              >
                {numHeads > 1 && <>H{hoveredCell.head}: </>}
                &ldquo;{tokens[hoveredCell.row]}&rdquo; &rarr; &ldquo;{tokens[hoveredCell.col]}&rdquo; = {hoveredCell.value.toFixed(4)}
              </div>
            )}
          </div>
          <div className="mt-2 text-[9px] font-mono text-[#6E6257]">
            Hover over a cell to see the attention weight
          </div>
        </div>

        {/* Output table */}
        <div className="bg-[#F4EFE6] border border-[#E5DDD0] p-4 rounded-2xl">
          <h5 className="text-[#2E251E] text-xs font-semibold font-mono mb-3 flex items-center gap-1.5">
            <BookOpen className="w-4 h-4 text-[#B6532B]" /> Attention Output (first 3 of {dim} dims)
          </h5>
          <div className="overflow-x-auto">
            <table className="w-full text-[10px] font-mono border-collapse">
              <thead>
                <tr className="border-b border-[#E5DDD0] text-[#6E6257]">
                  <th className="text-left py-1.5 pr-3 font-semibold">Token</th>
                  <th className="text-right py-1.5 px-2 font-semibold">v[0]</th>
                  <th className="text-right py-1.5 px-2 font-semibold">v[1]</th>
                  <th className="text-right py-1.5 px-2 font-semibold">v[2]</th>
                </tr>
              </thead>
              <tbody>
                {outputs.map((vec, i) => (
                  <tr key={i} className="border-b border-[#E5DDD0]/50 text-[#2E251E]">
                    <td className="py-1.5 pr-3 font-bold truncate max-w-[80px]">{tokens[i]}</td>
                    <td className="text-right py-1.5 px-2 text-[#B6532B]">{vec[0].toFixed(4)}</td>
                    <td className="text-right py-1.5 px-2">{vec[1].toFixed(4)}</td>
                    <td className="text-right py-1.5 px-2 text-[#6E6257]">{vec[2].toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
