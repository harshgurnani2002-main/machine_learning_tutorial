import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Layers, Play, RotateCcw, Cpu, Grid3x3, Minimize2, Sliders } from 'lucide-react';

const INPUT_SIZE = 8;
const KERNEL_SIZE = 3;
const OUTPUT_SIZE = INPUT_SIZE - KERNEL_SIZE + 1;
const POOL_SIZE = 2;
const POOLED_SIZE = Math.floor(OUTPUT_SIZE / POOL_SIZE);
const CELL_SIZE = 30;
const KERNEL_CELL = 36;

const COLORS = {
  bg: '#FAF6EE',
  border: '#E5DDD0',
  text: '#2E251E',
  muted: '#6E6257',
  accent: '#B6532B',
  panelBg: '#F4EFE6',
  gold: '#C18C3B',
  cellOn: '#2E251E',
  cellOff: '#FFFFFF',
};

const filters: Record<string, { name: string; matrix: number[][] }> = {
  edge: {
    name: 'Edge Detect',
    matrix: [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
  },
  blur: {
    name: 'Box Blur',
    matrix: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
  },
  sharpen: {
    name: 'Sharpen',
    matrix: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
  },
  custom: {
    name: 'Custom',
    matrix: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
  },
};

const initGrid = (rows: number, cols: number, val = 0) =>
  Array.from({ length: rows }, () => Array(cols).fill(val));

export const CNNSimulator: React.FC = () => {
  const [inputGrid, setInputGrid] = useState<number[][]>(() => initGrid(INPUT_SIZE, INPUT_SIZE));
  const [activeFilter, setActiveFilter] = useState<string>('edge');
  const [customKernel, setCustomKernel] = useState('0,0,0,0,1,0,0,0,0');
  const [outputGrid, setOutputGrid] = useState<number[][] | null>(null);
  const [pooledGrid, setPooledGrid] = useState<number[][] | null>(null);
  const [enablePooling, setEnablePooling] = useState(false);
  const [animating, setAnimating] = useState(false);
  const [animStep, setAnimStep] = useState(-1);
  const [animOutput, setAnimOutput] = useState<number[][]>(() => initGrid(OUTPUT_SIZE, OUTPUT_SIZE));

  const inputRef = useRef<HTMLCanvasElement>(null);
  const outputRef = useRef<HTMLCanvasElement>(null);
  const pooledRef = useRef<HTMLCanvasElement>(null);
  const kernelRef = useRef<HTMLCanvasElement>(null);

  const getKernel = useCallback((): number[][] => {
    if (activeFilter === 'custom') {
      const vals = customKernel.split(',').map(v => parseFloat(v.trim()));
      if (vals.length !== 9 || vals.some(isNaN)) return filters.edge.matrix;
      return [
        [vals[0], vals[1], vals[2]],
        [vals[3], vals[4], vals[5]],
        [vals[6], vals[7], vals[8]],
      ];
    }
    return filters[activeFilter].matrix;
  }, [activeFilter, customKernel]);

  const maxPool = (grid: number[][]) => {
    const result = initGrid(POOLED_SIZE, POOLED_SIZE);
    for (let py = 0; py < POOLED_SIZE; py++) {
      for (let px = 0; px < POOLED_SIZE; px++) {
        let maxVal = -Infinity;
        for (let dy = 0; dy < POOL_SIZE; dy++) {
          for (let dx = 0; dx < POOL_SIZE; dx++) {
            maxVal = Math.max(maxVal, grid[py * POOL_SIZE + dy][px * POOL_SIZE + dx]);
          }
        }
        result[py][px] = maxVal;
      }
    }
    return result;
  };

  const heatColor = (val: number, maxVal: number, r1: number, g1: number, b1: number) => {
    if (maxVal === 0) return COLORS.cellOff;
    const ratio = val / maxVal;
    return `rgb(${Math.round(255 - (255 - r1) * ratio)},${Math.round(255 - (255 - g1) * ratio)},${Math.round(255 - (255 - b1) * ratio)})`;
  };

  useEffect(() => {
    const c = inputRef.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    c.width = INPUT_SIZE * CELL_SIZE;
    c.height = INPUT_SIZE * CELL_SIZE;
    for (let y = 0; y < INPUT_SIZE; y++) {
      for (let x = 0; x < INPUT_SIZE; x++) {
        ctx.fillStyle = inputGrid[y][x] ? COLORS.cellOn : COLORS.cellOff;
        ctx.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
        ctx.strokeStyle = COLORS.border;
        ctx.strokeRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
      }
    }
    if (animating && animStep >= 0) {
      const oy = Math.floor(animStep / OUTPUT_SIZE);
      const ox = animStep % OUTPUT_SIZE;
      ctx.fillStyle = 'rgba(182, 83, 43, 0.2)';
      ctx.fillRect(ox * CELL_SIZE, oy * CELL_SIZE, KERNEL_SIZE * CELL_SIZE, KERNEL_SIZE * CELL_SIZE);
      ctx.strokeStyle = COLORS.accent;
      ctx.lineWidth = 2;
      ctx.strokeRect(ox * CELL_SIZE, oy * CELL_SIZE, KERNEL_SIZE * CELL_SIZE, KERNEL_SIZE * CELL_SIZE);
      ctx.lineWidth = 1;
    }
  }, [inputGrid, animating, animStep]);

  useEffect(() => {
    const c = outputRef.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    c.width = OUTPUT_SIZE * CELL_SIZE;
    c.height = OUTPUT_SIZE * CELL_SIZE;
    let maxVal = 0;
    for (let y = 0; y < OUTPUT_SIZE; y++)
      for (let x = 0; x < OUTPUT_SIZE; x++)
        maxVal = Math.max(maxVal, animOutput[y][x]);
    for (let y = 0; y < OUTPUT_SIZE; y++) {
      for (let x = 0; x < OUTPUT_SIZE; x++) {
        const val = animOutput[y][x];
        ctx.fillStyle = heatColor(val, maxVal, 182, 83, 43);
        ctx.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
        ctx.strokeStyle = COLORS.border;
        ctx.strokeRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
        ctx.fillStyle = maxVal > 0 && val > maxVal * 0.5 ? '#FFFFFF' : COLORS.text;
        ctx.font = 'bold 10px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(val.toFixed(1), x * CELL_SIZE + CELL_SIZE / 2, y * CELL_SIZE + CELL_SIZE / 2);
      }
    }
  }, [animOutput]);

  useEffect(() => {
    const c = pooledRef.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    c.width = POOLED_SIZE * CELL_SIZE;
    c.height = POOLED_SIZE * CELL_SIZE;
    const grid = pooledGrid || [];
    let maxVal = 0;
    for (let y = 0; y < POOLED_SIZE; y++)
      for (let x = 0; x < POOLED_SIZE; x++)
        maxVal = Math.max(maxVal, grid[y]?.[x] ?? 0);
    for (let y = 0; y < POOLED_SIZE; y++) {
      for (let x = 0; x < POOLED_SIZE; x++) {
        const val = grid[y]?.[x] ?? 0;
        ctx.fillStyle = heatColor(val, maxVal, 193, 140, 59);
        ctx.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
        ctx.strokeStyle = COLORS.border;
        ctx.strokeRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
        ctx.fillStyle = maxVal > 0 && val > maxVal * 0.5 ? '#FFFFFF' : COLORS.text;
        ctx.font = 'bold 10px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(val.toFixed(1), x * CELL_SIZE + CELL_SIZE / 2, y * CELL_SIZE + CELL_SIZE / 2);
      }
    }
  }, [pooledGrid]);

  useEffect(() => {
    const c = kernelRef.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    c.width = KERNEL_SIZE * KERNEL_CELL;
    c.height = KERNEL_SIZE * KERNEL_CELL;
    const kernel = getKernel();
    for (let y = 0; y < KERNEL_SIZE; y++) {
      for (let x = 0; x < KERNEL_SIZE; x++) {
        ctx.fillStyle = COLORS.panelBg;
        ctx.fillRect(x * KERNEL_CELL, y * KERNEL_CELL, KERNEL_CELL, KERNEL_CELL);
        ctx.strokeStyle = COLORS.border;
        ctx.strokeRect(x * KERNEL_CELL, y * KERNEL_CELL, KERNEL_CELL, KERNEL_CELL);
        ctx.fillStyle = COLORS.accent;
        ctx.font = 'bold 12px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const v = kernel[y][x];
        ctx.fillText(v % 1 !== 0 ? v.toFixed(2) : v.toFixed(0), x * KERNEL_CELL + KERNEL_CELL / 2, y * KERNEL_CELL + KERNEL_CELL / 2);
      }
    }
  }, [getKernel]);

  const handleInputClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (animating) return;
    const c = inputRef.current;
    if (!c) return;
    const rect = c.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / CELL_SIZE);
    const y = Math.floor((e.clientY - rect.top) / CELL_SIZE);
    if (x >= 0 && x < INPUT_SIZE && y >= 0 && y < INPUT_SIZE) {
      setInputGrid(prev => {
        const g = prev.map(r => [...r]);
        g[y][x] = g[y][x] ? 0 : 1;
        return g;
      });
    }
  };

  const runConvolution = () => {
    if (animating) return;
    const kernel = getKernel();
    const totalSteps = OUTPUT_SIZE * OUTPUT_SIZE;
    setAnimating(true);
    const fresh = initGrid(OUTPUT_SIZE, OUTPUT_SIZE);
    setAnimOutput(fresh);
    setOutputGrid(null);
    setPooledGrid(null);
    let step = 0;
    const partial = initGrid(OUTPUT_SIZE, OUTPUT_SIZE);
    const tick = () => {
      if (step >= totalSteps) {
        setAnimOutput(partial.map(r => [...r]));
        setOutputGrid(partial.map(r => [...r]));
        if (enablePooling) setPooledGrid(maxPool(partial));
        setAnimating(false);
        setAnimStep(-1);
        return;
      }
      const oy = Math.floor(step / OUTPUT_SIZE);
      const ox = step % OUTPUT_SIZE;
      let sum = 0;
      for (let ky = 0; ky < KERNEL_SIZE; ky++)
        for (let kx = 0; kx < KERNEL_SIZE; kx++)
          sum += inputGrid[oy + ky][ox + kx] * kernel[ky][kx];
      partial[oy][ox] = Math.max(0, sum);
      setAnimOutput(partial.map(r => [...r]));
      setAnimStep(step);
      step++;
      setTimeout(tick, 150);
    };
    setTimeout(tick, 300);
  };

  const reset = () => {
    setInputGrid(initGrid(INPUT_SIZE, INPUT_SIZE));
    setOutputGrid(null);
    setPooledGrid(null);
    setAnimOutput(initGrid(OUTPUT_SIZE, OUTPUT_SIZE));
    setAnimating(false);
    setAnimStep(-1);
  };

  const stats = (() => {
    const grid = outputGrid || animOutput;
    let min = Infinity;
    let max = -Infinity;
    for (let y = 0; y < OUTPUT_SIZE; y++)
      for (let x = 0; x < OUTPUT_SIZE; x++) {
        const v = grid[y][x];
        if (v < min) min = v;
        if (v > max) max = v;
      }
    return { min: min === Infinity ? 0 : min, max: max === -Infinity ? 0 : max };
  })();

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-6 h-full bg-[#FAF6EE] overflow-y-auto">
      <div className="lg:col-span-6 space-y-6">
        <div className="bg-[#F4EFE6] border border-[#E5DDD0] p-5 rounded-2xl shadow-xl">
          <h4 className="text-[#2E251E] font-bold text-lg tracking-tight flex items-center gap-2 mb-4">
            <Layers className="w-5 h-5 text-[#B6532B]" /> CNN Simulator
          </h4>
          <p className="text-[#6E6257] text-xs leading-relaxed mb-4">
            Draw a shape on the 8×8 grid, then convolve a filter over it.
          </p>
          <div className="mb-3">
            <label className="text-[10px] text-[#6E6257] font-mono uppercase font-bold block mb-1.5">
              <Sliders className="w-3 h-3 inline mr-1" />Filter Kernel
            </label>
            <div className="grid grid-cols-2 gap-1.5 bg-[#FAF6EE] p-1 rounded-xl border border-[#E5DDD0]">
              {Object.keys(filters).map(type => (
                <button
                  key={type}
                  onClick={() => setActiveFilter(type)}
                  className={`py-1.5 px-2 rounded-lg text-[10px] font-bold transition-all truncate ${
                    activeFilter === type
                      ? 'bg-white shadow-sm text-[#B6532B]'
                      : 'text-[#6E6257] hover:text-[#2E251E]'
                  }`}
                >
                  {filters[type].name}
                </button>
              ))}
            </div>
          </div>
          {activeFilter === 'custom' && (
            <div className="mb-3">
              <label className="text-[10px] text-[#6E6257] font-mono uppercase font-bold block mb-1">3×3 comma-separated</label>
              <input
                type="text"
                value={customKernel}
                onChange={e => setCustomKernel(e.target.value)}
                className="w-full bg-[#FAF6EE] border border-[#E5DDD0] rounded-lg px-2 py-1.5 text-xs font-mono text-[#2E251E] focus:outline-none focus:ring-1 focus:ring-[#B6532B]"
                placeholder="0,0,0,0,1,0,0,0,0"
              />
            </div>
          )}
          <div className="flex items-center gap-2 mb-3">
            <button
              onClick={() => setEnablePooling(!enablePooling)}
              className={`text-[10px] font-bold px-2.5 py-1.5 rounded-lg border transition-all flex items-center gap-1.5 ${
                enablePooling
                  ? 'border-[#C18C3B] bg-[#C18C3B]/10 text-[#C18C3B]'
                  : 'border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257]'
              }`}
            >
              <Minimize2 className="w-3 h-3" /> 2×2 Max Pool
            </button>
          </div>
          <div className="flex gap-2">
            <button
              onClick={runConvolution}
              disabled={animating}
              className="flex-1 flex items-center justify-center gap-1.5 bg-[#B6532B] text-white text-xs font-bold py-2 rounded-xl hover:bg-[#9E4625] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Play className="w-3.5 h-3.5" /> {animating ? 'Animating...' : 'Convolve'}
            </button>
            <button
              onClick={reset}
              disabled={animating}
              className="flex items-center justify-center gap-1.5 bg-[#F4EFE6] border border-[#E5DDD0] text-[#6E6257] text-xs font-bold py-2 px-3 rounded-xl hover:text-[#2E251E] transition-all disabled:opacity-50"
            >
              <RotateCcw className="w-3.5 h-3.5" /> Clear
            </button>
          </div>
          {outputGrid && (
            <div className="mt-3 bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-3 text-[10px] font-mono">
              <div className="flex justify-between items-center">
                <span className="text-[#2E251E] font-bold">{filters[activeFilter].name}</span>
                <span className="text-[#6E6257]">{OUTPUT_SIZE}×{OUTPUT_SIZE} output</span>
              </div>
              <div className="flex justify-between mt-1">
                <span className="text-[#6E6257]">Min: <span className="text-[#B6532B] font-bold">{stats.min.toFixed(1)}</span></span>
                <span className="text-[#6E6257]">Max: <span className="text-[#B6532B] font-bold">{stats.max.toFixed(1)}</span></span>
              </div>
              {enablePooling && pooledGrid && (
                <div className="flex justify-between mt-1 pt-1 border-t border-[#E5DDD0]">
                  <span className="text-[#6E6257]">Pooled: <span className="text-[#C18C3B] font-bold">{POOLED_SIZE}×{POOLED_SIZE}</span></span>
                </div>
              )}
            </div>
          )}
        </div>
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-sm flex flex-col items-center">
          <h5 className="text-xs font-mono font-bold text-[#2E251E] mb-3 tracking-wider flex items-center gap-1.5">
            <Grid3x3 className="w-4 h-4 text-[#B6532B]" /> Input Image (8×8)
          </h5>
          <canvas
            ref={inputRef}
            onClick={handleInputClick}
            className="cursor-pointer border-2 border-[#E5DDD0] rounded-lg"
          />
          <p className="text-[10px] text-[#6E6257] mt-2">Click cells to toggle black/white</p>
        </div>
      </div>
      <div className="lg:col-span-6 space-y-6">
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-sm flex flex-col items-center">
          <h5 className="text-xs font-mono font-bold text-[#2E251E] mb-3 tracking-wider flex items-center gap-1.5">
            <Cpu className="w-4 h-4 text-[#B6532B]" /> Feature Map ({OUTPUT_SIZE}×{OUTPUT_SIZE}) — ReLU
          </h5>
          <canvas ref={outputRef} className="border-2 border-[#E5DDD0] rounded-lg" />
          {!outputGrid && !animating && (
            <p className="text-[10px] text-[#B6532B] mt-2 font-bold">Click «Convolve»</p>
          )}
        </div>
        {enablePooling && (
          <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-sm flex flex-col items-center">
            <h5 className="text-xs font-mono font-bold text-[#2E251E] mb-3 tracking-wider flex items-center gap-1.5">
              <Minimize2 className="w-4 h-4 text-[#C18C3B]" /> Max Pooled ({POOLED_SIZE}×{POOLED_SIZE})
            </h5>
            <canvas ref={pooledRef} className="border-2 border-[#E5DDD0] rounded-lg" />
          </div>
        )}
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-sm flex flex-col items-center">
          <h5 className="text-xs font-mono font-bold text-[#2E251E] mb-3 tracking-wider">Kernel (3×3) — {filters[activeFilter].name}</h5>
          <canvas ref={kernelRef} className="rounded-lg" />
        </div>
      </div>
    </div>
  );
};
