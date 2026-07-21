import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';

type SearchMethod = 'grid' | 'random' | 'bayesian';
type SurfaceMode = 'simple' | 'complex' | 'ridge';

interface EvalPoint {
  x: number;
  y: number;
  score: number;
  order: number;
}

function computeSurface(x: number, y: number, mode: SurfaceMode): number {
  if (mode === 'simple') {
    const dx = x - 0.35, dy = y - 0.65;
    return Math.exp(-(dx * dx + dy * dy) / 0.025);
  }
  if (mode === 'complex') {
    const p1x = x - 0.25, p1y = y - 0.70;
    const p2x = x - 0.70, p2y = y - 0.30;
    const v1 = Math.exp(-(p1x * p1x + p1y * p1y) / 0.015);
    const v2 = Math.exp(-(p2x * p2x + p2y * p2y) / 0.015);
    return 0.95 * Math.max(v1, v2) + 0.05 * Math.min(v1, v2);
  }
  const d = y - (0.5 + 0.3 * x);
  const a = x - 0.5;
  return Math.exp(-(d * d / 0.003 + a * a / 0.2));
}

function getTrueOptimum(mode: SurfaceMode): { x: number; y: number; score: number } {
  const fixed: Record<SurfaceMode, { x: number; y: number; score: number }> = {
    simple: { x: 0.35, y: 0.65, score: 1.0 },
    complex: { x: 0.25, y: 0.70, score: 0.95 },
    ridge: { x: 0.50, y: 0.65, score: 1.0 },
  };
  return fixed[mode];
}

function heatmapRGB(val: number): [number, number, number] {
  const stops: { t: number; r: number; g: number; b: number }[] = [
    { t: 0.00, r: 244, g: 239, b: 230 },
    { t: 0.25, r: 229, g: 221, b: 208 },
    { t: 0.50, r: 183, g: 140, b: 59 },
    { t: 0.75, r: 160, g: 73, b: 38 },
    { t: 1.00, r: 46, g: 37, b: 30 },
  ];
  let lo = stops[0], hi = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (val >= stops[i].t && val <= stops[i + 1].t) {
      lo = stops[i]; hi = stops[i + 1]; break;
    }
  }
  const t = Math.max(0, Math.min(1, (val - lo.t) / (hi.t - lo.t)));
  return [
    Math.round(lo.r + t * (hi.r - lo.r)),
    Math.round(lo.g + t * (hi.g - lo.g)),
    Math.round(lo.b + t * (hi.b - lo.b)),
  ];
}

function generateGrid(nEvals: number, mode: SurfaceMode): EvalPoint[] {
  const n = Math.max(2, Math.floor(Math.sqrt(nEvals)));
  const pts: EvalPoint[] = [];
  let order = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const x = (i + 0.5) / n;
      const y = (j + 0.5) / n;
      pts.push({ x, y, score: computeSurface(x, y, mode), order: order++ });
    }
  }
  return pts;
}

function generateRandom(nEvals: number, mode: SurfaceMode): EvalPoint[] {
  const pts: EvalPoint[] = [];
  for (let i = 0; i < nEvals; i++) {
    const x = Math.random();
    const y = Math.random();
    pts.push({ x, y, score: computeSurface(x, y, mode), order: i });
  }
  return pts;
}

function generateBayesian(nEvals: number, mode: SurfaceMode): EvalPoint[] {
  const pts: EvalPoint[] = [];
  const nInit = Math.max(3, Math.min(6, Math.floor(nEvals * 0.15)));
  for (let i = 0; i < nInit; i++) {
    const x = Math.random();
    const y = Math.random();
    pts.push({ x, y, score: computeSurface(x, y, mode), order: i });
  }
  for (let i = nInit; i < nEvals; i++) {
    let best = pts[0];
    for (const p of pts) if (p.score > best.score) best = p;
    const progress = (i - nInit) / Math.max(1, nEvals - nInit);
    const std = 0.15 * (1 - progress * 0.7) + 0.02;
    const u1 = Math.random(), u2 = Math.random();
    const z1 = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
    const z2 = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.sin(2 * Math.PI * u2);
    let x = best.x + z1 * std;
    let y = best.y + z2 * std;
    if (Math.random() < 0.15 * (1 - progress)) {
      x = Math.random();
      y = Math.random();
    }
    pts.push({
      x: Math.max(0, Math.min(1, x)),
      y: Math.max(0, Math.min(1, y)),
      score: computeSurface(x, y, mode),
      order: i,
    });
  }
  return pts;
}

export const HyperparamTuningSimulator: React.FC = () => {
  const [searchMethod, setSearchMethod] = useState<SearchMethod>('grid');
  const [nEvals, setNEvals] = useState(30);
  const [surfaceMode, setSurfaceMode] = useState<SurfaceMode>('simple');
  const [showOptimum, setShowOptimum] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [displayCount, setDisplayCount] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const timerRef = useRef<number | null>(null);

  const allPoints = useMemo(() => {
    if (searchMethod === 'grid') return generateGrid(nEvals, surfaceMode);
    if (searchMethod === 'random') return generateRandom(nEvals, surfaceMode);
    return generateBayesian(nEvals, surfaceMode);
  }, [searchMethod, nEvals, surfaceMode]);

  const trueOpt = useMemo(() => getTrueOptimum(surfaceMode), [surfaceMode]);

  const visiblePoints = useMemo(() => allPoints.slice(0, displayCount), [allPoints, displayCount]);

  const bestPoint = useMemo(() => {
    if (visiblePoints.length === 0) return null;
    let best = visiblePoints[0];
    for (const p of visiblePoints) if (p.score > best.score) best = p;
    return best;
  }, [visiblePoints]);

  const regret = useMemo(() => {
    if (visiblePoints.length === 0) return null;
    return trueOpt.score - bestPoint!.score;
  }, [visiblePoints, trueOpt, bestPoint]);

  const simulatedTime = useMemo(() => {
    if (visiblePoints.length === 0) return 0;
    return searchMethod === 'grid'
      ? visiblePoints.length * 0.05
      : searchMethod === 'random'
        ? visiblePoints.length * 0.05
        : visiblePoints.length * 0.3;
  }, [visiblePoints, searchMethod]);

  useEffect(() => {
    if (!isRunning) return;
    if (displayCount >= allPoints.length) {
      setIsRunning(false);
      return;
    }
    timerRef.current = window.setTimeout(() => {
      setDisplayCount(d => d + 1);
    }, 150 + Math.random() * 100);
    return () => { if (timerRef.current !== null) clearTimeout(timerRef.current); };
  }, [isRunning, displayCount, allPoints.length]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const W = canvas.width = 540;
    const H = canvas.height = 480;

    const imgData = ctx.createImageData(W, H);
    for (let py = 0; py < H; py++) {
      for (let px = 0; px < W; px++) {
        const x = px / W;
        const y = 1 - py / H;
        const v = computeSurface(x, y, surfaceMode);
        const [r, g, b] = heatmapRGB(v);
        const idx = (py * W + px) * 4;
        imgData.data[idx] = r;
        imgData.data[idx + 1] = g;
        imgData.data[idx + 2] = b;
        imgData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);

    ctx.strokeStyle = 'rgba(46, 37, 30, 0.08)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 10; i++) {
      const px = (i / 10) * W;
      ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke();
      const py = (i / 10) * H;
      ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(W, py); ctx.stroke();
    }

    if (showOptimum) {
      const ox = trueOpt.x * W;
      const oy = (1 - trueOpt.y) * H;
      ctx.save();
      ctx.translate(ox, oy);
      ctx.fillStyle = '#2E251E';
      ctx.font = 'bold 22px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('★', 0, 0);
      ctx.restore();
    }

    const isGrid = searchMethod === 'grid';
    const ptRadius = isGrid ? 4 : 5;

    visiblePoints.forEach((p, i) => {
      const cx = p.x * W;
      const cy = (1 - p.y) * H;
      const alpha = isGrid ? 0.7 : 0.6 + 0.4 * (i / Math.max(1, visiblePoints.length - 1));
      const isBest = bestPoint && p.x === bestPoint.x && p.y === bestPoint.y;

      if (isBest && searchMethod !== 'grid') {
        ctx.beginPath();
        ctx.arc(cx, cy, ptRadius + 4, 0, Math.PI * 2);
        ctx.strokeStyle = '#2E251E';
        ctx.lineWidth = 2.5;
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(cx, cy, ptRadius + 7, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(46, 37, 30, 0.25)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      ctx.beginPath();
      ctx.arc(cx, cy, ptRadius, 0, Math.PI * 2);
      if (isGrid) {
        ctx.fillStyle = `rgba(244, 239, 230, ${alpha})`;
        ctx.strokeStyle = 'rgba(46, 37, 30, 0.5)';
        ctx.lineWidth = 1;
        ctx.fill();
        ctx.stroke();
      } else if (searchMethod === 'random') {
        const isBestR = bestPoint && p.x === bestPoint.x && p.y === bestPoint.y;
        ctx.fillStyle = isBestR ? '#2E251E' : `rgba(182, 83, 43, ${alpha})`;
        ctx.fill();
      } else {
        const hue = 30 + (1 - alpha) * 20;
        const sat = 40 + alpha * 30;
        ctx.fillStyle = `hsla(${hue}, ${sat}%, ${50 + alpha * 20}%, 0.85)`;
        ctx.strokeStyle = 'rgba(46, 37, 30, 0.4)';
        ctx.lineWidth = 0.8;
        ctx.fill();
        ctx.stroke();
      }
    });

    ctx.fillStyle = '#2E251E';
    ctx.font = 'bold 10px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('Accuracy →', 10, 16);
    const barW = 100, barH = 10, barX = W - barW - 10, barY = 8;
    for (let px = 0; px < barW; px++) {
      const v = px / barW;
      const [r, g, b] = heatmapRGB(v);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(barX + px, barY, 1, barH);
    }
    ctx.strokeStyle = '#2E251E';
    ctx.lineWidth = 0.5;
    ctx.strokeRect(barX, barY, barW, barH);
    ctx.fillStyle = '#2E251E';
    ctx.font = '8px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('0', barX, barY + barH + 12);
    ctx.textAlign = 'right';
    ctx.fillText('1', barX + barW, barY + barH + 12);

    ctx.fillStyle = 'rgba(46, 37, 30, 0.5)';
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Learning Rate →', W / 2, H - 4);
    ctx.save();
    ctx.translate(10, H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Regularization Strength →', 0, 0);
    ctx.restore();
  }, [visiblePoints, surfaceMode, showOptimum, trueOpt, searchMethod, bestPoint]);

  const run = useCallback(() => {
    if (isRunning) {
      setIsRunning(false);
      setDisplayCount(0);
      return;
    }
    if (searchMethod === 'bayesian') {
      setDisplayCount(0);
      setIsRunning(true);
    } else {
      setDisplayCount(allPoints.length);
    }
  }, [isRunning, searchMethod, allPoints.length]);

  const reset = useCallback(() => {
    setIsRunning(false);
    setDisplayCount(0);
  }, []);

  const handleMethodChange = useCallback((m: SearchMethod) => {
    if (isRunning) setIsRunning(false);
    setSearchMethod(m);
    setDisplayCount(0);
  }, [isRunning]);

  const handleSurfaceChange = useCallback((m: SurfaceMode) => {
    if (isRunning) setIsRunning(false);
    setSurfaceMode(m);
    setDisplayCount(0);
  }, [isRunning]);

  const handleNEvalsChange = useCallback((v: number) => {
    if (isRunning) setIsRunning(false);
    setNEvals(v);
    setDisplayCount(0);
  }, [isRunning]);

  const methodLabels: Record<SearchMethod, { label: string; desc: string }> = {
    grid: { label: 'Grid', desc: 'Fixed grid points covering the space' },
    random: { label: 'Random', desc: 'Uniformly sampled points' },
    bayesian: { label: 'Bayesian', desc: 'Adaptive sampling near optimum' },
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-12 gap-6 p-4">
      <div className="md:col-span-7">
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-3 rounded-2xl shadow-inner">
          <canvas ref={canvasRef} className="w-full rounded-lg" />
        </div>
      </div>

      <div className="md:col-span-5 space-y-4">
        <div className="bg-[#F4EFE6] border border-[#E5DDD0] p-4 rounded-2xl">
          <h4 className="text-[#2E251E] font-bold text-sm font-mono mb-2">Search Results</h4>
          <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-3 space-y-2 text-xs font-mono">
            <div className="flex justify-between">
              <span className="text-[#6E6257]">Best Score:</span>
              <span className="text-[#B6532B] font-bold">
                {bestPoint ? bestPoint.score.toFixed(4) : '—'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#6E6257]">Evaluations:</span>
              <span className="text-[#2E251E] font-bold">{displayCount}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#6E6257]">Sim. Time (s):</span>
              <span className="text-[#2E251E] font-bold">{simulatedTime.toFixed(1)}</span>
            </div>
            <div className="flex justify-between border-t border-[#E5DDD0] pt-2">
              <span className="text-[#6E6257]">Regret (gap):</span>
              <span className={`font-bold ${regret !== null && regret < 0.05 ? 'text-[#3B7A57]' : 'text-[#C18C3B]'}`}>
                {regret !== null ? regret.toFixed(4) : '—'}
              </span>
            </div>
          </div>
          {bestPoint && (
            <p className="text-[10px] text-[#6E6257] font-mono mt-2 text-center">
              Best at ({bestPoint.x.toFixed(3)}, {bestPoint.y.toFixed(3)})
              {regret !== null && regret < 0.01 ? ' ✓ Near optimum!' : ''}
            </p>
          )}
        </div>

        <div className="bg-[#F4EFE6] border border-[#E5DDD0] p-4 rounded-2xl space-y-4">
          <div>
            <label className="text-[10px] text-[#6E6257] font-mono font-semibold block mb-1.5">Search Method</label>
            <div className="grid grid-cols-3 gap-1.5">
              {(Object.keys(methodLabels) as SearchMethod[]).map(m => (
                <button
                  key={m}
                  onClick={() => handleMethodChange(m)}
                  className={`py-1.5 rounded-xl text-xs font-mono border transition-all ${
                    searchMethod === m
                      ? 'border-[#B6532B] bg-[#B6532B]/10 text-[#B6532B] font-bold'
                      : 'border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257] hover:text-[#2E251E]'
                  }`}
                >
                  {methodLabels[m].label}
                </button>
              ))}
            </div>
            <p className="text-[9px] text-[#6E6257] font-mono mt-1">{methodLabels[searchMethod].desc}</p>
          </div>

          <div>
            <div className="flex justify-between text-xs font-mono mb-1">
              <span className="text-[#6E6257]">Evaluations</span>
              <span className="text-[#B6532B] font-bold">{nEvals}</span>
            </div>
            <input
              type="range" min="5" max="100" step="1" value={nEvals}
              onChange={e => handleNEvalsChange(parseInt(e.target.value))}
              className="w-full accent-[#B6532B]"
            />
            <div className="flex justify-between text-[9px] text-[#6E6257] font-mono mt-0.5">
              <span>5</span><span>100</span>
            </div>
          </div>

          <div className="flex gap-2">
            <button
              onClick={run}
              className={`flex-1 py-2 rounded-xl text-xs font-mono font-bold border transition-all ${
                isRunning
                  ? 'border-[#C18C3B] bg-[#C18C3B]/10 text-[#C18C3B]'
                  : 'border-[#B6532B] bg-[#B6532B] text-white hover:bg-[#8B3A1F]'
              }`}
            >
              {isRunning ? '⏸ Stop' : '▶ Run Search'}
            </button>
            <button
              onClick={reset}
              className="flex-1 py-2 rounded-xl text-xs font-mono border border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257] hover:text-[#2E251E] transition-all"
            >
              ↺ Reset
            </button>
          </div>

          <div className="flex items-center gap-3">
            <label className="flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox" checked={showOptimum}
                onChange={e => setShowOptimum(e.target.checked)}
                className="accent-[#B6532B]"
              />
              <span className="text-xs font-mono text-[#6E6257]">Show True Optimum ★</span>
            </label>
          </div>

          <div>
            <label className="text-[10px] text-[#6E6257] font-mono font-semibold block mb-1.5">Surface Type</label>
            <div className="grid grid-cols-3 gap-1.5">
              {(['simple', 'complex', 'ridge'] as SurfaceMode[]).map(m => (
                <button
                  key={m}
                  onClick={() => handleSurfaceChange(m)}
                  className={`py-1.5 rounded-xl text-xs font-mono border transition-all capitalize ${
                    surfaceMode === m
                      ? 'border-[#B6532B] bg-[#B6532B]/10 text-[#B6532B] font-bold'
                      : 'border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257] hover:text-[#2E251E]'
                  }`}
                >
                  {m === 'simple' ? '1 Peak' : m === 'complex' ? '2 Peaks' : 'Ridge'}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
