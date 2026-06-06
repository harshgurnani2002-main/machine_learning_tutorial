import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Trash2, Sparkles, Play, Pause, RotateCcw, ChevronRight, TrendingDown, TrendingUp } from 'lucide-react';

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────
interface Point { x: number; y: number; label: 0 | 1; }
interface Stump {
  feature: 'x' | 'y';
  threshold: number;
  leftVal: number;
  rightVal: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Default dataset — overlapping clusters so accuracy isn't instant 100%
// ─────────────────────────────────────────────────────────────────────────────
const defaultPoints: Point[] = [
  { x: 110, y: 160, label: 0 }, { x: 160, y: 120, label: 0 }, { x: 190, y: 210, label: 0 },
  { x: 140, y: 270, label: 0 }, { x: 210, y: 300, label: 0 }, { x: 230, y: 155, label: 0 },
  { x: 170, y: 340, label: 0 }, { x: 120, y: 300, label: 0 }, { x: 260, y: 240, label: 0 },
  { x: 130, y: 100, label: 0 }, { x: 290, y: 200, label: 0 }, // slightly overlapping
  { x: 420, y: 150, label: 1 }, { x: 460, y: 220, label: 1 }, { x: 490, y: 300, label: 1 },
  { x: 440, y: 340, label: 1 }, { x: 390, y: 260, label: 1 }, { x: 510, y: 185, label: 1 },
  { x: 470, y: 380, label: 1 }, { x: 530, y: 280, label: 1 }, { x: 400, y: 120, label: 1 },
  { x: 320, y: 200, label: 1 }, { x: 340, y: 310, label: 1 }, // overlapping
];

// ─────────────────────────────────────────────────────────────────────────────
// Algorithm helpers
// ─────────────────────────────────────────────────────────────────────────────
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

const buildStump = (
  _pts: Point[],
  normPts: Array<{ x: number; y: number }>,
  residuals: number[]
): Stump => {
  let bestGain = -Infinity, bestFeat: 'x' | 'y' = 'x', bestThresh = 0.5;

  for (const feat of ['x', 'y'] as const) {
    const vals = [...new Set(normPts.map(p => p[feat]))].sort((a, b) => a - b);
    for (let k = 0; k < vals.length - 1; k++) {
      const t = (vals[k] + vals[k + 1]) / 2;
      const IL = normPts.map((p, i) => p[feat] <= t ? i : -1).filter(i => i >= 0);
      const IR = normPts.map((p, i) => p[feat] > t ? i : -1).filter(i => i >= 0);
      if (!IL.length || !IR.length) continue;
      const sL = IL.reduce((a, i) => a + residuals[i], 0);
      const sR = IR.reduce((a, i) => a + residuals[i], 0);
      const sA = residuals.reduce((a, b) => a + b, 0);
      const gain = sL * sL / IL.length + sR * sR / IR.length - sA * sA / normPts.length;
      if (gain > bestGain) { bestGain = gain; bestFeat = feat; bestThresh = t; }
    }
  }

  const IL = normPts.map((p, i) => p[bestFeat] <= bestThresh ? i : -1).filter(i => i >= 0);
  const IR = normPts.map((p, i) => p[bestFeat] > bestThresh ? i : -1).filter(i => i >= 0);
  const leftVal  = IL.length ? IL.reduce((a, i) => a + residuals[i], 0) / IL.length : 0;
  const rightVal = IR.length ? IR.reduce((a, i) => a + residuals[i], 0) / IR.length : 0;

  return { feature: bestFeat, threshold: bestThresh, leftVal, rightVal };
};

const predictStump = (s: Stump, nx: number, ny: number) =>
  (s.feature === 'x' ? nx : ny) <= s.threshold ? s.leftVal : s.rightVal;

const trainGB = (pts: Point[], nRounds: number, lr: number, normPts: Array<{ x: number; y: number }>) => {
  if (!pts.length) return { f0: 0, stumps: [], accHistory: [], lossHistory: [] };
  const pAvg = Math.max(0.01, Math.min(0.99, pts.reduce((a, p) => a + p.label, 0) / pts.length));
  const f0 = Math.log(pAvg / (1 - pAvg));
  const fVec = pts.map(() => f0);
  const stumps: Stump[] = [];
  const accHistory: number[] = [];
  const lossHistory: number[] = [];

  for (let m = 0; m < nRounds; m++) {
    const probs = fVec.map(f => sigmoid(f));
    const residuals = pts.map((p, i) => p.label - probs[i]);
    const stump = buildStump(pts, normPts, residuals);
    stumps.push(stump);
    for (let i = 0; i < pts.length; i++) fVec[i] += lr * predictStump(stump, normPts[i].x, normPts[i].y);

    // Metrics
    const correct = pts.filter((p, i) => (sigmoid(fVec[i]) >= 0.5 ? 1 : 0) === p.label).length;
    accHistory.push((correct / pts.length) * 100);
    const eps = 1e-7;
    const loss = -pts.reduce((sum, p, i) => {
      const prob = Math.max(eps, Math.min(1 - eps, sigmoid(fVec[i])));
      return sum + p.label * Math.log(prob) + (1 - p.label) * Math.log(1 - prob);
    }, 0) / pts.length;
    lossHistory.push(loss);
  }
  return { f0, stumps, accHistory, lossHistory };
};

// ─────────────────────────────────────────────────────────────────────────────
// Draw decision boundary canvas
// ─────────────────────────────────────────────────────────────────────────────
const drawCanvas = (
  ctx: CanvasRenderingContext2D, W: number, H: number,
  f0: number, stumps: Stump[], lr: number, pts: Point[]
) => {
  const bg = ctx.createLinearGradient(0, 0, 0, H);
  bg.addColorStop(0, '#FAF6EE'); bg.addColorStop(1, '#F4EFE6');
  ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);

  const step = 6;
  for (let x = 0; x < W; x += step) {
    for (let y = 0; y < H; y += step) {
      let f = f0;
      const nx = (x + step / 2) / W, ny = (y + step / 2) / H;
      stumps.forEach(s => { f += lr * predictStump(s, nx, ny); });
      const p = sigmoid(f);
      ctx.fillStyle = `rgba(${Math.round(182 * (1-p) + 59 * p)},${Math.round(83 * (1-p) + 122 * p)},${Math.round(43 * (1-p) + 87 * p)},0.18)`;
      ctx.fillRect(x, y, step, step);
    }
  }

  ctx.strokeStyle = 'rgba(110,98,87,0.07)'; ctx.lineWidth = 0.5;
  for (let x = 0; x < W; x += 50) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }
  for (let y = 0; y < H; y += 50) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }

  pts.forEach(pt => {
    const px = pt.x * (W / 600), py = pt.y * (H / 400);
    const col = pt.label === 0 ? '#B6532B' : '#3B7A57';
    ctx.beginPath(); ctx.arc(px, py, 7, 0, Math.PI * 2);
    ctx.fillStyle = col; ctx.fill();
    ctx.strokeStyle = '#FAF6EE'; ctx.lineWidth = 1.5; ctx.stroke();
  });
};

// ─────────────────────────────────────────────────────────────────────────────
// Miniature SVG line chart
// ─────────────────────────────────────────────────────────────────────────────
const MiniChart: React.FC<{ values: number[]; step: number; color: string; label: string; reverse?: boolean }> = ({ values, step, color, label, reverse }) => {
  const W = 260, H = 80, PL = 8, PR = 8, PT = 6, PB = 18;
  const cW = W - PL - PR, cH = H - PT - PB;
  const vis = values.slice(0, step + 1);
  const mn = Math.min(...vis, 0), mx = Math.max(...vis, 0.001);
  const range = mx - mn || 1;
  const toX = (i: number) => PL + (i / Math.max(values.length - 1, 1)) * cW;
  const toY = (v: number) => PT + cH - ((v - mn) / range) * cH;

  return (
    <div style={{ background: '#F4EFE6', borderRadius: 10, padding: '8px 10px', border: '1px solid #E5DDD0' }}>
      <div style={{ fontSize: 10, color: '#9E9183', fontFamily: 'JetBrains Mono, monospace', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          {reverse ? <TrendingDown size={10} color={color} /> : <TrendingUp size={10} color={color} />}
          {label}
        </span>
        <span style={{ color, fontWeight: 700 }}>{vis[vis.length - 1]?.toFixed(reverse ? 3 : 1)}{reverse ? '' : '%'}</span>
      </div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`}>
        <rect x={PL} y={PT} width={cW} height={cH} fill="#FAF6EE" rx="3" />
        {vis.length >= 2 && (
          <polyline
            points={vis.map((v, i) => `${toX(i)},${toY(v)}`).join(' ')}
            fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
          />
        )}
        {vis.length > 0 && <circle cx={toX(vis.length - 1)} cy={toY(vis[vis.length - 1])} r={3.5} fill={color} />}
        <text x={PL + cW / 2} y={H - 2} fill="#BDB0A2" fontSize="8" textAnchor="middle" fontFamily="JetBrains Mono, monospace">
          Boosting Round
        </text>
      </svg>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Residual bar chart — show per-point residuals shrinking
// ─────────────────────────────────────────────────────────────────────────────
const ResidualBars: React.FC<{ pts: Point[]; f0: number; stumps: Stump[]; lr: number; step: number }> = ({ pts, f0, stumps, lr, step }) => {
  if (!pts.length) return null;
  const normPts = pts.map(p => ({ x: p.x / 600, y: p.y / 400 }));
  const residuals = pts.map((p, idx) => {
    let f = f0;
    stumps.slice(0, step + 1).forEach(s => { f += lr * predictStump(s, normPts[idx].x, normPts[idx].y); });
    return p.label - sigmoid(f);
  });

  const W = 320, H = 70, PL = 30, PR = 8, PT = 6, PB = 20;
  const cW = W - PL - PR, cH = H - PT - PB;
  const maxAbs = Math.max(...residuals.map(Math.abs), 0.1);
  const barW = cW / pts.length;

  return (
    <div style={{ background: '#F4EFE6', borderRadius: 10, padding: '8px 10px', border: '1px solid #E5DDD0' }}>
      <div style={{ fontSize: 10, color: '#9E9183', fontFamily: 'JetBrains Mono, monospace', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 4 }}>
        Pseudo-Residuals (shrink with each round)
      </div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`}>
        <rect x={PL} y={PT} width={cW} height={cH} fill="#FAF6EE" rx="3" />
        <line x1={PL} y1={PT + cH / 2} x2={PL + cW} y2={PT + cH / 2} stroke="#D5C9B8" strokeWidth="1" />
        {residuals.map((r, i) => {
          const mid = PT + cH / 2;
          const h = Math.abs(r / maxAbs) * (cH / 2 - 1);
          const x = PL + i * barW + barW * 0.15;
          const bw = barW * 0.7;
          const col = pts[i].label === 0 ? '#B6532B' : '#3B7A57';
          return <rect key={i} x={x} y={r >= 0 ? mid - h : mid} width={Math.max(bw, 1)} height={Math.max(h, 1)} fill={`${col}90`} rx="1" />;
        })}
        <text x={PL - 4} y={PT + 8} fill="#BDB0A2" fontSize="7" textAnchor="end">+</text>
        <text x={PL - 4} y={PT + cH - 2} fill="#BDB0A2" fontSize="7" textAnchor="end">−</text>
        <text x={PL + cW / 2} y={H - 2} fill="#BDB0A2" fontSize="8" textAnchor="middle" fontFamily="JetBrains Mono, monospace">
          Each point
        </text>
      </svg>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Main: Basic Gradient Boosting Simulator
// ─────────────────────────────────────────────────────────────────────────────
export const BasicGBSimulator: React.FC = () => {
  const [pts, setPts] = useState<Point[]>(defaultPoints);
  const [activeLabel, setActiveLabel] = useState<0 | 1>(0);
  const [nRounds, setNRounds] = useState(15);
  const [lr, setLr] = useState(0.2);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(600);
  const playRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const normPts = React.useMemo(() => pts.map(p => ({ x: p.x / 600, y: p.y / 400 })), [pts]);
  const model = React.useMemo(() => trainGB(pts, nRounds, lr, normPts), [pts, nRounds, lr, normPts]);
  const maxStep = Math.max(model.stumps.length - 1, 0);
  const safeStep = Math.min(step, maxStep);

  const stopPlay = useCallback(() => {
    if (playRef.current) clearTimeout(playRef.current);
    setPlaying(false);
  }, []);

  useEffect(() => {
    if (!playing) return;
    if (safeStep >= maxStep) { stopPlay(); return; }
    playRef.current = setTimeout(() => setStep(s => s + 1), speed);
    return () => { if (playRef.current) clearTimeout(playRef.current); };
  }, [playing, safeStep, maxStep, speed, stopPlay]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    canvas.width = 600; canvas.height = 400;
    drawCanvas(ctx, 600, 400, model.f0, model.stumps.slice(0, safeStep + 1), lr, pts);
  }, [safeStep, model, lr, pts]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    setPts(prev => [...prev, {
      x: ((e.clientX - rect.left) / rect.width) * 600,
      y: ((e.clientY - rect.top) / rect.height) * 400,
      label: activeLabel,
    }]);
    stopPlay(); setStep(0);
  };

  const curAcc = model.accHistory[safeStep] ?? 0;
  const curLoss = model.lossHistory[safeStep] ?? 0;
  const curStump = model.stumps[safeStep];

  return (
    <div className="grid grid-cols-1 md:grid-cols-12 gap-5 p-5">

      {/* ── Sidebar ── */}
      <div className="md:col-span-4 flex flex-col gap-4">

        {/* Playback */}
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-2xl p-5 shadow-sm">
          <div className="text-[10px] font-mono text-[#9E9183] uppercase tracking-widest mb-3">Boosting Rounds</div>

          <div className="flex gap-2 mb-4">
            <button onClick={() => { stopPlay(); setStep(0); }}
              className="w-9 h-9 flex items-center justify-center rounded-lg border border-[#E5DDD0] bg-[#F4EFE6] text-[#6E6257] hover:bg-[#EDE6DA] transition-all cursor-pointer">
              <RotateCcw size={14} />
            </button>
            <button onClick={() => {
              if (playing) { stopPlay(); return; }
              if (safeStep >= maxStep) setStep(0);
              setPlaying(true);
            }}
              className="flex-1 h-9 flex items-center justify-center gap-2 rounded-lg font-bold text-sm transition-all cursor-pointer"
              style={{ background: playing ? '#B6532B' : '#B6532B15', color: '#B6532B', border: '1px solid #B6532B40' }}>
              {playing ? <><Pause size={14} /> Pause</> : <><Play size={14} /> Auto-Run</>}
            </button>
            <button onClick={() => { stopPlay(); setStep(s => Math.min(s + 1, maxStep)); }}
              className="w-9 h-9 flex items-center justify-center rounded-lg border border-[#E5DDD0] bg-[#F4EFE6] text-[#6E6257] hover:bg-[#EDE6DA] transition-all cursor-pointer">
              <ChevronRight size={14} />
            </button>
          </div>

          {/* Round scrubber */}
          <div className="mb-3">
            <div className="flex justify-between items-center mb-1.5">
              <span style={{ fontSize: 11, color: '#6E6257', fontFamily: 'JetBrains Mono, monospace' }}>Round</span>
              <span style={{ fontSize: 12, fontWeight: 700, color: '#B6532B', background: '#B6532B14', border: '1px solid #B6532B30', padding: '1px 8px', borderRadius: 6, fontFamily: 'JetBrains Mono, monospace' }}>
                {safeStep + 1} / {nRounds}
              </span>
            </div>
            <input type="range" min={0} max={maxStep} value={safeStep}
              onChange={e => { stopPlay(); setStep(Number(e.target.value)); }}
              style={{ width: '100%', accentColor: '#B6532B', cursor: 'pointer' }} />
          </div>

          {/* Speed */}
          <div>
            <div className="flex justify-between items-center mb-1.5">
              <span style={{ fontSize: 11, color: '#6E6257', fontFamily: 'JetBrains Mono, monospace' }}>Speed</span>
              <span style={{ fontSize: 10, color: '#9E9183', fontFamily: 'JetBrains Mono, monospace' }}>
                {speed <= 300 ? 'Fast' : speed <= 700 ? 'Normal' : 'Slow'}
              </span>
            </div>
            <input type="range" min={100} max={1200} step={100} value={speed}
              onChange={e => setSpeed(Number(e.target.value))}
              style={{ width: '100%', accentColor: '#C18C3B', cursor: 'pointer' }} />
          </div>
        </div>

        {/* Parameters */}
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-2xl p-5 shadow-sm">
          <div className="text-[10px] font-mono text-[#9E9183] uppercase tracking-widest mb-3">Parameters</div>

          {[
            { label: 'Boosting Rounds', val: nRounds, min: 2, max: 40, step: 1, fmt: (v: number) => `${v}`, set: setNRounds },
            { label: 'Learning Rate (η)', val: lr, min: 0.01, max: 1, step: 0.01, fmt: (v: number) => v.toFixed(2), set: setLr },
          ].map(({ label, val, min, max, step: st, fmt, set }) => (
            <div key={label} className="mb-3">
              <div className="flex justify-between items-center mb-1.5">
                <span style={{ fontSize: 11, color: '#6E6257', fontFamily: 'JetBrains Mono, monospace' }}>{label}</span>
                <span style={{ fontSize: 12, fontWeight: 700, color: '#B6532B', background: '#B6532B14', border: '1px solid #B6532B30', padding: '1px 8px', borderRadius: 6, fontFamily: 'JetBrains Mono, monospace' }}>{fmt(val)}</span>
              </div>
              <input type="range" min={min} max={max} step={st} value={val}
                onChange={e => { set(parseFloat(e.target.value) as never); stopPlay(); setStep(0); }}
                style={{ width: '100%', accentColor: '#B6532B', cursor: 'pointer' }} />
            </div>
          ))}
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl px-4 py-3 text-center shadow-sm">
            <div style={{ fontSize: 22, fontWeight: 800, color: '#B6532B', fontFamily: 'JetBrains Mono, monospace', lineHeight: 1 }}>{curAcc.toFixed(0)}%</div>
            <div style={{ fontSize: 10, color: '#9E9183', marginTop: 3, textTransform: 'uppercase', letterSpacing: '0.07em', fontFamily: 'JetBrains Mono, monospace' }}>Accuracy</div>
          </div>
          <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl px-4 py-3 text-center shadow-sm">
            <div style={{ fontSize: 22, fontWeight: 800, color: '#C18C3B', fontFamily: 'JetBrains Mono, monospace', lineHeight: 1 }}>{curLoss.toFixed(3)}</div>
            <div style={{ fontSize: 10, color: '#9E9183', marginTop: 3, textTransform: 'uppercase', letterSpacing: '0.07em', fontFamily: 'JetBrains Mono, monospace' }}>Log-Loss</div>
          </div>
        </div>

        {/* Current stump info */}
        {curStump && (
          <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-4 shadow-sm">
            <div style={{ fontSize: 10, color: '#9E9183', fontFamily: 'JetBrains Mono, monospace', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 8 }}>
              Round {safeStep + 1} — Weak Learner
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              {[
                { k: 'Split on', v: curStump.feature.toUpperCase() },
                { k: 'Threshold', v: curStump.threshold.toFixed(2) },
                { k: 'Left leaf', v: (curStump.leftVal * lr).toFixed(3) },
                { k: 'Right leaf', v: (curStump.rightVal * lr).toFixed(3) },
              ].map(({ k, v }) => (
                <div key={k} style={{ background: '#F4EFE6', borderRadius: 8, padding: '6px 8px' }}>
                  <div style={{ fontSize: 9, color: '#9E9183', fontFamily: 'JetBrains Mono, monospace', marginBottom: 2 }}>{k}</div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: '#2E251E', fontFamily: 'JetBrains Mono, monospace' }}>{v}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Add points */}
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-2xl p-4 shadow-sm">
          <div className="text-[10px] font-mono text-[#9E9183] uppercase tracking-widest mb-3">Add Points (click canvas)</div>
          <div className="flex gap-2 mb-2">
            {([0, 1] as const).map(lbl => (
              <button key={lbl} onClick={() => setActiveLabel(lbl)}
                className="flex-1 py-2 rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-1.5 cursor-pointer"
                style={{
                  background: activeLabel === lbl ? (lbl === 0 ? '#B6532B' : '#3B7A57') : 'transparent',
                  color: activeLabel === lbl ? '#fff' : (lbl === 0 ? '#B6532B' : '#3B7A57'),
                  border: `1.5px solid ${lbl === 0 ? '#B6532B' : '#3B7A57'}`,
                }}>
                <span style={{ width: 7, height: 7, borderRadius: '50%', background: activeLabel === lbl ? '#fff' : (lbl === 0 ? '#B6532B' : '#3B7A57'), display: 'inline-block' }} />
                Class {lbl}
              </button>
            ))}
          </div>
          <div className="flex gap-2">
            <button onClick={() => { setPts(defaultPoints); stopPlay(); setStep(0); }}
              className="flex-1 py-2 rounded-lg border border-[#E5DDD0] bg-[#F4EFE6] hover:bg-[#EDE6DA] text-[#6E6257] text-xs font-bold flex items-center justify-center gap-1 cursor-pointer">
              <Sparkles size={12} /> Reset
            </button>
            <button onClick={() => { setPts([]); stopPlay(); setStep(0); }}
              className="flex-1 py-2 rounded-lg border border-[#E5DDD0] bg-[#F4EFE6] hover:bg-[#EDE6DA] text-[#B6532B] text-xs font-bold flex items-center justify-center gap-1 cursor-pointer">
              <Trash2 size={12} /> Clear
            </button>
          </div>
        </div>
      </div>

      {/* ── Main Area ── */}
      <div className="md:col-span-8 flex flex-col gap-4">

        {/* Canvas */}
        <div className="bg-[#F4EFE6] border border-[#E5DDD0] rounded-2xl p-2 shadow-sm relative">
          <div style={{ position: 'absolute', top: 14, left: 14, zIndex: 5, background: 'rgba(46,37,30,0.82)', padding: '5px 10px', borderRadius: 8, fontSize: 10, fontFamily: 'JetBrains Mono, monospace', color: '#FAF6EE', display: 'flex', gap: 8, alignItems: 'center' }}>
            <span style={{ fontWeight: 700, color: '#C18C3B' }}>Gradient Boosting</span>
            <span style={{ opacity: 0.5 }}>·</span>
            <span>Round {safeStep + 1} / {nRounds}</span>
            <span style={{ opacity: 0.5 }}>·</span>
            <span style={{ color: '#B6532B' }}>Acc {curAcc.toFixed(0)}%</span>
          </div>
          <div style={{ position: 'absolute', bottom: 12, right: 14, zIndex: 5, background: 'rgba(46,37,30,0.55)', padding: '3px 8px', borderRadius: 6, fontSize: 9, fontFamily: 'JetBrains Mono, monospace', color: 'rgba(250,246,238,0.7)' }}>
            Click to add points
          </div>
          <canvas ref={canvasRef} onClick={handleClick}
            className="block w-full rounded-xl cursor-crosshair" style={{ aspectRatio: '3/2' }} />
        </div>

        {/* Charts + residuals */}
        <div className="grid grid-cols-2 gap-3">
          <MiniChart values={model.accHistory} step={safeStep} color="#B6532B" label="Accuracy" />
          <MiniChart values={model.lossHistory} step={safeStep} color="#C18C3B" label="Log-Loss" reverse />
        </div>

        <ResidualBars pts={pts} f0={model.f0} stumps={model.stumps} lr={lr} step={safeStep} />

        {/* Legend */}
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-4 shadow-sm">
          <div style={{ fontSize: 11, fontWeight: 700, color: '#2E251E', marginBottom: 8 }}>How it works</div>
          <div className="grid grid-cols-3 gap-3" style={{ fontSize: 10, color: '#6E6257', lineHeight: 1.6, fontFamily: 'JetBrains Mono, monospace' }}>
            <div><span style={{ color: '#B6532B', fontWeight: 700 }}>1. Residuals</span><br />Each round trains a shallow tree on the errors (pseudo-residuals) left by the ensemble so far.</div>
            <div><span style={{ color: '#B6532B', fontWeight: 700 }}>2. Learning rate</span><br />Each tree's contribution is scaled by η before adding. Smaller η = slower but more robust.</div>
            <div><span style={{ color: '#B6532B', fontWeight: 700 }}>3. Convergence</span><br />Watch residuals shrink and the boundary sharpen as more rounds are added.</div>
          </div>
        </div>
      </div>
    </div>
  );
};
