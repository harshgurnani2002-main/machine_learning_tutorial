import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Trash2, Sparkles, Play, Pause, RotateCcw, ChevronRight, TrendingUp, Info } from 'lucide-react';

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────
interface Point { x: number; y: number; label: 0 | 1; }
interface TreeNode {
  id: string; isLeaf: boolean;
  feature?: 'x' | 'y'; threshold?: number; value?: number;
  depth: number; left?: TreeNode; right?: TreeNode;
}
interface TrainedModel { f0: number; trees: TreeNode[]; accHistory: number[]; lossHistory: number[]; }
type AlgoMode = 'xgb' | 'lgb' | 'cat';

// ─────────────────────────────────────────────────────────────────────────────
// Default dataset — overlapping clusters
// ─────────────────────────────────────────────────────────────────────────────
const defaultPoints: Point[] = [
  { x: 110, y: 160, label: 0 }, { x: 160, y: 120, label: 0 }, { x: 190, y: 210, label: 0 },
  { x: 140, y: 270, label: 0 }, { x: 210, y: 300, label: 0 }, { x: 230, y: 155, label: 0 },
  { x: 170, y: 340, label: 0 }, { x: 120, y: 300, label: 0 }, { x: 260, y: 240, label: 0 },
  { x: 130, y: 100, label: 0 }, { x: 290, y: 200, label: 0 },
  { x: 420, y: 150, label: 1 }, { x: 460, y: 220, label: 1 }, { x: 490, y: 300, label: 1 },
  { x: 440, y: 340, label: 1 }, { x: 390, y: 260, label: 1 }, { x: 510, y: 185, label: 1 },
  { x: 470, y: 380, label: 1 }, { x: 530, y: 280, label: 1 }, { x: 400, y: 120, label: 1 },
  { x: 320, y: 200, label: 1 }, { x: 340, y: 310, label: 1 },
];

// ─────────────────────────────────────────────────────────────────────────────
// Algorithm palette (site theme colours)
// ─────────────────────────────────────────────────────────────────────────────
const ALGO: Record<AlgoMode, { name: string; desc: string; color: string; bg: string; border: string }> = {
  xgb: { name: 'XGBoost',  desc: 'Level-wise · Hessians · L2 reg',   color: '#B6532B', bg: '#B6532B12', border: '#B6532B35' },
  lgb: { name: 'LightGBM', desc: 'Leaf-wise · Best-first splits',    color: '#3B7A57', bg: '#3B7A5712', border: '#3B7A5735' },
  cat: { name: 'CatBoost', desc: 'Symmetric / Oblivious trees',      color: '#7B5EA7', bg: '#7B5EA712', border: '#7B5EA735' },
};

// ─────────────────────────────────────────────────────────────────────────────
// Tree Building
// ─────────────────────────────────────────────────────────────────────────────
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

const predictTree = (node: TreeNode, x: number, y: number): number => {
  if (node.isLeaf) return node.value ?? 0;
  if (!node.feature || node.threshold === undefined) return node.value ?? 0;
  const val = node.feature === 'x' ? x : y;
  const child = val <= node.threshold ? node.left : node.right;
  return child ? predictTree(child, x, y) : (node.value ?? 0);
};

const buildXGBTree = (
  idx: number[], depth: number, maxDepth: number,
  normPts: Array<{ x: number; y: number }>,
  grads: number[], hess: number[], lambda: number, gamma: number
): TreeNode => {
  const sumG = idx.reduce((a, i) => a + grads[i], 0);
  const sumH = idx.reduce((a, i) => a + hess[i], 0);
  const val = sumH + lambda > 0 ? -sumG / (sumH + lambda) : 0;
  if (depth >= maxDepth || idx.length <= 2)
    return { id: `${Math.random()}`, isLeaf: true, value: val, depth };

  let bestGain = 0, bF: 'x' | 'y' = 'x', bT = 0.5, bIL: number[] = [], bIR: number[] = [];
  for (const feat of ['x', 'y'] as const) {
    const vals = [...new Set(idx.map(i => normPts[i][feat]))].sort((a, b) => a - b);
    for (let k = 0; k < vals.length - 1; k++) {
      const t = (vals[k] + vals[k + 1]) / 2;
      const IL = idx.filter(i => normPts[i][feat] <= t), IR = idx.filter(i => normPts[i][feat] > t);
      if (!IL.length || !IR.length) continue;
      const GL = IL.reduce((a, i) => a + grads[i], 0), HL = IL.reduce((a, i) => a + hess[i], 0);
      const GR = IR.reduce((a, i) => a + grads[i], 0), HR = IR.reduce((a, i) => a + hess[i], 0);
      const gain = 0.5 * (GL * GL / (HL + lambda) + GR * GR / (HR + lambda) - sumG * sumG / (sumH + lambda)) - gamma;
      if (gain > bestGain) { bestGain = gain; bF = feat; bT = t; bIL = IL; bIR = IR; }
    }
  }
  if (!bIL.length || !bIR.length) return { id: `${Math.random()}`, isLeaf: true, value: val, depth };
  return {
    id: `${Math.random()}`, isLeaf: false, feature: bF, threshold: bT, value: val, depth,
    left: buildXGBTree(bIL, depth + 1, maxDepth, normPts, grads, hess, lambda, gamma),
    right: buildXGBTree(bIR, depth + 1, maxDepth, normPts, grads, hess, lambda, gamma),
  };
};

interface LGBMLeaf {
  id: string; indices: number[]; value: number; depth: number; isLeaf: boolean;
  feature?: 'x' | 'y'; threshold?: number; left?: LGBMLeaf; right?: LGBMLeaf;
  gain: number; bestIL?: number[]; bestIR?: number[];
}
const buildLGBTree = (rootIdx: number[], maxLeaves: number, normPts: Array<{ x: number; y: number }>, residuals: number[]): TreeNode => {
  const mean = (arr: number[]) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
  const evalLeaf = (leaf: LGBMLeaf) => {
    if (leaf.indices.length <= 2) { leaf.gain = -Infinity; return; }
    let best = -Infinity, bF: 'x' | 'y' = 'x', bT = 0.5, bIL: number[] = [], bIR: number[] = [];
    for (const feat of ['x', 'y'] as const) {
      const vals = [...new Set(leaf.indices.map(i => normPts[i][feat]))].sort((a, b) => a - b);
      for (let k = 0; k < vals.length - 1; k++) {
        const t = (vals[k] + vals[k + 1]) / 2;
        const IL = leaf.indices.filter(i => normPts[i][feat] <= t), IR = leaf.indices.filter(i => normPts[i][feat] > t);
        if (!IL.length || !IR.length) continue;
        const sL = IL.reduce((a, i) => a + residuals[i], 0), sR = IR.reduce((a, i) => a + residuals[i], 0);
        const sA = leaf.indices.reduce((a, i) => a + residuals[i], 0);
        const g = sL * sL / IL.length + sR * sR / IR.length - sA * sA / leaf.indices.length;
        if (g > best) { best = g; bF = feat; bT = t; bIL = IL; bIR = IR; }
      }
    }
    leaf.gain = best; leaf.feature = bF; leaf.threshold = bT; leaf.bestIL = bIL; leaf.bestIR = bIR;
  };
  const root: LGBMLeaf = { id: `${Math.random()}`, indices: rootIdx, value: mean(rootIdx.map(i => residuals[i])), depth: 0, isLeaf: true, gain: -Infinity };
  const leaves = [root]; evalLeaf(root);
  while (leaves.filter(l => l.isLeaf).length < maxLeaves) {
    const best = leaves.filter(l => l.isLeaf).reduce((a, b) => b.gain > a.gain ? b : a, { gain: -Infinity } as LGBMLeaf);
    if (!best.id || best.gain <= 0 || !best.bestIL?.length || !best.bestIR?.length) break;
    best.isLeaf = false;
    const lL: LGBMLeaf = { id: `${Math.random()}`, indices: best.bestIL, value: mean(best.bestIL.map(i => residuals[i])), depth: best.depth + 1, isLeaf: true, gain: -Infinity };
    const lR: LGBMLeaf = { id: `${Math.random()}`, indices: best.bestIR, value: mean(best.bestIR.map(i => residuals[i])), depth: best.depth + 1, isLeaf: true, gain: -Infinity };
    evalLeaf(lL); evalLeaf(lR); best.left = lL; best.right = lR; leaves.push(lL, lR);
  }
  const conv = (l: LGBMLeaf): TreeNode => ({ id: l.id, isLeaf: l.isLeaf, value: l.value, depth: l.depth, feature: l.feature, threshold: l.threshold, left: l.left ? conv(l.left) : undefined, right: l.right ? conv(l.right) : undefined });
  return conv(root);
};

const buildCatTree = (idx: number[], maxDepth: number, normPts: Array<{ x: number; y: number }>, residuals: number[]): TreeNode => {
  const splits: Array<{ feature: 'x' | 'y'; threshold: number }> = [];
  let parts: number[][] = [idx];
  for (let d = 0; d < maxDepth; d++) {
    let best = -Infinity, bF: 'x' | 'y' = 'x', bT = 0.5;
    for (const feat of ['x', 'y'] as const) {
      const vals = [...new Set(idx.map(i => normPts[i][feat]))].sort((a, b) => a - b);
      for (let k = 0; k < vals.length - 1; k++) {
        const t = (vals[k] + vals[k + 1]) / 2;
        let gain = 0, valid = false;
        for (const part of parts) {
          const IL = part.filter(i => normPts[i][feat] <= t), IR = part.filter(i => normPts[i][feat] > t);
          if (IL.length > 0 && IR.length > 0) {
            valid = true;
            const sL = IL.reduce((a, i) => a + residuals[i], 0), sR = IR.reduce((a, i) => a + residuals[i], 0);
            const sA = part.reduce((a, i) => a + residuals[i], 0);
            gain += sL * sL / IL.length + sR * sR / IR.length - sA * sA / part.length;
          }
        }
        if (valid && gain > best) { best = gain; bF = feat; bT = t; }
      }
    }
    if (best <= 0) break;
    splits.push({ feature: bF, threshold: bT });
    parts = parts.flatMap(p => [p.filter(i => normPts[i][bF] <= bT), p.filter(i => normPts[i][bF] > bT)]);
  }
  const build = (i2: number[], d: number): TreeNode => {
    const val = i2.length ? i2.reduce((a, i) => a + residuals[i], 0) / i2.length : 0;
    if (d >= splits.length || i2.length <= 1) return { id: `${Math.random()}`, isLeaf: true, value: val, depth: d };
    const sp = splits[d];
    const IL = i2.filter(i => normPts[i][sp.feature] <= sp.threshold), IR = i2.filter(i => normPts[i][sp.feature] > sp.threshold);
    return { id: `${Math.random()}`, isLeaf: false, feature: sp.feature, threshold: sp.threshold, value: val, depth: d, left: build(IL, d + 1), right: build(IR, d + 1) };
  };
  return build(idx, 0);
};

const trainModel = (pts: Point[], algo: AlgoMode, nTrees: number, lr: number, maxDepth: number, maxLeaves: number, lambda: number, gamma: number, normPts: Array<{ x: number; y: number }>): TrainedModel => {
  if (!pts.length) return { f0: 0, trees: [], accHistory: [], lossHistory: [] };
  const pAvg = Math.max(0.01, Math.min(0.99, pts.reduce((a, p) => a + p.label, 0) / pts.length));
  const f0 = Math.log(pAvg / (1 - pAvg));
  const fVec = pts.map(() => f0);
  const trees: TreeNode[] = [], accHistory: number[] = [], lossHistory: number[] = [];
  const allIdx = pts.map((_, i) => i);
  const eps = 1e-7;
  for (let m = 0; m < nTrees; m++) {
    const probs = fVec.map(f => sigmoid(f));
    const residuals = pts.map((p, i) => p.label - probs[i]);
    const grads = pts.map((p, i) => probs[i] - p.label);
    const hess = probs.map(p => Math.max(1e-4, p * (1 - p)));
    let tree: TreeNode;
    if (algo === 'xgb') tree = buildXGBTree(allIdx, 0, maxDepth, normPts, grads, hess, lambda, gamma);
    else if (algo === 'lgb') tree = buildLGBTree(allIdx, maxLeaves, normPts, residuals);
    else tree = buildCatTree(allIdx, maxDepth, normPts, residuals);
    trees.push(tree);
    for (let i = 0; i < pts.length; i++) fVec[i] += lr * predictTree(tree, normPts[i].x, normPts[i].y);
    const correct = pts.filter((p, i) => (sigmoid(fVec[i]) >= 0.5 ? 1 : 0) === p.label).length;
    accHistory.push((correct / pts.length) * 100);
    const loss = -pts.reduce((s, p, i) => { const pr = Math.max(eps, Math.min(1 - eps, sigmoid(fVec[i]))); return s + p.label * Math.log(pr) + (1 - p.label) * Math.log(1 - pr); }, 0) / pts.length;
    lossHistory.push(loss);
  }
  return { f0, trees, accHistory, lossHistory };
};

// ─────────────────────────────────────────────────────────────────────────────
// Canvas drawing
// ─────────────────────────────────────────────────────────────────────────────
const drawCanvas = (ctx: CanvasRenderingContext2D, W: number, H: number, f0: number, trees: TreeNode[], lr: number, pts: Point[], step = 5) => {
  const bg = ctx.createLinearGradient(0, 0, 0, H);
  bg.addColorStop(0, '#FAF6EE'); bg.addColorStop(1, '#F4EFE6');
  ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);
  if (pts.length > 0) {
    for (let x = 0; x < W; x += step) {
      for (let y = 0; y < H; y += step) {
        let f = f0; trees.forEach(t => { f += lr * predictTree(t, (x + step / 2) / W, (y + step / 2) / H); });
        const p = sigmoid(f);
        ctx.fillStyle = `rgba(${Math.round(182*(1-p)+59*p)},${Math.round(83*(1-p)+122*p)},${Math.round(43*(1-p)+87*p)},0.18)`;
        ctx.fillRect(x, y, step, step);
      }
    }
  }
  ctx.strokeStyle = 'rgba(110,98,87,0.07)'; ctx.lineWidth = 0.5;
  for (let x = 0; x < W; x += 50) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }
  for (let y = 0; y < H; y += 50) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }
  pts.forEach(pt => {
    const px = pt.x*(W/600), py = pt.y*(H/400), col = pt.label === 0 ? '#B6532B' : '#3B7A57';
    ctx.beginPath(); ctx.arc(px, py, 7, 0, Math.PI*2); ctx.fillStyle = col; ctx.fill();
    ctx.strokeStyle = '#FAF6EE'; ctx.lineWidth = 1.5; ctx.stroke();
  });
};

// ─────────────────────────────────────────────────────────────────────────────
// Accuracy line chart (3 lines for compare)
// ─────────────────────────────────────────────────────────────────────────────
const AccChart: React.FC<{ series: { label: string; color: string; values: number[] }[]; step: number }> = ({ series, step }) => {
  const W = 420, H = 90, PL = 28, PR = 10, PT = 8, PB = 22;
  const cW = W - PL - PR, cH = H - PT - PB;
  const maxN = Math.max(...series.map(s => s.values.length), 2);
  const toX = (i: number) => PL + (i / (maxN - 1)) * cW;
  const toY = (v: number) => PT + cH - (v / 100) * cH;
  const vis = Math.min(step + 1, maxN);
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ fontFamily: 'JetBrains Mono, monospace' }}>
      <rect x={PL} y={PT} width={cW} height={cH} fill="#FAF6EE" rx="3" />
      {[0, 50, 100].map(v => (
        <g key={v}>
          <line x1={PL} y1={toY(v)} x2={PL+cW} y2={toY(v)} stroke="#E5DDD0" strokeDasharray="3,4" />
          <text x={PL-4} y={toY(v)+4} fill="#BDB0A2" fontSize="7.5" textAnchor="end">{v}</text>
        </g>
      ))}
      {series.map(({ label, color, values }) => {
        const slice = values.slice(0, vis);
        if (slice.length < 2) return null;
        const pts = slice.map((v, i) => `${toX(i)},${toY(v)}`).join(' ');
        const lx = toX(slice.length - 1), ly = toY(slice[slice.length - 1]);
        return (
          <g key={label}>
            <polyline points={pts} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            <circle cx={lx} cy={ly} r={3.5} fill={color} />
          </g>
        );
      })}
      {maxN > 1 && <line x1={toX(Math.min(step, maxN-1))} y1={PT} x2={toX(Math.min(step, maxN-1))} y2={PT+cH} stroke="#2E251E" strokeWidth="1" strokeOpacity="0.12" />}
      <line x1={PL} y1={PT} x2={PL} y2={PT+cH} stroke="#D5C9B8" strokeWidth="1" />
      <line x1={PL} y1={PT+cH} x2={PL+cW} y2={PT+cH} stroke="#D5C9B8" strokeWidth="1" />
      <text x={PL+cW/2} y={H-3} fill="#BDB0A2" fontSize="8" textAnchor="middle">Boosting Rounds</text>
      <text x={8} y={PT+cH/2} fill="#BDB0A2" fontSize="8" textAnchor="middle" transform={`rotate(-90, 8, ${PT+cH/2})`}>Acc %</text>
    </svg>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Main: Advanced Boosting Simulator (XGBoost / LightGBM / CatBoost)
// ─────────────────────────────────────────────────────────────────────────────
export const GradientBoostingSimulator: React.FC = () => {
  const [algo, setAlgo] = useState<AlgoMode>('xgb');
  const [showCompare, setShowCompare] = useState(false);
  const [pts, setPts] = useState<Point[]>(defaultPoints);
  const [activeLabel, setActiveLabel] = useState<0 | 1>(0);
  const [nTrees, setNTrees] = useState(20);
  const [lr, setLr] = useState(0.15);
  const [maxDepth, setMaxDepth] = useState(3);
  const [maxLeaves, setMaxLeaves] = useState(4);
  const [lambda, setLambda] = useState(1.0);
  const [gamma, setGamma] = useState(0.0);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(600);
  const playRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mainRef = useRef<HTMLCanvasElement>(null);
  const cmpRefs = [useRef<HTMLCanvasElement>(null), useRef<HTMLCanvasElement>(null), useRef<HTMLCanvasElement>(null)];

  const normPts = React.useMemo(() => pts.map(p => ({ x: p.x / 600, y: p.y / 400 })), [pts]);
  const models = React.useMemo<Record<AlgoMode, TrainedModel>>(() => ({
    xgb: trainModel(pts, 'xgb', nTrees, lr, maxDepth, maxLeaves, lambda, gamma, normPts),
    lgb: trainModel(pts, 'lgb', nTrees, lr, maxDepth, maxLeaves, lambda, gamma, normPts),
    cat: trainModel(pts, 'cat', nTrees, lr, maxDepth, maxLeaves, lambda, gamma, normPts),
  }), [pts, nTrees, lr, maxDepth, maxLeaves, lambda, gamma, normPts]);

  const maxStep = Math.max(nTrees - 1, 0);
  const safeStep = Math.min(step, maxStep);
  const col = ALGO[algo];

  const stopPlay = useCallback(() => { if (playRef.current) clearTimeout(playRef.current); setPlaying(false); }, []);
  const doReset = useCallback(() => { stopPlay(); setStep(0); }, [stopPlay]);

  useEffect(() => {
    if (!playing) return;
    if (safeStep >= maxStep) { stopPlay(); return; }
    playRef.current = setTimeout(() => setStep(s => s + 1), speed);
    return () => { if (playRef.current) clearTimeout(playRef.current); };
  }, [playing, safeStep, maxStep, speed, stopPlay]);

  useEffect(() => {
    if (showCompare) return;
    const canvas = mainRef.current; if (!canvas) return;
    const ctx = canvas.getContext('2d'); if (!ctx) return;
    canvas.width = 600; canvas.height = 400;
    const m = models[algo];
    drawCanvas(ctx, 600, 400, m.f0, m.trees.slice(0, safeStep + 1), lr, pts);
  }, [safeStep, showCompare, algo, models, lr, pts]);

  useEffect(() => {
    if (!showCompare) return;
    (['xgb', 'lgb', 'cat'] as AlgoMode[]).forEach((a, idx) => {
      const canvas = cmpRefs[idx].current; if (!canvas) return;
      const ctx = canvas.getContext('2d'); if (!ctx) return;
      canvas.width = 360; canvas.height = 260;
      const m = models[a];
      drawCanvas(ctx, 360, 260, m.f0, m.trees.slice(0, safeStep + 1), lr, pts);
    });
  }, [safeStep, showCompare, models, lr, pts]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = mainRef.current; if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    setPts(prev => [...prev, { x: ((e.clientX - rect.left) / rect.width) * 600, y: ((e.clientY - rect.top) / rect.height) * 400, label: activeLabel }]);
    doReset();
  };

  const curAcc = models[algo].accHistory[safeStep] ?? 0;
  const curLoss = models[algo].lossHistory[safeStep] ?? 0;

  const Slider = ({ label, val, min, max, st, fmt, set }: { label: string; val: number; min: number; max: number; st: number; fmt: (v: number) => string; set: (v: number) => void }) => (
    <div className="mb-3">
      <div className="flex justify-between items-center mb-1.5">
        <span style={{ fontSize: 11, color: '#6E6257', fontFamily: 'JetBrains Mono, monospace' }}>{label}</span>
        <span style={{ fontSize: 12, fontWeight: 700, color: col.color, background: col.bg, border: `1px solid ${col.border}`, padding: '1px 7px', borderRadius: 6, fontFamily: 'JetBrains Mono, monospace' }}>{fmt(val)}</span>
      </div>
      <input type="range" min={min} max={max} step={st} value={val}
        onChange={e => { set(parseFloat(e.target.value)); doReset(); }}
        style={{ width: '100%', accentColor: col.color, cursor: 'pointer' }} />
    </div>
  );

  return (
    <div className="grid grid-cols-1 md:grid-cols-12 gap-5 p-5">

      {/* ── Sidebar ── */}
      <div className="md:col-span-4 flex flex-col gap-4">

        {/* Algorithm picker + view toggle */}
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-2xl p-5 shadow-sm">
          <div className="text-[10px] font-mono text-[#9E9183] uppercase tracking-widest mb-3">Algorithm</div>
          <div className="flex flex-col gap-2 mb-4">
            {(['xgb', 'lgb', 'cat'] as AlgoMode[]).map(a => {
              const c = ALGO[a]; const active = algo === a;
              return (
                <button key={a} onClick={() => { setAlgo(a); if (!showCompare) doReset(); }}
                  className="flex items-start gap-2.5 p-3 rounded-xl text-left transition-all duration-150 cursor-pointer"
                  style={{ background: active ? c.bg : 'transparent', border: `1px solid ${active ? c.border : '#E5DDD0'}` }}>
                  <span style={{ width: 7, height: 7, borderRadius: '50%', background: c.color, marginTop: 4, flexShrink: 0, display: 'block' }} />
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: active ? c.color : '#2E251E' }}>{c.name}</div>
                    <div style={{ fontSize: 10, color: '#9E9183', marginTop: 1, fontFamily: 'JetBrains Mono, monospace' }}>{c.desc}</div>
                  </div>
                </button>
              );
            })}
          </div>
          <div className="flex gap-2">
            {(['Single', 'Compare All'] as const).map((m, i) => {
              const isCmp = i === 1, active = showCompare === isCmp;
              return (
                <button key={m} onClick={() => { setShowCompare(isCmp); doReset(); }}
                  className="flex-1 py-2 rounded-xl text-xs font-bold transition-all cursor-pointer"
                  style={{ background: active ? '#B6532B' : 'transparent', color: active ? '#fff' : '#6E6257', border: `1px solid ${active ? '#B6532B' : '#E5DDD0'}` }}>
                  {m}
                </button>
              );
            })}
          </div>
        </div>

        {/* Playback */}
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-2xl p-5 shadow-sm">
          <div className="text-[10px] font-mono text-[#9E9183] uppercase tracking-widest mb-3">Playback</div>
          <div className="flex gap-2 mb-4">
            <button onClick={doReset} className="w-9 h-9 flex items-center justify-center rounded-lg border border-[#E5DDD0] bg-[#F4EFE6] text-[#6E6257] hover:bg-[#EDE6DA] cursor-pointer"><RotateCcw size={14} /></button>
            <button onClick={() => { if (playing) { stopPlay(); return; } if (safeStep >= maxStep) setStep(0); setPlaying(true); }}
              className="flex-1 h-9 flex items-center justify-center gap-2 rounded-lg font-bold text-sm cursor-pointer"
              style={{ background: playing ? col.color : col.bg, color: playing ? '#fff' : col.color, border: `1px solid ${col.border}` }}>
              {playing ? <><Pause size={14} />Pause</> : <><Play size={14} />Auto-Run</>}
            </button>
            <button onClick={() => { stopPlay(); setStep(s => Math.min(s+1, maxStep)); }} className="w-9 h-9 flex items-center justify-center rounded-lg border border-[#E5DDD0] bg-[#F4EFE6] text-[#6E6257] hover:bg-[#EDE6DA] cursor-pointer"><ChevronRight size={14} /></button>
          </div>
          <div className="mb-3">
            <div className="flex justify-between items-center mb-1.5">
              <span style={{ fontSize: 11, color: '#6E6257', fontFamily: 'JetBrains Mono, monospace' }}>Round</span>
              <span style={{ fontSize: 12, fontWeight: 700, color: col.color, background: col.bg, border: `1px solid ${col.border}`, padding: '1px 7px', borderRadius: 6, fontFamily: 'JetBrains Mono, monospace' }}>{safeStep+1}/{nTrees}</span>
            </div>
            <input type="range" min={0} max={maxStep} value={safeStep} onChange={e => { stopPlay(); setStep(Number(e.target.value)); }} style={{ width: '100%', accentColor: col.color, cursor: 'pointer' }} />
          </div>
          <div>
            <div className="flex justify-between items-center mb-1.5">
              <span style={{ fontSize: 11, color: '#6E6257', fontFamily: 'JetBrains Mono, monospace' }}>Speed</span>
              <span style={{ fontSize: 10, color: '#9E9183', fontFamily: 'JetBrains Mono, monospace' }}>{speed <= 300 ? 'Fast' : speed <= 700 ? 'Normal' : 'Slow'}</span>
            </div>
            <input type="range" min={100} max={1200} step={100} value={speed} onChange={e => setSpeed(Number(e.target.value))} style={{ width: '100%', accentColor: '#C18C3B', cursor: 'pointer' }} />
          </div>
        </div>

        {/* Parameters */}
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-2xl p-5 shadow-sm">
          <div className="text-[10px] font-mono text-[#9E9183] uppercase tracking-widest mb-3">Parameters</div>
          <Slider label="Trees" val={nTrees} min={2} max={40} st={1} fmt={v => `${v}`} set={setNTrees} />
          <Slider label="Learning Rate (η)" val={lr} min={0.01} max={1} st={0.01} fmt={v => v.toFixed(2)} set={setLr} />
          <Slider label="Max Depth" val={maxDepth} min={1} max={6} st={1} fmt={v => `${v}`} set={setMaxDepth} />
          {algo === 'lgb' && <Slider label="Max Leaves" val={maxLeaves} min={2} max={12} st={1} fmt={v => `${v}`} set={setMaxLeaves} />}
          {algo === 'xgb' && <>
            <Slider label="L2 Reg (λ)" val={lambda} min={0} max={10} st={0.5} fmt={v => v.toFixed(1)} set={setLambda} />
            <Slider label="Min Gain (γ)" val={gamma} min={0} max={5} st={0.1} fmt={v => v.toFixed(1)} set={setGamma} />
          </>}
        </div>

        {/* Add Points */}
        {!showCompare && (
          <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-2xl p-4 shadow-sm">
            <div className="text-[10px] font-mono text-[#9E9183] uppercase tracking-widest mb-3">Add Points (click canvas)</div>
            <div className="flex gap-2 mb-2">
              {([0, 1] as const).map(lbl => (
                <button key={lbl} onClick={() => setActiveLabel(lbl)}
                  className="flex-1 py-2 rounded-lg text-xs font-bold cursor-pointer flex items-center justify-center gap-1.5"
                  style={{ background: activeLabel === lbl ? (lbl === 0 ? '#B6532B' : '#3B7A57') : 'transparent', color: activeLabel === lbl ? '#fff' : (lbl === 0 ? '#B6532B' : '#3B7A57'), border: `1.5px solid ${lbl === 0 ? '#B6532B' : '#3B7A57'}` }}>
                  <span style={{ width: 7, height: 7, borderRadius: '50%', background: activeLabel === lbl ? '#fff' : (lbl === 0 ? '#B6532B' : '#3B7A57'), display: 'inline-block' }} />
                  Class {lbl}
                </button>
              ))}
            </div>
            <div className="flex gap-2">
              <button onClick={() => { setPts(defaultPoints); doReset(); }} className="flex-1 py-2 rounded-lg border border-[#E5DDD0] bg-[#F4EFE6] hover:bg-[#EDE6DA] text-[#6E6257] text-xs font-bold flex items-center justify-center gap-1 cursor-pointer"><Sparkles size={12} />Reset</button>
              <button onClick={() => { setPts([]); doReset(); }} className="flex-1 py-2 rounded-lg border border-[#E5DDD0] bg-[#F4EFE6] hover:bg-[#EDE6DA] text-[#B6532B] text-xs font-bold flex items-center justify-center gap-1 cursor-pointer"><Trash2 size={12} />Clear</button>
            </div>
          </div>
        )}
      </div>

      {/* ── Main ── */}
      <div className="md:col-span-8 flex flex-col gap-4">

        {/* Metrics (single mode) */}
        {!showCompare && (
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: 'Accuracy', value: `${curAcc.toFixed(1)}%`, color: col.color },
              { label: 'Log-Loss', value: curLoss.toFixed(3), color: '#C18C3B' },
              { label: 'Round', value: `${safeStep+1} / ${nTrees}`, color: '#6E6257' },
            ].map(({ label, value, color }) => (
              <div key={label} className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl px-4 py-3 text-center shadow-sm">
                <div style={{ fontSize: 20, fontWeight: 800, color, fontFamily: 'JetBrains Mono, monospace', lineHeight: 1 }}>{value}</div>
                <div style={{ fontSize: 10, color: '#9E9183', marginTop: 3, textTransform: 'uppercase', letterSpacing: '0.07em', fontFamily: 'JetBrains Mono, monospace' }}>{label}</div>
              </div>
            ))}
          </div>
        )}

        {/* Canvas or Compare */}
        {!showCompare ? (
          <>
            <div className="bg-[#F4EFE6] border border-[#E5DDD0] rounded-2xl p-2 shadow-sm relative">
              <div style={{ position: 'absolute', top: 14, left: 14, zIndex: 5, background: 'rgba(46,37,30,0.82)', padding: '5px 10px', borderRadius: 8, fontSize: 10, fontFamily: 'JetBrains Mono, monospace', color: '#FAF6EE', display: 'flex', gap: 8 }}>
                <span style={{ fontWeight: 700, color: col.color }}>{col.name}</span>
                <span style={{ opacity: 0.5 }}>·</span>
                <span>Round {safeStep+1}/{nTrees}</span>
                <span style={{ opacity: 0.5 }}>·</span>
                <span style={{ color: '#B6532B' }}>{curAcc.toFixed(0)}% acc</span>
              </div>
              <div style={{ position: 'absolute', bottom: 12, right: 14, zIndex: 5, background: 'rgba(46,37,30,0.55)', padding: '3px 8px', borderRadius: 6, fontSize: 9, fontFamily: 'JetBrains Mono, monospace', color: 'rgba(250,246,238,0.7)' }}>Click to add points</div>
              <canvas ref={mainRef} onClick={handleClick} className="block w-full rounded-xl cursor-crosshair" style={{ aspectRatio: '3/2' }} />
            </div>

            {/* Accuracy chart */}
            <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-4 shadow-sm">
              <div className="flex items-center gap-1.5 mb-2">
                <TrendingUp size={12} color={col.color} />
                <span style={{ fontSize: 10, color: '#9E9183', fontFamily: 'JetBrains Mono, monospace', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Accuracy over rounds</span>
              </div>
              <AccChart series={[{ label: col.name, color: col.color, values: models[algo].accHistory }]} step={safeStep} />
            </div>
          </>
        ) : (
          /* Compare mode */
          <>
            {/* Unified accuracy chart */}
            <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-4 shadow-sm">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-1.5">
                  <TrendingUp size={12} color="#B6532B" />
                  <span style={{ fontSize: 10, color: '#9E9183', fontFamily: 'JetBrains Mono, monospace', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Accuracy — Round {safeStep+1}/{nTrees}</span>
                </div>
                <div className="flex gap-4">
                  {(['xgb','lgb','cat'] as AlgoMode[]).map(a => (
                    <span key={a} style={{ fontSize: 10, color: ALGO[a].color, fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 3 }}>
                      <span style={{ display: 'inline-block', width: 8, height: 3, borderRadius: 2, background: ALGO[a].color }} />
                      {ALGO[a].name}: {(models[a].accHistory[safeStep] ?? 0).toFixed(1)}%
                    </span>
                  ))}
                </div>
              </div>
              <AccChart
                series={(['xgb','lgb','cat'] as AlgoMode[]).map(a => ({ label: ALGO[a].name, color: ALGO[a].color, values: models[a].accHistory }))}
                step={safeStep}
              />
            </div>

            {/* 3 canvases */}
            <div className="grid grid-cols-3 gap-3">
              {(['xgb','lgb','cat'] as AlgoMode[]).map((a, idx) => {
                const c = ALGO[a];
                const acc = models[a].accHistory[safeStep] ?? 0;
                return (
                  <div key={a} className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl overflow-hidden shadow-sm" style={{ borderTopColor: c.color, borderTopWidth: 2 }}>
                    <div style={{ padding: '7px 10px 4px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: 12, fontWeight: 800, color: c.color }}>{c.name}</span>
                      <span style={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, color: '#3B7A57' }}>{acc.toFixed(1)}%</span>
                    </div>
                    <canvas ref={cmpRefs[idx]} style={{ display: 'block', width: '100%', aspectRatio: '10/7' }} />
                    <div style={{ padding: '4px 10px 7px', fontSize: 9, color: '#9E9183', fontFamily: 'JetBrains Mono, monospace', textAlign: 'center' }}>{c.desc}</div>
                  </div>
                );
              })}
            </div>

            {/* Insight card */}
            <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-4 shadow-sm">
              <div className="flex items-center gap-1.5 mb-3">
                <Info size={13} color="#B6532B" />
                <span style={{ fontSize: 11, fontWeight: 700, color: '#2E251E' }}>Architecture Comparison</span>
              </div>
              <div className="grid grid-cols-3 gap-4" style={{ fontSize: 11, color: '#6E6257', lineHeight: 1.6 }}>
                <div><span style={{ fontWeight: 700, color: ALGO.xgb.color }}>XGBoost</span> — Level-wise growth using Newton's method (2nd-order Hessians). L2 regularization (λ, γ) controls leaf weight size.</div>
                <div><span style={{ fontWeight: 700, color: ALGO.lgb.color }}>LightGBM</span> — Leaf-wise (best-first) growth: always splits the leaf with the biggest gain, forming deep asymmetric trees fast.</div>
                <div><span style={{ fontWeight: 700, color: ALGO.cat.color }}>CatBoost</span> — Symmetric (oblivious) trees: same split used at every node on the same depth level, making grid-like boundaries.</div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};
