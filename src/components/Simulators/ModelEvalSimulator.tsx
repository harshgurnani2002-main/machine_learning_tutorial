import React, { useState, useEffect, useRef, useMemo } from 'react';
import { RotateCcw, Eye, EyeOff } from 'lucide-react';

const W = 480, H = 360, PAD = 44;
const NUM_PTS = 2000;
const THRESHOLD_STEPS = 120;

function seededRandom(seed: number) {
  let s = seed | 0;
  return () => { s = (s * 1664525 + 1013904223) | 0; return (s >>> 0) / 4294967296; };
}

function betaSample(alpha: number, beta: number, rng: () => number): number {
  const u = rng(), v = rng();
  const x = Math.pow(u, 1 / alpha);
  const y = Math.pow(v, 1 / beta);
  return x / (x + y + 1e-12);
}

function generateScores(scenario: string, classBalance: number) {
  const rng = seededRandom(73);
  const posCount = Math.round(NUM_PTS * classBalance);
  const negCount = NUM_PTS - posCount;
  const scores: { score: number; label: number }[] = [];

  let posAlpha: number, posBeta: number, negAlpha: number, negBeta: number;
  switch (scenario) {
    case 'perfect':
      posAlpha = 100; posBeta = 1; negAlpha = 1; negBeta = 100; break;
    case 'good':
      posAlpha = 6; posBeta = 2; negAlpha = 2; negBeta = 5; break;
    case 'random':
      posAlpha = 3; posBeta = 3; negAlpha = 3; negBeta = 3; break;
    default: // poor
      posAlpha = 2; posBeta = 5; negAlpha = 5; negBeta = 2; break;
  }

  for (let i = 0; i < posCount; i++) {
    let s = betaSample(posAlpha, posBeta, rng);
    if (scenario === 'poor') s = 1 - s;
    scores.push({ score: Math.max(0, Math.min(1, s)), label: 1 });
  }
  for (let i = 0; i < negCount; i++) {
    let s = betaSample(negAlpha, negBeta, rng);
    if (scenario === 'poor') s = 1 - s;
    scores.push({ score: Math.max(0, Math.min(1, s)), label: 0 });
  }
  return scores;
}

function computeROC(scores: { score: number; label: number }[]) {
  const thresholds: number[] = [];
  for (let i = 0; i <= THRESHOLD_STEPS; i++) thresholds.push(i / THRESHOLD_STEPS);
  const roc: { fpr: number; tpr: number; threshold: number }[] = [];
  const totalPos = scores.filter(s => s.label === 1).length;
  const totalNeg = scores.filter(s => s.label === 0).length;

  for (const t of thresholds) {
    let tp = 0, fp = 0, fn = 0, tn = 0;
    for (const s of scores) {
      if (s.score >= t) {
        if (s.label === 1) tp++; else fp++;
      } else {
        if (s.label === 1) fn++; else tn++;
      }
    }
    const tpr = totalPos > 0 ? tp / totalPos : 0;
    const fpr = totalNeg > 0 ? fp / totalNeg : 0;
    roc.push({ fpr, tpr, threshold: t });
  }
  return { roc, totalPos, totalNeg };
}

function computeAUC(roc: { fpr: number; tpr: number }[]): number {
  let auc = 0;
  for (let i = 1; i < roc.length; i++) {
    auc += (roc[i].tpr + roc[i - 1].tpr) * (roc[i].fpr - roc[i - 1].fpr) / 2;
  }
  return auc;
}

function getConfusionAtThreshold(scores: { score: number; label: number }[], threshold: number) {
  let tp = 0, fp = 0, fn = 0, tn = 0;
  for (const s of scores) {
    if (s.score >= threshold) {
      if (s.label === 1) tp++; else fp++;
    } else {
      if (s.label === 1) fn++; else tn++;
    }
  }
  return { tp, fp, fn, tn };
}

function computeMetrics(tp: number, fp: number, fn: number, tn: number) {
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const accuracy = tp + tn + fp + fn > 0 ? (tp + tn) / (tp + tn + fp + fn) : 0;
  const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
  return { precision, recall, accuracy, f1 };
}

export const ModelEvalSimulator: React.FC = () => {
  const [scenario, setScenario] = useState('good');
  const [threshold, setThreshold] = useState(0.5);
  const [classBalance, setClassBalance] = useState(0.5);
  const [showLearningCurve, setShowLearningCurve] = useState(false);
  const rocCanvasRef = useRef<HTMLCanvasElement>(null);
  const lcCanvasRef = useRef<HTMLCanvasElement>(null);
  const [seed, setSeed] = useState(0);

  const scores = useMemo(() => generateScores(scenario, classBalance), [scenario, classBalance, seed]);
  const { roc, totalPos, totalNeg } = useMemo(() => computeROC(scores), [scores]);
  const auc = useMemo(() => computeAUC(roc), [roc]);
  const cm = useMemo(() => getConfusionAtThreshold(scores, threshold), [scores, threshold]);
  const metrics = useMemo(() => computeMetrics(cm.tp, cm.fp, cm.fn, cm.tn), [cm]);

  const currentRocPoint = roc.find(r => Math.abs(r.threshold - threshold) < 1e-6) || roc[roc.length - 1];

  useEffect(() => {
    const canvas = rocCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    canvas.width = W; canvas.height = H;

    ctx.fillStyle = '#FAF6EE';
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = '#E5DDD0';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 10; i++) {
      const x = PAD + i * (W - 2 * PAD) / 10;
      ctx.beginPath(); ctx.moveTo(x, PAD); ctx.lineTo(x, H - PAD); ctx.stroke();
      const y = PAD + i * (H - 2 * PAD) / 10;
      ctx.beginPath(); ctx.moveTo(PAD, y); ctx.lineTo(W - PAD, y); ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = '#2E251E';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(PAD, PAD); ctx.lineTo(PAD, H - PAD); ctx.lineTo(W - PAD, H - PAD); ctx.stroke();

    ctx.fillStyle = '#6E6257';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('FPR (1 - Specificity)', W / 2, H - 8);
    ctx.save();
    ctx.translate(12, H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('TPR (Sensitivity)', 0, 0);
    ctx.restore();

    for (let i = 0; i <= 10; i += 2) {
      ctx.fillStyle = '#6E6257';
      ctx.font = '9px monospace';
      ctx.textAlign = 'center';
      ctx.fillText((i / 10).toFixed(1), PAD + i * (W - 2 * PAD) / 10, H - PAD + 14);
      ctx.textAlign = 'right';
      ctx.fillText((1 - i / 10).toFixed(1), PAD - 6, PAD + i * (H - 2 * PAD) / 10 + 3);
    }

    // Diagonal reference
    ctx.strokeStyle = '#D5C5B5';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(PAD, H - PAD); ctx.lineTo(W - PAD, PAD); ctx.stroke();
    ctx.setLineDash([]);

    const toX = (fpr: number) => PAD + fpr * (W - 2 * PAD);
    const toY = (tpr: number) => H - PAD - tpr * (H - 2 * PAD);

    // ROC curve fill
    const grad = ctx.createLinearGradient(0, PAD, 0, H - PAD);
    grad.addColorStop(0, 'rgba(182, 83, 43, 0.15)');
    grad.addColorStop(1, 'rgba(182, 83, 43, 0.01)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(0));
    for (const pt of roc) ctx.lineTo(toX(pt.fpr), toY(pt.tpr));
    ctx.lineTo(toX(1), toY(0));
    ctx.closePath();
    ctx.fill();

    // ROC curve
    ctx.strokeStyle = '#B6532B';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 0; i < roc.length; i++) {
      const x = toX(roc[i].fpr), y = toY(roc[i].tpr);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Current threshold point
    const cx = toX(currentRocPoint.fpr);
    const cy = toY(currentRocPoint.tpr);
    ctx.beginPath();
    ctx.arc(cx, cy, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#2E251E';
    ctx.fill();
    ctx.strokeStyle = '#FAF6EE';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Crosshairs
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = 'rgba(46, 37, 30, 0.4)';
    ctx.lineWidth = 0.8;
    ctx.beginPath(); ctx.moveTo(PAD, cy); ctx.lineTo(W - PAD, cy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx, PAD); ctx.lineTo(cx, H - PAD); ctx.stroke();
    ctx.setLineDash([]);

    // AUC text
    ctx.fillStyle = '#2E251E';
    ctx.font = 'bold 11px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`AUC = ${auc.toFixed(3)}`, W - PAD - 110, PAD + 16);

    // TPR/FPR at point
    ctx.font = '10px monospace';
    ctx.fillStyle = '#B6532B';
    ctx.fillText(`TPR = ${currentRocPoint.tpr.toFixed(3)}`, W - PAD - 110, PAD + 32);
    ctx.fillText(`FPR = ${currentRocPoint.fpr.toFixed(3)}`, W - PAD - 110, PAD + 46);

    const scenarioLabel: Record<string, string> = { perfect: 'Perfect', good: 'Good', random: 'Random', poor: 'Poor' };
    ctx.fillStyle = '#6E6257';
    ctx.font = '9px monospace';
    ctx.fillText(`Classifier: ${scenarioLabel[scenario]}  |  Balance: ${(classBalance * 100).toFixed(0)}/${((1 - classBalance) * 100).toFixed(0)}`, PAD, PAD - 6);
  }, [roc, currentRocPoint, auc, scenario, classBalance]);

  useEffect(() => {
    if (!showLearningCurve) return;
    const canvas = lcCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const LW = 360, LH = 180;
    const L_PAD = 38;
    canvas.width = LW; canvas.height = LH;

    ctx.fillStyle = '#FAF6EE';
    ctx.fillRect(0, 0, LW, LH);

    ctx.strokeStyle = '#E5DDD0';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 5; i++) {
      const y = L_PAD + i * (LH - 2 * L_PAD) / 5;
      ctx.beginPath(); ctx.moveTo(L_PAD, y); ctx.lineTo(LW - L_PAD, y); ctx.stroke();
    }

    ctx.strokeStyle = '#2E251E';
    ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(L_PAD, L_PAD); ctx.lineTo(L_PAD, LH - L_PAD); ctx.lineTo(LW - L_PAD, LH - L_PAD); ctx.stroke();

    ctx.fillStyle = '#6E6257';
    ctx.font = '8px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Training examples', LW / 2, LH - 4);
    ctx.save();
    ctx.translate(10, LH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Loss', 0, 0);
    ctx.restore();
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const v = (1 - i / 5).toFixed(1);
      ctx.fillText(v, L_PAD - 4, L_PAD + i * (LH - 2 * L_PAD) / 5 + 3);
    }

    const nPoints = 15;
    const trainLoss: number[] = [];
    const valLoss: number[] = [];

    let baseTrain: number, baseVal: number, gap: number;
    switch (scenario) {
      case 'perfect': baseTrain = 0.01; baseVal = 0.02; gap = 0.01; break;
      case 'good': baseTrain = 0.05; baseVal = 0.12; gap = 0.07; break;
      case 'random': baseTrain = 0.45; baseVal = 0.50; gap = 0.05; break;
      default: baseTrain = 0.02; baseVal = 0.35; gap = 0.33; break;
    }

    for (let i = 1; i <= nPoints; i++) {
      const frac = i / nPoints;
      const decay = Math.exp(-3 * frac);
      const train = baseTrain + (0.7 - baseTrain) * decay + (1 - frac) * 0.05;
      const val = baseVal + (0.7 - baseVal) * decay + gap * (1 - frac) + (1 - frac) * 0.08;
      trainLoss.push(Math.min(1, Math.max(0.001, train)));
      valLoss.push(Math.min(1, Math.max(0.001, val)));
    }

    const lx = (i: number) => L_PAD + (i / (nPoints - 1)) * (LW - 2 * L_PAD);
    const ly = (v: number) => LH - L_PAD - (v / 0.8) * (LH - 2 * L_PAD);

    ctx.strokeStyle = '#3B7A57';
    ctx.lineWidth = 2;
    ctx.beginPath();
    trainLoss.forEach((v, i) => { const x = lx(i), y = ly(v); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    ctx.stroke();

    ctx.strokeStyle = '#B6532B';
    ctx.lineWidth = 2;
    ctx.beginPath();
    valLoss.forEach((v, i) => { const x = lx(i), y = ly(v); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    ctx.stroke();

    ctx.fillStyle = '#3B7A57';
    ctx.font = '8px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('Train', LW - L_PAD - 32, L_PAD + 10);
    ctx.fillStyle = '#B6532B';
    ctx.fillText('Val', LW - L_PAD - 32, L_PAD + 22);

    const lastTrainGap = Math.abs(valLoss[valLoss.length - 1] - trainLoss[trainLoss.length - 1]);
    ctx.fillStyle = '#2E251E';
    ctx.font = '9px monospace';
    ctx.textAlign = 'left';
    const diagnosis = lastTrainGap > 0.15 ? 'Overfitting (high variance)' : lastTrainGap > 0.06 ? 'Slight overfit' : valLoss[valLoss.length - 1] > 0.4 ? 'Underfitting (high bias)' : 'Good fit';
    ctx.fillText(`Diagnosis: ${diagnosis}`, L_PAD, LH - L_PAD + 16);
  }, [showLearningCurve, scenario]);

  const scenarioLabel: Record<string, string> = { perfect: 'Perfect', good: 'Good', random: 'Random', poor: 'Poor' };

  return (
    <div className="grid grid-cols-1 md:grid-cols-12 gap-6 p-4">
      {/* Left column: ROC canvas + learning curve */}
      <div className="md:col-span-7 space-y-4">
        {/* ROC Curve */}
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-inner">
          <h5 className="text-[#2E251E] text-xs font-semibold font-mono mb-3">ROC Curve — TPR vs FPR across thresholds</h5>
          <canvas ref={rocCanvasRef} className="w-full rounded-lg" style={{ maxWidth: W }} />
          <div className="mt-3">
            <div className="flex justify-between text-xs font-mono mb-1">
              <span className="text-[#6E6257]">Threshold</span>
              <span className="text-[#B6532B] font-bold">{threshold.toFixed(2)}</span>
            </div>
            <input
              type="range" min="0" max="1" step="0.005" value={threshold}
              onChange={e => setThreshold(parseFloat(e.target.value))}
              className="w-full accent-[#B6532B]"
            />
            <div className="flex justify-between text-[10px] text-[#6E6257] font-mono mt-1">
              <span>0 (all positive)</span>
              <span>1 (all negative)</span>
            </div>
          </div>
        </div>

        {/* Learning Curve */}
        {showLearningCurve && (
          <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-inner">
            <h5 className="text-[#2E251E] text-xs font-semibold font-mono mb-3">Learning Curve — Train vs Validation Loss</h5>
            <canvas ref={lcCanvasRef} className="w-full rounded-lg" style={{ maxWidth: 360 }} />
          </div>
        )}
      </div>

      {/* Right column: Confusion Matrix + Metrics + Controls */}
      <div className="md:col-span-5 space-y-4">
        {/* Controls */}
        <div className="bg-[#F4EFE6] border border-[#E5DDD0] p-4 rounded-2xl space-y-3">
          <div className="flex justify-between items-center">
            <h4 className="text-[#2E251E] font-bold text-sm tracking-wide">Controls</h4>
            <button
              onClick={() => { setSeed(s => s + 1); setThreshold(0.5); }}
              className="flex items-center gap-1 text-[10px] font-mono text-[#6E6257] hover:text-[#B6532B] transition-colors"
            >
              <RotateCcw className="w-3 h-3" /> Reset
            </button>
          </div>

          <div>
            <div className="flex justify-between text-xs font-mono mb-1">
              <span className="text-[#6E6257]">Classifier</span>
              <span className="text-[#2E251E] font-semibold">{scenarioLabel[scenario]}</span>
            </div>
            <select
              value={scenario}
              onChange={e => setScenario(e.target.value)}
              className="w-full bg-[#FAF6EE] border border-[#E5DDD0] rounded-lg p-1.5 text-xs font-mono text-[#2E251E] outline-none focus:border-[#B6532B]"
            >
              <option value="perfect">Perfect</option>
              <option value="good">Good</option>
              <option value="random">Random</option>
              <option value="poor">Worse-than-random</option>
            </select>
          </div>

          <div>
            <div className="flex justify-between text-xs font-mono mb-1">
              <span className="text-[#6E6257]">Class Balance</span>
              <span className="text-[#2E251E] font-semibold">{(classBalance * 100).toFixed(0)}% / {((1 - classBalance) * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range" min="0.1" max="0.9" step="0.05" value={classBalance}
              onChange={e => setClassBalance(parseFloat(e.target.value))}
              className="w-full accent-[#B6532B]"
            />
            <div className="flex justify-between text-[10px] text-[#6E6257] font-mono mt-1">
              <span>90/10 (Imbalanced)</span>
              <span>50/50 (Balanced)</span>
            </div>
          </div>

          <button
            onClick={() => setShowLearningCurve(v => !v)}
            className="flex items-center gap-2 w-full bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-2 text-xs font-mono text-[#2E251E] hover:bg-[#F0E8D8] transition-colors"
          >
            {showLearningCurve ? <EyeOff className="w-3.5 h-3.5 text-[#B6532B]" /> : <Eye className="w-3.5 h-3.5 text-[#6E6257]" />}
            {showLearningCurve ? 'Hide Learning Curve' : 'Show Learning Curve'}
          </button>
        </div>

        {/* Confusion Matrix */}
        <div className="bg-[#F4EFE6] border border-[#E5DDD0] p-4 rounded-2xl">
          <h4 className="text-[#2E251E] font-bold text-xs font-mono mb-2.5 tracking-wide">Confusion Matrix</h4>
          <div className="grid grid-cols-2 gap-1.5">
            <div className="bg-[#FAF6EE] text-center p-3 rounded-lg border border-[#E5DDD0]">
              <p className="text-[9px] text-[#6E6257] font-mono mb-1">Actual Pos<br />Predicted Pos</p>
              <p className="text-lg font-bold text-[#3B7A57]">{cm.tp}</p>
              <p className="text-[9px] text-[#6E6257] font-mono">TP</p>
            </div>
            <div className="bg-[#FAF6EE] text-center p-3 rounded-lg border border-[#E5DDD0]">
              <p className="text-[9px] text-[#6E6257] font-mono mb-1">Actual Pos<br />Predicted Neg</p>
              <p className="text-lg font-bold text-[#A63A2B]">{cm.fn}</p>
              <p className="text-[9px] text-[#6E6257] font-mono">FN</p>
            </div>
            <div className="bg-[#FAF6EE] text-center p-3 rounded-lg border border-[#E5DDD0]">
              <p className="text-[9px] text-[#6E6257] font-mono mb-1">Actual Neg<br />Predicted Pos</p>
              <p className="text-lg font-bold text-[#A63A2B]">{cm.fp}</p>
              <p className="text-[9px] text-[#6E6257] font-mono">FP</p>
            </div>
            <div className="bg-[#FAF6EE] text-center p-3 rounded-lg border border-[#E5DDD0]">
              <p className="text-[9px] text-[#6E6257] font-mono mb-1">Actual Neg<br />Predicted Neg</p>
              <p className="text-lg font-bold text-[#3B7A57]">{cm.tn}</p>
              <p className="text-[9px] text-[#6E6257] font-mono">TN</p>
            </div>
          </div>
          <div className="flex justify-between text-[10px] font-mono text-[#6E6257] mt-2">
            <span>Total Pos: {totalPos}</span>
            <span>Total Neg: {totalNeg}</span>
          </div>
        </div>

        {/* Metrics Bars */}
        <div className="bg-[#F4EFE6] border border-[#E5DDD0] p-4 rounded-2xl">
          <h4 className="text-[#2E251E] font-bold text-xs font-mono mb-2.5 tracking-wide">Metrics at Threshold {threshold.toFixed(2)}</h4>
          <div className="space-y-2">
            {[
              { label: 'Accuracy', value: metrics.accuracy, color: '#3B7A57' },
              { label: 'Precision', value: metrics.precision, color: '#C18C3B' },
              { label: 'Recall', value: metrics.recall, color: '#B6532B' },
              { label: 'F1 Score', value: metrics.f1, color: '#2E251E' },
            ].map(({ label, value, color }) => (
              <div key={label}>
                <div className="flex justify-between text-[10px] font-mono mb-0.5">
                  <span className="text-[#6E6257]">{label}</span>
                  <span className="font-bold" style={{ color }}>{value.toFixed(3)}</span>
                </div>
                <div className="h-2.5 bg-[#E5DDD0] rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-150"
                    style={{ width: `${value * 100}%`, backgroundColor: color }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
