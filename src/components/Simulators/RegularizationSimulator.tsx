import React, { useState, useEffect, useRef } from 'react';
import { Sliders, RotateCcw } from 'lucide-react';

type RegType = 'none' | 'l1' | 'l2';

// Simulated "true" model with 8 features; some are relevant, some noise
const TRUE_COEFFICIENTS = [2.8, -1.5, 0.0, 0.0, 1.2, -0.8, 0.0, 0.05];
const FEATURE_NAMES = ['x₁', 'x₂', 'x₃', 'x₄', 'x₅', 'x₆', 'x₇', 'x₈'];

function computeCoefficients(lambda: number, regType: RegType): number[] {
  if (regType === 'none' || lambda === 0) {
    // Overfitting: add noise to true coefficients
    return TRUE_COEFFICIENTS.map((c, i) =>
      c + (i % 2 === 0 ? 0.3 : -0.4) + (Math.random() - 0.5) * 0.2
    );
  } else if (regType === 'l2') {
    // Ridge: shrinks all coefficients proportionally, never to 0
    const shrink = 1 / (1 + lambda * 2);
    return TRUE_COEFFICIENTS.map(c => c * shrink + (Math.random() - 0.5) * 0.05);
  } else {
    // Lasso: shrinks small coefficients to exactly 0 (soft-thresholding)
    return TRUE_COEFFICIENTS.map(c => {
      const threshold = lambda * 0.8;
      if (Math.abs(c) <= threshold) return 0;
      return c > 0 ? c - threshold : c + threshold;
    });
  }
}

function computeLoss(coeffs: number[], lambda: number, regType: RegType) {
  const baseMSE = coeffs.reduce((acc, c, i) => acc + (c - TRUE_COEFFICIENTS[i]) ** 2, 0) / coeffs.length;
  const trainLoss = Math.max(0.02, baseMSE + (regType === 'none' ? 0.02 : 0.06));
  const testLoss = Math.max(0.05, regType === 'none'
    ? 0.6 - lambda * 0.1 + 0.4  // overfitting: high test loss
    : 0.2 + Math.abs(lambda - 0.3) * 0.5  // regularized: sweet spot at some lambda
  );
  const penalty = regType === 'l2'
    ? lambda * coeffs.reduce((s, c) => s + c * c, 0)
    : lambda * coeffs.reduce((s, c) => s + Math.abs(c), 0);
  return { trainLoss, testLoss, penalty };
}

export const RegularizationSimulator: React.FC = () => {
  const [lambda, setLambda] = useState(0.1);
  const [regType, setRegType] = useState<RegType>('l2');
  const coeffsCanvasRef = useRef<HTMLCanvasElement>(null);
  const lossCanvasRef = useRef<HTMLCanvasElement>(null);

  const coeffs = computeCoefficients(lambda, regType);
  const { trainLoss, testLoss, penalty } = computeLoss(coeffs, lambda, regType);

  // Draw coefficient bar chart
  useEffect(() => {
    const canvas = coeffsCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width = 400;
    const H = canvas.height = 220;

    ctx.fillStyle = '#FAF6EE';
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = '#E5DDD0';
    ctx.lineWidth = 0.5;
    for (let y = 0; y < H; y += 40) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    // Zero baseline
    const baseline = H / 2;
    ctx.strokeStyle = '#2E251E';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(20, baseline); ctx.lineTo(W - 10, baseline); ctx.stroke();

    const barW = (W - 40) / (coeffs.length * 2);
    const maxVal = 3.5;

    coeffs.forEach((c, i) => {
      const x = 30 + i * (barW * 2 + 4);
      const barHeight = (Math.abs(c) / maxVal) * (H / 2 - 20);

      // True coefficient (ghost bar)
      const trueH = (Math.abs(TRUE_COEFFICIENTS[i]) / maxVal) * (H / 2 - 20);
      if (trueH > 1) {
        ctx.fillStyle = 'rgba(110, 98, 87, 0.15)';
        ctx.fillRect(x, TRUE_COEFFICIENTS[i] >= 0 ? baseline - trueH : baseline, barW, trueH);
      }

      // Model coefficient bar
      const isZeroed = Math.abs(c) < 0.01;
      ctx.fillStyle = isZeroed ? '#E5DDD0' : (c >= 0 ? '#B6532B' : '#C18C3B');
      ctx.fillRect(x, c >= 0 ? baseline - barHeight : baseline, barW, Math.max(2, barHeight));

      // Feature label
      ctx.fillStyle = '#6E6257';
      ctx.font = 'bold 9px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(FEATURE_NAMES[i], x + barW / 2, H - 6);

      // Value label
      if (Math.abs(c) > 0.1) {
        ctx.fillStyle = '#2E251E';
        ctx.font = '9px monospace';
        ctx.fillText(c.toFixed(1), x + barW / 2, c >= 0 ? baseline - barHeight - 4 : baseline + barHeight + 10);
      }
    });

    // Legend
    ctx.fillStyle = 'rgba(110, 98, 87, 0.4)';
    ctx.fillRect(W - 100, 8, 12, 10);
    ctx.fillStyle = '#6E6257';
    ctx.font = '9px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('True coeff', W - 84, 17);

  }, [coeffs, regType, lambda]);

  // Draw train vs test loss curve across lambda values
  useEffect(() => {
    const canvas = lossCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width = 400;
    const H = canvas.height = 140;

    ctx.fillStyle = '#FAF6EE';
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = '#E5DDD0';
    ctx.lineWidth = 0.5;
    for (let y = 0; y < H; y += 30) {
      ctx.beginPath(); ctx.moveTo(20, y); ctx.lineTo(W, y); ctx.stroke();
    }

    const nSteps = 60;
    const lambdaMax = 1.0;

    const drawCurve = (color: string, dash: number[], getValue: (l: number) => number) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.setLineDash(dash);
      ctx.beginPath();
      for (let i = 0; i <= nSteps; i++) {
        const l = (i / nSteps) * lambdaMax;
        const val = getValue(l);
        const x = 20 + (l / lambdaMax) * (W - 30);
        const y = H - 10 - (val * (H - 20));
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    };

    // Train loss (stays low even as lambda increases)
    drawCurve('#B6532B', [], (l) => {
      const c = computeCoefficients(l, regType);
      const base = c.reduce((a, v, i) => a + (v - TRUE_COEFFICIENTS[i]) ** 2, 0) / 8;
      return Math.min(0.95, 0.05 + base * 0.3);
    });

    // Test loss (U-shape — best at sweet spot lambda)
    drawCurve('#3B7A57', [5, 3], (l) => {
      return Math.min(0.9, 0.1 + (l < 0.2 ? (0.2 - l) * 2 : (l - 0.2) * 0.8));
    });

    // Current lambda marker
    const markerX = 20 + (lambda / lambdaMax) * (W - 30);
    ctx.strokeStyle = '#2E251E';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(markerX, 0); ctx.lineTo(markerX, H); ctx.stroke();
    ctx.setLineDash([]);

    // Labels
    ctx.font = 'bold 9px monospace';
    ctx.fillStyle = '#B6532B';
    ctx.textAlign = 'left';
    ctx.fillText('Train Loss', 28, 16);
    ctx.fillStyle = '#3B7A57';
    ctx.fillText('Test Loss ---', 28, 28);
    ctx.fillStyle = '#6E6257';
    ctx.font = '9px monospace';
    ctx.fillText(`λ = ${lambda.toFixed(2)}`, markerX + 3, H - 14);
    ctx.fillText('λ →', W - 25, H - 5);

  }, [lambda, regType]);

  const reset = () => {
    setLambda(0.1);
    setRegType('l2');
  };

  const sparsity = coeffs.filter(c => Math.abs(c) < 0.05).length;

  return (
    <div className="grid grid-cols-1 md:grid-cols-12 gap-6 p-4">
      {/* Controls */}
      <div className="md:col-span-4 bg-[#F4EFE6] border border-[#E5DDD0] p-5 rounded-2xl flex flex-col justify-between">
        <div className="space-y-5">
          <div>
            <h4 className="text-[#2E251E] font-bold text-base tracking-wide flex items-center gap-2">
              <Sliders className="w-5 h-5 text-[#B6532B]" /> Regularization
            </h4>
            <p className="text-[#6E6257] text-xs mt-1">
              See how L1 (Lasso) and L2 (Ridge) penalties shrink model coefficients to reduce overfitting.
            </p>
          </div>

          <div className="space-y-2">
            <label className="text-xs text-[#2E251E] font-mono font-semibold block">Penalty Type</label>
            <div className="grid grid-cols-3 gap-1.5">
              {(['none', 'l2', 'l1'] as RegType[]).map(t => (
                <button
                  key={t}
                  onClick={() => setRegType(t)}
                  className={`py-2 rounded-xl text-xs font-mono border transition-all ${
                    regType === t
                      ? 'border-[#B6532B] bg-[#B6532B]/10 text-[#B6532B] font-bold'
                      : 'border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257] hover:text-[#2E251E]'
                  }`}
                >
                  {t === 'none' ? 'None' : t.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          <div>
            <div className="flex justify-between text-xs font-mono mb-1.5">
              <span className="text-[#2E251E]">λ Penalty Strength</span>
              <span className="text-[#B6532B] font-bold">{lambda.toFixed(2)}</span>
            </div>
            <input
              type="range" min="0" max="1" step="0.01" value={lambda}
              onChange={e => setLambda(parseFloat(e.target.value))}
              className="w-full accent-[#B6532B]"
              disabled={regType === 'none'}
            />
            <div className="flex justify-between text-[10px] text-[#6E6257] font-mono mt-1">
              <span>0 (No reg)</span>
              <span>1.0 (Strong)</span>
            </div>
          </div>

          <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-3 space-y-2 text-xs font-mono">
            <div className="flex justify-between">
              <span className="text-[#6E6257]">Train Loss:</span>
              <span className="text-[#B6532B] font-bold">{trainLoss.toFixed(3)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#6E6257]">Test Loss:</span>
              <span className="text-[#3B7A57] font-bold">{testLoss.toFixed(3)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#6E6257]">Regularization Penalty:</span>
              <span className="text-[#C18C3B] font-bold">{penalty.toFixed(3)}</span>
            </div>
            {regType === 'l1' && (
              <div className="flex justify-between border-t border-[#E5DDD0] pt-2">
                <span className="text-[#6E6257]">Zeroed Features:</span>
                <span className="text-[#B6532B] font-bold">{sparsity} / 8</span>
              </div>
            )}
          </div>

          {regType === 'l1' && sparsity > 0 && (
            <div className="bg-[#B6532B]/5 border border-[#B6532B]/20 rounded-xl p-3">
              <p className="text-[#B6532B] text-[10px] font-mono font-bold">
                ✓ Lasso zeroed {sparsity} irrelevant feature{sparsity > 1 ? 's' : ''}!
              </p>
              <p className="text-[#6E6257] text-[10px] font-mono mt-0.5">
                Automatic feature selection in action.
              </p>
            </div>
          )}
        </div>

        <button
          onClick={reset}
          className="mt-4 py-2.5 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257] hover:text-[#2E251E] flex items-center justify-center gap-2 text-xs font-mono transition-all"
        >
          <RotateCcw className="w-3.5 h-3.5" /> Reset
        </button>
      </div>

      {/* Visualization */}
      <div className="md:col-span-8 space-y-4">
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-inner">
          <div className="flex justify-between items-center mb-2">
            <h5 className="text-[#2E251E] text-xs font-semibold font-mono">Model Coefficients</h5>
            <span className="text-[10px] font-mono text-[#6E6257]">Grey = true value · Color = model estimate</span>
          </div>
          <canvas ref={coeffsCanvasRef} className="w-full rounded-xl" />
        </div>

        <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-inner">
          <h5 className="text-[#2E251E] text-xs font-semibold font-mono mb-2">Train vs Test Loss Across λ Values</h5>
          <canvas ref={lossCanvasRef} className="w-full rounded-xl" />
          <p className="text-[10px] text-[#6E6257] font-mono mt-1 text-center">
            Best generalization at the sweet-spot λ where test loss is minimized
          </p>
        </div>
      </div>
    </div>
  );
};
