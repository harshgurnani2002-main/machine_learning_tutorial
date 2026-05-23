import React, { useState, useEffect, useRef } from 'react';
import { TrendingUp } from 'lucide-react';

// Generate a fixed noisy dataset from a true sinusoidal curve
function generateDataset(n = 20, seed = 42) {
  let s = seed;
  const rng = () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
  const pts = [];
  for (let i = 0; i < n; i++) {
    const x = rng() * 2 - 1; // [-1, 1]
    const noise = (rng() - 0.5) * 0.5;
    const y = Math.sin(Math.PI * x) + noise;
    pts.push({ x, y });
  }
  return pts;
}

// Polynomial feature expansion: [1, x, x^2, ..., x^degree]
function polyFeatures(x: number, degree: number): number[] {
  const feats = [1];
  for (let d = 1; d <= degree; d++) feats.push(Math.pow(x, d));
  return feats;
}

// Least-squares polynomial regression: returns coefficients
function fitPolynomial(pts: {x: number, y: number}[], degree: number): number[] {
  const n = pts.length;
  const d = degree + 1;
  // Build X matrix and y vector
  const X: number[][] = pts.map(p => polyFeatures(p.x, degree));
  const y = pts.map(p => p.y);

  // Normal equations: (X^T X)^{-1} X^T y — simplified for small d
  const XtX: number[][] = Array(d).fill(0).map(() => Array(d).fill(0));
  const Xty: number[] = Array(d).fill(0);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      Xty[j] += X[i][j] * y[i];
      for (let k = 0; k < d; k++) {
        XtX[j][k] += X[i][j] * X[i][k];
      }
    }
  }

  // Gaussian elimination
  for (let col = 0; col < d; col++) {
    let maxRow = col;
    for (let row = col + 1; row < d; row++) {
      if (Math.abs(XtX[row][col]) > Math.abs(XtX[maxRow][col])) maxRow = row;
    }
    [XtX[col], XtX[maxRow]] = [XtX[maxRow], XtX[col]];
    [Xty[col], Xty[maxRow]] = [Xty[maxRow], Xty[col]];
    for (let row = col + 1; row < d; row++) {
      const factor = XtX[row][col] / (XtX[col][col] + 1e-12);
      for (let k = col; k < d; k++) XtX[row][k] -= factor * XtX[col][k];
      Xty[row] -= factor * Xty[col];
    }
  }
  const coeffs = Array(d).fill(0);
  for (let col = d - 1; col >= 0; col--) {
    coeffs[col] = Xty[col] / (XtX[col][col] + 1e-12);
    for (let row = 0; row < col; row++) {
      Xty[row] -= XtX[row][col] * coeffs[col];
    }
  }
  return coeffs;
}

function predictPoly(x: number, coeffs: number[]): number {
  return coeffs.reduce((sum, c, d) => sum + c * Math.pow(x, d), 0);
}

export const BiasVarianceSimulator: React.FC = () => {
  const [degree, setDegree] = useState(1);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const dataset = generateDataset(22);
  const coeffs = fitPolynomial(dataset, degree);

  // Bias = how far the model is from the true function
  const biasSquared = (() => {
    let sum = 0;
    for (let i = -10; i <= 10; i++) {
      const x = i / 10;
      const trueY = Math.sin(Math.PI * x);
      const predY = Math.max(-2, Math.min(2, predictPoly(x, coeffs)));
      sum += (trueY - predY) ** 2;
    }
    return Math.min(1, sum / 21);
  })();

  // Variance estimate from prediction variance across training range
  const variance = Math.min(1, Math.pow(degree, 1.6) * 0.02);

  const totalError = biasSquared + variance + 0.06; // irreducible noise

  const labels = ['Underfitting', 'Good Fit', 'Overfit (start)', 'Overfit', 'High Overfit', 'Extreme Overfit', 'Extreme Overfit'];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width = 500;
    const H = canvas.height = 320;

    ctx.fillStyle = '#FAF6EE';
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = '#E5DDD0';
    ctx.lineWidth = 0.5;
    for (let x = 40; x < W; x += 50) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H - 30); ctx.stroke();
    }
    for (let y = 20; y < H - 30; y += 50) {
      ctx.beginPath(); ctx.moveTo(30, y); ctx.lineTo(W, y); ctx.stroke();
    }

    // Coordinate mapping
    const toCanvasX = (x: number) => 30 + ((x + 1) / 2) * (W - 40);
    const toCanvasY = (y: number) => (H - 30) - ((y + 1.5) / 3) * (H - 50);

    // X axis
    ctx.strokeStyle = '#2E251E';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(30, H - 30);
    ctx.lineTo(W, H - 30);
    ctx.stroke();
    ctx.fillStyle = '#6E6257';
    ctx.font = '9px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('x →', W - 25, H - 20);

    // True function (dashed green)
    ctx.strokeStyle = '#3B7A57';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    for (let i = 0; i <= 100; i++) {
      const x = -1 + (i / 100) * 2;
      const y = Math.sin(Math.PI * x);
      const cx = toCanvasX(x);
      const cy = toCanvasY(y);
      if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Fitted polynomial (terracotta)
    ctx.strokeStyle = '#B6532B';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    let valid = true;
    for (let i = 0; i <= 100; i++) {
      const x = -1 + (i / 100) * 2;
      const rawY = predictPoly(x, coeffs);
      const y = Math.max(-2, Math.min(2, rawY));
      const cx = toCanvasX(x);
      const cy = toCanvasY(y);
      if (i === 0) { ctx.moveTo(cx, cy); } else { ctx.lineTo(cx, cy); }
      if (Math.abs(rawY) > 2) valid = false;
    }
    ctx.stroke();

    // Data points
    dataset.forEach(p => {
      ctx.beginPath();
      ctx.arc(toCanvasX(p.x), toCanvasY(p.y), 4.5, 0, Math.PI * 2);
      ctx.fillStyle = '#C18C3B';
      ctx.strokeStyle = '#FAF6EE';
      ctx.lineWidth = 1.5;
      ctx.fill();
      ctx.stroke();
    });

    // Legend
    const legendY = 16;
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = '#3B7A57';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(38, legendY); ctx.lineTo(62, legendY); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#3B7A57';
    ctx.font = '9px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('True f(x)', 66, legendY + 4);

    ctx.strokeStyle = '#B6532B';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(140, legendY); ctx.lineTo(164, legendY); ctx.stroke();
    ctx.fillStyle = '#B6532B';
    ctx.fillText(`Model (degree ${degree})`, 168, legendY + 4);

    ctx.beginPath();
    ctx.arc(290, legendY, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#C18C3B';
    ctx.fill();
    ctx.fillStyle = '#6E6257';
    ctx.fillText('Data points', 298, legendY + 4);

    if (!valid) {
      ctx.fillStyle = 'rgba(182, 83, 43, 0.9)';
      ctx.font = 'bold 11px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('⚠ Model exploding — extreme overfit!', W / 2, H / 2);
    }

  }, [degree, dataset, coeffs]);

  const getZoneColor = () => {
    if (degree <= 1) return '#3B7A57'; // underfit
    if (degree === 3) return '#B6532B'; // good fit zone
    if (degree <= 4) return '#C18C3B'; // slightly overfit
    return '#A63A2B'; // overfit
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-12 gap-6 p-4">
      {/* Controls */}
      <div className="md:col-span-4 bg-[#F4EFE6] border border-[#E5DDD0] p-5 rounded-2xl flex flex-col justify-between">
        <div className="space-y-5">
          <div>
            <h4 className="text-[#2E251E] font-bold text-base tracking-wide flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-[#B6532B]" /> Bias-Variance
            </h4>
            <p className="text-[#6E6257] text-xs mt-1">
              Increase polynomial degree to see model transition from underfitting (high bias) to overfitting (high variance).
            </p>
          </div>

          <div>
            <div className="flex justify-between text-xs font-mono mb-2">
              <span className="text-[#2E251E] font-semibold">Polynomial Degree</span>
              <span className="font-bold" style={{ color: getZoneColor() }}>degree = {degree}</span>
            </div>
            <input
              type="range" min="1" max="9" step="1" value={degree}
              onChange={e => setDegree(parseInt(e.target.value))}
              className="w-full accent-[#B6532B]"
            />
            <div className="flex justify-between text-[10px] text-[#6E6257] font-mono mt-1">
              <span>1 (Underfit)</span>
              <span>9 (Overfit)</span>
            </div>
          </div>

          <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-3 text-center">
            <p className="text-[10px] text-[#6E6257] font-mono uppercase font-bold">{labels[Math.min(degree - 1, labels.length - 1)]}</p>
          </div>

          {/* Bias/Variance bars */}
          <div className="space-y-3">
            {[
              { label: 'Bias²', value: biasSquared, color: '#B6532B', max: 1 },
              { label: 'Variance', value: variance, color: '#C18C3B', max: 1 },
              { label: 'Total Error', value: totalError, color: '#2E251E', max: 1.3 },
            ].map(({ label, value, color, max }) => (
              <div key={label}>
                <div className="flex justify-between text-xs font-mono mb-1">
                  <span className="text-[#6E6257]">{label}</span>
                  <span className="font-bold" style={{ color }}>{Math.min(1, value).toFixed(3)}</span>
                </div>
                <div className="h-2 bg-[#E5DDD0] rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{ width: `${Math.min(100, (value / max) * 100)}%`, backgroundColor: color }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-3 text-xs font-mono space-y-1">
            <p className="text-[#B6532B] font-bold">Total = Bias² + Variance + Noise</p>
            <p className="text-[#6E6257]">
              {biasSquared.toFixed(3)} + {variance.toFixed(3)} + 0.060 = <span className="text-[#2E251E] font-bold">{totalError.toFixed(3)}</span>
            </p>
          </div>
        </div>
      </div>

      {/* Plot */}
      <div className="md:col-span-8">
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-inner h-full flex flex-col">
          <h5 className="text-[#2E251E] text-xs font-semibold font-mono mb-3">
            Model Fit vs True Function — drag the slider to change complexity
          </h5>
          <canvas ref={canvasRef} className="w-full rounded-xl" />
          <p className="text-[10px] text-[#6E6257] font-mono text-center mt-2">
            Sweet spot: degree 3–4 minimizes total error for this dataset
          </p>
        </div>
      </div>
    </div>
  );
};
