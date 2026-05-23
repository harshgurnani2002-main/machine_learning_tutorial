import React, { useState, useEffect, useRef } from 'react';
import { Trash2, LineChart } from 'lucide-react';

export const InteractiveRegression: React.FC = () => {
  const [points, setPoints] = useState<Array<{ x: number; y: number }>>([
    { x: 80, y: 220 },
    { x: 150, y: 180 },
    { x: 220, y: 130 },
    { x: 300, y: 90 },
  ]);
  const [fitType, setFitType] = useState<'linear' | 'quadratic'>('linear');
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragIdxRef = useRef<number | null>(null);

  const clearPoints = () => setPoints([]);

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const clickedIdx = points.findIndex(pt => {
      const dx = pt.x - x;
      const dy = pt.y - y;
      return dx * dx + dy * dy < 150;
    });

    if (clickedIdx !== -1) {
      dragIdxRef.current = clickedIdx;
    } else if (points.length < 20) {
      setPoints(prev => [...prev, { x, y }]);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (dragIdxRef.current === null) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(canvas.width, e.clientX - rect.left));
    const y = Math.max(0, Math.min(canvas.height, e.clientY - rect.top));

    setPoints(prev => {
      const next = [...prev];
      next[dragIdxRef.current!] = { x, y };
      return next;
    });
  };

  const handleMouseUp = () => {
    dragIdxRef.current = null;
  };

  const n = points.length;

  const { slope, intercept, quadraticCoeffs, mse } = React.useMemo(() => {
    let s = 0;
    let i = 0;
    let q = { a: 0, b: 0, c: 0 };
    let m = 0;

    if (n >= 2) {
      if (fitType === 'linear') {
        let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
        points.forEach(pt => {
          sumX += pt.x;
          sumY += pt.y;
          sumXY += pt.x * pt.y;
          sumXX += pt.x * pt.x;
        });

        const denom = (n * sumXX - sumX * sumX);
        if (Math.abs(denom) > 1e-5) {
          s = (n * sumXY - sumX * sumY) / denom;
          i = (sumY - s * sumX) / n;
        }

        let totalErr = 0;
        points.forEach(pt => {
          const predY = s * pt.x + i;
          totalErr += (pt.y - predY) * (pt.y - predY);
        });
        m = totalErr / n;
      } else {
        let sumX = 0, sumX2 = 0, sumX3 = 0, sumX4 = 0;
        let sumY = 0, sumXY = 0, sumX2Y = 0;

        points.forEach(pt => {
          const x = pt.x;
          const y = pt.y;
          const x2 = x * x;
          sumX += x;
          sumX2 += x2;
          sumX3 += x2 * x;
          sumX4 += x2 * x2;
          sumY += y;
          sumXY += x * y;
          sumX2Y += x2 * y;
        });

        const det = 
          sumX4 * (sumX2 * n - sumX * sumX) - 
          sumX3 * (sumX3 * n - sumX2 * sumX) + 
          sumX2 * (sumX3 * sumX - sumX2 * sumX2);

        if (Math.abs(det) > 1e-2) {
          const detA = 
            sumX2Y * (sumX2 * n - sumX * sumX) - 
            sumX3 * (sumXY * n - sumY * sumX) + 
            sumX2 * (sumXY * sumX - sumY * sumX2);
          
          const detB = 
            sumX4 * (sumXY * n - sumY * sumX) - 
            sumX2Y * (sumX3 * n - sumX2 * sumX) + 
            sumX2 * (sumX3 * sumY - sumX2 * sumXY);

          const detC = 
            sumX4 * (sumX2 * sumY - sumX * sumXY) - 
            sumX3 * (sumX3 * sumY - sumX2 * sumXY) + 
            sumX2Y * (sumX3 * sumX - sumX2 * sumX2);

          q.a = detA / det;
          q.b = detB / det;
          q.c = detC / det;
        }

        let totalErr = 0;
        points.forEach(pt => {
          const predY = q.a * pt.x * pt.x + q.b * pt.x + q.c;
          totalErr += (pt.y - predY) * (pt.y - predY);
        });
        m = totalErr / n;
      }
    }
    return { slope: s, intercept: i, quadraticCoeffs: q, mse: m };
  }, [points, fitType, n]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width = 600;
    const h = canvas.height = 400;

    // Premium dark gradient background
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, '#0f172a');
    grad.addColorStop(1, '#1e293b');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    // Subtle Grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 50) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += 50) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(50, 0); ctx.lineTo(50, h);
    ctx.moveTo(0, h - 50); ctx.lineTo(w, h - 50);
    ctx.stroke();

    if (points.length >= 2) {
      ctx.beginPath();
      ctx.lineWidth = 4;
      ctx.strokeStyle = '#38bdf8'; // Sky blue
      ctx.shadowColor = '#38bdf8';
      ctx.shadowBlur = 15;

      if (fitType === 'linear') {
        const yStart = slope * 0 + intercept;
        const yEnd = slope * w + intercept;
        ctx.moveTo(0, yStart);
        ctx.lineTo(w, yEnd);
      } else {
        for (let x = 0; x < w; x += 2) {
          const y = quadraticCoeffs.a * x * x + quadraticCoeffs.b * x + quadraticCoeffs.c;
          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      ctx.shadowBlur = 0; // reset
      
      // Draw residual lines
      ctx.strokeStyle = 'rgba(244, 63, 94, 0.5)'; // Rose 500
      ctx.setLineDash([5, 5]);
      ctx.lineWidth = 2;
      points.forEach(pt => {
        let predY = 0;
        if (fitType === 'linear') {
            predY = slope * pt.x + intercept;
        } else {
            predY = quadraticCoeffs.a * pt.x * pt.x + quadraticCoeffs.b * pt.x + quadraticCoeffs.c;
        }
        ctx.beginPath();
        ctx.moveTo(pt.x, pt.y);
        ctx.lineTo(pt.x, predY);
        ctx.stroke();
      });
      ctx.setLineDash([]);
    }

    // Draw Points
    points.forEach((pt) => {
      // Glow effect
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 12, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(248, 113, 113, 0.2)';
      ctx.fill();

      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#f87171'; // Red 400
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.fill();
      ctx.stroke();
    });

  }, [points, fitType, slope, intercept, quadraticCoeffs]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-12 gap-8 p-6">
      <div className="md:col-span-4 bg-[#FAF6EE] border border-[#E5DDD0] p-6 rounded-3xl flex flex-col justify-between shadow-xl">
        <div className="space-y-6">
          <h4 className="text-[#2E251E] font-bold text-xl tracking-tight flex items-center gap-3">
            <LineChart className="w-6 h-6 text-[#B6532B]" /> Linear Regression
          </h4>
          <p className="text-[#6E6257] text-sm leading-relaxed">
            Drag the red data points to see the <span className="font-semibold text-[#B6532B]">Ordinary Least Squares</span> (OLS) regression line recalibrate in real-time. Notice the dashed residual lines that the algorithm minimizes.
          </p>

          <div className="space-y-3">
            <label className="text-xs text-[#6E6257] font-bold uppercase tracking-wider block">Model Complexity</label>
            <div className="flex bg-[#F4EFE6] p-1 rounded-xl">
              <button
                onClick={() => setFitType('linear')}
                className={`flex-1 py-2.5 rounded-lg text-sm font-semibold transition-all ${
                  fitType === 'linear'
                    ? 'bg-[#FAF6EE] shadow-sm text-[#B6532B]'
                    : 'text-[#6E6257] hover:text-[#2E251E]'
                }`}
              >
                Linear (Degree 1)
              </button>
              <button
                onClick={() => setFitType('quadratic')}
                className={`flex-1 py-2.5 rounded-lg text-sm font-semibold transition-all ${
                  fitType === 'quadratic'
                    ? 'bg-[#FAF6EE] shadow-sm text-[#B6532B]'
                    : 'text-[#6E6257] hover:text-[#2E251E]'
                }`}
              >
                Polynomial (Degree 2)
              </button>
            </div>
          </div>

          <div className="p-5 bg-[#F4EFE6] border border-[#E5DDD0] rounded-2xl space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-[#6E6257] text-sm font-medium">Data Points</span>
              <span className="text-[#2E251E] font-bold bg-[#E5DDD0] px-3 py-1 rounded-full text-xs">{n}</span>
            </div>
            
            {n >= 2 && fitType === 'linear' && (
              <>
                <div className="flex justify-between items-center">
                  <span className="text-[#6E6257] text-sm font-medium">Weight (w)</span>
                  <span className="text-[#B6532B] font-mono font-bold text-lg">{(-slope).toFixed(3)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[#6E6257] text-sm font-medium">Bias (b)</span>
                  <span className="text-[#2E251E] font-mono font-bold text-lg">{(400 - intercept).toFixed(1)}</span>
                </div>
              </>
            )}
            
            <div className="pt-4 mt-2 border-t border-[#E5DDD0]">
              <div className="flex justify-between items-center">
                <span className="text-[#6E6257] text-sm font-bold">MSE (Error)</span>
                <span className="text-[#B6532B] font-mono font-bold text-xl">{(mse / 100).toFixed(2)}</span>
              </div>
            </div>
          </div>
        </div>

        <button
          onClick={clearPoints}
          className="w-full py-3 mt-6 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#F4EFE6] text-[#6E6257] hover:text-[#B6532B] transition-all flex items-center justify-center gap-2 text-sm font-bold shadow-sm"
        >
          <Trash2 className="w-5 h-5" /> Clear All Points
        </button>
      </div>

      <div className="md:col-span-8 flex flex-col items-center justify-center">
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-2 rounded-3xl w-full flex justify-center shadow-2xl relative overflow-hidden group">
          <canvas
            ref={canvasRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            className="rounded-2xl cursor-crosshair w-full aspect-[3/2] touch-none"
          />
          <div className="absolute top-6 left-6 bg-[#2E251E]/90 backdrop-blur-md px-4 py-2 rounded-xl border border-[#E5DDD0]/20 text-xs font-mono text-[#FAF6EE] shadow-lg pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity">
            Click to add • Drag to move
          </div>
        </div>
      </div>
    </div>
  );
};
