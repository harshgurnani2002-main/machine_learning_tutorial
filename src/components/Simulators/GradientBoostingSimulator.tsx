import React, { useState, useEffect, useRef } from 'react';
import { FastForward, Trash2 } from 'lucide-react';

export const GradientBoostingSimulator: React.FC = () => {
  const [points, setPoints] = useState<Array<{ x: number; y: number; label: 0 | 1 }>>([
    { x: 150, y: 150, label: 0 }, { x: 180, y: 180, label: 0 }, { x: 200, y: 100, label: 0 },
    { x: 400, y: 100, label: 1 }, { x: 450, y: 160, label: 1 }, { x: 500, y: 220, label: 1 },
    { x: 300, y: 280, label: 0 }, { x: 350, y: 350, label: 1 }, { x: 250, y: 200, label: 1 }
  ]);
  const [activeLabel, setActiveLabel] = useState<0 | 1>(0);
  const [numEstimators, setNumEstimators] = useState<number>(10);
  const [learningRate, setLearningRate] = useState<number>(0.1);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const clearPoints = () => setPoints([]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setPoints(prev => [...prev, { x, y, label: activeLabel }]);
  };

  interface WeakLearner {
    feature: 'x' | 'y';
    threshold: number;
    leftVal: number;
    rightVal: number;
  }

  // Very simple decision stump
  const buildStump = (pts: typeof points, residuals: number[]): WeakLearner => {
    let bestError = Infinity;
    let bestFeat: 'x' | 'y' = 'x';
    let bestThresh = 0;
    let bestLeft = 0;
    let bestRight = 0;

    ['x', 'y'].forEach(feat => {
      const vals = Array.from(new Set(pts.map(p => feat === 'x' ? p.x : p.y))).sort((a, b) => a - b);
      for (let i = 0; i < vals.length - 1; i++) {
        const thresh = (vals[i] + vals[i + 1]) / 2;
        
        let leftSum = 0, rightSum = 0;
        let leftCount = 0, rightCount = 0;
        
        for (let j = 0; j < pts.length; j++) {
          if ((feat === 'x' ? pts[j].x : pts[j].y) <= thresh) {
            leftSum += residuals[j]; leftCount++;
          } else {
            rightSum += residuals[j]; rightCount++;
          }
        }
        
        const leftVal = leftCount > 0 ? leftSum / leftCount : 0;
        const rightVal = rightCount > 0 ? rightSum / rightCount : 0;
        
        let error = 0;
        for (let j = 0; j < pts.length; j++) {
          const val = (feat === 'x' ? pts[j].x : pts[j].y) <= thresh ? leftVal : rightVal;
          error += (residuals[j] - val) ** 2;
        }

        if (error < bestError) {
          bestError = error;
          bestFeat = feat as 'x' | 'y';
          bestThresh = thresh;
          bestLeft = leftVal;
          bestRight = rightVal;
        }
      }
    });

    return { feature: bestFeat, threshold: bestThresh, leftVal: bestLeft, rightVal: bestRight };
  };

  const predictStump = (stump: WeakLearner, x: number, y: number) => {
    return (stump.feature === 'x' ? x : y) <= stump.threshold ? stump.leftVal : stump.rightVal;
  };

  const { f0, models } = React.useMemo(() => {
    if (points.length === 0) return { f0: 0, models: [] };

    let sum = 0;
    points.forEach(p => { sum += p.label; });
    const pAvg = sum / points.length;
    // Log odds
    const f0Initial = pAvg === 0 ? -10 : pAvg === 1 ? 10 : Math.log(pAvg / (1 - pAvg));

    const currentF = points.map(() => f0Initial);
    const ensemble: WeakLearner[] = [];

    for (let m = 0; m < numEstimators; m++) {
      const residuals = points.map((p, i) => {
        const prob = 1 / (1 + Math.exp(-currentF[i]));
        return p.label - prob;
      });

      const stump = buildStump(points, residuals);
      ensemble.push(stump);

      for (let i = 0; i < points.length; i++) {
        currentF[i] += learningRate * predictStump(stump, points[i].x, points[i].y);
      }
    }

    return { f0: f0Initial, models: ensemble };
  }, [points, numEstimators, learningRate]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width = 600;
    const h = canvas.height = 400;

    // Warm cream gradient background
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, '#FAF6EE');
    grad.addColorStop(1, '#F4EFE6');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    if (points.length > 0) {
      const step = 6;
      for (let x = 0; x < w; x += step) {
        for (let y = 0; y < h; y += step) {
          let f = f0;
          models.forEach(model => {
            f += learningRate * predictStump(model, x + step/2, y + step/2);
          });
          
          const prob = 1 / (1 + Math.exp(-f));
          
          const r = Math.floor(244 * (1 - prob) + 56 * prob);
          const g = Math.floor(63 * (1 - prob) + 189 * prob);
          const bCol = Math.floor(94 * (1 - prob) + 248 * prob);
          
          ctx.fillStyle = `rgba(${r}, ${g}, ${bCol}, 0.3)`;
          ctx.fillRect(x, y, step, step);
        }
      }
    }

    // Subtle warm grid
    ctx.strokeStyle = 'rgba(110, 98, 87, 0.1)';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 50) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y < h; y += 50) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Draw points
    points.forEach(pt => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 8, 0, Math.PI * 2);
      ctx.fillStyle = pt.label === 0 ? '#B6532B' : '#C18C3B';
      ctx.shadowBlur = 10;
      ctx.shadowColor = pt.label === 0 ? '#B6532B' : '#C18C3B';
      ctx.fill();
      ctx.shadowBlur = 0;
      
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 8, 0, Math.PI * 2);
      ctx.strokeStyle = '#FAF6EE';
      ctx.lineWidth = 2;
      ctx.stroke();
    });

  }, [points, f0, models, learningRate]);

  let correct = 0;
  if (models.length > 0) {
    points.forEach(p => {
      let f = f0;
      models.forEach(model => {
        f += learningRate * predictStump(model, p.x, p.y);
      });
      const prob = 1 / (1 + Math.exp(-f));
      const pred = prob >= 0.5 ? 1 : 0;
      if (pred === p.label) correct++;
    });
  }
  const accuracy = points.length > 0 ? (correct / points.length) * 100 : 0;

  return (
    <div className="grid grid-cols-1 md:grid-cols-12 gap-8 p-6">
      <div className="md:col-span-4 bg-[#FAF6EE] border border-[#E5DDD0] p-6 rounded-3xl flex flex-col justify-between shadow-xl">
        <div className="space-y-6">
          <h4 className="text-[#2E251E] font-bold text-xl tracking-tight flex items-center gap-3">
            <FastForward className="w-6 h-6 text-[#B6532B]" /> Gradient Boosting
          </h4>
          <p className="text-[#6E6257] text-sm leading-relaxed">
            Boosting combines weak learners (decision stumps) sequentially. Each new tree corrects the <span className="font-semibold text-[#B6532B]">residual errors</span> of the previous ensemble.
          </p>

          <div className="space-y-3">
            <label className="text-xs text-[#6E6257] font-bold uppercase tracking-wider block font-mono">Class Selector</label>
            <div className="flex bg-[#FAF6EE] border border-[#E5DDD0] p-1 rounded-xl gap-1">
              <button onClick={() => setActiveLabel(0)} className={`flex-1 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 ${activeLabel === 0 ? 'bg-[#B6532B] text-white shadow-md' : 'text-[#6E6257] hover:bg-[#F4EFE6]'}`}>
                <span className={`w-3 h-3 rounded-full ${activeLabel === 0 ? 'bg-white' : 'bg-[#B6532B]'}`} /> Class 0
              </button>
              <button onClick={() => setActiveLabel(1)} className={`flex-1 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 ${activeLabel === 1 ? 'bg-[#C18C3B] text-white shadow-md' : 'text-[#6E6257] hover:bg-[#F4EFE6]'}`}>
                <span className={`w-3 h-3 rounded-full ${activeLabel === 1 ? 'bg-white' : 'bg-[#C18C3B]'}`} /> Class 1
              </button>
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex justify-between items-center text-sm font-bold">
              <span className="text-[#6E6257]">Estimators (Stages):</span>
              <span className="text-[#B6532B] bg-[#B6532B]/10 px-3 py-1 rounded-full">{numEstimators}</span>
            </div>
            <input type="range" min="1" max="50" step="1" value={numEstimators} onChange={(e) => setNumEstimators(parseInt(e.target.value))} className="w-full h-2 bg-[#E5DDD0] rounded-lg appearance-none cursor-pointer accent-[#B6532B]" />
            
            <div className="flex justify-between items-center text-sm font-bold pt-2">
              <span className="text-[#6E6257]">Learning Rate:</span>
              <span className="text-[#B6532B] bg-[#B6532B]/10 px-3 py-1 rounded-full">{learningRate.toFixed(2)}</span>
            </div>
            <input type="range" min="0.01" max="1.0" step="0.01" value={learningRate} onChange={(e) => setLearningRate(parseFloat(e.target.value))} className="w-full h-2 bg-[#E5DDD0] rounded-lg appearance-none cursor-pointer accent-[#B6532B]" />
          </div>

          <div className="p-5 bg-[#FAF6EE] border border-[#E5DDD0] rounded-2xl space-y-3 shadow-inner mt-4">
            <div className="flex justify-between items-center">
              <span className="text-[#6E6257] text-sm font-medium">Model Accuracy</span>
              <span className="text-[#B6532B] font-mono font-bold text-lg">{accuracy.toFixed(1)}%</span>
            </div>
          </div>
        </div>

        <button onClick={clearPoints} className="w-full py-3 mt-6 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#F4EFE6] text-[#6E6257] hover:text-[#B6532B] text-sm font-bold transition-colors flex justify-center items-center gap-2 shadow-sm">
          <Trash2 className="w-5 h-5" /> Clear Data
        </button>
      </div>

      <div className="md:col-span-8 flex flex-col items-center justify-center">
        <div className="bg-[#F4EFE6] border border-[#E5DDD0] p-2 rounded-3xl w-full flex justify-center shadow-md relative overflow-hidden group">
          <canvas ref={canvasRef} onClick={handleCanvasClick} className="rounded-2xl cursor-crosshair w-full aspect-[3/2]" />
          <div className="absolute top-6 left-6 bg-[#2E251E]/80 backdrop-blur-md px-4 py-2 rounded-xl border border-[#E5DDD0]/30 text-xs font-mono text-[#FAF6EE] shadow-lg pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity">
            Watch boundaries evolve iteratively
          </div>
        </div>
      </div>
    </div>
  );
};
