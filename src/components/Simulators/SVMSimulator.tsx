import React, { useState, useEffect, useRef } from 'react';
import { Target, Trash2 } from 'lucide-react';

export const SVMSimulator: React.FC = () => {
  const [points, setPoints] = useState<Array<{ x: number; y: number; label: 0 | 1 }>>([
    { x: 150, y: 300, label: 0 }, { x: 200, y: 250, label: 0 }, { x: 180, y: 150, label: 0 },
    { x: 400, y: 200, label: 1 }, { x: 450, y: 120, label: 1 }, { x: 480, y: 280, label: 1 }
  ]);
  const [activeLabel, setActiveLabel] = useState<0 | 1>(0);
  const [kernel, setKernel] = useState<'linear' | 'rbf'>('linear');
  const [cParam, setCParam] = useState<number>(1.0); // Penalty parameter
  const [gamma, setGamma] = useState<number>(0.01); // RBF spread
  
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

  // Simplistic dummy SVM visualizer for educational purposes
  // In a real app, you'd run SMO or import a library like svm.js
  const computeDecisionValue = (px: number, py: number) => {
    if (points.length < 2) return 0;
    
    // Normalize coordinates
    const nx = px / 600;
    const ny = py / 400;

    let val = 0;
    
    // Very rough heuristic to simulate SVM boundaries
    if (kernel === 'linear') {
      // Approximate linear boundary based on centroids
      let cx0 = 0, cy0 = 0, n0 = 0;
      let cx1 = 0, cy1 = 0, n1 = 0;
      points.forEach(p => {
        if (p.label === 0) { cx0 += p.x/600; cy0 += p.y/400; n0++; }
        else { cx1 += p.x/600; cy1 += p.y/400; n1++; }
      });
      if (n0 > 0 && n1 > 0) {
        cx0 /= n0; cy0 /= n0; cx1 /= n1; cy1 /= n1;
        const dx = cx1 - cx0;
        const dy = cy1 - cy0;
        const midX = (cx0 + cx1) / 2;
        const midY = (cy0 + cy1) / 2;
        
        // Projection onto the vector between centroids
        val = (nx - midX) * dx + (ny - midY) * dy;
        val *= 10; // Scaling factor
      }
    } else {
      // RBF kernel approximation
      points.forEach(p => {
        const pnx = p.x / 600;
        const pny = p.y / 400;
        const distSq = (nx - pnx)**2 + (ny - pny)**2;
        const sign = p.label === 1 ? 1 : -1;
        // gamma controls the spread
        val += sign * Math.exp(-gamma * 1000 * distSq);
      });
    }
    
    return val;
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width = 600;
    const h = canvas.height = 400;

    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, w, h);

    if (points.length >= 2) {
      // Draw background heat map
      const step = 8;
      for (let x = 0; x < w; x += step) {
        for (let y = 0; y < h; y += step) {
          const val = computeDecisionValue(x + step/2, y + step/2);
          
          // val = 0 is boundary. >0 is class 1 (blue), <0 is class 0 (red)
          const prob = 1 / (1 + Math.exp(-val * 3)); // Squish to [0,1] for colors
          
          const r = Math.floor(244 * (1 - prob) + 56 * prob);
          const g = Math.floor(63 * (1 - prob) + 189 * prob);
          const bCol = Math.floor(94 * (1 - prob) + 248 * prob);
          
          // Highlight margin (where val is close to -1, 0, or 1)
          const marginIntensity = Math.max(0, 1 - Math.abs(val));
          const alpha = 0.2 + (marginIntensity * 0.3 * (1/cParam)); // Hard margin = less alpha spread
          
          ctx.fillStyle = `rgba(${r}, ${g}, ${bCol}, ${alpha})`;
          ctx.fillRect(x, y, step, step);
          
          // Draw Margin lines (approximate)
          if (Math.abs(val) < 0.05) {
            ctx.fillStyle = 'rgba(255,255,255,0.8)';
            ctx.fillRect(x, y, step, step);
          } else if (Math.abs(val - 1) < 0.05 || Math.abs(val + 1) < 0.05) {
            ctx.fillStyle = 'rgba(255,255,255,0.3)';
            ctx.fillRect(x, y, step, step);
          }
        }
      }
    }

    // Grid Overlay
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 50) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y < h; y += 50) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Draw points
    points.forEach(pt => {
      // Check if it's a "support vector" (inside the margin roughly)
      const val = computeDecisionValue(pt.x, pt.y);
      const isSupportVector = Math.abs(val) < 1.2;

      ctx.beginPath();
      ctx.arc(pt.x, pt.y, isSupportVector ? 12 : 8, 0, Math.PI * 2);
      ctx.fillStyle = pt.label === 0 ? '#f43f5e' : '#38bdf8';
      ctx.shadowBlur = 10;
      ctx.shadowColor = pt.label === 0 ? '#f43f5e' : '#38bdf8';
      ctx.fill();
      ctx.shadowBlur = 0;
      
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, isSupportVector ? 12 : 8, 0, Math.PI * 2);
      ctx.strokeStyle = isSupportVector ? '#fbbf24' : '#ffffff'; // Highlight SVs in yellow
      ctx.lineWidth = isSupportVector ? 3 : 2;
      ctx.stroke();
    });

  }, [points, kernel, cParam, gamma]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-12 gap-8 p-6">
      <div className="md:col-span-4 bg-[#FAF6EE] border border-[#E5DDD0] p-6 rounded-3xl flex flex-col justify-between shadow-xl">
        <div className="space-y-6">
          <h4 className="text-slate-900 font-bold text-xl tracking-tight flex items-center gap-3">
            <Target className="w-6 h-6 text-amber-500" /> Support Vector Machine
          </h4>
          <p className="text-slate-500 text-sm leading-relaxed">
            SVM finds the hyperplane that maximizes the margin between classes. Points with yellow rings are <span className="font-semibold text-amber-500">Support Vectors</span> that dictate the boundary.
          </p>

          <div className="space-y-3">
            <label className="text-xs text-slate-400 font-bold uppercase tracking-wider block">Class Selector</label>
            <div className="flex bg-slate-100 p-1 rounded-xl gap-1">
              <button onClick={() => setActiveLabel(0)} className={`flex-1 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 ${activeLabel === 0 ? 'bg-rose-500 text-white shadow-md' : 'text-slate-500 hover:bg-slate-200'}`}>
                <span className={`w-3 h-3 rounded-full ${activeLabel === 0 ? 'bg-white' : 'bg-rose-500'}`} /> Class 0
              </button>
              <button onClick={() => setActiveLabel(1)} className={`flex-1 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 ${activeLabel === 1 ? 'bg-sky-500 text-white shadow-md' : 'text-slate-500 hover:bg-slate-200'}`}>
                <span className={`w-3 h-3 rounded-full ${activeLabel === 1 ? 'bg-white' : 'bg-sky-500'}`} /> Class 1
              </button>
            </div>
          </div>

          <div className="space-y-3">
            <label className="text-xs text-slate-400 font-bold uppercase tracking-wider block">Kernel Type</label>
            <div className="flex bg-slate-100 p-1 rounded-xl gap-1">
              <button onClick={() => setKernel('linear')} className={`flex-1 py-2.5 rounded-lg text-sm font-bold transition-all ${kernel === 'linear' ? 'bg-amber-500 text-white shadow-md' : 'text-slate-500 hover:bg-slate-200'}`}>Linear</button>
              <button onClick={() => setKernel('rbf')} className={`flex-1 py-2.5 rounded-lg text-sm font-bold transition-all ${kernel === 'rbf' ? 'bg-amber-500 text-white shadow-md' : 'text-slate-500 hover:bg-slate-200'}`}>RBF (Non-linear)</button>
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex justify-between items-center text-sm font-bold">
              <span className="text-slate-500">Regularization (C):</span>
              <span className="text-amber-600 bg-amber-50 px-3 py-1 rounded-full">{cParam.toFixed(1)}</span>
            </div>
            <p className="text-[10px] text-slate-400">High C = Hard margin (strict), Low C = Soft margin (allows errors)</p>
            <input type="range" min="0.1" max="5.0" step="0.1" value={cParam} onChange={(e) => setCParam(parseFloat(e.target.value))} className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-amber-500" />
            
            {kernel === 'rbf' && (
              <>
                <div className="flex justify-between items-center text-sm font-bold pt-2">
                  <span className="text-slate-500">Gamma (Spread):</span>
                  <span className="text-amber-600 bg-amber-50 px-3 py-1 rounded-full">{gamma.toFixed(3)}</span>
                </div>
                <input type="range" min="0.001" max="0.05" step="0.001" value={gamma} onChange={(e) => setGamma(parseFloat(e.target.value))} className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-amber-500" />
              </>
            )}
          </div>
        </div>

        <button onClick={clearPoints} className="w-full py-3 mt-6 rounded-xl bg-red-50 text-red-600 hover:bg-red-100 text-sm font-bold transition-colors flex justify-center items-center gap-2">
          <Trash2 className="w-5 h-5" /> Clear Data
        </button>
      </div>

      <div className="md:col-span-8 flex flex-col items-center justify-center">
        <div className="bg-[#2E251E] border border-[#4A3D31] p-2 rounded-3xl w-full flex justify-center shadow-2xl relative overflow-hidden group">
          <canvas ref={canvasRef} onClick={handleCanvasClick} className="rounded-2xl cursor-crosshair w-full aspect-[3/2]" />
          <div className="absolute top-6 left-6 bg-white/10 backdrop-blur-md px-4 py-2 rounded-xl border border-white/20 text-xs font-mono text-white shadow-lg pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity">
            Solid white line = Decision Boundary. Faded white lines = Margins.
          </div>
        </div>
      </div>
    </div>
  );
};
