import React, { useState, useEffect, useRef } from 'react';
import { Network, Trash2 } from 'lucide-react';

export const KNNClassifierSimulator: React.FC = () => {
  const [points, setPoints] = useState<Array<{ x: number; y: number; label: 0 | 1 | 2 }>>([
    { x: 150, y: 150, label: 0 }, { x: 180, y: 180, label: 0 }, { x: 120, y: 200, label: 0 },
    { x: 450, y: 150, label: 1 }, { x: 400, y: 120, label: 1 }, { x: 480, y: 180, label: 1 },
    { x: 300, y: 320, label: 2 }, { x: 350, y: 350, label: 2 }, { x: 250, y: 300, label: 2 }
  ]);
  const [activeLabel, setActiveLabel] = useState<0 | 1 | 2>(0);
  const [kValue, setKValue] = useState<number>(3);
  const [metric, setMetric] = useState<'euclidean' | 'manhattan'>('euclidean');
  
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

  const getDistance = (x1: number, y1: number, x2: number, y2: number) => {
    if (metric === 'euclidean') {
      return Math.sqrt((x1 - x2)**2 + (y1 - y2)**2);
    } else {
      return Math.abs(x1 - x2) + Math.abs(y1 - y2);
    }
  };

  const predictKNN = (px: number, py: number) => {
    if (points.length === 0) return -1;
    
    const distances = points.map(pt => ({
      dist: getDistance(px, py, pt.x, pt.y),
      label: pt.label
    }));
    
    distances.sort((a, b) => a.dist - b.dist);
    const neighbors = distances.slice(0, Math.min(kValue, distances.length));
    
    const counts = [0, 0, 0];
    neighbors.forEach(n => counts[n.label]++);
    
    let maxCount = -1;
    let bestLabel = -1;
    counts.forEach((c, i) => {
      if (c > maxCount) {
        maxCount = c;
        bestLabel = i;
      }
    });
    return bestLabel;
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

    if (points.length > 0) {
      // Draw background decision regions (Voronoi-like for K=1, smooth for higher K)
      const step = 8;
      for (let x = 0; x < w; x += step) {
        for (let y = 0; y < h; y += step) {
          const pred = predictKNN(x + step/2, y + step/2);
          if (pred === 0) {
            ctx.fillStyle = 'rgba(244, 63, 94, 0.2)'; // Rose
          } else if (pred === 1) {
            ctx.fillStyle = 'rgba(56, 189, 248, 0.2)'; // Sky
          } else if (pred === 2) {
            ctx.fillStyle = 'rgba(167, 139, 250, 0.2)'; // Purple
          } else {
            continue;
          }
          ctx.fillRect(x, y, step, step);
        }
      }
    }

    // Grid Overlay
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
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
      ctx.fillStyle = pt.label === 0 ? '#f43f5e' : pt.label === 1 ? '#38bdf8' : '#a78bfa';
      ctx.shadowBlur = 10;
      ctx.shadowColor = ctx.fillStyle;
      ctx.fill();
      ctx.shadowBlur = 0;
      
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 8, 0, Math.PI * 2);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();
    });

  }, [points, kValue, metric]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-12 gap-8 p-6">
      <div className="md:col-span-4 bg-[#FAF6EE] border border-[#E5DDD0] p-6 rounded-3xl flex flex-col justify-between shadow-xl">
        <div className="space-y-6">
          <h4 className="text-slate-900 font-bold text-xl tracking-tight flex items-center gap-3">
            <Network className="w-6 h-6 text-purple-500" /> K-Nearest Neighbors
          </h4>
          <p className="text-slate-500 text-sm leading-relaxed">
            KNN classifies a point based on the majority vote of its <span className="font-semibold text-purple-500">K</span> closest training examples. Notice how K=1 creates sharp Voronoi boundaries, while higher K values create smoother regions.
          </p>

          <div className="space-y-3">
            <label className="text-xs text-slate-400 font-bold uppercase tracking-wider block">Class Selector</label>
            <div className="flex bg-slate-100 p-1 rounded-xl gap-1">
              <button onClick={() => setActiveLabel(0)} className={`flex-1 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 ${activeLabel === 0 ? 'bg-rose-500 text-white shadow-md' : 'text-slate-500 hover:bg-slate-200'}`}>
                Class 0
              </button>
              <button onClick={() => setActiveLabel(1)} className={`flex-1 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 ${activeLabel === 1 ? 'bg-sky-500 text-white shadow-md' : 'text-slate-500 hover:bg-slate-200'}`}>
                Class 1
              </button>
              <button onClick={() => setActiveLabel(2)} className={`flex-1 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 ${activeLabel === 2 ? 'bg-purple-500 text-white shadow-md' : 'text-slate-500 hover:bg-slate-200'}`}>
                Class 2
              </button>
            </div>
          </div>

          <div className="space-y-3">
            <label className="text-xs text-slate-400 font-bold uppercase tracking-wider block">Distance Metric</label>
            <div className="flex bg-slate-100 p-1 rounded-xl gap-1">
              <button onClick={() => setMetric('euclidean')} className={`flex-1 py-2.5 rounded-lg text-sm font-bold transition-all ${metric === 'euclidean' ? 'bg-purple-500 text-white shadow-md' : 'text-slate-500 hover:bg-slate-200'}`}>Euclidean (L2)</button>
              <button onClick={() => setMetric('manhattan')} className={`flex-1 py-2.5 rounded-lg text-sm font-bold transition-all ${metric === 'manhattan' ? 'bg-purple-500 text-white shadow-md' : 'text-slate-500 hover:bg-slate-200'}`}>Manhattan (L1)</button>
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex justify-between items-center text-sm font-bold">
              <span className="text-slate-500">Neighbors (K):</span>
              <span className="text-purple-600 bg-purple-50 px-3 py-1 rounded-full">{kValue}</span>
            </div>
            <input type="range" min="1" max="15" step="2" value={kValue} onChange={(e) => setKValue(parseInt(e.target.value))} className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-purple-500" />
            <p className="text-[10px] text-slate-400 mt-1">We use odd values of K to avoid ties during voting.</p>
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
            Decision regions are painted based on K-closest points
          </div>
        </div>
      </div>
    </div>
  );
};
