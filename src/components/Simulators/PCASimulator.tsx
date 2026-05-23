import React, { useState, useEffect, useRef } from 'react';
import { Compass, Trash2 } from 'lucide-react';

export const PCASimulator: React.FC = () => {
  const [points, setPoints] = useState<Array<{ x: number; y: number }>>([
    { x: 100, y: 120 }, { x: 150, y: 160 }, { x: 200, y: 200 },
    { x: 250, y: 260 }, { x: 300, y: 290 }, { x: 350, y: 340 },
    { x: 130, y: 180 }, { x: 220, y: 250 }, { x: 280, y: 320 }
  ]);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const clearPoints = () => setPoints([]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setPoints(prev => [...prev, { x, y }]);
  };

  const { meanX, meanY, pc1, pc2, variance } = React.useMemo(() => {
    const n = points.length;
    if (n < 2) return { meanX: 300, meanY: 200, pc1: { dx: 1, dy: 0 }, pc2: { dx: 0, dy: 1 }, variance: { v1: 0, v2: 0 } };

    let mx = 0, my = 0;
    points.forEach(p => { mx += p.x; my += p.y; });
    mx /= n;
    my /= n;

    let cxx = 0, cyy = 0, cxy = 0;
    points.forEach(p => {
      const dx = p.x - mx;
      const dy = p.y - my;
      cxx += dx * dx;
      cyy += dy * dy;
      cxy += dx * dy;
    });
    cxx /= n;
    cyy /= n;
    cxy /= n;

    // Eigenvalues of covariance matrix
    const trace = cxx + cyy;
    const det = cxx * cyy - cxy * cxy;
    const l1 = (trace + Math.sqrt(trace * trace - 4 * det)) / 2;
    const l2 = (trace - Math.sqrt(trace * trace - 4 * det)) / 2;

    // Eigenvectors
    let e1 = { dx: l1 - cyy, dy: cxy };
    const mag1 = Math.sqrt(e1.dx * e1.dx + e1.dy * e1.dy);
    if (mag1 === 0) e1 = { dx: 1, dy: 0 };
    else { e1.dx /= mag1; e1.dy /= mag1; }

    let e2 = { dx: l2 - cyy, dy: cxy };
    const mag2 = Math.sqrt(e2.dx * e2.dx + e2.dy * e2.dy);
    if (mag2 === 0) e2 = { dx: -e1.dy, dy: e1.dx };
    else { e2.dx /= mag2; e2.dy /= mag2; }

    const totalVar = l1 + l2;

    return { 
      meanX: mx, 
      meanY: my, 
      pc1: e1, 
      pc2: e2, 
      variance: { v1: totalVar === 0 ? 0 : l1 / totalVar, v2: totalVar === 0 ? 0 : l2 / totalVar }
    };
  }, [points]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width = 600;
    const h = canvas.height = 400;

    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, w, h);

    // Grid Overlay
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 50) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y < h; y += 50) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    if (points.length >= 2) {
      // Draw PC1 line
      ctx.beginPath();
      ctx.moveTo(meanX - pc1.dx * 1000, meanY - pc1.dy * 1000);
      ctx.lineTo(meanX + pc1.dx * 1000, meanY + pc1.dy * 1000);
      ctx.strokeStyle = 'rgba(244, 63, 94, 0.5)'; // Rose
      ctx.lineWidth = 3;
      ctx.stroke();

      // Draw PC2 line
      ctx.beginPath();
      ctx.moveTo(meanX - pc2.dx * 1000, meanY - pc2.dy * 1000);
      ctx.lineTo(meanX + pc2.dx * 1000, meanY + pc2.dy * 1000);
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.3)'; // Sky
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw Projections to PC1
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1.5;
      points.forEach(p => {
        const dx = p.x - meanX;
        const dy = p.y - meanY;
        const proj = dx * pc1.dx + dy * pc1.dy;
        const projX = meanX + proj * pc1.dx;
        const projY = meanY + proj * pc1.dy;

        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(projX, projY);
        ctx.stroke();

        // Draw projected point on the line
        ctx.beginPath();
        ctx.arc(projX, projY, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#f43f5e';
        ctx.fill();
      });
      ctx.setLineDash([]);
    }

    // Draw Mean Point
    if (points.length >= 1) {
      ctx.beginPath();
      ctx.arc(meanX, meanY, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
    }

    // Draw points
    points.forEach(pt => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#94a3b8';
      ctx.fill();
      
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

  }, [points, meanX, meanY, pc1, pc2]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-12 gap-8 p-6">
      <div className="md:col-span-4 bg-white border border-slate-200 p-6 rounded-3xl flex flex-col justify-between shadow-xl">
        <div className="space-y-6">
          <h4 className="text-slate-900 font-bold text-xl tracking-tight flex items-center gap-3">
            <Compass className="w-6 h-6 text-orange-500" /> Principal Component Analysis
          </h4>
          <p className="text-slate-500 text-sm leading-relaxed">
            PCA finds orthogonal axes of maximum variance. The <span className="font-semibold text-rose-500">First Principal Component (PC1)</span> is the line that minimizes projection errors and captures the most information.
          </p>

          <div className="p-5 bg-slate-50 border border-slate-200 rounded-2xl space-y-4 shadow-inner">
            <h5 className="text-xs text-slate-400 font-bold uppercase tracking-wider block border-b border-slate-200 pb-2">Variance Explained</h5>
            
            <div>
              <div className="flex justify-between items-center mb-1">
                <span className="text-rose-500 text-sm font-bold">PC1 (Red Line)</span>
                <span className="text-slate-700 font-mono font-bold">{(variance.v1 * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                <div className="h-full bg-rose-500 transition-all duration-300" style={{ width: `${variance.v1 * 100}%` }} />
              </div>
            </div>

            <div>
              <div className="flex justify-between items-center mb-1">
                <span className="text-sky-500 text-sm font-bold">PC2 (Blue Line)</span>
                <span className="text-slate-700 font-mono font-bold">{(variance.v2 * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                <div className="h-full bg-sky-500 transition-all duration-300" style={{ width: `${variance.v2 * 100}%` }} />
              </div>
            </div>
          </div>
        </div>

        <button onClick={clearPoints} className="w-full py-3 mt-6 rounded-xl bg-red-50 text-red-600 hover:bg-red-100 text-sm font-bold transition-colors flex justify-center items-center gap-2">
          <Trash2 className="w-5 h-5" /> Clear Data
        </button>
      </div>

      <div className="md:col-span-8 flex flex-col items-center justify-center">
        <div className="bg-slate-900 border border-slate-800 p-2 rounded-3xl w-full flex justify-center shadow-2xl relative overflow-hidden group">
          <canvas ref={canvasRef} onClick={handleCanvasClick} className="rounded-2xl cursor-crosshair w-full aspect-[3/2]" />
          <div className="absolute top-6 left-6 bg-white/10 backdrop-blur-md px-4 py-2 rounded-xl border border-white/20 text-xs font-mono text-white shadow-lg pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity">
            Dashed lines represent projection error onto PC1
          </div>
        </div>
      </div>
    </div>
  );
};
