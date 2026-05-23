import React, { useState, useEffect, useRef } from 'react';
import { Activity, Play, Pause, Trash2 } from 'lucide-react';

const drawLinearBoundary = (
  ctx: CanvasRenderingContext2D,
  w0: number,
  w1: number,
  b: number,
  width: number,
  height: number,
  color: string = '#2E251E',
  lineWidth: number = 2.5
) => {
  const A = w0 / width;
  const B = w1 / height;
  const C = b;

  const points: Array<{ x: number; y: number }> = [];

  if (Math.abs(B) > 1e-9) {
    const y = -C / B;
    if (y >= 0 && y <= height) points.push({ x: 0, y });
  }
  if (Math.abs(B) > 1e-9) {
    const y = -(A * width + C) / B;
    if (y >= 0 && y <= height) points.push({ x: width, y });
  }
  if (Math.abs(A) > 1e-9) {
    const x = -C / A;
    if (x >= 0 && x <= width) points.push({ x, y: 0 });
  }
  if (Math.abs(A) > 1e-9) {
    const x = -(B * height + C) / A;
    if (x >= 0 && x <= width) points.push({ x, y: height });
  }

  const uniquePoints: Array<{ x: number; y: number }> = [];
  points.forEach(p => {
    if (!uniquePoints.some(up => Math.abs(up.x - p.x) < 1e-3 && Math.abs(up.y - p.y) < 1e-3)) {
      uniquePoints.push(p);
    }
  });

  if (uniquePoints.length >= 2) {
    ctx.beginPath();
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = color;
    ctx.moveTo(uniquePoints[0].x, uniquePoints[0].y);
    ctx.lineTo(uniquePoints[1].x, uniquePoints[1].y);
    ctx.stroke();
  }
};

export const LogisticRegressionSimulator: React.FC = () => {
  const [points, setPoints] = useState<Array<{ x: number; y: number; label: 0 | 1 }>>([
    { x: 150, y: 300, label: 0 },
    { x: 200, y: 250, label: 0 },
    { x: 180, y: 150, label: 0 },
    { x: 400, y: 200, label: 1 },
    { x: 450, y: 120, label: 1 },
    { x: 480, y: 280, label: 1 },
  ]);
  const [activeLabel, setActiveLabel] = useState<0 | 1>(0);
  const [w, setW] = useState<[number, number]>([-1.2, 0.6]);
  const [b, setB] = useState<number>(0.2);
  const [epochs, setEpochs] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const clearPoints = () => {
    setPoints([]);
    setEpochs(0);
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setPoints(prev => [...prev, { x, y, label: activeLabel }]);
  };

  const trainStep = () => {
    if (points.length === 0) return;
    let dw0 = 0;
    let dw1 = 0;
    let db = 0;
    const lr = 0.1; // Increased learning rate for visibility

    points.forEach(pt => {
      const nx = pt.x / 600;
      const ny = pt.y / 400;
      const z = w[0] * nx + w[1] * ny + b;
      const prob = 1 / (1 + Math.exp(-z));
      const diff = prob - pt.label;

      dw0 += diff * nx;
      dw1 += diff * ny;
      db += diff;
    });

    const m = points.length;
    setW(prev => [prev[0] - lr * (dw0 / m), prev[1] - lr * (dw1 / m)]);
    setB(prev => prev - lr * (db / m));
    setEpochs(prev => prev + 1);
  };

  useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(trainStep, 40);
      return () => clearInterval(interval);
    }
  }, [isPlaying, points, w, b]);

  let loss = 0;
  if (points.length > 0) {
    let sumLoss = 0;
    points.forEach(pt => {
      const nx = pt.x / 600;
      const ny = pt.y / 400;
      const z = w[0] * nx + w[1] * ny + b;
      const prob = Math.max(1e-15, Math.min(1 - 1e-15, 1 / (1 + Math.exp(-z))));
      sumLoss += pt.label * Math.log(prob) + (1 - pt.label) * Math.log(1 - prob);
    });
    loss = -sumLoss / points.length;
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width = 600;
    const height = canvas.height = 400;

    // Draw background gradients (Sigmoid Field)
    const step = 6;
    for (let x = 0; x < width; x += step) {
      for (let y = 0; y < height; y += step) {
        const nx = (x + step/2) / width;
        const ny = (y + step/2) / height;
        const z = w[0] * nx + w[1] * ny + b;
        const prob = 1 / (1 + Math.exp(-z));

        // Blend: 0 = Rose (rgba(244, 63, 94)), 1 = Sky Blue (rgba(56, 189, 248))
        const r = Math.floor(244 * (1 - prob) + 56 * prob);
        const g = Math.floor(63 * (1 - prob) + 189 * prob);
        const bCol = Math.floor(94 * (1 - prob) + 248 * prob);
        
        ctx.fillStyle = `rgba(${r}, ${g}, ${bCol}, 0.25)`;
        ctx.fillRect(x, y, step, step);
      }
    }

    // Grid Overlay
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 50) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
    }
    for (let y = 0; y < height; y += 50) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
    }

    // Draw Decision Line (prob = 0.5)
    ctx.shadowBlur = 10;
    ctx.shadowColor = '#ffffff';
    drawLinearBoundary(ctx, w[0], w[1], b, width, height, '#ffffff', 3);
    ctx.shadowBlur = 0;

    // Draw coordinate points
    points.forEach(pt => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 10, 0, Math.PI * 2);
      ctx.fillStyle = pt.label === 0 ? '#f43f5e' : '#38bdf8';
      ctx.shadowBlur = 10;
      ctx.shadowColor = pt.label === 0 ? '#f43f5e' : '#38bdf8';
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 10, 0, Math.PI * 2);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2.5;
      ctx.stroke();
    });

  }, [points, w, b]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-12 gap-8 p-6">
      <div className="md:col-span-4 bg-[#FAF6EE] border border-[#E5DDD0] p-6 rounded-3xl flex flex-col justify-between shadow-xl">
        <div className="space-y-6">
          <h4 className="text-[#2E251E] font-bold text-xl tracking-tight flex items-center gap-3">
            <Activity className="w-6 h-6 text-[#B6532B]" /> Logistic Boundaries
          </h4>
          <p className="text-[#6E6257] text-sm leading-relaxed">
            Click on the canvas to place points of either class. Run the optimizer to see the <span className="font-semibold text-[#B6532B]">Sigmoid probability field</span> shift and establish the best decision boundary using Gradient Descent.
          </p>

          <div className="space-y-3">
            <label className="text-xs text-[#6E6257] font-bold uppercase tracking-wider block">Class Selector</label>
            <div className="flex bg-[#F4EFE6] p-1 rounded-xl gap-1">
              <button
                onClick={() => setActiveLabel(0)}
                className={`flex-1 py-3 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 ${
                  activeLabel === 0
                    ? 'bg-[#B6532B] text-white shadow-md'
                    : 'text-[#6E6257] hover:bg-[#E5DDD0]'
                }`}
              >
                <span className={`w-3 h-3 rounded-full ${activeLabel === 0 ? 'bg-white' : 'bg-rose-500'}`} /> Class 0
              </button>
              <button
                onClick={() => setActiveLabel(1)}
                className={`flex-1 py-3 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 ${
                  activeLabel === 1
                    ? 'bg-sky-500 text-white shadow-md'
                    : 'text-[#6E6257] hover:bg-[#E5DDD0]'
                }`}
              >
                <span className={`w-3 h-3 rounded-full ${activeLabel === 1 ? 'bg-white' : 'bg-sky-500'}`} /> Class 1
              </button>
            </div>
          </div>

          <div className="p-5 bg-[#2E251E] border border-[#4A3D31] rounded-2xl space-y-4 shadow-inner">
            <div className="flex justify-between items-center">
              <span className="text-slate-400 text-sm font-medium">Epochs</span>
              <span className="text-white font-mono font-bold">{epochs}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-400 text-sm font-medium">BCE Loss</span>
              <span className="text-emerald-400 font-mono font-bold text-lg">{loss.toFixed(4)}</span>
            </div>
            <div className="pt-4 mt-2 border-t border-[#4A3D31]">
              <span className="text-xs text-[#CFC5B4] font-mono">Weights:</span>
              <div className="text-sm text-[#E5DDD0] font-mono mt-1">
                w₁={w[0].toFixed(2)}, w₂={w[1].toFixed(2)}, b={b.toFixed(2)}
              </div>
            </div>
          </div>
        </div>

        <div className="pt-6 border-t border-[#E5DDD0] space-y-3 mt-4">
          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="py-3 rounded-xl bg-[#B6532B] text-white hover:bg-[#9F4825] font-bold text-sm transition-all shadow-md flex items-center justify-center gap-2"
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {isPlaying ? 'Pause' : 'Train Auto'}
            </button>
            <button
              onClick={trainStep}
              className="py-3 rounded-xl border-2 border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#F4EFE6] text-[#2E251E] font-bold text-sm shadow-sm"
            >
              Step Update
            </button>
          </div>
          <button
            onClick={clearPoints}
            className="w-full py-3 rounded-xl bg-red-50 text-red-600 hover:bg-red-100 text-sm font-bold transition-colors flex justify-center items-center gap-2"
          >
            <Trash2 className="w-4 h-4" /> Clear Canvas
          </button>
        </div>
      </div>

      <div className="md:col-span-8 flex flex-col items-center justify-center">
        <div className="bg-[#2E251E] border border-[#4A3D31] p-2 rounded-3xl w-full flex justify-center shadow-2xl relative overflow-hidden group">
          <canvas
            ref={canvasRef}
            onClick={handleCanvasClick}
            className="rounded-2xl cursor-crosshair w-full aspect-[3/2]"
          />
          <div className="absolute top-6 left-6 bg-white/10 backdrop-blur-md px-4 py-2 rounded-xl border border-white/20 text-xs font-mono text-white shadow-lg pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity">
            Click to plot {activeLabel === 0 ? 'Class 0 (Red)' : 'Class 1 (Blue)'}
          </div>
        </div>
      </div>
    </div>
  );
};
