import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Crosshair, Sliders, Info } from 'lucide-react';

interface Point {
  x: number;
  y: number;
  loss: number;
}

interface OptState {
  name: string;
  key: 'sgd' | 'momentum' | 'rmsprop' | 'adam';
  color: string;
  posX: number;
  posY: number;
  history: Point[];
  vx: number;
  vy: number;
  sx: number;
  sy: number;
  mx: number;
  my: number;
}

export const GradientDescent: React.FC = () => {
  const [surfaceType, setSurfaceType] = useState<'bowl' | 'valley' | 'saddle'>('valley');
  const [learningRate, setLearningRate] = useState<number>(0.15);
  const [momentumBeta, setMomentumBeta] = useState<number>(0.9);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [epoch, setEpoch] = useState<number>(0);
  const [startPoint, setStartPoint] = useState<{ x: number; y: number }>({ x: 1.6, y: 1.4 });

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lossCanvasRef = useRef<HTMLCanvasElement>(null);
  const backgroundRef = useRef<ImageData | null>(null);
  const backgroundSurfaceRef = useRef<string>('');

  const [optimizersState, setOptimizersState] = useState<Record<string, OptState>>({
    sgd: { name: 'SGD', key: 'sgd', color: '#EF4444', posX: 1.6, posY: 1.4, history: [], vx: 0, vy: 0, sx: 0, sy: 0, mx: 0, my: 0 },
    momentum: { name: 'Momentum', key: 'momentum', color: '#3B82F6', posX: 1.6, posY: 1.4, history: [], vx: 0, vy: 0, sx: 0, sy: 0, mx: 0, my: 0 },
    rmsprop: { name: 'RMSprop', key: 'rmsprop', color: '#10B981', posX: 1.6, posY: 1.4, history: [], vx: 0, vy: 0, sx: 0, sy: 0, mx: 0, my: 0 },
    adam: { name: 'Adam', key: 'adam', color: '#8B5CF6', posX: 1.6, posY: 1.4, history: [], vx: 0, vy: 0, sx: 0, sy: 0, mx: 0, my: 0 },
  });

  // 1. Loss Functions and Gradients
  const getLoss = (x: number, y: number, type: 'bowl' | 'valley' | 'saddle') => {
    switch (type) {
      case 'bowl':
        return 0.5 * (x * x + y * y);
      case 'valley':
        return 0.05 * x * x + 1.0 * y * y;
      case 'saddle':
        return 0.2 * (x * x - y * y) + 1.0;
    }
  };

  const getGradient = (x: number, y: number, type: 'bowl' | 'valley' | 'saddle') => {
    switch (type) {
      case 'bowl':
        return { dx: x, dy: y };
      case 'valley':
        return { dx: 0.1 * x, dy: 2.0 * y };
      case 'saddle':
        return { dx: 0.4 * x, dy: -0.4 * y };
    }
  };

  const getNormalizedRatio = (loss: number, type: 'bowl' | 'valley' | 'saddle') => {
    switch (type) {
      case 'bowl':
        return Math.min(1, Math.max(0, loss / 4.84));
      case 'valley':
        return Math.min(1, Math.max(0, loss / 5.0));
      case 'saddle':
        return Math.min(1, Math.max(0, (loss - 0.03) / 1.94));
    }
  };

  // Reset/Restart when start point or surface type changes
  useEffect(() => {
    const startX = startPoint.x;
    const startY = startPoint.y;

    setOptimizersState({
      sgd: { name: 'SGD', key: 'sgd', color: '#EF4444', posX: startX, posY: startY, history: [{ x: startX, y: startY, loss: getLoss(startX, startY, surfaceType) }], vx: 0, vy: 0, sx: 0, sy: 0, mx: 0, my: 0 },
      momentum: { name: 'Momentum', key: 'momentum', color: '#3B82F6', posX: startX, posY: startY, history: [{ x: startX, y: startY, loss: getLoss(startX, startY, surfaceType) }], vx: 0, vy: 0, sx: 0, sy: 0, mx: 0, my: 0 },
      rmsprop: { name: 'RMSprop', key: 'rmsprop', color: '#10B981', posX: startX, posY: startY, history: [{ x: startX, y: startY, loss: getLoss(startX, startY, surfaceType) }], vx: 0, vy: 0, sx: 0, sy: 0, mx: 0, my: 0 },
      adam: { name: 'Adam', key: 'adam', color: '#8B5CF6', posX: startX, posY: startY, history: [{ x: startX, y: startY, loss: getLoss(startX, startY, surfaceType) }], vx: 0, vy: 0, sx: 0, sy: 0, mx: 0, my: 0 },
    });
    setEpoch(0);
    setIsPlaying(false);
  }, [startPoint, surfaceType]);

  // Simulation steps
  const step = () => {
    setOptimizersState(prev => {
      const nextState = { ...prev };
      let anyActive = false;

      (Object.keys(nextState) as Array<keyof typeof prev>).forEach(key => {
        const opt = { ...nextState[key] };

        // Boundary check
        if (Math.abs(opt.posX) > 2.2 || Math.abs(opt.posY) > 2.2) {
          return;
        }

        anyActive = true;
        const { dx, dy } = getGradient(opt.posX, opt.posY, surfaceType);
        let nextX = opt.posX;
        let nextY = opt.posY;

        let nextVx = opt.vx;
        let nextVy = opt.vy;
        let nextMx = opt.mx;
        let nextMy = opt.my;
        let nextSx = opt.sx;
        let nextSy = opt.sy;

        if (key === 'sgd') {
          nextX = opt.posX - learningRate * dx;
          nextY = opt.posY - learningRate * dy;
        } else if (key === 'momentum') {
          nextVx = momentumBeta * opt.vx + learningRate * dx;
          nextVy = momentumBeta * opt.vy + learningRate * dy;
          nextX = opt.posX - nextVx;
          nextY = opt.posY - nextVy;
        } else if (key === 'rmsprop') {
          const betaRms = 0.9;
          const eps = 1e-8;
          nextSx = betaRms * opt.sx + (1 - betaRms) * dx * dx;
          nextSy = betaRms * opt.sy + (1 - betaRms) * dy * dy;
          nextX = opt.posX - (learningRate * dx) / (Math.sqrt(nextSx) + eps);
          nextY = opt.posY - (learningRate * dy) / (Math.sqrt(nextSy) + eps);
        } else if (key === 'adam') {
          const beta1 = 0.9;
          const beta2 = 0.999;
          const eps = 1e-8;
          const t = epoch + 1;

          nextMx = beta1 * opt.mx + (1 - beta1) * dx;
          nextMy = beta1 * opt.my + (1 - beta1) * dy;

          nextSx = beta2 * opt.sx + (1 - beta2) * dx * dx;
          nextSy = beta2 * opt.sy + (1 - beta2) * dy * dy;

          const mHatX = nextMx / (1 - Math.pow(beta1, t));
          const mHatY = nextMy / (1 - Math.pow(beta1, t));
          const sHatX = nextSx / (1 - Math.pow(beta2, t));
          const sHatY = nextSy / (1 - Math.pow(beta2, t));

          nextX = opt.posX - (learningRate * mHatX) / (Math.sqrt(sHatX) + eps);
          nextY = opt.posY - (learningRate * mHatY) / (Math.sqrt(sHatY) + eps);
        }

        opt.posX = nextX;
        opt.posY = nextY;
        opt.vx = nextVx;
        opt.vy = nextVy;
        opt.mx = nextMx;
        opt.my = nextMy;
        opt.sx = nextSx;
        opt.sy = nextSy;

        const nextLoss = getLoss(nextX, nextY, surfaceType);
        opt.history = [...opt.history, { x: nextX, y: nextY, loss: nextLoss }];

        nextState[key] = opt;
      });

      if (!anyActive) {
        setIsPlaying(false);
      } else {
        setEpoch(prev => prev + 1);
      }

      return nextState;
    });
  };

  const stepRef = useRef(step);
  useEffect(() => {
    stepRef.current = step;
  }, [step]);

  // Auto-start descent on mount
  useEffect(() => {
    const timer = setTimeout(() => setIsPlaying(true), 800);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(() => {
        stepRef.current();
      }, 100);
      return () => clearInterval(interval);
    }
  }, [isPlaying]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    const cx = ((px / canvas.width) - 0.5) * 4.4;
    const cy = ((py / canvas.height) - 0.5) * 4.4;

    setStartPoint({ x: cx, y: cy });
  };

  const resetSimulation = () => {
    setIsPlaying(false);
    setEpoch(0);
    const startX = startPoint.x;
    const startY = startPoint.y;

    setOptimizersState({
      sgd: { name: 'SGD', key: 'sgd', color: '#EF4444', posX: startX, posY: startY, history: [{ x: startX, y: startY, loss: getLoss(startX, startY, surfaceType) }], vx: 0, vy: 0, sx: 0, sy: 0, mx: 0, my: 0 },
      momentum: { name: 'Momentum', key: 'momentum', color: '#3B82F6', posX: startX, posY: startY, history: [{ x: startX, y: startY, loss: getLoss(startX, startY, surfaceType) }], vx: 0, vy: 0, sx: 0, sy: 0, mx: 0, my: 0 },
      rmsprop: { name: 'RMSprop', key: 'rmsprop', color: '#10B981', posX: startX, posY: startY, history: [{ x: startX, y: startY, loss: getLoss(startX, startY, surfaceType) }], vx: 0, vy: 0, sx: 0, sy: 0, mx: 0, my: 0 },
      adam: { name: 'Adam', key: 'adam', color: '#8B5CF6', posX: startX, posY: startY, history: [{ x: startX, y: startY, loss: getLoss(startX, startY, surfaceType) }], vx: 0, vy: 0, sx: 0, sy: 0, mx: 0, my: 0 },
    });
  };

  // Draw 2D Landscape & Trajectories
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width = 400;
    const height = canvas.height = 400;

    // Cache background image if not already cached
    if (!backgroundRef.current || backgroundSurfaceRef.current !== surfaceType) {
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = 400;
      tempCanvas.height = 400;
      const tempCtx = tempCanvas.getContext('2d')!;
      const imgData = tempCtx.createImageData(400, 400);

      for (let py = 0; py < 400; py++) {
        for (let px = 0; px < 400; px++) {
          const cx = ((px / 400) - 0.5) * 4.4;
          const cy = ((py / 400) - 0.5) * 4.4;
          const loss = getLoss(cx, cy, surfaceType);
          const ratio = getNormalizedRatio(loss, surfaceType);
          const idx = (py * 400 + px) * 4;

          // Cool Sky Blue (low loss) = #E0F2FE (224, 242, 254)
          // Warm Terracotta (high loss) = #B6532B (182, 83, 43)
          const r = Math.floor(224 * (1 - ratio) + 182 * ratio);
          const g = Math.floor(242 * (1 - ratio) + 83 * ratio);
          const b = Math.floor(254 * (1 - ratio) + 43 * ratio);

          imgData.data[idx] = r;
          imgData.data[idx + 1] = g;
          imgData.data[idx + 2] = b;
          imgData.data[idx + 3] = 255;
        }
      }
      backgroundRef.current = imgData;
      backgroundSurfaceRef.current = surfaceType;
    }

    // Draw background
    ctx.putImageData(backgroundRef.current, 0, 0);

    // Draw contours
    ctx.lineWidth = 1;
    if (surfaceType === 'bowl') {
      for (let level = 1; level <= 6; level++) {
        ctx.strokeStyle = `rgba(46, 37, 30, ${0.03 * level})`;
        ctx.beginPath();
        const r = Math.sqrt(level * 0.7) * (width / 4.4);
        ctx.arc(width / 2, height / 2, r, 0, Math.PI * 2);
        ctx.stroke();
      }
    } else if (surfaceType === 'valley') {
      for (let level = 1; level <= 7; level++) {
        ctx.strokeStyle = `rgba(46, 37, 30, ${0.03 * level})`;
        ctx.beginPath();
        const rx = Math.sqrt(level * 0.9) * (width / 4.4) * 2;
        const ry = Math.sqrt(level * 0.9) * (height / 4.4) * 0.45;
        ctx.ellipse(width / 2, height / 2, rx, ry, 0, 0, Math.PI * 2);
        ctx.stroke();
      }
    } else if (surfaceType === 'saddle') {
      ctx.strokeStyle = 'rgba(46, 37, 30, 0.08)';
      for (let level = -5; level <= 5; level++) {
        if (level === 0) continue;
        const constVal = level * 0.4;

        // Horizontal hyperbola
        ctx.beginPath();
        for (let py = 0; py < height; py += 4) {
          const cy = ((py / height) - 0.5) * 4.4;
          const cx2 = cy * cy + constVal;
          if (cx2 >= 0) {
            const cx = Math.sqrt(cx2);
            const px = ((cx / 4.4) + 0.5) * width;
            if (py === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          }
        }
        ctx.stroke();

        ctx.beginPath();
        for (let py = 0; py < height; py += 4) {
          const cy = ((py / height) - 0.5) * 4.4;
          const cx2 = cy * cy + constVal;
          if (cx2 >= 0) {
            const cx = Math.sqrt(cx2);
            const px = (((-cx) / 4.4) + 0.5) * width;
            if (py === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          }
        }
        ctx.stroke();

        // Vertical hyperbola
        ctx.beginPath();
        for (let px = 0; px < width; px += 4) {
          const cx = ((px / width) - 0.5) * 4.4;
          const cy2 = cx * cx + constVal;
          if (cy2 >= 0) {
            const cy = Math.sqrt(cy2);
            const py = ((cy / 4.4) + 0.5) * height;
            if (px === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          }
        }
        ctx.stroke();

        ctx.beginPath();
        for (let px = 0; px < width; px += 4) {
          const cx = ((px / width) - 0.5) * 4.4;
          const cy2 = cx * cx + constVal;
          if (cy2 >= 0) {
            const cy = Math.sqrt(cy2);
            const py = (((-cy) / 4.4) + 0.5) * height;
            if (px === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          }
        }
        ctx.stroke();
      }
    }

    // Grid lines
    ctx.strokeStyle = 'rgba(46, 37, 30, 0.08)';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(width / 2, 0);
    ctx.lineTo(width / 2, height);
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Trajectories
    Object.values(optimizersState).forEach(opt => {
      if (opt.history.length > 0) {
        ctx.strokeStyle = opt.color;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        opt.history.forEach((pos, idx) => {
          const px = ((pos.x / 4.4) + 0.5) * width;
          const py = ((pos.y / 4.4) + 0.5) * height;
          if (idx === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();

        // Historical markers
        opt.history.forEach((pos, idx) => {
          if (idx % 3 === 0 || idx === opt.history.length - 1) {
            const px = ((pos.x / 4.4) + 0.5) * width;
            const py = ((pos.y / 4.4) + 0.5) * height;
            ctx.beginPath();
            ctx.arc(px, py, 2.5, 0, Math.PI * 2);
            ctx.fillStyle = opt.color;
            ctx.fill();
          }
        });

        // Current position indicator
        const last = opt.history[opt.history.length - 1];
        const px = ((last.x / 4.4) + 0.5) * width;
        const py = ((last.y / 4.4) + 0.5) * height;
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.fillStyle = opt.color;
        ctx.fill();
        ctx.strokeStyle = '#FAF6EE';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    });
  }, [optimizersState, surfaceType]);

  // Draw Loss Curves (Convergence Panel)
  useEffect(() => {
    const canvas = lossCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width = 400;
    const h = canvas.height = 140;

    ctx.fillStyle = '#FAF6EE';
    ctx.fillRect(0, 0, w, h);

    // Chart grid lines
    ctx.strokeStyle = '#E5DDD0';
    ctx.lineWidth = 0.5;
    for (let i = 20; i < w; i += 40) {
      ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, h); ctx.stroke();
    }
    for (let i = 20; i < h; i += 30) {
      ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(w, i); ctx.stroke();
    }

    let allLosses: number[] = [];
    Object.values(optimizersState).forEach(opt => {
      opt.history.forEach(pt => allLosses.push(pt.loss));
    });
    if (allLosses.length === 0) return;

    const maxLoss = Math.max(0.5, ...allLosses);
    const maxHistoryLen = Math.max(20, ...Object.values(optimizersState).map(opt => opt.history.length));

    // Curves
    Object.values(optimizersState).forEach(opt => {
      if (opt.history.length === 0) return;
      ctx.beginPath();
      ctx.strokeStyle = opt.color;
      ctx.lineWidth = 2;

      opt.history.forEach((pt, idx) => {
        const x = (idx / maxHistoryLen) * (w - 30) + 15;
        const boundedLoss = Math.max(0, pt.loss);
        const y = h - (boundedLoss / maxLoss) * (h - 25) - 10;
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });

    // Legend
    ctx.font = '9px monospace';
    let legendX = 15;
    Object.values(optimizersState).forEach(opt => {
      ctx.fillStyle = opt.color;
      ctx.fillRect(legendX, 8, 8, 8);
      ctx.fillStyle = '#2E251E';
      ctx.fillText(opt.name, legendX + 12, 15);
      legendX += 75;
    });

    // Label epochs
    ctx.fillStyle = '#6E6257';
    ctx.fillText(`Epochs: ${epoch}`, 15, h - 5);
  }, [optimizersState, epoch]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-12 gap-6 p-4">
      {/* Parameter sidebar */}
      <div className="md:col-span-5 bg-[#F4EFE6] border border-[#E5DDD0] p-5 rounded-2xl flex flex-col justify-between shadow-sm">
        <div className="space-y-6">
          <h4 className="text-[#2E251E] font-bold text-base mb-1 tracking-wide flex items-center gap-2">
            <Sliders className="w-5 h-5 text-[#B6532B]" /> Simulation Controls
          </h4>

          {/* Landscape Selector */}
          <div>
            <label className="text-xs text-[#2E251E] font-mono block mb-2 font-semibold uppercase tracking-wider">Loss Landscape</label>
            <div className="grid grid-cols-3 gap-2">
              {(['bowl', 'valley', 'saddle'] as const).map((type) => (
                <button
                  key={type}
                  onClick={() => setSurfaceType(type)}
                  className={`py-2 rounded-xl text-xs font-mono transition-all capitalize border ${
                    surfaceType === type
                      ? 'border-[#B6532B] bg-[#B6532B]/10 text-[#B6532B] font-semibold shadow-sm'
                      : 'border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257] hover:text-[#2E251E] hover:bg-[#F4EFE6]'
                  }`}
                >
                  {type === 'bowl' ? 'Convex Bowl' : type === 'valley' ? 'Elliptical Ravine' : 'Saddle Point'}
                </button>
              ))}
            </div>
          </div>

          {/* Sliders */}
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-xs font-mono mb-2">
                <span className="text-[#2E251E]">Learning Rate (η)</span>
                <span className="text-[#B6532B] font-semibold">{learningRate.toFixed(3)}</span>
              </div>
              <input
                type="range"
                min="0.01"
                max="1.5"
                step="0.01"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                className="w-full h-1 bg-[#E5DDD0] rounded-lg appearance-none cursor-pointer accent-[#B6532B]"
              />
              <div className="flex justify-between text-[9px] text-[#6E6257] font-mono mt-1">
                <span>0.01 (Cautious)</span>
                <span>1.50 (Exploding)</span>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs font-mono mb-2">
                <span className="text-[#2E251E]">Momentum Decay (β)</span>
                <span className="text-[#B6532B] font-semibold">{momentumBeta.toFixed(3)}</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="0.999"
                step="0.005"
                value={momentumBeta}
                onChange={(e) => setMomentumBeta(parseFloat(e.target.value))}
                className="w-full h-1 bg-[#E5DDD0] rounded-lg appearance-none cursor-pointer accent-[#B6532B]"
              />
              <div className="flex justify-between text-[9px] text-[#6E6257] font-mono mt-1">
                <span>0.50 (Low Inertia)</span>
                <span>0.999 (High Spiral Overshoot)</span>
              </div>
            </div>
          </div>

          {/* Comparison Table */}
          <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-3.5 space-y-2">
            <span className="text-[10px] text-[#6E6257] font-mono block uppercase tracking-wider font-bold">Optimizer Status Table</span>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono text-[#2E251E]">
                <thead>
                  <tr className="border-b border-[#E5DDD0] text-[#6E6257] text-[10px]">
                    <th className="pb-1.5 font-semibold">Optimizer</th>
                    <th className="pb-1.5 font-semibold text-right">Coords (w1, w2)</th>
                    <th className="pb-1.5 font-semibold text-right">Loss J(w)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#E5DDD0]/50">
                  {Object.values(optimizersState).map(opt => {
                    const lastLoss = opt.history.length > 0 ? opt.history[opt.history.length - 1].loss : 0;
                    return (
                      <tr key={opt.key}>
                        <td className="py-2 flex items-center gap-1.5 font-bold">
                          <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: opt.color }} />
                          {opt.name}
                        </td>
                        <td className="py-2 text-right">({opt.posX.toFixed(2)}, {opt.posY.toFixed(2)})</td>
                        <td className="py-2 text-right font-semibold text-[#B6532B]">{lastLoss.toFixed(4)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Action Controls */}
        <div className="pt-5 border-t border-[#E5DDD0] space-y-3">
          <div className="flex gap-2">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className={`flex-1 py-3 rounded-xl flex items-center justify-center gap-2 text-xs font-semibold shadow-sm transition-all cursor-pointer ${
                isPlaying
                  ? 'bg-amber-600 border-amber-600 text-white'
                  : 'bg-[#B6532B] border-[#B6532B] hover:bg-[#B6532B]/95 text-white'
              }`}
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {isPlaying ? 'Pause optimization' : 'Start descent'}
            </button>

            <button
              onClick={resetSimulation}
              className="p-3 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257] hover:text-[#2E251E] hover:bg-[#F4EFE6] transition-colors cursor-pointer"
              title="Reset weights"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

          <div className="text-[10px] text-[#6E6257] flex items-center gap-1.5 font-mono justify-center">
            <Crosshair className="w-3.5 h-3.5 text-[#B6532B]" />
            <span>Protip: Click directly on the contour map to set new start weights.</span>
          </div>
        </div>
      </div>

      {/* Landscape Canvas display */}
      <div className="md:col-span-7 space-y-4 flex flex-col justify-between">
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl flex flex-col items-center shadow-inner relative">
          <div className="relative border border-[#E5DDD0] rounded-xl overflow-hidden cursor-crosshair">
            <canvas ref={canvasRef} onClick={handleCanvasClick} className="w-[300px] h-[300px] sm:w-[400px] sm:h-[400px] block" />
            <div className="absolute top-3 left-3 bg-[#2E251E]/90 px-2.5 py-1 rounded-lg border border-[#E5DDD0]/20 text-[9px] font-mono text-[#FAF6EE] flex items-center gap-1 shadow-md">
              <Info className="w-3.5 h-3.5 text-[#C18C3B]" />
              <span>Loss Basin J(w_1, w_2)</span>
            </div>
          </div>
        </div>

        {/* Live Loss Chart (Convergence Panel) */}
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-inner">
          <div className="flex items-center justify-between mb-2">
            <h5 className="text-[#2E251E] text-xs font-semibold tracking-wide font-mono">Real-time Optimization Plot</h5>
            <span className="text-[10px] text-[#B6532B] font-mono font-bold">Objective: J(w) → 0</span>
          </div>
          <div className="border border-[#E5DDD0] rounded-xl bg-white p-2 overflow-hidden flex justify-center shadow-sm">
            <canvas ref={lossCanvasRef} className="block" />
          </div>
        </div>
      </div>
    </div>
  );
};
