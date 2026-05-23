import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Crosshair } from 'lucide-react';

export const GradientDescent: React.FC = () => {
  // Params
  const [learningRate, setLearningRate] = useState<number>(0.1);
  const [optimizer, setOptimizer] = useState<'sgd' | 'momentum' | 'adam'>('sgd');
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [epoch, setEpoch] = useState<number>(0);
  
  // Simulation Coordinates (from -2 to 2)
  const [posX, setPosX] = useState<number>(1.6);
  const [posY, setPosY] = useState<number>(1.4);
  const [history, setHistory] = useState<Array<{ x: number; y: number; loss: number }>>([]);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lossCanvasRef = useRef<HTMLCanvasElement>(null);

  // Optimizer parameters
  const velocityRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const mRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const vRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  // Mathematical Loss Function: Himmelblau-like Wavy Basin
  const getLoss = (x: number, y: number) => {
    return 0.25 * (x * x + y * y * 1.5) - 0.1 * Math.cos(4 * x) - 0.1 * Math.cos(4 * y) + 0.2;
  };

  const getGradient = (x: number, y: number) => {
    return {
      dx: 0.5 * x + 0.4 * Math.sin(4 * x),
      dy: 0.75 * y + 0.4 * Math.sin(4 * y)
    };
  };

  const resetSimulation = () => {
    setIsPlaying(false);
    setEpoch(0);
    const startX = 1.6;
    const startY = 1.4;
    setPosX(startX);
    setPosY(startY);
    setHistory([{ x: startX, y: startY, loss: getLoss(startX, startY) }]);
    velocityRef.current = { x: 0, y: 0 };
    mRef.current = { x: 0, y: 0 };
    vRef.current = { x: 0, y: 0 };
  };

  // Perform single training step
  const step = () => {
    let currentX = posX;
    let currentY = posY;

    // Boundary check
    if (Math.abs(currentX) > 2.2 || Math.abs(currentY) > 2.2) {
      setIsPlaying(false);
      return;
    }

    const { dx, dy } = getGradient(currentX, currentY);
    let nextX = currentX;
    let nextY = currentY;

    if (optimizer === 'sgd') {
      nextX = currentX - learningRate * dx;
      nextY = currentY - learningRate * dy;
    } else if (optimizer === 'momentum') {
      const beta = 0.9;
      velocityRef.current.x = beta * velocityRef.current.x + learningRate * dx;
      velocityRef.current.y = beta * velocityRef.current.y + learningRate * dy;
      nextX = currentX - velocityRef.current.x;
      nextY = currentY - velocityRef.current.y;
    } else if (optimizer === 'adam') {
      const beta1 = 0.9;
      const beta2 = 0.999;
      const eps = 1e-8;
      const t = epoch + 1;

      // Update biased first moment estimate
      mRef.current.x = beta1 * mRef.current.x + (1 - beta1) * dx;
      mRef.current.y = beta1 * mRef.current.y + (1 - beta1) * dy;

      // Update biased second raw moment estimate
      vRef.current.x = beta2 * vRef.current.x + (1 - beta2) * dx * dx;
      vRef.current.y = beta2 * vRef.current.y + (1 - beta2) * dy * dy;

      // Compute bias-corrected estimates
      const mHatX = mRef.current.x / (1 - Math.pow(beta1, t));
      const mHatY = mRef.current.y / (1 - Math.pow(beta1, t));
      const vHatX = vRef.current.x / (1 - Math.pow(beta2, t));
      const vHatY = vRef.current.y / (1 - Math.pow(beta2, t));

      nextX = currentX - (learningRate * mHatX) / (Math.sqrt(vHatX) + eps);
      nextY = currentY - (learningRate * mHatY) / (Math.sqrt(vHatY) + eps);
    }

    setPosX(nextX);
    setPosY(nextY);
    setEpoch(prev => prev + 1);

    const nextLoss = getLoss(nextX, nextY);
    setHistory(prev => [...prev, { x: nextX, y: nextY, loss: nextLoss }]);

  };

  const stepRef = useRef(step);
  useEffect(() => {
    stepRef.current = step;
  }, [step]);

  // Auto-start descent on mount for one-click experience
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

  // Click on contour to move starting position
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    // Convert pixel to coordinates (-2.2 to 2.2)
    const cx = ((px / canvas.width) - 0.5) * 4.4;
    const cy = ((py / canvas.height) - 0.5) * 4.4;

    setPosX(cx);
    setPosY(cy);
    setEpoch(0);
    setHistory([{ x: cx, y: cy, loss: getLoss(cx, cy) }]);
    setIsPlaying(false);
    velocityRef.current = { x: 0, y: 0 };
    mRef.current = { x: 0, y: 0 };
    vRef.current = { x: 0, y: 0 };
  };

  // Draw Loss Surface contour map
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width = 400;
    const height = canvas.height = 400;

    // Render Contour background pixels (cream-to-terracotta gradient map)
    const imgData = ctx.createImageData(width, height);
    for (let py = 0; py < height; py++) {
      for (let px = 0; px < width; px++) {
        // Map pixel coordinates to mathematical domain [-2.2, 2.2]
        const cx = ((px / width) - 0.5) * 4.4;
        const cy = ((py / height) - 0.5) * 4.4;
        
        const loss = getLoss(cx, cy);

        // Map loss to pixel color
        const idx = (py * width + px) * 4;
        
        // Low loss = Cream #FAF6EE (250, 246, 238)
        // High loss = Soft Terracotta #B6532B (182, 83, 43)
        const ratio = Math.min(1, loss / 1.6);
        const r = Math.floor(250 * (1 - ratio) + 182 * ratio);
        const g = Math.floor(246 * (1 - ratio) + 83 * ratio);
        const b = Math.floor(238 * (1 - ratio) + 43 * ratio);
        
        imgData.data[idx] = r;
        imgData.data[idx+1] = g;
        imgData.data[idx+2] = b;
        imgData.data[idx+3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);

    // Draw Contour level curves (nested outlines)
    ctx.lineWidth = 1;
    for (let level = 1; level <= 6; level++) {
      ctx.strokeStyle = `rgba(46, 37, 30, ${0.04 * level})`;
      ctx.beginPath();
      // Simple circle contours for rendering visual lines
      const r = Math.sqrt(level * 0.7) * (width / 4.4);
      ctx.arc(width / 2, height / 2, r, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Draw grid axes
    ctx.strokeStyle = '#E5DDD0';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(width / 2, 0);
    ctx.lineTo(width / 2, height);
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Draw optimization history trail
    if (history.length > 0) {
      ctx.strokeStyle = '#B6532B'; // Terracotta
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      
      history.forEach((pos, idx) => {
        // Coordinate to pixel mapping
        const px = ((pos.x / 4.4) + 0.5) * width;
        const py = ((pos.y / 4.4) + 0.5) * height;
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();

      // Draw historical dots
      history.forEach((pos) => {
        const px = ((pos.x / 4.4) + 0.5) * width;
        const py = ((pos.y / 4.4) + 0.5) * height;
        ctx.beginPath();
        ctx.arc(px, py, 3, 0, Math.PI * 2);
        ctx.fillStyle = '#C18C3B'; // Gold/Ochre
        ctx.fill();
      });

      // Draw current active dot
      const last = history[history.length - 1];
      const px = ((last.x / 4.4) + 0.5) * width;
      const py = ((last.y / 4.4) + 0.5) * height;
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#2E251E'; // Charcoal
      ctx.fill();
      ctx.strokeStyle = '#FAF6EE';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

  }, [history]);

  // Draw live update Loss Chart
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
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, h);
      ctx.stroke();
    }
    for (let i = 20; i < h; i += 30) {
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(w, i);
      ctx.stroke();
    }

    if (history.length === 0) return;

    // Draw Loss Path Curve
    ctx.beginPath();
    ctx.strokeStyle = '#B6532B'; // Terracotta
    ctx.lineWidth = 2;

    const maxLoss = Math.max(0.5, ...history.map(pt => pt.loss));
    history.forEach((pt, idx) => {
      const x = (idx / Math.max(20, history.length)) * (w - 20) + 10;
      const y = h - (pt.loss / maxLoss) * (h - 20) - 10;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Text labels
    ctx.fillStyle = '#2E251E';
    ctx.font = '9px monospace';
    ctx.fillText(`Loss: ${history[history.length - 1].loss.toFixed(4)}`, 15, 20);
    ctx.fillText(`Epochs: ${epoch}`, 15, 32);

  }, [history, epoch]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-12 gap-6 p-4">
      {/* Parameters Panel */}
      <div className="md:col-span-5 bg-[#F4EFE6] border border-[#E5DDD0] p-5 rounded-2xl flex flex-col justify-between">
        <div>
          <h4 className="text-[#2E251E] font-bold text-base mb-1 tracking-wide flex items-center gap-2">
            <Crosshair className="w-5 h-5 text-[#B6532B]" /> Gradient Settings
          </h4>
          <p className="text-[#6E6257] text-xs mb-5">
            Tweak hyper-parameters to guide the agent down the contour basin.
          </p>

          <div className="space-y-5">
            {/* Learning Rate Slider */}
            <div>
              <div className="flex justify-between text-xs font-mono mb-2">
                <span className="text-[#2E251E]">Learning Rate (η)</span>
                <span className="text-[#B6532B] font-semibold">{learningRate.toFixed(3)}</span>
              </div>
              <input
                type="range"
                min="0.01"
                max="0.8"
                step="0.01"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                className="w-full h-1 bg-[#E5DDD0] rounded-lg appearance-none cursor-pointer accent-[#B6532B]"
              />
              <div className="flex justify-between text-[10px] text-[#6E6257] font-mono mt-1">
                <span>0.01 (Cautious)</span>
                <span>0.8 (Overshooting)</span>
              </div>
            </div>

            {/* Optimizer selector */}
            <div>
              <label className="text-xs text-[#2E251E] font-mono block mb-2 font-semibold">Optimizer selection</label>
              <div className="grid grid-cols-3 gap-2">
                {(['sgd', 'momentum', 'adam'] as const).map((opt) => (
                  <button
                    key={opt}
                    onClick={() => setOptimizer(opt)}
                    className={`py-2 rounded-xl text-xs font-mono transition-all capitalize border ${
                      optimizer === opt
                        ? 'border-[#B6532B] bg-[#B6532B]/10 text-[#B6532B] font-semibold'
                        : 'border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257] hover:text-[#2E251E] hover:bg-[#F4EFE6]'
                    }`}
                  >
                    {opt}
                  </button>
                ))}
              </div>
            </div>

            {/* Coordinates */}
            <div className="p-3 bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl space-y-2">
              <span className="text-[10px] text-[#6E6257] font-mono block uppercase tracking-wider">Position Log</span>
              <div className="grid grid-cols-2 text-xs font-mono text-[#2E251E]">
                <div>
                  <span className="text-[#6E6257]">w_1:</span>
                  <span className="font-bold ml-2">{posX.toFixed(3)}</span>
                </div>
                <div>
                  <span className="text-[#6E6257]">w_2:</span>
                  <span className="font-bold ml-2">{posY.toFixed(3)}</span>
                </div>
              </div>
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
                  : 'bg-[#B6532B] border-[#B6532B] text-white hover:bg-[#B6532B]/95'
              }`}
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {isPlaying ? 'Pause optimization' : 'Start descent'}
            </button>
            
            <button
              onClick={resetSimulation}
              className="p-3 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257] hover:text-[#2E251E] transition-colors cursor-pointer"
              title="Reset weights"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

          <div className="text-[10px] text-[#6E6257] flex items-center gap-1.5 font-mono justify-center">
            <Crosshair className="w-3.5 h-3.5 text-[#B6532B]" />
            <span>Protip: Click directly on the contour map to set start weights.</span>
          </div>
        </div>
      </div>

      {/* Vis Canvas Display */}
      <div className="md:col-span-7 space-y-6">
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl flex flex-col items-center shadow-inner">
          <div className="relative border border-[#E5DDD0] rounded-xl overflow-hidden cursor-crosshair">
            <canvas ref={canvasRef} onClick={handleCanvasClick} className="w-[300px] h-[300px] sm:w-[400px] sm:h-[400px]" />
            <div className="absolute top-3 left-3 bg-[#2E251E]/90 px-2.5 py-1 rounded-lg border border-[#E5DDD0]/20 text-[9px] font-mono text-[#FAF6EE]">
              Loss Terrain Basin J(w_1, w_2)
            </div>
            

          </div>
        </div>

        {/* Live Loss Chart */}
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl">
          <div className="flex items-center justify-between mb-2">
            <h5 className="text-[#2E251E] text-xs font-semibold tracking-wide font-mono">Real-time Optimization Plot</h5>
            <span className="text-[10px] text-[#B6532B] font-mono font-bold">Objective: J(w) → 0</span>
          </div>
          <div className="border border-[#E5DDD0] rounded-xl bg-white p-2 overflow-hidden flex justify-center">
            <canvas ref={lossCanvasRef} />
          </div>
        </div>
      </div>
    </div>
  );
};
