import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { RotateCcw, Crosshair, Sliders, Eye, EyeOff } from 'lucide-react';

interface Point {
  x: number;
  y: number;
}

interface OptimizerPath {
  key: string;
  name: string;
  color: string;
  points: Point[];
  losses: number[];
}

const SURFACE = {
  loss: (x: number, y: number) => 0.1 * x * x + 0.8 * y * y,
  grad: (x: number, y: number) => ({ dx: 0.2 * x, dy: 1.6 * y }),
};

const OPTIMIZER_CONFIGS: Record<string, { name: string; color: string }> = {
  sgd: { name: 'SGD', color: '#B6532B' },
  momentum: { name: 'Momentum', color: '#C18C3B' },
  rmsprop: { name: 'RMSprop', color: '#3B7A57' },
  adam: { name: 'Adam', color: '#2E251E' },
};

const CANVAS_SIZE = 400;
const WORLD_BOUNDS = 2.2;
const EPS = 1e-8;

function toCanvas(wx: number, wy: number): [number, number] {
  return [
    ((wx / WORLD_BOUNDS) + 1) / 2 * CANVAS_SIZE,
    ((wy / WORLD_BOUNDS) + 1) / 2 * CANVAS_SIZE,
  ];
}

function toWorld(px: number, py: number): [number, number] {
  return [
    (px / CANVAS_SIZE * 2 - 1) * WORLD_BOUNDS,
    (py / CANVAS_SIZE * 2 - 1) * WORLD_BOUNDS,
  ];
}

function computePaths(start: Point, lr: number, n: number): OptimizerPath[] {
  const results: OptimizerPath[] = [];
  const sx = start.x, sy = start.y;

  {
    const pts: Point[] = [{ x: sx, y: sy }];
    for (let i = 0; i < n; i++) {
      const g = SURFACE.grad(pts[i].x, pts[i].y);
      pts.push({ x: pts[i].x - lr * g.dx, y: pts[i].y - lr * g.dy });
    }
    results.push({ key: 'sgd', ...OPTIMIZER_CONFIGS.sgd, points: pts, losses: pts.map(p => SURFACE.loss(p.x, p.y)) });
  }

  {
    const pts: Point[] = [{ x: sx, y: sy }];
    let vx = 0, vy = 0;
    for (let i = 0; i < n; i++) {
      const g = SURFACE.grad(pts[i].x, pts[i].y);
      vx = 0.9 * vx + lr * g.dx;
      vy = 0.9 * vy + lr * g.dy;
      pts.push({ x: pts[i].x - vx, y: pts[i].y - vy });
    }
    results.push({ key: 'momentum', ...OPTIMIZER_CONFIGS.momentum, points: pts, losses: pts.map(p => SURFACE.loss(p.x, p.y)) });
  }

  {
    const pts: Point[] = [{ x: sx, y: sy }];
    let sx2 = 0, sy2 = 0;
    for (let i = 0; i < n; i++) {
      const g = SURFACE.grad(pts[i].x, pts[i].y);
      sx2 = 0.9 * sx2 + 0.1 * g.dx * g.dx;
      sy2 = 0.9 * sy2 + 0.1 * g.dy * g.dy;
      pts.push({
        x: pts[i].x - lr * g.dx / (Math.sqrt(sx2) + EPS),
        y: pts[i].y - lr * g.dy / (Math.sqrt(sy2) + EPS),
      });
    }
    results.push({ key: 'rmsprop', ...OPTIMIZER_CONFIGS.rmsprop, points: pts, losses: pts.map(p => SURFACE.loss(p.x, p.y)) });
  }

  {
    const pts: Point[] = [{ x: sx, y: sy }];
    let mx = 0, my = 0, vx = 0, vy = 0;
    for (let i = 0; i < n; i++) {
      const g = SURFACE.grad(pts[i].x, pts[i].y);
      const t = i + 1;
      mx = 0.9 * mx + 0.1 * g.dx;
      my = 0.9 * my + 0.1 * g.dy;
      vx = 0.999 * vx + 0.001 * g.dx * g.dx;
      vy = 0.999 * vy + 0.001 * g.dy * g.dy;
      const mhx = mx / (1 - Math.pow(0.9, t));
      const mhy = my / (1 - Math.pow(0.9, t));
      const vhx = vx / (1 - Math.pow(0.999, t));
      const vhy = vy / (1 - Math.pow(0.999, t));
      pts.push({
        x: pts[i].x - lr * mhx / (Math.sqrt(vhx) + EPS),
        y: pts[i].y - lr * mhy / (Math.sqrt(vhy) + EPS),
      });
    }
    results.push({ key: 'adam', ...OPTIMIZER_CONFIGS.adam, points: pts, losses: pts.map(p => SURFACE.loss(p.x, p.y)) });
  }

  return results;
}

function drawContours(ctx: CanvasRenderingContext2D) {
  const cx = CANVAS_SIZE / 2, cy = CANVAS_SIZE / 2;
  const levels = [0.1, 0.3, 0.6, 1.0, 1.5, 2.5, 4.0];
  ctx.lineWidth = 0.5;
  levels.forEach((level, i) => {
    const rx = Math.sqrt(level / 0.1);
    const ry = Math.sqrt(level / 0.8);
    const px = (rx / WORLD_BOUNDS) * (CANVAS_SIZE / 2);
    const py = (ry / WORLD_BOUNDS) * (CANVAS_SIZE / 2);
    ctx.strokeStyle = `rgba(46, 37, 30, ${0.06 + i * 0.025})`;
    ctx.beginPath();
    ctx.ellipse(cx, cy, px, py, 0, 0, Math.PI * 2);
    ctx.stroke();
  });
}

function drawBackground(ctx: CanvasRenderingContext2D) {
  const imgData = ctx.createImageData(CANVAS_SIZE, CANVAS_SIZE);
  const maxLoss = SURFACE.loss(WORLD_BOUNDS, WORLD_BOUNDS);
  for (let py = 0; py < CANVAS_SIZE; py++) {
    for (let px = 0; px < CANVAS_SIZE; px++) {
      const [wx, wy] = toWorld(px, py);
      const ratio = Math.min(1, SURFACE.loss(wx, wy) / maxLoss);
      const idx = (py * CANVAS_SIZE + px) * 4;
      imgData.data[idx] = Math.floor(250 * (1 - ratio) + 229 * ratio);
      imgData.data[idx + 1] = Math.floor(246 * (1 - ratio) + 221 * ratio);
      imgData.data[idx + 2] = Math.floor(238 * (1 - ratio) + 208 * ratio);
      imgData.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
}

export const OptimizerCompare: React.FC = () => {
  const [startPoint, setStartPoint] = useState<Point>({ x: 2.0, y: 1.5 });
  const [learningRate, setLearningRate] = useState(0.12);
  const [numSteps, setNumSteps] = useState(25);
  const [toggles, setToggles] = useState<Record<string, boolean>>({
    sgd: true, momentum: true, rmsprop: true, adam: true,
  });

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lossCanvasRef = useRef<HTMLCanvasElement>(null);
  const bgDrawn = useRef(false);

  const paths = useMemo(
    () => computePaths(startPoint, learningRate, numSteps),
    [startPoint, learningRate, numSteps]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;

    if (!bgDrawn.current) {
      drawBackground(ctx);
      bgDrawn.current = true;
    }

    drawContours(ctx);

    const [sx, sy] = toCanvas(startPoint.x, startPoint.y);
    ctx.beginPath();
    ctx.arc(sx, sy, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#2E251E';
    ctx.fill();
    ctx.strokeStyle = '#FAF6EE';
    ctx.lineWidth = 2;
    ctx.stroke();

    paths.forEach(path => {
      if (!toggles[path.key]) return;

      ctx.strokeStyle = path.color;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      path.points.forEach((pt, i) => {
        const [px, py] = toCanvas(pt.x, pt.y);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();

      path.points.forEach((pt, i) => {
        if (i === 0) return;
        const [px, py] = toCanvas(pt.x, pt.y);
        ctx.beginPath();
        ctx.arc(px, py, 2, 0, Math.PI * 2);
        ctx.fillStyle = path.color;
        ctx.fill();
      });

      const last = path.points[path.points.length - 1];
      const [lpx, lpy] = toCanvas(last.x, last.y);
      ctx.beginPath();
      ctx.arc(lpx, lpy, 5, 0, Math.PI * 2);
      ctx.fillStyle = path.color;
      ctx.fill();
      ctx.strokeStyle = '#FAF6EE';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });
  }, [paths, toggles, startPoint]);

  useEffect(() => {
    const canvas = lossCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width = 400;
    const h = canvas.height = 160;

    ctx.fillStyle = '#FAF6EE';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = '#E5DDD0';
    ctx.lineWidth = 0.5;
    for (let i = 30; i < w; i += 35) {
      ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, h); ctx.stroke();
    }
    for (let i = 20; i < h; i += 25) {
      ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(w, i); ctx.stroke();
    }

    let maxLoss = 0.5;
    paths.forEach(p => {
      if (toggles[p.key]) p.losses.forEach(l => { if (l > maxLoss) maxLoss = l; });
    });
    maxLoss = Math.max(0.1, maxLoss);

    paths.forEach(path => {
      if (!toggles[path.key]) return;
      ctx.beginPath();
      ctx.strokeStyle = path.color;
      ctx.lineWidth = 2;
      path.losses.forEach((lossVal, idx) => {
        const x = (idx / Math.max(1, numSteps)) * (w - 30) + 15;
        const y = h - 10 - (Math.max(0, lossVal) / maxLoss) * (h - 30);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });

    ctx.font = '9px monospace';
    let lx = 15;
    paths.forEach(path => {
      if (!toggles[path.key]) return;
      ctx.fillStyle = path.color;
      ctx.fillRect(lx, 6, 8, 8);
      ctx.fillStyle = '#2E251E';
      ctx.fillText(path.name, lx + 12, 13);
      lx += 70;
    });

    ctx.fillStyle = '#6E6257';
    ctx.font = '8px monospace';
    ctx.fillText('Iteration', w - 55, h - 4);
  }, [paths, toggles, numSteps]);

  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width;
    const sy = canvas.height / rect.height;
    const px = (e.clientX - rect.left) * sx;
    const py = (e.clientY - rect.top) * sy;
    const [wx, wy] = toWorld(px, py);
    setStartPoint({
      x: Math.max(-WORLD_BOUNDS, Math.min(WORLD_BOUNDS, wx)),
      y: Math.max(-WORLD_BOUNDS, Math.min(WORLD_BOUNDS, wy)),
    });
  }, []);

  const toggleOptimizer = useCallback((key: string) => {
    setToggles(prev => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const reset = useCallback(() => {
    setStartPoint({ x: 2.0, y: 1.5 });
    setLearningRate(0.12);
    setNumSteps(25);
    setToggles({ sgd: true, momentum: true, rmsprop: true, adam: true });
  }, []);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 p-3 h-full">
      <div className="lg:col-span-6 flex flex-col items-center justify-center bg-[#FAF6EE] border border-[#E5DDD0] rounded-2xl p-3 shadow-inner">
        <div className="relative border border-[#E5DDD0] rounded-xl overflow-hidden cursor-crosshair">
          <canvas
            ref={canvasRef}
            onClick={handleCanvasClick}
            className="w-[300px] h-[300px] sm:w-[360px] sm:h-[360px] block"
          />
          <div className="absolute top-2 left-2 bg-[#2E251E]/85 px-2 py-1 rounded-md text-[8px] font-mono text-[#FAF6EE] flex items-center gap-1">
            <Crosshair className="w-2.5 h-2.5 text-[#C18C3B]" />
            Click contour to set start
          </div>
        </div>
      </div>

      <div className="lg:col-span-6 flex flex-col gap-3">
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-2xl p-3 shadow-inner">
          <div className="flex items-center justify-between mb-1">
            <h5 className="text-[#2E251E] text-xs font-semibold font-mono">Loss vs Iteration</h5>
          </div>
          <div className="border border-[#E5DDD0] rounded-xl bg-white overflow-hidden">
            <canvas ref={lossCanvasRef} className="block w-full" />
          </div>
        </div>

        <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-2xl p-4 shadow-inner flex flex-col gap-3">
          <div className="flex items-center gap-2 text-[#2E251E] text-xs font-semibold font-mono mb-1">
            <Sliders className="w-4 h-4 text-[#B6532B]" />
            Controls
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-[#2E251E]">Learning Rate (η)</span>
                <span className="text-[#B6532B] font-semibold">{learningRate.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.01"
                max="1.0"
                step="0.01"
                value={learningRate}
                onChange={e => setLearningRate(parseFloat(e.target.value))}
                className="w-full h-1 bg-[#E5DDD0] rounded-lg appearance-none cursor-pointer accent-[#B6532B]"
              />
            </div>
            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-[#2E251E]">Steps</span>
                <span className="text-[#B6532B] font-semibold">{numSteps}</span>
              </div>
              <input
                type="range"
                min="5"
                max="50"
                step="1"
                value={numSteps}
                onChange={e => setNumSteps(parseInt(e.target.value))}
                className="w-full h-1 bg-[#E5DDD0] rounded-lg appearance-none cursor-pointer accent-[#B6532B]"
              />
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            {Object.entries(OPTIMIZER_CONFIGS).map(([key, config]) => (
              <button
                key={key}
                onClick={() => toggleOptimizer(key)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono border transition-all cursor-pointer ${
                  toggles[key]
                    ? 'border-transparent text-white shadow-sm'
                    : 'border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257] opacity-50'
                }`}
                style={toggles[key] ? { backgroundColor: config.color, borderColor: config.color } : {}}
              >
                {toggles[key] ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
                {config.name}
              </button>
            ))}
          </div>

          <div className="flex items-center justify-between pt-2 border-t border-[#E5DDD0]">
            <div className="text-[10px] text-[#6E6257] font-mono">
              Start: ({startPoint.x.toFixed(2)}, {startPoint.y.toFixed(2)})
            </div>
            <button
              onClick={reset}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[#E5DDD0] bg-[#F4EFE6] text-[#6E6257] hover:text-[#2E251E] hover:bg-[#EEE8DA] text-xs font-mono transition-colors cursor-pointer"
            >
              <RotateCcw className="w-3 h-3" />
              Reset
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
