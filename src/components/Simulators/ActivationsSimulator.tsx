import React, { useState, useEffect, useRef } from 'react';
import { Activity } from 'lucide-react';

type ActivationType = 'sigmoid' | 'tanh' | 'relu' | 'leaky_relu';

export const ActivationsSimulator: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [actType, setActType] = useState<ActivationType>('sigmoid');
  const [z, setZ] = useState(0); // Input value
  const [isDragging, setIsDragging] = useState(false);

  // Activation functions and their derivatives
  const funcs = {
    sigmoid: {
      f: (x: number) => 1 / (1 + Math.exp(-x)),
      df: (x: number) => {
        const s = 1 / (1 + Math.exp(-x));
        return s * (1 - s);
      },
      name: 'Sigmoid',
      range: [0, 1]
    },
    tanh: {
      f: (x: number) => Math.tanh(x),
      df: (x: number) => 1 - Math.tanh(x)**2,
      name: 'Tanh',
      range: [-1, 1]
    },
    relu: {
      f: (x: number) => Math.max(0, x),
      df: (x: number) => x > 0 ? 1 : 0,
      name: 'ReLU',
      range: [0, '∞']
    },
    leaky_relu: {
      f: (x: number) => x > 0 ? x : 0.1 * x,
      df: (x: number) => x > 0 ? 1 : 0.1,
      name: 'Leaky ReLU',
      range: ['-∞', '∞']
    }
  };

  const { f, df } = funcs[actType];
  const a = f(z);
  const grad = df(z);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width = 400;
    const height = canvas.height = 300;
    const cx = width / 2;
    const cy = height / 2;
    
    // Scale: x goes from -5 to 5. y goes from -2 to 2 (except for ReLU which we might scale differently if needed, but -2 to 2 is fine for visualization)
    const scaleX = width / 10;
    const scaleY = height / 4;

    ctx.fillStyle = '#FAF6EE';
    ctx.fillRect(0, 0, width, height);

    // Grid
    ctx.strokeStyle = 'rgba(46, 37, 30, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 0; x <= width; x += scaleX) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
    }
    for (let y = 0; y <= height; y += scaleY) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = 'rgba(46, 37, 30, 0.2)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, cy); ctx.lineTo(width, cy); // X axis
    ctx.moveTo(cx, 0); ctx.lineTo(cx, height); // Y axis
    ctx.stroke();

    // Draw Function f(x)
    ctx.beginPath();
    ctx.strokeStyle = '#3B7A57'; // Green
    ctx.lineWidth = 3;
    for (let px = 0; px <= width; px += 2) {
      const vx = (px - cx) / scaleX;
      const vy = f(vx);
      const py = cy - vy * scaleY;
      if (px === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Draw Derivative f'(x)
    ctx.beginPath();
    ctx.strokeStyle = '#B6532B'; // Terracotta
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    for (let px = 0; px <= width; px += 2) {
      const vx = (px - cx) / scaleX;
      const vy = df(vx);
      const py = cy - vy * scaleY;
      if (px === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw point Z
    const pzX = cx + z * scaleX;
    const pzY = cy - a * scaleY;

    // Tangent line at Z
    const tLen = 1.5; // length of tangent line in plot units
    ctx.beginPath();
    ctx.strokeStyle = '#C18C3B'; // Ochre
    ctx.lineWidth = 2;
    const tx1 = pzX - tLen * scaleX;
    const ty1 = pzY + tLen * grad * scaleY; // + because y is flipped
    const tx2 = pzX + tLen * scaleX;
    const ty2 = pzY - tLen * grad * scaleY;
    ctx.moveTo(tx1, ty1);
    ctx.lineTo(tx2, ty2);
    ctx.stroke();

    // Lines to axes
    ctx.strokeStyle = 'rgba(46, 37, 30, 0.3)';
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(pzX, pzY); ctx.lineTo(pzX, cy);
    ctx.moveTo(pzX, pzY); ctx.lineTo(cx, pzY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Dot at Z
    ctx.beginPath();
    ctx.arc(pzX, pzY, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#C18C3B';
    ctx.fill();
    ctx.strokeStyle = '#FAF6EE';
    ctx.stroke();

  }, [actType, z, f, df, a, grad]);

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    updateZ(e);
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (isDragging) updateZ(e);
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    setIsDragging(false);
    (e.target as HTMLElement).releasePointerCapture(e.pointerId);
  };

  const updateZ = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const scaleX = canvas.width / rect.width;
    const px = x * scaleX;
    
    // Map px to [-5, 5]
    let newZ = (px - (canvas.width / 2)) / (canvas.width / 10);
    newZ = Math.max(-5, Math.min(5, newZ));
    setZ(newZ);
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-6 h-full bg-[#FAF6EE]">
      <div className="lg:col-span-5 bg-[#F4EFE6] border border-[#E5DDD0] p-6 rounded-2xl flex flex-col justify-between shadow-xl">
        <div className="space-y-6">
          <h4 className="text-[#2E251E] font-bold text-xl tracking-tight flex items-center gap-3">
            <Activity className="w-6 h-6 text-[#B6532B]" /> Activation Functions
          </h4>
          <p className="text-[#6E6257] text-sm leading-relaxed">
            Drag along the X-axis to change the input $z$. The solid line is the activation $a = f(z)$, and the dashed line is its derivative $f'(z)$. 
            Notice how Sigmoid and Tanh gradients vanish at large $|z|$.
          </p>
          
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-2 bg-[#FAF6EE] p-1 rounded-xl border border-[#E5DDD0]">
              {(Object.keys(funcs) as ActivationType[]).map(type => (
                <button
                  key={type}
                  onClick={() => setActType(type)}
                  className={`py-2 rounded-lg text-xs font-bold transition-all ${
                    actType === type
                      ? 'bg-white shadow-sm text-[#B6532B]'
                      : 'text-[#6E6257] hover:text-[#2E251E]'
                  }`}
                >
                  {funcs[type].name}
                </button>
              ))}
            </div>
            
            <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-4 flex flex-col gap-3">
              <div className="flex justify-between items-center text-xs font-mono">
                <span className="text-[#6E6257] font-bold">Input (z)</span>
                <span className="text-[#2E251E] font-bold bg-[#E5DDD0] px-2 py-1 rounded w-16 text-center">{z.toFixed(2)}</span>
              </div>
              <div className="flex justify-between items-center text-xs font-mono">
                <span className="text-[#6E6257] font-bold">Output (a)</span>
                <span className="text-[#3B7A57] font-bold bg-[#E5DDD0] px-2 py-1 rounded w-16 text-center">{a.toFixed(2)}</span>
              </div>
              <div className="flex justify-between items-center text-xs font-mono">
                <span className="text-[#6E6257] font-bold">Gradient (f')</span>
                <span className="text-[#B6532B] font-bold bg-[#E5DDD0] px-2 py-1 rounded w-16 text-center">{grad.toFixed(2)}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="lg:col-span-7 flex flex-col items-center justify-center">
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-2 rounded-2xl shadow-xl overflow-hidden mb-4 relative group">
          <canvas 
            ref={canvasRef} 
            className="rounded-xl cursor-ew-resize touch-none w-full max-w-[400px] h-auto"
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerUp}
          />
          <div className="absolute top-4 left-4 bg-[#F4EFE6]/90 backdrop-blur-md px-3 py-2 rounded-xl border border-[#E5DDD0] text-[10px] font-mono text-[#2E251E] shadow-lg flex flex-col gap-1 pointer-events-none">
            <span className="flex items-center gap-2"><div className="w-3 h-0.5 bg-[#3B7A57]"></div> f(z)</span>
            <span className="flex items-center gap-2"><div className="w-3 h-0 border-t border-dashed border-[#B6532B]"></div> f'(z) [Gradient]</span>
          </div>
        </div>
      </div>
    </div>
  );
};
