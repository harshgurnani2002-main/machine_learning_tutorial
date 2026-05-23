import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Shuffle } from 'lucide-react';

export const GANSimulator: React.FC = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [genLR, setGenLR] = useState(0.01);
  const [discLR, setDiscLR] = useState(0.01);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Target Distribution (Real Data) - e.g., a circle
  const [realData] = useState(() => {
    const data = [];
    for (let i = 0; i < 50; i++) {
      const angle = Math.random() * Math.PI * 2;
      const r = 0.6 + Math.random() * 0.1;
      data.push({ x: Math.cos(angle) * r, y: Math.sin(angle) * r });
    }
    return data;
  });

  // Generator "Weights" (simply moving fake points towards the real distribution)
  const [fakeData, setFakeData] = useState(() => {
    const data = [];
    for (let i = 0; i < 50; i++) {
      data.push({ x: (Math.random() - 0.5) * 0.5, y: (Math.random() - 0.5) * 0.5 });
    }
    return data;
  });
  
  // Discriminator "Weights" (a simple center/radius or basic network simulated here)
  const [discCenter, setDiscCenter] = useState({ x: 0, y: 0 });
  const [discRadius, setDiscRadius] = useState(0.3);

  const initGAN = () => {
    setFakeData(() => {
      const data = [];
      for (let i = 0; i < 50; i++) {
        data.push({ x: (Math.random() - 0.5) * 0.5, y: (Math.random() - 0.5) * 0.5 });
      }
      return data;
    });
    setDiscCenter({ x: 0, y: 0 });
    setDiscRadius(0.3);
    setEpoch(0);
    setIsPlaying(false);
  };

  const trainStep = () => {
    // 1. Train Discriminator
    // Disc wants to correctly classify Real (distance ~ 0.65) vs Fake
    // In this toy example, Discriminator learns the center and radius of real data.
    let newDX = discCenter.x;
    let newDY = discCenter.y;
    let newDR = discRadius;

    // Real data pulls disc towards true radius (0.65) and center (0,0)
    let avgRealR = 0;
    realData.forEach(p => {
      avgRealR += Math.sqrt(p.x*p.x + p.y*p.y);
    });
    avgRealR /= realData.length;
    
    // Gradient step for Discriminator
    newDR += discLR * (avgRealR - newDR);
    
    // 2. Train Generator
    // Gen wants fake data to fool discriminator (move towards discRadius and discCenter)
    const newFake = fakeData.map(p => {
      const dist = Math.sqrt((p.x - newDX)**2 + (p.y - newDY)**2);
      const angle = Math.atan2(p.y - newDY, p.x - newDX);
      
      // Target distance for generator is the discriminator's current boundary
      const targetDist = newDR;
      const moveAmount = (targetDist - dist) * genLR * 5; // *5 for visual speed
      
      return {
        x: p.x + Math.cos(angle) * moveAmount + (Math.random() - 0.5) * 0.05,
        y: p.y + Math.sin(angle) * moveAmount + (Math.random() - 0.5) * 0.05
      };
    });

    setDiscCenter({ x: newDX, y: newDY });
    setDiscRadius(newDR);
    setFakeData(newFake);
    setEpoch(prev => prev + 1);
  };

  // Auto-start training on mount for one-click experience
  useEffect(() => {
    const timer = setTimeout(() => setIsPlaying(true), 700);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    let animId: number;
    if (isPlaying) {
      const run = () => {
        trainStep();
        animId = requestAnimationFrame(run);
      };
      animId = requestAnimationFrame(run);
    }
    return () => cancelAnimationFrame(animId);
  }, [isPlaying, fakeData, discCenter, discRadius, genLR, discLR]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const size = canvas.width = 400;
    const hSize = canvas.height = 400;
    const cx = size / 2;
    const cy = hSize / 2;
    const scale = size / 3; // mapping [-1.5, 1.5] to canvas

    ctx.fillStyle = '#FAF6EE';
    ctx.fillRect(0, 0, size, hSize);

    // Draw grid
    ctx.strokeStyle = 'rgba(46, 37, 30, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 0; x < size; x += 40) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, hSize); ctx.stroke();
    }
    for (let y = 0; y < hSize; y += 40) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(size, y); ctx.stroke();
    }

    // Draw Discriminator Decision Boundary (Heatmap representation)
    const grd = ctx.createRadialGradient(
      cx + discCenter.x * scale, cy + discCenter.y * scale, Math.max(0, discRadius * scale - 20),
      cx + discCenter.x * scale, cy + discCenter.y * scale, Math.max(0, discRadius * scale + 20)
    );
    grd.addColorStop(0, 'rgba(182, 83, 43, 0.0)'); // transparent inside
    grd.addColorStop(0.5, 'rgba(182, 83, 43, 0.2)'); // Terracotta glow at boundary
    grd.addColorStop(1, 'rgba(182, 83, 43, 0.0)'); // transparent outside
    
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, size, hSize);

    ctx.beginPath();
    ctx.arc(cx + discCenter.x * scale, cy + discCenter.y * scale, discRadius * scale, 0, 2 * Math.PI);
    ctx.strokeStyle = 'rgba(182, 83, 43, 0.5)';
    ctx.setLineDash([5, 5]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw Real Data
    ctx.fillStyle = '#3B7A57'; // Green for real
    ctx.strokeStyle = '#FAF6EE';
    ctx.lineWidth = 1;
    realData.forEach(p => {
      ctx.beginPath();
      ctx.arc(cx + p.x * scale, cy + p.y * scale, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });

    // Draw Fake Data
    ctx.fillStyle = '#C18C3B'; // Ochre for fake
    fakeData.forEach(p => {
      ctx.beginPath();
      ctx.arc(cx + p.x * scale, cy + p.y * scale, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });

  }, [realData, fakeData, discCenter, discRadius]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-6 h-full bg-[#FAF6EE]">
      <div className="lg:col-span-5 bg-[#F4EFE6] border border-[#E5DDD0] p-6 rounded-2xl flex flex-col justify-between shadow-xl">
        <div className="space-y-6">
          <h4 className="text-[#2E251E] font-bold text-xl tracking-tight flex items-center gap-3">
            <Shuffle className="w-6 h-6 text-[#B6532B]" /> GAN Simulator
          </h4>
          <p className="text-[#6E6257] text-sm leading-relaxed">
            Watch the <span className="font-bold text-[#C18C3B]">Generator (Fake)</span> try to mimic the <span className="font-bold text-[#3B7A57]">Real Data</span>, while the <span className="font-bold text-[#B6532B]">Discriminator (Boundary)</span> learns to tell them apart.
          </p>
          
          <div className="space-y-4 bg-[#FAF6EE] p-4 rounded-xl border border-[#E5DDD0]">
            <div>
              <div className="flex justify-between mb-1">
                <label className="text-[10px] text-[#6E6257] font-mono uppercase font-bold">Generator LR</label>
                <span className="text-[10px] font-mono font-bold text-[#C18C3B]">{genLR.toFixed(3)}</span>
              </div>
              <input 
                type="range" min="0.001" max="0.05" step="0.001" value={genLR} 
                onChange={e => setGenLR(parseFloat(e.target.value))}
                className="w-full accent-[#C18C3B]"
              />
            </div>
            <div>
              <div className="flex justify-between mb-1">
                <label className="text-[10px] text-[#6E6257] font-mono uppercase font-bold">Discriminator LR</label>
                <span className="text-[10px] font-mono font-bold text-[#B6532B]">{discLR.toFixed(3)}</span>
              </div>
              <input 
                type="range" min="0.001" max="0.05" step="0.001" value={discLR} 
                onChange={e => setDiscLR(parseFloat(e.target.value))}
                className="w-full accent-[#B6532B]"
              />
            </div>
          </div>

          <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-3 grid grid-cols-2 text-center text-xs font-mono text-[#2E251E]">
            <div className="border-r border-[#E5DDD0]">
              <span className="text-[#6E6257] text-[9px] block">EPOCH</span>
              <span className="font-bold">{epoch}</span>
            </div>
            <div>
              <span className="text-[#6E6257] text-[9px] block">D-RADIUS</span>
              <span className="text-[#B6532B] font-bold">{discRadius.toFixed(2)}</span>
            </div>
          </div>
        </div>

        <div className="flex gap-2 mt-6">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`flex-1 py-3 rounded-xl flex items-center justify-center gap-2 text-sm font-bold transition-all shadow-sm ${
              isPlaying
                ? 'bg-amber-600 border-amber-600 text-white'
                : 'bg-[#B6532B] border-[#B6532B] text-white hover:bg-[#B6532B]/90'
            }`}
          >
            {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
            {isPlaying ? 'Pause' : 'Train GAN'}
          </button>
          <button
            onClick={initGAN}
            className="p-3 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257] hover:text-[#2E251E] transition-colors"
            title="Reset"
          >
            <RotateCcw className="w-5 h-5" />
          </button>
        </div>
      </div>

      <div className="lg:col-span-7 flex flex-col items-center justify-center gap-4">
        <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-2 rounded-2xl shadow-xl overflow-hidden inline-block">
          <canvas ref={canvasRef} className="rounded-xl" />
        </div>
        <div className="flex gap-4 text-[10px] font-mono text-[#6E6257] font-bold uppercase tracking-wider">
          <span className="flex items-center gap-1.5"><div className="w-3 h-3 bg-[#3B7A57] rounded-full border border-white shadow-sm"></div> Real Data</span>
          <span className="flex items-center gap-1.5"><div className="w-3 h-3 bg-[#C18C3B] rounded-full border border-white shadow-sm"></div> Generated (Fake)</span>
          <span className="flex items-center gap-1.5"><div className="w-4 h-0 border-t-2 border-dashed border-[#B6532B]"></div> D-Boundary</span>
        </div>
      </div>
    </div>
  );
};
