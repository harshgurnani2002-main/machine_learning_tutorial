import React, { useState, useEffect, useRef } from 'react';
import { BoxSelect, Activity } from 'lucide-react';

export const VAESimulator: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Latent space coordinate [-1, 1]
  const [latentZ, setLatentZ] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  
  // VAE Parameters
  const [beta, setBeta] = useState(1.0); // KL Divergence weight

  // Generate a mock output image (8x8) based on latent Z and beta
  const generateImage = (z: {x: number, y: number}, b: number) => {
    const img = [];
    for (let y = 0; y < 8; y++) {
      const row = [];
      for (let x = 0; x < 8; x++) {
        // Map 2D latent to 8x8 pattern. 
        // We'll create a pattern that shifts based on z.x and z.y, and blurs based on beta.
        const px = (x - 3.5) / 3.5;
        const py = (y - 3.5) / 3.5;
        
        // Distance to the "latent center"
        const dist = Math.sqrt((px - z.x)**2 + (py - z.y)**2);
        
        // Beta adds regularization, pulling things to the center, meaning less extreme features
        const sharpness = Math.max(0.5, 3 - b * 0.5);
        let val = Math.max(0, 1 - dist * sharpness);
        
        // Add some noise or secondary features to make it look like a digit/feature
        if (z.x * px > 0 && z.y * py > 0) val += 0.2;
        
        row.push(Math.min(1, val));
      }
      img.push(row);
    }
    return img;
  };

  const [outputImg, setOutputImg] = useState(generateImage(latentZ, beta));

  useEffect(() => {
    setOutputImg(generateImage(latentZ, beta));
  }, [latentZ, beta]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const size = canvas.width = 300;
    const hSize = canvas.height = 300;
    const cx = size / 2;
    const cy = hSize / 2;

    // Draw Background
    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, size);
    grad.addColorStop(0, '#FAF6EE');
    grad.addColorStop(1, '#F4EFE6');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, size, hSize);

    // Draw Gaussian prior distribution heatmap
    const imgData = ctx.createImageData(size, hSize);
    for (let y = 0; y < hSize; y++) {
      for (let x = 0; x < size; x++) {
        const dx = (x - cx) / (size / 2);
        const dy = (y - cy) / (hSize / 2);
        const dist = Math.sqrt(dx*dx + dy*dy);
        // Gaussian probability density
        const p = Math.exp(-0.5 * (dist / 0.4)**2);
        
        const idx = (y * size + x) * 4;
        imgData.data[idx] = 182; // R
        imgData.data[idx+1] = 83; // G
        imgData.data[idx+2] = 43; // B
        imgData.data[idx+3] = Math.floor(p * 50); // Alpha
      }
    }
    ctx.putImageData(imgData, 0, 0);

    // Axes
    ctx.strokeStyle = 'rgba(46, 37, 30, 0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, 0); ctx.lineTo(cx, hSize);
    ctx.moveTo(0, cy); ctx.lineTo(size, cy);
    ctx.stroke();

    // Prior unit circle (sigma=1)
    ctx.beginPath();
    ctx.arc(cx, cy, size/2 * 0.4, 0, 2*Math.PI);
    ctx.strokeStyle = 'rgba(46, 37, 30, 0.2)';
    ctx.setLineDash([5, 5]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw Z point
    const zx = cx + latentZ.x * (size / 2);
    const zy = cy + latentZ.y * (size / 2);

    ctx.beginPath();
    ctx.arc(zx, zy, 8, 0, 2 * Math.PI);
    ctx.fillStyle = '#B6532B';
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#FAF6EE';
    ctx.stroke();

    // Draw lines to axes
    ctx.strokeStyle = 'rgba(182, 83, 43, 0.5)';
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(zx, zy); ctx.lineTo(zx, cy);
    ctx.moveTo(zx, zy); ctx.lineTo(cx, zy);
    ctx.stroke();
    ctx.setLineDash([]);

  }, [latentZ, beta]);

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
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    
    // Normalize to [-1, 1]
    const nzX = Math.max(-1, Math.min(1, (x / (canvas.width / 2)) - 1));
    const nzY = Math.max(-1, Math.min(1, (y / (canvas.height / 2)) - 1));
    
    setLatentZ({ x: nzX, y: nzY });
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-6 h-full bg-[#FAF6EE]">
      <div className="lg:col-span-5 bg-[#F4EFE6] border border-[#E5DDD0] p-6 rounded-2xl flex flex-col justify-between shadow-xl">
        <div className="space-y-6">
          <h4 className="text-[#2E251E] font-bold text-xl tracking-tight flex items-center gap-3">
            <Activity className="w-6 h-6 text-[#B6532B]" /> Variational Autoencoder
          </h4>
          <p className="text-[#6E6257] text-sm leading-relaxed">
            Drag the point $z$ in the <span className="font-semibold text-[#B6532B]">Latent Space</span>. The VAE reconstructs an image based on these continuous coordinates. The $\beta$ parameter controls how closely the latent space matches a standard Gaussian distribution.
          </p>
          
          <div className="space-y-4 bg-[#FAF6EE] p-4 rounded-xl border border-[#E5DDD0]">
            <div>
              <div className="flex justify-between mb-1">
                <label className="text-[10px] text-[#6E6257] font-mono uppercase font-bold">Beta ($\beta$) - KL Weight</label>
                <span className="text-[10px] font-mono font-bold text-[#B6532B]">{beta.toFixed(2)}</span>
              </div>
              <input 
                type="range" min="0" max="5" step="0.1" value={beta} 
                onChange={e => setBeta(parseFloat(e.target.value))}
                className="w-full accent-[#B6532B]"
              />
              <p className="text-[9px] text-[#6E6257] mt-2 leading-tight">
                Higher $\beta$ forces the latent space to be a dense Gaussian centered at (0,0), but may blur reconstructions.
              </p>
            </div>
          </div>

          <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-4 flex gap-4 items-center">
            <div className="flex-1 text-center">
              <span className="text-[10px] text-[#6E6257] font-mono font-bold uppercase block mb-1">Z1 (Mean)</span>
              <span className="font-mono font-bold text-[#2E251E] bg-[#F4EFE6] px-2 py-1 rounded">{latentZ.x.toFixed(2)}</span>
            </div>
            <div className="flex-1 text-center">
              <span className="text-[10px] text-[#6E6257] font-mono font-bold uppercase block mb-1">Z2 (Mean)</span>
              <span className="font-mono font-bold text-[#2E251E] bg-[#F4EFE6] px-2 py-1 rounded">{latentZ.y.toFixed(2)}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="lg:col-span-7 flex flex-col md:flex-row items-center justify-center gap-8">
        {/* Latent Space Canvas */}
        <div className="flex flex-col items-center">
          <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-2 rounded-2xl shadow-xl overflow-hidden mb-2 cursor-crosshair">
            <canvas 
              ref={canvasRef} 
              className="rounded-xl touch-none"
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerCancel={handlePointerUp}
            />
          </div>
          <span className="text-[10px] font-mono text-[#6E6257] font-bold uppercase tracking-wider">
            2D Continuous Latent Space (Z)
          </span>
        </div>

        {/* Decoder Output */}
        <div className="flex flex-col items-center">
          <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-3 rounded-2xl shadow-inner mb-2 w-32 h-32 md:w-48 md:h-48 grid grid-cols-8 grid-rows-8 gap-0.5 relative">
            {outputImg.map((row, y) => (
              row.map((val, x) => (
                <div 
                  key={`${x}-${y}`} 
                  className="w-full h-full rounded-sm"
                  style={{ backgroundColor: `rgba(46, 37, 30, ${val})` }}
                />
              ))
            ))}
            {/* Overlay grid lines */}
            <div className="absolute inset-0 border border-[#E5DDD0] rounded-lg pointer-events-none" />
          </div>
          <span className="text-[10px] font-mono text-[#6E6257] font-bold uppercase tracking-wider flex items-center gap-1.5">
            <BoxSelect className="w-3.5 h-3.5 text-[#B6532B]" /> Reconstructed Output
          </span>
        </div>
      </div>
    </div>
  );
};
