import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useState, useEffect, useRef } from 'react';
import { BoxSelect, Activity } from 'lucide-react';
export const VAESimulator = () => {
    const canvasRef = useRef(null);
    // Latent space coordinate [-1, 1]
    const [latentZ, setLatentZ] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    // VAE Parameters
    const [beta, setBeta] = useState(1.0); // KL Divergence weight
    // Generate a mock output image (8x8) based on latent Z and beta
    const generateImage = (z, b) => {
        const img = [];
        for (let y = 0; y < 8; y++) {
            const row = [];
            for (let x = 0; x < 8; x++) {
                // Map 2D latent to 8x8 pattern. 
                // We'll create a pattern that shifts based on z.x and z.y, and blurs based on beta.
                const px = (x - 3.5) / 3.5;
                const py = (y - 3.5) / 3.5;
                // Distance to the "latent center"
                const dist = Math.sqrt((px - z.x) ** 2 + (py - z.y) ** 2);
                // Beta adds regularization, pulling things to the center, meaning less extreme features
                const sharpness = Math.max(0.5, 3 - b * 0.5);
                let val = Math.max(0, 1 - dist * sharpness);
                // Add some noise or secondary features to make it look like a digit/feature
                if (z.x * px > 0 && z.y * py > 0)
                    val += 0.2;
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
        if (!canvas)
            return;
        const ctx = canvas.getContext('2d');
        if (!ctx)
            return;
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
                const dist = Math.sqrt(dx * dx + dy * dy);
                // Gaussian probability density
                const p = Math.exp(-0.5 * (dist / 0.4) ** 2);
                const idx = (y * size + x) * 4;
                imgData.data[idx] = 182; // R
                imgData.data[idx + 1] = 83; // G
                imgData.data[idx + 2] = 43; // B
                imgData.data[idx + 3] = Math.floor(p * 50); // Alpha
            }
        }
        ctx.putImageData(imgData, 0, 0);
        // Axes
        ctx.strokeStyle = 'rgba(46, 37, 30, 0.1)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(cx, 0);
        ctx.lineTo(cx, hSize);
        ctx.moveTo(0, cy);
        ctx.lineTo(size, cy);
        ctx.stroke();
        // Prior unit circle (sigma=1)
        ctx.beginPath();
        ctx.arc(cx, cy, size / 2 * 0.4, 0, 2 * Math.PI);
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
        ctx.moveTo(zx, zy);
        ctx.lineTo(zx, cy);
        ctx.moveTo(zx, zy);
        ctx.lineTo(cx, zy);
        ctx.stroke();
        ctx.setLineDash([]);
    }, [latentZ, beta]);
    const handlePointerDown = (e) => {
        setIsDragging(true);
        updateZ(e);
        e.target.setPointerCapture(e.pointerId);
    };
    const handlePointerMove = (e) => {
        if (isDragging)
            updateZ(e);
    };
    const handlePointerUp = (e) => {
        setIsDragging(false);
        e.target.releasePointerCapture(e.pointerId);
    };
    const updateZ = (e) => {
        const canvas = canvasRef.current;
        if (!canvas)
            return;
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
    return (_jsxs("div", { className: "grid grid-cols-1 lg:grid-cols-12 gap-6 p-6 h-full bg-[#FAF6EE]", children: [_jsx("div", { className: "lg:col-span-5 bg-[#F4EFE6] border border-[#E5DDD0] p-6 rounded-2xl flex flex-col justify-between shadow-xl", children: _jsxs("div", { className: "space-y-6", children: [_jsxs("h4", { className: "text-[#2E251E] font-bold text-xl tracking-tight flex items-center gap-3", children: [_jsx(Activity, { className: "w-6 h-6 text-[#B6532B]" }), " Variational Autoencoder"] }), _jsxs("p", { className: "text-[#6E6257] text-sm leading-relaxed", children: ["Drag the point $z$ in the ", _jsx("span", { className: "font-semibold text-[#B6532B]", children: "Latent Space" }), ". The VAE reconstructs an image based on these continuous coordinates. The $\\beta$ parameter controls how closely the latent space matches a standard Gaussian distribution."] }), _jsx("div", { className: "space-y-4 bg-[#FAF6EE] p-4 rounded-xl border border-[#E5DDD0]", children: _jsxs("div", { children: [_jsxs("div", { className: "flex justify-between mb-1", children: [_jsx("label", { className: "text-[10px] text-[#6E6257] font-mono uppercase font-bold", children: "Beta ($\\beta$) - KL Weight" }), _jsx("span", { className: "text-[10px] font-mono font-bold text-[#B6532B]", children: beta.toFixed(2) })] }), _jsx("input", { type: "range", min: "0", max: "5", step: "0.1", value: beta, onChange: e => setBeta(parseFloat(e.target.value)), className: "w-full accent-[#B6532B]" }), _jsx("p", { className: "text-[9px] text-[#6E6257] mt-2 leading-tight", children: "Higher $\\beta$ forces the latent space to be a dense Gaussian centered at (0,0), but may blur reconstructions." })] }) }), _jsxs("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-4 flex gap-4 items-center", children: [_jsxs("div", { className: "flex-1 text-center", children: [_jsx("span", { className: "text-[10px] text-[#6E6257] font-mono font-bold uppercase block mb-1", children: "Z1 (Mean)" }), _jsx("span", { className: "font-mono font-bold text-[#2E251E] bg-[#F4EFE6] px-2 py-1 rounded", children: latentZ.x.toFixed(2) })] }), _jsxs("div", { className: "flex-1 text-center", children: [_jsx("span", { className: "text-[10px] text-[#6E6257] font-mono font-bold uppercase block mb-1", children: "Z2 (Mean)" }), _jsx("span", { className: "font-mono font-bold text-[#2E251E] bg-[#F4EFE6] px-2 py-1 rounded", children: latentZ.y.toFixed(2) })] })] })] }) }), _jsxs("div", { className: "lg:col-span-7 flex flex-col md:flex-row items-center justify-center gap-8", children: [_jsxs("div", { className: "flex flex-col items-center", children: [_jsx("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] p-2 rounded-2xl shadow-xl overflow-hidden mb-2 cursor-crosshair", children: _jsx("canvas", { ref: canvasRef, className: "rounded-xl touch-none", onPointerDown: handlePointerDown, onPointerMove: handlePointerMove, onPointerUp: handlePointerUp, onPointerCancel: handlePointerUp }) }), _jsx("span", { className: "text-[10px] font-mono text-[#6E6257] font-bold uppercase tracking-wider", children: "2D Continuous Latent Space (Z)" })] }), _jsxs("div", { className: "flex flex-col items-center", children: [_jsxs("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] p-3 rounded-2xl shadow-inner mb-2 w-32 h-32 md:w-48 md:h-48 grid grid-cols-8 grid-rows-8 gap-0.5 relative", children: [outputImg.map((row, y) => (row.map((val, x) => (_jsx("div", { className: "w-full h-full rounded-sm", style: { backgroundColor: `rgba(46, 37, 30, ${val})` } }, `${x}-${y}`))))), _jsx("div", { className: "absolute inset-0 border border-[#E5DDD0] rounded-lg pointer-events-none" })] }), _jsxs("span", { className: "text-[10px] font-mono text-[#6E6257] font-bold uppercase tracking-wider flex items-center gap-1.5", children: [_jsx(BoxSelect, { className: "w-3.5 h-3.5 text-[#B6532B]" }), " Reconstructed Output"] })] })] })] }));
};
