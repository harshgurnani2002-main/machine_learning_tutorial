import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useState, useEffect, useRef } from 'react';
import { Target, Trash2 } from 'lucide-react';
export const FeatureScalingSimulator = () => {
    const [points, setPoints] = useState([
        { x: 50, y: 150 },
        { x: 80, y: 120 },
        { x: 100, y: 160 },
        { x: 120, y: 140 },
        { x: 160, y: 180 },
    ]);
    const [scalingMethod, setScalingMethod] = useState('minmax');
    const [showOutlier, setShowOutlier] = useState(false);
    const canvasRef = useRef(null);
    const scaledCanvasRef = useRef(null);
    const activePoints = showOutlier ? [...points, { x: 380, y: 280 }] : points;
    // Calculate statistics
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    let sumX = 0, sumY = 0;
    if (activePoints.length > 0) {
        activePoints.forEach(p => {
            if (p.x < minX)
                minX = p.x;
            if (p.x > maxX)
                maxX = p.x;
            if (p.y < minY)
                minY = p.y;
            if (p.y > maxY)
                maxY = p.y;
            sumX += p.x;
            sumY += p.y;
        });
    }
    const meanX = sumX / activePoints.length;
    const meanY = sumY / activePoints.length;
    let varX = 0, varY = 0;
    activePoints.forEach(p => {
        varX += (p.x - meanX) ** 2;
        varY += (p.y - meanY) ** 2;
    });
    const stdX = Math.sqrt(varX / activePoints.length) || 1; // prevent div by zero
    const stdY = Math.sqrt(varY / activePoints.length) || 1;
    const rangeX = (maxX - minX) || 1;
    const rangeY = (maxY - minY) || 1;
    const handleCanvasClick = (e) => {
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        setPoints(prev => [...prev, { x, y }]);
    };
    useEffect(() => {
        // Draw original points
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const ctx = canvas.getContext('2d');
        if (!ctx)
            return;
        const w = canvas.width = 400;
        const h = canvas.height = 300;
        ctx.fillStyle = '#FAF6EE';
        ctx.fillRect(0, 0, w, h);
        // Draw grid
        ctx.strokeStyle = '#E5DDD0';
        ctx.lineWidth = 0.5;
        for (let x = 0; x < w; x += 40) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }
        for (let y = 0; y < h; y += 40) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }
        // Draw points
        activePoints.forEach((pt, i) => {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2);
            ctx.fillStyle = (showOutlier && i === activePoints.length - 1) ? '#D9534F' : '#B6532B';
            ctx.strokeStyle = '#FAF6EE';
            ctx.lineWidth = 1.5;
            ctx.fill();
            ctx.stroke();
        });
    }, [activePoints, showOutlier]);
    useEffect(() => {
        // Draw scaled points
        const canvas = scaledCanvasRef.current;
        if (!canvas)
            return;
        const ctx = canvas.getContext('2d');
        if (!ctx)
            return;
        const w = canvas.width = 400;
        const h = canvas.height = 300;
        ctx.fillStyle = '#FAF6EE';
        ctx.fillRect(0, 0, w, h);
        // Draw grid axes
        ctx.strokeStyle = '#E5DDD0';
        ctx.lineWidth = 0.5;
        for (let x = 0; x < w; x += 40) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }
        for (let y = 0; y < h; y += 40) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }
        // Draw center axes (0,0)
        ctx.strokeStyle = '#2E251E';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        if (scalingMethod === 'standard') {
            ctx.beginPath();
            ctx.moveTo(w / 2, 0);
            ctx.lineTo(w / 2, h);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(0, h / 2);
            ctx.lineTo(w, h / 2);
            ctx.stroke();
        }
        else {
            // MinMax: origin is bottom-left ideally, let's map [0,1] to [40, 360] 
            ctx.beginPath();
            ctx.moveTo(40, 0);
            ctx.lineTo(40, h);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(0, h - 40);
            ctx.lineTo(w, h - 40);
            ctx.stroke();
        }
        ctx.setLineDash([]);
        // Draw points scaled
        activePoints.forEach((pt, i) => {
            let scaledX = 0, scaledY = 0;
            if (scalingMethod === 'minmax') {
                scaledX = 40 + ((pt.x - minX) / rangeX) * (w - 80);
                scaledY = (h - 40) - ((pt.y - minY) / rangeY) * (h - 80);
            }
            else {
                // Standard: z = (x - mean)/std
                const zX = (pt.x - meanX) / stdX;
                const zY = (pt.y - meanY) / stdY;
                // Map z=[-3, 3] to [0, w]
                scaledX = w / 2 + (zX / 3) * (w / 2);
                scaledY = h / 2 - (zY / 3) * (h / 2); // invert Y for canvas
            }
            ctx.beginPath();
            ctx.arc(scaledX, scaledY, 6, 0, Math.PI * 2);
            ctx.fillStyle = (showOutlier && i === activePoints.length - 1) ? '#D9534F' : '#C18C3B';
            ctx.strokeStyle = '#FAF6EE';
            ctx.lineWidth = 1.5;
            ctx.fill();
            ctx.stroke();
        });
    }, [activePoints, scalingMethod, minX, minY, rangeX, rangeY, meanX, meanY, stdX, stdY, showOutlier]);
    return (_jsxs("div", { className: "grid grid-cols-1 md:grid-cols-12 gap-6 p-4", children: [_jsxs("div", { className: "md:col-span-4 bg-[#F4EFE6] border border-[#E5DDD0] p-5 rounded-2xl flex flex-col justify-between", children: [_jsxs("div", { className: "space-y-4", children: [_jsxs("h4", { className: "text-[#2E251E] font-bold text-base tracking-wide flex items-center gap-2", children: [_jsx(Target, { className: "w-5 h-5 text-[#B6532B]" }), " Feature Scaling"] }), _jsx("p", { className: "text-[#6E6257] text-xs", children: "Add points on the left to see how they are transformed on the right. Toggle the outlier to see how it squashes the rest of the data." }), _jsxs("div", { className: "space-y-2", children: [_jsx("label", { className: "text-xs text-[#2E251E] font-mono block", children: "Scaling Method" }), _jsxs("div", { className: "flex gap-2", children: [_jsx("button", { onClick: () => setScalingMethod('minmax'), className: `flex-1 py-2 rounded-xl text-xs font-mono border ${scalingMethod === 'minmax'
                                                    ? 'border-[#B6532B] bg-[#B6532B]/15 text-[#B6532B] font-bold'
                                                    : 'border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257]'}`, children: "MinMax" }), _jsx("button", { onClick: () => setScalingMethod('standard'), className: `flex-1 py-2 rounded-xl text-xs font-mono border ${scalingMethod === 'standard'
                                                    ? 'border-[#C18C3B] bg-[#C18C3B]/15 text-[#C18C3B] font-bold'
                                                    : 'border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257]'}`, children: "Standard" })] })] }), _jsxs("label", { className: "flex items-center gap-2 text-sm text-[#2E251E] cursor-pointer mt-4", children: [_jsx("input", { type: "checkbox", checked: showOutlier, onChange: (e) => setShowOutlier(e.target.checked), className: "rounded text-[#B6532B] focus:ring-[#B6532B]" }), _jsx("span", { className: "font-mono text-xs", children: "Inject Extreme Outlier" })] }), _jsxs("div", { className: "p-3 bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl space-y-2 text-xs font-mono mt-4", children: [_jsxs("div", { className: "flex justify-between", children: [_jsx("span", { className: "text-[#6E6257]", children: "Data Mean (X, Y):" }), _jsxs("span", { className: "text-[#2E251E] font-bold", children: ["(", meanX.toFixed(1), ", ", meanY.toFixed(1), ")"] })] }), _jsxs("div", { className: "flex justify-between", children: [_jsx("span", { className: "text-[#6E6257]", children: "Data Std (X, Y):" }), _jsxs("span", { className: "text-[#2E251E] font-bold", children: ["(", stdX.toFixed(1), ", ", stdY.toFixed(1), ")"] })] }), _jsxs("div", { className: "flex justify-between", children: [_jsx("span", { className: "text-[#6E6257]", children: "Range (X, Y):" }), _jsxs("span", { className: "text-[#2E251E] font-bold", children: ["(", rangeX.toFixed(1), ", ", rangeY.toFixed(1), ")"] })] })] })] }), _jsxs("button", { onClick: () => { setPoints([]); setShowOutlier(false); }, className: "w-full py-2.5 mt-4 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#FAF6EE]/80 text-[#6E6257] hover:text-[#2E251E] transition-all flex items-center justify-center gap-2 text-xs font-semibold", children: [_jsx(Trash2, { className: "w-4 h-4" }), " Clear Data"] })] }), _jsx("div", { className: "md:col-span-4 flex flex-col items-center", children: _jsxs("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl relative w-full flex justify-center shadow-inner", children: [_jsx("canvas", { ref: canvasRef, onClick: handleCanvasClick, className: "rounded-xl border border-[#E5DDD0] cursor-crosshair w-full max-w-[400px] h-[300px]" }), _jsx("div", { className: "absolute top-7 left-7 bg-[#2E251E]/90 px-2.5 py-1 rounded-lg border border-[#E5DDD0]/20 text-[9px] font-mono text-[#FAF6EE] uppercase tracking-wide", children: "Original Data (Click to Add)" })] }) }), _jsx("div", { className: "md:col-span-4 flex flex-col items-center", children: _jsxs("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl relative w-full flex justify-center shadow-inner", children: [_jsx("canvas", { ref: scaledCanvasRef, className: "rounded-xl border border-[#E5DDD0] w-full max-w-[400px] h-[300px]" }), _jsx("div", { className: "absolute top-7 left-7 bg-[#2E251E]/90 px-2.5 py-1 rounded-lg border border-[#E5DDD0]/20 text-[9px] font-mono text-[#FAF6EE] uppercase tracking-wide", children: scalingMethod === 'minmax' ? 'MinMax Scaled [0, 1]' : 'Standardized (Z-Score)' })] }) })] }));
};
