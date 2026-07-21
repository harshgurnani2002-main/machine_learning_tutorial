import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useState, useEffect, useRef } from 'react';
import { Network, Trash2 } from 'lucide-react';
export const KNNClassifierSimulator = () => {
    const [points, setPoints] = useState([
        { x: 150, y: 150, label: 0 }, { x: 180, y: 180, label: 0 }, { x: 120, y: 200, label: 0 },
        { x: 450, y: 150, label: 1 }, { x: 400, y: 120, label: 1 }, { x: 480, y: 180, label: 1 },
        { x: 300, y: 320, label: 2 }, { x: 350, y: 350, label: 2 }, { x: 250, y: 300, label: 2 }
    ]);
    const [activeLabel, setActiveLabel] = useState(0);
    const [kValue, setKValue] = useState(3);
    const [metric, setMetric] = useState('euclidean');
    const canvasRef = useRef(null);
    const clearPoints = () => setPoints([]);
    const handleCanvasClick = (e) => {
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        setPoints(prev => [...prev, { x, y, label: activeLabel }]);
    };
    const getDistance = (x1, y1, x2, y2) => {
        if (metric === 'euclidean') {
            return Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2);
        }
        else {
            return Math.abs(x1 - x2) + Math.abs(y1 - y2);
        }
    };
    const predictKNN = (px, py) => {
        if (points.length === 0)
            return -1;
        const distances = points.map(pt => ({
            dist: getDistance(px, py, pt.x, pt.y),
            label: pt.label
        }));
        distances.sort((a, b) => a.dist - b.dist);
        const neighbors = distances.slice(0, Math.min(kValue, distances.length));
        const counts = [0, 0, 0];
        neighbors.forEach(n => counts[n.label]++);
        let maxCount = -1;
        let bestLabel = -1;
        counts.forEach((c, i) => {
            if (c > maxCount) {
                maxCount = c;
                bestLabel = i;
            }
        });
        return bestLabel;
    };
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const ctx = canvas.getContext('2d');
        if (!ctx)
            return;
        const w = canvas.width = 600;
        const h = canvas.height = 400;
        // Warm cream gradient background
        const grad = ctx.createLinearGradient(0, 0, 0, h);
        grad.addColorStop(0, '#FAF6EE');
        grad.addColorStop(1, '#F4EFE6');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, w, h);
        if (points.length > 0) {
            // Draw background decision regions (Voronoi-like for K=1, smooth for higher K)
            const step = 8;
            for (let x = 0; x < w; x += step) {
                for (let y = 0; y < h; y += step) {
                    const pred = predictKNN(x + step / 2, y + step / 2);
                    if (pred === 0) {
                        ctx.fillStyle = 'rgba(182, 83, 43, 0.15)'; // Terracotta
                    }
                    else if (pred === 1) {
                        ctx.fillStyle = 'rgba(193, 140, 59, 0.15)'; // Ochre
                    }
                    else if (pred === 2) {
                        ctx.fillStyle = 'rgba(59, 122, 87, 0.15)'; // Green
                    }
                    else {
                        continue;
                    }
                    ctx.fillRect(x, y, step, step);
                }
            }
        }
        // Subtle warm grid
        ctx.strokeStyle = 'rgba(110, 98, 87, 0.1)';
        ctx.lineWidth = 1;
        for (let x = 0; x < w; x += 50) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }
        for (let y = 0; y < h; y += 50) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }
        // Draw points
        points.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 8, 0, Math.PI * 2);
            ctx.fillStyle = pt.label === 0 ? '#B6532B' : pt.label === 1 ? '#C18C3B' : '#3B7A57';
            ctx.shadowBlur = 10;
            ctx.shadowColor = ctx.fillStyle;
            ctx.fill();
            ctx.shadowBlur = 0;
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 8, 0, Math.PI * 2);
            ctx.strokeStyle = '#FAF6EE';
            ctx.lineWidth = 2;
            ctx.stroke();
        });
    }, [points, kValue, metric]);
    return (_jsxs("div", { className: "grid grid-cols-1 md:grid-cols-12 gap-8 p-6", children: [_jsxs("div", { className: "md:col-span-4 bg-[#F4EFE6] border border-[#E5DDD0] p-6 rounded-2xl flex flex-col justify-between", children: [_jsxs("div", { className: "space-y-6", children: [_jsxs("h4", { className: "text-[#2E251E] font-bold text-xl tracking-tight flex items-center gap-3", children: [_jsx(Network, { className: "w-6 h-6 text-[#B6532B]" }), " K-Nearest Neighbors"] }), _jsxs("p", { className: "text-[#6E6257] text-sm leading-relaxed", children: ["KNN classifies a point based on the majority vote of its ", _jsx("span", { className: "font-semibold text-[#B6532B]", children: "K" }), " closest training examples. K=1 creates sharp boundaries, while higher K creates smoother regions."] }), _jsxs("div", { className: "space-y-3", children: [_jsx("label", { className: "text-xs text-[#6E6257] font-bold uppercase tracking-wider block font-mono", children: "Class Selector" }), _jsxs("div", { className: "flex bg-[#FAF6EE] border border-[#E5DDD0] p-1 rounded-xl gap-1", children: [_jsx("button", { onClick: () => setActiveLabel(0), className: `flex-1 py-2 rounded-lg text-xs font-bold transition-all ${activeLabel === 0 ? 'bg-[#B6532B] text-white shadow-md' : 'text-[#6E6257] hover:bg-[#F4EFE6]'}`, children: "Class 0" }), _jsx("button", { onClick: () => setActiveLabel(1), className: `flex-1 py-2 rounded-lg text-xs font-bold transition-all ${activeLabel === 1 ? 'bg-[#C18C3B] text-white shadow-md' : 'text-[#6E6257] hover:bg-[#F4EFE6]'}`, children: "Class 1" }), _jsx("button", { onClick: () => setActiveLabel(2), className: `flex-1 py-2 rounded-lg text-xs font-bold transition-all ${activeLabel === 2 ? 'bg-[#3B7A57] text-white shadow-md' : 'text-[#6E6257] hover:bg-[#F4EFE6]'}`, children: "Class 2" })] })] }), _jsxs("div", { className: "space-y-3", children: [_jsx("label", { className: "text-xs text-[#6E6257] font-bold uppercase tracking-wider block font-mono", children: "Distance Metric" }), _jsxs("div", { className: "flex bg-[#FAF6EE] border border-[#E5DDD0] p-1 rounded-xl gap-1", children: [_jsx("button", { onClick: () => setMetric('euclidean'), className: `flex-1 py-2 rounded-lg text-xs font-bold transition-all ${metric === 'euclidean' ? 'bg-[#B6532B] text-white shadow-md' : 'text-[#6E6257] hover:bg-[#F4EFE6]'}`, children: "Euclidean (L2)" }), _jsx("button", { onClick: () => setMetric('manhattan'), className: `flex-1 py-2 rounded-lg text-xs font-bold transition-all ${metric === 'manhattan' ? 'bg-[#B6532B] text-white shadow-md' : 'text-[#6E6257] hover:bg-[#F4EFE6]'}`, children: "Manhattan (L1)" })] })] }), _jsxs("div", { className: "space-y-3", children: [_jsxs("div", { className: "flex justify-between items-center text-sm font-bold", children: [_jsx("span", { className: "text-[#6E6257] font-mono text-xs", children: "Neighbors (K):" }), _jsx("span", { className: "text-[#B6532B] bg-[#B6532B]/10 px-3 py-1 rounded-full text-xs font-mono font-bold border border-[#B6532B]/20", children: kValue })] }), _jsx("input", { type: "range", min: "1", max: "15", step: "2", value: kValue, onChange: (e) => setKValue(parseInt(e.target.value)), className: "w-full accent-[#B6532B]" }), _jsx("p", { className: "text-[10px] text-[#6E6257] font-mono mt-1", children: "Odd values of K avoid ties during voting." })] })] }), _jsxs("button", { onClick: clearPoints, className: "w-full py-3 mt-6 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257] hover:text-[#2E251E] hover:bg-[#F4EFE6] text-sm font-bold transition-colors flex justify-center items-center gap-2", children: [_jsx(Trash2, { className: "w-5 h-5" }), " Clear Data"] })] }), _jsx("div", { className: "md:col-span-8 flex flex-col items-center justify-center", children: _jsxs("div", { className: "bg-[#F4EFE6] border border-[#E5DDD0] p-2 rounded-2xl w-full flex justify-center shadow-md relative overflow-hidden group", children: [_jsx("canvas", { ref: canvasRef, onClick: handleCanvasClick, className: "rounded-xl cursor-crosshair w-full aspect-[3/2]" }), _jsx("div", { className: "absolute top-4 left-4 bg-[#2E251E]/80 backdrop-blur-sm px-3 py-1.5 rounded-lg border border-[#E5DDD0]/30 text-[9px] font-mono text-[#FAF6EE] pointer-events-none", children: "Click to add points \u00B7 K-decision regions shown" })] }) })] }));
};
