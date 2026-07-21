import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useState, useEffect, useRef } from 'react';
import { Activity, Play, Pause, Trash2 } from 'lucide-react';
const drawLinearBoundary = (ctx, w0, w1, b, width, height, color = '#2E251E', lineWidth = 2.5) => {
    const A = w0 / width;
    const B = w1 / height;
    const C = b;
    const points = [];
    if (Math.abs(B) > 1e-9) {
        const y = -C / B;
        if (y >= 0 && y <= height)
            points.push({ x: 0, y });
    }
    if (Math.abs(B) > 1e-9) {
        const y = -(A * width + C) / B;
        if (y >= 0 && y <= height)
            points.push({ x: width, y });
    }
    if (Math.abs(A) > 1e-9) {
        const x = -C / A;
        if (x >= 0 && x <= width)
            points.push({ x, y: 0 });
    }
    if (Math.abs(A) > 1e-9) {
        const x = -(B * height + C) / A;
        if (x >= 0 && x <= width)
            points.push({ x, y: height });
    }
    const uniquePoints = [];
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
export const LogisticRegressionSimulator = () => {
    const [points, setPoints] = useState([
        { x: 150, y: 300, label: 0 },
        { x: 200, y: 250, label: 0 },
        { x: 180, y: 150, label: 0 },
        { x: 400, y: 200, label: 1 },
        { x: 450, y: 120, label: 1 },
        { x: 480, y: 280, label: 1 },
    ]);
    const [activeLabel, setActiveLabel] = useState(0);
    const [w, setW] = useState([-1.2, 0.6]);
    const [b, setB] = useState(0.2);
    const [epochs, setEpochs] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const canvasRef = useRef(null);
    const clearPoints = () => {
        setPoints([]);
        setEpochs(0);
    };
    const handleCanvasClick = (e) => {
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        setPoints(prev => [...prev, { x, y, label: activeLabel }]);
    };
    const trainStep = () => {
        if (points.length === 0)
            return;
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
        if (!canvas)
            return;
        const ctx = canvas.getContext('2d');
        if (!ctx)
            return;
        const width = canvas.width = 600;
        const height = canvas.height = 400;
        // Draw background gradients (Sigmoid Field)
        const step = 6;
        for (let x = 0; x < width; x += step) {
            for (let y = 0; y < height; y += step) {
                const nx = (x + step / 2) / width;
                const ny = (y + step / 2) / height;
                const z = w[0] * nx + w[1] * ny + b;
                const prob = 1 / (1 + Math.exp(-z));
                // Blend: 0 = Terracotta, 1 = Ochre
                const r = Math.floor(182 * (1 - prob) + 193 * prob);
                const g = Math.floor(83 * (1 - prob) + 140 * prob);
                const bCol = Math.floor(43 * (1 - prob) + 59 * prob);
                ctx.fillStyle = `rgba(${r}, ${g}, ${bCol}, 0.25)`;
                ctx.fillRect(x, y, step, step);
            }
        }
        // Subtle warm grid
        ctx.strokeStyle = 'rgba(110, 98, 87, 0.1)';
        ctx.lineWidth = 1;
        for (let x = 0; x < width; x += 50) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        for (let y = 0; y < height; y += 50) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
        // Draw Decision Line (prob = 0.5)
        ctx.shadowBlur = 10;
        ctx.shadowColor = '#FAF6EE';
        drawLinearBoundary(ctx, w[0], w[1], b, width, height, '#FAF6EE', 3);
        ctx.shadowBlur = 0;
        // Draw coordinate points
        points.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 10, 0, Math.PI * 2);
            ctx.fillStyle = pt.label === 0 ? '#B6532B' : '#C18C3B';
            ctx.shadowBlur = 10;
            ctx.shadowColor = pt.label === 0 ? '#B6532B' : '#C18C3B';
            ctx.fill();
            ctx.shadowBlur = 0;
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 10, 0, Math.PI * 2);
            ctx.strokeStyle = '#FAF6EE';
            ctx.lineWidth = 2.5;
            ctx.stroke();
        });
    }, [points, w, b]);
    return (_jsxs("div", { className: "grid grid-cols-1 md:grid-cols-12 gap-8 p-6", children: [_jsxs("div", { className: "md:col-span-4 bg-[#FAF6EE] border border-[#E5DDD0] p-6 rounded-3xl flex flex-col justify-between shadow-xl", children: [_jsxs("div", { className: "space-y-6", children: [_jsxs("h4", { className: "text-[#2E251E] font-bold text-xl tracking-tight flex items-center gap-3", children: [_jsx(Activity, { className: "w-6 h-6 text-[#B6532B]" }), " Logistic Boundaries"] }), _jsxs("p", { className: "text-[#6E6257] text-sm leading-relaxed", children: ["Click on the canvas to place points of either class. Run the optimizer to see the ", _jsx("span", { className: "font-semibold text-[#B6532B]", children: "Sigmoid probability field" }), " shift and establish the best decision boundary using Gradient Descent."] }), _jsxs("div", { className: "space-y-3", children: [_jsx("label", { className: "text-xs text-[#6E6257] font-bold uppercase tracking-wider block", children: "Class Selector" }), _jsxs("div", { className: "flex bg-[#F4EFE6] p-1 rounded-xl gap-1", children: [_jsxs("button", { onClick: () => setActiveLabel(0), className: `flex-1 py-3 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 ${activeLabel === 0
                                                    ? 'bg-[#B6532B] text-white shadow-md'
                                                    : 'text-[#6E6257] hover:bg-[#E5DDD0]'}`, children: [_jsx("span", { className: `w-3 h-3 rounded-full ${activeLabel === 0 ? 'bg-white' : 'bg-[#B6532B]'}` }), " Class 0"] }), _jsxs("button", { onClick: () => setActiveLabel(1), className: `flex-1 py-3 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 ${activeLabel === 1
                                                    ? 'bg-[#C18C3B] text-white shadow-md'
                                                    : 'text-[#6E6257] hover:bg-[#E5DDD0]'}`, children: [_jsx("span", { className: `w-3 h-3 rounded-full ${activeLabel === 1 ? 'bg-white' : 'bg-[#C18C3B]'}` }), " Class 1"] })] })] }), _jsxs("div", { className: "p-5 bg-[#FAF6EE] border border-[#E5DDD0] rounded-2xl space-y-4 shadow-inner", children: [_jsxs("div", { className: "flex justify-between items-center", children: [_jsx("span", { className: "text-[#6E6257] text-sm font-medium", children: "Epochs" }), _jsx("span", { className: "text-[#B6532B] font-mono font-bold", children: epochs })] }), _jsxs("div", { className: "flex justify-between items-center", children: [_jsx("span", { className: "text-[#6E6257] text-sm font-medium", children: "BCE Loss" }), _jsx("span", { className: "text-[#3B7A57] font-mono font-bold text-lg", children: loss.toFixed(4) })] }), _jsxs("div", { className: "pt-4 mt-2 border-t border-[#E5DDD0]", children: [_jsx("span", { className: "text-xs text-[#6E6257] font-mono", children: "Weights:" }), _jsxs("div", { className: "text-sm text-[#2E251E] font-mono mt-1", children: ["w\u2081=", w[0].toFixed(2), ", w\u2082=", w[1].toFixed(2), ", b=", b.toFixed(2)] })] })] })] }), _jsxs("div", { className: "pt-6 border-t border-[#E5DDD0] space-y-3 mt-4", children: [_jsxs("div", { className: "grid grid-cols-2 gap-3", children: [_jsxs("button", { onClick: () => setIsPlaying(!isPlaying), className: "py-3 rounded-xl bg-[#B6532B] text-white hover:bg-[#9F4825] font-bold text-sm transition-all shadow-md flex items-center justify-center gap-2", children: [isPlaying ? _jsx(Pause, { className: "w-4 h-4" }) : _jsx(Play, { className: "w-4 h-4" }), isPlaying ? 'Pause' : 'Train Auto'] }), _jsx("button", { onClick: trainStep, className: "py-3 rounded-xl border-2 border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#F4EFE6] text-[#2E251E] font-bold text-sm shadow-sm", children: "Step Update" })] }), _jsxs("button", { onClick: clearPoints, className: "w-full py-3 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#F4EFE6] text-[#6E6257] hover:text-[#B6532B] text-sm font-bold transition-colors flex justify-center items-center gap-2 shadow-sm", children: [_jsx(Trash2, { className: "w-4 h-4" }), " Clear Canvas"] })] })] }), _jsx("div", { className: "md:col-span-8 flex flex-col items-center justify-center", children: _jsxs("div", { className: "bg-[#F4EFE6] border border-[#E5DDD0] p-2 rounded-3xl w-full flex justify-center shadow-md relative overflow-hidden group", children: [_jsx("canvas", { ref: canvasRef, onClick: handleCanvasClick, className: "rounded-2xl cursor-crosshair w-full aspect-[3/2]" }), _jsxs("div", { className: "absolute top-6 left-6 bg-[#2E251E]/80 backdrop-blur-md px-4 py-2 rounded-xl border border-[#E5DDD0]/30 text-xs font-mono text-[#FAF6EE] shadow-lg pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity", children: ["Click to plot ", activeLabel === 0 ? 'Class 0' : 'Class 1'] })] }) })] }));
};
