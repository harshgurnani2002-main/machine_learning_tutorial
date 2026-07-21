import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import React, { useState, useEffect, useRef } from 'react';
import { Target, Trash2 } from 'lucide-react';
export const SVMSimulator = () => {
    const [points, setPoints] = useState([
        { x: 150, y: 300, label: 0 }, { x: 200, y: 250, label: 0 }, { x: 180, y: 150, label: 0 },
        { x: 400, y: 200, label: 1 }, { x: 450, y: 120, label: 1 }, { x: 480, y: 280, label: 1 }
    ]);
    const [activeLabel, setActiveLabel] = useState(0);
    const [kernel, setKernel] = useState('linear');
    const [cParam, setCParam] = useState(1.0); // Penalty parameter
    const [gamma, setGamma] = useState(0.01); // RBF spread
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
    // Simplistic dummy SVM visualizer for educational purposes
    // In a real app, you'd run SMO or import a library like svm.js
    const computeDecisionValue = (px, py) => {
        if (points.length < 2)
            return 0;
        // Normalize coordinates
        const nx = px / 600;
        const ny = py / 400;
        let val = 0;
        // Very rough heuristic to simulate SVM boundaries
        if (kernel === 'linear') {
            // Approximate linear boundary based on centroids
            let cx0 = 0, cy0 = 0, n0 = 0;
            let cx1 = 0, cy1 = 0, n1 = 0;
            points.forEach(p => {
                if (p.label === 0) {
                    cx0 += p.x / 600;
                    cy0 += p.y / 400;
                    n0++;
                }
                else {
                    cx1 += p.x / 600;
                    cy1 += p.y / 400;
                    n1++;
                }
            });
            if (n0 > 0 && n1 > 0) {
                cx0 /= n0;
                cy0 /= n0;
                cx1 /= n1;
                cy1 /= n1;
                const dx = cx1 - cx0;
                const dy = cy1 - cy0;
                const midX = (cx0 + cx1) / 2;
                const midY = (cy0 + cy1) / 2;
                // Projection onto the vector between centroids
                val = (nx - midX) * dx + (ny - midY) * dy;
                val *= 10; // Scaling factor
            }
        }
        else {
            // RBF kernel approximation
            points.forEach(p => {
                const pnx = p.x / 600;
                const pny = p.y / 400;
                const distSq = (nx - pnx) ** 2 + (ny - pny) ** 2;
                const sign = p.label === 1 ? 1 : -1;
                // gamma controls the spread
                val += sign * Math.exp(-gamma * 1000 * distSq);
            });
        }
        return val;
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
        if (points.length >= 2) {
            // Draw background heat map
            const step = 8;
            for (let x = 0; x < w; x += step) {
                for (let y = 0; y < h; y += step) {
                    const val = computeDecisionValue(x + step / 2, y + step / 2);
                    // val = 0 is boundary. >0 is class 1 (blue), <0 is class 0 (red)
                    const prob = 1 / (1 + Math.exp(-val * 3)); // Squish to [0,1] for colors
                    const r = Math.floor(182 * (1 - prob) + 193 * prob);
                    const g = Math.floor(83 * (1 - prob) + 140 * prob);
                    const bCol = Math.floor(43 * (1 - prob) + 59 * prob);
                    // Highlight margin (where val is close to -1, 0, or 1)
                    const marginIntensity = Math.max(0, 1 - Math.abs(val));
                    const alpha = 0.2 + (marginIntensity * 0.3 * (1 / cParam)); // Hard margin = less alpha spread
                    ctx.fillStyle = `rgba(${r}, ${g}, ${bCol}, ${alpha})`;
                    ctx.fillRect(x, y, step, step);
                    // Draw Margin lines (approximate)
                    if (Math.abs(val) < 0.05) {
                        ctx.fillStyle = 'rgba(110, 98, 87, 0.4)';
                        ctx.fillRect(x, y, step, step);
                    }
                    else if (Math.abs(val - 1) < 0.05 || Math.abs(val + 1) < 0.05) {
                        ctx.fillStyle = 'rgba(110, 98, 87, 0.2)';
                        ctx.fillRect(x, y, step, step);
                    }
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
            // Check if it's a "support vector" (inside the margin roughly)
            const val = computeDecisionValue(pt.x, pt.y);
            const isSupportVector = Math.abs(val) < 1.2;
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, isSupportVector ? 12 : 8, 0, Math.PI * 2);
            ctx.fillStyle = pt.label === 0 ? '#B6532B' : '#C18C3B';
            ctx.shadowBlur = 10;
            ctx.shadowColor = pt.label === 0 ? '#B6532B' : '#C18C3B';
            ctx.fill();
            ctx.shadowBlur = 0;
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, isSupportVector ? 12 : 8, 0, Math.PI * 2);
            ctx.strokeStyle = isSupportVector ? '#2E251E' : '#FAF6EE'; // Highlight SVs
            ctx.lineWidth = isSupportVector ? 3 : 2;
            ctx.stroke();
        });
    }, [points, kernel, cParam, gamma]);
    return (_jsxs("div", { className: "grid grid-cols-1 md:grid-cols-12 gap-8 p-6", children: [_jsxs("div", { className: "md:col-span-4 bg-[#FAF6EE] border border-[#E5DDD0] p-6 rounded-3xl flex flex-col justify-between shadow-xl", children: [_jsxs("div", { className: "space-y-6", children: [_jsxs("h4", { className: "text-[#2E251E] font-bold text-xl tracking-tight flex items-center gap-3", children: [_jsx(Target, { className: "w-6 h-6 text-[#C18C3B]" }), " Support Vector Machine"] }), _jsxs("p", { className: "text-[#6E6257] text-sm leading-relaxed", children: ["SVM finds the hyperplane that maximizes the margin between classes. Points with dark rings are ", _jsx("span", { className: "font-semibold text-[#C18C3B]", children: "Support Vectors" }), " that dictate the boundary."] }), _jsxs("div", { className: "space-y-3", children: [_jsx("label", { className: "text-xs text-[#6E6257] font-bold uppercase tracking-wider block font-mono", children: "Class Selector" }), _jsxs("div", { className: "flex bg-[#FAF6EE] border border-[#E5DDD0] p-1 rounded-xl gap-1", children: [_jsxs("button", { onClick: () => setActiveLabel(0), className: `flex-1 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 ${activeLabel === 0 ? 'bg-[#B6532B] text-white shadow-md' : 'text-[#6E6257] hover:bg-[#F4EFE6]'}`, children: [_jsx("span", { className: `w-3 h-3 rounded-full ${activeLabel === 0 ? 'bg-white' : 'bg-[#B6532B]'}` }), " Class 0"] }), _jsxs("button", { onClick: () => setActiveLabel(1), className: `flex-1 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 ${activeLabel === 1 ? 'bg-[#C18C3B] text-white shadow-md' : 'text-[#6E6257] hover:bg-[#F4EFE6]'}`, children: [_jsx("span", { className: `w-3 h-3 rounded-full ${activeLabel === 1 ? 'bg-white' : 'bg-[#C18C3B]'}` }), " Class 1"] })] })] }), _jsxs("div", { className: "space-y-3", children: [_jsx("label", { className: "text-xs text-[#6E6257] font-bold uppercase tracking-wider block font-mono", children: "Kernel Type" }), _jsxs("div", { className: "flex bg-[#FAF6EE] border border-[#E5DDD0] p-1 rounded-xl gap-1", children: [_jsx("button", { onClick: () => setKernel('linear'), className: `flex-1 py-2.5 rounded-lg text-sm font-bold transition-all ${kernel === 'linear' ? 'bg-[#C18C3B] text-white shadow-md' : 'text-[#6E6257] hover:bg-[#F4EFE6]'}`, children: "Linear" }), _jsx("button", { onClick: () => setKernel('rbf'), className: `flex-1 py-2.5 rounded-lg text-sm font-bold transition-all ${kernel === 'rbf' ? 'bg-[#C18C3B] text-white shadow-md' : 'text-[#6E6257] hover:bg-[#F4EFE6]'}`, children: "RBF (Non-linear)" })] })] }), _jsxs("div", { className: "space-y-4", children: [_jsxs("div", { className: "flex justify-between items-center text-sm font-bold", children: [_jsx("span", { className: "text-[#6E6257]", children: "Regularization (C):" }), _jsx("span", { className: "text-[#C18C3B] bg-[#C18C3B]/10 px-3 py-1 rounded-full", children: cParam.toFixed(1) })] }), _jsx("p", { className: "text-[10px] text-[#6E6257]", children: "High C = Hard margin (strict), Low C = Soft margin (allows errors)" }), _jsx("input", { type: "range", min: "0.1", max: "5.0", step: "0.1", value: cParam, onChange: (e) => setCParam(parseFloat(e.target.value)), className: "w-full h-2 bg-[#E5DDD0] rounded-lg appearance-none cursor-pointer accent-[#C18C3B]" }), kernel === 'rbf' && (_jsxs(_Fragment, { children: [_jsxs("div", { className: "flex justify-between items-center text-sm font-bold pt-2", children: [_jsx("span", { className: "text-[#6E6257]", children: "Gamma (Spread):" }), _jsx("span", { className: "text-[#C18C3B] bg-[#C18C3B]/10 px-3 py-1 rounded-full", children: gamma.toFixed(3) })] }), _jsx("input", { type: "range", min: "0.001", max: "0.05", step: "0.001", value: gamma, onChange: (e) => setGamma(parseFloat(e.target.value)), className: "w-full h-2 bg-[#E5DDD0] rounded-lg appearance-none cursor-pointer accent-[#C18C3B]" })] }))] })] }), _jsxs("button", { onClick: clearPoints, className: "w-full py-3 mt-6 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#F4EFE6] text-[#6E6257] hover:text-[#B6532B] text-sm font-bold transition-colors flex justify-center items-center gap-2 shadow-sm", children: [_jsx(Trash2, { className: "w-5 h-5" }), " Clear Data"] })] }), _jsx("div", { className: "md:col-span-8 flex flex-col items-center justify-center", children: _jsxs("div", { className: "bg-[#F4EFE6] border border-[#E5DDD0] p-2 rounded-3xl w-full flex justify-center shadow-md relative overflow-hidden group", children: [_jsx("canvas", { ref: canvasRef, onClick: handleCanvasClick, className: "rounded-2xl cursor-crosshair w-full aspect-[3/2]" }), _jsx("div", { className: "absolute top-6 left-6 bg-[#2E251E]/80 backdrop-blur-md px-4 py-2 rounded-xl border border-[#E5DDD0]/30 text-xs font-mono text-[#FAF6EE] shadow-lg pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity", children: "Solid line = Decision Boundary. Faded lines = Margins." })] }) })] }));
};
