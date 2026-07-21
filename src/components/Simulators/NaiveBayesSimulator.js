import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useState, useEffect, useRef } from 'react';
import { Database, Trash2 } from 'lucide-react';
export const NaiveBayesSimulator = () => {
    const [points, setPoints] = useState([
        { x: 200, y: 150, label: 0 }, { x: 220, y: 180, label: 0 }, { x: 180, y: 160, label: 0 }, { x: 210, y: 140, label: 0 },
        { x: 400, y: 250, label: 1 }, { x: 420, y: 280, label: 1 }, { x: 380, y: 260, label: 1 }, { x: 410, y: 240, label: 1 }
    ]);
    const [activeLabel, setActiveLabel] = useState(0);
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
    const computeStats = (label) => {
        const classPoints = points.filter(p => p.label === label);
        const n = classPoints.length;
        if (n === 0)
            return { meanX: 0, meanY: 0, varX: 1, varY: 1, prior: 0 };
        const prior = n / points.length;
        let sumX = 0, sumY = 0;
        classPoints.forEach(p => { sumX += p.x; sumY += p.y; });
        const meanX = sumX / n;
        const meanY = sumY / n;
        let sumSqX = 0, sumSqY = 0;
        classPoints.forEach(p => {
            sumSqX += (p.x - meanX) ** 2;
            sumSqY += (p.y - meanY) ** 2;
        });
        // Add small epsilon to variance to prevent division by zero
        const varX = Math.max(100, sumSqX / n);
        const varY = Math.max(100, sumSqY / n);
        return { meanX, meanY, varX, varY, prior };
    };
    const gaussianPdf = (x, mean, variance) => {
        const exponent = Math.exp(-((x - mean) ** 2) / (2 * variance));
        return (1 / Math.sqrt(2 * Math.PI * variance)) * exponent;
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
        const stats0 = computeStats(0);
        const stats1 = computeStats(1);
        if (points.length > 0) {
            const step = 8;
            for (let x = 0; x < w; x += step) {
                for (let y = 0; y < h; y += step) {
                    const cx = x + step / 2;
                    const cy = y + step / 2;
                    let prob0 = 0;
                    let prob1 = 0;
                    if (stats0.prior > 0) {
                        const likelihoodX = gaussianPdf(cx, stats0.meanX, stats0.varX);
                        const likelihoodY = gaussianPdf(cy, stats0.meanY, stats0.varY);
                        // Naive assumption: P(X,Y|C) = P(X|C) * P(Y|C)
                        prob0 = stats0.prior * likelihoodX * likelihoodY;
                    }
                    if (stats1.prior > 0) {
                        const likelihoodX = gaussianPdf(cx, stats1.meanX, stats1.varX);
                        const likelihoodY = gaussianPdf(cy, stats1.meanY, stats1.varY);
                        prob1 = stats1.prior * likelihoodX * likelihoodY;
                    }
                    if (prob0 === 0 && prob1 === 0)
                        continue;
                    // Normalize
                    const totalProb = prob0 + prob1;
                    const normProb1 = prob1 / totalProb;
                    const r = Math.floor(182 * (1 - normProb1) + 193 * normProb1);
                    const g = Math.floor(83 * (1 - normProb1) + 140 * normProb1);
                    const bCol = Math.floor(43 * (1 - normProb1) + 59 * normProb1);
                    // Calculate intensity based on total probability density (how "confident" or close to the cluster)
                    // Scale it up so it's visible
                    const maxProbApprox = 1 / (2 * Math.PI * Math.sqrt(Math.min(stats0.varX, stats1.varX) * Math.min(stats0.varY, stats1.varY)));
                    const intensity = Math.min(0.5, (totalProb / maxProbApprox) * 2);
                    ctx.fillStyle = `rgba(${r}, ${g}, ${bCol}, ${intensity + 0.1})`;
                    ctx.fillRect(x, y, step, step);
                    // Draw Decision Boundary (approx)
                    if (Math.abs(normProb1 - 0.5) < 0.05 && totalProb > maxProbApprox * 0.001) {
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
        // Draw Covariance Ellipses approx (since we assume independence, axes are aligned to x/y)
        const drawEllipse = (stats, color) => {
            if (stats.prior === 0)
                return;
            ctx.beginPath();
            // 2 std deviations
            ctx.ellipse(stats.meanX, stats.meanY, Math.sqrt(stats.varX) * 2, Math.sqrt(stats.varY) * 2, 0, 0, 2 * Math.PI);
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.stroke();
            ctx.setLineDash([]);
            // Center
            ctx.beginPath();
            ctx.arc(stats.meanX, stats.meanY, 4, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
        };
        drawEllipse(stats0, '#B6532B');
        drawEllipse(stats1, '#C18C3B');
        // Draw points
        points.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2);
            ctx.fillStyle = pt.label === 0 ? '#B6532B' : '#C18C3B';
            ctx.shadowBlur = 10;
            ctx.shadowColor = ctx.fillStyle;
            ctx.fill();
            ctx.shadowBlur = 0;
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2);
            ctx.strokeStyle = '#FAF6EE';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        });
    }, [points]);
    return (_jsxs("div", { className: "grid grid-cols-1 md:grid-cols-12 gap-8 p-6", children: [_jsxs("div", { className: "md:col-span-4 bg-[#FAF6EE] border border-[#E5DDD0] p-6 rounded-3xl flex flex-col justify-between shadow-xl", children: [_jsxs("div", { className: "space-y-6", children: [_jsxs("h4", { className: "text-[#2E251E] font-bold text-xl tracking-tight flex items-center gap-3", children: [_jsx(Database, { className: "w-6 h-6 text-[#B6532B]" }), " Gaussian Naive Bayes"] }), _jsxs("p", { className: "text-[#6E6257] text-sm leading-relaxed", children: ["Naive Bayes assumes features are conditionally independent. Notice how the dashed ellipses representing the ", _jsx("span", { className: "font-semibold text-[#B6532B]", children: "Gaussian distributions" }), " are always aligned to the X and Y axes, never tilted."] }), _jsxs("div", { className: "space-y-3", children: [_jsx("label", { className: "text-xs text-[#6E6257] font-bold uppercase tracking-wider block font-mono", children: "Class Selector" }), _jsxs("div", { className: "flex bg-[#FAF6EE] border border-[#E5DDD0] p-1 rounded-xl gap-1", children: [_jsx("button", { onClick: () => setActiveLabel(0), className: `flex-1 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 ${activeLabel === 0 ? 'bg-[#B6532B] text-white shadow-md' : 'text-[#6E6257] hover:bg-[#F4EFE6]'}`, children: "Class 0" }), _jsx("button", { onClick: () => setActiveLabel(1), className: `flex-1 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 ${activeLabel === 1 ? 'bg-[#C18C3B] text-white shadow-md' : 'text-[#6E6257] hover:bg-[#F4EFE6]'}`, children: "Class 1" })] })] })] }), _jsxs("button", { onClick: clearPoints, className: "w-full py-3 mt-6 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#F4EFE6] text-[#6E6257] hover:text-[#B6532B] text-sm font-bold transition-colors flex justify-center items-center gap-2 shadow-sm", children: [_jsx(Trash2, { className: "w-5 h-5" }), " Clear Data"] })] }), _jsx("div", { className: "md:col-span-8 flex flex-col items-center justify-center", children: _jsxs("div", { className: "bg-[#F4EFE6] border border-[#E5DDD0] p-2 rounded-3xl w-full flex justify-center shadow-md relative overflow-hidden group", children: [_jsx("canvas", { ref: canvasRef, onClick: handleCanvasClick, className: "rounded-2xl cursor-crosshair w-full aspect-[3/2]" }), _jsx("div", { className: "absolute top-6 left-6 bg-[#2E251E]/80 backdrop-blur-md px-4 py-2 rounded-xl border border-[#E5DDD0]/30 text-xs font-mono text-[#FAF6EE] shadow-lg pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity", children: "Probability heatmaps via Bayes Theorem" })] }) })] }));
};
