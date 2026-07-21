import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useState, useEffect, useRef } from 'react';
import { Activity, Play, Trash2 } from 'lucide-react';
export const TimeSeriesSimulator = () => {
    const [dataPoints, setDataPoints] = useState([
        10, 15, 12, 18, 22, 19, 25, 30, 28, 35, 32, 40, 38, 45, 42, 50, 48, 55
    ]);
    const [lagWindow, setLagWindow] = useState(1);
    const [splitIndex, setSplitIndex] = useState(12);
    const canvasRef = useRef(null);
    const addRandomPoint = () => {
        setDataPoints(prev => {
            const last = prev[prev.length - 1] || 10;
            // Add trend and noise
            const next = Math.max(0, last + (Math.random() - 0.2) * 5);
            return [...prev, next];
        });
    };
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const ctx = canvas.getContext('2d');
        if (!ctx)
            return;
        const w = canvas.width = 600;
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
        if (dataPoints.length === 0)
            return;
        // Find max value for scaling
        const maxVal = Math.max(...dataPoints, 60);
        const minVal = 0;
        const range = maxVal - minVal;
        const getX = (idx) => (idx / Math.max(20, dataPoints.length)) * (w - 40) + 20;
        const getY = (val) => (h - 20) - ((val - minVal) / range) * (h - 40);
        // Draw Train/Test split region
        const splitX = getX(splitIndex);
        ctx.fillStyle = '#B6532B11';
        ctx.fillRect(0, 0, splitX, h);
        ctx.fillStyle = '#C18C3B11';
        ctx.fillRect(splitX, 0, w - splitX, h);
        // Split line
        ctx.strokeStyle = '#2E251E';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(splitX, 0);
        ctx.lineTo(splitX, h);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#2E251E';
        ctx.font = '10px monospace';
        ctx.fillText('TRAIN', splitX / 2 - 15, 20);
        ctx.fillText('TEST', splitX + (w - splitX) / 2 - 15, 20);
        // Draw Original Time Series
        ctx.strokeStyle = '#2E251E';
        ctx.lineWidth = 2;
        ctx.beginPath();
        dataPoints.forEach((val, idx) => {
            if (idx === 0)
                ctx.moveTo(getX(idx), getY(val));
            else
                ctx.lineTo(getX(idx), getY(val));
        });
        ctx.stroke();
        // Draw Points
        dataPoints.forEach((val, idx) => {
            ctx.beginPath();
            ctx.arc(getX(idx), getY(val), 4, 0, Math.PI * 2);
            ctx.fillStyle = idx < splitIndex ? '#B6532B' : '#C18C3B';
            ctx.fill();
        });
        // Draw Lagged Series
        if (lagWindow > 0) {
            ctx.strokeStyle = '#6E6257';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([2, 4]);
            ctx.beginPath();
            dataPoints.forEach((val, idx) => {
                // Shifting right by lagWindow
                const lagIdx = idx + lagWindow;
                if (lagIdx < dataPoints.length) {
                    if (idx === 0)
                        ctx.moveTo(getX(lagIdx), getY(val));
                    else
                        ctx.lineTo(getX(lagIdx), getY(val));
                }
            });
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }, [dataPoints, lagWindow, splitIndex]);
    return (_jsxs("div", { className: "grid grid-cols-1 md:grid-cols-12 gap-6 p-4", children: [_jsxs("div", { className: "md:col-span-4 bg-[#F4EFE6] border border-[#E5DDD0] p-5 rounded-2xl flex flex-col justify-between", children: [_jsxs("div", { className: "space-y-4", children: [_jsxs("h4", { className: "text-[#2E251E] font-bold text-base tracking-wide flex items-center gap-2", children: [_jsx(Activity, { className: "w-5 h-5 text-[#B6532B]" }), " Time Series Analysis"] }), _jsx("p", { className: "text-[#6E6257] text-xs", children: "Generate sequential data and observe the effects of temporal features and strict chronological splitting (TimeSeriesSplit)." }), _jsxs("div", { className: "space-y-3 pt-2", children: [_jsxs("div", { className: "flex justify-between text-xs font-mono", children: [_jsx("span", { className: "text-[#6E6257]", children: "Lag Shift:" }), _jsxs("span", { className: "text-[#B6532B] font-bold", children: [lagWindow, " steps"] })] }), _jsx("input", { type: "range", min: "0", max: "5", step: "1", value: lagWindow, onChange: (e) => setLagWindow(parseInt(e.target.value)), className: "w-full h-1 bg-[#E5DDD0] rounded-lg appearance-none cursor-pointer accent-[#B6532B]" })] }), _jsxs("div", { className: "space-y-3 pt-2", children: [_jsxs("div", { className: "flex justify-between text-xs font-mono", children: [_jsx("span", { className: "text-[#6E6257]", children: "Split Index (Train/Test):" }), _jsxs("span", { className: "text-[#2E251E] font-bold", children: ["t = ", splitIndex] })] }), _jsx("input", { type: "range", min: "3", max: Math.max(4, dataPoints.length - 2), step: "1", value: splitIndex, onChange: (e) => setSplitIndex(parseInt(e.target.value)), className: "w-full h-1 bg-[#E5DDD0] rounded-lg appearance-none cursor-pointer accent-[#2E251E]" })] }), _jsxs("div", { className: "p-3 bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl space-y-2 text-xs font-mono mt-4", children: [_jsxs("div", { className: "flex justify-between", children: [_jsx("span", { className: "text-[#6E6257]", children: "Data Points:" }), _jsx("span", { className: "text-[#2E251E] font-bold", children: dataPoints.length })] }), _jsxs("div", { className: "flex justify-between", children: [_jsx("span", { className: "text-[#6E6257]", children: "Train Size:" }), _jsx("span", { className: "text-[#B6532B] font-bold", children: splitIndex })] }), _jsxs("div", { className: "flex justify-between", children: [_jsx("span", { className: "text-[#6E6257]", children: "Test Size:" }), _jsx("span", { className: "text-[#C18C3B] font-bold", children: dataPoints.length - splitIndex })] })] })] }), _jsxs("div", { className: "flex gap-2 mt-4", children: [_jsxs("button", { onClick: addRandomPoint, className: "flex-1 py-2.5 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#FAF6EE]/80 text-[#2E251E] transition-all flex items-center justify-center gap-2 text-xs font-semibold", children: [_jsx(Play, { className: "w-4 h-4" }), " Next Step"] }), _jsx("button", { onClick: () => { setDataPoints([10, 15, 12, 18, 22]); setSplitIndex(3); }, className: "py-2.5 px-3 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#FAF6EE]/80 text-[#D9534F] transition-all flex items-center justify-center", title: "Reset Series", children: _jsx(Trash2, { className: "w-4 h-4" }) })] })] }), _jsx("div", { className: "md:col-span-8 flex flex-col items-center", children: _jsxs("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl relative w-full flex justify-center shadow-inner overflow-hidden", children: [_jsx("canvas", { ref: canvasRef, className: "rounded-xl border border-[#E5DDD0] w-full max-w-[600px] h-[300px]" }), _jsx("div", { className: "absolute top-7 left-7 bg-[#2E251E]/90 px-2.5 py-1 rounded-lg border border-[#E5DDD0]/20 text-[9px] font-mono text-[#FAF6EE] uppercase tracking-wide", children: "Sequence Value Over Time" }), lagWindow > 0 && (_jsxs("div", { className: "absolute top-7 right-7 bg-[#6E6257]/90 px-2.5 py-1 rounded-lg border border-[#E5DDD0]/20 text-[9px] font-mono text-[#FAF6EE] uppercase tracking-wide flex items-center gap-2", children: [_jsx("div", { className: "w-4 h-px border-b-2 border-dashed border-[#FAF6EE]" }), "Lag-", lagWindow, " Series"] }))] }) })] }));
};
