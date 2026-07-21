import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useState, useEffect, useRef } from 'react';
import { Trees, Trash2 } from 'lucide-react';
export const RandomForestSimulator = () => {
    const [points, setPoints] = useState([
        { x: 150, y: 150, label: 0 }, { x: 180, y: 180, label: 0 }, { x: 200, y: 100, label: 0 },
        { x: 400, y: 100, label: 1 }, { x: 450, y: 160, label: 1 }, { x: 500, y: 220, label: 1 },
        { x: 300, y: 280, label: 0 }, { x: 350, y: 350, label: 1 }, { x: 250, y: 200, label: 1 }
    ]);
    const [activeLabel, setActiveLabel] = useState(0);
    const [numTrees, setNumTrees] = useState(5);
    const [maxDepth, setMaxDepth] = useState(3);
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
    // Gini Impurity
    const computeGini = (pts) => {
        if (pts.length === 0)
            return 0;
        const p0 = pts.filter(p => p.label === 0).length / pts.length;
        return 1 - (p0 * p0 + (1 - p0) * (1 - p0));
    };
    const buildTree = (pts, depth) => {
        const c0 = pts.filter(p => p.label === 0).length;
        const val = (pts.length - c0) > c0 ? 1 : 0;
        if (depth >= maxDepth || pts.length < 2 || computeGini(pts) === 0) {
            return { isLeaf: true, val };
        }
        let bestGain = -1;
        let bestFeat = 'x';
        let bestThresh = 0;
        ['x', 'y'].forEach(feat => {
            const vals = Array.from(new Set(pts.map(p => feat === 'x' ? p.x : p.y))).sort((a, b) => a - b);
            for (let i = 0; i < vals.length - 1; i++) {
                const thresh = (vals[i] + vals[i + 1]) / 2;
                const left = pts.filter(p => (feat === 'x' ? p.x : p.y) <= thresh);
                const right = pts.filter(p => (feat === 'x' ? p.x : p.y) > thresh);
                if (left.length === 0 || right.length === 0)
                    continue;
                const gain = computeGini(pts) - ((left.length / pts.length) * computeGini(left) + (right.length / pts.length) * computeGini(right));
                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeat = feat;
                    bestThresh = thresh;
                }
            }
        });
        if (bestGain <= 0)
            return { isLeaf: true, val };
        return {
            isLeaf: false,
            feature: bestFeat,
            threshold: bestThresh,
            val,
            left: buildTree(pts.filter(p => (bestFeat === 'x' ? p.x : p.y) <= bestThresh), depth + 1),
            right: buildTree(pts.filter(p => (bestFeat === 'x' ? p.x : p.y) > bestThresh), depth + 1)
        };
    };
    const predict = (tree, x, y) => {
        let curr = tree;
        while (!curr.isLeaf) {
            if ((curr.feature === 'x' ? x : y) <= curr.threshold)
                curr = curr.left;
            else
                curr = curr.right;
        }
        return curr.val;
    };
    const forest = React.useMemo(() => {
        const trees = [];
        if (points.length === 0)
            return trees;
        for (let t = 0; t < numTrees; t++) {
            // Bootstrap sampling
            const sample = [];
            for (let i = 0; i < points.length; i++) {
                sample.push(points[Math.floor(Math.random() * points.length)]);
            }
            trees.push(buildTree(sample, 0));
        }
        return trees;
    }, [points, numTrees, maxDepth]);
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const ctx = canvas.getContext('2d');
        if (!ctx)
            return;
        const w = canvas.width = 600;
        const h = canvas.height = 400;
        // Draw Voting Background
        const step = 6;
        for (let x = 0; x < w; x += step) {
            for (let y = 0; y < h; y += step) {
                let votesFor1 = 0;
                forest.forEach(tree => {
                    if (predict(tree, x + step / 2, y + step / 2) === 1)
                        votesFor1++;
                });
                const prob = forest.length > 0 ? votesFor1 / forest.length : 0.5;
                const r = Math.floor(244 * (1 - prob) + 56 * prob);
                const g = Math.floor(63 * (1 - prob) + 189 * prob);
                const bCol = Math.floor(94 * (1 - prob) + 248 * prob);
                ctx.fillStyle = `rgba(${r}, ${g}, ${bCol}, 0.3)`;
                ctx.fillRect(x, y, step, step);
            }
        }
        // Grid Overlay
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
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
            ctx.fillStyle = pt.label === 0 ? '#f43f5e' : '#38bdf8';
            ctx.shadowBlur = 10;
            ctx.shadowColor = pt.label === 0 ? '#f43f5e' : '#38bdf8';
            ctx.fill();
            ctx.shadowBlur = 0;
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 8, 0, Math.PI * 2);
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.stroke();
        });
    }, [points, forest]);
    let correct = 0;
    if (forest.length > 0) {
        points.forEach(p => {
            let votes = 0;
            forest.forEach(t => {
                if (predict(t, p.x, p.y) === 1)
                    votes++;
            });
            const finalPred = votes > forest.length / 2 ? 1 : 0;
            if (finalPred === p.label)
                correct++;
        });
    }
    const accuracy = points.length > 0 ? (correct / points.length) * 100 : 0;
    return (_jsxs("div", { className: "grid grid-cols-1 md:grid-cols-12 gap-8 p-6", children: [_jsxs("div", { className: "md:col-span-4 bg-[#FAF6EE] border border-[#E5DDD0] p-6 rounded-3xl flex flex-col justify-between shadow-xl", children: [_jsxs("div", { className: "space-y-6", children: [_jsxs("h4", { className: "text-slate-900 font-bold text-xl tracking-tight flex items-center gap-3", children: [_jsx(Trees, { className: "w-6 h-6 text-emerald-500" }), " Random Forest"] }), _jsxs("p", { className: "text-slate-500 text-sm leading-relaxed", children: ["An ensemble of independent decision trees trained on bootstrap samples of the data. The final boundary is determined by ", _jsx("span", { className: "font-semibold text-emerald-500", children: "majority voting" }), ", creating a smoother boundary than a single tree."] }), _jsxs("div", { className: "space-y-3", children: [_jsx("label", { className: "text-xs text-slate-400 font-bold uppercase tracking-wider block", children: "Class Selector" }), _jsxs("div", { className: "flex bg-slate-100 p-1 rounded-xl gap-1", children: [_jsxs("button", { onClick: () => setActiveLabel(0), className: `flex-1 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 ${activeLabel === 0 ? 'bg-rose-500 text-white shadow-md' : 'text-slate-500 hover:bg-slate-200'}`, children: [_jsx("span", { className: `w-3 h-3 rounded-full ${activeLabel === 0 ? 'bg-white' : 'bg-rose-500'}` }), " Class 0"] }), _jsxs("button", { onClick: () => setActiveLabel(1), className: `flex-1 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 ${activeLabel === 1 ? 'bg-sky-500 text-white shadow-md' : 'text-slate-500 hover:bg-slate-200'}`, children: [_jsx("span", { className: `w-3 h-3 rounded-full ${activeLabel === 1 ? 'bg-white' : 'bg-sky-500'}` }), " Class 1"] })] })] }), _jsxs("div", { className: "space-y-4", children: [_jsxs("div", { className: "flex justify-between items-center text-sm font-bold", children: [_jsx("span", { className: "text-slate-500", children: "Number of Trees:" }), _jsx("span", { className: "text-emerald-600 bg-emerald-50 px-3 py-1 rounded-full", children: numTrees })] }), _jsx("input", { type: "range", min: "1", max: "25", step: "1", value: numTrees, onChange: (e) => setNumTrees(parseInt(e.target.value)), className: "w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-500" }), _jsxs("div", { className: "flex justify-between items-center text-sm font-bold pt-2", children: [_jsx("span", { className: "text-slate-500", children: "Max Depth Limit:" }), _jsx("span", { className: "text-emerald-600 bg-emerald-50 px-3 py-1 rounded-full", children: maxDepth })] }), _jsx("input", { type: "range", min: "1", max: "8", step: "1", value: maxDepth, onChange: (e) => setMaxDepth(parseInt(e.target.value)), className: "w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-500" })] }), _jsx("div", { className: "p-5 bg-slate-50 border border-slate-200 rounded-2xl space-y-3 shadow-inner mt-4", children: _jsxs("div", { className: "flex justify-between items-center", children: [_jsx("span", { className: "text-slate-500 text-sm font-medium", children: "Forest Accuracy" }), _jsxs("span", { className: "text-emerald-600 font-mono font-bold text-lg", children: [accuracy.toFixed(1), "%"] })] }) })] }), _jsxs("button", { onClick: clearPoints, className: "w-full py-3 mt-6 rounded-xl bg-red-50 text-red-600 hover:bg-red-100 text-sm font-bold transition-colors flex justify-center items-center gap-2", children: [_jsx(Trash2, { className: "w-5 h-5" }), " Clear Data"] })] }), _jsx("div", { className: "md:col-span-8 flex flex-col items-center justify-center", children: _jsxs("div", { className: "bg-[#2E251E] border border-[#4A3D31] p-2 rounded-3xl w-full flex justify-center shadow-2xl relative overflow-hidden group", children: [_jsx("canvas", { ref: canvasRef, onClick: handleCanvasClick, className: "rounded-2xl cursor-crosshair w-full aspect-[3/2]" }), _jsx("div", { className: "absolute top-6 left-6 bg-white/10 backdrop-blur-md px-4 py-2 rounded-xl border border-white/20 text-xs font-mono text-white shadow-lg pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity", children: "Watch the boundaries smooth out as trees increase" })] }) })] }));
};
