import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useState, useEffect, useRef } from 'react';
import { Trash2, GitMerge } from 'lucide-react';
export const DecisionTreeSimulator = () => {
    const [points, setPoints] = useState([
        { x: 150, y: 150, label: 0 },
        { x: 180, y: 180, label: 0 },
        { x: 200, y: 100, label: 0 },
        { x: 400, y: 100, label: 1 },
        { x: 450, y: 160, label: 1 },
        { x: 500, y: 220, label: 1 },
        { x: 300, y: 280, label: 0 },
        { x: 350, y: 350, label: 1 },
    ]);
    const [activeLabel, setActiveLabel] = useState(0);
    const [maxDepth, setMaxDepth] = useState(3);
    const [criterion, setCriterion] = useState('gini');
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
    const computeImpurity = (pts) => {
        if (pts.length === 0)
            return 0;
        const c0 = pts.filter(p => p.label === 0).length;
        const c1 = pts.length - c0;
        const p0 = c0 / pts.length;
        const p1 = c1 / pts.length;
        if (criterion === 'gini') {
            return 1 - (p0 * p0 + p1 * p1);
        }
        else {
            let ent = 0;
            if (p0 > 0)
                ent -= p0 * Math.log2(p0);
            if (p1 > 0)
                ent -= p1 * Math.log2(p1);
            return ent;
        }
    };
    const buildTree = (pts, depth, region) => {
        const c0 = pts.filter(p => p.label === 0).length;
        const c1 = pts.length - c0;
        const val = c1 > c0 ? 1 : 0;
        const impurity = computeImpurity(pts);
        const leaf = {
            isLeaf: true,
            val,
            depth,
            impurity,
            samples: pts.length,
            classCounts: [c0, c1],
            region
        };
        if (depth >= maxDepth || pts.length < 2 || impurity === 0) {
            return leaf;
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
                const pLeft = left.length / pts.length;
                const pRight = right.length / pts.length;
                const gain = impurity - (pLeft * computeImpurity(left) + pRight * computeImpurity(right));
                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeat = feat;
                    bestThresh = thresh;
                }
            }
        });
        if (bestGain <= 0)
            return leaf;
        const leftPts = pts.filter(p => (bestFeat === 'x' ? p.x : p.y) <= bestThresh);
        const rightPts = pts.filter(p => (bestFeat === 'x' ? p.x : p.y) > bestThresh);
        const leftRegion = { ...region, [bestFeat === 'x' ? 'maxX' : 'maxY']: bestThresh };
        const rightRegion = { ...region, [bestFeat === 'x' ? 'minX' : 'minY']: bestThresh };
        return {
            isLeaf: false,
            feature: bestFeat,
            threshold: bestThresh,
            depth,
            impurity,
            samples: pts.length,
            classCounts: [c0, c1],
            region,
            val,
            left: buildTree(leftPts, depth + 1, leftRegion),
            right: buildTree(rightPts, depth + 1, rightRegion)
        };
    };
    const tree = buildTree(points, 0, { minX: 0, maxX: 600, minY: 0, maxY: 400 });
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
        const drawRegions = (node) => {
            if (node.isLeaf) {
                // Terracotta for 0, Ochre for 1
                ctx.fillStyle = node.val === 0 ? 'rgba(182, 83, 43, 0.15)' : 'rgba(193, 140, 59, 0.15)';
                ctx.fillRect(node.region.minX, node.region.minY, node.region.maxX - node.region.minX, node.region.maxY - node.region.minY);
            }
            else {
                if (node.left)
                    drawRegions(node.left);
                if (node.right)
                    drawRegions(node.right);
                ctx.strokeStyle = 'rgba(46, 37, 30, 0.5)';
                ctx.lineWidth = 2;
                ctx.setLineDash([4, 4]);
                ctx.beginPath();
                if (node.feature === 'x') {
                    ctx.moveTo(node.threshold, node.region.minY);
                    ctx.lineTo(node.threshold, node.region.maxY);
                }
                else {
                    ctx.moveTo(node.region.minX, node.threshold);
                    ctx.lineTo(node.region.maxX, node.threshold);
                }
                ctx.stroke();
                ctx.setLineDash([]);
            }
        };
        drawRegions(tree);
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
            ctx.fillStyle = pt.label === 0 ? '#B6532B' : '#C18C3B';
            ctx.shadowBlur = 10;
            ctx.shadowColor = pt.label === 0 ? '#B6532B' : '#C18C3B';
            ctx.fill();
            ctx.shadowBlur = 0;
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 8, 0, Math.PI * 2);
            ctx.strokeStyle = '#FAF6EE';
            ctx.lineWidth = 2;
            ctx.stroke();
        });
    }, [points, maxDepth, criterion, tree]);
    let totalNodes = 0;
    let totalLeaves = 0;
    const traverseStats = (node) => {
        totalNodes++;
        if (node.isLeaf)
            totalLeaves++;
        else {
            if (node.left)
                traverseStats(node.left);
            if (node.right)
                traverseStats(node.right);
        }
    };
    traverseStats(tree);
    let correct = 0;
    points.forEach(p => {
        let curr = tree;
        while (!curr.isLeaf) {
            if ((curr.feature === 'x' ? p.x : p.y) <= curr.threshold)
                curr = curr.left;
            else
                curr = curr.right;
        }
        if (curr.val === p.label)
            correct++;
    });
    const accuracy = points.length > 0 ? (correct / points.length) * 100 : 0;
    return (_jsxs("div", { className: "grid grid-cols-1 md:grid-cols-12 gap-8 p-6", children: [_jsxs("div", { className: "md:col-span-4 bg-[#FAF6EE] border border-[#E5DDD0] p-6 rounded-3xl flex flex-col justify-between shadow-xl", children: [_jsxs("div", { className: "space-y-6", children: [_jsxs("h4", { className: "text-[#2E251E] font-bold text-xl tracking-tight flex items-center gap-3", children: [_jsx(GitMerge, { className: "w-6 h-6 text-[#3B7A57]" }), " Decision Tree"] }), _jsxs("p", { className: "text-[#6E6257] text-sm leading-relaxed", children: ["Trees partition space into pure rectangular regions using recursive orthogonal splits. Watch how ", _jsx("span", { className: "font-semibold text-[#3B7A57]", children: "Max Depth" }), " prevents overfitting."] }), _jsxs("div", { className: "space-y-3", children: [_jsx("label", { className: "text-xs text-[#6E6257] font-bold uppercase tracking-wider block font-mono", children: "Class to Plot" }), _jsxs("div", { className: "flex bg-[#FAF6EE] border border-[#E5DDD0] p-1 rounded-xl gap-1", children: [_jsxs("button", { onClick: () => setActiveLabel(0), className: `flex-1 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 ${activeLabel === 0 ? 'bg-[#B6532B] text-white shadow-md' : 'text-[#6E6257] hover:bg-[#F4EFE6]'}`, children: [_jsx("span", { className: `w-3 h-3 rounded-full ${activeLabel === 0 ? 'bg-white' : 'bg-[#B6532B]'}` }), " Class 0"] }), _jsxs("button", { onClick: () => setActiveLabel(1), className: `flex-1 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 ${activeLabel === 1 ? 'bg-[#C18C3B] text-white shadow-md' : 'text-[#6E6257] hover:bg-[#F4EFE6]'}`, children: [_jsx("span", { className: `w-3 h-3 rounded-full ${activeLabel === 1 ? 'bg-white' : 'bg-[#C18C3B]'}` }), " Class 1"] })] })] }), _jsxs("div", { className: "space-y-3", children: [_jsx("label", { className: "text-xs text-[#6E6257] font-bold uppercase tracking-wider block font-mono", children: "Impurity Metric" }), _jsxs("div", { className: "flex bg-[#FAF6EE] border border-[#E5DDD0] p-1 rounded-xl gap-1", children: [_jsx("button", { onClick: () => setCriterion('gini'), className: `flex-1 py-2.5 rounded-lg text-sm font-bold transition-all ${criterion === 'gini' ? 'bg-[#3B7A57] text-white shadow-md' : 'text-[#6E6257] hover:bg-[#F4EFE6]'}`, children: "Gini Impurity" }), _jsx("button", { onClick: () => setCriterion('entropy'), className: `flex-1 py-2.5 rounded-lg text-sm font-bold transition-all ${criterion === 'entropy' ? 'bg-[#3B7A57] text-white shadow-md' : 'text-[#6E6257] hover:bg-[#F4EFE6]'}`, children: "Entropy" })] })] }), _jsxs("div", { className: "space-y-4", children: [_jsxs("div", { className: "flex justify-between items-center text-sm font-bold", children: [_jsx("span", { className: "text-[#6E6257]", children: "Max Depth Limit:" }), _jsx("span", { className: "text-[#3B7A57] bg-[#3B7A57]/10 px-3 py-1 rounded-full", children: maxDepth })] }), _jsx("input", { type: "range", min: "1", max: "8", step: "1", value: maxDepth, onChange: (e) => setMaxDepth(parseInt(e.target.value)), className: "w-full h-2 bg-[#E5DDD0] rounded-lg appearance-none cursor-pointer accent-[#3B7A57]" })] }), _jsxs("div", { className: "p-5 bg-[#FAF6EE] border border-[#E5DDD0] rounded-2xl space-y-3 shadow-inner", children: [_jsxs("div", { className: "flex justify-between items-center", children: [_jsx("span", { className: "text-[#6E6257] text-sm font-medium", children: "Accuracy" }), _jsxs("span", { className: "text-[#3B7A57] font-mono font-bold text-lg", children: [accuracy.toFixed(1), "%"] })] }), _jsxs("div", { className: "flex justify-between items-center", children: [_jsx("span", { className: "text-[#6E6257] text-sm font-medium", children: "Tree Size" }), _jsxs("span", { className: "text-[#2E251E] font-mono font-bold", children: [totalNodes, " nodes"] })] })] })] }), _jsxs("button", { onClick: clearPoints, className: "w-full py-3 mt-6 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#F4EFE6] text-[#6E6257] hover:text-[#B6532B] transition-all flex items-center justify-center gap-2 text-sm font-bold shadow-sm", children: [_jsx(Trash2, { className: "w-5 h-5" }), " Clear Data"] })] }), _jsx("div", { className: "md:col-span-8 flex flex-col items-center justify-center", children: _jsxs("div", { className: "bg-[#F4EFE6] border border-[#E5DDD0] p-2 rounded-3xl w-full flex justify-center shadow-md relative overflow-hidden group", children: [_jsx("canvas", { ref: canvasRef, onClick: handleCanvasClick, className: "rounded-2xl cursor-crosshair w-full aspect-[3/2]" }), _jsx("div", { className: "absolute top-6 left-6 bg-[#2E251E]/80 backdrop-blur-md px-4 py-2 rounded-xl border border-[#E5DDD0]/30 text-xs font-mono text-[#FAF6EE] shadow-lg pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity", children: "Click to plot \u2022 Observe recursive splits" })] }) })] }));
};
