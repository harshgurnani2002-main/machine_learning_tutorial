import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useState, useMemo, useCallback } from 'react';
// Calculate Variance (MSE) for regression splits
function calculateVariance(points) {
    if (points.length === 0)
        return 0;
    const mean = points.reduce((sum, p) => sum + p.cls, 0) / points.length;
    return points.reduce((sum, p) => sum + Math.pow(p.cls - mean, 2), 0) / points.length;
}
// Build a CART Regression Tree
function buildTree(points, depth, maxDepth, minSamples = 2) {
    if (depth >= maxDepth || points.length < minSamples) {
        const mean = points.length > 0 ? points.reduce((sum, p) => sum + p.cls, 0) / points.length : 0;
        return { value: mean, feature: 'x', threshold: 0 };
    }
    let bestFeature = 'x';
    let bestThreshold = 0;
    let bestCost = Infinity;
    let bestGroups = [[], []];
    ['x', 'y'].forEach(feat => {
        const sorted = [...points].sort((a, b) => a[feat] - b[feat]);
        for (let i = 0; i < sorted.length - 1; i++) {
            if (sorted[i][feat] === sorted[i + 1][feat])
                continue;
            const thresh = (sorted[i][feat] + sorted[i + 1][feat]) / 2;
            const left = [];
            const right = [];
            for (const p of points) {
                if (p[feat] <= thresh)
                    left.push(p);
                else
                    right.push(p);
            }
            if (left.length === 0 || right.length === 0)
                continue;
            const cost = (left.length * calculateVariance(left) + right.length * calculateVariance(right)) / points.length;
            if (cost < bestCost) {
                bestCost = cost;
                bestFeature = feat;
                bestThreshold = thresh;
                bestGroups = [left, right];
            }
        }
    });
    if (bestCost === Infinity || bestGroups[0].length === 0 || bestGroups[1].length === 0) {
        const mean = points.length > 0 ? points.reduce((sum, p) => sum + p.cls, 0) / points.length : 0;
        return { value: mean, feature: 'x', threshold: 0 };
    }
    return {
        feature: bestFeature,
        threshold: bestThreshold,
        left: buildTree(bestGroups[0], depth + 1, maxDepth, minSamples),
        right: buildTree(bestGroups[1], depth + 1, maxDepth, minSamples)
    };
}
function predictTree(node, x, y) {
    if (node.value !== undefined)
        return node.value;
    const val = node.feature === 'x' ? x : y;
    if (val <= node.threshold)
        return predictTree(node.left, x, y);
    return predictTree(node.right, x, y);
}
// Seeded random for bootstrap sampling
function seededRandom(seed) {
    let s = seed;
    return () => {
        s = Math.sin(s) * 10000;
        return s - Math.floor(s);
    };
}
export const EnsembleSimulator = () => {
    const [mode, setMode] = useState('single');
    const [points, setPoints] = useState([]);
    const [activeClass, setActiveClass] = useState(0);
    // Controls
    const [rfTrees, setRfTrees] = useState(10);
    const [gbmIters, setGbmIters] = useState(5);
    const [maxDepthSingle, setMaxDepthSingle] = useState(6);
    const W = 500, H = 400;
    const GRID_SIZE = 10;
    // Generate Initial Data
    const generateData = useCallback(() => {
        const newPoints = [];
        const rand = Math.random;
        for (let i = 0; i < 30; i++) {
            newPoints.push({ id: `p${rand()}`, x: 50 + rand() * 150, y: 50 + rand() * 200, cls: 0 });
            newPoints.push({ id: `p${rand()}`, x: 250 + rand() * 200, y: 150 + rand() * 200, cls: 1 });
        }
        // Add noise
        for (let i = 0; i < 15; i++) {
            newPoints.push({ id: `p${rand()}`, x: 150 + rand() * 200, y: 100 + rand() * 200, cls: rand() > 0.5 ? 1 : 0 });
        }
        setPoints(newPoints);
    }, []);
    // Initialize once
    React.useEffect(() => {
        generateData();
    }, [generateData]);
    // Train Models dynamically based on current points and mode
    const { boundary, trainError } = useMemo(() => {
        const cells = [];
        if (points.length === 0)
            return { boundary: [], trainError: '0.0' };
        let modelPredict;
        if (mode === 'single') {
            const tree = buildTree(points, 0, maxDepthSingle, 2);
            modelPredict = (x, y) => predictTree(tree, x, y);
        }
        else if (mode === 'bagging') {
            const trees = [];
            const rng = seededRandom(42); // deterministic bagging
            for (let i = 0; i < rfTrees; i++) {
                const bootstrap = [];
                for (let j = 0; j < points.length; j++) {
                    bootstrap.push(points[Math.floor(rng() * points.length)]);
                }
                trees.push(buildTree(bootstrap, 0, 6, 2)); // deeper trees for RF
            }
            modelPredict = (x, y) => {
                let sum = 0;
                for (const t of trees)
                    sum += predictTree(t, x, y);
                return sum / trees.length;
            };
        }
        else { // boosting
            const trees = [];
            const learningRate = 0.3;
            let currentF = points.map(() => 0.5); // Initial F0
            for (let i = 0; i < gbmIters; i++) {
                // Compute residuals
                const residuals = points.map((p, idx) => ({ ...p, cls: p.cls - currentF[idx] }));
                const tree = buildTree(residuals, 0, 2, 2); // Shallow trees (stumps/depth 2) for Boosting
                trees.push(tree);
                // Update F
                currentF = currentF.map((f, idx) => f + learningRate * predictTree(tree, points[idx].x, points[idx].y));
            }
            modelPredict = (x, y) => {
                let f = 0.5;
                for (const t of trees) {
                    f += learningRate * predictTree(t, x, y);
                }
                return f;
            };
        }
        // Compute boundary grid
        for (let gx = 0; gx < W; gx += GRID_SIZE) {
            for (let gy = 0; gy < H; gy += GRID_SIZE) {
                const px = gx + GRID_SIZE / 2;
                const py = gy + GRID_SIZE / 2;
                let score = modelPredict(px, py);
                score = Math.max(0, Math.min(1, score));
                cells.push({ x: gx, y: gy, score });
            }
        }
        // Calculate training error
        let errCount = 0;
        for (const p of points) {
            const pred = modelPredict(p.x, p.y) > 0.5 ? 1 : 0;
            if (pred !== p.cls)
                errCount++;
        }
        return { boundary: cells, trainError: ((errCount / points.length) * 100).toFixed(1) };
    }, [points, mode, maxDepthSingle, rfTrees, gbmIters]);
    const handleCanvasClick = (e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        setPoints(prev => [...prev, { id: `p${Math.random()}`, x, y, cls: activeClass }]);
    };
    const clearData = () => setPoints([]);
    // Theme Constants (Premium Light Theme)
    const colors = {
        bg: '#F8FAFC',
        card: '#FFFFFF',
        text: '#1E293B',
        textMuted: '#64748B',
        border: '#E2E8F0',
        primary: '#6366F1', // Indigo
        class0: '#F43F5E', // Rose
        class1: '#0EA5E9', // Sky Blue
        accentBg: '#EEF2FF'
    };
    return (_jsxs("div", { className: "ens-container", style: {
            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
            backgroundColor: colors.bg,
            color: colors.text,
            padding: '32px',
            borderRadius: '24px',
            boxShadow: '0 10px 30px -5px rgba(0,0,0,0.05), border 1px solid rgba(0,0,0,0.05)',
            display: 'flex',
            flexDirection: 'column',
            gap: '24px',
            maxWidth: '1000px',
            margin: '0 auto',
        }, children: [_jsx("style", { children: `
        .ens-btn {
          padding: 10px 18px;
          border-radius: 12px;
          font-weight: 600;
          font-size: 14px;
          cursor: pointer;
          transition: all 0.2s ease;
          border: 1px solid ${colors.border};
          background: ${colors.card};
          color: ${colors.textMuted};
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .ens-btn:hover {
          background: ${colors.accentBg};
          border-color: ${colors.primary};
          color: ${colors.primary};
        }
        .ens-btn.active {
          background: ${colors.primary};
          color: white;
          border-color: ${colors.primary};
          box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }
        
        .ens-class-btn {
          width: 32px; height: 32px;
          border-radius: 50%;
          border: 3px solid transparent;
          cursor: pointer;
          transition: transform 0.2s;
        }
        .ens-class-btn:hover { transform: scale(1.1); }
        .ens-class-btn.active { border-color: ${colors.text}; transform: scale(1.1); }

        .ens-slider {
          -webkit-appearance: none;
          width: 100%;
          height: 6px;
          background: #E2E8F0;
          border-radius: 3px;
          outline: none;
        }
        .ens-slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 18px;
          height: 18px;
          border-radius: 50%;
          background: ${colors.primary};
          cursor: pointer;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
          transition: transform 0.1s;
        }
        .ens-slider::-webkit-slider-thumb:hover { transform: scale(1.2); }
      ` }), _jsxs("div", { style: { textAlign: 'center', marginBottom: '8px' }, children: [_jsx("h2", { style: { margin: 0, fontSize: '32px', fontWeight: 800, color: colors.text, letterSpacing: '-0.5px' }, children: "Interactive True Ensemble Simulator" }), _jsx("p", { style: { margin: '8px 0 0', fontSize: '16px', color: colors.textMuted }, children: "Add points to the canvas and watch real math algorithms construct the decision boundaries live!" })] }), _jsxs("div", { style: { display: 'flex', gap: '24px', flexWrap: 'wrap' }, children: [_jsxs("div", { style: { flex: '1', minWidth: '500px', display: 'flex', flexDirection: 'column', gap: '16px' }, children: [_jsxs("div", { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: colors.card, padding: '12px 20px', borderRadius: '16px', border: `1px solid ${colors.border}`, boxShadow: '0 2px 4px rgba(0,0,0,0.02)' }, children: [_jsxs("div", { style: { display: 'flex', alignItems: 'center', gap: '12px' }, children: [_jsx("span", { style: { fontSize: '14px', fontWeight: 600, color: colors.textMuted }, children: "Draw Tool:" }), _jsx("button", { className: `ens-class-btn ${activeClass === 0 ? 'active' : ''}`, style: { backgroundColor: colors.class0 }, onClick: () => setActiveClass(0), title: "Add Class 0 (Red)" }), _jsx("button", { className: `ens-class-btn ${activeClass === 1 ? 'active' : ''}`, style: { backgroundColor: colors.class1 }, onClick: () => setActiveClass(1), title: "Add Class 1 (Blue)" }), _jsx("span", { style: { fontSize: '12px', color: colors.textMuted, marginLeft: '8px' }, children: "(Click canvas to add)" })] }), _jsxs("div", { style: { display: 'flex', gap: '8px' }, children: [_jsx("button", { onClick: generateData, className: "ens-btn", style: { padding: '6px 12px', fontSize: '12px' }, children: "\uD83C\uDFB2 Randomize" }), _jsx("button", { onClick: clearData, className: "ens-btn", style: { padding: '6px 12px', fontSize: '12px' }, children: "\uD83D\uDDD1\uFE0F Clear" })] })] }), _jsx("div", { style: { position: 'relative', borderRadius: '20px', overflow: 'hidden', border: `1px solid ${colors.border}`, background: '#FFFFFF', boxShadow: '0 10px 25px -5px rgba(0,0,0,0.05)' }, children: _jsxs("svg", { width: W, height: H, onClick: handleCanvasClick, style: { cursor: 'crosshair', display: 'block' }, children: [boundary.map((cell, i) => (_jsx("rect", { x: cell.x, y: cell.y, width: GRID_SIZE + 0.5, height: GRID_SIZE + 0.5, fill: cell.score > 0.5 ? colors.class1 : colors.class0, opacity: 0.1 + Math.abs(cell.score - 0.5) * 0.8, style: { transition: 'opacity 0.2s, fill 0.2s' } }, `bg-${i}`))), points.map((pt) => (_jsx("circle", { cx: pt.x, cy: pt.y, r: 6, fill: pt.cls === 0 ? colors.class0 : colors.class1, stroke: "#FFFFFF", strokeWidth: 2, style: { transition: 'all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)' } }, pt.id)))] }) })] }), _jsxs("div", { style: { flex: '0 0 340px', display: 'flex', flexDirection: 'column', gap: '16px' }, children: [_jsxs("div", { style: { background: colors.card, padding: '20px', borderRadius: '20px', border: `1px solid ${colors.border}`, boxShadow: '0 4px 6px -1px rgba(0,0,0,0.05)' }, children: [_jsx("h3", { style: { margin: '0 0 16px 0', fontSize: '14px', textTransform: 'uppercase', color: colors.textMuted, letterSpacing: '1px' }, children: "Algorithm Selection" }), _jsxs("div", { style: { display: 'flex', flexDirection: 'column', gap: '10px' }, children: [_jsx("button", { className: `ens-btn ${mode === 'single' ? 'active' : ''}`, onClick: () => setMode('single'), children: "\uD83C\uDF33 Single Decision Tree" }), _jsx("button", { className: `ens-btn ${mode === 'bagging' ? 'active' : ''}`, onClick: () => setMode('bagging'), children: "\uD83C\uDF32 Bagging (Random Forest)" }), _jsx("button", { className: `ens-btn ${mode === 'boosting' ? 'active' : ''}`, onClick: () => setMode('boosting'), children: "\u26A1 Boosting (Gradient Boost)" })] })] }), _jsxs("div", { style: { background: colors.card, padding: '20px', borderRadius: '20px', border: `1px solid ${colors.border}`, boxShadow: '0 4px 6px -1px rgba(0,0,0,0.05)', flex: 1 }, children: [_jsx("div", { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '20px' }, children: _jsx("h3", { style: { margin: 0, fontSize: '18px', fontWeight: 700, color: colors.primary }, children: mode === 'single' ? 'Single Tree Setup' : mode === 'bagging' ? 'Forest Settings' : 'Boosting Stages' }) }), mode === 'single' && (_jsxs("div", { style: { display: 'flex', flexDirection: 'column', gap: '12px' }, children: [_jsx("p", { style: { margin: 0, fontSize: '13px', color: colors.textMuted, lineHeight: 1.5 }, children: "A single unconstrained tree perfectly memorizes data but creates jagged, unnatural boundaries. Try adding outlier points!" }), _jsxs("div", { style: { marginTop: '16px' }, children: [_jsxs("div", { style: { display: 'flex', justifyContent: 'space-between', fontSize: '13px', fontWeight: 600, marginBottom: '8px' }, children: [_jsx("span", { children: "Max Depth" }), _jsx("span", { children: maxDepthSingle })] }), _jsx("input", { type: "range", min: 1, max: 10, value: maxDepthSingle, onChange: e => setMaxDepthSingle(Number(e.target.value)), className: "ens-slider" })] })] })), mode === 'bagging' && (_jsxs("div", { style: { display: 'flex', flexDirection: 'column', gap: '12px' }, children: [_jsx("p", { style: { margin: 0, fontSize: '13px', color: colors.textMuted, lineHeight: 1.5 }, children: "Trains many deep trees on random subsets of points and averages them. Notice how the boundary becomes remarkably smooth and resistant to outliers!" }), _jsxs("div", { style: { marginTop: '16px' }, children: [_jsxs("div", { style: { display: 'flex', justifyContent: 'space-between', fontSize: '13px', fontWeight: 600, marginBottom: '8px' }, children: [_jsx("span", { children: "Number of Trees" }), _jsx("span", { children: rfTrees })] }), _jsx("input", { type: "range", min: 1, max: 50, value: rfTrees, onChange: e => setRfTrees(Number(e.target.value)), className: "ens-slider" })] })] })), mode === 'boosting' && (_jsxs("div", { style: { display: 'flex', flexDirection: 'column', gap: '12px' }, children: [_jsx("p", { style: { margin: 0, fontSize: '13px', color: colors.textMuted, lineHeight: 1.5 }, children: "Sequentially adds shallow trees (stumps). Each new tree focuses only on correcting the mistakes of the previous sequence." }), _jsxs("div", { style: { marginTop: '16px' }, children: [_jsxs("div", { style: { display: 'flex', justifyContent: 'space-between', fontSize: '13px', fontWeight: 600, marginBottom: '8px' }, children: [_jsx("span", { children: "Boosting Iterations" }), _jsx("span", { children: gbmIters })] }), _jsx("input", { type: "range", min: 1, max: 30, value: gbmIters, onChange: e => setGbmIters(Number(e.target.value)), className: "ens-slider" })] })] })), _jsxs("div", { style: { marginTop: '24px', paddingTop: '20px', borderTop: `1px solid ${colors.border}` }, children: [_jsx("div", { style: { fontSize: '12px', color: colors.textMuted, textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '8px' }, children: "Live Metrics" }), _jsxs("div", { style: { background: colors.accentBg, borderRadius: '12px', padding: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }, children: [_jsx("span", { style: { fontSize: '14px', fontWeight: 600, color: colors.primary }, children: "Training Error" }), _jsxs("span", { style: { fontSize: '24px', fontWeight: 800, color: trainError === '0.0' ? '#10B981' : colors.text, fontFamily: 'monospace' }, children: [trainError, "%"] })] }), _jsx("div", { style: { fontSize: '11px', color: colors.textMuted, marginTop: '8px', textAlign: 'center' }, children: "(Models automatically retrain when data changes)" })] })] })] })] })] }));
};
