import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Pause, RotateCcw, Layers } from 'lucide-react';
// Generate 4 high-dimensional clusters (simulated as 3D, then projected)
function generateHDClusters(n = 60, seed = 7) {
    let s = seed;
    const rng = () => {
        s = (s * 1664525 + 1013904223) & 0xffffffff;
        return (s >>> 0) / 0xffffffff;
    };
    const centers = [
        [2, 2, 1, -1, 0.5],
        [-2, -2, -1, 1, -0.5],
        [2, -2, 0, 0, 1],
        [-2, 2, 0.5, -0.5, -1],
    ];
    const pts = [];
    for (let i = 0; i < n; i++) {
        const cluster = Math.floor(rng() * centers.length);
        const center = centers[cluster];
        const hd = center.map(c => c + (rng() - 0.5) * 1.5);
        pts.push({ hd, cluster });
    }
    return pts;
}
// PCA-like projection (random orthogonal projection, fixed per session)
function projectPCA(pts) {
    return pts.map((p, i) => ({
        x: p.hd[0] * 0.7 + p.hd[1] * 0.4 + p.hd[2] * 0.3 + (Math.sin(i * 0.5) * 0.2),
        y: p.hd[1] * 0.6 - p.hd[0] * 0.3 + p.hd[3] * 0.4 + (Math.cos(i * 0.5) * 0.2),
        cluster: p.cluster,
    }));
}
// t-SNE simulation: iterative attractive/repulsive forces
function tsneStep(points, perplexity) {
    const n = points.length;
    const sigma = Math.max(0.3, perplexity / 20);
    return points.map((pi, i) => {
        let fx = 0, fy = 0;
        points.forEach((pj, j) => {
            if (i === j)
                return;
            const dx = pj.x - pi.x;
            const dy = pj.y - pi.y;
            const dist2 = dx * dx + dy * dy + 0.01;
            // Attractive force within same cluster (simulated)
            const attraction = pi.cluster === pj.cluster ? 0.015 / (dist2 * 0.5 + sigma) : 0;
            // Repulsive force (Student-t kernel)
            const repulsion = -0.008 / (n * dist2);
            const force = attraction + repulsion;
            fx += force * dx;
            fy += force * dy;
        });
        // Boundary force to keep in [-2, 2]
        const bk = 0.05;
        if (pi.x > 2)
            fx -= bk * (pi.x - 2);
        if (pi.x < -2)
            fx -= bk * (pi.x + 2);
        if (pi.y > 2)
            fy -= bk * (pi.y - 2);
        if (pi.y < -2)
            fy -= bk * (pi.y + 2);
        return {
            x: pi.x + fx,
            y: pi.y + fy,
            cluster: pi.cluster,
        };
    });
}
const CLUSTER_COLORS = ['#B6532B', '#C18C3B', '#3B7A57', '#6E6257'];
const CLUSTER_LABELS = ['Class A', 'Class B', 'Class C', 'Class D'];
export const TSNESimulator = () => {
    const [perplexity, setPerplexity] = useState(15);
    const [isPlaying, setIsPlaying] = useState(false);
    const [iteration, setIteration] = useState(0);
    const [mode, setMode] = useState('pca');
    const [points, setPoints] = useState([]);
    const canvasRef = useRef(null);
    const rawData = generateHDClusters(64);
    const initPoints = useCallback(() => {
        const pca = projectPCA(rawData);
        setPoints(pca);
        setIteration(0);
        setIsPlaying(false);
        setMode('pca');
    }, []);
    useEffect(() => {
        initPoints();
    }, []);
    // Auto-start t-SNE after a short delay
    useEffect(() => {
        const timer = setTimeout(() => {
            setMode('tsne');
            setIsPlaying(true);
        }, 1000);
        return () => clearTimeout(timer);
    }, []);
    // Animation loop
    useEffect(() => {
        if (!isPlaying || mode === 'pca')
            return;
        const id = setInterval(() => {
            setPoints(prev => {
                const next = tsneStep(prev, perplexity);
                return next;
            });
            setIteration(prev => {
                const next = prev + 1;
                if (next >= 200)
                    setIsPlaying(false);
                return next;
            });
        }, 60);
        return () => clearInterval(id);
    }, [isPlaying, mode, perplexity]);
    // Draw
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const ctx = canvas.getContext('2d');
        if (!ctx)
            return;
        const W = canvas.width = 420;
        const H = canvas.height = 380;
        const cx = W / 2, cy = H / 2;
        const scale = 80;
        ctx.fillStyle = '#FAF6EE';
        ctx.fillRect(0, 0, W, H);
        // Grid
        ctx.strokeStyle = '#E5DDD0';
        ctx.lineWidth = 0.5;
        for (let x = 0; x < W; x += 42) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, H);
            ctx.stroke();
        }
        for (let y = 0; y < H; y += 38) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(W, y);
            ctx.stroke();
        }
        // Axes
        ctx.strokeStyle = 'rgba(46, 37, 30, 0.15)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(cx, 10);
        ctx.lineTo(cx, H - 10);
        ctx.moveTo(10, cy);
        ctx.lineTo(W - 10, cy);
        ctx.stroke();
        // Points
        points.forEach(p => {
            const px = cx + p.x * scale;
            const py = cy + p.y * scale;
            if (px < 5 || px > W - 5 || py < 5 || py > H - 5)
                return;
            ctx.beginPath();
            ctx.arc(px, py, 5.5, 0, Math.PI * 2);
            ctx.fillStyle = CLUSTER_COLORS[p.cluster] + 'CC';
            ctx.strokeStyle = '#FAF6EE';
            ctx.lineWidth = 1.5;
            ctx.fill();
            ctx.stroke();
        });
        // Legend
        CLUSTER_COLORS.forEach((col, i) => {
            ctx.beginPath();
            ctx.arc(16, 16 + i * 18, 5, 0, Math.PI * 2);
            ctx.fillStyle = col;
            ctx.fill();
            ctx.fillStyle = '#6E6257';
            ctx.font = '9px monospace';
            ctx.textAlign = 'left';
            ctx.fillText(CLUSTER_LABELS[i], 26, 20 + i * 18);
        });
        // Status
        ctx.fillStyle = '#2E251E';
        ctx.font = 'bold 10px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(mode === 'pca' ? 'PCA Projection' : `t-SNE (iter ${iteration})`, W - 10, H - 10);
    }, [points, mode, iteration]);
    const reset = () => {
        initPoints();
    };
    const runTSNE = () => {
        setMode('tsne');
        setIteration(0);
        const pca = projectPCA(rawData);
        setPoints(pca);
        setTimeout(() => setIsPlaying(true), 100);
    };
    // Separation quality metric
    const separationScore = (() => {
        if (points.length === 0)
            return 0;
        let interDist = 0, intraDist = 0, interCount = 0, intraCount = 0;
        for (let i = 0; i < points.length; i++) {
            for (let j = i + 1; j < points.length; j++) {
                const d = Math.sqrt((points[i].x - points[j].x) ** 2 + (points[i].y - points[j].y) ** 2);
                if (points[i].cluster === points[j].cluster) {
                    intraDist += d;
                    intraCount++;
                }
                else {
                    interDist += d;
                    interCount++;
                }
            }
        }
        const avgIntra = intraCount > 0 ? intraDist / intraCount : 1;
        const avgInter = interCount > 0 ? interDist / interCount : 1;
        return Math.min(1, Math.max(0, (avgInter - avgIntra) / (avgInter + avgIntra + 0.01)));
    })();
    return (_jsxs("div", { className: "grid grid-cols-1 md:grid-cols-12 gap-6 p-4", children: [_jsxs("div", { className: "md:col-span-4 bg-[#F4EFE6] border border-[#E5DDD0] p-5 rounded-2xl flex flex-col justify-between", children: [_jsxs("div", { className: "space-y-5", children: [_jsxs("div", { children: [_jsxs("h4", { className: "text-[#2E251E] font-bold text-base tracking-wide flex items-center gap-2", children: [_jsx(Layers, { className: "w-5 h-5 text-[#B6532B]" }), " t-SNE Visualizer"] }), _jsx("p", { className: "text-[#6E6257] text-xs mt-1", children: "Watch 64 high-dimensional points (5D clusters) being compressed into 2D using t-SNE's attractive/repulsive forces." })] }), _jsxs("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-3 space-y-2", children: [_jsxs("div", { className: "flex justify-between text-xs font-mono", children: [_jsx("span", { className: "text-[#6E6257]", children: "Mode:" }), _jsx("span", { className: "font-bold text-[#B6532B]", children: mode === 'pca' ? 'PCA (Linear)' : 't-SNE (Non-linear)' })] }), _jsxs("div", { className: "flex justify-between text-xs font-mono", children: [_jsx("span", { className: "text-[#6E6257]", children: "Iteration:" }), _jsxs("span", { className: "font-bold text-[#2E251E]", children: [iteration, " / 200"] })] }), _jsxs("div", { className: "flex justify-between text-xs font-mono", children: [_jsx("span", { className: "text-[#6E6257]", children: "Cluster Separation:" }), _jsxs("span", { className: "font-bold", style: { color: separationScore > 0.4 ? '#3B7A57' : '#C18C3B' }, children: [(separationScore * 100).toFixed(0), "%"] })] })] }), _jsxs("div", { children: [_jsx("div", { className: "h-2 bg-[#E5DDD0] rounded-full overflow-hidden", children: _jsx("div", { className: "h-full bg-[#B6532B] rounded-full transition-all", style: { width: `${(iteration / 200) * 100}%` } }) }), _jsx("p", { className: "text-[10px] font-mono text-[#6E6257] mt-1 text-center", children: iteration < 200 ? 'Optimizing layout...' : '✓ Converged' })] }), _jsxs("div", { children: [_jsxs("div", { className: "flex justify-between text-xs font-mono mb-1.5", children: [_jsx("span", { className: "text-[#2E251E] font-semibold", children: "Perplexity" }), _jsx("span", { className: "text-[#B6532B] font-bold", children: perplexity })] }), _jsx("input", { type: "range", min: "5", max: "40", step: "1", value: perplexity, onChange: e => setPerplexity(parseInt(e.target.value)), className: "w-full accent-[#B6532B]" }), _jsxs("div", { className: "flex justify-between text-[10px] text-[#6E6257] font-mono mt-1", children: [_jsx("span", { children: "5 (local)" }), _jsx("span", { children: "40 (global)" })] })] }), _jsxs("div", { className: "text-[10px] text-[#6E6257] font-mono bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-3", children: [_jsx("p", { className: "font-bold text-[#2E251E] mb-1", children: "How it works:" }), _jsx("p", { children: "\u2022 Each point is attracted to its cluster neighbors" }), _jsx("p", { children: "\u2022 All points repel each other (Student-t distribution)" }), _jsx("p", { children: "\u2022 Higher perplexity = more global structure preserved" })] })] }), _jsxs("div", { className: "space-y-2 mt-4", children: [_jsxs("button", { onClick: () => setIsPlaying(p => !p), disabled: mode === 'pca', className: `w-full py-3 rounded-xl flex items-center justify-center gap-2 text-sm font-bold transition-all shadow-sm ${isPlaying
                                    ? 'bg-amber-600 text-white'
                                    : 'bg-[#B6532B] text-white hover:bg-[#B6532B]/90'} disabled:opacity-50 disabled:cursor-not-allowed`, children: [isPlaying ? _jsx(Pause, { className: "w-4 h-4" }) : _jsx(Play, { className: "w-4 h-4" }), isPlaying ? 'Pause' : 'Resume t-SNE'] }), _jsxs("div", { className: "grid grid-cols-2 gap-2", children: [_jsx("button", { onClick: runTSNE, className: "py-2.5 rounded-xl border border-[#B6532B]/30 bg-[#B6532B]/5 text-[#B6532B] text-xs font-bold font-mono transition-all hover:bg-[#B6532B]/10", children: "Run t-SNE" }), _jsxs("button", { onClick: reset, className: "py-2.5 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257] text-xs font-mono hover:text-[#2E251E] flex items-center justify-center gap-1.5", children: [_jsx(RotateCcw, { className: "w-3.5 h-3.5" }), " Reset"] })] })] })] }), _jsx("div", { className: "md:col-span-8", children: _jsxs("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-inner h-full flex flex-col", children: [_jsx("h5", { className: "text-[#2E251E] text-xs font-semibold font-mono mb-3", children: "2D Embedding Space \u2014 clusters separate as t-SNE converges" }), _jsx("div", { className: "flex justify-center", children: _jsx("canvas", { ref: canvasRef, className: "rounded-xl w-full max-w-[420px]" }) }), _jsx("p", { className: "text-[10px] text-[#6E6257] font-mono text-center mt-2", children: mode === 'pca'
                                ? 'PCA: linear projection — clusters overlap in 2D'
                                : iteration < 50
                                    ? 'Early t-SNE: random placement with repulsion starting...'
                                    : iteration < 150
                                        ? 'Mid t-SNE: clusters beginning to form distinct groups...'
                                        : '✓ Clusters now clearly separated in 2D embedding space' })] }) })] }));
};
