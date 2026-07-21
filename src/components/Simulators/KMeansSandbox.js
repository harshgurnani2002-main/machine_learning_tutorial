import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useState, useEffect, useRef } from 'react';
import { GitPullRequest, ArrowRight, Play, Pause, RotateCcw } from 'lucide-react';
export const KMeansSandbox = () => {
    const [k, setK] = useState(3);
    const [isPlaying, setIsPlaying] = useState(false);
    const [step, setStep] = useState('assign');
    const [iterations, setIterations] = useState(0);
    const canvasRef = useRef(null);
    // Simulation state
    const [points, setPoints] = useState([]);
    const [centroids, setCentroids] = useState([]);
    // Curated Light Sepia cluster palette
    const centroidColors = ['#B6532B', '#C18C3B', '#3B7A57', '#6E6257', '#2E251E'];
    // Initialize random scatter data and initial centroids
    const initializeSimulation = () => {
        setIsPlaying(false);
        setIterations(0);
        setStep('assign');
        const w = 400;
        const h = 320;
        // Generate clusters of points around some centers
        const newPoints = [];
        const centers = [
            { x: w * 0.3, y: h * 0.3 },
            { x: w * 0.7, y: h * 0.4 },
            { x: w * 0.45, y: h * 0.75 }
        ];
        for (let i = 0; i < 70; i++) {
            const center = centers[Math.floor(Math.random() * centers.length)];
            newPoints.push({
                x: center.x + (Math.random() - 0.5) * 80,
                y: center.y + (Math.random() - 0.5) * 80,
                cluster: -1
            });
        }
        // Set centroids
        const newCentroids = [];
        for (let i = 0; i < k; i++) {
            newCentroids.push({
                x: Math.random() * (w - 60) + 30,
                y: Math.random() * (h - 60) + 30,
                color: centroidColors[i]
            });
        }
        setPoints(newPoints);
        setCentroids(newCentroids);
    };
    useEffect(() => {
        initializeSimulation();
    }, [k]);
    // Auto-start after mount for one-click experience
    useEffect(() => {
        const timer = setTimeout(() => setIsPlaying(true), 700);
        return () => clearTimeout(timer);
    }, []);
    // Execute single iteration step
    const executeStep = () => {
        if (centroids.length === 0)
            return;
        if (step === 'assign') {
            // Step 1: Assign each point to nearest centroid
            const updatedPoints = points.map(pt => {
                let minDist = Infinity;
                let closestCluster = -1;
                centroids.forEach((c, idx) => {
                    const dx = pt.x - c.x;
                    const dy = pt.y - c.y;
                    const dist = dx * dx + dy * dy;
                    if (dist < minDist) {
                        minDist = dist;
                        closestCluster = idx;
                    }
                });
                return { ...pt, cluster: closestCluster };
            });
            setPoints(updatedPoints);
            setStep('update');
        }
        else {
            // Step 2: Recalculate centroids (mean of points in that cluster)
            const updatedCentroids = centroids.map((c, idx) => {
                const clusterPoints = points.filter(p => p.cluster === idx);
                if (clusterPoints.length === 0)
                    return c; // keep same
                const sumX = clusterPoints.reduce((acc, p) => acc + p.x, 0);
                const sumY = clusterPoints.reduce((acc, p) => acc + p.y, 0);
                return {
                    ...c,
                    x: sumX / clusterPoints.length,
                    y: sumY / clusterPoints.length
                };
            });
            // Check if centroids stopped moving (convergence)
            let shifted = false;
            centroids.forEach((c, idx) => {
                const dist = Math.sqrt((c.x - updatedCentroids[idx].x) ** 2 + (c.y - updatedCentroids[idx].y) ** 2);
                if (dist > 1.0)
                    shifted = true;
            });
            if (!shifted && iterations > 1) {
                setIsPlaying(false);
            }
            setCentroids(updatedCentroids);
            setIterations(prev => prev + 1);
            setStep('assign');
        }
    };
    // Play loop
    useEffect(() => {
        if (isPlaying) {
            const interval = setInterval(executeStep, 1000);
            return () => clearInterval(interval);
        }
    }, [isPlaying, points, centroids, step]);
    // Canvas drawing: Voronoi grid backgrounds + scatter
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const ctx = canvas.getContext('2d');
        if (!ctx)
            return;
        const w = canvas.width = 400;
        const h = canvas.height = 320;
        // Base background (Cream)
        ctx.fillStyle = '#FAF6EE';
        ctx.fillRect(0, 0, w, h);
        // Draw Voronoi cells backgrounds
        if (centroids.length > 0) {
            const scale = 4; // grid downsampling for rendering performance
            const sw = Math.ceil(w / scale);
            const sh = Math.ceil(h / scale);
            for (let y = 0; y < sh; y++) {
                for (let x = 0; x < sw; x++) {
                    const px = x * scale;
                    const py = y * scale;
                    let closestIdx = 0;
                    let minDist = Infinity;
                    centroids.forEach((c, idx) => {
                        const dx = px - c.x;
                        const dy = py - c.y;
                        const dist = dx * dx + dy * dy;
                        if (dist < minDist) {
                            minDist = dist;
                            closestIdx = idx;
                        }
                    });
                    // Soft background watercolor wash (8% opacity)
                    ctx.fillStyle = centroids[closestIdx].color + '14';
                    ctx.fillRect(px, py, scale, scale);
                }
            }
        }
        // Draw Grid lines
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
        // Draw Scatter points
        points.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 4.5, 0, Math.PI * 2);
            ctx.fillStyle = pt.cluster === -1 ? '#CFC5B4' : centroids[pt.cluster]?.color || '#CFC5B4';
            ctx.strokeStyle = '#FAF6EE';
            ctx.lineWidth = 1;
            ctx.fill();
            ctx.stroke();
        });
        // Draw Centroids
        centroids.forEach((c, idx) => {
            // Halo glow
            ctx.beginPath();
            ctx.arc(c.x, c.y, 14, 0, Math.PI * 2);
            ctx.fillStyle = c.color + '1a'; // 10% opacity
            ctx.fill();
            // Outer ring
            ctx.strokeStyle = c.color;
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            ctx.arc(c.x, c.y, 7, 0, Math.PI * 2);
            ctx.stroke();
            // Inner center target
            ctx.beginPath();
            ctx.arc(c.x, c.y, 2, 0, Math.PI * 2);
            ctx.fillStyle = '#2E251E'; // Espresso center
            ctx.fill();
            // Label index with shadow background for readability
            ctx.fillStyle = '#2E251E';
            ctx.font = 'bold 9px monospace';
            ctx.fillText(`μ${idx + 1}`, c.x + 10, c.y - 6);
        });
    }, [points, centroids]);
    return (_jsxs("div", { className: "grid grid-cols-1 md:grid-cols-12 gap-6 p-4", children: [_jsxs("div", { className: "md:col-span-5 bg-[#F4EFE6] border border-[#E5DDD0] p-5 rounded-2xl flex flex-col justify-between", children: [_jsxs("div", { children: [_jsxs("h4", { className: "text-[#2E251E] font-bold text-base mb-1 tracking-wide flex items-center gap-2", children: [_jsx(GitPullRequest, { className: "w-5 h-5 text-[#B6532B]" }), " K-Means Sandbox"] }), _jsx("p", { className: "text-[#6E6257] text-xs mb-5", children: "Watch clusters organize dynamically using Voronoi partitioning space maps." }), _jsxs("div", { className: "space-y-6", children: [_jsxs("div", { children: [_jsxs("div", { className: "flex justify-between text-xs font-mono mb-2", children: [_jsx("span", { className: "text-[#2E251E]", children: "Centroids Count (K)" }), _jsx("span", { className: "text-[#B6532B] font-semibold", children: k })] }), _jsx("input", { type: "range", min: "2", max: "5", step: "1", value: k, onChange: (e) => setK(parseInt(e.target.value)), className: "w-full h-1 bg-[#E5DDD0] rounded-lg appearance-none cursor-pointer accent-[#B6532B]" })] }), _jsxs("div", { className: "p-3.5 bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl space-y-2 text-xs font-mono text-[#2E251E]", children: [_jsxs("div", { className: "flex justify-between", children: [_jsx("span", { className: "text-[#6E6257]", children: "Active State:" }), _jsxs("span", { className: "uppercase flex items-center gap-1 font-bold", children: [step === 'assign' ? 'Assign clusters' : 'Recalculate centroids', _jsx(ArrowRight, { className: "w-3.5 h-3.5 text-[#B6532B]" })] })] }), _jsxs("div", { className: "flex justify-between", children: [_jsx("span", { className: "text-[#6E6257]", children: "Iterations:" }), _jsx("span", { className: "text-[#B6532B] font-bold", children: iterations })] }), _jsxs("div", { className: "flex justify-between", children: [_jsx("span", { className: "text-[#6E6257]", children: "Convergence:" }), _jsx("span", { className: "text-[#2E251E] font-bold", children: isPlaying ? 'Running' : 'Converged / Idle' })] })] })] })] }), _jsxs("div", { className: "pt-5 border-t border-[#E5DDD0] space-y-3", children: [_jsxs("div", { className: "grid grid-cols-2 gap-2", children: [_jsxs("button", { onClick: () => setIsPlaying(!isPlaying), className: `py-3 rounded-xl flex items-center justify-center gap-1.5 text-xs font-semibold shadow-sm transition-all cursor-pointer ${isPlaying
                                            ? 'bg-amber-600 border-amber-600 text-white'
                                            : 'bg-[#B6532B] border-[#B6532B] text-white hover:bg-[#B6532B]/95'}`, children: [isPlaying ? _jsx(Pause, { className: "w-4 h-4" }) : _jsx(Play, { className: "w-4 h-4" }), isPlaying ? 'Pause' : 'Auto Step'] }), _jsx("button", { onClick: executeStep, disabled: isPlaying, className: "py-3 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#FAF6EE]/80 text-[#2E251E] disabled:opacity-50 transition-all text-xs font-mono font-bold cursor-pointer", children: "Single Step" })] }), _jsxs("button", { onClick: initializeSimulation, className: "w-full py-2.5 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#FAF6EE]/85 text-[#6E6257] hover:text-[#2E251E] transition-colors flex items-center justify-center gap-2 text-xs font-semibold cursor-pointer", children: [_jsx(RotateCcw, { className: "w-4 h-4" }), " Reset centroids"] })] })] }), _jsx("div", { className: "md:col-span-7 flex flex-col items-center", children: _jsxs("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl w-full flex justify-center shadow-inner relative", children: [_jsx("canvas", { ref: canvasRef, className: "rounded-xl border border-[#E5DDD0] w-full max-w-[400px] h-[320px]" }), _jsx("div", { className: "absolute top-7 left-7 bg-[#2E251E]/90 px-2.5 py-1 rounded-lg border border-[#E5DDD0]/20 text-[9px] font-mono text-[#FAF6EE] uppercase tracking-wide", children: "Voronoi Cluster Margins" })] }) })] }));
};
