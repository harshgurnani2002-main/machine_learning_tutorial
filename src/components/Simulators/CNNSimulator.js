import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Layers, Play, RotateCcw, Cpu, Grid3x3, Minimize2, Sliders } from 'lucide-react';
const INPUT_SIZE = 8;
const KERNEL_SIZE = 3;
const OUTPUT_SIZE = INPUT_SIZE - KERNEL_SIZE + 1;
const POOL_SIZE = 2;
const POOLED_SIZE = Math.floor(OUTPUT_SIZE / POOL_SIZE);
const CELL_SIZE = 30;
const KERNEL_CELL = 36;
const COLORS = {
    bg: '#FAF6EE',
    border: '#E5DDD0',
    text: '#2E251E',
    muted: '#6E6257',
    accent: '#B6532B',
    panelBg: '#F4EFE6',
    gold: '#C18C3B',
    cellOn: '#2E251E',
    cellOff: '#FFFFFF',
};
const filters = {
    edge: {
        name: 'Edge Detect',
        matrix: [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
    },
    blur: {
        name: 'Box Blur',
        matrix: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
    },
    sharpen: {
        name: 'Sharpen',
        matrix: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    },
    custom: {
        name: 'Custom',
        matrix: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    },
};
const initGrid = (rows, cols, val = 0) => Array.from({ length: rows }, () => Array(cols).fill(val));
export const CNNSimulator = () => {
    const [inputGrid, setInputGrid] = useState(() => initGrid(INPUT_SIZE, INPUT_SIZE));
    const [activeFilter, setActiveFilter] = useState('edge');
    const [customKernel, setCustomKernel] = useState('0,0,0,0,1,0,0,0,0');
    const [outputGrid, setOutputGrid] = useState(null);
    const [pooledGrid, setPooledGrid] = useState(null);
    const [enablePooling, setEnablePooling] = useState(false);
    const [animating, setAnimating] = useState(false);
    const [animStep, setAnimStep] = useState(-1);
    const [animOutput, setAnimOutput] = useState(() => initGrid(OUTPUT_SIZE, OUTPUT_SIZE));
    const inputRef = useRef(null);
    const outputRef = useRef(null);
    const pooledRef = useRef(null);
    const kernelRef = useRef(null);
    const getKernel = useCallback(() => {
        if (activeFilter === 'custom') {
            const vals = customKernel.split(',').map(v => parseFloat(v.trim()));
            if (vals.length !== 9 || vals.some(isNaN))
                return filters.edge.matrix;
            return [
                [vals[0], vals[1], vals[2]],
                [vals[3], vals[4], vals[5]],
                [vals[6], vals[7], vals[8]],
            ];
        }
        return filters[activeFilter].matrix;
    }, [activeFilter, customKernel]);
    const maxPool = (grid) => {
        const result = initGrid(POOLED_SIZE, POOLED_SIZE);
        for (let py = 0; py < POOLED_SIZE; py++) {
            for (let px = 0; px < POOLED_SIZE; px++) {
                let maxVal = -Infinity;
                for (let dy = 0; dy < POOL_SIZE; dy++) {
                    for (let dx = 0; dx < POOL_SIZE; dx++) {
                        maxVal = Math.max(maxVal, grid[py * POOL_SIZE + dy][px * POOL_SIZE + dx]);
                    }
                }
                result[py][px] = maxVal;
            }
        }
        return result;
    };
    const heatColor = (val, maxVal, r1, g1, b1) => {
        if (maxVal === 0)
            return COLORS.cellOff;
        const ratio = val / maxVal;
        return `rgb(${Math.round(255 - (255 - r1) * ratio)},${Math.round(255 - (255 - g1) * ratio)},${Math.round(255 - (255 - b1) * ratio)})`;
    };
    useEffect(() => {
        const c = inputRef.current;
        if (!c)
            return;
        const ctx = c.getContext('2d');
        if (!ctx)
            return;
        c.width = INPUT_SIZE * CELL_SIZE;
        c.height = INPUT_SIZE * CELL_SIZE;
        for (let y = 0; y < INPUT_SIZE; y++) {
            for (let x = 0; x < INPUT_SIZE; x++) {
                ctx.fillStyle = inputGrid[y][x] ? COLORS.cellOn : COLORS.cellOff;
                ctx.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                ctx.strokeStyle = COLORS.border;
                ctx.strokeRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
            }
        }
        if (animating && animStep >= 0) {
            const oy = Math.floor(animStep / OUTPUT_SIZE);
            const ox = animStep % OUTPUT_SIZE;
            ctx.fillStyle = 'rgba(182, 83, 43, 0.2)';
            ctx.fillRect(ox * CELL_SIZE, oy * CELL_SIZE, KERNEL_SIZE * CELL_SIZE, KERNEL_SIZE * CELL_SIZE);
            ctx.strokeStyle = COLORS.accent;
            ctx.lineWidth = 2;
            ctx.strokeRect(ox * CELL_SIZE, oy * CELL_SIZE, KERNEL_SIZE * CELL_SIZE, KERNEL_SIZE * CELL_SIZE);
            ctx.lineWidth = 1;
        }
    }, [inputGrid, animating, animStep]);
    useEffect(() => {
        const c = outputRef.current;
        if (!c)
            return;
        const ctx = c.getContext('2d');
        if (!ctx)
            return;
        c.width = OUTPUT_SIZE * CELL_SIZE;
        c.height = OUTPUT_SIZE * CELL_SIZE;
        let maxVal = 0;
        for (let y = 0; y < OUTPUT_SIZE; y++)
            for (let x = 0; x < OUTPUT_SIZE; x++)
                maxVal = Math.max(maxVal, animOutput[y][x]);
        for (let y = 0; y < OUTPUT_SIZE; y++) {
            for (let x = 0; x < OUTPUT_SIZE; x++) {
                const val = animOutput[y][x];
                ctx.fillStyle = heatColor(val, maxVal, 182, 83, 43);
                ctx.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                ctx.strokeStyle = COLORS.border;
                ctx.strokeRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                ctx.fillStyle = maxVal > 0 && val > maxVal * 0.5 ? '#FFFFFF' : COLORS.text;
                ctx.font = 'bold 10px monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(val.toFixed(1), x * CELL_SIZE + CELL_SIZE / 2, y * CELL_SIZE + CELL_SIZE / 2);
            }
        }
    }, [animOutput]);
    useEffect(() => {
        const c = pooledRef.current;
        if (!c)
            return;
        const ctx = c.getContext('2d');
        if (!ctx)
            return;
        c.width = POOLED_SIZE * CELL_SIZE;
        c.height = POOLED_SIZE * CELL_SIZE;
        const grid = pooledGrid || [];
        let maxVal = 0;
        for (let y = 0; y < POOLED_SIZE; y++)
            for (let x = 0; x < POOLED_SIZE; x++)
                maxVal = Math.max(maxVal, grid[y]?.[x] ?? 0);
        for (let y = 0; y < POOLED_SIZE; y++) {
            for (let x = 0; x < POOLED_SIZE; x++) {
                const val = grid[y]?.[x] ?? 0;
                ctx.fillStyle = heatColor(val, maxVal, 193, 140, 59);
                ctx.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                ctx.strokeStyle = COLORS.border;
                ctx.strokeRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                ctx.fillStyle = maxVal > 0 && val > maxVal * 0.5 ? '#FFFFFF' : COLORS.text;
                ctx.font = 'bold 10px monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(val.toFixed(1), x * CELL_SIZE + CELL_SIZE / 2, y * CELL_SIZE + CELL_SIZE / 2);
            }
        }
    }, [pooledGrid]);
    useEffect(() => {
        const c = kernelRef.current;
        if (!c)
            return;
        const ctx = c.getContext('2d');
        if (!ctx)
            return;
        c.width = KERNEL_SIZE * KERNEL_CELL;
        c.height = KERNEL_SIZE * KERNEL_CELL;
        const kernel = getKernel();
        for (let y = 0; y < KERNEL_SIZE; y++) {
            for (let x = 0; x < KERNEL_SIZE; x++) {
                ctx.fillStyle = COLORS.panelBg;
                ctx.fillRect(x * KERNEL_CELL, y * KERNEL_CELL, KERNEL_CELL, KERNEL_CELL);
                ctx.strokeStyle = COLORS.border;
                ctx.strokeRect(x * KERNEL_CELL, y * KERNEL_CELL, KERNEL_CELL, KERNEL_CELL);
                ctx.fillStyle = COLORS.accent;
                ctx.font = 'bold 12px monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                const v = kernel[y][x];
                ctx.fillText(v % 1 !== 0 ? v.toFixed(2) : v.toFixed(0), x * KERNEL_CELL + KERNEL_CELL / 2, y * KERNEL_CELL + KERNEL_CELL / 2);
            }
        }
    }, [getKernel]);
    const handleInputClick = (e) => {
        if (animating)
            return;
        const c = inputRef.current;
        if (!c)
            return;
        const rect = c.getBoundingClientRect();
        const x = Math.floor((e.clientX - rect.left) / CELL_SIZE);
        const y = Math.floor((e.clientY - rect.top) / CELL_SIZE);
        if (x >= 0 && x < INPUT_SIZE && y >= 0 && y < INPUT_SIZE) {
            setInputGrid(prev => {
                const g = prev.map(r => [...r]);
                g[y][x] = g[y][x] ? 0 : 1;
                return g;
            });
        }
    };
    const runConvolution = () => {
        if (animating)
            return;
        const kernel = getKernel();
        const totalSteps = OUTPUT_SIZE * OUTPUT_SIZE;
        setAnimating(true);
        const fresh = initGrid(OUTPUT_SIZE, OUTPUT_SIZE);
        setAnimOutput(fresh);
        setOutputGrid(null);
        setPooledGrid(null);
        let step = 0;
        const partial = initGrid(OUTPUT_SIZE, OUTPUT_SIZE);
        const tick = () => {
            if (step >= totalSteps) {
                setAnimOutput(partial.map(r => [...r]));
                setOutputGrid(partial.map(r => [...r]));
                if (enablePooling)
                    setPooledGrid(maxPool(partial));
                setAnimating(false);
                setAnimStep(-1);
                return;
            }
            const oy = Math.floor(step / OUTPUT_SIZE);
            const ox = step % OUTPUT_SIZE;
            let sum = 0;
            for (let ky = 0; ky < KERNEL_SIZE; ky++)
                for (let kx = 0; kx < KERNEL_SIZE; kx++)
                    sum += inputGrid[oy + ky][ox + kx] * kernel[ky][kx];
            partial[oy][ox] = Math.max(0, sum);
            setAnimOutput(partial.map(r => [...r]));
            setAnimStep(step);
            step++;
            setTimeout(tick, 150);
        };
        setTimeout(tick, 300);
    };
    const reset = () => {
        setInputGrid(initGrid(INPUT_SIZE, INPUT_SIZE));
        setOutputGrid(null);
        setPooledGrid(null);
        setAnimOutput(initGrid(OUTPUT_SIZE, OUTPUT_SIZE));
        setAnimating(false);
        setAnimStep(-1);
    };
    const stats = (() => {
        const grid = outputGrid || animOutput;
        let min = Infinity;
        let max = -Infinity;
        for (let y = 0; y < OUTPUT_SIZE; y++)
            for (let x = 0; x < OUTPUT_SIZE; x++) {
                const v = grid[y][x];
                if (v < min)
                    min = v;
                if (v > max)
                    max = v;
            }
        return { min: min === Infinity ? 0 : min, max: max === -Infinity ? 0 : max };
    })();
    return (_jsxs("div", { className: "grid grid-cols-1 lg:grid-cols-12 gap-6 p-6 h-full bg-[#FAF6EE] overflow-y-auto", children: [_jsxs("div", { className: "lg:col-span-6 space-y-6", children: [_jsxs("div", { className: "bg-[#F4EFE6] border border-[#E5DDD0] p-5 rounded-2xl shadow-xl", children: [_jsxs("h4", { className: "text-[#2E251E] font-bold text-lg tracking-tight flex items-center gap-2 mb-4", children: [_jsx(Layers, { className: "w-5 h-5 text-[#B6532B]" }), " CNN Simulator"] }), _jsx("p", { className: "text-[#6E6257] text-xs leading-relaxed mb-4", children: "Draw a shape on the 8\u00D78 grid, then convolve a filter over it." }), _jsxs("div", { className: "mb-3", children: [_jsxs("label", { className: "text-[10px] text-[#6E6257] font-mono uppercase font-bold block mb-1.5", children: [_jsx(Sliders, { className: "w-3 h-3 inline mr-1" }), "Filter Kernel"] }), _jsx("div", { className: "grid grid-cols-2 gap-1.5 bg-[#FAF6EE] p-1 rounded-xl border border-[#E5DDD0]", children: Object.keys(filters).map(type => (_jsx("button", { onClick: () => setActiveFilter(type), className: `py-1.5 px-2 rounded-lg text-[10px] font-bold transition-all truncate ${activeFilter === type
                                                ? 'bg-white shadow-sm text-[#B6532B]'
                                                : 'text-[#6E6257] hover:text-[#2E251E]'}`, children: filters[type].name }, type))) })] }), activeFilter === 'custom' && (_jsxs("div", { className: "mb-3", children: [_jsx("label", { className: "text-[10px] text-[#6E6257] font-mono uppercase font-bold block mb-1", children: "3\u00D73 comma-separated" }), _jsx("input", { type: "text", value: customKernel, onChange: e => setCustomKernel(e.target.value), className: "w-full bg-[#FAF6EE] border border-[#E5DDD0] rounded-lg px-2 py-1.5 text-xs font-mono text-[#2E251E] focus:outline-none focus:ring-1 focus:ring-[#B6532B]", placeholder: "0,0,0,0,1,0,0,0,0" })] })), _jsx("div", { className: "flex items-center gap-2 mb-3", children: _jsxs("button", { onClick: () => setEnablePooling(!enablePooling), className: `text-[10px] font-bold px-2.5 py-1.5 rounded-lg border transition-all flex items-center gap-1.5 ${enablePooling
                                        ? 'border-[#C18C3B] bg-[#C18C3B]/10 text-[#C18C3B]'
                                        : 'border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257]'}`, children: [_jsx(Minimize2, { className: "w-3 h-3" }), " 2\u00D72 Max Pool"] }) }), _jsxs("div", { className: "flex gap-2", children: [_jsxs("button", { onClick: runConvolution, disabled: animating, className: "flex-1 flex items-center justify-center gap-1.5 bg-[#B6532B] text-white text-xs font-bold py-2 rounded-xl hover:bg-[#9E4625] transition-all disabled:opacity-50 disabled:cursor-not-allowed", children: [_jsx(Play, { className: "w-3.5 h-3.5" }), " ", animating ? 'Animating...' : 'Convolve'] }), _jsxs("button", { onClick: reset, disabled: animating, className: "flex items-center justify-center gap-1.5 bg-[#F4EFE6] border border-[#E5DDD0] text-[#6E6257] text-xs font-bold py-2 px-3 rounded-xl hover:text-[#2E251E] transition-all disabled:opacity-50", children: [_jsx(RotateCcw, { className: "w-3.5 h-3.5" }), " Clear"] })] }), outputGrid && (_jsxs("div", { className: "mt-3 bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-3 text-[10px] font-mono", children: [_jsxs("div", { className: "flex justify-between items-center", children: [_jsx("span", { className: "text-[#2E251E] font-bold", children: filters[activeFilter].name }), _jsxs("span", { className: "text-[#6E6257]", children: [OUTPUT_SIZE, "\u00D7", OUTPUT_SIZE, " output"] })] }), _jsxs("div", { className: "flex justify-between mt-1", children: [_jsxs("span", { className: "text-[#6E6257]", children: ["Min: ", _jsx("span", { className: "text-[#B6532B] font-bold", children: stats.min.toFixed(1) })] }), _jsxs("span", { className: "text-[#6E6257]", children: ["Max: ", _jsx("span", { className: "text-[#B6532B] font-bold", children: stats.max.toFixed(1) })] })] }), enablePooling && pooledGrid && (_jsx("div", { className: "flex justify-between mt-1 pt-1 border-t border-[#E5DDD0]", children: _jsxs("span", { className: "text-[#6E6257]", children: ["Pooled: ", _jsxs("span", { className: "text-[#C18C3B] font-bold", children: [POOLED_SIZE, "\u00D7", POOLED_SIZE] })] }) }))] }))] }), _jsxs("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-sm flex flex-col items-center", children: [_jsxs("h5", { className: "text-xs font-mono font-bold text-[#2E251E] mb-3 tracking-wider flex items-center gap-1.5", children: [_jsx(Grid3x3, { className: "w-4 h-4 text-[#B6532B]" }), " Input Image (8\u00D78)"] }), _jsx("canvas", { ref: inputRef, onClick: handleInputClick, className: "cursor-pointer border-2 border-[#E5DDD0] rounded-lg" }), _jsx("p", { className: "text-[10px] text-[#6E6257] mt-2", children: "Click cells to toggle black/white" })] })] }), _jsxs("div", { className: "lg:col-span-6 space-y-6", children: [_jsxs("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-sm flex flex-col items-center", children: [_jsxs("h5", { className: "text-xs font-mono font-bold text-[#2E251E] mb-3 tracking-wider flex items-center gap-1.5", children: [_jsx(Cpu, { className: "w-4 h-4 text-[#B6532B]" }), " Feature Map (", OUTPUT_SIZE, "\u00D7", OUTPUT_SIZE, ") \u2014 ReLU"] }), _jsx("canvas", { ref: outputRef, className: "border-2 border-[#E5DDD0] rounded-lg" }), !outputGrid && !animating && (_jsx("p", { className: "text-[10px] text-[#B6532B] mt-2 font-bold", children: "Click \u00ABConvolve\u00BB" }))] }), enablePooling && (_jsxs("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-sm flex flex-col items-center", children: [_jsxs("h5", { className: "text-xs font-mono font-bold text-[#2E251E] mb-3 tracking-wider flex items-center gap-1.5", children: [_jsx(Minimize2, { className: "w-4 h-4 text-[#C18C3B]" }), " Max Pooled (", POOLED_SIZE, "\u00D7", POOLED_SIZE, ")"] }), _jsx("canvas", { ref: pooledRef, className: "border-2 border-[#E5DDD0] rounded-lg" })] })), _jsxs("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-sm flex flex-col items-center", children: [_jsxs("h5", { className: "text-xs font-mono font-bold text-[#2E251E] mb-3 tracking-wider", children: ["Kernel (3\u00D73) \u2014 ", filters[activeFilter].name] }), _jsx("canvas", { ref: kernelRef, className: "rounded-lg" })] })] })] }));
};
