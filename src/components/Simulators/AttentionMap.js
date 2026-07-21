import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Sparkles, BookOpen, Type, Layers, Scale, Send } from 'lucide-react';
function hashString(str) {
    let hash = 5381;
    for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5) + hash) + str.charCodeAt(i);
        hash = hash & hash;
    }
    return hash >>> 0;
}
function mulberry32(seed) {
    let s = seed | 0;
    return () => {
        s = s + 0x6D2B79F5 | 0;
        let t = Math.imul(s ^ s >>> 15, 1 | s);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}
function softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => v / sum);
}
function interpolateColor(t) {
    const r = Math.round(255 - t * (255 - 182));
    const g = Math.round(255 - t * (255 - 83));
    const b = Math.round(255 - t * (255 - 43));
    return `rgb(${r},${g},${b})`;
}
function generateVectors(tokens, dim) {
    const genVec = (token, kind, d) => {
        const seed = hashString(token + "_" + kind);
        const rng = mulberry32(seed);
        return Array.from({ length: d }, () => (rng() - 0.5) * 2);
    };
    return {
        Q: tokens.map(t => genVec(t, "Q", dim)),
        K: tokens.map(t => genVec(t, "K", dim)),
        V: tokens.map(t => genVec(t, "V", dim))
    };
}
function computeAttentionMatrices(Q, K, numHeads, headDim, useScale) {
    const n = Q.length;
    const result = [];
    for (let h = 0; h < numHeads; h++) {
        const hStart = h * headDim;
        const matrix = [];
        for (let i = 0; i < n; i++) {
            const row = [];
            for (let j = 0; j < n; j++) {
                let dot = 0;
                for (let d = 0; d < headDim; d++) {
                    dot += Q[i][hStart + d] * K[j][hStart + d];
                }
                if (useScale)
                    dot /= Math.sqrt(headDim);
                row.push(dot);
            }
            matrix.push(softmax(row));
        }
        result.push(matrix);
    }
    return result;
}
function computeOutputs(attentionMatrices, V, numHeads, headDim, dim) {
    const n = V.length;
    const output = [];
    for (let i = 0; i < n; i++) {
        const vec = new Array(dim).fill(0);
        for (let h = 0; h < numHeads; h++) {
            const hStart = h * headDim;
            const attn = attentionMatrices[h];
            for (let j = 0; j < n; j++) {
                const w = attn[i][j];
                for (let d = 0; d < headDim; d++) {
                    vec[hStart + d] += w * V[j][hStart + d];
                }
            }
        }
        output.push(vec);
    }
    return output;
}
export const AttentionMap = () => {
    const presets = [
        "The cat sat on the mat",
        "Attention is all you need",
        "I love machine learning",
        "The dog chased the ball"
    ];
    const [sentence, setSentence] = useState(presets[0]);
    const [customInput, setCustomInput] = useState("");
    const [numHeads, setNumHeads] = useState(1);
    const [useScale, setUseScale] = useState(true);
    const [hoveredCell, setHoveredCell] = useState(null);
    const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
    const canvasRef = useRef(null);
    const dim = 8;
    const tokens = useMemo(() => {
        return sentence.trim().split(/\s+/).filter(Boolean);
    }, [sentence]);
    const headDim = dim / numHeads;
    const vectors = useMemo(() => generateVectors(tokens, dim), [tokens, dim]);
    const attentionMatrices = useMemo(() => computeAttentionMatrices(vectors.Q, vectors.K, numHeads, headDim, useScale), [vectors, numHeads, headDim, useScale]);
    const outputs = useMemo(() => computeOutputs(attentionMatrices, vectors.V, numHeads, headDim, dim), [attentionMatrices, vectors.V, numHeads, headDim, dim]);
    const isValidSentence = tokens.length >= 1 && tokens.length <= 10;
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const ctx = canvas.getContext('2d');
        if (!ctx)
            return;
        const cw = 400;
        const ch = 320;
        canvas.width = cw;
        canvas.height = ch;
        ctx.clearRect(0, 0, cw, ch);
        ctx.fillStyle = '#FAF6EE';
        ctx.fillRect(0, 0, cw, ch);
        const n = tokens.length;
        if (n === 0)
            return;
        const drawHeatmap = (matrix, tkns, ox, oy, w, h, label) => {
            const m = tkns.length;
            const lm = label ? 24 : 20;
            const tm = label ? 22 : 18;
            const gw = w - lm - 4;
            const gh = h - tm - 4;
            const cw = gw / m;
            const ch = gh / m;
            for (let i = 0; i < m; i++) {
                for (let j = 0; j < m; j++) {
                    const val = matrix[i][j];
                    const x = ox + lm + j * cw;
                    const y = oy + tm + i * ch;
                    ctx.fillStyle = interpolateColor(val);
                    ctx.fillRect(x, y, cw, ch);
                    ctx.strokeStyle = '#E5DDD0';
                    ctx.lineWidth = 0.5;
                    ctx.strokeRect(x, y, cw, ch);
                }
            }
            const fontSize = Math.max(6.5, Math.min(9, cw * 0.75));
            ctx.fillStyle = '#2E251E';
            ctx.font = `bold ${fontSize}px monospace`;
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            for (let i = 0; i < m; i++) {
                ctx.fillText(tkns[i].slice(0, 4), ox + lm - 3, oy + tm + i * ch + ch / 2);
            }
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';
            for (let j = 0; j < m; j++) {
                ctx.fillText(tkns[j].slice(0, 4), ox + lm + j * cw + cw / 2, oy + tm - 3);
            }
            if (label) {
                ctx.fillStyle = '#6E6257';
                ctx.font = 'bold 7px monospace';
                ctx.textAlign = 'left';
                ctx.textBaseline = 'top';
                ctx.fillText(label, ox + 3, oy + 3);
            }
        };
        if (numHeads === 1) {
            drawHeatmap(attentionMatrices[0], tokens, 0, 0, cw, ch, "");
        }
        else {
            const hw = cw / 2;
            const hh = ch / 2;
            for (let h = 0; h < 4; h++) {
                const col = h % 2;
                const row = Math.floor(h / 2);
                drawHeatmap(attentionMatrices[h], tokens, col * hw, row * hh, hw, hh, `H${h + 1}`);
            }
        }
    }, [attentionMatrices, tokens, numHeads]);
    const getCellFromPos = (cx, cy) => {
        const n = tokens.length;
        if (n === 0)
            return null;
        const cw = 400;
        const ch = 320;
        if (numHeads === 1) {
            const lm = 20;
            const tm = 18;
            const gw = cw - lm - 4;
            const gh = ch - tm - 4;
            const cellW = gw / n;
            const cellH = gh / n;
            const j = Math.floor((cx - lm) / cellW);
            const i = Math.floor((cy - tm) / cellH);
            if (i < 0 || i >= n || j < 0 || j >= n)
                return null;
            return { head: 1, row: i, col: j, value: attentionMatrices[0][i][j] };
        }
        const hw = cw / 2;
        const hh = ch / 2;
        const hx = Math.floor(cx / hw);
        const hy = Math.floor(cy / hh);
        if (hx < 0 || hx > 1 || hy < 0 || hy > 1)
            return null;
        const headIdx = hy * 2 + hx;
        const ox = hx * hw;
        const oy = hy * hh;
        const lm = 24;
        const tm = 22;
        const gw = hw - lm - 4;
        const gh = hh - tm - 4;
        const cellW = gw / n;
        const cellH = gh / n;
        const j = Math.floor((cx - ox - lm) / cellW);
        const i = Math.floor((cy - oy - tm) / cellH);
        if (i < 0 || i >= n || j < 0 || j >= n)
            return null;
        return { head: headIdx + 1, row: i, col: j, value: attentionMatrices[headIdx][i][j] };
    };
    const handleCanvasMouseMove = (e) => {
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const rect = canvas.getBoundingClientRect();
        const sx = canvas.width / rect.width;
        const sy = canvas.height / rect.height;
        const cx = (e.clientX - rect.left) * sx;
        const cy = (e.clientY - rect.top) * sy;
        const cell = getCellFromPos(cx, cy);
        if (cell) {
            setHoveredCell(cell);
            setTooltipPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
        }
        else {
            setHoveredCell(null);
        }
    };
    const handleCanvasMouseLeave = () => {
        setHoveredCell(null);
    };
    const handleCustomSubmit = (e) => {
        e.preventDefault();
        const trimmed = customInput.trim();
        if (trimmed.length > 0) {
            const words = trimmed.split(/\s+/);
            if (words.length >= 1 && words.length <= 10) {
                setSentence(trimmed);
            }
        }
    };
    const handleCustomKeyDown = (e) => {
        if (e.key === 'Enter') {
            handleCustomSubmit(e);
        }
    };
    return (_jsxs("div", { className: "grid grid-cols-1 lg:grid-cols-12 gap-6 p-6 h-full bg-[#FAF6EE] overflow-y-auto", children: [_jsxs("div", { className: "lg:col-span-5 bg-[#F4EFE6] border border-[#E5DDD0] p-5 rounded-2xl flex flex-col justify-between", children: [_jsxs("div", { children: [_jsxs("h4", { className: "text-[#2E251E] font-bold text-base mb-1 tracking-wide flex items-center gap-2", children: [_jsx(Sparkles, { className: "w-5 h-5 text-[#B6532B]" }), " Scaled Dot-Product Attention"] }), _jsx("p", { className: "text-[#6E6257] text-xs mb-5", children: "Real Q\u00B7K matching with softmax normalization." }), _jsxs("div", { className: "space-y-4", children: [_jsxs("div", { children: [_jsxs("label", { className: "text-[10px] text-[#6E6257] font-mono block mb-1.5 uppercase font-bold", children: [_jsx(BookOpen, { className: "w-3 h-3 inline mr-1" }), " Preset Sentences"] }), _jsx("div", { className: "space-y-1.5", children: presets.map((sent, idx) => (_jsx("button", { onClick: () => { setSentence(sent); setCustomInput(""); }, className: `w-full text-left p-2.5 rounded-xl border text-xs font-mono transition-all leading-normal cursor-pointer ${sentence === sent && customInput === ""
                                                        ? 'border-[#B6532B] bg-[#B6532B]/10 text-[#2E251E] font-bold'
                                                        : 'border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257] hover:text-[#2E251E] hover:bg-[#F4EFE6]'}`, children: sent }, idx))) })] }), _jsxs("div", { children: [_jsxs("label", { className: "text-[10px] text-[#6E6257] font-mono block mb-1.5 uppercase font-bold", children: [_jsx(Type, { className: "w-3 h-3 inline mr-1" }), " Custom Sentence"] }), _jsxs("form", { onSubmit: handleCustomSubmit, className: "flex gap-2", children: [_jsx("input", { type: "text", value: customInput, onChange: e => setCustomInput(e.target.value), onKeyDown: handleCustomKeyDown, placeholder: "Type 1-10 words...", className: "flex-1 bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl px-3 py-2 text-xs font-mono text-[#2E251E] placeholder:text-[#CFC5B4] focus:outline-none focus:border-[#B6532B]" }), _jsx("button", { type: "submit", className: "px-3 py-2 bg-[#B6532B] text-white rounded-xl hover:bg-[#9E4521] transition-colors cursor-pointer", children: _jsx(Send, { className: "w-4 h-4" }) })] }), !isValidSentence && customInput.trim().length > 0 && (_jsx("p", { className: "text-[#B6532B] text-[10px] font-mono mt-1", children: "Enter 1 to 10 words" }))] }), _jsxs("div", { children: [_jsxs("label", { className: "text-[10px] text-[#6E6257] font-mono block mb-1.5 uppercase font-bold", children: [_jsx(Layers, { className: "w-3 h-3 inline mr-1" }), " Attention Heads"] }), _jsxs("div", { className: "grid grid-cols-2 gap-2", children: [_jsx("button", { onClick: () => setNumHeads(1), className: `py-2 rounded-xl text-xs font-mono border transition-all cursor-pointer ${numHeads === 1
                                                            ? 'border-[#B6532B] bg-[#B6532B]/15 text-[#B6532B] font-semibold'
                                                            : 'border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257]'}`, children: "Single Head" }), _jsx("button", { onClick: () => setNumHeads(4), className: `py-2 rounded-xl text-xs font-mono border transition-all cursor-pointer ${numHeads === 4
                                                            ? 'border-[#B6532B] bg-[#B6532B]/15 text-[#B6532B] font-semibold'
                                                            : 'border-[#E5DDD0] bg-[#FAF6EE] text-[#6E6257]'}`, children: "4 Heads" })] })] }), _jsxs("div", { children: [_jsxs("label", { className: "text-[10px] text-[#6E6257] font-mono block mb-1.5 uppercase font-bold", children: [_jsx(Scale, { className: "w-3 h-3 inline mr-1" }), " Scaling"] }), _jsxs("label", { className: "flex items-center gap-2.5 p-2.5 bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl cursor-pointer", children: [_jsx("input", { type: "checkbox", checked: useScale, onChange: e => setUseScale(e.target.checked), className: "accent-[#B6532B] w-4 h-4" }), _jsxs("span", { className: "text-xs font-mono text-[#2E251E]", children: ["Apply ", "\u221A", "d", _jsx("sub", { children: "k" }), " scaling"] })] }), _jsx("p", { className: "text-[10px] text-[#6E6257] font-mono mt-1 ml-1", children: useScale
                                                    ? "Dividing by \u221Ad\u2096 prevents softmax saturation"
                                                    : "Without scaling, logits grow large and softmax becomes peaked" })] })] })] }), _jsxs("div", { className: "p-3 bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl flex gap-2.5 items-start text-xs font-mono leading-normal pt-4 mt-4 text-[#2E251E]", children: [_jsx(Sparkles, { className: "w-4 h-4 text-[#B6532B] shrink-0 mt-0.5" }), _jsxs("div", { children: [_jsx("span", { className: "font-bold block mb-0.5", children: "How it works:" }), _jsxs("span", { className: "text-[#6E6257] text-[11px]", children: ["Each token has random Q/K/V vectors (dim=", dim, ") seeded by a hash of the word. Attention = softmax(Q\u00B7K\u1D40 / ", "\u221A", "d", _jsx("sub", { children: "k" }), "). Same sentence always produces the same matrix."] })] })] })] }), _jsxs("div", { className: "lg:col-span-7 space-y-5 flex flex-col", children: [_jsxs("div", { className: "bg-[#F4EFE6] border border-[#E5DDD0] p-4 rounded-2xl flex flex-col items-center shadow-inner relative", children: [_jsxs("div", { className: "relative inline-block", children: [_jsx("canvas", { ref: canvasRef, className: "rounded-xl border border-[#E5DDD0] bg-white", style: { width: 400, height: 320, cursor: hoveredCell ? 'crosshair' : 'default' }, onMouseMove: handleCanvasMouseMove, onMouseLeave: handleCanvasMouseLeave }), hoveredCell && (_jsxs("div", { className: "absolute pointer-events-none bg-[#2E251E] text-[#FAF6EE] text-[10px] font-mono px-2 py-1 rounded-lg border border-[#E5DDD0]/20 whitespace-nowrap z-10", style: {
                                            left: tooltipPos.x + 12,
                                            top: tooltipPos.y - 20,
                                        }, children: [numHeads > 1 && _jsxs(_Fragment, { children: ["H", hoveredCell.head, ": "] }), "\u201C", tokens[hoveredCell.row], "\u201D \u2192 \u201C", tokens[hoveredCell.col], "\u201D = ", hoveredCell.value.toFixed(4)] }))] }), _jsx("div", { className: "mt-2 text-[9px] font-mono text-[#6E6257]", children: "Hover over a cell to see the attention weight" })] }), _jsxs("div", { className: "bg-[#F4EFE6] border border-[#E5DDD0] p-4 rounded-2xl", children: [_jsxs("h5", { className: "text-[#2E251E] text-xs font-semibold font-mono mb-3 flex items-center gap-1.5", children: [_jsx(BookOpen, { className: "w-4 h-4 text-[#B6532B]" }), " Attention Output (first 3 of ", dim, " dims)"] }), _jsx("div", { className: "overflow-x-auto", children: _jsxs("table", { className: "w-full text-[10px] font-mono border-collapse", children: [_jsx("thead", { children: _jsxs("tr", { className: "border-b border-[#E5DDD0] text-[#6E6257]", children: [_jsx("th", { className: "text-left py-1.5 pr-3 font-semibold", children: "Token" }), _jsx("th", { className: "text-right py-1.5 px-2 font-semibold", children: "v[0]" }), _jsx("th", { className: "text-right py-1.5 px-2 font-semibold", children: "v[1]" }), _jsx("th", { className: "text-right py-1.5 px-2 font-semibold", children: "v[2]" })] }) }), _jsx("tbody", { children: outputs.map((vec, i) => (_jsxs("tr", { className: "border-b border-[#E5DDD0]/50 text-[#2E251E]", children: [_jsx("td", { className: "py-1.5 pr-3 font-bold truncate max-w-[80px]", children: tokens[i] }), _jsx("td", { className: "text-right py-1.5 px-2 text-[#B6532B]", children: vec[0].toFixed(4) }), _jsx("td", { className: "text-right py-1.5 px-2", children: vec[1].toFixed(4) }), _jsx("td", { className: "text-right py-1.5 px-2 text-[#6E6257]", children: vec[2].toFixed(4) })] }, i))) })] }) })] })] })] }));
};
