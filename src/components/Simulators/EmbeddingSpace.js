import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useState, useEffect, useRef } from 'react';
import { Compass, RotateCcw, HelpCircle } from 'lucide-react';
export const EmbeddingSpace = () => {
    const words = [
        // Royalty cluster
        { word: 'King', x: 1.2, y: 1.2, category: 'royalty' },
        { word: 'Queen', x: 1.15, y: 0.3, category: 'royalty' },
        { word: 'Man', x: 0.3, y: 1.2, category: 'royalty' },
        { word: 'Woman', x: 0.25, y: 0.3, category: 'royalty' },
        { word: 'Prince', x: 1.4, y: 0.8, category: 'royalty' },
        // Tech cluster
        { word: 'Python', x: -0.8, y: -0.5, category: 'tech' },
        { word: 'Code', x: -1.1, y: -0.9, category: 'tech' },
        { word: 'AI', x: -0.6, y: -1.1, category: 'tech' },
        { word: 'GPU', x: -0.4, y: -0.7, category: 'tech' },
        { word: 'Data', x: -0.9, y: -0.3, category: 'tech' }
    ];
    // Vector math state: Word A - Word B + Word C = Target
    const [wordA, setWordA] = useState('King');
    const [wordB, setWordB] = useState('Man');
    const [wordC, setWordC] = useState('Woman');
    const [resultWord, setResultWord] = useState('Queen');
    const [resultCoord, setResultCoord] = useState({ x: 1.15, y: 0.3 });
    const canvasRef = useRef(null);
    // Run vector calculation: A - B + C
    const calculateVectorMath = () => {
        const vecA = words.find(w => w.word === wordA);
        const vecB = words.find(w => w.word === wordB);
        const vecC = words.find(w => w.word === wordC);
        if (!vecA || !vecB || !vecC)
            return;
        // A - B + C
        const targetX = vecA.x - vecB.x + vecC.x;
        const targetY = vecA.y - vecB.y + vecC.y;
        setResultCoord({ x: targetX, y: targetY });
        // Find nearest neighbor by Euclidean distance
        let nearest = words[0];
        let minDist = Infinity;
        words.forEach(w => {
            // Exclude operands from answers to make math meaningful
            if (w.word === wordA || w.word === wordB || w.word === wordC)
                return;
            const dx = w.x - targetX;
            const dy = w.y - targetY;
            const dist = dx * dx + dy * dy;
            if (dist < minDist) {
                minDist = dist;
                nearest = w;
            }
        });
        setResultWord(nearest.word);
    };
    useEffect(() => {
        calculateVectorMath();
    }, [wordA, wordB, wordC]);
    // Draw 2D Embedding Projection
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const ctx = canvas.getContext('2d');
        if (!ctx)
            return;
        const w = canvas.width = 400;
        const h = canvas.height = 320;
        // Draw background (Light Sepia grid)
        ctx.fillStyle = '#FAF6EE';
        ctx.fillRect(0, 0, w, h);
        // Grid details
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
        // Grid center axes
        ctx.strokeStyle = '#CFC5B4';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(w / 2, 0);
        ctx.lineTo(w / 2, h);
        ctx.moveTo(0, h / 2);
        ctx.lineTo(w, h / 2);
        ctx.stroke();
        // Map float coordinate to pixel: uniform scale to prevent mismatch
        const scale = Math.min(w / 4, h / 4);
        const offsetX = (w / scale) / 2;
        const offsetY = (h / scale) / 2;
        const toPixel = (x, y) => {
            const px = (x + offsetX) * scale;
            // invert y axis
            const py = h - (y + offsetY) * scale;
            return { x: px, y: py };
        };
        // Draw word points
        words.forEach(wd => {
            const p = toPixel(wd.x, wd.y);
            ctx.beginPath();
            ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
            ctx.fillStyle = wd.category === 'royalty' ? '#B6532B' : '#C18C3B'; // Terracotta vs Ochre
            ctx.fill();
            // Word text label
            ctx.fillStyle = '#2E251E'; // Deep charcoal text
            ctx.font = 'bold 10px monospace';
            ctx.fillText(wd.word, p.x + 8, p.y + 3);
        });
        // Draw Vector Math arrows (Step 1: Origin to A)
        const vecA = words.find(w => w.word === wordA);
        const vecB = words.find(w => w.word === wordB);
        const vecC = words.find(w => w.word === wordC);
        if (vecA && vecB && vecC) {
            const pZero = toPixel(0, 0);
            const pA = toPixel(vecA.x, vecA.y);
            const pAMinusB = toPixel(vecA.x - vecB.x, vecA.y - vecB.y);
            const pResult = toPixel(resultCoord.x, resultCoord.y);
            // Line A (Terracotta)
            ctx.beginPath();
            ctx.moveTo(pZero.x, pZero.y);
            ctx.lineTo(pA.x, pA.y);
            ctx.strokeStyle = '#B6532B';
            ctx.lineWidth = 2.5;
            ctx.stroke();
            // Line -B (Ochre - dashed)
            ctx.beginPath();
            ctx.moveTo(pA.x, pA.y);
            ctx.lineTo(pAMinusB.x, pAMinusB.y);
            ctx.strokeStyle = '#C18C3B';
            ctx.lineWidth = 1.8;
            ctx.setLineDash([4, 4]); // dashed for subtraction
            ctx.stroke();
            ctx.setLineDash([]); // reset
            // Line +C (Forest Green)
            ctx.beginPath();
            ctx.moveTo(pAMinusB.x, pAMinusB.y);
            ctx.lineTo(pResult.x, pResult.y);
            ctx.strokeStyle = '#3B7A57';
            ctx.lineWidth = 2.2;
            ctx.stroke();
            // Draw Result coordinates highlight ring
            ctx.beginPath();
            ctx.arc(pResult.x, pResult.y, 8, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(59, 122, 87, 0.15)';
            ctx.strokeStyle = '#3B7A57';
            ctx.lineWidth = 1.5;
            ctx.fill();
            ctx.stroke();
            // Highlight nearest result word
            const nearestVec = words.find(w => w.word === resultWord);
            if (nearestVec) {
                const pNearest = toPixel(nearestVec.x, nearestVec.y);
                ctx.beginPath();
                ctx.arc(pNearest.x, pNearest.y, 14, 0, Math.PI * 2);
                ctx.strokeStyle = '#3B7A57';
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        }
    }, [wordA, wordB, wordC, resultWord, resultCoord]);
    return (_jsxs("div", { className: "grid grid-cols-1 md:grid-cols-12 gap-6 p-4", children: [_jsxs("div", { className: "md:col-span-5 bg-[#F4EFE6] border border-[#E5DDD0] p-5 rounded-2xl flex flex-col justify-between", children: [_jsxs("div", { children: [_jsxs("h4", { className: "text-[#2E251E] font-bold text-base mb-1 tracking-wide flex items-center gap-2", children: [_jsx(Compass, { className: "w-5 h-5 text-[#B6532B]" }), " Vector Arithmetic"] }), _jsx("p", { className: "text-[#6E6257] text-xs mb-5", children: "Embeddings capture conceptual semantics using coordinate geometry." }), _jsxs("div", { className: "space-y-4 font-mono text-xs", children: [_jsxs("div", { children: [_jsx("label", { className: "text-[10px] text-[#6E6257] block mb-1.5 uppercase font-semibold", children: "Word A (Base Concept)" }), _jsx("select", { value: wordA, onChange: (e) => setWordA(e.target.value), className: "w-full bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl px-3 py-2 text-[#2E251E] focus:outline-none focus:border-[#B6532B] font-mono", children: words.map(w => (_jsx("option", { value: w.word, children: w.word }, w.word))) })] }), _jsxs("div", { className: "flex items-center gap-2", children: [_jsx("span", { className: "text-[#C18C3B] font-bold text-lg px-1", children: "-" }), _jsxs("div", { className: "flex-1", children: [_jsx("label", { className: "text-[10px] text-[#6E6257] block mb-1.5 uppercase font-semibold", children: "Word B (Subtract Attribute)" }), _jsx("select", { value: wordB, onChange: (e) => setWordB(e.target.value), className: "w-full bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl px-3 py-2 text-[#2E251E] focus:outline-none focus:border-[#B6532B] font-mono", children: words.map(w => (_jsx("option", { value: w.word, children: w.word }, w.word))) })] })] }), _jsxs("div", { className: "flex items-center gap-2", children: [_jsx("span", { className: "text-[#3B7A57] font-bold text-lg px-1", children: "+" }), _jsxs("div", { className: "flex-1", children: [_jsx("label", { className: "text-[10px] text-[#6E6257] block mb-1.5 uppercase font-semibold", children: "Word C (Add Attribute)" }), _jsx("select", { value: wordC, onChange: (e) => setWordC(e.target.value), className: "w-full bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl px-3 py-2 text-[#2E251E] focus:outline-none focus:border-[#B6532B] font-mono", children: words.map(w => (_jsx("option", { value: w.word, children: w.word }, w.word))) })] })] })] })] }), _jsxs("div", { className: "pt-5 border-t border-[#E5DDD0] space-y-3.5", children: [_jsxs("div", { className: "p-3.5 bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl space-y-2 text-center", children: [_jsx("span", { className: "text-[9px] text-[#6E6257] font-mono uppercase tracking-wider block", children: "Nearest Semantic Match" }), _jsxs("span", { className: "text-xl font-bold font-mono text-[#3B7A57] flex items-center justify-center gap-1.5", children: ["= ", resultWord] })] }), _jsxs("button", { onClick: () => {
                                    setWordA('King');
                                    setWordB('Man');
                                    setWordC('Woman');
                                }, className: "w-full py-2.5 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#FAF6EE]/80 text-[#6E6257] hover:text-[#2E251E] transition-colors flex items-center justify-center gap-2 text-xs font-semibold cursor-pointer font-mono", children: [_jsx(RotateCcw, { className: "w-4 h-4" }), " Reset Equation"] })] })] }), _jsx("div", { className: "md:col-span-7 flex flex-col items-center", children: _jsxs("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl w-full flex flex-col justify-center relative shadow-inner", children: [_jsx("div", { className: "relative overflow-hidden flex justify-center", children: _jsx("canvas", { ref: canvasRef, className: "rounded-xl border border-[#E5DDD0] w-full max-w-[400px] h-[320px]" }) }), _jsx("div", { className: "absolute top-7 left-7 bg-[#2E251E]/90 px-2.5 py-1 rounded-lg border border-[#E5DDD0]/20 text-[9px] font-mono text-[#FAF6EE] uppercase tracking-wide", children: "2D Latent Vector Projection (PCA)" }), _jsxs("div", { className: "mt-3 p-3 bg-[#F4EFE6] border border-[#E5DDD0] rounded-xl text-[10px] font-mono text-[#6E6257] flex gap-2", children: [_jsx(HelpCircle, { className: "w-4 h-4 text-[#B6532B] shrink-0" }), _jsxs("span", { children: ["Vectors are color-coded: ", _jsx("span", { className: "text-[#B6532B] font-semibold", children: "Royalty terms" }), " vs. ", _jsx("span", { className: "text-[#C18C3B] font-semibold", children: "Coding tech terms" }), ". Notice how subtraction shifts opposite and addition jumps clusters correctly!"] })] })] }) })] }));
};
