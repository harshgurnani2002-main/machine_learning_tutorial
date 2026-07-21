import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useState, useEffect, useRef } from 'react';
import { useApp } from '../context/AppContext';
import { Play, Terminal, AlertCircle, CheckCircle, Lightbulb, BookOpen, Code2 } from 'lucide-react';
export const CodingPlayground = () => {
    const { activeModule, activeCodingStage, setActiveCodingStage, completeCodingStage, userStats } = useApp();
    const activeStage = activeModule.coding[activeCodingStage];
    const hasAllStages = Boolean(activeModule.coding?.tutorial &&
        activeModule.coding?.project &&
        activeModule.coding?.assignment);
    const [code, setCode] = useState('');
    const [terminalLogs, setTerminalLogs] = useState([]);
    const [status, setStatus] = useState('idle');
    const [hintIdx, setHintIdx] = useState(-1);
    const [feedback, setFeedback] = useState('');
    const [showSolution, setShowSolution] = useState(false);
    const [showPseudo, setShowPseudo] = useState(true);
    const [isDescExpanded, setIsDescExpanded] = useState(true);
    const plotCanvasRef = useRef(null);
    const terminalRef = useRef(null);
    // Set code when stage or module changes
    useEffect(() => {
        if (activeStage) {
            const initialCode = activeStage.starterCode;
            setCode(initialCode); // eslint-disable-line react-hooks/set-state-in-effect
            setTerminalLogs([
                `[env] Python 3.11 sandbox initialized`,
                `[env] Libraries: NumPy, Pandas, Scikit-Learn, Matplotlib`,
                `[ready] ${activeStage.title}`
            ]);
            setStatus('idle');
            setHintIdx(-1);
            setFeedback('');
            setShowSolution(false);
            setShowPseudo(true);
        }
    }, [activeModule.id, activeCodingStage, activeStage]);
    // Auto-scroll terminal
    useEffect(() => {
        if (terminalRef.current) {
            terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
        }
    }, [terminalLogs]);
    // Render visualization canvas
    useEffect(() => {
        const canvas = plotCanvasRef.current;
        if (!canvas)
            return;
        const ctx = canvas.getContext('2d');
        if (!ctx)
            return;
        const w = canvas.width = 300;
        const h = canvas.height = 200;
        const grad = ctx.createLinearGradient(0, 0, 0, h);
        grad.addColorStop(0, '#2E251E');
        grad.addColorStop(1, '#3A3027');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.lineWidth = 1;
        for (let x = 0; x < w; x += 30) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }
        for (let y = 0; y < h; y += 30) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }
        const isSuccess = status === 'success';
        const isCompiling = status === 'compiling';
        if (activeModule.id === 'linear-regression') {
            const pts = [{ x: 50, y: 150 }, { x: 100, y: 120 }, { x: 150, y: 90 }, { x: 210, y: 60 }, { x: 250, y: 40 }];
            pts.forEach(p => {
                ctx.beginPath();
                ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
                ctx.fillStyle = '#C18C3B';
                ctx.shadowBlur = 10;
                ctx.shadowColor = '#C18C3B';
                ctx.fill();
                ctx.shadowBlur = 0;
            });
            ctx.beginPath();
            ctx.strokeStyle = isSuccess ? '#3B7A57' : isCompiling ? '#C18C3B' : '#B6532B';
            ctx.lineWidth = 3;
            if (isSuccess) {
                ctx.moveTo(30, 165);
                ctx.lineTo(270, 30);
            }
            else {
                ctx.moveTo(30, 110);
                ctx.lineTo(270, 110);
            }
            ctx.stroke();
            ctx.fillStyle = 'rgba(255,255,255,0.5)';
            ctx.font = '10px monospace';
            ctx.fillText('MSE', 10, 20);
            ctx.fillText('Fit \u2192', w - 45, h - 10);
        }
        else if (activeModule.id === 'logistic-regression' || activeModule.id === 'activation-functions') {
            ctx.beginPath();
            ctx.strokeStyle = isSuccess ? '#3B7A57' : '#CFC5B4';
            ctx.lineWidth = 3;
            ctx.lineJoin = 'round';
            for (let x = 0; x < w; x++) {
                const z = ((x / w) - 0.5) * 10;
                const sig = 1 / (1 + Math.exp(-z));
                const py = h - sig * (h - 40) - 20;
                if (x === 0)
                    ctx.moveTo(x, py);
                else
                    ctx.lineTo(x, py);
            }
            ctx.stroke();
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(0, 20);
            ctx.lineTo(w, 20);
            ctx.moveTo(0, h - 20);
            ctx.lineTo(w, h - 20);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = 'rgba(255,255,255,0.5)';
            ctx.font = '10px monospace';
            ctx.fillText('\u03c3(z)', 10, 15);
        }
        else if (activeModule.id === 'kmeans-clustering' || activeModule.id === 'dbscan-clustering') {
            const clusters = [{ x: 80, y: 70, col: '#B6532B' }, { x: 220, y: 65, col: '#C18C3B' }, { x: 150, y: 140, col: '#3B7A57' }];
            clusters.forEach((c) => {
                for (let i = 0; i < 8; i++) {
                    const a = (i / 8) * Math.PI * 2;
                    const r = 20 + Math.sin(i * 1.5) * 10;
                    ctx.beginPath();
                    ctx.arc(c.x + Math.cos(a) * r, c.y + Math.sin(a) * r, 4, 0, Math.PI * 2);
                    ctx.fillStyle = c.col + 'AA';
                    ctx.fill();
                }
                ctx.beginPath();
                ctx.arc(c.x, c.y, 8, 0, Math.PI * 2);
                ctx.fillStyle = c.col;
                ctx.shadowBlur = 15;
                ctx.shadowColor = c.col;
                ctx.fill();
                ctx.shadowBlur = 0;
                if (isSuccess) {
                    ctx.beginPath();
                    ctx.arc(c.x, c.y, 35, 0, Math.PI * 2);
                    ctx.strokeStyle = c.col + '88';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }
            });
        }
        else {
            ctx.beginPath();
            ctx.strokeStyle = isSuccess ? '#3B7A57' : isCompiling ? '#C18C3B' : '#B6532B';
            ctx.lineWidth = 3;
            ctx.lineJoin = 'round';
            for (let x = 0; x < w; x += 3) {
                const progress = x / w;
                let yVal;
                if (isSuccess) {
                    yVal = h - 30 - (h - 60) * Math.exp(-progress * 6);
                }
                else if (isCompiling) {
                    yVal = h / 2 + 25 * Math.sin(progress * 20);
                }
                else {
                    yVal = h - 120 + 40 * Math.sin(progress * 12) * Math.exp(-progress * 1.5);
                }
                if (x === 0)
                    ctx.moveTo(x, yVal);
                else
                    ctx.lineTo(x, yVal);
            }
            ctx.stroke();
            ctx.fillStyle = 'rgba(255,255,255,0.5)';
            ctx.font = '10px monospace';
            ctx.fillText('LOSS', 10, 20);
            ctx.fillText('EPOCHS \u2192', w - 70, h - 10);
        }
    }, [activeModule.id, status]);
    if (!hasAllStages || !activeStage) {
        return (_jsx("div", { className: "p-6 bg-[#FAF6EE] min-h-screen", children: _jsxs("div", { className: "max-w-2xl mx-auto bg-[#FAF6EE] border border-[#E5DDD0] rounded-2xl p-6 text-[#6E6257]", children: [_jsx("h3", { className: "text-lg font-bold text-[#2E251E] mb-2", children: "Coding Sandbox Data Missing" }), _jsx("p", { className: "text-sm", children: "This module is missing one or more coding stages (`tutorial`, `project`, `assignment`)." })] }) }));
    }
    // Run simulated script compiler
    const runCode = () => {
        const executionCode = showSolution && activeStage.solution ? activeStage.solution : code;
        setStatus('compiling');
        setTerminalLogs(prev => [...prev, ``, `>>> python main.py`]);
        setFeedback('');
        setTimeout(() => {
            let isCorrect = false;
            let output = '';
            let hasTodo = false;
            let todoLine = -1;
            let syntaxError = '';
            const lines = executionCode.split('\n');
            lines.forEach((line, i) => {
                const trimmed = line.trim();
                if (trimmed.startsWith('#'))
                    return;
                if (!hasTodo && (line.toUpperCase().includes('TODO') || line.includes('placeholder') || trimmed === 'pass' || trimmed === '...')) {
                    hasTodo = true;
                    todoLine = i + 1;
                }
                const blockStarts = ['def ', 'for ', 'while ', 'class '];
                if (!syntaxError) {
                    if (blockStarts.some(bs => trimmed.startsWith(bs)) && !trimmed.endsWith(':') && !trimmed.endsWith('(') && !trimmed.endsWith('\\') && !trimmed.endsWith(',')) {
                        syntaxError = `File "main.py", line ${i + 1}\nSyntaxError: expected ':'`;
                    }
                    if (['else', 'try', 'finally'].includes(trimmed)) {
                        syntaxError = `File "main.py", line ${i + 1}\nSyntaxError: expected ':'`;
                    }
                }
            });
            const missingKeywords = activeStage.testKeywords.filter(kw => !executionCode.includes(kw));
            if (syntaxError) {
                output = syntaxError;
            }
            else if (hasTodo) {
                output = `File "main.py", line ${todoLine}\nNotImplementedError: Unresolved placeholder detected.\nPlease implement the required logic and remove "TODO", "pass", or "..." before running.`;
            }
            else if (missingKeywords.length > 0) {
                const missing = missingKeywords.map(kw => `"${kw}"`).join(', ');
                output = `AssertionError: Validation failed.\nMissing required concepts/keywords: ${missing}.\nHint: Check the pseudocode.`;
            }
            else {
                isCorrect = true;
                output = activeStage.expectedOutput;
            }
            setTerminalLogs(prev => [...prev, output]);
            if (isCorrect) {
                setStatus('success');
                completeCodingStage(activeModule.id, activeCodingStage);
                setFeedback(`✓ All tests passed! Your implementation of "${activeStage.title}" is correct. Key concepts used: ${activeStage.testKeywords.join(', ')}.`);
                setTerminalLogs(prev => [...prev, ``, `[✓] Tests passed: ${activeStage.testKeywords.length}/${activeStage.testKeywords.length}`]);
            }
            else {
                setStatus('failed');
                setFeedback(`Review the pseudocode and ensure your implementation uses: ${activeStage.testKeywords.join(', ')}. Remove all TODO placeholders before running.`);
                setTerminalLogs(prev => [...prev, ``, `[✗] Tests failed`]);
            }
        }, 1000);
    };
    const stagesList = [
        { id: 'tutorial', label: 'Tutorial', icon: '📖' },
        { id: 'project', label: 'Project', icon: '🔬' },
        { id: 'assignment', label: 'Challenge', icon: '🏆' }
    ];
    const stageCompleted = (stageId) => {
        return userStats.completedCodingStages[activeModule.id]?.includes(stageId);
    };
    return (_jsx("div", { className: "bg-[#FAF6EE] p-3 sm:p-4 md:p-5", children: _jsxs("div", { className: "grid grid-cols-1 xl:grid-cols-12 gap-5", children: [_jsxs("aside", { className: "xl:col-span-3 bg-[#FAF6EE] border border-[#E5DDD0] rounded-2xl p-4 md:p-5 space-y-4", children: [_jsxs("div", { children: [_jsx("p", { className: "text-[10px] uppercase font-mono font-bold tracking-widest text-[#B6532B]", children: "Coding Sandbox" }), _jsx("h3", { className: "text-[#2E251E] font-bold text-base mt-1 leading-tight", children: activeStage.title })] }), _jsx("div", { className: "grid grid-cols-3 xl:grid-cols-1 gap-2", children: stagesList.map((stg) => {
                                const isActive = activeCodingStage === stg.id;
                                const isDone = stageCompleted(stg.id);
                                return (_jsxs("button", { onClick: () => setActiveCodingStage(stg.id), className: `w-full text-center xl:text-left px-2 sm:px-3 py-2 sm:py-2.5 rounded-xl border text-[10px] sm:text-xs md:text-sm flex flex-col sm:flex-row xl:flex-row items-center justify-between gap-1 transition-all cursor-pointer ${isActive
                                        ? 'bg-[#F4EFE6] border-[#B6532B] text-[#2E251E]'
                                        : 'bg-[#FAF6EE] border-[#E5DDD0] text-[#6E6257] hover:bg-[#F4EFE6]'}`, children: [_jsxs("span", { className: "flex items-center gap-1.5 sm:gap-2", children: [_jsx("span", { children: stg.icon }), _jsx("span", { className: "font-semibold hidden sm:inline", children: stg.label }), _jsxs("span", { className: "font-semibold inline sm:hidden", children: [stg.label.substring(0, 4), "."] })] }), _jsx("span", { className: `text-[9px] sm:text-xs font-mono font-medium ${isDone ? 'text-[#3B7A57]' : 'text-[#CFC5B4]'}`, children: isDone ? 'Done' : 'Open' })] }, stg.id));
                            }) }), _jsxs("div", { className: "border border-[#E5DDD0] rounded-xl bg-[#F4EFE6] p-3", children: [_jsxs("button", { onClick: () => setIsDescExpanded(!isDescExpanded), className: "w-full flex justify-between items-center text-[10px] uppercase font-mono font-bold tracking-wider text-[#B6532B] cursor-pointer outline-none", children: [_jsx("span", { children: "Problem Statement" }), _jsx("span", { className: "text-[10px] text-[#6E6257] font-semibold lowercase font-sans", children: isDescExpanded ? '[hide]' : '[show]' })] }), isDescExpanded && (_jsx("p", { className: "text-[#2E251E] text-xs leading-relaxed whitespace-pre-line mt-2 transition-all", children: activeStage.description }))] }), _jsxs("div", { className: "grid grid-cols-2 xl:grid-cols-1 gap-2", children: [_jsxs("button", { onClick: () => setHintIdx(prev => (prev + 1) % activeStage.hints.length), className: "px-3 py-2 sm:py-2.5 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#F4EFE6] text-[#2E251E] text-xs sm:text-sm flex items-center justify-center gap-2 cursor-pointer transition-colors", children: [_jsx(Lightbulb, { className: "w-3.5 h-3.5 text-[#C18C3B]" }), "Hint"] }), _jsxs("button", { onClick: () => setShowSolution(!showSolution), className: "px-3 py-2 sm:py-2.5 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#F4EFE6] text-[#2E251E] text-xs sm:text-sm flex items-center justify-center gap-2 cursor-pointer transition-colors", children: [_jsx(BookOpen, { className: "w-3.5 h-3.5 text-[#B6532B]" }), showSolution ? 'Hide Solution' : 'View Solution'] })] }), hintIdx >= 0 && (_jsxs("div", { className: "border border-[#E7D8B4] bg-[#F9F3E5] rounded-xl p-3", children: [_jsxs("p", { className: "text-[10px] uppercase font-mono font-bold tracking-wider text-[#6E4F1F]", children: ["Hint ", hintIdx + 1] }), _jsx("p", { className: "text-xs text-[#6E4F1F] mt-1", children: activeStage.hints[hintIdx] })] }))] }), _jsxs("div", { className: "xl:col-span-6 flex flex-col gap-4", children: [_jsxs("main", { className: "bg-[#2E251E] border border-[#4A3D31] rounded-2xl overflow-hidden min-h-[380px] sm:min-h-[500px] xl:min-h-[620px] flex flex-col", children: [_jsxs("div", { className: "px-4 py-3 bg-[#3A3027] border-b border-[#4A3D31] flex items-center justify-between", children: [_jsxs("div", { className: "flex items-center gap-3", children: [_jsx(Code2, { className: "w-4 h-4 text-[#C18C3B]" }), _jsx("span", { className: "text-xs font-mono text-[#E5DDD0]", children: "main.py" }), showSolution && (_jsx("span", { className: "text-[10px] px-2 py-0.5 rounded bg-[#3B7A57]/20 text-[#B7D8C4] font-mono", children: "SOLUTION" }))] }), _jsxs("button", { onClick: runCode, disabled: status === 'compiling', className: "px-4 py-2 rounded-lg bg-[#B6532B] hover:bg-[#9F4825] disabled:opacity-50 text-white text-xs font-bold flex items-center gap-2 cursor-pointer transition-colors", children: [_jsx(Play, { className: "w-4 h-4" }), status === 'compiling' ? 'Running...' : 'Run Code'] })] }), _jsx("div", { className: "flex-1 p-3", children: _jsx("textarea", { value: showSolution && activeStage.solution ? activeStage.solution : code, onChange: (e) => {
                                            if (!showSolution)
                                                setCode(e.target.value);
                                        }, readOnly: showSolution, spellCheck: false, className: "w-full h-full min-h-[280px] sm:min-h-[400px] xl:min-h-[520px] p-4 bg-[#2A211A] border border-[#4A3D31] rounded-xl text-[#F4EFE6] font-mono text-xs sm:text-sm leading-relaxed resize-none outline-none" }) })] }), feedback && (_jsxs("div", { className: `p-4 rounded-xl border flex items-center gap-3 text-xs sm:text-sm ${status === 'success'
                                ? 'border-[#B7D8C4] bg-[#E8F3EC] text-[#2A5D42]'
                                : 'border-[#E7D8B4] bg-[#F9F3E5] text-[#6E4F1F]'}`, children: [status === 'success' ? (_jsx(CheckCircle, { className: "w-5 h-5 text-[#3B7A57] shrink-0" })) : (_jsx(AlertCircle, { className: "w-5 h-5 text-[#C18C3B] shrink-0" })), _jsx("p", { children: feedback })] }))] }), _jsxs("section", { className: "xl:col-span-3 space-y-5", children: [_jsxs("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] rounded-2xl p-4", children: [_jsx("p", { className: "text-[10px] uppercase font-mono font-bold tracking-wider text-[#B6532B] mb-2", children: "Output Preview" }), _jsx("div", { className: "rounded-xl overflow-hidden bg-[#2E251E] border border-[#4A3D31] aspect-video", children: _jsx("canvas", { ref: plotCanvasRef, className: "w-full h-full object-cover" }) }), _jsx("p", { className: "text-[10px] text-[#6E6257] font-mono text-center mt-2", children: status === 'success' ? 'Executed successfully' : status === 'compiling' ? 'Computing...' : 'Ready' })] }), _jsxs("div", { className: "bg-[#2E251E] border border-[#4A3D31] rounded-2xl overflow-hidden min-h-[200px] sm:min-h-[240px] flex flex-col", children: [_jsxs("div", { className: "bg-[#3A3027] border-b border-[#4A3D31] px-4 py-2 text-[#E5DDD0] text-xs font-mono font-bold uppercase tracking-wider flex items-center gap-2", children: [_jsx(Terminal, { className: "w-4 h-4 text-[#C18C3B]" }), "Terminal"] }), _jsx("div", { ref: terminalRef, className: "flex-1 p-3 text-[10px] sm:text-xs font-mono overflow-y-auto space-y-1.5 text-[#E5DDD0]", children: terminalLogs.map((log, idx) => (_jsx("div", { className: `whitespace-pre-wrap break-words ${log.startsWith('Error') || log.startsWith('Syntax') || log.startsWith('Assertion') || log.startsWith('File') || log.startsWith('NotImplementedError') || log.includes('[✗]')
                                            ? 'text-[#D9534F]'
                                            : log.includes('[✓]') || log.includes('passed')
                                                ? 'text-[#3B7A57]'
                                                : log.startsWith('>>>')
                                                    ? 'text-[#C18C3B]'
                                                    : 'text-[#CFC5B4]'}`, children: log }, idx))) })] })] }), activeStage.pseudoCode && showPseudo && (_jsxs("section", { className: "xl:col-span-12 bg-[#2E251E] border border-[#4A3D31] rounded-2xl p-4", children: [_jsxs("div", { className: "flex items-center justify-between mb-2", children: [_jsx("p", { className: "text-[10px] uppercase font-mono font-bold tracking-wider text-[#C18C3B]", children: "Algorithm Pseudocode" }), _jsx("button", { onClick: () => setShowPseudo(false), className: "text-xs text-[#E5DDD0] hover:text-white font-mono cursor-pointer", children: "Hide" })] }), _jsx("pre", { className: "text-[#E5DDD0] text-[10px] sm:text-xs font-mono leading-relaxed whitespace-pre-wrap", children: activeStage.pseudoCode })] })), !showPseudo && (_jsx("section", { className: "xl:col-span-12", children: _jsx("button", { onClick: () => setShowPseudo(true), className: "px-3 py-2 rounded-lg border border-[#E5DDD0] bg-[#FAF6EE] text-[#2E251E] text-xs font-mono cursor-pointer hover:bg-[#F4EFE6]", children: "Show Pseudocode" }) }))] }) }));
};
