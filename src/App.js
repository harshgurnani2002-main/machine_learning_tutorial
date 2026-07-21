import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useMemo, useState } from 'react';
import { AppProvider, useApp } from './context/AppContext';
import { NeuralBackground } from './components/NeuralBackground';
import { CodingPlayground } from './components/CodingPlayground';
import { KaggleNotebook } from './components/KaggleNotebook';
import { QuizEngine } from './components/QuizEngine';
import { InterviewPrep } from './components/InterviewPrep';
import { ModuleSimulator } from './components/Simulators/ModuleSimulator';
import { modulesData } from './data/modules';
import { Sparkles, BookOpen, Activity, HelpCircle, Code, Award, RefreshCw, Compass, Scale, Eye, Brain, Settings, Menu, X, } from 'lucide-react';
import katex from 'katex';
/* eslint-disable no-control-regex */
const normalizeLatex = (raw) => raw
    .replace(/\u0007/g, '\\a')
    .replace(/\u0008/g, '\\b')
    .replace(/\u000c/g, '\\f')
    .replace(/\u000b/g, '\\v')
    .replace(/\r/g, '\\r')
    .replace(/\t/g, '\\t')
    .replace(/\u0000/g, '')
    .replace(/[\u0001-\u0006\u000e-\u001f]/g, '');
/* eslint-enable no-control-regex */
// Math theory notation parser and visual viewer
const TheoryView = ({ module }) => {
    const content = module.theory;
    const blocks = useMemo(() => {
        const lines = content.split('\n');
        const parsedBlocks = [];
        let inCode = false;
        let codeContent = '';
        let inMath = false;
        let mathContent = '';
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            // Code block detection
            if (line.trim().startsWith('```')) {
                if (inCode) {
                    parsedBlocks.push({ type: 'code', text: codeContent, codeLang: 'python' });
                    codeContent = '';
                    inCode = false;
                }
                else {
                    inCode = true;
                    codeContent = '';
                }
                continue;
            }
            if (inCode) {
                codeContent += line + '\n';
                continue;
            }
            // Math block detection
            if (line.trim().startsWith('$$')) {
                const trimmed = line.trim();
                if (trimmed.length > 2 && trimmed.endsWith('$$')) {
                    parsedBlocks.push({ type: 'math-block', text: trimmed.substring(2, trimmed.length - 2) });
                }
                else {
                    if (inMath) {
                        parsedBlocks.push({ type: 'math-block', text: mathContent });
                        mathContent = '';
                        inMath = false;
                    }
                    else {
                        inMath = true;
                        mathContent = '';
                    }
                }
                continue;
            }
            if (inMath) {
                mathContent += line + ' ';
                continue;
            }
            // Headers
            if (line.startsWith('### ')) {
                parsedBlocks.push({ type: 'h3', text: line.substring(4) });
                continue;
            }
            if (line.startsWith('#### ')) {
                parsedBlocks.push({ type: 'h4', text: line.substring(5) });
                continue;
            }
            // Lists
            if (line.trim().startsWith('* ') || line.trim().startsWith('- ')) {
                parsedBlocks.push({ type: 'list', text: line.trim().substring(2) });
                continue;
            }
            // Paragraphs
            if (line.trim() !== '') {
                parsedBlocks.push({ type: 'p', text: line });
            }
        }
        return parsedBlocks;
    }, [content]);
    // Inline styling for inline formulas ($ ... $) and bold text (** ... **)
    const formatText = (text) => {
        const formatted = [];
        const parts = text.split(/(\$[^$]+\$|\*\*[^*]+\*\*)/);
        parts.forEach((part, idx) => {
            if (part.startsWith('$') && part.endsWith('$')) {
                const math = normalizeLatex(part.substring(1, part.length - 1));
                try {
                    const html = katex.renderToString(math, { displayMode: false, throwOnError: false });
                    formatted.push(_jsx("span", { dangerouslySetInnerHTML: { __html: html } }, idx));
                }
                catch {
                    formatted.push(_jsx("span", { children: part }, idx));
                }
            }
            else if (part.startsWith('**') && part.endsWith('**')) {
                formatted.push(_jsx("strong", { className: "font-bold text-[#2E251E]", children: part.substring(2, part.length - 2) }, idx));
            }
            else {
                formatted.push(_jsx("span", { children: part }, idx));
            }
        });
        return formatted;
    };
    return (_jsxs("div", { className: "max-w-3xl mx-auto space-y-4", children: [_jsxs("div", { className: "bg-[#F4EFE6] border border-[#E5DDD0] px-5 py-3 rounded-xl flex items-center gap-3", children: [_jsx("span", { className: "text-lg", children: "\uD83D\uDCD6" }), _jsxs("div", { children: [_jsx("p", { className: "text-[10px] text-[#B6532B] font-mono font-bold uppercase tracking-wider", children: "Theory & Math \u2014 Absolute Beginner Friendly" }), _jsx("p", { className: "text-[9px] text-[#6E6257] font-mono mt-0.5", children: "No prior ML knowledge required. Read top to bottom for a complete understanding." })] })] }), _jsx("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] p-6 sm:p-8 rounded-2xl space-y-5 shadow-sm text-[#2E251E]", children: blocks.map((block, idx) => {
                    switch (block.type) {
                        case 'h3':
                            return (_jsx("h3", { className: "text-lg sm:text-xl font-bold text-[#2E251E] border-b border-[#E5DDD0] pb-1.5 pt-3 font-serif", children: block.text }, idx));
                        case 'h4':
                            return (_jsxs("h4", { className: "text-sm sm:text-base font-bold text-[#B6532B] pt-1 font-mono", children: ["\u25B8 ", block.text] }, idx));
                        case 'math-block': {
                            const html = katex.renderToString(normalizeLatex(block.text), { displayMode: true, throwOnError: false });
                            return (_jsxs("div", { className: "my-4 p-4 bg-[#F4EFE6] border-l-4 border-[#B6532B] rounded-r-xl overflow-x-auto", children: [_jsx("span", { className: "text-[9px] text-[#6E6257] font-mono uppercase block mb-1", children: "Mathematical Formula" }), _jsx("span", { className: "font-mono text-xs sm:text-sm text-[#B6532B] font-semibold whitespace-pre-wrap block", dangerouslySetInnerHTML: { __html: html } })] }, idx));
                        }
                        case 'code':
                            return (_jsx("pre", { className: "my-3 p-3.5 dark-sepia-editor rounded-xl overflow-x-auto text-[11px] leading-relaxed text-[#E5E0D8]", children: _jsx("code", { children: block.text }) }, idx));
                        case 'list':
                            return (_jsx("ul", { className: "list-none pl-0 text-xs text-[#2E251E] font-mono leading-relaxed", children: _jsxs("li", { className: "flex gap-2 items-start py-0.5", children: [_jsx("span", { className: "text-[#B6532B] shrink-0 mt-0.5", children: "\u2192" }), _jsx("span", { children: formatText(block.text) })] }) }, idx));
                        case 'p':
                        default:
                            return (_jsx("p", { className: "text-xs text-[#6E6257] leading-relaxed font-sans font-medium", children: formatText(block.text) }, idx));
                    }
                }) })] }));
};
function MainAppContent() {
    const { activeTab, setActiveTab, activeModuleId, setActiveModuleId, activeModule, userStats, resetProgress } = useApp();
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);
    const handleReset = () => {
        if (window.confirm("Are you sure you want to reset all your learning achievements and coding sandboxes? This cannot be undone.")) {
            resetProgress();
        }
    };
    const categories = useMemo(() => ['Foundations & Math', 'Supervised Learning', 'Unsupervised Learning', 'Deep Learning', 'Advanced & MLOps', 'Kaggle Real-World Projects'], []);
    const totalModules = modulesData.length;
    const isKaggleModule = activeModule.category === 'Kaggle Real-World Projects';
    const modulesByCategory = useMemo(() => {
        const groups = {};
        categories.forEach(cat => {
            groups[cat] = modulesData.filter(m => m.category === cat);
        });
        return groups;
    }, [categories]);
    useEffect(() => {
        if (isKaggleModule && activeTab !== 'theory' && activeTab !== 'sandbox') {
            setActiveTab('theory');
        }
    }, [isKaggleModule, activeTab, setActiveTab]);
    const getCategoryIcon = (category) => {
        switch (category) {
            case 'Foundations & Math':
                return _jsx(Compass, { className: "w-3.5 h-3.5 text-[#C18C3B]" });
            case 'Supervised Learning':
                return _jsx(Scale, { className: "w-3.5 h-3.5 text-[#B6532B]" });
            case 'Unsupervised Learning':
                return _jsx(Eye, { className: "w-3.5 h-3.5 text-[#6E6257]" });
            case 'Deep Learning':
                return _jsx(Brain, { className: "w-3.5 h-3.5 text-[#B6532B]" });
            case 'Advanced & MLOps':
                return _jsx(Settings, { className: "w-3.5 h-3.5 text-[#2E251E]" });
            case 'Kaggle Real-World Projects':
                return _jsx(BookOpen, { className: "w-3.5 h-3.5 text-[#B6532B]" });
            default:
                return _jsx(BookOpen, { className: "w-3.5 h-3.5 text-[#6E6257]" });
        }
    };
    const getModuleProgress = (moduleId) => {
        const codingStages = userStats.completedCodingStages[moduleId] || [];
        const quizDone = userStats.completedQuizzes.includes(moduleId) ? 1 : 0;
        const interviewDone = userStats.completedInterviews.includes(moduleId) ? 1 : 0;
        const totalCompleted = codingStages.length + quizDone + interviewDone;
        const module = modulesData.find(m => m.id === moduleId);
        const isKaggle = module?.category === 'Kaggle Real-World Projects';
        if (isKaggle) {
            const codingDone = codingStages.length;
            return {
                total: 2,
                completed: codingDone > 0 ? (codingDone === 3 ? 2 : 1) : 0,
                percentage: codingDone > 0 ? (codingDone === 3 ? 100 : 50) : 0,
                isCompleted: codingDone === 3,
                isInProgress: codingDone > 0 && codingDone < 3,
                isNotStarted: codingDone === 0,
                codingCount: codingDone,
                quizDone: false,
                interviewDone: false
            };
        }
        return {
            total: 5,
            completed: totalCompleted,
            percentage: (totalCompleted / 5) * 100,
            isCompleted: totalCompleted === 5,
            isInProgress: totalCompleted > 0 && totalCompleted < 5,
            isNotStarted: totalCompleted === 0,
            codingCount: codingStages.length,
            quizDone: quizDone === 1,
            interviewDone: interviewDone === 1
        };
    };
    const activeModuleProgress = getModuleProgress(activeModuleId);
    return (_jsxs("div", { className: "relative min-h-screen flex flex-col z-10 bg-[#FAF6EE] text-[#2E251E]", children: [_jsx(NeuralBackground, {}), _jsx("header", { className: "sticky top-0 z-40 w-full px-4 sm:px-6 py-3.5 bg-[#FAF6EE]/90 backdrop-blur-md border-b border-[#E5DDD0] shadow-sm", children: _jsxs("div", { className: "max-w-7xl mx-auto flex items-center justify-between gap-4", children: [_jsxs("div", { className: "flex items-center gap-3", children: [_jsx("button", { onClick: () => setIsSidebarOpen(true), className: "lg:hidden p-2 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#F4EFE6] text-[#6E6257] hover:text-[#2E251E] transition-all cursor-pointer flex items-center justify-center", title: "Open Syllabus", children: _jsx(Menu, { className: "w-4 h-4" }) }), _jsxs("button", { onClick: () => {
                                        setActiveModuleId(modulesData[0].id);
                                        setIsSidebarOpen(false);
                                    }, className: "flex items-center gap-2 group cursor-pointer", children: [_jsx("div", { className: "w-8 h-8 rounded-lg bg-[#B6532B] flex items-center justify-center shadow-md shadow-[#B6532B]/10 group-hover:scale-105 transition-transform", children: _jsx(Sparkles, { className: "w-[18px] h-[18px] text-[#FAF6EE]" }) }), _jsxs("span", { className: "font-bold text-base text-[#2E251E] tracking-tight font-serif", children: ["Latent", _jsx("span", { className: "text-[#B6532B] font-light", children: "Academy" })] })] })] }), _jsx("div", { className: "flex items-center gap-3", children: _jsxs("button", { onClick: handleReset, className: "px-3.5 py-1.5 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#F4EFE6] text-[#6E6257] hover:text-[#2E251E] text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer font-mono", children: [_jsx(RefreshCw, { className: "w-3.5 h-3.5" }), _jsx("span", { children: "Reset" })] }) })] }) }), _jsx("div", { className: "sticky top-[61px] z-30 lg:hidden w-full bg-[#FAF6EE]/90 backdrop-blur-md border-b border-[#E5DDD0] px-4 py-2 overflow-x-auto flex gap-1.5 scrollbar-none shadow-sm", children: (isKaggleModule
                    ? [
                        { id: 'theory', label: 'Theory', icon: _jsx(BookOpen, { className: "w-3.5 h-3.5" }) },
                        { id: 'sandbox', label: 'Notebook', icon: _jsx(Code, { className: "w-3.5 h-3.5" }) }
                    ]
                    : [
                        { id: 'theory', label: 'Theory', icon: _jsx(BookOpen, { className: "w-3.5 h-3.5" }) },
                        { id: 'visualizer', label: 'Simulator', icon: _jsx(Activity, { className: "w-3.5 h-3.5" }) },
                        { id: 'quiz', label: 'Quiz', icon: _jsx(HelpCircle, { className: "w-3.5 h-3.5" }) },
                        { id: 'sandbox', label: 'Coding', icon: _jsx(Code, { className: "w-3.5 h-3.5" }) },
                        { id: 'interview', label: 'Interview', icon: _jsx(Award, { className: "w-3.5 h-3.5" }) }
                    ]).map((tab) => {
                    const isActive = activeTab === tab.id;
                    let activeStyle = 'border-transparent bg-[#F4EFE6]/60 text-[#6E6257]';
                    if (isActive) {
                        activeStyle = 'border-[#B6532B] text-[#B6532B] font-semibold bg-[#FAF6EE] shadow-sm border';
                    }
                    return (_jsxs("button", { onClick: () => setActiveTab(tab.id), className: `px-3.5 py-1.5 rounded-xl flex items-center gap-1.5 text-xs font-mono transition-all cursor-pointer whitespace-nowrap ${activeStyle}`, children: [tab.icon, _jsx("span", { children: tab.label })] }, tab.id));
                }) }), isSidebarOpen && (_jsx("div", { className: "fixed inset-0 z-40 bg-[#2E251E]/40 backdrop-blur-sm lg:hidden transition-opacity duration-300", onClick: () => setIsSidebarOpen(false) })), _jsxs("main", { className: "flex-1 w-full max-w-7xl mx-auto px-4 py-6 flex flex-col lg:flex-row gap-6 relative", children: [_jsxs("aside", { className: `
          fixed top-[61px] bottom-0 left-0 z-50 w-80 max-w-[85vw] flex flex-col bg-[#F4EFE6] border-r border-[#E5DDD0] shadow-2xl transition-transform duration-300 ease-in-out transform
          ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}
          lg:relative lg:top-auto lg:bottom-auto lg:translate-x-0 lg:flex lg:w-80 lg:shrink-0 lg:border lg:rounded-2xl lg:max-h-[85vh] lg:overflow-hidden lg:shadow-none lg:z-auto
        `, children: [_jsxs("div", { className: "p-4 border-b border-[#E5DDD0] bg-[#F4EFE6] sticky top-0 z-10 flex justify-between items-center", children: [_jsxs("div", { children: [_jsx("span", { className: "text-[10px] text-[#B6532B] font-mono uppercase tracking-wider block font-bold", children: "Learning Syllabus" }), _jsxs("div", { className: "flex justify-between items-center mt-1 gap-2", children: [_jsxs("h2", { className: "text-[#2E251E] font-bold text-sm font-serif", children: [totalModules, " ML Modules"] }), _jsxs("span", { className: "text-[10px] font-mono bg-[#FAF6EE] border border-[#E5DDD0] px-1.5 py-0.5 rounded text-[#6E6257]", children: [userStats.completedModules.length, "/", totalModules] })] })] }), _jsx("button", { onClick: () => setIsSidebarOpen(false), className: "lg:hidden p-2 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#F4EFE6] text-[#6E6257] hover:text-[#2E251E] transition-all cursor-pointer flex items-center justify-center", children: _jsx(X, { className: "w-3.5 h-3.5" }) })] }), _jsx("div", { className: "flex-1 overflow-y-auto p-3 space-y-4 scroller scrollbar-thin", children: categories.map((cat) => (_jsxs("div", { className: "space-y-1.5", children: [_jsxs("div", { className: "flex items-center gap-1.5 px-2 py-0.5 font-mono text-[9px] text-[#6E6257] uppercase tracking-wider font-bold", children: [getCategoryIcon(cat), _jsx("span", { children: cat })] }), _jsx("div", { className: "space-y-1", children: modulesByCategory[cat]?.map((m, idx) => {
                                                const isActive = m.id === activeModuleId;
                                                const progress = getModuleProgress(m.id);
                                                let statusColor = 'border-transparent text-[#6E6257] hover:bg-[#FAF6EE]/50';
                                                let progressLabel = 'Unlocked';
                                                let labelBadgeStyle = 'bg-[#6E6257]/10 text-[#6E6257] border-[#6E6257]/10';
                                                if (isActive) {
                                                    statusColor = 'border-[#B6532B] bg-[#FAF6EE] text-[#2E251E] font-semibold shadow-sm';
                                                }
                                                if (progress.isCompleted) {
                                                    progressLabel = 'Completed';
                                                    labelBadgeStyle = 'bg-[#3B7A57]/10 text-[#3B7A57] border-[#3B7A57]/20';
                                                }
                                                else if (progress.isInProgress) {
                                                    progressLabel = `${progress.completed}/${progress.total} Completed`;
                                                    labelBadgeStyle = 'bg-[#C18C3B]/10 text-[#C18C3B] border-[#C18C3B]/20';
                                                }
                                                return (_jsxs("button", { onClick: () => {
                                                        setActiveModuleId(m.id);
                                                        setIsSidebarOpen(false);
                                                    }, className: `w-full text-left p-2.5 rounded-xl border flex items-center justify-between gap-2 text-xs font-mono transition-all cursor-pointer ${statusColor}`, children: [_jsxs("div", { className: "truncate pr-1", children: [_jsxs("span", { className: "text-[#C18C3B] font-bold mr-1.5", children: [idx + 1, "."] }), _jsx("span", { className: "text-[#2E251E] font-medium leading-snug", children: m.title })] }), _jsx("span", { className: `text-[8px] font-bold px-1.5 py-0.5 rounded border whitespace-nowrap shrink-0 ${labelBadgeStyle}`, children: progressLabel })] }, m.id));
                                            }) })] }, cat))) })] }), _jsxs("section", { className: "flex-1 min-w-0 flex flex-col space-y-6", children: [_jsxs("div", { className: "bg-[#F4EFE6] border border-[#E5DDD0] p-6 rounded-2xl space-y-4", children: [_jsxs("div", { className: "flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 border-b border-[#E5DDD0] pb-3", children: [_jsxs("div", { children: [_jsx("span", { className: "px-2 py-0.5 rounded-md text-[9px] font-mono uppercase bg-[#B6532B]/10 text-[#B6532B] border border-[#B6532B]/20 font-bold", children: activeModule.category }), _jsx("h1", { className: "text-xl sm:text-2xl font-bold text-[#2E251E] font-serif leading-tight mt-1", children: activeModule.title })] }), _jsxs("div", { className: "flex items-center gap-3", children: [_jsxs("div", { className: "text-right", children: [_jsx("span", { className: "text-[10px] text-[#6E6257] font-mono block", children: "Module Progress" }), _jsxs("span", { className: "text-xs font-mono font-bold text-[#2E251E]", children: [activeModuleProgress.completed, " of ", activeModuleProgress.total, " components"] })] }), _jsxs("div", { className: "w-12 h-12 rounded-full border-2 border-[#E5DDD0] flex items-center justify-center font-mono text-xs font-bold text-[#B6532B] relative overflow-hidden bg-[#FAF6EE]", children: [_jsxs("span", { children: [Math.round(activeModuleProgress.percentage), "%"] }), _jsx("div", { className: "absolute inset-0 border-2 border-[#B6532B] transition-transform pointer-events-none rounded-full", style: {
                                                                    clipPath: `polygon(50% 50%, 50% 0%, ${activeModuleProgress.percentage >= 25 ? '100% 0%,' : ''} ${activeModuleProgress.percentage >= 50 ? '100% 100%,' : ''} ${activeModuleProgress.percentage >= 75 ? '0% 100%,' : ''} ${activeModuleProgress.percentage >= 100 ? '0% 0%,' : ''} 50% 0%)`,
                                                                    opacity: activeModuleProgress.percentage > 0 ? 0.35 : 0
                                                                } })] })] })] }), _jsxs("div", { className: "grid grid-cols-1 md:grid-cols-12 gap-4", children: [_jsxs("div", { className: "md:col-span-8 space-y-2", children: [_jsx("span", { className: "text-[9px] text-[#6E6257] font-mono uppercase block", children: "Core Concept Summary" }), _jsx("p", { className: "text-[#6E6257] text-xs font-medium leading-relaxed font-sans", children: activeModule.description })] }), _jsxs("div", { className: "md:col-span-4 bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-xl flex flex-col justify-center text-center space-y-1 shadow-inner", children: [_jsx("span", { className: "text-[8px] text-[#B6532B] font-mono uppercase tracking-wider block font-bold", children: "Mathematical Formula" }), _jsx("div", { className: "text-xs font-mono font-bold text-[#2E251E] overflow-x-auto whitespace-nowrap py-1", dangerouslySetInnerHTML: { __html: katex.renderToString(normalizeLatex(activeModule.formula), { displayMode: true, throwOnError: false }) } })] })] }), _jsx("div", { className: "hidden lg:flex border-b border-[#E5DDD0] pt-2 overflow-x-auto gap-1 scrollbar-thin", children: (isKaggleModule
                                            ? [
                                                { id: 'theory', label: 'Theory', icon: _jsx(BookOpen, { className: "w-3.5 h-3.5" }) },
                                                { id: 'sandbox', label: 'Coding Notebook', icon: _jsx(Code, { className: "w-3.5 h-3.5" }) }
                                            ]
                                            : [
                                                { id: 'theory', label: 'Theory & Math', icon: _jsx(BookOpen, { className: "w-3.5 h-3.5" }) },
                                                { id: 'visualizer', label: 'Interactive Simulator', icon: _jsx(Activity, { className: "w-3.5 h-3.5" }) },
                                                { id: 'quiz', label: 'Quiz Checkpoint', icon: _jsx(HelpCircle, { className: "w-3.5 h-3.5" }) },
                                                { id: 'sandbox', label: 'Coding Sandbox', icon: _jsx(Code, { className: "w-3.5 h-3.5" }) },
                                                { id: 'interview', label: 'Interview Board', icon: _jsx(Award, { className: "w-3.5 h-3.5" }) }
                                            ]).map((tab) => {
                                            const isActive = activeTab === tab.id;
                                            let activeStyle = 'border-transparent text-[#6E6257] hover:text-[#2E251E]';
                                            if (isActive) {
                                                activeStyle = 'border-[#B6532B] text-[#B6532B] font-semibold border-b-2 bg-[#FAF6EE]/40';
                                            }
                                            return (_jsxs("button", { onClick: () => setActiveTab(tab.id), className: `px-4 py-2.5 flex items-center gap-1.5 text-xs font-mono transition-all border-b-2 border-transparent cursor-pointer whitespace-nowrap ${activeStyle}`, children: [tab.icon, _jsx("span", { children: tab.label })] }, tab.id));
                                        }) })] }), _jsxs("div", { className: "animate-fade-in flex-1", children: [activeTab === 'theory' && _jsx(TheoryView, { module: activeModule }), activeTab === 'visualizer' && (_jsxs("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] rounded-2xl shadow-sm overflow-hidden", children: [_jsxs("div", { className: "bg-[#F4EFE6] border-b border-[#E5DDD0] p-4 space-y-2", children: [_jsx("span", { className: "text-[10px] text-[#B6532B] font-mono font-bold uppercase tracking-wider block", children: "Interactive Simulator" }), _jsx("p", { className: "text-[#2E251E] text-xs font-medium leading-relaxed font-sans", children: activeModule.interactiveSummary }), _jsxs("div", { className: "flex gap-3 pt-1 flex-wrap", children: [_jsx("span", { className: "text-[9px] bg-[#C18C3B]/10 text-[#C18C3B] border border-[#C18C3B]/20 px-2 py-0.5 rounded font-mono font-semibold", children: "\uD83D\uDDB1 Click to interact" }), _jsx("span", { className: "text-[9px] bg-[#3B7A57]/10 text-[#3B7A57] border border-[#3B7A57]/20 px-2 py-0.5 rounded font-mono font-semibold", children: "\u26A1 Real-time updates" }), _jsx("span", { className: "text-[9px] bg-[#B6532B]/10 text-[#B6532B] border border-[#B6532B]/20 px-2 py-0.5 rounded font-mono font-semibold", children: "\u25B6 Auto-running" })] })] }), _jsx("div", { className: "p-4", children: _jsx(ModuleSimulator, {}) })] })), activeTab === 'quiz' && (_jsx("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-sm", children: _jsx(QuizEngine, {}) })), activeTab === 'sandbox' && (isKaggleModule ? (_jsx(KaggleNotebook, {})) : (_jsx("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-sm", children: _jsx(CodingPlayground, {}) }))), activeTab === 'interview' && (_jsx("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-sm", children: _jsx(InterviewPrep, {}) }))] })] })] }), _jsx("footer", { className: "w-full px-6 py-6 mt-12 bg-[#F4EFE6] border-t border-[#E5DDD0] text-center text-[10px] text-[#6E6257] font-mono leading-normal", children: _jsxs("div", { className: "max-w-7xl mx-auto flex flex-col sm:flex-row justify-between items-center gap-3", children: [_jsx("span", { children: "\u00A9 2026 Latent Academy \u2014 Interactive Machine Learning Education" }), _jsx("div", { className: "flex gap-4", children: _jsxs("span", { className: "text-[#B6532B] font-semibold", children: [totalModules, " Modules \u00B7 ", modulesData.reduce((s, m) => s + m.quiz.length, 0), " Quiz Questions \u00B7 ", modulesData.reduce((s, m) => s + m.interviewQuestions.length, 0), " FAANG Q&As"] }) })] }) })] }));
}
export default function App() {
    return (_jsx(AppProvider, { children: _jsx(MainAppContent, {}) }));
}
