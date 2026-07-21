import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useState } from 'react';
import { useApp } from '../context/AppContext';
import { Building2, ChevronDown, ChevronUp, CheckCircle, Search, Filter } from 'lucide-react';
const DIFFICULTY_COLORS = {
    'Easy': 'bg-[#3B7A57]/10 text-[#3B7A57] border-[#3B7A57]/20',
    'Medium': 'bg-[#C18C3B]/10 text-[#C18C3B] border-[#C18C3B]/20',
    'Hard': 'bg-[#B6532B]/10 text-[#B6532B] border-[#B6532B]/20',
    'Advanced': 'bg-[#2E251E]/10 text-[#2E251E] border-[#2E251E]/20',
    'Intermediate': 'bg-[#6E6257]/10 text-[#6E6257] border-[#6E6257]/20',
};
export const InterviewPrep = () => {
    const { activeModule, completeInterview, userStats } = useApp();
    const questions = activeModule.interviewQuestions || [];
    const [expandedIdx, setExpandedIdx] = useState(null);
    const [searchQuery, setSearchQuery] = useState('');
    const [filterDifficulty, setFilterDifficulty] = useState('All');
    const isCompleted = userStats.completedInterviews.includes(activeModule.id);
    const toggleExpand = (idx) => {
        setExpandedIdx(prev => prev === idx ? null : idx);
    };
    const handleComplete = () => {
        completeInterview(activeModule.id);
    };
    const difficulties = ['All', ...Array.from(new Set(questions.map(q => q.difficulty)))];
    const filteredQuestions = questions.filter(q => {
        const matchesSearch = searchQuery === '' ||
            q.question.toLowerCase().includes(searchQuery.toLowerCase()) ||
            q.answer.toLowerCase().includes(searchQuery.toLowerCase());
        const matchesDiff = filterDifficulty === 'All' || q.difficulty === filterDifficulty;
        return matchesSearch && matchesDiff;
    });
    const diffCounts = {
        Easy: questions.filter(q => q.difficulty === 'Easy').length,
        Medium: questions.filter(q => q.difficulty === 'Medium').length,
        Hard: questions.filter(q => q.difficulty === 'Hard' || q.difficulty === 'Advanced' || q.difficulty === 'Intermediate').length,
    };
    return (_jsxs("div", { className: "max-w-4xl mx-auto space-y-5 p-2", children: [_jsxs("div", { className: "bg-[#F4EFE6] border border-[#E5DDD0] p-5 rounded-2xl", children: [_jsxs("div", { className: "flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4", children: [_jsxs("div", { children: [_jsx("h2", { className: "text-[#2E251E] font-bold text-lg font-serif leading-tight", children: activeModule.title }), _jsxs("p", { className: "text-[#6E6257] text-xs mt-0.5 font-mono", children: ["FAANG Interview Preparation \u2014 ", questions.length, " Questions"] })] }), !isCompleted ? (_jsx("button", { onClick: handleComplete, className: "px-4 py-2 rounded-xl bg-[#B6532B] hover:bg-[#A6431B] text-[#FAF6EE] text-xs font-semibold transition-all font-mono shadow-md cursor-pointer shrink-0", children: "Mark as Reviewed" })) : (_jsxs("div", { className: "px-4 py-2 rounded-xl bg-[#3B7A57]/10 border border-[#3B7A57]/20 text-[#3B7A57] text-xs font-semibold font-mono flex items-center gap-2", children: [_jsx(CheckCircle, { className: "w-4 h-4" }), " Reviewed"] }))] }), _jsx("div", { className: "grid grid-cols-3 gap-3 mt-4", children: [
                            { label: 'Easy', count: diffCounts.Easy, color: '#3B7A57' },
                            { label: 'Medium', count: diffCounts.Medium, color: '#C18C3B' },
                            { label: 'Hard/Adv', count: diffCounts.Hard, color: '#B6532B' },
                        ].map(stat => (_jsxs("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-3 text-center", children: [_jsx("div", { className: "text-lg font-bold font-mono", style: { color: stat.color }, children: stat.count }), _jsx("div", { className: "text-[9px] text-[#6E6257] font-mono uppercase tracking-wider", children: stat.label })] }, stat.label))) })] }), _jsxs("div", { className: "flex flex-col sm:flex-row gap-3", children: [_jsxs("div", { className: "flex-1 relative", children: [_jsx(Search, { className: "w-3.5 h-3.5 text-[#6E6257] absolute left-3 top-1/2 -translate-y-1/2" }), _jsx("input", { type: "text", placeholder: "Search questions...", value: searchQuery, onChange: e => setSearchQuery(e.target.value), className: "w-full pl-8 pr-4 py-2 bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl text-xs font-mono text-[#2E251E] outline-none focus:border-[#B6532B] transition-colors" })] }), _jsxs("div", { className: "flex items-center gap-2", children: [_jsx(Filter, { className: "w-3.5 h-3.5 text-[#6E6257] shrink-0" }), _jsx("div", { className: "flex gap-1.5 flex-wrap", children: difficulties.map(d => (_jsx("button", { onClick: () => setFilterDifficulty(d), className: `px-2.5 py-1.5 rounded-lg text-[10px] font-mono font-semibold transition-all cursor-pointer border ${filterDifficulty === d
                                        ? 'bg-[#B6532B] text-[#FAF6EE] border-[#B6532B]'
                                        : 'bg-[#FAF6EE] text-[#6E6257] border-[#E5DDD0] hover:border-[#B6532B]/40'}`, children: d }, d))) })] })] }), _jsxs("div", { className: "space-y-3", children: [filteredQuestions.length === 0 && (_jsx("div", { className: "text-center py-10 bg-[#F4EFE6] border border-[#E5DDD0] rounded-2xl", children: _jsx("p", { className: "text-[#6E6257] text-sm font-mono", children: "No questions match your filter." }) })), filteredQuestions.map((q) => {
                        // Find original index for expand tracking
                        const origIdx = questions.indexOf(q);
                        const isExpanded = expandedIdx === origIdx;
                        const diffColor = DIFFICULTY_COLORS[q.difficulty] || DIFFICULTY_COLORS['Medium'];
                        return (_jsxs("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl overflow-hidden transition-all hover:border-[#CFC5B4]", children: [_jsxs("button", { onClick: () => toggleExpand(origIdx), className: "w-full text-left p-4 flex items-start justify-between gap-4 hover:bg-[#F4EFE6] transition-colors cursor-pointer", children: [_jsxs("div", { className: "flex-1 space-y-2", children: [_jsxs("div", { className: "flex items-center gap-2 flex-wrap", children: [_jsxs("span", { className: "text-[9px] font-mono font-bold text-[#6E6257] bg-[#F4EFE6] border border-[#E5DDD0] px-1.5 py-0.5 rounded", children: ["Q", origIdx + 1] }), _jsx("span", { className: `px-2 py-0.5 rounded-md text-[9px] font-mono uppercase font-bold border ${diffColor}`, children: q.difficulty }), _jsxs("div", { className: "flex flex-wrap items-center gap-1.5", children: [_jsx(Building2, { className: "w-3 h-3 text-[#B6532B]" }), q.companyTags?.map(tag => (_jsx("span", { className: "px-1.5 py-0.5 rounded bg-[#F4EFE6] border border-[#E5DDD0] text-[9px] font-mono text-[#2E251E] font-semibold", children: tag }, tag)))] })] }), _jsx("p", { className: "text-[#2E251E] font-semibold text-sm leading-snug", children: q.question })] }), _jsx("div", { className: "text-[#6E6257] shrink-0 mt-1", children: isExpanded ? _jsx(ChevronUp, { className: "w-4 h-4" }) : _jsx(ChevronDown, { className: "w-4 h-4" }) })] }), isExpanded && (_jsx("div", { className: "px-4 pb-4 border-t border-[#E5DDD0] bg-[#F4EFE6]", children: _jsxs("div", { className: "pt-3 space-y-2", children: [_jsx("span", { className: "text-[10px] text-[#B6532B] font-mono uppercase font-bold tracking-wider", children: "Model Answer" }), _jsx("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-4", children: _jsx("p", { className: "text-[#2E251E] text-sm leading-relaxed whitespace-pre-line font-sans", children: q.answer }) }), _jsx("p", { className: "text-[9px] text-[#6E6257] font-mono italic", children: "Practice tip: Answer out loud, then compare to the model answer above." })] }) }))] }, origIdx));
                    })] }), questions.length === 0 && (_jsxs("div", { className: "text-center py-10 bg-[#F4EFE6] border border-[#E5DDD0] rounded-2xl p-6", children: [_jsx("h5", { className: "text-[#2E251E] font-bold text-sm", children: "No Interview Questions Available" }), _jsx("p", { className: "text-[#6E6257] text-xs mt-1", children: "This module does not have interview questions configured yet." })] }))] }));
};
