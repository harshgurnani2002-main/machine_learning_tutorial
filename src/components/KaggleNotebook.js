import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React from 'react';
import { CheckCircle, Code2, FileText } from 'lucide-react';
import { useApp } from '../context/AppContext';
const stageLabels = {
    tutorial: 'Tutorial',
    project: 'Project',
    assignment: 'Challenge'
};
export const KaggleNotebook = () => {
    const { activeModule, completeCodingStage, userStats } = useApp();
    const completedStages = userStats.completedCodingStages[activeModule.id] || [];
    const fallbackCells = [
        {
            id: 'tutorial-cell',
            type: 'code',
            title: activeModule.coding.tutorial.title,
            summary: activeModule.coding.tutorial.description,
            code: activeModule.coding.tutorial.solution || activeModule.coding.tutorial.starterCode,
            output: activeModule.coding.tutorial.expectedOutput,
            stageId: 'tutorial'
        },
        {
            id: 'project-cell',
            type: 'code',
            title: activeModule.coding.project.title,
            summary: activeModule.coding.project.description,
            code: activeModule.coding.project.solution || activeModule.coding.project.starterCode,
            output: activeModule.coding.project.expectedOutput,
            stageId: 'project'
        },
        {
            id: 'assignment-cell',
            type: 'code',
            title: activeModule.coding.assignment.title,
            summary: activeModule.coding.assignment.description,
            code: activeModule.coding.assignment.solution || activeModule.coding.assignment.starterCode,
            output: activeModule.coding.assignment.expectedOutput,
            stageId: 'assignment'
        }
    ];
    const notebookCells = activeModule.notebookCells && activeModule.notebookCells.length > 0
        ? activeModule.notebookCells
        : fallbackCells;
    return (_jsxs("div", { className: "bg-[#FAF6EE] min-h-screen p-4 md:p-5", children: [_jsxs("div", { className: "bg-[#F4EFE6] border border-[#E5DDD0] rounded-2xl p-4 mb-5", children: [_jsx("p", { className: "text-[10px] uppercase font-mono font-bold tracking-widest text-[#B6532B]", children: "Kaggle Coding Notebook" }), _jsx("h3", { className: "text-[#2E251E] font-bold text-base mt-1 leading-tight", children: "Jupyter-style project flow" }), _jsx("p", { className: "text-xs text-[#6E6257] leading-relaxed mt-2", children: activeModule.description })] }), _jsx("div", { className: "space-y-4", children: notebookCells.map((cell, idx) => {
                    const stageId = cell.stageId;
                    const isStageComplete = stageId ? completedStages.includes(stageId) : false;
                    const stageLabel = stageId ? stageLabels[stageId] : null;
                    return (_jsxs("div", { className: "flex gap-3", children: [_jsx("div", { className: "w-16 text-right text-[10px] font-mono text-[#6E6257] pt-3", children: cell.type === 'code' ? `In [${idx + 1}]` : 'Markdown' }), _jsxs("div", { className: "flex-1 bg-[#FAF6EE] border border-[#E5DDD0] rounded-2xl overflow-hidden shadow-sm", children: [_jsxs("div", { className: "flex items-center justify-between gap-3 px-4 py-2 border-b border-[#E5DDD0] bg-[#F4EFE6]", children: [_jsxs("div", { className: "flex items-center gap-2", children: [cell.type === 'code' ? (_jsx(Code2, { className: "w-4 h-4 text-[#C18C3B]" })) : (_jsx(FileText, { className: "w-4 h-4 text-[#B6532B]" })), _jsx("span", { className: "text-xs font-semibold text-[#2E251E]", children: cell.title })] }), stageId && (isStageComplete ? (_jsxs("span", { className: "text-[10px] font-mono flex items-center gap-1 text-[#3B7A57]", children: [_jsx(CheckCircle, { className: "w-3.5 h-3.5" }), stageLabel, " complete"] })) : (_jsxs("button", { onClick: () => completeCodingStage(activeModule.id, stageId), className: "text-[10px] font-mono px-2 py-1 rounded-lg border border-[#E5DDD0] bg-[#FAF6EE] text-[#2E251E] hover:bg-[#F4EFE6]", children: ["Mark ", stageLabel, " done"] })))] }), _jsxs("div", { className: "p-4 space-y-3", children: [_jsx("p", { className: "text-xs text-[#6E6257] leading-relaxed whitespace-pre-line", children: cell.summary }), cell.type === 'code' && cell.code && (_jsx("pre", { className: "text-xs font-mono text-[#F4EFE6] bg-[#2E251E] border border-[#4A3D31] rounded-xl p-4 whitespace-pre-wrap leading-relaxed", children: cell.code })), cell.output && (_jsxs("div", { className: "text-xs font-mono text-[#2E251E] bg-[#F4EFE6] border border-[#E5DDD0] rounded-xl px-3 py-2", children: ["Out [", idx + 1, "]: ", cell.output] }))] })] })] }, cell.id));
                }) })] }));
};
