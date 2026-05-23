import React from 'react';
import { CheckCircle, Code2, FileText } from 'lucide-react';
import { useApp } from '../context/AppContext';

const stageLabels: Record<'tutorial' | 'project' | 'assignment', string> = {
  tutorial: 'Tutorial',
  project: 'Project',
  assignment: 'Challenge'
};

export const KaggleNotebook: React.FC = () => {
  const { activeModule, completeCodingStage, userStats } = useApp();
  const completedStages = userStats.completedCodingStages[activeModule.id] || [];

  const fallbackCells = [
    {
      id: 'tutorial-cell',
      type: 'code' as const,
      title: activeModule.coding.tutorial.title,
      summary: activeModule.coding.tutorial.description,
      code: activeModule.coding.tutorial.solution || activeModule.coding.tutorial.starterCode,
      output: activeModule.coding.tutorial.expectedOutput,
      stageId: 'tutorial' as const
    },
    {
      id: 'project-cell',
      type: 'code' as const,
      title: activeModule.coding.project.title,
      summary: activeModule.coding.project.description,
      code: activeModule.coding.project.solution || activeModule.coding.project.starterCode,
      output: activeModule.coding.project.expectedOutput,
      stageId: 'project' as const
    },
    {
      id: 'assignment-cell',
      type: 'code' as const,
      title: activeModule.coding.assignment.title,
      summary: activeModule.coding.assignment.description,
      code: activeModule.coding.assignment.solution || activeModule.coding.assignment.starterCode,
      output: activeModule.coding.assignment.expectedOutput,
      stageId: 'assignment' as const
    }
  ];

  const notebookCells = activeModule.notebookCells && activeModule.notebookCells.length > 0
    ? activeModule.notebookCells
    : fallbackCells;

  return (
    <div className="bg-[#FAF6EE] min-h-screen p-4 md:p-5">
      <div className="bg-[#F4EFE6] border border-[#E5DDD0] rounded-2xl p-4 mb-5">
        <p className="text-[10px] uppercase font-mono font-bold tracking-widest text-[#B6532B]">Kaggle Coding Notebook</p>
        <h3 className="text-[#2E251E] font-bold text-base mt-1 leading-tight">Jupyter-style project flow</h3>
        <p className="text-xs text-[#6E6257] leading-relaxed mt-2">
          {activeModule.description}
        </p>
      </div>

      <div className="space-y-4">
        {notebookCells.map((cell, idx) => {
          const stageId = cell.stageId;
          const isStageComplete = stageId ? completedStages.includes(stageId) : false;
          const stageLabel = stageId ? stageLabels[stageId] : null;

          return (
            <div key={cell.id} className="flex gap-3">
              <div className="w-16 text-right text-[10px] font-mono text-[#6E6257] pt-3">
                {cell.type === 'code' ? `In [${idx + 1}]` : 'Markdown'}
              </div>
              <div className="flex-1 bg-[#FAF6EE] border border-[#E5DDD0] rounded-2xl overflow-hidden shadow-sm">
                <div className="flex items-center justify-between gap-3 px-4 py-2 border-b border-[#E5DDD0] bg-[#F4EFE6]">
                  <div className="flex items-center gap-2">
                    {cell.type === 'code' ? (
                      <Code2 className="w-4 h-4 text-[#C18C3B]" />
                    ) : (
                      <FileText className="w-4 h-4 text-[#B6532B]" />
                    )}
                    <span className="text-xs font-semibold text-[#2E251E]">{cell.title}</span>
                  </div>
                  {stageId && (
                    isStageComplete ? (
                      <span className="text-[10px] font-mono flex items-center gap-1 text-[#3B7A57]">
                        <CheckCircle className="w-3.5 h-3.5" />
                        {stageLabel} complete
                      </span>
                    ) : (
                      <button
                        onClick={() => completeCodingStage(activeModule.id, stageId)}
                        className="text-[10px] font-mono px-2 py-1 rounded-lg border border-[#E5DDD0] bg-[#FAF6EE] text-[#2E251E] hover:bg-[#F4EFE6]"
                      >
                        Mark {stageLabel} done
                      </button>
                    )
                  )}
                </div>
                <div className="p-4 space-y-3">
                  <p className="text-xs text-[#6E6257] leading-relaxed whitespace-pre-line">{cell.summary}</p>
                  {cell.type === 'code' && cell.code && (
                    <pre className="text-xs font-mono text-[#F4EFE6] bg-[#2E251E] border border-[#4A3D31] rounded-xl p-4 whitespace-pre-wrap leading-relaxed">
                      {cell.code}
                    </pre>
                  )}
                  {cell.output && (
                    <div className="text-xs font-mono text-[#2E251E] bg-[#F4EFE6] border border-[#E5DDD0] rounded-xl px-3 py-2">
                      Out [{idx + 1}]: {cell.output}
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
