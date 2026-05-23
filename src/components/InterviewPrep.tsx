import React, { useState } from 'react';
import { useApp } from '../context/AppContext';
import { Building2, ChevronDown, ChevronUp, CheckCircle, Search, Filter } from 'lucide-react';

const DIFFICULTY_COLORS: Record<string, string> = {
  'Easy': 'bg-[#3B7A57]/10 text-[#3B7A57] border-[#3B7A57]/20',
  'Medium': 'bg-[#C18C3B]/10 text-[#C18C3B] border-[#C18C3B]/20',
  'Hard': 'bg-[#B6532B]/10 text-[#B6532B] border-[#B6532B]/20',
  'Advanced': 'bg-[#2E251E]/10 text-[#2E251E] border-[#2E251E]/20',
  'Intermediate': 'bg-[#6E6257]/10 text-[#6E6257] border-[#6E6257]/20',
};

export const InterviewPrep: React.FC = () => {
  const { activeModule, completeInterview, userStats } = useApp();
  const questions = activeModule.interviewQuestions || [];

  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [filterDifficulty, setFilterDifficulty] = useState<string>('All');

  const isCompleted = userStats.completedInterviews.includes(activeModule.id);

  const toggleExpand = (idx: number) => {
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

  return (
    <div className="max-w-4xl mx-auto space-y-5 p-2">
      {/* Header Card */}
      <div className="bg-[#F4EFE6] border border-[#E5DDD0] p-5 rounded-2xl">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <div>
            <h2 className="text-[#2E251E] font-bold text-lg font-serif leading-tight">{activeModule.title}</h2>
            <p className="text-[#6E6257] text-xs mt-0.5 font-mono">FAANG Interview Preparation — {questions.length} Questions</p>
          </div>
          {!isCompleted ? (
            <button
              onClick={handleComplete}
              className="px-4 py-2 rounded-xl bg-[#B6532B] hover:bg-[#A6431B] text-[#FAF6EE] text-xs font-semibold transition-all font-mono shadow-md cursor-pointer shrink-0"
            >
              Mark as Reviewed
            </button>
          ) : (
            <div className="px-4 py-2 rounded-xl bg-[#3B7A57]/10 border border-[#3B7A57]/20 text-[#3B7A57] text-xs font-semibold font-mono flex items-center gap-2">
              <CheckCircle className="w-4 h-4" /> Reviewed
            </div>
          )}
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-3 gap-3 mt-4">
          {[
            { label: 'Easy', count: diffCounts.Easy, color: '#3B7A57' },
            { label: 'Medium', count: diffCounts.Medium, color: '#C18C3B' },
            { label: 'Hard/Adv', count: diffCounts.Hard, color: '#B6532B' },
          ].map(stat => (
            <div key={stat.label} className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-3 text-center">
              <div className="text-lg font-bold font-mono" style={{ color: stat.color }}>{stat.count}</div>
              <div className="text-[9px] text-[#6E6257] font-mono uppercase tracking-wider">{stat.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Search & Filter */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="flex-1 relative">
          <Search className="w-3.5 h-3.5 text-[#6E6257] absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search questions..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            className="w-full pl-8 pr-4 py-2 bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl text-xs font-mono text-[#2E251E] outline-none focus:border-[#B6532B] transition-colors"
          />
        </div>
        <div className="flex items-center gap-2">
          <Filter className="w-3.5 h-3.5 text-[#6E6257] shrink-0" />
          <div className="flex gap-1.5 flex-wrap">
            {difficulties.map(d => (
              <button
                key={d}
                onClick={() => setFilterDifficulty(d)}
                className={`px-2.5 py-1.5 rounded-lg text-[10px] font-mono font-semibold transition-all cursor-pointer border ${
                  filterDifficulty === d
                    ? 'bg-[#B6532B] text-[#FAF6EE] border-[#B6532B]'
                    : 'bg-[#FAF6EE] text-[#6E6257] border-[#E5DDD0] hover:border-[#B6532B]/40'
                }`}
              >
                {d}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Questions list */}
      <div className="space-y-3">
        {filteredQuestions.length === 0 && (
          <div className="text-center py-10 bg-[#F4EFE6] border border-[#E5DDD0] rounded-2xl">
            <p className="text-[#6E6257] text-sm font-mono">No questions match your filter.</p>
          </div>
        )}
        {filteredQuestions.map((q) => {
          // Find original index for expand tracking
          const origIdx = questions.indexOf(q);
          const isExpanded = expandedIdx === origIdx;
          const diffColor = DIFFICULTY_COLORS[q.difficulty] || DIFFICULTY_COLORS['Medium'];

          return (
            <div key={origIdx} className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl overflow-hidden transition-all hover:border-[#CFC5B4]">
              <button
                onClick={() => toggleExpand(origIdx)}
                className="w-full text-left p-4 flex items-start justify-between gap-4 hover:bg-[#F4EFE6] transition-colors cursor-pointer"
              >
                <div className="flex-1 space-y-2">
                  {/* Question meta */}
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-[9px] font-mono font-bold text-[#6E6257] bg-[#F4EFE6] border border-[#E5DDD0] px-1.5 py-0.5 rounded">
                      Q{origIdx + 1}
                    </span>
                    <span className={`px-2 py-0.5 rounded-md text-[9px] font-mono uppercase font-bold border ${diffColor}`}>
                      {q.difficulty}
                    </span>
                    <div className="flex flex-wrap items-center gap-1.5">
                      <Building2 className="w-3 h-3 text-[#B6532B]" />
                      {q.companyTags?.map(tag => (
                        <span key={tag} className="px-1.5 py-0.5 rounded bg-[#F4EFE6] border border-[#E5DDD0] text-[9px] font-mono text-[#2E251E] font-semibold">
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                  {/* Question text */}
                  <p className="text-[#2E251E] font-semibold text-sm leading-snug">
                    {q.question}
                  </p>
                </div>
                <div className="text-[#6E6257] shrink-0 mt-1">
                  {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </div>
              </button>

              {/* Answer panel */}
              {isExpanded && (
                <div className="px-4 pb-4 border-t border-[#E5DDD0] bg-[#F4EFE6]">
                  <div className="pt-3 space-y-2">
                    <span className="text-[10px] text-[#B6532B] font-mono uppercase font-bold tracking-wider">
                      Model Answer
                    </span>
                    <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-xl p-4">
                      <p className="text-[#2E251E] text-sm leading-relaxed whitespace-pre-line font-sans">
                        {q.answer}
                      </p>
                    </div>
                    <p className="text-[9px] text-[#6E6257] font-mono italic">
                      Practice tip: Answer out loud, then compare to the model answer above.
                    </p>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {questions.length === 0 && (
        <div className="text-center py-10 bg-[#F4EFE6] border border-[#E5DDD0] rounded-2xl p-6">
          <h5 className="text-[#2E251E] font-bold text-sm">No Interview Questions Available</h5>
          <p className="text-[#6E6257] text-xs mt-1">This module does not have interview questions configured yet.</p>
        </div>
      )}
    </div>
  );
};
