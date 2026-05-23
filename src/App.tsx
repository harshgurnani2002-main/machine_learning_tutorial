import { useEffect, useMemo } from 'react';
import { AppProvider, useApp } from './context/AppContext';
import { NeuralBackground } from './components/NeuralBackground';
import { CodingPlayground } from './components/CodingPlayground';
import { KaggleNotebook } from './components/KaggleNotebook';
import { QuizEngine } from './components/QuizEngine';
import { InterviewPrep } from './components/InterviewPrep';
import { ModuleSimulator } from './components/Simulators/ModuleSimulator';
import { modulesData } from './data/modules';
import type { MLModule } from './types';
import { 
  Sparkles, 
  BookOpen, 
  Activity, 
  HelpCircle, 
  Code, 
  Award, 
  RefreshCw, 
  Compass, 
  Scale, 
  Eye, 
  Brain, 
  Settings, 
} from 'lucide-react';
import katex from 'katex';

const normalizeLatex = (raw: string) =>
  raw
    .replace(/\u0007/g, '\\a')
    .replace(/\u0008/g, '\\b')
    .replace(/\u000c/g, '\\f')
    .replace(/\u000b/g, '\\v')
    .replace(/\r/g, '\\r')
    .replace(/\t/g, '\\t')
    .replace(/\u0000/g, '')
    .replace(/[\u0001-\u0006\u000e-\u001f]/g, '');

// Math theory notation parser and visual viewer
const TheoryView: React.FC<{ module: MLModule }> = ({ module }) => {
  const content = module.theory;
  
  const blocks = useMemo(() => {
    const lines = content.split('\n');
    const parsedBlocks: Array<{ type: 'h3' | 'h4' | 'math-block' | 'code' | 'list' | 'p'; text: string; codeLang?: string }> = [];
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
        } else {
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
        } else {
          if (inMath) {
            parsedBlocks.push({ type: 'math-block', text: mathContent });
            mathContent = '';
            inMath = false;
          } else {
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
  const formatText = (text: string) => {
    const formatted: React.ReactNode[] = [];
    const parts = text.split(/(\$[^\$]+\$|\*\*[^\*]+\*\*)/);
    
    parts.forEach((part, idx) => {
      if (part.startsWith('$') && part.endsWith('$')) {
        const math = normalizeLatex(part.substring(1, part.length - 1));
        try {
          const html = katex.renderToString(math, { displayMode: false, throwOnError: false });
          formatted.push(<span key={idx} dangerouslySetInnerHTML={{ __html: html }} />);
        } catch (e) {
          formatted.push(<span key={idx}>{part}</span>);
        }
      } else if (part.startsWith('**') && part.endsWith('**')) {
        formatted.push(<strong key={idx} className="font-bold text-[#2E251E]">{part.substring(2, part.length - 2)}</strong>);
      } else {
        formatted.push(<span key={idx}>{part}</span>);
      }
    });
    
    return formatted;
  };

  return (
    <div className="max-w-3xl mx-auto space-y-4">
      {/* Beginner callout */}
      <div className="bg-[#F4EFE6] border border-[#E5DDD0] px-5 py-3 rounded-xl flex items-center gap-3">
        <span className="text-lg">📖</span>
        <div>
          <p className="text-[10px] text-[#B6532B] font-mono font-bold uppercase tracking-wider">Theory & Math — Absolute Beginner Friendly</p>
          <p className="text-[9px] text-[#6E6257] font-mono mt-0.5">No prior ML knowledge required. Read top to bottom for a complete understanding.</p>
        </div>
      </div>

      <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-6 sm:p-8 rounded-2xl space-y-5 shadow-sm text-[#2E251E]">
        {blocks.map((block, idx) => {
          switch (block.type) {
            case 'h3':
              return (
                <h3 key={idx} className="text-lg sm:text-xl font-bold text-[#2E251E] border-b border-[#E5DDD0] pb-1.5 pt-3 font-serif">
                  {block.text}
                </h3>
              );
            case 'h4':
              return (
                <h4 key={idx} className="text-sm sm:text-base font-bold text-[#B6532B] pt-1 font-mono">
                  ▸ {block.text}
                </h4>
              );
            case 'math-block': {
              const html = katex.renderToString(normalizeLatex(block.text), { displayMode: true, throwOnError: false });
              return (
                <div key={idx} className="my-4 p-4 bg-[#F4EFE6] border-l-4 border-[#B6532B] rounded-r-xl overflow-x-auto">
                  <span className="text-[9px] text-[#6E6257] font-mono uppercase block mb-1">Mathematical Formula</span>
                  <span 
                    className="font-mono text-xs sm:text-sm text-[#B6532B] font-semibold whitespace-pre-wrap block"
                    dangerouslySetInnerHTML={{ __html: html }}
                  />
                </div>
              );
            }
            case 'code':
              return (
                <pre key={idx} className="my-3 p-3.5 dark-sepia-editor rounded-xl overflow-x-auto text-[11px] leading-relaxed text-[#E5E0D8]">
                  <code>{block.text}</code>
                </pre>
              );
            case 'list':
              return (
                <ul key={idx} className="list-none pl-0 text-xs text-[#2E251E] font-mono leading-relaxed">
                  <li className="flex gap-2 items-start py-0.5">
                    <span className="text-[#B6532B] shrink-0 mt-0.5">→</span>
                    <span>{formatText(block.text)}</span>
                  </li>
                </ul>
              );
            case 'p':
            default:
              return (
                <p key={idx} className="text-xs text-[#6E6257] leading-relaxed font-sans font-medium">
                  {formatText(block.text)}
                </p>
              );
          }
        })}
      </div>
    </div>
  );
};

function MainAppContent() {
  const { 
    activeTab, 
    setActiveTab, 
    activeModuleId, 
    setActiveModuleId, 
    activeModule, 
    userStats, 
    resetProgress 
  } = useApp();

  const handleReset = () => {
    if (window.confirm("Are you sure you want to reset all your learning achievements and coding sandboxes? This cannot be undone.")) {
      resetProgress();
    }
  };

  const categories = ['Foundations & Math', 'Supervised Learning', 'Unsupervised Learning', 'Deep Learning', 'Advanced & MLOps', 'Kaggle Real-World Projects'] as const;
  const totalModules = modulesData.length;
  const isKaggleModule = activeModule.category === 'Kaggle Real-World Projects';

  const modulesByCategory = useMemo(() => {
    const groups: { [key: string]: typeof modulesData } = {};
    categories.forEach(cat => {
      groups[cat] = modulesData.filter(m => m.category === cat);
    });
    return groups;
  }, []);

  useEffect(() => {
    if (isKaggleModule && activeTab !== 'theory' && activeTab !== 'sandbox') {
      setActiveTab('theory');
    }
  }, [isKaggleModule, activeTab, setActiveTab]);

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'Foundations & Math':
        return <Compass className="w-3.5 h-3.5 text-[#C18C3B]" />;
      case 'Supervised Learning':
        return <Scale className="w-3.5 h-3.5 text-[#B6532B]" />;
      case 'Unsupervised Learning':
        return <Eye className="w-3.5 h-3.5 text-[#6E6257]" />;
      case 'Deep Learning':
        return <Brain className="w-3.5 h-3.5 text-[#B6532B]" />;
      case 'Advanced & MLOps':
        return <Settings className="w-3.5 h-3.5 text-[#2E251E]" />;
      case 'Kaggle Real-World Projects':
        return <BookOpen className="w-3.5 h-3.5 text-[#B6532B]" />;
      default:
        return <BookOpen className="w-3.5 h-3.5 text-[#6E6257]" />;
    }
  };

  const getModuleProgress = (moduleId: string) => {
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

  return (
    <div className="relative min-h-screen flex flex-col z-10 bg-[#FAF6EE] text-[#2E251E]">
      <NeuralBackground />

      <header className="sticky top-0 z-40 w-full px-6 py-4 bg-[#FAF6EE]/90 backdrop-blur-md border-b border-[#E5DDD0] shadow-sm">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <button 
            onClick={() => setActiveModuleId(modulesData[0].id)}
            className="flex items-center gap-2.5 group cursor-pointer"
          >
            <div className="w-8 h-8 rounded-lg bg-[#B6532B] flex items-center justify-center shadow-md shadow-[#B6532B]/10 group-hover:scale-105 transition-transform">
              <Sparkles className="w-[18px] h-[18px] text-[#FAF6EE]" />
            </div>
            <span className="font-bold text-base text-[#2E251E] tracking-tight font-serif">
              Latent<span className="text-[#B6532B] font-light">Academy</span>
            </span>
          </button>

          <div className="flex flex-wrap items-center gap-3 md:gap-5">

            <button
              onClick={handleReset}
              className="px-3.5 py-1.5 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#F4EFE6] text-[#6E6257] hover:text-[#2E251E] text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer font-mono"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Reset</span>
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1 w-full max-w-7xl mx-auto px-4 py-6 flex flex-col lg:flex-row gap-6 relative">
        <aside className="w-full lg:w-80 shrink-0 flex flex-col bg-[#F4EFE6] border border-[#E5DDD0] rounded-2xl max-h-[85vh] overflow-hidden">
          <div className="p-4 border-b border-[#E5DDD0] bg-[#F4EFE6] sticky top-0 z-10">
            <span className="text-[10px] text-[#B6532B] font-mono uppercase tracking-wider block font-bold">
              Learning Syllabus
            </span>
            <div className="flex justify-between items-center mt-1">
              <h2 className="text-[#2E251E] font-bold text-sm font-serif">{totalModules} Machine Learning Modules</h2>
              <span className="text-[10px] font-mono bg-[#FAF6EE] border border-[#E5DDD0] px-1.5 py-0.5 rounded text-[#6E6257]">
                {userStats.completedModules.length} / {totalModules} Done
              </span>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-3 space-y-4 scroller scrollbar-thin">
            {categories.map((cat) => (
              <div key={cat} className="space-y-1.5">
                <div className="flex items-center gap-1.5 px-2 py-0.5 font-mono text-[9px] text-[#6E6257] uppercase tracking-wider font-bold">
                  {getCategoryIcon(cat)}
                  <span>{cat}</span>
                </div>

                <div className="space-y-1">
                  {modulesByCategory[cat]?.map((m, idx) => {
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
                    } else if (progress.isInProgress) {
                      progressLabel = `${progress.completed}/${progress.total} Completed`;
                      labelBadgeStyle = 'bg-[#C18C3B]/10 text-[#C18C3B] border-[#C18C3B]/20';
                    }

                    return (
                      <button
                        key={m.id}
                        onClick={() => setActiveModuleId(m.id)}
                        className={`w-full text-left p-2.5 rounded-xl border flex items-center justify-between gap-2 text-xs font-mono transition-all cursor-pointer ${statusColor}`}
                      >
                        <div className="truncate pr-1">
                          <span className="text-[#C18C3B] font-bold mr-1.5">{idx + 1}.</span>
                          <span className="text-[#2E251E] font-medium leading-snug">{m.title}</span>
                        </div>
                        <span className={`text-[8px] font-bold px-1.5 py-0.5 rounded border whitespace-nowrap shrink-0 ${labelBadgeStyle}`}>
                          {progressLabel}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </aside>

        <section className="flex-1 min-w-0 flex flex-col space-y-6">
          <div className="bg-[#F4EFE6] border border-[#E5DDD0] p-6 rounded-2xl space-y-4">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 border-b border-[#E5DDD0] pb-3">
              <div>
                <span className="px-2 py-0.5 rounded-md text-[9px] font-mono uppercase bg-[#B6532B]/10 text-[#B6532B] border border-[#B6532B]/20 font-bold">
                  {activeModule.category}
                </span>
                <h1 className="text-xl sm:text-2xl font-bold text-[#2E251E] font-serif leading-tight mt-1">
                  {activeModule.title}
                </h1>
              </div>

              <div className="flex items-center gap-3">
                <div className="text-right">
                  <span className="text-[10px] text-[#6E6257] font-mono block">Module Progress</span>
                  <span className="text-xs font-mono font-bold text-[#2E251E]">
                    {activeModuleProgress.completed} of {activeModuleProgress.total} components
                  </span>
                </div>
                <div className="w-12 h-12 rounded-full border-2 border-[#E5DDD0] flex items-center justify-center font-mono text-xs font-bold text-[#B6532B] relative overflow-hidden bg-[#FAF6EE]">
                  <span>{Math.round(activeModuleProgress.percentage)}%</span>
                  <div 
                    className="absolute inset-0 border-2 border-[#B6532B] transition-transform pointer-events-none rounded-full"
                    style={{
                      clipPath: `polygon(50% 50%, 50% 0%, ${activeModuleProgress.percentage >= 25 ? '100% 0%,' : ''} ${activeModuleProgress.percentage >= 50 ? '100% 100%,' : ''} ${activeModuleProgress.percentage >= 75 ? '0% 100%,' : ''} ${activeModuleProgress.percentage >= 100 ? '0% 0%,' : ''} 50% 0%)`,
                      opacity: activeModuleProgress.percentage > 0 ? 0.35 : 0
                    }}
                  />
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-12 gap-4">
              <div className="md:col-span-8 space-y-2">
                <span className="text-[9px] text-[#6E6257] font-mono uppercase block">Core Concept Summary</span>
                <p className="text-[#6E6257] text-xs font-medium leading-relaxed font-sans">
                  {activeModule.description}
                </p>
              </div>

              <div className="md:col-span-4 bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-xl flex flex-col justify-center text-center space-y-1 shadow-inner">
                <span className="text-[8px] text-[#B6532B] font-mono uppercase tracking-wider block font-bold">Mathematical Formula</span>
                <div 
                  className="text-xs font-mono font-bold text-[#2E251E] overflow-x-auto whitespace-nowrap py-1"
                  dangerouslySetInnerHTML={{ __html: katex.renderToString(normalizeLatex(activeModule.formula), { displayMode: true, throwOnError: false }) }}
                />
              </div>
            </div>

            <div className="flex border-b border-[#E5DDD0] pt-2 overflow-x-auto gap-1 scrollbar-thin">
              {(isKaggleModule
                ? [
                    { id: 'theory' as const, label: 'Theory', icon: <BookOpen className="w-3.5 h-3.5" /> },
                    { id: 'sandbox' as const, label: 'Coding Notebook', icon: <Code className="w-3.5 h-3.5" /> }
                  ]
                : [
                    { id: 'theory' as const, label: 'Theory & Math', icon: <BookOpen className="w-3.5 h-3.5" /> },
                    { id: 'visualizer' as const, label: 'Interactive Simulator', icon: <Activity className="w-3.5 h-3.5" /> },
                    { id: 'quiz' as const, label: 'Quiz Checkpoint', icon: <HelpCircle className="w-3.5 h-3.5" /> },
                    { id: 'sandbox' as const, label: 'Coding Sandbox', icon: <Code className="w-3.5 h-3.5" /> },
                    { id: 'interview' as const, label: 'Interview Board', icon: <Award className="w-3.5 h-3.5" /> }
                  ]).map((tab) => {
                const isActive = activeTab === tab.id;
                let activeStyle = 'border-transparent text-[#6E6257] hover:text-[#2E251E]';
                if (isActive) {
                  activeStyle = 'border-[#B6532B] text-[#B6532B] font-semibold border-b-2 bg-[#FAF6EE]/40';
                }
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`px-4 py-2.5 flex items-center gap-1.5 text-xs font-mono transition-all border-b-2 border-transparent cursor-pointer whitespace-nowrap ${activeStyle}`}
                  >
                    {tab.icon}
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="animate-fade-in flex-1">
            {activeTab === 'theory' && <TheoryView module={activeModule} />}
            {activeTab === 'visualizer' && (
              <div className="bg-[#FAF6EE] border border-[#E5DDD0] rounded-2xl shadow-sm overflow-hidden">
                {/* Simulator header */}
                <div className="bg-[#F4EFE6] border-b border-[#E5DDD0] p-4 space-y-2">
                  <span className="text-[10px] text-[#B6532B] font-mono font-bold uppercase tracking-wider block">Interactive Simulator</span>
                  <p className="text-[#2E251E] text-xs font-medium leading-relaxed font-sans">{activeModule.interactiveSummary}</p>
                  <div className="flex gap-3 pt-1 flex-wrap">
                    <span className="text-[9px] bg-[#C18C3B]/10 text-[#C18C3B] border border-[#C18C3B]/20 px-2 py-0.5 rounded font-mono font-semibold">🖱 Click to interact</span>
                    <span className="text-[9px] bg-[#3B7A57]/10 text-[#3B7A57] border border-[#3B7A57]/20 px-2 py-0.5 rounded font-mono font-semibold">⚡ Real-time updates</span>
                    <span className="text-[9px] bg-[#B6532B]/10 text-[#B6532B] border border-[#B6532B]/20 px-2 py-0.5 rounded font-mono font-semibold">🎛 Adjust parameters</span>
                  </div>
                </div>
                <div className="p-4">
                  <ModuleSimulator />
                </div>
              </div>
            )}
            {activeTab === 'quiz' && (
              <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-sm">
                <QuizEngine />
              </div>
            )}
            {activeTab === 'sandbox' && (
              isKaggleModule ? (
                <KaggleNotebook />
              ) : (
                <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-sm">
                  <CodingPlayground />
                </div>
              )
            )}
            {activeTab === 'interview' && (
              <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-4 rounded-2xl shadow-sm">
                <InterviewPrep />
              </div>
            )}
          </div>
        </section>
      </main>

      <footer className="w-full px-6 py-6 mt-12 bg-[#F4EFE6] border-t border-[#E5DDD0] text-center text-[10px] text-[#6E6257] font-mono leading-normal">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row justify-between items-center gap-3">
          <span>© 2026 Latent Academy — Interactive Machine Learning Education</span>
          <div className="flex gap-4">
            <span className="text-[#B6532B] font-semibold">{totalModules} Modules · 285 Quiz Questions · 549 FAANG Q&As</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <MainAppContent />
    </AppProvider>
  );
}
