import React, { useState, useEffect } from 'react';
import { useApp } from '../context/AppContext';
import { BookOpen, CheckCircle2, AlertCircle, RefreshCw, Trophy, ArrowRight } from 'lucide-react';

export const QuizEngine: React.FC = () => {
  const { activeModule, completeQuiz, userStats } = useApp();
  const quizQuestions = activeModule.quiz || [];

  const [activeIdx, setActiveIdx] = useState<number>(0);
  const [selectedAns, setSelectedAns] = useState<string>('');
  const [isAnswered, setIsAnswered] = useState<boolean>(false);
  const [isCorrect, setIsCorrect] = useState<boolean>(false);
  const [score, setScore] = useState<number>(0);
  const [showResult, setShowResult] = useState<boolean>(false);
  const [answers, setAnswers] = useState<boolean[]>([]);

  // Restart quiz when active module changes
  useEffect(() => {
    restartQuiz();
  }, [activeModule.id]);

  const activeQuestion = quizQuestions[activeIdx];

  const handleOptionSelect = (opt: string) => {
    if (isAnswered) return;
    setSelectedAns(opt);
  };

  const checkAnswer = () => {
    if (!selectedAns) return;
    const correct = selectedAns === activeQuestion.correctAnswer;
    setIsCorrect(correct);
    setIsAnswered(true);
    if (correct) {
      setScore(prev => prev + 1);
    }
    setAnswers(prev => [...prev, correct]);
  };

  const handleNext = () => {
    setIsAnswered(false);
    setSelectedAns('');
    if (activeIdx + 1 < quizQuestions.length) {
      setActiveIdx(prev => prev + 1);
    } else {
      setShowResult(true);
      if (score >= Math.ceil(quizQuestions.length * 0.7)) {
        completeQuiz(activeModule.id);
      }
    }
  };

  const restartQuiz = () => {
    setActiveIdx(0);
    setSelectedAns('');
    setIsAnswered(false);
    setIsCorrect(false);
    setScore(0);
    setShowResult(false);
    setAnswers([]);
  };

  if (quizQuestions.length === 0) {
    return (
      <div className="text-center py-10 bg-[#F4EFE6] border border-[#E5DDD0] rounded-2xl p-6 max-w-md mx-auto">
        <BookOpen className="w-10 h-10 text-[#C18C3B] mx-auto mb-3 animate-pulse" />
        <h5 className="text-[#2E251E] font-bold text-sm">No Quiz Available</h5>
        <p className="text-[#6E6257] text-xs mt-1">This module does not have quiz questions configured.</p>
      </div>
    );
  }

  const isCompleted = userStats.completedQuizzes.includes(activeModule.id);
  const progressPercent = ((activeIdx + (isAnswered ? 1 : 0)) / quizQuestions.length) * 100;

  // Option label A/B/C/D
  const optionLabels = ['A', 'B', 'C', 'D', 'E'];

  return (
    <div className="max-w-2xl mx-auto p-2">
      {!showResult ? (
        <div className="space-y-4">
          {/* Progress header */}
          <div className="bg-[#F4EFE6] border border-[#E5DDD0] p-4 rounded-2xl space-y-3">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-2">
                <BookOpen className="w-4 h-4 text-[#B6532B]" />
                <span className="text-xs text-[#B6532B] font-mono font-bold uppercase tracking-wider">
                  Knowledge Check
                </span>
              </div>
              <span className="text-xs text-[#6E6257] font-mono font-semibold">
                Q{activeIdx + 1} / {quizQuestions.length} · Score: {score}
              </span>
            </div>

            {/* Progress dots */}
            <div className="flex gap-1 flex-wrap">
              {quizQuestions.map((_, i) => (
                <div
                  key={i}
                  className={`h-1.5 rounded-full transition-all flex-1 min-w-[6px] max-w-[24px] ${
                    i < answers.length
                      ? answers[i] ? 'bg-[#3B7A57]' : 'bg-[#A63A2B]'
                      : i === activeIdx
                      ? 'bg-[#B6532B]'
                      : 'bg-[#E5DDD0]'
                  }`}
                />
              ))}
            </div>

            {/* Linear progress bar */}
            <div className="h-1 bg-[#E5DDD0] rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-[#B6532B] to-[#C18C3B] rounded-full transition-all duration-300"
                style={{ width: `${progressPercent}%` }}
              />
            </div>
          </div>

          {/* Question card */}
          <div className="bg-[#FAF6EE] border border-[#E5DDD0] p-6 rounded-2xl space-y-5">
            {/* Question number badge + text */}
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <span className="text-[10px] font-mono font-bold text-[#FAF6EE] bg-[#B6532B] px-2 py-0.5 rounded shrink-0 mt-0.5">
                  Q{activeIdx + 1}
                </span>
                <h4 className="text-[#2E251E] font-semibold text-sm leading-relaxed">
                  {activeQuestion.question}
                </h4>
              </div>
            </div>

            {/* MCQ options */}
            <div className="space-y-2.5">
              {activeQuestion.options?.map((opt, idx) => {
                const isSelected = selectedAns === opt;
                const isCorrectOpt = opt === activeQuestion.correctAnswer;
                const label = optionLabels[idx] || String(idx + 1);

                let btnStyle = 'border-[#E5DDD0] bg-[#FAF6EE] text-[#2E251E] hover:bg-[#F4EFE6] hover:border-[#CFC5B4]';
                let labelStyle = 'bg-[#F4EFE6] text-[#6E6257]';

                if (isSelected && !isAnswered) {
                  btnStyle = 'border-[#B6532B] bg-[#B6532B]/8 text-[#2E251E]';
                  labelStyle = 'bg-[#B6532B] text-[#FAF6EE]';
                } else if (isAnswered) {
                  if (isCorrectOpt) {
                    btnStyle = 'border-[#3B7A57] bg-[#3B7A57]/8 text-[#3B7A57]';
                    labelStyle = 'bg-[#3B7A57] text-[#FAF6EE]';
                  } else if (isSelected && !isCorrect) {
                    btnStyle = 'border-[#A63A2B] bg-[#A63A2B]/8 text-[#A63A2B]';
                    labelStyle = 'bg-[#A63A2B] text-[#FAF6EE]';
                  } else {
                    btnStyle = 'border-[#E5DDD0] bg-[#FAF6EE] opacity-40 text-[#6E6257]';
                    labelStyle = 'bg-[#E5DDD0] text-[#6E6257]';
                  }
                }

                return (
                  <button
                    key={idx}
                    onClick={() => handleOptionSelect(opt)}
                    disabled={isAnswered}
                    className={`w-full text-left p-3.5 rounded-xl border text-xs font-sans transition-all leading-normal flex items-center gap-3 cursor-pointer ${btnStyle}`}
                  >
                    <span className={`w-5 h-5 rounded-md flex items-center justify-center text-[10px] font-bold font-mono shrink-0 ${labelStyle}`}>
                      {label}
                    </span>
                    <span className="flex-1">{opt}</span>
                    {isAnswered && isCorrectOpt && <CheckCircle2 className="w-4 h-4 text-[#3B7A57] shrink-0" />}
                    {isAnswered && isSelected && !isCorrect && <AlertCircle className="w-4 h-4 text-[#A63A2B] shrink-0" />}
                  </button>
                );
              })}
            </div>

            {/* Explanation panel */}
            {isAnswered && (
              <div className={`p-4 rounded-xl border text-xs leading-relaxed animate-fade-in ${
                isCorrect
                  ? 'border-[#3B7A57]/20 bg-[#3B7A57]/5'
                  : 'border-[#A63A2B]/20 bg-[#A63A2B]/5'
              }`}>
                <div className="flex items-center gap-2 mb-1.5">
                  {isCorrect
                    ? <CheckCircle2 className="w-4 h-4 text-[#3B7A57] shrink-0" />
                    : <AlertCircle className="w-4 h-4 text-[#A63A2B] shrink-0" />
                  }
                  <span className={`font-bold text-sm ${isCorrect ? 'text-[#3B7A57]' : 'text-[#A63A2B]'}`}>
                    {isCorrect ? 'Correct!' : 'Not quite — here\'s why:'}
                  </span>
                </div>
                <p className="text-[#6E6257] text-xs font-mono leading-relaxed">
                  {activeQuestion.explanation}
                </p>
              </div>
            )}

            {/* Controls */}
            <div className="pt-2 border-t border-[#E5DDD0] flex justify-between items-center">
              <span className="text-[10px] text-[#6E6257] font-mono">
                {isAnswered ? (isCorrect ? '✓ +1 point' : '✗ no points') : 'Choose an answer above'}
              </span>
              {!isAnswered ? (
                <button
                  onClick={checkAnswer}
                  disabled={!selectedAns}
                  className="px-5 py-2.5 rounded-xl bg-[#B6532B] disabled:opacity-50 text-[#FAF6EE] text-xs font-semibold hover:bg-[#A6431B] transition-all flex items-center gap-1.5 cursor-pointer font-mono"
                >
                  Submit Answer
                </button>
              ) : (
                <button
                  onClick={handleNext}
                  className="px-5 py-2.5 rounded-xl bg-[#2E251E] hover:bg-[#2E251E]/90 text-[#FAF6EE] text-xs font-semibold transition-all flex items-center gap-1.5 cursor-pointer font-mono"
                >
                  {activeIdx + 1 < quizQuestions.length ? 'Next Question' : 'See Results'} <ArrowRight className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          </div>
        </div>
      ) : (
        /* Result dashboard */
        <div className="bg-[#F4EFE6] border border-[#E5DDD0] p-8 rounded-2xl text-center space-y-6 animate-scale-up">
          <div className="w-16 h-16 rounded-full bg-gradient-to-tr from-[#C18C3B] to-[#D5A75C] flex items-center justify-center mx-auto shadow-md">
            <Trophy className="w-8 h-8 text-[#FAF6EE]" />
          </div>

          <div className="space-y-2">
            <h4 className="text-[#2E251E] text-lg font-bold font-serif">
              {score >= Math.ceil(quizQuestions.length * 0.7) ? '🎉 Quiz Passed!' : 'Keep Studying!'}
            </h4>
            <p className="text-[#6E6257] text-xs leading-relaxed max-w-sm mx-auto">
              {score >= Math.ceil(quizQuestions.length * 0.7)
                ? `You scored ${score}/${quizQuestions.length} on ${activeModule.title}. Module checkpoint cleared!`
                : `You scored ${score}/${quizQuestions.length}. You need at least ${Math.ceil(quizQuestions.length * 0.7)} correct to pass. Review the theory and try again!`}
            </p>
          </div>

          {/* Answer breakdown */}
          <div className="max-w-sm mx-auto p-4 bg-[#FAF6EE] border border-[#E5DDD0] rounded-2xl font-mono text-xs space-y-3">
            <div className="text-center">
              <span className="text-[#6E6257] text-[9px] block uppercase tracking-wide">Your Score</span>
              <span className="text-[#2E251E] font-bold text-2xl">{score}</span>
              <span className="text-[#6E6257] text-sm">/{quizQuestions.length}</span>
            </div>
            <div className="flex gap-1.5 justify-center">
              {answers.map((correct, i) => (
                <div
                  key={i}
                  className={`w-5 h-5 rounded text-[8px] font-bold flex items-center justify-center ${
                    correct ? 'bg-[#3B7A57] text-white' : 'bg-[#A63A2B] text-white'
                  }`}
                  title={`Q${i+1}: ${correct ? 'Correct' : 'Wrong'}`}
                >
                  {correct ? '✓' : '✗'}
                </div>
              ))}
            </div>
            <div className="flex justify-between text-[10px] text-[#6E6257] border-t border-[#E5DDD0] pt-2">
              <span>✓ Correct: {score}</span>
              <span>✗ Wrong: {quizQuestions.length - score}</span>
              <span>Score: {Math.round((score / quizQuestions.length) * 100)}%</span>
            </div>
          </div>

          {isCompleted && (
            <p className="text-[10px] font-mono text-[#3B7A57]">✓ Checkpoint already recorded in your progress.</p>
          )}

          <div className="flex gap-3 justify-center pt-2">
            <button
              onClick={restartQuiz}
              className="px-5 py-2.5 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#F4EFE6] text-[#2E251E] transition-colors text-xs font-semibold flex items-center gap-1.5 cursor-pointer font-mono"
            >
              <RefreshCw className="w-3.5 h-3.5" /> Retake Quiz
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
