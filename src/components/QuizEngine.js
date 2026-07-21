import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useState, useEffect } from 'react';
import { useApp } from '../context/AppContext';
import { BookOpen, CheckCircle2, AlertCircle, RefreshCw, Trophy, ArrowRight } from 'lucide-react';
export const QuizEngine = () => {
    const { activeModule, completeQuiz, userStats } = useApp();
    const quizQuestions = activeModule.quiz || [];
    const [activeIdx, setActiveIdx] = useState(0);
    const [selectedAns, setSelectedAns] = useState('');
    const [isAnswered, setIsAnswered] = useState(false);
    const [isCorrect, setIsCorrect] = useState(false);
    const [score, setScore] = useState(0);
    const [showResult, setShowResult] = useState(false);
    const [answers, setAnswers] = useState([]);
    const restartQuiz = () => {
        setActiveIdx(0);
        setSelectedAns('');
        setIsAnswered(false);
        setIsCorrect(false);
        setScore(0);
        setShowResult(false);
        setAnswers([]);
    };
    // Restart quiz when active module changes
    useEffect(() => {
        restartQuiz(); // eslint-disable-line react-hooks/set-state-in-effect
    }, [activeModule.id]);
    const activeQuestion = quizQuestions[activeIdx];
    const handleOptionSelect = (opt) => {
        if (isAnswered)
            return;
        setSelectedAns(opt);
    };
    const checkAnswer = () => {
        if (!selectedAns)
            return;
        const correct = selectedAns === activeQuestion.correctAnswer;
        setIsCorrect(correct);
        setIsAnswered(true);
        if (correct) {
            setScore(prev => prev + 1);
        }
        setAnswers(prev => [...prev, correct]);
    };
    const handleNext = () => {
        const updatedScore = score + (isCorrect ? 1 : 0);
        setIsAnswered(false);
        setSelectedAns('');
        if (activeIdx + 1 < quizQuestions.length) {
            setActiveIdx(prev => prev + 1);
        }
        else {
            setShowResult(true);
            if (updatedScore >= Math.ceil(quizQuestions.length * 0.7)) {
                completeQuiz(activeModule.id);
            }
        }
    };
    if (quizQuestions.length === 0) {
        return (_jsxs("div", { className: "text-center py-10 bg-[#F4EFE6] border border-[#E5DDD0] rounded-2xl p-6 max-w-md mx-auto", children: [_jsx(BookOpen, { className: "w-10 h-10 text-[#C18C3B] mx-auto mb-3 animate-pulse" }), _jsx("h5", { className: "text-[#2E251E] font-bold text-sm", children: "No Quiz Available" }), _jsx("p", { className: "text-[#6E6257] text-xs mt-1", children: "This module does not have quiz questions configured." })] }));
    }
    const isCompleted = userStats.completedQuizzes.includes(activeModule.id);
    const progressPercent = ((activeIdx + (isAnswered ? 1 : 0)) / quizQuestions.length) * 100;
    // Option label A/B/C/D
    const optionLabels = ['A', 'B', 'C', 'D', 'E'];
    return (_jsx("div", { className: "max-w-2xl mx-auto p-2", children: !showResult ? (_jsxs("div", { className: "space-y-4", children: [_jsxs("div", { className: "bg-[#F4EFE6] border border-[#E5DDD0] p-4 rounded-2xl space-y-3", children: [_jsxs("div", { className: "flex justify-between items-center", children: [_jsxs("div", { className: "flex items-center gap-2", children: [_jsx(BookOpen, { className: "w-4 h-4 text-[#B6532B]" }), _jsx("span", { className: "text-xs text-[#B6532B] font-mono font-bold uppercase tracking-wider", children: "Knowledge Check" })] }), _jsxs("span", { className: "text-xs text-[#6E6257] font-mono font-semibold", children: ["Q", activeIdx + 1, " / ", quizQuestions.length, " \u00B7 Score: ", score] })] }), _jsx("div", { className: "flex gap-1 flex-wrap", children: quizQuestions.map((_, i) => (_jsx("div", { className: `h-1.5 rounded-full transition-all flex-1 min-w-[6px] max-w-[24px] ${i < answers.length
                                    ? answers[i] ? 'bg-[#3B7A57]' : 'bg-[#A63A2B]'
                                    : i === activeIdx
                                        ? 'bg-[#B6532B]'
                                        : 'bg-[#E5DDD0]'}` }, i))) }), _jsx("div", { className: "h-1 bg-[#E5DDD0] rounded-full overflow-hidden", children: _jsx("div", { className: "h-full bg-gradient-to-r from-[#B6532B] to-[#C18C3B] rounded-full transition-all duration-300", style: { width: `${progressPercent}%` } }) })] }), _jsxs("div", { className: "bg-[#FAF6EE] border border-[#E5DDD0] p-6 rounded-2xl space-y-5", children: [_jsx("div", { className: "space-y-3", children: _jsxs("div", { className: "flex items-start gap-3", children: [_jsxs("span", { className: "text-[10px] font-mono font-bold text-[#FAF6EE] bg-[#B6532B] px-2 py-0.5 rounded shrink-0 mt-0.5", children: ["Q", activeIdx + 1] }), _jsx("h4", { className: "text-[#2E251E] font-semibold text-sm leading-relaxed", children: activeQuestion.question })] }) }), _jsx("div", { className: "space-y-2.5", children: activeQuestion.options?.map((opt, idx) => {
                                const isSelected = selectedAns === opt;
                                const isCorrectOpt = opt === activeQuestion.correctAnswer;
                                const label = optionLabels[idx] || String(idx + 1);
                                let btnStyle = 'border-[#E5DDD0] bg-[#FAF6EE] text-[#2E251E] hover:bg-[#F4EFE6] hover:border-[#CFC5B4]';
                                let labelStyle = 'bg-[#F4EFE6] text-[#6E6257]';
                                if (isSelected && !isAnswered) {
                                    btnStyle = 'border-[#B6532B] bg-[#B6532B]/8 text-[#2E251E]';
                                    labelStyle = 'bg-[#B6532B] text-[#FAF6EE]';
                                }
                                else if (isAnswered) {
                                    if (isCorrectOpt) {
                                        btnStyle = 'border-[#3B7A57] bg-[#3B7A57]/8 text-[#3B7A57]';
                                        labelStyle = 'bg-[#3B7A57] text-[#FAF6EE]';
                                    }
                                    else if (isSelected && !isCorrect) {
                                        btnStyle = 'border-[#A63A2B] bg-[#A63A2B]/8 text-[#A63A2B]';
                                        labelStyle = 'bg-[#A63A2B] text-[#FAF6EE]';
                                    }
                                    else {
                                        btnStyle = 'border-[#E5DDD0] bg-[#FAF6EE] opacity-40 text-[#6E6257]';
                                        labelStyle = 'bg-[#E5DDD0] text-[#6E6257]';
                                    }
                                }
                                return (_jsxs("button", { onClick: () => handleOptionSelect(opt), disabled: isAnswered, className: `w-full text-left p-3.5 rounded-xl border text-xs font-sans transition-all leading-normal flex items-center gap-3 cursor-pointer ${btnStyle}`, children: [_jsx("span", { className: `w-5 h-5 rounded-md flex items-center justify-center text-[10px] font-bold font-mono shrink-0 ${labelStyle}`, children: label }), _jsx("span", { className: "flex-1", children: opt }), isAnswered && isCorrectOpt && _jsx(CheckCircle2, { className: "w-4 h-4 text-[#3B7A57] shrink-0" }), isAnswered && isSelected && !isCorrect && _jsx(AlertCircle, { className: "w-4 h-4 text-[#A63A2B] shrink-0" })] }, idx));
                            }) }), isAnswered && (_jsxs("div", { className: `p-4 rounded-xl border text-xs leading-relaxed animate-fade-in ${isCorrect
                                ? 'border-[#3B7A57]/20 bg-[#3B7A57]/5'
                                : 'border-[#A63A2B]/20 bg-[#A63A2B]/5'}`, children: [_jsxs("div", { className: "flex items-center gap-2 mb-1.5", children: [isCorrect
                                            ? _jsx(CheckCircle2, { className: "w-4 h-4 text-[#3B7A57] shrink-0" })
                                            : _jsx(AlertCircle, { className: "w-4 h-4 text-[#A63A2B] shrink-0" }), _jsx("span", { className: `font-bold text-sm ${isCorrect ? 'text-[#3B7A57]' : 'text-[#A63A2B]'}`, children: isCorrect ? 'Correct!' : 'Not quite — here\'s why:' })] }), _jsx("p", { className: "text-[#6E6257] text-xs font-mono leading-relaxed", children: activeQuestion.explanation })] })), _jsxs("div", { className: "pt-2 border-t border-[#E5DDD0] flex justify-between items-center", children: [_jsx("span", { className: "text-[10px] text-[#6E6257] font-mono", children: isAnswered ? (isCorrect ? '✓ +1 point' : '✗ no points') : 'Choose an answer above' }), !isAnswered ? (_jsx("button", { onClick: checkAnswer, disabled: !selectedAns, className: "px-5 py-2.5 rounded-xl bg-[#B6532B] disabled:opacity-50 text-[#FAF6EE] text-xs font-semibold hover:bg-[#A6431B] transition-all flex items-center gap-1.5 cursor-pointer font-mono", children: "Submit Answer" })) : (_jsxs("button", { onClick: handleNext, className: "px-5 py-2.5 rounded-xl bg-[#2E251E] hover:bg-[#2E251E]/90 text-[#FAF6EE] text-xs font-semibold transition-all flex items-center gap-1.5 cursor-pointer font-mono", children: [activeIdx + 1 < quizQuestions.length ? 'Next Question' : 'See Results', " ", _jsx(ArrowRight, { className: "w-3.5 h-3.5" })] }))] })] })] })) : (_jsxs("div", { className: "bg-[#F4EFE6] border border-[#E5DDD0] p-8 rounded-2xl text-center space-y-6 animate-scale-up", children: [_jsx("div", { className: "w-16 h-16 rounded-full bg-gradient-to-tr from-[#C18C3B] to-[#D5A75C] flex items-center justify-center mx-auto shadow-md", children: _jsx(Trophy, { className: "w-8 h-8 text-[#FAF6EE]" }) }), _jsxs("div", { className: "space-y-2", children: [_jsx("h4", { className: "text-[#2E251E] text-lg font-bold font-serif", children: score >= Math.ceil(quizQuestions.length * 0.7) ? '🎉 Quiz Passed!' : 'Keep Studying!' }), _jsx("p", { className: "text-[#6E6257] text-xs leading-relaxed max-w-sm mx-auto", children: score >= Math.ceil(quizQuestions.length * 0.7)
                                ? `You scored ${score}/${quizQuestions.length} on ${activeModule.title}. Module checkpoint cleared!`
                                : `You scored ${score}/${quizQuestions.length}. You need at least ${Math.ceil(quizQuestions.length * 0.7)} correct to pass. Review the theory and try again!` })] }), _jsxs("div", { className: "max-w-sm mx-auto p-4 bg-[#FAF6EE] border border-[#E5DDD0] rounded-2xl font-mono text-xs space-y-3", children: [_jsxs("div", { className: "text-center", children: [_jsx("span", { className: "text-[#6E6257] text-[9px] block uppercase tracking-wide", children: "Your Score" }), _jsx("span", { className: "text-[#2E251E] font-bold text-2xl", children: score }), _jsxs("span", { className: "text-[#6E6257] text-sm", children: ["/", quizQuestions.length] })] }), _jsx("div", { className: "flex gap-1.5 justify-center", children: answers.map((correct, i) => (_jsx("div", { className: `w-5 h-5 rounded text-[8px] font-bold flex items-center justify-center ${correct ? 'bg-[#3B7A57] text-white' : 'bg-[#A63A2B] text-white'}`, title: `Q${i + 1}: ${correct ? 'Correct' : 'Wrong'}`, children: correct ? '✓' : '✗' }, i))) }), _jsxs("div", { className: "flex justify-between text-[10px] text-[#6E6257] border-t border-[#E5DDD0] pt-2", children: [_jsxs("span", { children: ["\u2713 Correct: ", score] }), _jsxs("span", { children: ["\u2717 Wrong: ", quizQuestions.length - score] }), _jsxs("span", { children: ["Score: ", Math.round((score / quizQuestions.length) * 100), "%"] })] })] }), isCompleted && (_jsx("p", { className: "text-[10px] font-mono text-[#3B7A57]", children: "\u2713 Checkpoint already recorded in your progress." })), _jsx("div", { className: "flex gap-3 justify-center pt-2", children: _jsxs("button", { onClick: restartQuiz, className: "px-5 py-2.5 rounded-xl border border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#F4EFE6] text-[#2E251E] transition-colors text-xs font-semibold flex items-center gap-1.5 cursor-pointer font-mono", children: [_jsx(RefreshCw, { className: "w-3.5 h-3.5" }), " Retake Quiz"] }) })] })) }));
};
