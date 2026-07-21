import { jsx as _jsx } from "react/jsx-runtime";
import React, { createContext, useContext, useState, useEffect } from 'react';
import { modulesData } from '../data/modules';
const AppContext = createContext(undefined);
export const AppProvider = ({ children }) => {
    const isKaggleModule = (moduleId) => modulesData.find(m => m.id === moduleId)?.category === 'Kaggle Real-World Projects';
    const [activeModuleId, setActiveModuleId] = useState(() => {
        const saved = localStorage.getItem('latent_academy_active_module');
        return saved && modulesData.some(m => m.id === saved) ? saved : modulesData[0].id;
    });
    const [activeCodingStage, setActiveCodingStage] = useState('tutorial');
    const [activeTab, setActiveTab] = useState('theory');
    const activeModule = modulesData.find(m => m.id === activeModuleId) || modulesData[0];
    const [userStats, setUserStats] = useState(() => {
        const saved = localStorage.getItem('latent_academy_stats_2');
        if (saved) {
            try {
                return JSON.parse(saved);
            }
            catch (e) {
                console.error("Failed to parse saved user stats", e);
            }
        }
        return {
            completedModules: [],
            completedCodingStages: {},
            completedQuizzes: [],
            completedInterviews: []
        };
    });
    useEffect(() => {
        localStorage.setItem('latent_academy_active_module', activeModuleId);
    }, [activeModuleId]);
    useEffect(() => {
        localStorage.setItem('latent_academy_stats_2', JSON.stringify(userStats));
    }, [userStats]);
    const completeCodingStage = (moduleId, stage) => {
        setUserStats(prev => {
            const currentStages = prev.completedCodingStages[moduleId] || [];
            if (currentStages.includes(stage))
                return prev;
            const newStages = [...currentStages, stage];
            const updatedCodingStages = {
                ...prev.completedCodingStages,
                [moduleId]: newStages
            };
            // Check if this module is fully completed
            const quizCompleted = prev.completedQuizzes.includes(moduleId);
            const interviewCompleted = prev.completedInterviews.includes(moduleId);
            const allStagesCompleted = newStages.includes('tutorial') && newStages.includes('project') && newStages.includes('assignment');
            const completedModules = [...prev.completedModules];
            const moduleCompleted = isKaggleModule(moduleId)
                ? allStagesCompleted
                : quizCompleted && interviewCompleted && allStagesCompleted;
            if (moduleCompleted && !completedModules.includes(moduleId)) {
                completedModules.push(moduleId);
            }
            return {
                ...prev,
                completedCodingStages: updatedCodingStages,
                completedModules
            };
        });
    };
    const completeQuiz = (moduleId) => {
        setUserStats(prev => {
            if (prev.completedQuizzes.includes(moduleId))
                return prev;
            const completedQuizzes = [...prev.completedQuizzes, moduleId];
            const currentStages = prev.completedCodingStages[moduleId] || [];
            const interviewCompleted = prev.completedInterviews.includes(moduleId);
            const allStagesCompleted = currentStages.includes('tutorial') && currentStages.includes('project') && currentStages.includes('assignment');
            const completedModules = [...prev.completedModules];
            if (!isKaggleModule(moduleId) && interviewCompleted && allStagesCompleted && !completedModules.includes(moduleId)) {
                completedModules.push(moduleId);
            }
            return {
                ...prev,
                completedQuizzes,
                completedModules
            };
        });
    };
    const completeInterview = (moduleId) => {
        setUserStats(prev => {
            if (prev.completedInterviews.includes(moduleId))
                return prev;
            const completedInterviews = [...prev.completedInterviews, moduleId];
            const currentStages = prev.completedCodingStages[moduleId] || [];
            const quizCompleted = prev.completedQuizzes.includes(moduleId);
            const allStagesCompleted = currentStages.includes('tutorial') && currentStages.includes('project') && currentStages.includes('assignment');
            const completedModules = [...prev.completedModules];
            if (!isKaggleModule(moduleId) && quizCompleted && allStagesCompleted && !completedModules.includes(moduleId)) {
                completedModules.push(moduleId);
            }
            return {
                ...prev,
                completedInterviews,
                completedModules
            };
        });
    };
    const resetProgress = () => {
        setUserStats({
            completedModules: [],
            completedCodingStages: {},
            completedQuizzes: [],
            completedInterviews: []
        });
        setActiveCodingStage('tutorial');
        setActiveTab('theory');
        setActiveModuleId(modulesData[0].id);
    };
    return (_jsx(AppContext.Provider, { value: {
            userStats,
            activeModuleId,
            activeModule,
            activeCodingStage,
            activeTab,
            setActiveModuleId,
            setActiveCodingStage,
            setActiveTab,
            completeCodingStage,
            completeQuiz,
            completeInterview,
            resetProgress
        }, children: children }));
};
export const useApp = () => {
    const context = useContext(AppContext);
    if (!context) {
        throw new Error('useApp must be used within an AppProvider');
    }
    return context;
};
