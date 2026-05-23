import React, { createContext, useContext, useState, useEffect } from 'react';
import type { UserStats, MLModule } from '../types';
import { modulesData } from '../data/modules';

interface AppContextType {
  userStats: UserStats;
  activeModuleId: string;
  activeModule: MLModule;
  activeCodingStage: 'tutorial' | 'project' | 'assignment';
  activeTab: 'theory' | 'visualizer' | 'quiz' | 'sandbox' | 'interview';
  setActiveModuleId: (id: string) => void;
  setActiveCodingStage: (stage: 'tutorial' | 'project' | 'assignment') => void;
  setActiveTab: (tab: 'theory' | 'visualizer' | 'quiz' | 'sandbox' | 'interview') => void;
  completeCodingStage: (moduleId: string, stage: 'tutorial' | 'project' | 'assignment') => void;
  completeQuiz: (moduleId: string) => void;
  completeInterview: (moduleId: string) => void;
  resetProgress: () => void;
}



const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const isKaggleModule = (moduleId: string) =>
    modulesData.find(m => m.id === moduleId)?.category === 'Kaggle Real-World Projects';

  const [activeModuleId, setActiveModuleId] = useState<string>(() => {
    const saved = localStorage.getItem('latent_academy_active_module');
    return saved && modulesData.some(m => m.id === saved) ? saved : modulesData[0].id;
  });

  const [activeCodingStage, setActiveCodingStage] = useState<'tutorial' | 'project' | 'assignment'>('tutorial');
  const [activeTab, setActiveTab] = useState<'theory' | 'visualizer' | 'quiz' | 'sandbox' | 'interview'>('theory');

  const activeModule = modulesData.find(m => m.id === activeModuleId) || modulesData[0];

  const [userStats, setUserStats] = useState<UserStats>(() => {
    const saved = localStorage.getItem('latent_academy_stats_2');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {
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



  const completeCodingStage = (moduleId: string, stage: 'tutorial' | 'project' | 'assignment') => {
    setUserStats(prev => {
      const currentStages = prev.completedCodingStages[moduleId] || [];
      if (currentStages.includes(stage)) return prev;

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

  const completeQuiz = (moduleId: string) => {
    setUserStats(prev => {
      if (prev.completedQuizzes.includes(moduleId)) return prev;

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

  const completeInterview = (moduleId: string) => {
    setUserStats(prev => {
      if (prev.completedInterviews.includes(moduleId)) return prev;

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

  return (
    <AppContext.Provider value={{
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
    }}>
      {children}
    </AppContext.Provider>
  );
};

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
};
