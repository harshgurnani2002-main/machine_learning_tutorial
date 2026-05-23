export interface UserStats {
  completedModules: string[]; // list of completed module IDs
  completedCodingStages: { [moduleId: string]: ('tutorial' | 'project' | 'assignment')[] };
  completedQuizzes: string[]; // list of module IDs with completed quizzes
  completedInterviews: string[]; // list of module IDs with completed interviews
}

export interface QuizQuestion {
  id: string;
  question: string;
  options: string[];
  correctAnswer: string;
  explanation: string;
}

export interface CodingStage {
  title: string;
  description: string;
  pseudoCode?: string;
  starterCode: string;
  expectedOutput: string;
  solution: string;
  hints: string[];
  testKeywords: string[];
}

export interface NotebookCell {
  id: string;
  type: 'markdown' | 'code';
  title: string;
  summary: string;
  code?: string;
  output?: string;
  stageId?: 'tutorial' | 'project' | 'assignment';
}

export interface ModuleCoding {
  tutorial: CodingStage;
  project: CodingStage;
  assignment: CodingStage;
}

export interface InterviewQA {
  question: string;
  answer: string;
  companyTags: string[];
  difficulty: string;
}

export interface MLModule {
  id: string;
  title: string;
  category: 'Foundations & Math' | 'Supervised Learning' | 'Unsupervised Learning' | 'Deep Learning' | 'Advanced & MLOps' | 'Kaggle Real-World Projects';
  description: string;
  formula: string;
  theory: string; // Detailed markdown + LaTeX text
  interactiveSummary: string;
  simulatorId?: string;
  quiz: QuizQuestion[];
  notebookCells?: NotebookCell[];
  coding: ModuleCoding;
  interviewQuestions: InterviewQA[];
}
