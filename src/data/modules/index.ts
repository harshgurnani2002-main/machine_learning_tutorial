import type { MLModule } from '../../types';

import { linearRegression } from './linearRegression';
import { logisticRegression } from './logisticRegression';
import { decisionTrees } from './decisionTrees';
import { randomForests } from './randomForests';
import { kmeansClustering } from './kmeansClustering';
import { supportVectorMachines } from './supportVectorMachines';
import { naiveBayes } from './naiveBayes';
import { knnClassification } from './knnClassification';
import { pcaReduction } from './pcaReduction';
import { gradientDescent } from './gradientDescent';
import { decisionTreeAlgorithm } from './decisionTreeAlgorithm';
import { gradientBoosting } from './gradientBoosting';
import { singlePerceptron } from './singlePerceptron';
import { multiLayerPerceptron } from './multiLayerPerceptron';
import { convolutionalNetworks } from './convolutionalNetworks';
import { recurrentNetworks } from './recurrentNetworks';
import { transformerAttention } from './transformerAttention';
import { wordEmbeddings } from './wordEmbeddings';
import { qLearningRl } from './qLearningRl';
import { generativeAdversarialNetworks } from './generativeAdversarialNetworks';
import { variationalAutoencoders } from './variationalAutoencoders';
import { tsneVisualization } from './tsneVisualization';
import { ridgeLassoRegularization } from './ridgeLassoRegularization';
import { dbscanClustering } from './dbscanClustering';
import { neuralActivations } from './neuralActivations';
import { optimizationAlgorithms } from './optimizationAlgorithms';
import { biasVarianceTradeoff } from './biasVarianceTradeoff';
import { featureEngineering } from './featureEngineering';
import { gradientBoostingAdvanced } from './gradientBoostingAdvanced';
import { timeSeriesAnalysis } from './timeSeriesAnalysis';
import { kaggleTitanicProject } from './kaggleTitanicProject';
import { kaggleHousePricesProject } from './kaggleHousePricesProject';
import { kaggleCreditCardFraudProject } from './kaggleCreditCardFraudProject';
import { kaggleCustomerChurnProject } from './kaggleCustomerChurnProject';
import { kaggleStoreSalesForecastingProject } from './kaggleStoreSalesForecastingProject';
import { kaggleDigitRecognizerProject } from './kaggleDigitRecognizerProject';
import { kaggleDisasterTweetsProject } from './kaggleDisasterTweetsProject';
import { kaggleLoanDefaultRiskProject } from './kaggleLoanDefaultRiskProject';

import { introductionToMl } from './introductionToMl';
import { imbalancedDatasets } from './imbalancedDatasets';
import { anomalyDetection } from './anomalyDetection';
import { ensembleTechniquesBaggingBoosting } from './ensembleTechniquesBaggingBoosting';
import { kaggleEdaProject } from './kaggleEdaProject';

// ─────────────────────────────────────────────────────────────────────────────
// Module list ordered from beginner → advanced → real-world projects
// ─────────────────────────────────────────────────────────────────────────────
export const modulesData: MLModule[] = [
  // ── 1. FOUNDATIONS & MATH ─────────────────────────────────────────────────
  introductionToMl,               // What is ML, paradigms, lifecycle
  linearRegression,               // Simplest supervised model
  gradientDescent,                // Optimization core — needed before trees
  logisticRegression,             // Classification intro
  ridgeLassoRegularization,       // Regularization before ensembles

  // ── 2. TREE-BASED & ENSEMBLE MODELS ──────────────────────────────────────
  decisionTrees,                  // Base learner
  decisionTreeAlgorithm,          // Deep dive: ID3, Gini, splitting
  ensembleTechniquesBaggingBoosting, // Parallel vs sequential ensembles
  randomForests,                  // Bagging applied
  gradientBoosting,               // Boosting applied
  gradientBoostingAdvanced,       // XGBoost / LightGBM / CatBoost

  // ── 3. CLASSICAL SUPERVISED ML ───────────────────────────────────────────
  supportVectorMachines,          // Max-margin classifiers
  naiveBayes,                     // Probabilistic classifiers
  knnClassification,              // Instance-based learning

  // ── 4. UNSUPERVISED LEARNING ─────────────────────────────────────────────
  kmeansClustering,               // Centroid-based clustering
  dbscanClustering,               // Density-based clustering
  anomalyDetection,               // Outlier & anomaly detection
  pcaReduction,                   // Dimensionality reduction

  // ── 5. FEATURE WORK & ML ENGINEERING ─────────────────────────────────────
  featureEngineering,             // Feature creation & selection
  imbalancedDatasets,             // Class imbalance — SMOTE, class weights
  biasVarianceTradeoff,           // Bias-variance decomposition
  timeSeriesAnalysis,             // Temporal data modeling

  // ── 6. DEEP LEARNING ──────────────────────────────────────────────────────
  neuralActivations,              // Activation functions
  optimizationAlgorithms,         // Adam, SGD, schedulers
  singlePerceptron,               // Single neuron
  multiLayerPerceptron,           // Fully connected networks
  convolutionalNetworks,          // CNNs
  recurrentNetworks,              // RNNs / LSTMs
  transformerAttention,           // Self-attention & transformers
  wordEmbeddings,                 // Word2Vec / GloVe / FastText
  variationalAutoencoders,        // VAEs — generative models
  generativeAdversarialNetworks,  // GANs
  tsneVisualization,              // t-SNE visualization
  qLearningRl,                    // Reinforcement learning

  // ── 7. KAGGLE REAL-WORLD PROJECTS ────────────────────────────────────────
  kaggleEdaProject,               // EDA & preprocessing — start here
  kaggleTitanicProject,           // Classic binary classification
  kaggleHousePricesProject,       // Regression
  kaggleCreditCardFraudProject,   // Imbalanced classification
  kaggleCustomerChurnProject,     // Business ML
  kaggleStoreSalesForecastingProject, // Time series forecasting
  kaggleDigitRecognizerProject,   // Computer vision
  kaggleDisasterTweetsProject,    // NLP
  kaggleLoanDefaultRiskProject,   // Risk modeling
];
