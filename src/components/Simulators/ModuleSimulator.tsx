import React from 'react';
import { useApp } from '../../context/AppContext';
import { InteractiveRegression } from './InteractiveRegression';
import { LogisticRegressionSimulator } from './LogisticRegressionSimulator';
import { DecisionTreeSimulator } from './DecisionTreeSimulator';
import { RandomForestSimulator } from './RandomForestSimulator';
import { SVMSimulator } from './SVMSimulator';
import { NaiveBayesSimulator } from './NaiveBayesSimulator';
import { KNNClassifierSimulator } from './KNNClassifierSimulator';
import { PCASimulator } from './PCASimulator';
import { GradientBoostingSimulator } from './GradientBoostingSimulator';
import { BasicGBSimulator } from './BasicGBSimulator';
import { KMeansSandbox } from './KMeansSandbox';
import { NeuralNetSandbox } from './NeuralNetSandbox';
import { GradientDescent } from './GradientDescent';
import { AttentionMap } from './AttentionMap';
import { EmbeddingSpace } from './EmbeddingSpace';
import { TimeSeriesSimulator } from './TimeSeriesSimulator';
import { FeatureScalingSimulator } from './FeatureScalingSimulator';
import { ImbalancedDataSimulator } from './ImbalancedDataSimulator';
import { AnomalyDetectionSimulator } from './AnomalyDetectionSimulator';
import { EnsembleSimulator } from './EnsembleSimulator';
import { QLearningSimulator } from './QLearningSimulator';
import { GANSimulator } from './GANSimulator';
import { VAESimulator } from './VAESimulator';
import { PerceptronSimulator } from './PerceptronSimulator';
import { ActivationsSimulator } from './ActivationsSimulator';
import { CNNSimulator } from './CNNSimulator';
import { RNNSimulator } from './RNNSimulator';
import { RegularizationSimulator } from './RegularizationSimulator';
import { BiasVarianceSimulator } from './BiasVarianceSimulator';
import { TSNESimulator } from './TSNESimulator';
import { Database } from 'lucide-react';

export const ModuleSimulator: React.FC = () => {
  const { activeModule } = useApp();

  const renderSimulator = () => {
    switch (activeModule.simulatorId) {
      case 'intro-simulator':
      case 'kaggle-notebook':
        return (
          <div className="flex flex-col items-center justify-center h-full p-12 text-center text-[#6E6257]">
            <Database className="w-16 h-16 mb-4 text-[#CFC5B4]" />
            <h3 className="text-xl font-bold text-[#2E251E] mb-2">Conceptual Overview</h3>
            <p>This module is designed for reading and reflection rather than interactive simulation. Explore the theory tab for a deep dive.</p>
          </div>
        );
      case 'lin-reg':
        return <InteractiveRegression />;
      case 'log-reg':
        return <LogisticRegressionSimulator />;
      case 'decision-tree':
      case 'tree-splits':
        return <DecisionTreeSimulator />;
      case 'random-forest':
        return <RandomForestSimulator />;
      case 'svm':
        return <SVMSimulator />;
      case 'naive-bayes':
        return <NaiveBayesSimulator />;
      case 'knn':
        return <KNNClassifierSimulator />;
      case 'pca':
        return <PCASimulator />;
      case 'gradient-boosting':
        return <BasicGBSimulator />;
      case 'gradient-boosting-advanced':
        return <GradientBoostingSimulator />;
      case 'kmeans':
        return <KMeansSandbox />;
      case 'dbscan':
        return (
          <div className="flex flex-col items-center justify-center h-full p-12 text-center text-[#6E6257]">
            <Database className="w-16 h-16 mb-4 text-[#CFC5B4]" />
            <h3 className="text-xl font-bold text-[#2E251E] mb-2">DBSCAN Simulator Coming Soon</h3>
            <p>DBSCAN is a density-based clustering algorithm. The K-Means simulator is available for comparison in the meantime.</p>
          </div>
        );
      case 'gradient-descent':
        return <GradientDescent />;
      case 'attention':
        return <AttentionMap />;
      case 'word-embeddings':
        return <EmbeddingSpace />;
      case 'time-series':
        return <TimeSeriesSimulator />;
      case 'feature-scaling':
        return <FeatureScalingSimulator />;
      case 'regularization':
        return <RegularizationSimulator />;
      case 'bias-variance':
        return <BiasVarianceSimulator />;
      case 'mlp':
        return <NeuralNetSandbox />;
      case 'optimizers':
        return <GradientDescent />;
      case 'tsne':
        return <TSNESimulator />;
      case 'perceptron':
        return <PerceptronSimulator />;
      case 'cnn':
        return <CNNSimulator />;
      case 'rnn':
        return <RNNSimulator />;
      case 'q-learning':
        return <QLearningSimulator />;
      case 'gan':
        return <GANSimulator />;
      case 'vae':
        return <VAESimulator />;
      case 'activations':
        return <ActivationsSimulator />;
      case 'imbalanced-simulator':
        return <ImbalancedDataSimulator />;
      case 'anomaly-simulator':
        return <AnomalyDetectionSimulator />;
      case 'ensemble-simulator':
        return <EnsembleSimulator />;
      default:
        return (
          <div className="flex flex-col items-center justify-center h-full p-12 text-center text-[#6E6257]">
            <Database className="w-16 h-16 mb-4 text-[#CFC5B4]" />
            <h3 className="text-xl font-bold text-[#2E251E] mb-2">No Simulator Available</h3>
            <p>This module does not have an interactive visualization simulator associated with it.</p>
          </div>
        );
    }
  };

  return (
    <div className="w-full h-full min-h-[500px] bg-[#FAF6EE] rounded-2xl overflow-hidden shadow-inner border border-[#E5DDD0]">
      {renderSimulator()}
    </div>
  );
};
