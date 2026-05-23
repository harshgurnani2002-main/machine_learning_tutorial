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
import { Database } from 'lucide-react';

export const ModuleSimulator: React.FC = () => {
  const { activeModule } = useApp();

  const renderSimulator = () => {
    switch (activeModule.simulatorId) {
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
        return <GradientBoostingSimulator />;
      case 'kmeans':
      case 'dbscan':
        return <KMeansSandbox />;
      case 'gradient-descent':
        return <GradientDescent />;
      case 'attention':
        return <AttentionMap />;
      case 'word-embeddings':
        return <EmbeddingSpace />;
      case 'time-series':
        return <TimeSeriesSimulator />;
      case 'feature-scaling':
      case 'regularization':
      case 'bias-variance':
        return <FeatureScalingSimulator />;
      case 'mlp':
      case 'perceptron':
      case 'cnn':
      case 'rnn':
      case 'q-learning':
      case 'gan':
      case 'vae':
      case 'tsne':
      case 'activations':
      case 'optimizers':
        return <NeuralNetSandbox />;
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
