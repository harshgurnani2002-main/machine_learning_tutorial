import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React from 'react';
import { useApp } from '../../context/AppContext';
import { KaggleDataPreview } from './KaggleDataPreview';
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
import { DbscanSandbox } from './DbscanSandbox';
import { NeuralNetSandbox } from './NeuralNetSandbox';
import { GradientDescent } from './GradientDescent';
import { OptimizerCompare } from './OptimizerCompare';
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
import { NLPSentimentSimulator } from './NLPSentimentSimulator';
import { RegularizationSimulator } from './RegularizationSimulator';
import { BiasVarianceSimulator } from './BiasVarianceSimulator';
import { TSNESimulator } from './TSNESimulator';
import { RecommenderSimulator } from './RecommenderSimulator';
import { ModelEvalSimulator } from './ModelEvalSimulator';
import { HyperparamTuningSimulator } from './HyperparamTuningSimulator';
import { Database } from 'lucide-react';
export const ModuleSimulator = () => {
    const { activeModule } = useApp();
    const renderSimulator = () => {
        switch (activeModule.simulatorId) {
            case 'intro-simulator':
            case 'kaggle-notebook':
                return _jsx(KaggleDataPreview, {});
            case 'lin-reg':
                return _jsx(InteractiveRegression, {});
            case 'log-reg':
                return _jsx(LogisticRegressionSimulator, {});
            case 'decision-tree':
            case 'tree-splits':
                return _jsx(DecisionTreeSimulator, {});
            case 'random-forest':
                return _jsx(RandomForestSimulator, {});
            case 'svm':
                return _jsx(SVMSimulator, {});
            case 'naive-bayes':
                return _jsx(NaiveBayesSimulator, {});
            case 'knn':
                return _jsx(KNNClassifierSimulator, {});
            case 'pca':
                return _jsx(PCASimulator, {});
            case 'gradient-boosting':
                return _jsx(BasicGBSimulator, {});
            case 'gradient-boosting-advanced':
                return _jsx(GradientBoostingSimulator, {});
            case 'kmeans':
                return _jsx(KMeansSandbox, {});
            case 'dbscan':
                return _jsx(DbscanSandbox, {});
            case 'gradient-descent':
                return _jsx(GradientDescent, {});
            case 'attention':
                return _jsx(AttentionMap, {});
            case 'word-embeddings':
                return _jsx(EmbeddingSpace, {});
            case 'time-series':
                return _jsx(TimeSeriesSimulator, {});
            case 'feature-scaling':
                return _jsx(FeatureScalingSimulator, {});
            case 'regularization':
                return _jsx(RegularizationSimulator, {});
            case 'bias-variance':
                return _jsx(BiasVarianceSimulator, {});
            case 'mlp':
                return _jsx(NeuralNetSandbox, {});
            case 'optimizers':
                return _jsx(OptimizerCompare, {});
            case 'tsne':
                return _jsx(TSNESimulator, {});
            case 'perceptron':
                return _jsx(PerceptronSimulator, {});
            case 'cnn':
                return _jsx(CNNSimulator, {});
            case 'rnn':
                return _jsx(RNNSimulator, {});
            case 'nlp-sentiment':
                return _jsx(NLPSentimentSimulator, {});
            case 'q-learning':
                return _jsx(QLearningSimulator, {});
            case 'gan':
                return _jsx(GANSimulator, {});
            case 'vae':
                return _jsx(VAESimulator, {});
            case 'activations':
                return _jsx(ActivationsSimulator, {});
            case 'imbalanced-simulator':
                return _jsx(ImbalancedDataSimulator, {});
            case 'anomaly-simulator':
                return _jsx(AnomalyDetectionSimulator, {});
            case 'ensemble-simulator':
                return _jsx(EnsembleSimulator, {});
            case 'recommender':
                return _jsx(RecommenderSimulator, {});
            case 'model-eval':
                return _jsx(ModelEvalSimulator, {});
            case 'hyperparam-tuning':
                return _jsx(HyperparamTuningSimulator, {});
            default:
                return (_jsxs("div", { className: "flex flex-col items-center justify-center h-full p-12 text-center text-[#6E6257]", children: [_jsx(Database, { className: "w-16 h-16 mb-4 text-[#CFC5B4]" }), _jsx("h3", { className: "text-xl font-bold text-[#2E251E] mb-2", children: "No Simulator Available" }), _jsx("p", { children: "This module does not have an interactive visualization simulator associated with it." })] }));
        }
    };
    return (_jsx("div", { className: "w-full h-full min-h-[500px] bg-[#FAF6EE] rounded-2xl overflow-hidden shadow-inner border border-[#E5DDD0]", children: renderSimulator() }));
};
