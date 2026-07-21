# Session Changes

> Branch: `main` (914c31a)
> All changes uncommitted.

---

## Part 1 — Professor Review Improvements (Phase 1)

### DBSCAN Simulator rewrite — `src/components/Simulators/DbscanSandbox.tsx`
- New interactive sandbox: scatter plot with adjustable `eps` and `min_samples` sliders
- Real-time cluster coloring, noise visualization, legend toggles
- Registered in `ModuleSimulator.tsx` as `case 'dbscan'`

### CNN Simulator improvements — `src/components/Simulators/CNNSimulator.tsx`
- Fixed: filter count stuck at 1 (was accidentally constant due to stale state closure)
- Added: epoch counter, configurable kernel size, pooling toggle, learning rate slider
- Training now updates live with per-class accuracy breakdown

### AttentionMap visual enhancements — `src/components/Simulators/AttentionMap.tsx`
- Added: temperature slider for softmax sharpness, multi-head attention visualization
- Fixed: attention weight rendering was clipped on narrow viewports

### OptimizationAlgorithms comparison panel — `src/components/Simulators/OptimizerCompare.tsx`
- New side-by-side comparison: SGD, Momentum, Adam, RMSprop on 2D Rosenbrock landscape
- Convergence traces plotted with iteration markers
- Registered in `ModuleSimulator.tsx` as `case 'optimization'`

### ModuleSimulator routing fixes — `src/components/Simulators/ModuleSimulator.tsx`
- Added missing switch cases for all 7 simulators
- `case 'vae'` → VAESimulator, `case 'gan'` → GANSimulator, `case 'mlp'` → NeuralNetSandbox
- `case 'optimization'` → OptimizerCompare, `case 'dbscan'` → DbscanSandbox
- `case 'kaggle-preview'` → KaggleDataPreview

---

## Part 2 — Professor Review Improvements (Phase 2) — New Modules

### 4 New Module files (data + coding exercises)
| File | Topic |
|------|-------|
| `src/data/modules/naturalLanguageProcessing.ts` | NLP: TF-IDF, Bag-of-Words, sequence models |
| `src/data/modules/recommendationSystems.ts` | Collaborative & content-based filtering (SVD, KNN) |
| `src/data/modules/modelEvaluation.ts` | Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix |
| `src/data/modules/hyperparameterTuning.ts` | GridSearchCV, RandomSearch, Bayesian optimization |

Each includes `tutorial`, `project`, `assignment` coding stages with starter code, solution, hints, and test keywords.

### 5 New Simulator components
| File | Function |
|------|----------|
| `KaggleDataPreview.tsx` | Tabular data explorer: scatter, histogram, correlation heatmap |
| `NLPSentimentSimulator.tsx` | Real-time sentiment classifier with TF-IDF + Naive Bayes |
| `RecommenderSimulator.tsx` | MovieLens SVD recommender with rating prediction |
| `ModelEvalSimulator.tsx` | Confusion matrix, ROC curve, PR curve playground |
| `HyperparamTuningSimulator.tsx` | 2D grid search heatmap with contour overlay |

### Module index registration
- `src/data/modules/index.ts` updated to export all 4 new modules
- `src/data/moduleImports.ts` updated for dynamic imports
- All 5 new simulators registered in `ModuleSimulator.tsx` switch

---

## Part 3 — Python Code Snippets in Theory Sections

Added `#### Python Implementation` code blocks to all 34 module files' theory sections:

| Module | Snippet topic |
|--------|--------------|
| anomalyDetection | IsolationForest example |
| biasVarianceTradeoff | cross_val_score with DecisionTree |
| convolutionalNetworks | Keras Conv2D model summary |
| dbscanClustering | DBSCAN fit_predict |
| decisionTreeAlgorithm | entropy calculation |
| decisionTrees | DecisionTreeClassifier |
| ensembleTechniquesBaggingBoosting | Bagging vs AdaBoost comparison |
| featureEngineering | PolynomialFeatures |
| generativeAdversarialNetworks | GAN discriminator/generator (Keras) |
| gradientBoosting | GradientBoostingRegressor |
| gradientBoostingAdvanced | XGBoost DMatrix training |
| gradientDescent | Manual numpy GD loop |
| hyperparameterTuning | GridSearchCV |
| imbalancedDatasets | SMOTE resampling |
| introductionToMl | sklearn train/test split + RF |
| kmeansClustering | KMeans with inertia output |
| knnClassification | KNeighborsClassifier on iris |
| linearRegression | LinearRegression |
| logisticRegression | LogisticRegression |
| modelEvaluation | accuracy, precision, recall, F1 |
| multiLayerPerceptron | MLPClassifier |
| naiveBayes | GaussianNB on iris |
| naturalLanguageProcessing | TfidfVectorizer + MultinomialNB |
| neuralActivations | sigmoid, relu, tanh implementations |
| optimizationAlgorithms | Manual SGD function |
| pcaReduction | PCA on digits dataset |
| qLearningRl | numpy Q-table learning loop |
| randomForests | RandomForestClassifier |
| recommendationSystems | Surprise SVD cross-validation |
| recurrentNetworks | Keras LSTM model summary |
| ridgeLassoRegularization | Ridge vs Lasso comparison |
| singlePerceptron | numpy perceptron training |
| supportVectorMachines | SVC with RBF kernel |
| timeSeriesAnalysis | ARIMA model summary |
| transformerAttention | Scaled dot-product attention (numpy) |
| tsneVisualization | t-SNE on digits |
| variationalAutoencoders | PyTorch VAE loss function |
| wordEmbeddings | Word2Vec with gensim |

All backticks properly escaped for TypeScript template literals.

### Format conversion (6 files)
- `decisionTrees.ts`, `linearRegression.ts`, `logisticRegression.ts`, `recurrentNetworks.ts`, `transformerAttention.ts`, `wordEmbeddings.ts` — converted from single-line `"theory": "..."` JSON-style string to multi-line template literal `theory: \`...\`` for snippet insertion compatibility.

---

## Verification

- `npx tsc --noEmit` — **0 errors**
