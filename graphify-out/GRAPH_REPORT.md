# Graph Report - .  (2026-07-20)

## Corpus Check
- 109 files · ~211,794 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 375 nodes · 565 edges · 70 communities (26 shown, 44 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS · INFERRED: 2 edges (avg confidence: 0.82)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Package Config & Dependencies|Package Config & Dependencies]]
- [[_COMMUNITY_Core UI Components|Core UI Components]]
- [[_COMMUNITY_ML Module Content|ML Module Content]]
- [[_COMMUNITY_Simulator Framework|Simulator Framework]]
- [[_COMMUNITY_TypeScript App Config|TypeScript App Config]]
- [[_COMMUNITY_TypeScript Node Config|TypeScript Node Config]]
- [[_COMMUNITY_Gradient Boosting Simulator|Gradient Boosting Simulator]]
- [[_COMMUNITY_Bug Fixes & Issues|Bug Fixes & Issues]]
- [[_COMMUNITY_Kaggle Projects & PCA|Kaggle Projects & PCA]]
- [[_COMMUNITY_Module Data & Decision Trees|Module Data & Decision Trees]]
- [[_COMMUNITY_Basic GB Simulator|Basic GB Simulator]]
- [[_COMMUNITY_Decision Tree & Ensemble Sims|Decision Tree & Ensemble Sims]]
- [[_COMMUNITY_t-SNE Simulator|t-SNE Simulator]]
- [[_COMMUNITY_Social Icons|Social Icons]]
- [[_COMMUNITY_Regularization Simulator|Regularization Simulator]]
- [[_COMMUNITY_Bias-Variance Simulator|Bias-Variance Simulator]]
- [[_COMMUNITY_Anomaly Detection Simulator|Anomaly Detection Simulator]]
- [[_COMMUNITY_CNN Simulator|CNN Simulator]]
- [[_COMMUNITY_Gradient Descent Simulator|Gradient Descent Simulator]]
- [[_COMMUNITY_Imbalanced Data Simulator|Imbalanced Data Simulator]]
- [[_COMMUNITY_Perceptron Simulator|Perceptron Simulator]]
- [[_COMMUNITY_RNN Simulator|RNN Simulator]]
- [[_COMMUNITY_Gemini Settings|Gemini Settings]]
- [[_COMMUNITY_Site Index & README|Site Index & README]]
- [[_COMMUNITY_OpenCode Config|OpenCode Config]]
- [[_COMMUNITY_OpenCode Plugin|OpenCode Plugin]]
- [[_COMMUNITY_Activations Simulator|Activations Simulator]]
- [[_COMMUNITY_Embedding Space Simulator|Embedding Space Simulator]]
- [[_COMMUNITY_Logistic Regression Simulator|Logistic Regression Simulator]]
- [[_COMMUNITY_Q-Learning Simulator|Q-Learning Simulator]]
- [[_COMMUNITY_Test Files|Test Files]]
- [[_COMMUNITY_TypeScript Config Ref|TypeScript Config Ref]]
- [[_COMMUNITY_Update Theory Script|Update Theory Script]]
- [[_COMMUNITY_React Framework Assets|React Framework Assets]]
- [[_COMMUNITY_Vite Assets|Vite Assets]]
- [[_COMMUNITY_Gemini Graphify|Gemini Graphify]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 68|Community 68]]

## God Nodes (most connected - your core abstractions)
1. `MLModule` - 49 edges
2. `compilerOptions` - 17 edges
3. `compilerOptions` - 16 edges
4. `useApp()` - 13 edges
5. `Latent Academy Fix Report` - 10 edges
6. `Icons SVG Sprite Sheet` - 6 edges
7. `scripts` - 5 edges
8. `trainModel()` - 5 edges
9. `Sklearn For Project PyTorch For Assignment Convention` - 5 edges
10. `BiasVarianceSimulator()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `DecisionTreeSimulator()` --calls--> `buildTree()`  [INFERRED]
  src/components/Simulators/DecisionTreeSimulator.tsx → src/components/Simulators/EnsembleSimulator.tsx
- `AppContextType` --references--> `MLModule`  [EXTRACTED]
  src/context/AppContext.tsx → src/types.ts
- `MainAppContent()` --calls--> `useApp()`  [EXTRACTED]
  src/App.tsx → src/context/AppContext.tsx
- `CodingPlayground()` --calls--> `useApp()`  [EXTRACTED]
  src/components/CodingPlayground.tsx → src/context/AppContext.tsx
- `InterviewPrep()` --calls--> `useApp()`  [EXTRACTED]
  src/components/InterviewPrep.tsx → src/context/AppContext.tsx

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Sklearn PyTorch Module Classification Groups** — fixes_sklearn_group_a, fixes_sklearn_pytorch_group_b, fixes_pytorch_only_group_c [EXTRACTED 1.00]

## Communities (70 total, 44 thin omitted)

### Community 0 - "Package Config & Dependencies"
Cohesion: 0.06
Nodes (33): dependencies, katex, lucide-react, prismjs, react, react-dom, react-simple-code-editor, tailwindcss (+25 more)

### Community 1 - "Core UI Components"
Cohesion: 0.15
Nodes (16): CodingPlayground(), DIFFICULTY_COLORS, InterviewPrep(), KaggleNotebook(), stageLabels, NeuralBackground(), QuizEngine(), AppContext (+8 more)

### Community 2 - "ML Module Content"
Cohesion: 0.13
Nodes (11): anomalyDetection, gradientBoosting, introductionToMl, kaggleDigitRecognizerProject, kaggleDisasterTweetsProject, kaggleEdaProject, kaggleLoanDefaultRiskProject, kaggleStoreSalesForecastingProject (+3 more)

### Community 3 - "Simulator Framework"
Cohesion: 0.14
Nodes (10): FeatureScalingSimulator(), GANSimulator(), KMeansSandbox(), KNNClassifierSimulator(), NaiveBayesSimulator(), NeuralNetSandbox(), PCASimulator(), RandomForestSimulator() (+2 more)

### Community 4 - "TypeScript App Config"
Cohesion: 0.11
Nodes (18): compilerOptions, allowImportingTsExtensions, erasableSyntaxOnly, jsx, lib, module, moduleDetection, moduleResolution (+10 more)

### Community 5 - "TypeScript Node Config"
Cohesion: 0.11
Nodes (17): compilerOptions, allowImportingTsExtensions, erasableSyntaxOnly, lib, module, moduleDetection, moduleResolution, noEmit (+9 more)

### Community 6 - "Gradient Boosting Simulator"
Cohesion: 0.15
Nodes (15): ALGO, AlgoMode, buildCatTree(), buildLGBTree(), buildXGBTree(), defaultPoints, drawCanvas(), GradientBoostingSimulator() (+7 more)

### Community 7 - "Bug Fixes & Issues"
Cohesion: 0.19
Nodes (14): Latent Academy Fix Report, 27 Curriculum Modules, CNN Sklearn Alternative Gap, Interactive Simulator System, Latent Academy Project, Lucide React Version Bug, Orphan And Scratch Files, Group C CNN Only Pytorch (+6 more)

### Community 8 - "Kaggle Projects & PCA"
Cohesion: 0.14
Nodes (9): kaggleTitanicProject, pcaReduction, recurrentNetworks, tsneVisualization, CodingStage, InterviewQA, ModuleCoding, NotebookCell (+1 more)

### Community 9 - "Module Data & Decision Trees"
Cohesion: 0.15
Nodes (7): modulesData, decisionTreeAlgorithm, decisionTrees, gradientDescent, linearRegression, supportVectorMachines, MLModule

### Community 10 - "Basic GB Simulator"
Cohesion: 0.21
Nodes (9): BasicGBSimulator(), buildStump(), defaultPoints, drawCanvas(), Point, predictStump(), sigmoid(), Stump (+1 more)

### Community 11 - "Decision Tree & Ensemble Sims"
Cohesion: 0.17
Nodes (7): DecisionTreeSimulator(), buildTree(), ClassType, DataPoint, EnsembleSimulator(), Mode, TreeNode

### Community 12 - "t-SNE Simulator"
Cohesion: 0.29
Nodes (5): CLUSTER_COLORS, CLUSTER_LABELS, generateHDClusters(), Point2D, TSNESimulator()

### Community 13 - "Social Icons"
Cohesion: 0.29
Nodes (7): Bluesky Icon, Discord Icon, Documentation Icon, GitHub Icon, Social Sharing Icon, Icons SVG Sprite Sheet, X/Twitter Icon

### Community 14 - "Regularization Simulator"
Cohesion: 0.38
Nodes (6): computeCoefficients(), computeLoss(), FEATURE_NAMES, RegType, RegularizationSimulator(), TRUE_COEFFICIENTS

### Community 15 - "Bias-Variance Simulator"
Cohesion: 0.47
Nodes (3): BiasVarianceSimulator(), fitPolynomial(), generateDataset()

### Community 17 - "CNN Simulator"
Cohesion: 0.50
Nodes (3): CNNSimulator(), defaultImage, filters

### Community 18 - "Gradient Descent Simulator"
Cohesion: 0.50
Nodes (3): GradientDescent(), OptState, Point

### Community 20 - "Perceptron Simulator"
Cohesion: 0.50
Nodes (3): datasets, DatasetType, PerceptronSimulator()

### Community 21 - "RNN Simulator"
Cohesion: 0.50
Nodes (3): embeddings, RNNSimulator(), sequence

## Knowledge Gaps
- **134 isolated node(s):** `BeforeTool`, `$schema`, `plugin`, `@opencode-ai/plugin`, `name` (+129 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **44 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MLModule` connect `Module Data & Decision Trees` to `Core UI Components`, `ML Module Content`, `Kaggle Projects & PCA`, `Community 36`, `Community 37`, `Community 38`, `Community 39`, `Community 40`, `Community 41`, `Community 42`, `Community 43`, `Community 44`, `Community 45`, `Community 46`, `Community 47`, `Community 48`, `Community 49`, `Community 50`, `Community 51`, `Community 52`, `Community 53`, `Community 54`, `Community 55`, `Community 56`, `Community 57`, `Community 58`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Why does `useApp()` connect `Core UI Components` to `Simulator Framework`?**
  _High betweenness centrality (0.009) - this node is a cross-community bridge._
- **What connects `BeforeTool`, `$schema`, `plugin` to the rest of the system?**
  _134 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Package Config & Dependencies` be split into smaller, more focused modules?**
  _Cohesion score 0.058823529411764705 - nodes in this community are weakly interconnected._
- **Should `ML Module Content` be split into smaller, more focused modules?**
  _Cohesion score 0.13043478260869565 - nodes in this community are weakly interconnected._
- **Should `Simulator Framework` be split into smaller, more focused modules?**
  _Cohesion score 0.14285714285714285 - nodes in this community are weakly interconnected._
- **Should `TypeScript App Config` be split into smaller, more focused modules?**
  _Cohesion score 0.10526315789473684 - nodes in this community are weakly interconnected._