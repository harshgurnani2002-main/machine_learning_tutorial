# Graph Report - machine_learning_tutorial  (2026-06-06)

## Corpus Check
- 99 files · ~208,845 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 379 nodes · 567 edges · 60 communities (25 shown, 35 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `242ad71b`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
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

## God Nodes (most connected - your core abstractions)
1. `MLModule` - 49 edges
2. `PART 6 — COMPLETE SKLEARN SOLUTIONS FOR ALL 27 MODULES` - 28 edges
3. `compilerOptions` - 17 edges
4. `compilerOptions` - 16 edges
5. `useApp()` - 15 edges
6. `Latent Academy — Comprehensive Fix & Enhancement Report` - 10 edges
7. `GROUP B — Mixed sklearn + PyTorch modules` - 9 edges
8. `scripts` - 5 edges
9. `BiasVarianceSimulator()` - 4 edges
10. `GradientBoostingSimulator()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `GradientBoostingSimulator()` --calls--> `useApp()`  [EXTRACTED]
  src/components/Simulators/GradientBoostingSimulator.tsx → src/context/AppContext.tsx
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

## Communities (60 total, 35 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (33): dependencies, katex, lucide-react, prismjs, react, react-dom, react-simple-code-editor, tailwindcss (+25 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (28): 🔴 BUG-1 — `multiLayerPerceptron.ts` simulatorId mismatch, 🔴 BUG-2 — `generativeAdversarialNetworks.ts` missing `simulatorId`, 🔴 BUG-3 — `variationalAutoencoders.ts` missing `simulatorId`, 🔴 BUG-4 — `lucide-react` version `^1.16.0` does not exist, 🟡 BUG-5 — `w-4.5 h-4.5` are not valid Tailwind classes, 🟢 CLEAN-1 — `biasVarianceTradeoff.ts.draft` orphan file, 🟢 CLEAN-2 — Scratch/generator scripts in project root, `generativeAdversarialNetworks.ts` — Project uses `sklearn.neural_network.MLPClassifier` as discriminator proxy ✅ (+20 more)

### Community 2 - "Community 2"
Cohesion: 0.07
Nodes (28): 10. Gradient Descent — project solution (sklearn), 11. Decision Tree Algorithm (Tree Splits) — project solution (sklearn), 12. Gradient Boosting — project solution (sklearn), 13. Single Perceptron — project solution (sklearn), 14. Multi-Layer Perceptron — project solution (sklearn), 15. CNN — project solution (sklearn — new, see Part 5 above), 16. RNN — project solution (sklearn — text classification proxy), 17. Transformer Attention — project solution (sklearn + numpy) (+20 more)

### Community 3 - "Community 3"
Cohesion: 0.11
Nodes (14): AttentionMap(), DecisionTreeSimulator(), FeatureScalingSimulator(), GANSimulator(), InteractiveRegression(), KMeansSandbox(), KNNClassifierSimulator(), NaiveBayesSimulator() (+6 more)

### Community 4 - "Community 4"
Cohesion: 0.15
Nodes (16): CodingPlayground(), DIFFICULTY_COLORS, InterviewPrep(), KaggleNotebook(), stageLabels, NeuralBackground(), QuizEngine(), AppContext (+8 more)

### Community 5 - "Community 5"
Cohesion: 0.13
Nodes (11): anomalyDetection, gradientBoosting, introductionToMl, kaggleDigitRecognizerProject, kaggleDisasterTweetsProject, kaggleEdaProject, kaggleLoanDefaultRiskProject, kaggleStoreSalesForecastingProject (+3 more)

### Community 6 - "Community 6"
Cohesion: 0.11
Nodes (18): compilerOptions, allowImportingTsExtensions, erasableSyntaxOnly, jsx, lib, module, moduleDetection, moduleResolution (+10 more)

### Community 7 - "Community 7"
Cohesion: 0.11
Nodes (17): compilerOptions, allowImportingTsExtensions, erasableSyntaxOnly, lib, module, moduleDetection, moduleResolution, noEmit (+9 more)

### Community 8 - "Community 8"
Cohesion: 0.14
Nodes (9): linearRegression, pcaReduction, supportVectorMachines, tsneVisualization, CodingStage, InterviewQA, ModuleCoding, NotebookCell (+1 more)

### Community 9 - "Community 9"
Cohesion: 0.15
Nodes (7): modulesData, decisionTreeAlgorithm, decisionTrees, gradientDescent, kaggleTitanicProject, recurrentNetworks, MLModule

### Community 10 - "Community 10"
Cohesion: 0.17
Nodes (6): computeAccuracy(), defaultPoints, GradientBoostingSimulator(), LightGBMLeaf, Point, TreeNode

### Community 11 - "Community 11"
Cohesion: 0.29
Nodes (5): CLUSTER_COLORS, CLUSTER_LABELS, generateHDClusters(), Point2D, TSNESimulator()

### Community 12 - "Community 12"
Cohesion: 0.29
Nodes (4): BoundaryCell, DataPoint, EnsembleSimulator(), ModelType

### Community 13 - "Community 13"
Cohesion: 0.38
Nodes (6): computeCoefficients(), computeLoss(), FEATURE_NAMES, RegType, RegularizationSimulator(), TRUE_COEFFICIENTS

### Community 14 - "Community 14"
Cohesion: 0.47
Nodes (3): BiasVarianceSimulator(), fitPolynomial(), generateDataset()

### Community 15 - "Community 15"
Cohesion: 0.50
Nodes (3): Expanding the ESLint configuration, React Compiler, React + TypeScript + Vite

### Community 17 - "Community 17"
Cohesion: 0.50
Nodes (3): CNNSimulator(), defaultImage, filters

### Community 18 - "Community 18"
Cohesion: 0.50
Nodes (3): GradientDescent(), OptState, Point

### Community 20 - "Community 20"
Cohesion: 0.50
Nodes (3): datasets, DatasetType, PerceptronSimulator()

### Community 22 - "Community 22"
Cohesion: 0.50
Nodes (3): embeddings, RNNSimulator(), sequence

## Knowledge Gaps
- **155 isolated node(s):** `BeforeTool`, `name`, `private`, `version`, `type` (+150 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **35 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MLModule` connect `Community 9` to `Community 4`, `Community 5`, `Community 8`, `Community 30`, `Community 31`, `Community 32`, `Community 33`, `Community 34`, `Community 35`, `Community 36`, `Community 37`, `Community 38`, `Community 39`, `Community 40`, `Community 41`, `Community 42`, `Community 43`, `Community 44`, `Community 45`, `Community 46`, `Community 47`, `Community 48`, `Community 49`, `Community 50`, `Community 51`, `Community 52`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._
- **Why does `PART 6 — COMPLETE SKLEARN SOLUTIONS FOR ALL 27 MODULES` connect `Community 2` to `Community 1`?**
  _High betweenness centrality (0.016) - this node is a cross-community bridge._
- **Why does `Latent Academy — Comprehensive Fix & Enhancement Report` connect `Community 1` to `Community 2`?**
  _High betweenness centrality (0.015) - this node is a cross-community bridge._
- **What connects `BeforeTool`, `name`, `private` to the rest of the system?**
  _155 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.058823529411764705 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.06896551724137931 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.07142857142857142 - nodes in this community are weakly interconnected._