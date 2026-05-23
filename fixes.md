# Latent Academy — Comprehensive Fix & Enhancement Report

> Audited: `machine_learning_tutorial` (React 19 + TypeScript + Vite)  
> 27 Modules · 5 simulators imported · 1 MonsterSimulator file (4400+ lines)

---

## SEVERITY LEGEND

| Symbol | Meaning |
|--------|---------|
| 🔴 **CRITICAL** | Broken / wrong output right now |
| 🟡 **IMPORTANT** | Degraded UX / missing feature |
| 🟢 **ENHANCEMENT** | Clean-up / quality improvement |

---

## PART 1 — SIMULATOR BUGS (Interactive Simulator tab broken for 3 modules)

### 🔴 BUG-1 — `multiLayerPerceptron.ts` simulatorId mismatch

**File:** `src/data/modules/multiLayerPerceptron.ts`  
**Problem:** Module declares `simulatorId: 'mlp'`, but the router switch in `ModuleSimulator.tsx` has **no `case 'mlp':`** — it only has `case 'neural-network':`. When you click the Interactive Simulator tab for Multi-Layer Perceptron, it falls to the `default` branch and renders `<InteractiveRegression />` (a scatter plot for Linear Regression). Totally wrong.

**Fix A — Easiest (1 line):** Change the simulatorId in the module data to match the existing `NeuralNetSandbox`:
```ts
// src/data/modules/multiLayerPerceptron.ts  line ~69
simulatorId: 'neural-network',   // was: 'mlp'
```

**Fix B — Correct (add missing case):** Add the case to the router and wire it to `NeuralNetSandbox`:
```tsx
// src/components/Simulators/ModuleSimulator.tsx — inside the switch
case 'mlp':
  return <NeuralNetSandbox />;
```
Fix B is more explicit and avoids coupling two different module IDs to the same string. Prefer Fix B.

---

### 🔴 BUG-2 — `generativeAdversarialNetworks.ts` missing `simulatorId`

**File:** `src/data/modules/generativeAdversarialNetworks.ts`  
**Problem:** The module object has **no `simulatorId` field at all**. `activeModule.simulatorId` resolves to `undefined`, so `ModuleSimulator` receives `simulatorId=""`, hits the `default` case, and renders `<InteractiveRegression />` instead of `<GANSimulator />`.

**Fix:** Add the field:
```ts
// src/data/modules/generativeAdversarialNetworks.ts — after the `interactiveSummary` field
simulatorId: 'gan',
```

---

### 🔴 BUG-3 — `variationalAutoencoders.ts` missing `simulatorId`

**File:** `src/data/modules/variationalAutoencoders.ts`  
**Problem:** Same as BUG-2. No `simulatorId` field → renders `<InteractiveRegression />` instead of `<VAESimulator />`.

**Fix:**
```ts
// src/data/modules/variationalAutoencoders.ts — after the `interactiveSummary` field
simulatorId: 'vae',
```

---

## PART 2 — DEPENDENCY ISSUES

### 🔴 BUG-4 — `lucide-react` version `^1.16.0` does not exist

**File:** `package.json`  
**Problem:** `lucide-react` uses a `0.x.y` versioning scheme. Version `1.16.0` does not exist on npm. The current stable release is in the `0.4xx` range. This means:
- `npm install` in a fresh clone will fail with `No matching version found`
- Anyone who clones the repo cannot build it
- The `dist/` in your zip built on your machine only because `node_modules` was cached

**Fix:** Pin to the actual latest stable version:
```json
// package.json
"lucide-react": "^0.511.0"
```
Then run `npm install` again. All icon imports (`Sparkles`, `BookOpen`, `Activity`, etc.) are identical — no source changes needed.

---

### 🟡 WARN-1 — `framer-motion` installed but never imported

**File:** `package.json`  
**Problem:** `framer-motion` (a large animation library, ~50KB gzipped) is in `dependencies` but is not imported anywhere in the source code. It inflates the bundle for no reason.

**Fix:** Remove it from `package.json` and run `npm install`:
```json
// Remove this line from "dependencies":
"framer-motion": "^12.40.0"
```

---

## PART 3 — TAILWIND / CSS BUGS

### 🟡 BUG-5 — `w-4.5 h-4.5` are not valid Tailwind classes

**File:** `src/App.tsx` — line 348  
**Problem:** Tailwind's default spacing scale uses integers (`w-4` = 16px, `w-5` = 20px). There is no `w-4.5` or `h-4.5` in Tailwind v4's pre-defined utilities. These classes silently compile to nothing — the Sparkles icon in the header logo gets the browser's default size instead of the intended 18px.

**Fix:** Use an arbitrary value or the nearest scale step:
```tsx
// Option A — arbitrary value (exact 18px):
<Sparkles className="w-[18px] h-[18px] text-[#FAF6EE]" />

// Option B — nearest scale (20px):
<Sparkles className="w-5 h-5 text-[#FAF6EE]" />
```

---

## PART 4 — ORPHAN / JUNK FILES

### 🟢 CLEAN-1 — `biasVarianceTradeoff.ts.draft` orphan file

**File:** `src/data/modules/biasVarianceTradeoff.ts.draft`  
**Problem:** This file is a leftover scratch artifact containing only:
```ts
export const biasVarianceTradeoff = {
    // ... we will use write_to_file or a python script to generate it
};
```
The real module (`biasVarianceTradeoff.ts`) exists and is fully complete. The `.draft` file is never imported but sits in the modules directory and may confuse your IDE or linter.

**Fix:** Delete the file:
```bash
rm src/data/modules/biasVarianceTradeoff.ts.draft
```

---

### 🟢 CLEAN-2 — Scratch/generator scripts in project root

**Files in root:** `generate_gd.py`, `generate_dt.py`, `generate_lr.py`, `generate_logr.py`, `generate_gb.py`, `generate_ml_modules.py`, `update_naive_bayes.py`, `format_ts.py`, `fix_lines.py`, `fix_types.py`, `split.ts`, `expand.ts`, `gradientDescent_updated.json`, `decisionTreeAlgorithm_updated.json`, `gradientBoosting_updated.json`  
**Problem:** These are one-off data generation scripts used during authoring. They have no runtime role and clutter the repo root.

**Fix:** Move them to a `scripts/` or `tooling/` directory, or add them to `.gitignore` if they won't be needed again. At minimum, add to `.gitignore`:
```
/generate_*.py
/update_*.py
/format_ts.py
/fix_*.py
/split.ts
/expand.ts
/*_updated.json
/scratch/
```

---

## PART 5 — CODING SANDBOX — SKLEARN PROJECT & ASSIGNMENT AUDIT

All 27 modules have been audited. **All modules have `tutorial`, `project`, and `assignment` coding stages with `starterCode`, `solution`, `testKeywords`, and `hints` defined.** Sklearn is correctly used in the project/assignment stages for every module that can reasonably use it.

Below is the complete breakdown of which stages use sklearn vs PyTorch, with the full sklearn-based code included for every module that needs it.

---

### GROUP A — Pure sklearn modules (project + assignment use sklearn)
These are complete and correct. No changes needed.

| Module | Project Uses | Assignment Uses |
|--------|-------------|----------------|
| Linear Regression | `LinearRegression`, `StandardScaler`, `r2_score` | `Ridge`, `Pipeline`, `train_test_split` |
| Logistic Regression | `LogisticRegression`, `StandardScaler` | `LogisticRegression`, `GridSearchCV` |
| Decision Trees | `DecisionTreeClassifier` | `DecisionTreeClassifier`, `cross_val_score` |
| Random Forests | `RandomForestClassifier` | `RandomForestClassifier`, `feature_importances_` |
| K-Means Clustering | `KMeans`, `StandardScaler` | `KMeans`, `TfidfVectorizer` |
| SVM | `SVC`, `StandardScaler` | `SVC`, `GridSearchCV`, `classification_report` |
| Naive Bayes | `GaussianNB` | `MultinomialNB`, `TfidfVectorizer` |
| KNN | `KNeighborsClassifier` | `KNeighborsClassifier`, `cross_val_score` |
| PCA | `PCA`, `StandardScaler` | `PCA`, `LogisticRegression`, Pipeline |
| Gradient Descent | `SGDRegressor` | `SGDClassifier`, `learning_curve` |
| Decision Tree Algorithm | `DecisionTreeClassifier`, `export_text` | `DecisionTreeClassifier`, `plot_tree` |
| Gradient Boosting | `GradientBoostingClassifier` | `GradientBoostingClassifier`, `XGBClassifier` |
| Bias-Variance Tradeoff | `PolynomialFeatures`, `LinearRegression` | `PolynomialFeatures`, `validation_curve` |
| DBSCAN | `DBSCAN`, `StandardScaler` | `DBSCAN`, silhouette analysis |
| Ridge/Lasso | `Ridge`, `Lasso`, `ElasticNet` | `RidgeCV`, `LassoCV`, `cross_val_score` |
| t-SNE | `TSNE`, `StandardScaler` | `TSNE`, `PCA`, scatter comparison |
| Neural Activations | `MLPClassifier` | `MLPClassifier`, activation comparison |
| Optimization Algorithms | `SGDClassifier`, `learning_rate` | `SGDClassifier`, custom LR schedule |

---

### GROUP B — Mixed sklearn + PyTorch modules

These modules correctly use sklearn for the **project** stage and PyTorch for deeper stages where sklearn has no equivalent. The `project` stage (middle difficulty) always has a sklearn path.

#### `multiLayerPerceptron.ts` — Project uses `sklearn.neural_network.MLPClassifier` ✅
```python
# CURRENT PROJECT starterCode (correct — already uses sklearn):
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = ...  # Moon dataset features
y = ...  # Binary labels

# TODO: Scale, split, fit MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500), predict
```

#### `singlePerceptron.ts` — Project uses `sklearn.linear_model.Perceptron` ✅
```python
# CURRENT PROJECT starterCode (correct):
from sklearn.linear_model import Perceptron
```

#### `recurrentNetworks.ts` — Project uses `sklearn` sequence classification proxy ✅
```python
# CURRENT PROJECT starterCode (correct — uses sklearn text features as proxy):
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
```

#### `wordEmbeddings.ts` — Project uses `sklearn` cosine similarity ✅
```python
# CURRENT PROJECT starterCode (correct):
from sklearn.metrics.pairwise import cosine_similarity
```

#### `transformerAttention.ts` — Project uses `sklearn` attention approximation ✅
```python
# CURRENT PROJECT starterCode (correct — uses sklearn for scaled dot-product proxy):
from sklearn.preprocessing import normalize
import numpy as np
```

#### `qLearningRl.ts` — Project uses numpy-only Q-table ✅  
(sklearn has no RL primitives; numpy Q-table is the correct choice)

#### `generativeAdversarialNetworks.ts` — Project uses `sklearn.neural_network.MLPClassifier` as discriminator proxy ✅
```python
# CURRENT PROJECT starterCode (correct — uses sklearn MLP as discriminator):
from sklearn.neural_network import MLPClassifier
```

#### `variationalAutoencoders.ts` — Project uses `sklearn.decomposition.PCA` as linear analog ✅
```python
# CURRENT PROJECT starterCode (correct — notes sklearn PCA as VAE precursor):
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
```

---

### GROUP C — CNN module needs sklearn alternative added to project stage 🟡

**File:** `src/data/modules/convolutionalNetworks.ts`  
**Problem:** The CNN module is the **only module** where neither the project nor the assignment stage uses sklearn at all. Both use raw PyTorch (`torch.nn`). This is fine for the assignment, but the project stage should offer a sklearn-equivalent path so the module fits the consistent platform convention (Project = sklearn, Assignment = PyTorch).

**Fix — Add sklearn project stage alternative for CNN:**

Replace the current project stage `starterCode` in `convolutionalNetworks.ts` with a sklearn `MLPClassifier` approximation that simulates what a CNN does conceptually (flattened image classification), and move the PyTorch CNN to the assignment:

```python
# NEW PROJECT starterCode — sklearn MLPClassifier on flattened images
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Simulated flattened 16x16 grayscale images (sklearn cannot use 2D convolutions,
# so we flatten first — this is exactly what a CNN does before its Dense head)
np.random.seed(42)
X = np.random.rand(80, 256)   # 80 images, 16x16 = 256 features each
y = np.array([0]*20 + [1]*20 + [2]*20 + [3]*20)  # 4 classes

# TODO: Scale X with StandardScaler
# TODO: Split 75/25 with train_test_split(random_state=42)
# TODO: Fit MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
#         max_iter=500, random_state=42) on training data
# TODO: Predict on X_test and print accuracy rounded to 2 decimal places

scaler = None
model = None
print("Accuracy:", 0.0)
```

```python
# NEW PROJECT solution:
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X = np.random.rand(80, 256)
y = np.array([0]*20 + [1]*20 + [2]*20 + [3]*20)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=500, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, preds), 2))
```

```python
# Update testKeywords for the new project stage:
testKeywords: ['MLPClassifier', 'StandardScaler', 'train_test_split', 'accuracy_score']
```

Also update the `project.description` to:
> "Use `sklearn.neural_network.MLPClassifier` to classify simulated flattened image vectors — this mirrors the Dense head of a CNN after spatial features have been extracted and flattened. sklearn has no Conv2d layers; the project stage demonstrates the classification component using sklearn, while the assignment upgrades the full pipeline to PyTorch."

---

## PART 6 — COMPLETE SKLEARN SOLUTIONS FOR ALL 27 MODULES

Below is the full sklearn-based `project` solution code for every module, ready to paste into the relevant `solution:` fields. All solutions have been verified against the existing `testKeywords` in each module.

---

### 1. Linear Regression — project solution (sklearn)
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

X = np.array([
    [8.3, 6.98, 41.0, 322],
    [8.3, 7.0, 21.0, 2401],
    [5.64, 4.94, 52.0, 496],
    [4.58, 6.05, 52.0, 558],
    [3.85, 4.88, 52.0, 565],
    [6.13, 5.94, 34.0, 1200],
    [3.12, 4.76, 42.0, 890],
    [7.65, 6.35, 18.0, 1100],
])
y = np.array([4.526, 3.585, 3.521, 3.413, 3.422, 2.697, 2.992, 4.100])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LinearRegression()
model.fit(X_scaled, y)
predictions = model.predict(X_scaled)
print("R^2 Score:", round(r2_score(y, predictions), 2))
```

---

### 2. Logistic Regression — project solution (sklearn)
```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X = np.array([
    [25, 50000], [30, 60000], [35, 75000], [40, 90000],
    [45, 110000], [50, 130000], [55, 150000], [60, 170000],
    [22, 30000], [28, 45000],
])
y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression(random_state=42)
model.fit(X_scaled, y)
predictions = model.predict(X_scaled)
print("Accuracy:", round(accuracy_score(y, predictions), 2))
```

---

### 3. Decision Trees — project solution (sklearn)
```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = np.array([
    [5.1,3.5],[4.9,3.0],[6.7,3.1],[6.3,3.3],
    [5.8,2.7],[7.1,3.0],[6.3,2.9],[6.5,3.0],
    [5.0,3.4],[4.7,3.2],
])
y = np.array([0,0,2,2,1,2,1,2,0,0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, predictions), 2))
```

---

### 4. Random Forests — project solution (sklearn)
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = np.array([
    [62,1,1,120,268,0,0,160,0,3.6],
    [67,1,4,160,286,0,2,108,1,1.5],
    [57,1,4,120,354,0,0,163,1,0.6],
    [63,1,4,130,254,0,2,147,0,1.4],
    [53,1,4,140,203,1,2,155,1,3.1],
    [56,0,2,140,294,0,2,153,0,1.3],
    [48,1,2,130,245,0,2,180,0,0.2],
    [58,1,3,150,270,0,2,111,1,0.8],
    [41,0,2,130,204,0,2,172,0,1.4],
    [60,1,4,140,293,0,2,170,0,1.2],
])
y = np.array([1,1,1,1,1,0,0,1,0,0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, predictions), 2))
importances = model.feature_importances_
print("Top Feature Index:", importances.argmax())
```

---

### 5. K-Means Clustering — project solution (sklearn)
```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X = np.array([
    [25, 30000], [30, 45000], [22, 28000],
    [45, 90000], [50, 120000], [55, 95000],
    [35, 65000], [28, 55000], [40, 75000],
    [60, 150000], [58, 140000], [52, 110000],
    [48, 100000], [33, 60000], [27, 40000],
])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = KMeans(n_clusters=3, n_init=10, random_state=42)
model.fit(X_scaled)
labels = model.labels_
centroids_original = scaler.inverse_transform(model.cluster_centers_)
print("Labels:", labels)
print("Centroids (original scale):", centroids_original.round(0))
```

---

### 6. SVM — project solution (sklearn)
```python
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X = np.array([
    [0.1, 0.2],[0.4, 0.1],[0.2, 0.3],[0.35, 0.25],
    [0.8, 0.9],[0.9, 0.7],[0.7, 0.85],[0.85, 0.8],
    [0.5, 0.5],[0.45, 0.55],
])
y = np.array([0,0,0,0,1,1,1,1,0,1])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions, zero_division=0))
```

---

### 7. Naive Bayes — project solution (sklearn)
```python
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = np.array([
    [5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2],[4.7,3.2,1.3,0.2],
    [7.0,3.2,4.7,1.4],[6.4,3.2,4.5,1.5],[6.9,3.1,4.9,1.5],
    [6.3,3.3,6.0,2.5],[5.8,2.7,5.1,1.9],[7.1,3.0,5.9,2.1],
    [5.0,3.6,1.4,0.2],[5.4,3.9,1.7,0.4],[6.7,3.0,5.0,1.7],
])
y = np.array([0,0,0,1,1,1,2,2,2,0,0,1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, predictions), 2))
```

---

### 8. KNN Classification — project solution (sklearn)
```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = np.array([
    [1.0, 2.0],[1.5, 1.8],[1.2, 2.1],[3.0, 3.5],
    [3.5, 3.0],[3.2, 3.3],[5.0, 1.0],[5.5, 0.8],
    [5.2, 1.3],[2.0, 2.5],[4.0, 2.0],[2.5, 3.0],
])
y = np.array([0,0,0,1,1,1,2,2,2,0,2,1])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, predictions), 2))
```

---

### 9. PCA — project solution (sklearn)
```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = np.array([
    [2.5, 2.4, 3.1],[0.5, 0.7, 1.2],[2.2, 2.9, 2.8],[1.9, 2.2, 2.0],
    [3.1, 3.0, 3.5],[2.3, 2.7, 2.6],[2.0, 1.6, 1.9],[1.0, 1.1, 1.3],
    [1.5, 1.6, 1.7],[1.1, 0.9, 1.0],
])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)
print("Original shape:", X_scaled.shape)
print("Reduced shape:", X_reduced.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_.round(2))
```

---

### 10. Gradient Descent — project solution (sklearn)
```python
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]], dtype=float)
y = np.array([1.5, 3.2, 4.8, 6.1, 8.0, 9.5, 11.3, 13.0, 14.7, 16.2])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = SGDRegressor(learning_rate='constant', eta0=0.01, max_iter=1000, random_state=42)
model.fit(X_scaled, y)
predictions = model.predict(X_scaled)
rmse = np.sqrt(mean_squared_error(y, predictions))
print(f"RMSE: {rmse:.2f}")
print(f"Coef: {model.coef_[0]:.2f}, Intercept: {model.intercept_[0]:.2f}")
```

---

### 11. Decision Tree Algorithm (Tree Splits) — project solution (sklearn)
```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

X = np.array([
    [2.5,1.0],[1.0,4.5],[3.5,2.0],[4.0,1.5],
    [1.5,3.5],[3.0,3.0],[4.5,4.0],[2.0,2.5],
    [1.0,1.0],[4.0,4.0],
])
y = np.array([0,1,0,0,1,1,1,0,1,1])

model = DecisionTreeClassifier(max_depth=2, criterion='gini', random_state=42)
model.fit(X, y)
predictions = model.predict(X)
print("Accuracy:", round(accuracy_score(y, predictions), 2))
print(export_text(model, feature_names=['x1','x2']))
```

---

### 12. Gradient Boosting — project solution (sklearn)
```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = np.array([
    [23,1,0,1],[45,0,1,0],[31,1,1,1],[52,0,0,0],
    [29,1,0,0],[38,0,1,1],[47,1,1,0],[25,0,0,1],
    [34,1,0,1],[41,0,1,0],[27,1,1,0],[55,0,0,1],
])
y = np.array([0,1,0,1,0,1,1,0,0,1,0,1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, predictions), 2))
```

---

### 13. Single Perceptron — project solution (sklearn)
```python
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X = np.array([
    [0,0],[0,1],[1,0],[1,1],
    [2,2],[2,3],[3,2],[3,3],
])
y = np.array([0,0,0,0,1,1,1,1])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = Perceptron(max_iter=1000, random_state=42)
model.fit(X_scaled, y)
predictions = model.predict(X_scaled)
print("Accuracy:", round(accuracy_score(y, predictions), 2))
print("Weights:", model.coef_)
print("Bias:", model.intercept_)
```

---

### 14. Multi-Layer Perceptron — project solution (sklearn)
```python
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Moons-like dataset
np.random.seed(42)
angles = np.linspace(0, np.pi, 30)
X0 = np.column_stack([np.cos(angles), np.sin(angles)]) + 0.1*np.random.randn(30, 2)
X1 = np.column_stack([1 - np.cos(angles), 1 - np.sin(angles) - 0.5]) + 0.1*np.random.randn(30, 2)
X = np.vstack([X0, X1])
y = np.array([0]*30 + [1]*30)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=500, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, predictions), 2))
```

---

### 15. CNN — project solution (sklearn — new, see Part 5 above)
```python
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X = np.random.rand(80, 256)   # 80 images, 16x16 = 256 flattened features
y = np.array([0]*20 + [1]*20 + [2]*20 + [3]*20)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=500, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, preds), 2))
```

---

### 16. RNN — project solution (sklearn — text classification proxy)
```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sentiment proxy dataset (TF-IDF features simulate text sequence encoding)
texts = [
    "great movie loved it","amazing performance","enjoyed the story",
    "wonderful film highly recommend","brilliant acting superb",
    "terrible movie hated it","boring film waste of time",
    "awful performance disappointing","did not enjoy poor story","bad film skip it",
]
y = np.array([1,1,1,1,1,0,0,0,0,0])

vectorizer = TfidfVectorizer(max_features=20)
X = vectorizer.fit_transform(texts)
model = LogisticRegression(random_state=42, max_iter=200)
model.fit(X, y)
predictions = model.predict(X)
print("Accuracy:", round(accuracy_score(y, predictions), 2))
print("Vocabulary size:", len(vectorizer.vocabulary_))
```

---

### 17. Transformer Attention — project solution (sklearn + numpy)
```python
import numpy as np
from sklearn.preprocessing import normalize

# Scaled dot-product attention simulation
np.random.seed(42)
seq_len, d_k = 5, 4

Q = normalize(np.random.randn(seq_len, d_k), axis=1)
K = normalize(np.random.randn(seq_len, d_k), axis=1)
V = np.random.randn(seq_len, d_k)

# Scaled dot-product
scores = Q @ K.T / np.sqrt(d_k)

# Softmax
exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
attention_weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)

output = attention_weights @ V
print("Attention weights shape:", attention_weights.shape)
print("Output shape:", output.shape)
print("Sum of attention row 0:", round(attention_weights[0].sum(), 4))
```

---

### 18. Word Embeddings — project solution (sklearn cosine similarity)
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Simulated word vectors (GloVe-like 50-dim)
np.random.seed(7)
words = ['king', 'queen', 'man', 'woman', 'prince', 'paris', 'france', 'london']
embeddings = {w: np.random.randn(50) for w in words}
# Inject semantic structure
embeddings['queen'] = embeddings['king'] - embeddings['man'] + embeddings['woman']
embeddings['prince'] = embeddings['king'] * 0.8

E = np.array(list(embeddings.values()))
sim_matrix = cosine_similarity(E)

# king - man + woman ≈ queen
analogy = embeddings['king'] - embeddings['man'] + embeddings['woman']
sims = cosine_similarity([analogy], E)[0]
best_idx = sims.argsort()[-2]  # exclude king itself
print("king - man + woman ≈", words[best_idx])
print("Similarity:", round(sims[best_idx], 4))
```

---

### 19. Q-Learning RL — project solution (numpy Q-table)
```python
import numpy as np

# 4x4 grid world: 0=empty, 1=wall, 2=goal
grid = np.array([
    [0,0,0,0],
    [0,1,0,1],
    [0,0,0,0],
    [1,0,0,2]
])
n_states = 16
n_actions = 4  # up, down, left, right
Q = np.zeros((n_states, n_actions))

alpha, gamma, epsilon = 0.1, 0.9, 0.3
goal_state = 15
actions = [(-1,0),(1,0),(0,-1),(0,1)]

def step(r, c, a):
    nr, nc = r + actions[a][0], c + actions[a][1]
    if nr < 0 or nr > 3 or nc < 0 or nc > 3 or grid[nr,nc] == 1:
        return r, c, -1, False
    if grid[nr,nc] == 2:
        return nr, nc, 10, True
    return nr, nc, -0.1, False

for ep in range(500):
    r, c = 0, 0
    for _ in range(50):
        s = r*4+c
        a = np.random.randint(4) if np.random.rand() < epsilon else Q[s].argmax()
        nr, nc, reward, done = step(r, c, a)
        ns = nr*4+nc
        Q[s,a] += alpha*(reward + gamma*Q[ns].max() - Q[s,a])
        r, c = nr, nc
        if done: break

print("Q-table max per state:", Q.max(axis=1).round(2))
print("Best action from start:", ['up','down','left','right'][Q[0].argmax()])
```

---

### 20. GAN — project solution (sklearn MLP discriminator proxy)
```python
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

np.random.seed(42)

# Real data distribution: Gaussian centered at (2, 2)
real_data = np.random.randn(100, 2) + 2

# Generator: random noise -> fake samples (starts far from real)
def generator(n, epoch):
    noise = np.random.randn(n, 2)
    # Simulate generator improving over epochs
    shift = min(epoch * 0.2, 2.0)
    return noise + shift

# Train discriminator to distinguish real vs fake
epochs_log = []
for epoch in range(1, 11):
    fake_data = generator(100, epoch)
    X_disc = np.vstack([real_data, fake_data])
    y_disc = np.array([1]*100 + [0]*100)

    disc = MLPClassifier(hidden_layer_sizes=(16,), max_iter=200, random_state=42)
    disc.fit(X_disc, y_disc)
    proba = disc.predict_proba(X_disc)
    loss = log_loss(y_disc, proba)
    epochs_log.append(round(loss, 3))

print("Discriminator loss per epoch:", epochs_log)
print("Final (converging to ~0.693 = random guess if GAN balanced):", epochs_log[-1])
```

---

### 21. VAE — project solution (sklearn PCA as linear analog)
```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Simulated high-dimensional data (images as flat vectors)
np.random.seed(42)
X = np.random.randn(120, 64)  # 120 samples, 64-dim (8x8 "images")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encoder: PCA to 2D latent space (like VAE encoder)
pca = PCA(n_components=2)
Z = pca.fit_transform(X_scaled)  # latent representation

# Decoder: inverse_transform (like VAE decoder)
X_reconstructed = pca.inverse_transform(Z)
X_reconstructed = scaler.inverse_transform(X_reconstructed)

reconstruction_error = np.mean((X - X_reconstructed) ** 2)
print("Latent space shape:", Z.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_.round(4))
print("Reconstruction MSE:", round(reconstruction_error, 4))
```

---

### 22. t-SNE — project solution (sklearn)
```python
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
# 3 clusters in 10-dimensional space
X_cluster0 = np.random.randn(30, 10) + 3
X_cluster1 = np.random.randn(30, 10) - 3
X_cluster2 = np.random.randn(30, 10) + np.array([3,-3,0,0,0,0,0,0,0,0])
X = np.vstack([X_cluster0, X_cluster1, X_cluster2])
y_true = np.array([0]*30 + [1]*30 + [2]*30)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
tsne = TSNE(n_components=2, perplexity=10, random_state=42, max_iter=300)
X_2d = tsne.fit_transform(X_scaled)

print("Reduced shape:", X_2d.shape)
print("X_2d range: [{:.2f}, {:.2f}]".format(X_2d.min(), X_2d.max()))
```

---

### 23. DBSCAN — project solution (sklearn)
```python
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
cluster_a = np.random.randn(30, 2) * 0.5 + [1, 1]
cluster_b = np.random.randn(30, 2) * 0.5 + [5, 5]
noise = np.array([[10, 10], [-5, -5], [8, 2]])
X = np.vstack([cluster_a, cluster_b, noise])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print("Clusters found:", n_clusters)
print("Noise points:", n_noise)
print("Labels:", labels)
```

---

### 24. Ridge/Lasso Regularization — project solution (sklearn)
```python
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

np.random.seed(42)
X = np.random.randn(50, 10)
true_coef = np.array([3.0, -2.0, 1.5, 0, 0, 0, 0, 0, 0, 0])
y = X @ true_coef + 0.5 * np.random.randn(50)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)
lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, y)

print("Ridge R2:", round(r2_score(y, ridge.predict(X_scaled)), 2))
print("Lasso R2:", round(r2_score(y, lasso.predict(X_scaled)), 2))
print("Ridge coef (first 4):", ridge.coef_[:4].round(2))
print("Lasso coef (first 4):", lasso.coef_[:4].round(2))
print("Lasso zero coefs:", (lasso.coef_ == 0).sum())
```

---

### 25. Bias-Variance Tradeoff — project solution (sklearn)
```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

np.random.seed(42)
X = np.sort(np.random.rand(30, 1) * 6 - 3, axis=0)
y = np.sin(X.ravel()) + 0.3 * np.random.randn(30)

X_train, y_train = X[:20], y[:20]
X_test, y_test = X[20:], y[20:]

for degree in [1, 3, 10]:
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('lr', LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, pipe.predict(X_train))
    test_mse = mean_squared_error(y_test, pipe.predict(X_test))
    print(f"Degree {degree}: Train MSE={train_mse:.4f}, Test MSE={test_mse:.4f}")
```

---

### 26. Neural Activations — project solution (sklearn)
```python
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)
X = np.random.randn(200, 4)
y = (X[:, 0] * X[:, 1] > 0).astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

for activation in ['relu', 'tanh', 'logistic']:
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation=activation,
        max_iter=300,
        random_state=42
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Activation '{activation}': Accuracy = {acc:.2f}")
```

---

### 27. Optimization Algorithms — project solution (sklearn SGD variants)
```python
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)
X = np.random.randn(300, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

configs = [
    ('SGD (constant)', 'constant', 1.0, None),
    ('SGD (invscaling)', 'invscaling', 0.01, None),
    ('SGD (adaptive)', 'adaptive', 0.1, None),
]

for name, lr_sched, eta0, momentum in configs:
    model = SGDClassifier(
        loss='hinge',
        learning_rate=lr_sched,
        eta0=eta0,
        max_iter=200,
        random_state=42,
        tol=1e-3
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: Accuracy = {acc:.2f}")
```

---

## PART 7 — SUMMARY CHECKLIST

| # | File | Issue | Severity | Fix |
|---|------|-------|----------|-----|
| 1 | `multiLayerPerceptron.ts` | simulatorId `'mlp'` has no switch case | 🔴 | Add `case 'mlp': return <NeuralNetSandbox />;` |
| 2 | `generativeAdversarialNetworks.ts` | Missing `simulatorId` field | 🔴 | Add `simulatorId: 'gan'` |
| 3 | `variationalAutoencoders.ts` | Missing `simulatorId` field | 🔴 | Add `simulatorId: 'vae'` |
| 4 | `package.json` | `lucide-react: ^1.16.0` doesn't exist | 🔴 | Change to `^0.511.0` |
| 5 | `App.tsx:348` | `w-4.5 h-4.5` invalid Tailwind classes | 🟡 | Replace with `w-[18px] h-[18px]` |
| 6 | `package.json` | `framer-motion` installed but never used | 🟡 | Remove from dependencies |
| 7 | `convolutionalNetworks.ts` | Project stage uses only PyTorch, no sklearn | 🟡 | Add sklearn MLPClassifier project stage (see Part 5) |
| 8 | `biasVarianceTradeoff.ts.draft` | Orphan draft file in modules directory | 🟢 | Delete file |
| 9 | Project root | 15+ generator/scratch scripts cluttering root | 🟢 | Move to `scripts/` or add to `.gitignore` |

---

## QUICK-START PATCH COMMANDS

```bash
# 1. Fix lucide-react version
npm install lucide-react@latest

# 2. Remove unused framer-motion
npm uninstall framer-motion

# 3. Delete orphan draft
rm src/data/modules/biasVarianceTradeoff.ts.draft

# 4-6. Edit these 3 files manually as described above:
#   src/data/modules/multiLayerPerceptron.ts     -> simulatorId: 'neural-network'
#   src/data/modules/generativeAdversarialNetworks.ts  -> add simulatorId: 'gan'
#   src/data/modules/variationalAutoencoders.ts  -> add simulatorId: 'vae'
#   src/App.tsx:348                              -> w-[18px] h-[18px]

# Rebuild after all fixes
npm run build
```
