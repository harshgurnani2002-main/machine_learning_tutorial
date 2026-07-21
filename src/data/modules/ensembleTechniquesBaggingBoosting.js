export const ensembleTechniquesBaggingBoosting = {
    id: 'ensemble-techniques',
    title: 'Ensemble Techniques (Bagging & Boosting)',
    category: 'Supervised Learning',
    description: 'Master the fundamental difference between parallel (Bagging) and sequential (Boosting) ensemble methods — including AdaBoost, Gradient Boosting, and XGBoost with their mathematical foundations.',
    formula: 'F_M(x) = \\sum_{m=1}^{M} \\gamma_m h_m(x)',
    interactiveSummary: 'Use the Ensemble Simulator to visually compare how a Single Tree, Bagging (Random Forest), and Boosting (Gradient Boosting) create fundamentally different decision boundaries. Use the iteration slider to watch boosting iteratively sharpen its focus on misclassified regions.',
    simulatorId: 'ensemble-simulator',
    theory: `### Ensemble Techniques: Bagging & Boosting

#### What Are Ensemble Methods?
Ensemble learning combines multiple models (called **base learners** or **weak learners**) into a single, more powerful model. The core idea comes from the "wisdom of crowds" — a diverse group of moderately skilled people often outperforms even the best individual expert.

There are three main ensemble strategies:
1. **Bagging** (Bootstrap Aggregating) — train models in **parallel** on random subsets
2. **Boosting** — train models **sequentially**, each fixing the previous one's errors
3. **Stacking** — train a meta-learner on top of base learner predictions

#### The Bias-Variance Decomposition
To understand why ensembles work, we must understand the bias-variance tradeoff mathematically.

The expected prediction error for any model can be decomposed as:

$$\\text{Error} = \\text{Bias}^2 + \\text{Variance} + \\text{Irreducible Noise}$$

- **Bias**: Error from wrong assumptions. A shallow decision tree (stump) has high bias.
- **Variance**: Error from sensitivity to training data fluctuations. A very deep tree has high variance.
- **Bagging reduces Variance** — it averages many high-variance models, smoothing out their individual noise.
- **Boosting reduces Bias** — it iteratively corrects the systematic errors of weak learners.

#### Bagging (Bootstrap Aggregating)

**Bootstrap Sampling**: Given a dataset $D$ of $n$ samples, create $B$ bootstrap samples $D_1, D_2, ..., D_B$, each of size $n$, drawn **with replacement**. Each bootstrap sample includes approximately $63.2\\%$ of the original data (the rest are out-of-bag):

$$P(\\text{not selected}) = \\left(1 - \\frac{1}{n}\\right)^n \\xrightarrow{n \\to \\infty} e^{-1} \\approx 0.368$$

**The Bagging Algorithm**:
1. For $b = 1$ to $B$:
   - Draw bootstrap sample $D_b$ from $D$
   - Train model $f_b$ on $D_b$
2. **Aggregate**:
   - **Classification**: Majority vote $\\hat{y} = \\text{mode}(f_1(x), f_2(x), ..., f_B(x))$
   - **Regression**: Average $\\hat{y} = \\frac{1}{B} \\sum_{b=1}^B f_b(x)$

**Out-Of-Bag (OOB) Error**: Each tree sees ~63.2% of training data. The remaining ~36.8% (OOB samples) can be used to evaluate each tree without a separate validation set. Average OOB error across all trees gives a free, unbiased performance estimate.

**Variance of the Ensemble**: Let $\\rho$ be the pairwise correlation between trees and $\\sigma^2$ be the single-tree variance:

$$\\text{Var}\\left(\\bar{f}\\right) = \\rho\\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2$$

As $B \\to \\infty$, the second term vanishes. Variance is bounded by $\\rho\\sigma^2$. **Random Forests** reduce $\\rho$ further by only considering $\\sqrt{p}$ random features at each split.

#### Random Forests — Bagging Applied
Random Forest is Bagging with two key innovations:
1. **Bootstrap sampling** of training data (standard Bagging)
2. **Random feature subsets**: At each split, only a random subset of $\\sqrt{p}$ features (for classification) or $p/3$ features (for regression) are considered

This forces de-correlation between trees — they cannot all rely on the same strong predictor. The combined prediction is far more stable and accurate than any single tree.

#### Boosting — Sequential Error Correction

Boosting trains models sequentially, with each new model focusing on the mistakes of the previous ensemble.

**AdaBoost (Adaptive Boosting)**:
1. Initialize equal sample weights: $w_i = \\frac{1}{n}$ for all $i$
2. For $m = 1$ to $M$:
   - Train weak learner $h_m$ on weighted dataset
   - Compute weighted error: $\\epsilon_m = \\frac{\\sum_i w_i \\cdot \\mathbf{1}[y_i \\neq h_m(x_i)]}{\\sum_i w_i}$
   - Compute learner weight: $\\alpha_m = \\frac{1}{2} \\ln \\frac{1 - \\epsilon_m}{\\epsilon_m}$
   - Update sample weights: $w_i \\leftarrow w_i \\cdot \\exp(-\\alpha_m y_i h_m(x_i))$
   - Renormalize weights
3. Final prediction: $F(x) = \\text{sign}\\left(\\sum_{m=1}^{M} \\alpha_m h_m(x)\\right)$

Misclassified samples get higher weights → next model focuses on them.

**Gradient Boosting — Functional Gradient Descent**:
Gradient Boosting frames boosting as **gradient descent in function space**. Instead of adjusting weights, it fits each new tree to the **negative gradient of the loss function** (i.e., the residuals):

1. Initialize with a constant prediction: $F_0(x) = \\arg\\min_\\gamma \\sum_i L(y_i, \\gamma)$
2. For $m = 1$ to $M$:
   - Compute pseudo-residuals (negative gradient):
     $$r_{im} = -\\left[\\frac{\\partial L(y_i, F(x_i))}{\\partial F(x_i)}\\right]_{F=F_{m-1}}$$
   - Fit a tree $h_m$ to the residuals $\\{r_{im}\\}$
   - Update the ensemble: $F_m(x) = F_{m-1}(x) + \\eta \\cdot h_m(x)$
   
where $\\eta$ is the **learning rate** (shrinkage parameter). A small $\\eta$ requires more trees but reduces overfitting.

For **MSE loss** (regression), the pseudo-residuals are simply the actual residuals: $r_i = y_i - F_{m-1}(x_i)$

**XGBoost — Regularized Gradient Boosting**:
XGBoost adds second-order Taylor expansion of the loss and explicit L1/L2 tree regularization:

$$\\text{Obj} = \\sum_i L(y_i, \\hat{y}_i) + \\sum_k \\Omega(f_k)$$

where $\\Omega(f) = \\gamma T + \\frac{1}{2}\\lambda \\|w\\|^2$ ($T$ = number of leaves, $w$ = leaf weights).

Key innovations over vanilla Gradient Boosting:
- **Histogram-based splitting**: Pre-sort features into histograms → much faster split finding
- **Column subsampling**: Random features per tree (like Random Forests) → reduces overfitting
- **Second-order gradients**: Uses both gradient and Hessian for better step size
- **Regularization**: $\\gamma$ and $\\lambda$ control tree complexity directly

**LightGBM** takes this further with **Gradient-based One-Side Sampling (GOSS)** and **Exclusive Feature Bundling (EFB)** for even faster training on large datasets.

#### Stacking (Stacked Generalization)
Stacking trains a **meta-learner** (often Logistic Regression) on the out-of-fold predictions of base models. The meta-learner learns how to best combine base model outputs.

$$\\hat{y}_{\\text{stack}} = g(f_1(x), f_2(x), ..., f_K(x))$$

where $g$ is the meta-learner and $f_k$ are the base learners. This often outperforms individual models and simple averaging.

#### Summary Comparison

| Method | Trains Models | Targets | Key Effect | Example |
|---|---|---|---|---|
| Bagging | Parallel | Random subsets | Reduces Variance | Random Forest |
| Boosting | Sequential | Residuals/errors | Reduces Bias | XGBoost, AdaBoost |
| Stacking | Parallel + Meta | Full dataset | Combines strengths | Kaggle Blending |

#### Python Implementation

\`\`\`python
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, random_state=42)
bag = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10)
ada = AdaBoostClassifier(n_estimators=50)
print(f"Bagging: {bag.fit(X, y).score(X, y):.3f}")
print(f"Boosting: {ada.fit(X, y).score(X, y):.3f}")
\`\`\`
`,
    quiz: [
        {
            id: 'ens_q1',
            question: 'Which fundamental statistical property does Bagging primarily reduce?',
            options: [
                'Bias — by correcting systematic errors',
                'Variance — by averaging many diverse models',
                'Irreducible noise — by filtering outliers',
                'Overfitting — by limiting tree depth'
            ],
            correctAnswer: 'Variance — by averaging many diverse models',
            explanation: 'Bagging trains many high-variance models (like fully grown trees) and averages them. The variance of an average of B correlated models is ρσ² + (1-ρ)σ²/B, which shrinks as B grows. Boosting, not Bagging, targets bias reduction.'
        },
        {
            id: 'ens_q2',
            question: 'In AdaBoost, what happens to the weight of a sample that was MISCLASSIFIED by the previous learner?',
            options: [
                'Its weight is set to zero and it is excluded from the next tree',
                'Its weight is increased so the next learner focuses on it',
                'Its weight is decreased to prevent overfitting',
                'Its weight stays the same — only the model weight changes'
            ],
            correctAnswer: 'Its weight is increased so the next learner focuses on it',
            explanation: 'AdaBoost updates weights as: w_i ← w_i * exp(-α * y_i * h(x_i)). For a misclassified sample, y_i * h(x_i) = -1, so the exponent is +α, increasing the weight. This forces the next weak learner to prioritize those hard examples.'
        },
        {
            id: 'ens_q3',
            question: 'What does the "learning rate" (η) control in Gradient Boosting, and what is the typical tradeoff?',
            options: [
                'The maximum depth of each tree; smaller depth = less overfitting',
                'The step size when updating the ensemble; smaller η = more trees needed but better generalization',
                'The fraction of features used per tree; similar to Random Forests',
                'The number of samples in each bootstrap; larger samples = less variance'
            ],
            correctAnswer: 'The step size when updating the ensemble; smaller η = more trees needed but better generalization',
            explanation: 'In F_m(x) = F_{m-1}(x) + η * h_m(x), a small learning rate η (e.g., 0.01-0.1) makes smaller updates, requiring more trees (n_estimators) to reach the same training error, but it reduces overfitting significantly. This η vs n_estimators tradeoff is the most important hyperparameter relationship in Gradient Boosting.'
        },
        {
            id: 'ens_q4',
            question: 'What is an Out-Of-Bag (OOB) sample in a Random Forest?',
            options: [
                'A sample that was incorrectly predicted by the forest',
                'The 36.8% of training samples not selected in a given bootstrap — used as free validation',
                'A test sample completely unseen during training',
                'A sample that is used for cross-validation only'
            ],
            correctAnswer: 'The 36.8% of training samples not selected in a given bootstrap — used as free validation',
            explanation: 'Bootstrap sampling selects n samples with replacement from n training points. The probability of any single sample not being selected is (1-1/n)^n → e^{-1} ≈ 36.8% as n grows. These OOB samples are used to estimate error for each tree without a separate validation set.'
        },
        {
            id: 'ens_q5',
            question: 'Why does XGBoost outperform standard Gradient Boosting in practice?',
            options: [
                'XGBoost uses deeper trees and more estimators by default',
                'XGBoost adds regularization terms (γ, λ), uses second-order gradients, and histogram-based splits for speed',
                'XGBoost uses k-fold cross-validation internally during training',
                'XGBoost can only be used for classification, making it more specialized'
            ],
            correctAnswer: 'XGBoost adds regularization terms (γ, λ), uses second-order gradients, and histogram-based splits for speed',
            explanation: 'XGBoost\'s objective includes Ω(f) = γT + λ||w||² which directly penalizes tree complexity. Second-order (Hessian) information gives better step size approximations. Histogram-based splitting dramatically reduces computation from O(n) to O(bins) per feature. These combine to make it faster, more regularized, and more accurate than vanilla GBDT.'
        },
        {
            id: 'ens_q6',
            question: 'In a stacking ensemble, what is the role of the meta-learner?',
            options: [
                'It replaces all base models and trains on the original features',
                'It learns how to optimally combine the predictions of the base learners',
                'It selects which base learner to use on a per-sample basis',
                'It applies cross-validation to the base learners\' training process'
            ],
            correctAnswer: 'It learns how to optimally combine the predictions of the base learners',
            explanation: 'In stacking, base learners (e.g., RF, XGBoost, SVM) produce out-of-fold predictions on the training set. The meta-learner (often Logistic Regression) is trained on these predictions to learn their relative strengths. At test time, base predictions are fed to the meta-learner for the final output — it learns that one base model is more reliable for certain regions of the feature space.'
        }
    ],
    coding: {
        tutorial: {
            title: 'AdaBoost vs Gradient Boosting Comparison',
            description: 'Train both AdaBoost and Gradient Boosting on a binary classification dataset. Compare their accuracy and understand the impact of the number of estimators.',
            pseudoCode: `1. Load Breast Cancer dataset
2. Split 80/20 train/test
3. Train AdaBoostClassifier(n_estimators=100)
4. Train GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
5. Print accuracy for both
6. Show how performance changes with n_estimators`,
            starterCode: `from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TODO: Train AdaBoostClassifier(n_estimators=100, random_state=42)
# ada = ...
# ada.fit(...)

# TODO: Train GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
# gbc = ...
# gbc.fit(...)

# TODO: Print accuracy for both models
# print("AdaBoost Accuracy:", ...)
# print("Gradient Boosting Accuracy:", ...)

# Bonus: Compare staged predictions (how accuracy evolves per iteration)
# ada_scores = [accuracy_score(y_test, p) for p in ada.staged_predict(X_test)]
# print("Ada Final:", ada_scores[-1], "Best:", max(ada_scores))`,
            expectedOutput: `AdaBoost Accuracy:`,
            solution: `from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)

gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbc.fit(X_train, y_train)

print("AdaBoost Accuracy:", accuracy_score(y_test, ada.predict(X_test)))
print("Gradient Boosting Accuracy:", accuracy_score(y_test, gbc.predict(X_test)))

ada_scores = [accuracy_score(y_test, p) for p in ada.staged_predict(X_test)]
print("Ada Final:", ada_scores[-1], "Best:", max(ada_scores))`,
            hints: [
                'AdaBoostClassifier(n_estimators=100, random_state=42)',
                'GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)',
                'staged_predict() yields predictions at each boosting iteration',
                'max_depth=3 is the standard shallow tree used in GBM'
            ],
            testKeywords: ['AdaBoostClassifier', 'GradientBoostingClassifier', 'accuracy_score', 'staged_predict', 'n_estimators']
        },
        project: {
            title: 'XGBoost on Titanic with Hyperparameter Tuning',
            description: 'Train an XGBoost model on a synthetic Titanic-like dataset. Use GridSearchCV to tune key hyperparameters (learning_rate, max_depth, n_estimators) and report the best configuration.',
            pseudoCode: `1. Create synthetic Titanic-like features (age, fare, pclass, sex_enc, family_size)
2. Split 80/20
3. Define XGBClassifier
4. Define param_grid: learning_rate, max_depth, n_estimators
5. Run GridSearchCV(cv=3, scoring='accuracy')
6. Print best params and best score`,
            starterCode: `import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Simulate Titanic-like dataset
np.random.seed(42)
n = 800
df = pd.DataFrame({
    'pclass': np.random.choice([1, 2, 3], n, p=[0.2, 0.3, 0.5]),
    'age': np.random.normal(30, 12, n).clip(1, 80),
    'fare': np.random.exponential(30, n),
    'sex_enc': np.random.choice([0, 1], n),  # 0=male, 1=female
    'family_size': np.random.choice([1,2,3,4,5], n, p=[0.4,0.3,0.15,0.1,0.05])
})
# Survival probability based on features
prob = (0.1 + 0.3*(df.sex_enc) + 0.2*(df.pclass==1) - 0.1*(df.age>50)).clip(0.05, 0.95)
df['survived'] = (np.random.rand(n) < prob).astype(int)

X = df.drop('survived', axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TODO: Import XGBClassifier from xgboost
# TODO: Define param_grid with at least:
#   'n_estimators': [50, 100],
#   'learning_rate': [0.05, 0.1, 0.2],
#   'max_depth': [3, 5]

# TODO: Run GridSearchCV(XGBClassifier(random_state=42, eval_metric='logloss'), param_grid, cv=3)
# TODO: Fit and print best_params_ and best_score_
# TODO: Evaluate on test set`,
            expectedOutput: `Best params:`,
            solution: `import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

np.random.seed(42)
n = 800
df = pd.DataFrame({
    'pclass': np.random.choice([1, 2, 3], n, p=[0.2, 0.3, 0.5]),
    'age': np.random.normal(30, 12, n).clip(1, 80),
    'fare': np.random.exponential(30, n),
    'sex_enc': np.random.choice([0, 1], n),
    'family_size': np.random.choice([1,2,3,4,5], n, p=[0.4,0.3,0.15,0.1,0.05])
})
prob = (0.1 + 0.3*(df.sex_enc) + 0.2*(df.pclass==1) - 0.1*(df.age>50)).clip(0.05, 0.95)
df['survived'] = (np.random.rand(n) < prob).astype(int)
X = df.drop('survived', axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5]
}

xgb = XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
search = GridSearchCV(xgb, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
search.fit(X_train, y_train)

print("Best params:", search.best_params_)
print("Best CV accuracy:", search.best_score_)
print("Test accuracy:", accuracy_score(y_test, search.predict(X_test)))`,
            hints: [
                'from xgboost import XGBClassifier',
                'GridSearchCV(estimator, param_grid, cv=3, scoring=\'accuracy\')',
                'search.best_params_ shows the winning combination',
                'Use n_jobs=-1 to parallelize the search across CPU cores'
            ],
            testKeywords: ['XGBClassifier', 'GridSearchCV', 'best_params_', 'best_score_', 'accuracy_score']
        },
        assignment: {
            title: 'Build a Stacking Ensemble from Scratch',
            description: 'Implement a stacking ensemble using out-of-fold predictions. Combine Random Forest, Gradient Boosting, and KNN as base learners with Logistic Regression as the meta-learner. Compare stacking against each base learner individually.',
            pseudoCode: `1. Load Wine dataset
2. Use StratifiedKFold(5) for out-of-fold predictions
3. For each base model (RF, GBC, KNN):
   a. Generate 5-fold OOF predictions on train set
   b. Generate predictions on test set (average across folds)
4. Stack OOF predictions as new features → train meta-learner (LR)
5. Stack test predictions → meta-learner predicts
6. Compare: RF alone, GBC alone, KNN alone, Stacking`,
            starterCode: `import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Base learners
base_models = [
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('KNN', KNeighborsClassifier(n_neighbors=5))
]

# TODO: Split into train/test (80/20, stratified)

# TODO: For each base model, compute cross_val_score on X_train and print accuracy

# --- Build Stacking ---
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# TODO: For each base model:
#   - Create oof_preds array of shape (len(X_train), n_classes)
#   - For each fold: fit on train_idx, predict_proba on val_idx
#   - Also predict_proba on X_test and average across folds

# TODO: Stack OOF predictions → train meta-learner LR
# TODO: Stack test predictions → evaluate
# TODO: Print comparison table`,
            expectedOutput: `Random Forest CV:`,
            solution: `import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.2, stratify=y, random_state=42)

base_models = [
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('KNN', KNeighborsClassifier(n_neighbors=5))
]

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_classes = len(np.unique(y))

oof_preds_list = []
test_preds_list = []

for name, model in base_models:
    oof = np.zeros((len(X_train), n_classes))
    test_fold_preds = []
    
    for train_idx, val_idx in kf.split(X_train, y_train):
        Xtr, Xval = X_train[train_idx], X_train[val_idx]
        ytr = y_train[train_idx]
        model.fit(Xtr, ytr)
        oof[val_idx] = model.predict_proba(Xval)
        test_fold_preds.append(model.predict_proba(X_test))
    
    solo_acc = accuracy_score(y_train, np.argmax(oof, axis=1))
    print(f"{name} OOF accuracy: {solo_acc:.4f}")
    
    oof_preds_list.append(oof)
    test_preds_list.append(np.mean(test_fold_preds, axis=0))

# Stack
X_meta_train = np.hstack(oof_preds_list)
X_meta_test = np.hstack(test_preds_list)

meta = LogisticRegression(max_iter=500, random_state=42)
meta.fit(X_meta_train, y_train)
stack_acc = accuracy_score(y_test, meta.predict(X_meta_test))
print(f"Stacking Test accuracy: {stack_acc:.4f}")`,
            hints: [
                'oof array shape: (len(X_train), n_classes) for multi-class proba',
                'Average test predictions across folds: np.mean(test_fold_preds, axis=0)',
                'Stack with np.hstack(oof_preds_list) to get meta-features',
                'The meta-learner trains on OOF predictions, not the original features'
            ],
            testKeywords: ['StratifiedKFold', 'predict_proba', 'LogisticRegression', 'accuracy_score', 'hstack']
        }
    },
    interviewQuestions: [
        {
            question: 'Explain the difference between Bagging and Boosting in terms of what error they reduce.',
            answer: 'Bagging reduces Variance. It trains multiple models in parallel on different bootstrap samples and averages them. The variance of the average is ρσ² + (1-ρ)σ²/B — averaging reduces the (1-ρ)/B term and Random Feature Selection reduces ρ. This is why a single deep tree (high variance) + Bagging = Random Forest (low variance). Boosting reduces Bias. It trains models sequentially, each correcting the residuals of the previous ensemble. An ensemble of weak learners (high bias stumps) + Boosting = a strong learner with much lower bias. The tradeoff: Bagging is safer (harder to overfit), Boosting can overfit if you add too many estimators without regularization.',
            companyTags: ['Google', 'Meta', 'Amazon'],
            difficulty: 'Medium'
        },
        {
            question: 'What is the learning rate in Gradient Boosting and what is its tradeoff with n_estimators?',
            answer: 'The learning rate η (also called shrinkage) scales each tree\'s contribution: F_m = F_{m-1} + η * h_m. Smaller η means smaller steps, requiring more trees (n_estimators) to reach the same training error. However, smaller η with more trees typically generalizes better because each tree corrects less aggressively. This prevents any single tree from dominating the ensemble. The standard practice is to use η = 0.01-0.1 and tune n_estimators with early stopping on a validation set. In XGBoost: use n_estimators=10000 with early_stopping_rounds=50 to find the optimal number of trees automatically.',
            companyTags: ['Netflix', 'Spotify', 'Airbnb'],
            difficulty: 'Hard'
        },
        {
            question: 'When would you use XGBoost vs LightGBM vs CatBoost?',
            answer: 'All three are gradient boosting frameworks but with different strengths: XGBoost: Most widely used, great default performance, excellent documentation, best for medium datasets (up to ~1M rows). LightGBM: Fastest training on large datasets (100M+ rows) due to GOSS (Gradient-based One-Side Sampling) and EFB (Exclusive Feature Bundling). Uses leaf-wise (best-first) tree growth instead of level-wise. Preferred for Kaggle competitions with huge data. CatBoost: Best for high-cardinality categorical features (uses ordered target statistics, no need for manual encoding). Most resistant to overfitting by design. Handles missing values natively. As a rule of thumb: start with XGBoost, switch to LightGBM for speed on big data, use CatBoost when you have many categorical columns.',
            companyTags: ['Yandex', 'Microsoft', 'Kaggle'],
            difficulty: 'Hard'
        }
    ]
};
