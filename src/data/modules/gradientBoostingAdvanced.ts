import type { MLModule } from '../../types';

export const gradientBoostingAdvanced: MLModule = {
  id: 'gradient-boosting-advanced',
  title: 'Advanced Gradient Boosting',
  category: 'Supervised Learning',
  description: 'Unleash state-of-the-art tree frameworks: XGBoost, LightGBM, and CatBoost for competitive datasets.',
  formula: 'F_M(x) = \\sum_{m=1}^M \\gamma_m h_m(x)',
  theory: `### Advanced Gradient Boosting: XGBoost, LightGBM, & CatBoost

#### What is it?
In competitive machine learning (e.g., Kaggle competitions) and industry tabular data applications, gradient boosted decision trees (GBDT) are the dominant paradigm. While neural networks rule image and text processing, boosted trees consistently outperform them on structured, tabular data.
Modern GBDT libraries—**XGBoost** (Extreme Gradient Boosting), **LightGBM** (Light Gradient Boosting Machine), and **CatBoost** (Categorical Boosting)—are highly optimized implementations of gradient boosting that introduce unique architectures, split criteria, and categorical handling to achieve state-of-the-art speed and accuracy.

#### Why do we need it?
1. **Unbeatable Baseline:** GBDTs natively handle mixed feature types (numerical and categorical), missing values, and unscaled inputs without heavy pre-processing.
2. **Speed & Efficiency:** Classic Gradient Boosting is computationally slow. Modern libraries use histogram-based splitting or symmetric trees to train on millions of rows in seconds.
3. **Overfitting Control:** They include built-in regularization (L1/L2 penalties on leaf weights) and early stopping, which halts training when validation scores plateau.

#### XGBoost vs. LightGBM vs. CatBoost

| Feature / Library | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| **Tree Growth** | Depth-wise (Level-wise) | Leaf-wise (Best-first) | Symmetric Trees (Oblivious) |
| **Categorical Support** | One-Hot / Partitioning | Numerical indexing | Native Target Statistics (Ordered Boosting) |
| **Split Strategy** | Pre-sorted / Histogram | Histogram-based | Histogram-based |
| **Speed** | Fast | Extremely Fast | Fast (GPU optimized) |

- **XGBoost:** Grows trees level-by-level (depth-wise). It uses a pre-sorted algorithm and histogram-based algorithm to find splits and applies second-order Taylor expansion on the loss function.
- **LightGBM:** Grows trees leaf-wise (best-first). It splits the leaf that minimizes the loss the most, regardless of tree depth. This achieves much lower loss but is highly prone to overfitting on small datasets unless \`max_depth\` is constrained.
- **CatBoost:** Uses symmetric (oblivious) trees, meaning the same split criterion is used at all nodes of a given level. This provides regularized predictions and makes inference incredibly fast. It also uses "Ordered Boosting" to prevent target leakage in categorical columns.

#### The Math Behind It

In GBDT, we iteratively add trees to minimize the loss. Given a loss function $L(y, F(x))$:

**1. Residual Estimation:**
At iteration $m$, we compute the negative gradient (pseudo-residuals) with respect to the previous ensemble $F_{m-1}(x)$:
$$r_{im} = - \\left[ \frac{\\partial L(y_i, F(x_i))}{\\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}$$

**2. XGBoost Objective Function (Second-Order Taylor Approximation):**
Instead of just first-order gradients, XGBoost uses second-order gradients (Hessians) to compute optimal leaf values:
$$\\mathcal{L}^{(m)} \\approx \\sum_{i=1}^n \\left[ L(y_i, F_{m-1}(x_i)) + g_i f_m(x_i) + \frac{1}{2} h_i f_m^2(x_i) \right] + \\Omega(f_m)$$
where:
- $g_i = \frac{\\partial L(y_i, F_{m-1}(x_i))}{\\partial F_{m-1}(x_i)}$ (Gradient)
- $h_i = \frac{\\partial^2 L(y_i, F_{m-1}(x_i))}{\\partial F_{m-1}^2(x_i)}$ (Hessian)
- $\\Omega(f_m) = \\gamma T + \frac{1}{2} \\lambda \\sum_{j=1}^T w_j^2$ is the regularization penalty on $T$ leaves and $w_j$ leaf weights.

The optimal weight $w_j^*$ for leaf $j$ containing sample indices $I_j$ is:
$$w_j^* = - \frac{\\sum_{i \\in I_j} g_i}{\\sum_{i \\in I_j} h_i + \\lambda}$$

#### Hyperparameter Cheat Sheet
- **num_leaves:** (LightGBM) Max leaves per tree. Set $< 2^{max\\_depth}$ to prevent overfitting.
- **learning_rate:** Shrinks the contribution of each tree. Keep it low (0.01 - 0.1) and increase trees.
- **subsample / colsample_bytree:** Column and row subsampling ratios to prevent overfitting.
- **min_child_weight:** Minimum sum of instance weight (Hessian) needed in a child leaf.

#### Worked Example: Gradient Splitting
Suppose we are predicting salary residuals.
- Left node has gradients sum $G_L = -10$, Hessian sum $H_L = 4$.
- Right node has gradients sum $G_R = 12$, Hessian sum $H_R = 5$.
- Using $\\lambda = 1.0$:
  - Left Leaf weight: $w_L^* = -(-10) / (4 + 1.0) = 2.0$
  - Right Leaf weight: $w_R^* = -(12) / (5 + 1.0) = -2.0$

#### Common Pitfalls
1. **Under-tuning LightGBM leaves:** Leaf-wise growth can grow very deep trees, leading to severe overfitting. You must set \`max_depth\` or control \`num_leaves\` and \`min_data_in_leaf\`.
2. **Ignoring Categorical Indices:** Standard libraries treat integers as ordered numerical values. You must explicitly mark categorical columns using categorical features parameters.

#### Python Implementation

\`\`\`python
import xgboost as xgb
import numpy as np

X = np.random.randn(100, 5)
y = np.random.randn(100)
dtrain = xgb.DMatrix(X, label=y)
params = {"max_depth": 4, "eta": 0.1, "objective": "reg:squarederror"}
model = xgb.train(params, dtrain, num_boost_round=50)
predictions = model.predict(dtrain)
print(f"RMSE: {np.sqrt(np.mean((predictions - y)**2)):.4f}")
\`\`\`
`,
  interactiveSummary: 'This simulator lets you visualize advanced tree boosting. See how individual shallow decision trees are added sequentially to target residuals of previous trees. Change model parameters to see how learning rate and tree depth control model variance.',
  simulatorId: 'gradient-boosting-advanced',
  quiz: [
    {
      id: 'gb_q1',
      question: 'Which tree growth strategy is used by LightGBM?',
      options: [
        'Leaf-wise (Best-first) growth.',
        'Depth-wise (Level-wise) growth.',
        'Symmetric oblivious tree growth.',
        'Balanced binary tree growth.'
      ],
      correctAnswer: 'Leaf-wise (Best-first) growth.',
      explanation: 'LightGBM grows trees leaf-wise, splitting the leaf that yields the maximum loss reduction. This is highly efficient but requires depth constraints to prevent overfitting.'
    },
    {
      id: 'gb_q2',
      question: 'What is the role of Hessians in XGBoost leaf weight calculation?',
      options: [
        'They represent second-order derivatives that scale the gradient steps, acting as a Newton-Raphson update.',
        'They are used to normalize feature inputs.',
        'They act as a regularization penalty.',
        'They determine the random state of the tree split.'
      ],
      correctAnswer: 'They represent second-order derivatives that scale the gradient steps, acting as a Newton-Raphson update.',
      explanation: 'XGBoost uses a second-order Taylor expansion on the loss function. The Hessians (second derivatives) scale the gradients (first derivatives) to compute optimal leaf weights directly.'
    },
    {
      id: 'gb_q3',
      question: 'How does CatBoost prevent target leakage when encoding categorical variables?',
      options: [
        'By using Ordered Boosting (computing statistics on shuffled historical rows).',
        'By running one-hot encoding on all categorical features.',
        'By automatically removing categorical columns.',
        'By standardizing the categories before tree splits.'
      ],
      correctAnswer: 'By using Ordered Boosting (computing statistics on shuffled historical rows).',
      explanation: 'CatBoost implements Ordered Boosting, which calculates target statistics using only the historical data points preceding a sample in a random permutation, preventing training set leakage.'
    },
    {
      id: 'gb_q4',
      question: 'What happens when early stopping is enabled during boosting?',
      options: [
        'Training stops when performance on validation data stops improving for a specified number of epochs.',
        'The model automatically cuts off feature counts.',
        'The learning rate is reduced to zero.',
        'The trees are pruned starting from the roots.'
      ],
      correctAnswer: 'Training stops when performance on validation data stops improving for a specified number of epochs.',
      explanation: 'Early stopping monitors a validation set. If validation loss does not decrease for `early_stopping_rounds`, training stops to prevent overfitting.'
    },
    {
      id: 'gb_q5',
      question: 'Which hyperparameter controls the contribution size of each tree in the ensemble?',
      options: [
        'learning_rate (or shrinkage)',
        'max_depth',
        'subsample',
        'colsample_bytree'
      ],
      correctAnswer: 'learning_rate (or shrinkage)',
      explanation: 'The learning rate (often called shrinkage or eta) scales the weights of new trees. A smaller learning rate requires more trees but results in better generalization.'
    }
  ],
  coding: {
    tutorial: {
      title: 'Pseudo-Residual Computation',
      description: 'Implement a vectorized function to compute pseudo-residuals for regression under squared error loss ($L = \frac{1}{2}(y - \\hat{y})^2$).',
      pseudoCode: '1. Accept true values y and predictions y_pred\n2. Compute derivative dL/dy_pred = (y_pred - y)\n3. Residual is -dL/dy_pred = (y - y_pred)\n4. Return residuals',
      starterCode: `import numpy as np

def compute_residuals(y, y_pred):
    # TODO: Calculate pseudo-residuals (y - y_pred)
    return np.zeros_like(y)

y_true = np.array([10.0, 15.0, 20.0])
y_p = np.array([8.5, 16.0, 19.5])
print("Residuals:", compute_residuals(y_true, y_p))`,
      expectedOutput: 'Residuals: [ 1.5 -1.   0.5]',
      solution: `import numpy as np

def compute_residuals(y, y_pred):
    return y - y_pred

y_true = np.array([10.0, 15.0, 20.0])
y_p = np.array([8.5, 16.0, 19.5])
print("Residuals:", compute_residuals(y_true, y_p))`,
      hints: [
        'Subtract y_pred from y.',
        'Under squared loss, the residual is exactly the prediction error: y - y_pred.'
      ],
      testKeywords: [
        'y - y_pred'
      ]
    },
    project: {
      title: 'Scikit-Learn Histogram Boosting',
      description: 'Train a `HistGradientBoostingClassifier` (scikit-learn\'s highly optimized LightGBM equivalent) with early stopping enabled.',
      pseudoCode: '1. Instantiate HistGradientBoostingClassifier(early_stopping=True, random_state=42).\n2. Fit model to X_train, y_train.\n3. Predict on X_test.\n4. Print model accuracy.',
      starterCode: `import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Tabular Dataset (subset of real values)
np.random.seed(42)
X = np.random.rand(150, 6)
y = (X[:, 0] + X[:, 1]*0.5 + X[:, 2]*0.2 > 0.8).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TODO: Initialize HistGradientBoostingClassifier with early_stopping=True, random_state=42
# TODO: Fit model and predict on X_test
# TODO: Print final test accuracy

model = None
preds = []
print("Accuracy:", 0.0)`,
      expectedOutput: 'Accuracy: 0.93',
      solution: `import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)
X = np.random.rand(150, 6)
y = (X[:, 0] + X[:, 1]*0.5 + X[:, 2]*0.2 > 0.8).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = HistGradientBoostingClassifier(early_stopping=True, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, preds), 2))`,
      hints: [
        'Set early_stopping=True in constructor.',
        'Use model.fit(X_train, y_train) followed by model.predict(X_test).'
      ],
      testKeywords: [
        'HistGradientBoostingClassifier',
        'early_stopping=True',
        'fit',
        'predict'
      ]
    },
    assignment: {
      title: 'GridSearch Hyperparameter Optimization',
      description: 'Run hyperparameter optimization using `GridSearchCV` on `HistGradientBoostingClassifier` to find the best tree depth and learning rate.',
      pseudoCode: '1. Create hyperparameter grid (max_iter, max_depth, learning_rate).\n2. Set up GridSearchCV with 3-fold cross validation.\n3. Fit grid search on training data.\n4. Print best parameters.',
      starterCode: `import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

np.random.seed(42)
X = np.random.rand(100, 4)
y = (X[:, 0] * X[:, 1] > 0.25).astype(int)

# TODO: Instantiate HistGradientBoostingClassifier(random_state=42)
# TODO: Define param_grid: {'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
# TODO: Run GridSearchCV with cv=3
# TODO: Print grid.best_params_

model = None
grid = None
print("Best Params: {}")`,
      expectedOutput: "Best Params: {'learning_rate': 0.1, 'max_depth': 3}",
      solution: `import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

np.random.seed(42)
X = np.random.rand(100, 4)
y = (X[:, 0] * X[:, 1] > 0.25).astype(int)

model = HistGradientBoostingClassifier(random_state=42)
param_grid = {'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid.fit(X, y)
print("Best Params:", grid.best_params_)`,
      hints: [
        'Pass estimator and param_grid to GridSearchCV.',
        'Ensure cv=3 is specified.',
        'Call grid.fit(X, y) and read grid.best_params_.'
      ],
      testKeywords: [
        'GridSearchCV',
        'param_grid',
        'fit',
        'best_params_'
      ]
    }
  },
  interviewQuestions: [
    {
      question: 'Explain the difference between Leaf-wise tree growth and Depth-wise tree growth. Which libraries use which?',
      answer: 'Depth-wise tree growth (used by XGBoost) splits all nodes at a given depth level simultaneously. Leaf-wise tree growth (used by LightGBM) chooses the single node with the highest loss reduction (best-first) and splits it, regardless of its depth level. Leaf-wise reduces loss faster but is more prone to overfitting unless controlled by max_depth constraints.',
      companyTags: ['Kaggle', 'Microsoft'],
      difficulty: 'Advanced'
    },
    {
      question: 'How does XGBoost utilize second-order optimization (Taylor expansion) during training?',
      answer: 'Classic boosting libraries use only first-order gradients. XGBoost uses both first-order gradients (Jacobian) and second-order gradients (Hessians) of the loss function. This Taylor expansion allows the algorithm to calculate the optimal leaf weights directly in a single step (Newton-Raphson update), stabilizing training on custom loss functions.',
      companyTags: ['Google', 'Meta'],
      difficulty: 'Hard'
    },
    {
      question: 'What are Oblivious Trees, and which boosting library uses them?',
      answer: 'Oblivious trees (used by CatBoost) are symmetric trees where the same splitting criterion (feature + threshold) is applied across all nodes at the same tree depth. This makes the trees highly regularized, reduces prediction variance, and enables extremely fast GPU-based inference since splits can be represented as simple bit masks.',
      companyTags: ['CatBoost', 'Yandex'],
      difficulty: 'Advanced'
    }
  ]
};
