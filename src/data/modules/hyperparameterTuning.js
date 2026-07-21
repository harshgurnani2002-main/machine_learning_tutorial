export const hyperparameterTuning = {
    id: 'hyperparameter-tuning',
    title: 'Hyperparameter Tuning & Search',
    category: 'Advanced & MLOps',
    description: 'Grid search, random search, and Bayesian optimization for finding optimal model hyperparameters.',
    formula: 'x^* = \\arg\\max_{x \\in \\mathcal{X}} f(x)',
    theory: `### Hyperparameter Tuning & Search Theory
  
  **What is Hyperparameter Tuning?**
  Hyperparameters are configuration variables set before training begins — they define model architecture, optimization procedure, and regularization strategy. Unlike model parameters (weights, biases) which are learned from data, hyperparameters must be specified by the practitioner: learning rate, tree depth, regularization strength, number of estimators, kernel parameters, batch size. Tuning systematically searches for the hyperparameter configuration that minimizes validation error.
  
  **Why it matters**
  Hyperparameters dramatically affect model performance. A poorly chosen learning rate causes divergence or excruciatingly slow convergence. Excessive tree depth causes overfitting; insufficient depth causes underfitting. The gap between a poorly tuned and a well-tuned model can be enormous — often the difference between a state-of-the-art result and random guessing. Modern ML systems have dozens of hyperparameters, creating high-dimensional search spaces requiring principled strategies.
  
  **Grid Search**
  The simplest strategy: define a finite set of values for each hyperparameter, then exhaustively evaluate all combinations. With k-fold CV per combination, total fits = k × ∏ nⱼ. The curse of dimensionality is brutal — 10 values each for 5 HPs gives 100,000 combinations. Bergstra & Bengio (2012) proved grid search is strictly less efficient than random search because each HP is explored at only n^{1/d} distinct values in an n-point grid. Grid search wastes evaluations by fixing all but one dimension at a time.
  
  **Random Search**
  Sample hyperparameters uniformly from defined distributions — log-uniform for positive parameters (learning rate, regularization) and uniform for bounded parameters (tree depth, batch size). With n trials, each HP is explored at n distinct values regardless of dimensionality d. This is dramatically more efficient when the objective has low effective dimensionality (only 2–3 HPs matter), which is empirically common. Random search is embarrassingly parallel and trivially simple.
  
  **Bayesian Optimization (GP-based)**
  A sequential model-based approach. Build a probabilistic surrogate model of the objective f(hp), then use an acquisition function to decide where to sample next. The surrogate is typically a Gaussian Process (GP), providing both a predicted mean μ(hp) and uncertainty σ(hp):
  
  $$\\mu(y^*) = k(y^*, Y)K^{-1}f \\quad \\sigma^2(y^*) = k(y^*, y^*) - k(y^*, Y)K^{-1}k(Y, y^*)$$
  
  High uncertainty in unexplored regions drives exploration; high predicted mean drives exploitation.
  
  **Acquisition Functions**
  The acquisition function α(hp) scores candidates, balancing exploration vs exploitation. hp_{n+1} = argmax α(hp):
  
  - **Expected Improvement (EI)**: E[max(f(hp) - f*, 0)]. Most popular. Naturally balances exploration (potential improvement in uncertain regions) and exploitation (likely improvement where μ is high).
  - **Probability of Improvement (PI)**: Φ((μ(hp) - f* - ξ) / σ(hp)). Simpler but risks premature convergence.
  - **Upper Confidence Bound (UCB)**: μ(hp) + κ·σ(hp). Parameter κ explicitly controls the tradeoff.
  
  **Exploration vs Exploitation**
  Central to sequential decision-making under uncertainty. Exploitation greedily samples where μ is highest; exploration samples where σ is high to improve the surrogate. BO elegantly transitions from exploration to exploitation as the GP becomes confident.
  
  **Cross-Validation in Tuning**
  1. Partition data into training + held-out test set (never touch test set during tuning).
  2. Evaluate each HP config using k-fold CV on the training set.
  3. Select config with highest average CV score.
  4. Retrain on full training set with selected config.
  5. Evaluate once on the held-out test set.
  
  **Common Pitfalls**
  
  *Data Leakage*: Using test data during preprocessing (scaling, imputation, PCA) before CV. Always fit preprocessing inside the CV loop.
  
  *Overfitting the Validation Set*: Multiple comparisons across many trials can find a config that scores well on the validation fold by pure chance. The more trials, the higher the risk.
  
  *Nested Cross-Validation*: An outer loop (5-fold) provides unbiased performance estimation; an inner loop handles HP selection for each outer fold. This gives an honest estimate of the tuned model's generalization error.
  
  *Log Scales*: Learning rates, regularization strengths, and kernel bandwidths operate over orders of magnitude. Always search these on a log scale.
  
  **Practical Hyperparameters**
  
  | Model | Key HPs | Typical Range | Scale |
  |-------|---------|--------------|-------|
  | SGD / Adam | Learning rate η | [1e-5, 1e-1] | Log |
  | Decision Trees | max_depth, min_samples_split | [3, 20], [2, 50] | Linear |
  | Random Forest | n_estimators, max_features | [50, 1000], [sqrt, log2] | Linear |
  | XGBoost | lr, max_depth, subsample | [0.01, 0.3], [3, 10], [0.5, 1.0] | Log/Linear |
  | SVM (RBF) | C, γ | [1e-3, 1e3], [1e-4, 1e1] | Log |
  | Ridge / Lasso | α | [1e-4, 1e2] | Log |
  
  **Summary**
  - Grid search: deterministic, expensive; useful for coarse exploration on 1–3 HPs.
  - Random search: better default; simple, parallel, empirically efficient.
  - Bayesian optimization: sample-efficient; best for expensive evaluations (large models, large data).
  - Always use CV; never touch the test set until final evaluation.
  - Use log scales for HP ranges spanning orders of magnitude.
  
  #### Python Implementation
  
  \`\`\`python
  from sklearn.model_selection import GridSearchCV
  from sklearn.svm import SVC
  
  X, y = [[1,2],[2,3],[3,1],[6,5],[7,7],[8,6]], [0,0,0,1,1,1]
  params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
  grid = GridSearchCV(SVC(), params, cv=3)
  grid.fit(X, y)
  print(f"Best params: {grid.best_params_}")
  \`\`\`,
  `,
    interactiveSummary: 'A visual simulator comparing grid search, random search, and Bayesian optimization on a 2D hyperparameter loss surface. Watch grid points march in lockstep across the space, random points scatter unpredictably, or Bayesian points converge adaptively toward the optimum. Toggle between surface modes to see how each method handles single peaks, multiple optima, and elongated ridges. The regret metric tracks how each method approaches the true optimum over successive evaluations.',
    simulatorId: 'hyperparam-tuning',
    quiz: [
        { id: 'hp_q1', question: 'What is the key advantage of random search over grid search for hyperparameter tuning?', options: ['It explores each hyperparameter at more distinct values for the same budget', 'It always finds the global optimum', 'It does not require cross-validation', 'It is deterministic and reproducible'], correctAnswer: 'It explores each hyperparameter at more distinct values for the same budget', explanation: 'With n trials, random search explores each HP at n distinct values, while grid search only explores n^{1/d} values. This is more efficient when the objective has low effective dimensionality.' },
        { id: 'hp_q2', question: 'In Bayesian optimization, what does the acquisition function balance?', options: ['Exploration vs exploitation', 'Bias vs variance', 'Precision vs recall', 'Training speed vs accuracy'], correctAnswer: 'Exploration vs exploitation', explanation: 'Acquisition functions like EI, PI, and UCB balance exploring uncertain regions (high σ) against exploiting regions with high predicted performance (high μ).' },
        { id: 'hp_q3', question: 'What information does a Gaussian Process surrogate model provide that a simple regression does not?', options: ['A measure of predictive uncertainty (σ) at each point', 'The exact value of the objective at unobserved points', 'A closed-form derivative of the objective', 'The global minimum of the objective function'], correctAnswer: 'A measure of predictive uncertainty (σ) at each point', explanation: 'The GP provides both a predicted mean μ and a posterior uncertainty σ at any point. High σ in unexplored regions drives exploration.' },
        { id: 'hp_q4', question: 'Why is log-uniform sampling preferred over uniform sampling for learning rate?', options: ['Learning rate operates over multiple orders of magnitude', 'It is computationally cheaper', 'It guarantees finding the optimum', 'It reduces the number of required CV folds'], correctAnswer: 'Learning rate operates over multiple orders of magnitude', explanation: 'Learning rates like 0.0001, 0.001, 0.01, 0.1 span 4 orders of magnitude. Uniform sampling would waste most points in [0.5, 1.0] where learning rates rarely work.' },
        { id: 'hp_q5', question: 'What is the purpose of nested cross-validation in hyperparameter tuning?', options: ['To provide an unbiased estimate of the tuned model\'s performance', 'To reduce the number of hyperparameter evaluations needed', 'To eliminate the need for a test set', 'To automatically select the best search method'], correctAnswer: 'To provide an unbiased estimate of the tuned model\'s performance', explanation: 'Outer CV provides unbiased performance estimation; inner CV handles HP selection. This prevents the optimistic bias from multiple comparisons during tuning.' },
        { id: 'hp_q6', question: 'What is the primary risk of tuning too many hyperparameters with too many trials?', options: ['Overfitting the validation set', 'Underfitting the training data', 'Excessive data leakage', 'Numerical instability in the optimizer'], correctAnswer: 'Overfitting the validation set', explanation: 'With enough trials, random chance will produce a config that performs well on the validation fold by chance but generalizes poorly. This is overfitting the validation set.' },
        { id: 'hp_q7', question: 'In the context of hyperparameter tuning, what constitutes data leakage?', options: ['Using test set information during preprocessing before cross-validation', 'Training the model on too much data', 'Using too few cross-validation folds', 'Evaluating the model on the training data'], correctAnswer: 'Using test set information during preprocessing before cross-validation', explanation: 'Any preprocessing that uses test data information (scaling, imputation, feature selection) before CV leaks information, producing artificially high performance estimates.' },
        { id: 'hp_q8', question: 'Which acquisition function is defined as μ(hp) + κ·σ(hp) and has an explicit exploration parameter?', options: ['Upper Confidence Bound (UCB)', 'Expected Improvement (EI)', 'Probability of Improvement (PI)', 'Thompson Sampling'], correctAnswer: 'Upper Confidence Bound (UCB)', explanation: 'UCB = μ(hp) + κ·σ(hp). The parameter κ explicitly controls exploration: higher κ favors exploring uncertain regions; κ=0 reduces to pure exploitation.' },
    ],
    coding: {
        tutorial: {
            title: 'Grid Search from Scratch',
            description: 'Implement a function that performs exhaustive grid search over two hyperparameters (learning rate and regularization strength). Evaluate all combinations on a mock scoring function and return the best configuration.',
            pseudoCode: `function grid_search(lr_values, reg_values):
    best_score = -inf
    best_params = None
    for lr in lr_values:
        for reg in reg_values:
            score = evaluate_model(lr, reg)
            if score > best_score:
                best_score = score
                best_params = (lr, reg)
    return best_params, best_score`,
            starterCode: `import numpy as np

def evaluate_model(lr, reg):
    """Mock evaluation: higher is better. True optimum near lr=0.01, reg=0.1"""
    return np.exp(-((np.log10(lr) + 2)**2 + (np.log10(reg) + 1)**2) / 0.5)

def grid_search(lr_values, reg_values):
    # TODO: iterate over all combinations, track best
    best_params = None
    best_score = -np.inf
    # Your code here
    return best_params, best_score

lr_vals = [0.001, 0.01, 0.1, 1.0]
reg_vals = [0.001, 0.01, 0.1, 1.0]
params, score = grid_search(lr_vals, reg_vals)
print(f"Best: lr={params[0]}, reg={params[1]}, score={score:.3f}")`,
            expectedOutput: 'Best: lr=0.01, reg=0.1, score=1.000',
            solution: `import numpy as np

def evaluate_model(lr, reg):
    return np.exp(-((np.log10(lr) + 2)**2 + (np.log10(reg) + 1)**2) / 0.5)

def grid_search(lr_values, reg_values):
    best_params = None
    best_score = -np.inf
    for lr in lr_values:
        for reg in reg_values:
            score = evaluate_model(lr, reg)
            if score > best_score:
                best_score = score
                best_params = (lr, reg)
    return best_params, best_score

lr_vals = [0.001, 0.01, 0.1, 1.0]
reg_vals = [0.001, 0.01, 0.1, 1.0]
params, score = grid_search(lr_vals, reg_vals)
print(f"Best: lr={params[0]}, reg={params[1]}, score={score:.3f}")`,
            hints: ['Use nested for loops over lr_values and reg_values.', 'Call evaluate_model(lr, reg) for each combination.', 'Track the best score and corresponding params with an if statement.'],
            testKeywords: ['for', 'range', 'evaluate_model', 'best_score']
        },
        project: {
            title: 'Random Forest Tuning with RandomizedSearchCV',
            description: 'Use sklearn\'s RandomizedSearchCV to tune a RandomForestClassifier on a synthetic binary classification dataset. Search over n_estimators, max_depth, and min_samples_split. Report the best parameters and CV score.',
            pseudoCode: `# Step 1: Create synthetic dataset
X, y = make_classification(n_samples=500, n_features=10, random_state=42)

# Step 2: Define model and hyperparameter distribution
model = RandomForestClassifier(random_state=42)
param_dist = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

# Step 3: Run RandomizedSearchCV with 5-fold CV
search = RandomizedSearchCV(model, param_dist, n_iter=20, cv=5, scoring='accuracy')
search.fit(X, y)

# Step 4: Report results
print(search.best_params_)
print(search.best_score_)`,
            starterCode: `import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

X, y = make_classification(n_samples=500, n_features=10, random_state=42)

model = RandomForestClassifier(random_state=42)

param_dist = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

# TODO: Create RandomizedSearchCV with n_iter=20, cv=5, scoring='accuracy'
# TODO: Fit on X, y
# TODO: Print best_params_ and best_score_

search = None`,
            expectedOutput: "Best params: {'n_estimators': 200, 'min_samples_split': 2, 'max_depth': 10}\nBest CV accuracy: 0.932",
            solution: `import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

X, y = make_classification(n_samples=500, n_features=10, random_state=42)

model = RandomForestClassifier(random_state=42)

param_dist = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

search = RandomizedSearchCV(model, param_dist, n_iter=20, cv=5, scoring='accuracy', random_state=42)
search.fit(X, y)
print(f"Best params: {search.best_params_}")
print(f"Best CV accuracy: {search.best_score_:.3f}")`,
            hints: ['Import RandomizedSearchCV from sklearn.model_selection.', 'Pass model, param_dist, n_iter=20, cv=5, scoring=\'accuracy\' to RandomizedSearchCV.', 'Access best params via search.best_params_ and best score via search.best_score_.'],
            testKeywords: ['RandomizedSearchCV', 'best_params_', 'best_score_', 'fit']
        },
        assignment: {
            title: 'Simple Bayesian Optimizer with Gaussian Process',
            description: 'Implement a basic Bayesian optimizer using sklearn\'s GaussianProcessRegressor. Use Expected Improvement as the acquisition function to sequentially suggest the next hyperparameter to evaluate. Track how points converge toward the optimum over iterations.',
            pseudoCode: `# Step 1: Define objective, bounds, and GP
def objective(x):
    return -((x - 0.5)**2)  # max at x=0.5

gp = GaussianProcessRegressor(kernel=RBF(1.0), alpha=1e-6)

# Step 2: Initial random points
X = np.random.rand(3, 1)
y = objective(X)

# Step 3: Bayesian loop
for i in range(10):
    gp.fit(X, y)
    # Candidate points
    X_candidates = np.linspace(0, 1, 200).reshape(-1, 1)
    mu, sigma = gp.predict(X_candidates, return_std=True)
    # EI acquisition
    best_y = y.max()
    imp = mu - best_y - 0.01
    Z = imp / (sigma + 1e-9)
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    # Select next point
    x_next = X_candidates[ei.argmax()]
    X = np.vstack([X, [x_next]])
    y = np.append(y, objective([[x_next]]))`,
            starterCode: `import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm

def objective(x):
    """Simple 1D function with max at x=0.5"""
    return -((x - 0.5)**2) + 0.25

np.random.seed(42)

# Initial random points
X = np.random.rand(3, 1)
y = objective(X)

gp = GaussianProcessRegressor(kernel=RBF(1.0), alpha=1e-6)

# TODO: Implement the Bayesian optimization loop (10 iterations)
# For each iteration:
#   1. Fit GP on (X, y)
#   2. Generate candidate points: np.linspace(0, 1, 200).reshape(-1, 1)
#   3. Predict mu and sigma with return_std=True
#   4. Compute Expected Improvement
#   5. Select the point with maximum EI
#   6. Append to X and y

print("Final best x:", X[y.argmax()][0])
print("Final best y:", y.max())`,
            expectedOutput: 'Final best x: 0.498\nFinal best y: 0.250',
            solution: `import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm

def objective(x):
    return -((x - 0.5)**2) + 0.25

np.random.seed(42)
X = np.random.rand(3, 1)
y = objective(X)

gp = GaussianProcessRegressor(kernel=RBF(1.0), alpha=1e-6)

for i in range(10):
    gp.fit(X, y)
    X_candidates = np.linspace(0, 1, 200).reshape(-1, 1)
    mu, sigma = gp.predict(X_candidates, return_std=True)
    best_y = y.max()
    imp = mu - best_y - 0.01
    Z = imp / (sigma + 1e-9)
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    x_next = X_candidates[ei.argmax()].reshape(1, -1)
    y_next = objective(x_next)
    X = np.vstack([X, x_next])
    y = np.append(y, y_next)

print(f"Final best x: {X[y.argmax()][0]:.3f}")
print(f"Final best y: {y.max():.3f}")`,
            hints: ['Use gp.predict(X_candidates, return_std=True) to get mu and sigma.', 'EI formula: imp = mu - best_y - xi; Z = imp/(sigma+eps); ei = imp*norm.cdf(Z) + sigma*norm.pdf(Z).', 'Use X_candidates[ei.argmax()] to select the next evaluation point.'],
            testKeywords: ['GaussianProcessRegressor', 'predict', 'return_std', 'norm.cdf', 'norm.pdf', 'argmax']
        }
    },
    interviewQuestions: [
        { question: 'Compare grid search and random search for hyperparameter tuning. Under what conditions is one preferred over the other?', answer: 'Grid search exhaustively evaluates all combinations of predefined HP values; it is deterministic, trivially parallel, and suitable for coarse exploration of 1-3 HPs but suffers from the curse of dimensionality. Random search samples HPs uniformly from defined distributions; Bergstra & Bengio proved it is more efficient than grid search because each HP is explored at more distinct values for the same budget. Random search is the better default for most problems, especially when only a few HPs matter (low effective dimensionality). Grid search may be preferred when the search space is very small (≤3 HPs with few values each) or when deterministic behavior is required for reproducibility.', companyTags: ['Google', 'Meta'], difficulty: 'Medium' },
        { question: 'Explain how Bayesian optimization works for hyperparameter tuning, including the role of the Gaussian Process surrogate and acquisition function.', answer: 'Bayesian optimization builds a probabilistic surrogate model (typically a GP) of the objective f(hp). The GP provides both a predicted mean μ(hp) and uncertainty σ(hp) at any point. An acquisition function (EI, PI, or UCB) scores candidate points by balancing exploration (high σ) and exploitation (high μ). The next evaluation is at the point maximizing the acquisition function. After each evaluation, the GP is updated with the new observation. This sequential process naturally transitions from exploration to exploitation as the surrogate becomes more confident, making BO highly sample-efficient for expensive evaluation functions.', companyTags: ['Amazon', 'Microsoft'], difficulty: 'Hard' },
        { question: 'What is the exploration-exploitation tradeoff in hyperparameter search, and how do different search methods address it?', answer: 'Exploration means sampling regions with high uncertainty to understand the objective landscape; exploitation means sampling regions where performance is predicted to be high. Grid search and random search do not explicitly model this tradeoff — they simply cover the space. Bayesian optimization explicitly balances exploration vs exploitation through the acquisition function: EI naturally trades off both, UCB has an explicit parameter κ, and PI can be overly greedy. Random search implicitly explores well but does not exploit. The tradeoff is central to sequential decision-making: too much exploration wastes evaluations, while too much exploitation risks missing the global optimum.', companyTags: ['Apple', 'Uber'], difficulty: 'Medium' },
        { question: 'How would you design a rigorous hyperparameter tuning pipeline that avoids overfitting the validation set?', answer: '1) Split data into training + held-out test set (lock test set away). 2) Use k-fold cross-validation (k=5) on the training set to evaluate each HP configuration. 3) For rigorous performance estimation, use nested CV: an outer loop provides unbiased performance estimates, while an inner loop handles HP selection. 4) Limit the number of tuning trials to avoid multiple comparison inflation. 5) Use Bayesian optimization for efficient search when evaluations are expensive. 6) Report both the best CV score and a separate test set evaluation. 7) Always fit preprocessing (scaling, feature selection) inside the CV loop to prevent data leakage.', companyTags: ['Google', 'Netflix'], difficulty: 'Hard' },
        { question: 'What are the most important hyperparameters for tree-based ensemble models (Random Forest, XGBoost), and what search strategies work well for them?', answer: 'For Random Forest: n_estimators (50-1000, larger is better but diminishing returns), max_depth (3-20 or None), min_samples_split (2-50), max_features (sqrt, log2, or auto). For XGBoost: learning_rate (0.01-0.3, log scale), max_depth (3-10), subsample (0.5-1.0), colsample_bytree (0.5-1.0), n_estimators (50-2000). Strategy: use random search or Bayesian optimization — grid search is inefficient due to the number of HPs. Start with a broad random search (50-100 trials) to identify promising regions, then refine with Bayesian optimization. Use learning rate and n_estimators together (lower learning rate needs more estimators). Always use early stopping with XGBoost to dynamically determine n_estimators.', companyTags: ['Meta', 'Kaggle'], difficulty: 'Medium' },
    ]
};
