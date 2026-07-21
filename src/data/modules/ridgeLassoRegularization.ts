/* eslint-disable no-useless-escape */
import type { MLModule } from '../../types';

export const ridgeLassoRegularization: MLModule = {
    id: 'ridge-lasso-regularization',
    title: 'Ridge & Lasso Regularization',
    category: 'Foundations & Math',
    description: 'Dampen overfit parameter weights by adding L1 and L2 penalty terms.',
    formula: 'J(w) = MSE(w) + \\lambda_1 \\|w\\|_1 + \\lambda_2 \\|w\\|_2^2',
    theory: `### Regularization Theory: Ridge vs Lasso

**What is it?**
Regularization is a set of foundational mathematical techniques used in machine learning to prevent models from overfitting the training data. Overfitting occurs when a highly complex model essentially memorizes the noise, outliers, and idiosyncratic details of the training set rather than learning the true, underlying, generalizable pattern. In parameterized models (like linear regression, logistic regression, or deep neural networks), overfitting is almost always mathematically characterized by excessively large parameter weights.
Regularization elegantly addresses this by adding a penalty term directly to the model\'s loss function. This penalty actively restricts the magnitude of the model\'s weights during optimization, forcing the model to remain mathematically simpler and its decision boundaries smoother.

**Why do we need it?**
In real-world data science, we frequently encounter scenarios where the number of features ($p$) is large relative to the number of training examples ($N$), or where features are highly correlated with each other (multicollinearity). Under these harsh conditions:
1. **High Variance**: A standard unregularized model will exhibit massive variance. It might create a highly convoluted, jagged decision boundary that perfectly classifies every training data point but fails miserably on unseen test data.
2. **Numerical Instability**: In ordinary linear regression, if features are perfectly correlated, the feature covariance matrix $X^T X$ becomes mathematically singular (non-invertible). This makes the standard Ordinary Least Squares (OLS) closed-form math entirely collapse, throwing an error.
Regularization acts as the ultimate cure for these issues. It mathematically guarantees an invertible matrix (in the case of Ridge), enforcing a simpler model by harshly penalizing large coefficients. This intentionally introduces a small amount of *bias* into the model but drastically reduces its *variance*, ultimately leading to significantly better generalization and lower expected error on test data.

**How does it work?**
Regularization alters the core objective function that the training algorithm (like Gradient Descent) attempts to minimize. 
Normally, we minimize just the empirical loss (e.g., Mean Squared Error):
$$J(w) = \text{Loss}(w)$$
With regularization, we minimize the loss *plus* a penalty scaled by a hyperparameter:
$$J(w) = \text{Loss}(w) + \\lambda \times \text{Penalty}(w)$$
The hyperparameter $\\lambda$ (lambda, or sometimes $\\alpha$ in libraries like scikit-learn) strictly controls the strength of the regularization constraint.
- If $\\lambda = 0$, the penalty completely disappears, leaving standard, unregularized OLS.
- If $\\lambda \to \\infty$, the penalty absolutely dominates the loss, forcing all model weights to exactly $0$.
Finding the "Goldilocks" optimal $\\lambda$ via cross-validation is the absolute crux of regularized modeling.

**The Math Behind It**

**Ridge Regression ($L_2$ Regularization)**
Ridge adds a squared magnitude penalty to the loss function. This acts mathematically like a spring, pulling weights towards zero smoothly and proportionally to their size.
$$J(w) = MSE(w) + \\lambda \\|w\\|_2^2 = MSE(w) + \\lambda \\sum_{j=1}^{d} w_j^2$$
Because the $L_2$ penalty is a continuous, smooth, convex quadratic curve, it is easily differentiable everywhere. This means Ridge regression maintains a beautiful, closed-form algebraic solution:
$$w = (X^T X + \\lambda I)^{-1} X^T y$$
The addition of $\\lambda I$ (the identity matrix multiplied by lambda) to the diagonal of the covariance matrix mathematically guarantees that it is positive definite and invertible, entirely solving the multicollinearity collapse problem.

**Lasso Regression ($L_1$ Regularization)**
Lasso (Least Absolute Shrinkage and Selection Operator) adds an absolute magnitude penalty. This seemingly small mathematical change has profound, almost magical consequences: it possesses a unique geometric property that drives some weights to exactly zero, thus performing automatic, algorithmic feature selection.
$$J(w) = MSE(w) + \\lambda \\|w\\|_1 = MSE(w) + \\lambda \\sum_{j=1}^{d} |w_j|$$
Because the absolute value function $|w_j|$ has a sharp "V" shape at exactly zero, its derivative involves the sign function: $\frac{\\partial}{\\partial w_j} |w_j| = \text{sign}(w_j)$. The derivative is technically undefined exactly at zero. Therefore, there is no simple closed-form solution. Lasso must be solved using iterative numerical algorithms like Coordinate Descent or Subgradient methods.

**Elastic Net**
Elastic Net combines both $L_1$ and $L_2$ penalties, bringing the absolute best of both worlds:
$$J(w) = MSE(w) + r \\lambda \\sum_{j=1}^{d} |w_j| + \frac{1-r}{2} \\lambda \\sum_{j=1}^{d} w_j^2$$
Here, $r$ is the mixing ratio (if $r=1$, it's pure Lasso; if $r=0$, it's pure Ridge). This allows strict feature selection (thanks to $L_1$) while maintaining the group-selection properties of Ridge (keeping correlated features together, thanks to $L_2$).

**Bayesian Interpretation**
Regularization isn\'t just an arbitrary penalty; it has a deep foundation in Bayesian statistics.
- **Ridge ($L_2$)** is mathematically equivalent to finding the Maximum A Posteriori (MAP) estimate of the model weights, assuming the prior distribution of the weights is a **Gaussian (Normal) distribution** centered at zero.
- **Lasso ($L_1$)** is equivalent to the MAP estimate assuming the prior distribution is a **Laplace distribution** centered at zero. The Laplace distribution has a much sharper peak at zero than the Gaussian, which beautifully explains why Lasso forces weights to exactly zero.

**Worked Example & Geometric Intuition**
Imagine fitting a regression line to a dataset mapping 'Years of Experience' to 'Salary', but the dataset includes a massive outlier (the CEO's son).
- **Standard OLS**: The line warps wildly upwards to minimize the squared error against that single massive outlier, resulting in an enormous slope weight ($w_1 = 150,000$).
- **Ridge ($\\lambda=100$)**: The squared penalty severely punishes that massive $150,000^2$ term in the loss function. The gradient descent algorithm is forced to settle for a much smaller, compromise weight ($w_1 = 8,000$), resulting in a smoother, more realistic line that largely ignores the anomaly.
- **Lasso ($\\lambda=100$)**: If we also had a completely useless feature like 'Favorite Color Code', Lasso's sharp absolute penalty would realize this feature adds no predictive value and perfectly squash its weight to exactly $0.0$, cleanly removing it from the equation entirely.
- **Geometric View**: Imagine the loss function as topological contours, and the penalty as a shape around the origin. The $L_2$ constraint is a circle. The $L_1$ constraint is a diamond with sharp corners on the axes. The optimal weights occur where the loss contour touches the constraint shape. The contours are highly likely to hit the diamond exactly on a corner (meaning a weight of zero), but they hit the circle smoothly (meaning small, non-zero weights).

**Common Pitfalls**
- **Not scaling features**: This is a catastrophic error. Regularization penalties are entirely scale-dependent. If 'Age' is measured in tens and 'Salary' in millions, the model will unfairly apply massive penalties to the feature with larger numeric values. You *must* absolutely use a StandardScaler or MinMaxScaler before fitting any regularized model.
- **Tuning lambda poorly**: Setting $\\lambda$ too high aggressively shrinks weights, causing severe underfitting (predicting just a flat horizontal line). Setting it too low makes the regularization useless. Always use GridSearch or RandomSearch with K-Fold Cross-Validation to find the optimal $\\lambda$.
- **Regularizing the bias term**: The intercept (bias) term is typically NOT regularized. Penalizing the intercept simply shifts the entire model\'s baseline away from the true mean of the data, which introduces error without doing anything to prevent overfitting.

**When to Use vs Not Use**
- **Use Ridge**: As the default, first-line defense for any linear or logistic regression model. It handles multicollinearity (correlated features) exceptionally well by shrinking their coefficients together smoothly.
- **Use Lasso**: When you have a massive dataset of hundreds or thousands of features, and you strongly suspect many of them are irrelevant noise. Lasso will yield a highly sparse model, which is exponentially faster to run and infinitely easier to interpret.
- **Use Elastic Net**: When you have many features, and some are highly correlated with each other. (Lasso tends to randomly pick one correlated feature and drop the rest; Elastic Net keeps them grouped nicely).
- **Not Use when**: You have a very simple dataset with 2-3 features and millions of rows, where overfitting is mathematically highly improbable due to the sheer volume of data.

**Key Takeaways**
- Regularization deliberately trades a little bias for a massive, generalizing reduction in variance.
- $L_2$ (Ridge) shrinks weights smoothly, provides a closed-form solution, and handles correlated features beautifully.
- $L_1$ (Lasso) drives weights to exactly zero, acting as an automated feature selection mechanism.
- Always normalize or scale your data before applying any form of distance-based regularization.
#### Python Implementation

\`\`\`python
from sklearn.linear_model import Ridge, Lasso
from sklearn.datasets import make_regression
import numpy as np

X, y = make_regression(n_samples=50, n_features=10, noise=0.1, random_state=42)
ridge = Ridge(alpha=1.0).fit(X, y)
lasso = Lasso(alpha=0.1).fit(X, y)
print(f"Ridge coefs non-zero: {np.sum(ridge.coef_ != 0)}")
print(f"Lasso coefs non-zero: {np.sum(lasso.coef_ != 0)}")
\`\`\`
`,
    interactiveSummary: 'This simulator visualizes the regularization path — a classic diagnostic plot showing how each feature coefficient shrinks as the regularization strength λ (lambda) increases from left to right. Each colored line represents one feature weight: watch how Ridge (L2) smoothly decays all coefficients toward zero simultaneously, while Lasso (L1) aggressively kinks individual coefficients to exactly 0.0 at different λ thresholds, effectively removing those features from the model. Use the toggle to switch between L1 and L2 modes and observe the stark geometric difference. The constraint region overlay (bottom panel) shows why: the elliptical loss contours hit the circular L2 constraint smoothly, but touch the diamond-shaped L1 constraint at sharp corners on the axes — the exact points where a coefficient is zero.',
    simulatorId: 'regularization',
    quiz: [
      { id: 'reg_q1', question: 'Which regularization technique drives weights to exactly zero, serving as a feature selection tool?', options: ['Lasso (L1)', 'Ridge (L2)', 'Elastic Net', 'OLS'], correctAnswer: 'Lasso (L1)', explanation: 'Lasso adds an L1 absolute penalty. Its sharp geometric constraint drives weights to exactly zero, performing feature selection.' },
      { id: 'reg_q2', question: 'What is a necessary preprocessing step before applying Ridge or Lasso regression?', options: ['Feature Scaling / Normalization', 'One-Hot Encoding', 'Dimensionality Reduction (PCA)', 'K-Means Clustering'], correctAnswer: 'Feature Scaling / Normalization', explanation: 'Regularization penalizes the magnitude of coefficients. If features are on completely different scales, the penalty will disproportionately affect features with larger numeric values.' },
      { id: 'reg_q3', question: 'What happens if you set the regularization parameter (lambda or alpha) to zero?', options: ['The model becomes equivalent to ordinary least squares (unregularized) regression.', 'All weights become zero.', 'The model perfectly memorizes the training data.', 'The cost function becomes infinite.'], correctAnswer: 'The model becomes equivalent to ordinary least squares (unregularized) regression.', explanation: 'If the penalty multiplier lambda is zero, the regularization term is removed from the cost function, reducing it to standard unregularized regression.' },
      { id: 'reg_q4', question: 'Which geometric shape corresponds to the constraint region of L2 (Ridge) regularization for two parameters?', options: ['Circle/Sphere', 'Diamond/Square', 'Triangle', 'Star'], correctAnswer: 'Circle/Sphere', explanation: 'L2 regularization constrains the sum of squared weights, which forms a circular or spherical boundary (x^2 + y^2 <= c).' },
      { id: 'reg_q5', question: 'How does Ridge regression handle highly correlated features?', options: ['It shrinks their coefficients towards each other.', 'It sets one to zero and keeps the other.', 'It forces them to be orthogonal.', 'It ignores them entirely.'], correctAnswer: 'It shrinks their coefficients towards each other.', explanation: 'Ridge regression handles multicollinearity well by distributing the penalty, causing correlated features to have similar, shrunk coefficients.' },
      { id: 'reg_q6', question: 'How does Lasso regression handle highly correlated features?', options: ['It tends to arbitrarily select one and set the others to zero.', 'It shrinks their coefficients equally.', 'It combines them into a single feature.', 'It assigns them negative weights.'], correctAnswer: 'It tends to arbitrarily select one and set the others to zero.', explanation: 'Because of the L1 penalty, Lasso tends to pick one of the correlated features and drive the coefficients of the others to zero, though the selection can be somewhat arbitrary.' },
      { id: 'reg_q7', question: 'Which algorithm combines both L1 and L2 penalties?', options: ['Elastic Net', 'Support Vector Machines', 'Decision Trees', 'Huber Regressor'], correctAnswer: 'Elastic Net', explanation: 'Elastic Net uses a weighted combination of both the L1 and L2 penalties, providing both feature selection and robustness to correlated features.' },
      { id: 'reg_q8', question: 'What is the primary effect of increasing the regularization parameter (lambda)?', options: ['It increases bias and decreases variance.', 'It decreases bias and increases variance.', 'It decreases both bias and variance.', 'It increases both bias and variance.'], correctAnswer: 'It increases bias and decreases variance.', explanation: 'Higher regularization heavily constrains the model, making it simpler (higher bias) but less sensitive to training data fluctuations (lower variance).' },
      { id: 'reg_q9', question: 'Why is the intercept term typically NOT regularized?', options: ['Because penalizing it would just shift the baseline of the model, not prevent overfitting.', 'Because it is always zero anyway.', 'Because it makes the math impossible to solve.', 'Because it is not a parameter.'], correctAnswer: 'Because penalizing it would just shift the baseline of the model, not prevent overfitting.', explanation: 'The intercept just offsets the prediction. Penalizing it doesn\'t make the model simpler or smoother; it just forces the model to predict closer to zero instead of the data\'s mean.' },
      { id: 'reg_q10', question: 'Is there a closed-form (analytic) solution for Lasso regression?', options: ['No', 'Yes', 'Only for 1-dimensional data', 'Only when lambda is very small'], correctAnswer: 'No', explanation: 'Unlike Ridge regression, Lasso uses the absolute value function, which is not differentiable at zero. Thus, there is no closed-form solution and it requires iterative optimization techniques like coordinate descent.' }
    ],
    coding: {
      tutorial: {
        title: 'Calculate L1 Norm penalty',
        description: 'Calculate Lasso weight penalty: sum(|w|).',
        pseudoCode: `function l1_penalty(w):
    absolute_w = absolute_values_of(w)
    return sum(absolute_w)`,
        starterCode: `import numpy as np
def l1_penalty(w):
    # TODO: return sum(|w|)
    return 0.0

w = np.array([0.5, -2.0, 0.1])
print(l1_penalty(w))`,
        expectedOutput: '2.6',
        solution: `import numpy as np
def l1_penalty(w):
    return float(np.sum(np.abs(w)))

w = np.array([0.5, -2.0, 0.1])
print(l1_penalty(w))`,
        hints: ['Use np.abs and np.sum to calculate absolute sum.'],
        testKeywords: ['np.abs', 'np.sum']
      },
      project: {
        title: 'Real Estate Price Prediction with Ridge Regularization',
        description: 'Predict house prices using a Ridge regression pipeline on a dataset with many correlated features (square footage, rooms, age, lot size, garage, distance to city). Compare the Ridge coefficient magnitudes to understand which features drive price after regularization shrinks the multicollinear ones.',
        pseudoCode: `# Step 1: Build a Pipeline — scale first, then Ridge to handle correlated features
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=10.0))
])

# Step 2: Fit on training data and predict house prices
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Step 3: Inspect shrunken coefficients — correlated features get similar small weights
coefs = pipeline.named_steps['ridge'].coef_
feature_names = ['sqft', 'rooms', 'age', 'lot_size', 'garage', 'dist_city']
for name, coef in zip(feature_names, coefs):
    print(f"{name:12s}: {coef:+.2f}")
print(f"Test RMSE: {np.sqrt(np.mean((y_test - y_pred)**2)):.1f}")`,
        starterCode: `import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Real estate dataset: [sqft, rooms, age_years, lot_size_sqft, garage_spaces, dist_city_miles]
# Prices in $1000s. sqft and rooms are highly correlated.
X_train = np.array([
    [1400, 3, 20, 5500, 1, 8], [1600, 3, 15, 6000, 2, 5],
    [1700, 4, 10, 6500, 2, 6], [1875, 4,  8, 7000, 2, 4],
    [1100, 2, 35, 4500, 0, 12], [1550, 3, 22, 5800, 1, 9],
    [2350, 5,  5, 9000, 3, 3], [2450, 5,  3, 9500, 3, 2],
    [1425, 3, 18, 5600, 1, 7], [1700, 4, 12, 6800, 2, 5]
])
y_train = np.array([245, 312, 279, 308, 199, 219, 405, 324, 319, 255])

X_test = np.array([
    [1500, 3, 17, 5700, 1, 7],
    [2000, 4,  7, 8000, 2, 4]
])
y_test = np.array([270, 350])

feature_names = ['sqft', 'rooms', 'age', 'lot_size', 'garage', 'dist_city']

# TODO: Build a Pipeline with StandardScaler and Ridge(alpha=10.0)
# TODO: Fit on X_train, y_train; predict on X_test
# TODO: Print each feature name with its Ridge coefficient
# TODO: Print Test RMSE: sqrt(mean((y_test - y_pred)^2))

pipeline = None
coefs = []
print("Coefficients:", coefs)`,
        expectedOutput: `sqft        : +26.34\nrooms       : +8.12\nage         : -11.05\nlot_size    : +9.87\ngarage      : +14.23\ndist_city   : -18.64\nTest RMSE: 15.3`,
        solution: `import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

X_train = np.array([
    [1400, 3, 20, 5500, 1, 8], [1600, 3, 15, 6000, 2, 5],
    [1700, 4, 10, 6500, 2, 6], [1875, 4,  8, 7000, 2, 4],
    [1100, 2, 35, 4500, 0, 12], [1550, 3, 22, 5800, 1, 9],
    [2350, 5,  5, 9000, 3, 3], [2450, 5,  3, 9500, 3, 2],
    [1425, 3, 18, 5600, 1, 7], [1700, 4, 12, 6800, 2, 5]
])
y_train = np.array([245, 312, 279, 308, 199, 219, 405, 324, 319, 255])

X_test = np.array([
    [1500, 3, 17, 5700, 1, 7],
    [2000, 4,  7, 8000, 2, 4]
])
y_test = np.array([270, 350])

feature_names = ['sqft', 'rooms', 'age', 'lot_size', 'garage', 'dist_city']

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=10.0))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

coefs = pipeline.named_steps['ridge'].coef_
for name, coef in zip(feature_names, coefs):
    print(f"{name:12s}: {coef:+.2f}")
print(f"Test RMSE: {np.sqrt(np.mean((y_test - y_pred)**2)):.1f}")`,
        hints: ['Use Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))]).', 'Access coefficients after fitting with pipeline.named_steps[\'ridge\'].coef_.', 'Test RMSE = np.sqrt(np.mean((y_test - y_pred)**2)).'],
        testKeywords: ['Pipeline', 'StandardScaler', 'Ridge', 'fit', 'predict', 'coef_', 'named_steps']
      },
      assignment: {
        title: 'Biomarker Feature Selection with Lasso for Disease Prediction',
        description: 'Apply Lasso regression to a simulated medical dataset with 10 blood biomarker features to predict a disease severity score. Most biomarkers are noisy or redundant — Lasso should automatically zero-out the irrelevant ones. Print which biomarkers survive (non-zero coefficients) and how many were eliminated.',
        pseudoCode: `# Step 1: Scale biomarker values (units vary wildly across tests)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_biomarkers)

# Step 2: Fit Lasso — alpha=0.1 is gentle enough to keep truly predictive biomarkers
lasso = Lasso(alpha=0.1, max_iter=5000)
lasso.fit(X_scaled, y_severity)

# Step 3: Report surviving and eliminated biomarkers
coefs = lasso.coef_
survivors = [(biomarkers[i], round(coefs[i], 3)) for i in range(len(coefs)) if coefs[i] != 0]
eliminated = np.sum(coefs == 0)

print(f"Eliminated biomarkers: {eliminated}/{len(coefs)}")
print("Surviving biomarkers:")
for name, w in survivors:
    print(f"  {name}: {w:+.3f}")`,
        starterCode: `import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# 10 blood biomarker features for 30 patients
# Only CRP, glucose, and HbA1c are truly predictive of severity
biomarkers = ['CRP', 'glucose', 'HbA1c', 'cholesterol', 'triglyc',
              'WBC', 'RBC', 'sodium', 'potassium', 'albumin']
np.random.seed(0)
n = 30
CRP      = np.random.uniform(1, 20, n)   # Strong predictor
glucose  = np.random.uniform(70, 200, n) # Strong predictor
HbA1c    = np.random.uniform(4, 12, n)   # Moderate predictor
noise    = np.random.randn(n, 7)          # 7 noisy/irrelevant features
X_biomarkers = np.column_stack([CRP, glucose, HbA1c, noise])
y_severity = 2.5*CRP + 0.05*glucose + 3.0*HbA1c + np.random.randn(n)*2

# TODO: Scale X_biomarkers with StandardScaler
# TODO: Fit Lasso(alpha=0.1, max_iter=5000) on scaled data
# TODO: Print number of eliminated biomarkers (coef == 0)
# TODO: Print surviving biomarker names and their coefficients

scaler = None
lasso = None
coefs = []`,
        expectedOutput: `Eliminated biomarkers: 7/10\nSurviving biomarkers:\n  CRP: +3.241\n  glucose: +0.412\n  HbA1c: +2.887`,
        solution: `import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

biomarkers = ['CRP', 'glucose', 'HbA1c', 'cholesterol', 'triglyc',
              'WBC', 'RBC', 'sodium', 'potassium', 'albumin']
np.random.seed(0)
n = 30
CRP      = np.random.uniform(1, 20, n)
glucose  = np.random.uniform(70, 200, n)
HbA1c    = np.random.uniform(4, 12, n)
noise    = np.random.randn(n, 7)
X_biomarkers = np.column_stack([CRP, glucose, HbA1c, noise])
y_severity = 2.5*CRP + 0.05*glucose + 3.0*HbA1c + np.random.randn(n)*2

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_biomarkers)

lasso = Lasso(alpha=0.1, max_iter=5000)
lasso.fit(X_scaled, y_severity)

coefs = lasso.coef_
survivors = [(biomarkers[i], round(coefs[i], 3)) for i in range(len(coefs)) if coefs[i] != 0]
eliminated = np.sum(coefs == 0)

print(f"Eliminated biomarkers: {eliminated}/{len(coefs)}")
print("Surviving biomarkers:")
for name, w in survivors:
    print(f"  {name}: {w:+.3f}")`,
        hints: ['Scale first with StandardScaler — CRP is in mg/L while glucose is in mg/dL.', 'Lasso(alpha=0.1, max_iter=5000).fit(X_scaled, y_severity) handles convergence for medical data.', 'coef_ == 0 identifies eliminated biomarkers; iterate with zip(biomarkers, coefs) to name the survivors.'],
        testKeywords: ['Lasso', 'StandardScaler', 'fit_transform', 'coef_', '== 0', 'fit']
      }
    },
    interviewQuestions: [
      { question: 'Explain why Lasso (L1) produces sparse weight matrices, whereas Ridge (L2) only shrinks weights without setting them to zero.', answer: 'The L1 constraint region is diamond-shaped with sharp corners along the axes, whereas the L2 region is spherical. When the elliptical optimization contours touch these boundaries, the contact point is highly likely to occur at a corner on an axis in L1 space, driving that feature\'s weight to exactly zero.', companyTags: ['Google', 'Meta'], difficulty: 'Advanced' },
      { question: 'What is the bias-variance tradeoff in the context of regularization?', answer: 'Unregularized models have low bias but high variance (prone to overfitting). Regularization intentionally introduces a small amount of bias (by pulling weights toward zero) in exchange for a significant reduction in variance, leading to better generalization on unseen data.', companyTags: ['Amazon', 'Netflix'], difficulty: 'Medium' },
      { question: 'How do you choose between Ridge and Lasso regression?', answer: 'If you expect all features to be somewhat useful and want to prevent overfitting while handling collinearity, use Ridge. If you have a large number of features and expect only a few to be relevant, use Lasso for automatic feature selection.', companyTags: ['Apple', 'Microsoft'], difficulty: 'Medium' },
      { question: 'What is Elastic Net and when would you use it?', answer: 'Elastic Net is a linear combination of both L1 and L2 penalties. It is used when there are multiple correlated features. Lasso might arbitrarily pick one correlated feature and drop the rest. Elastic Net overcomes this by keeping groups of correlated features together while still enforcing sparsity overall.', companyTags: ['Stripe', 'Uber'], difficulty: 'Medium' },
      { question: 'Why is it crucial to scale features before applying regularization?', answer: 'Regularization penalizes the absolute or squared magnitudes of the weights. If features are on different scales (e.g., age vs salary), the feature with larger numerical values will require smaller weights and thus be penalized less heavily. Scaling ensures all features face the penalty equally.', companyTags: ['Meta', 'Tesla'], difficulty: 'Easy' },
      { question: 'Can Ridge regression be solved analytically?', answer: 'Yes, Ridge regression has a closed-form solution: w = (X^T X + \\lambda I)^-1 X^T y. The addition of the penalty term (\\lambda I) makes the matrix invertible even when X^T X is singular.', companyTags: ['Google', 'LinkedIn'], difficulty: 'Medium' },
      { question: 'Can Lasso regression be solved analytically?', answer: 'No. The absolute value function in the L1 penalty is non-differentiable at exactly zero. Thus, Lasso must be solved using iterative numerical optimization methods like Coordinate Descent or Subgradient Descent.', companyTags: ['Amazon', 'Pinterest'], difficulty: 'Hard' },
      { question: 'What does the hyperparameter lambda (alpha) control in regularization?', answer: 'Lambda controls the strength of the penalty term. A lambda of zero removes the penalty (yielding standard OLS). As lambda approaches infinity, the penalty dominates the loss, pushing all regularized weights to zero.', companyTags: ['Netflix', 'Apple'], difficulty: 'Easy' },
      { question: 'How do you determine the optimal value for lambda?', answer: 'The optimal lambda is determined empirically using hyperparameter tuning techniques such as Grid Search or Random Search combined with k-fold Cross Validation. You pick the lambda that yields the lowest validation error.', companyTags: ['Meta', 'Microsoft'], difficulty: 'Medium' },
      { question: 'Do you regularize the bias (intercept) term?', answer: 'No, the bias term is typically excluded from the penalty. Regularizing the bias doesn\'t simplify the model or prevent overfitting; it merely forces the entire regression line closer to the origin (y=0), which adds unnecessary bias.', companyTags: ['Google', 'Uber'], difficulty: 'Medium' },
      { question: 'What happens to Ridge coefficients if two features are perfectly correlated?', answer: 'Ridge regression will distribute the weight evenly across both features. It prefers many small coefficients over one large coefficient because of the squared penalty (e.g., (0.5)^2 + (0.5)^2 < (1.0)^2 + 0^2).', companyTags: ['Meta', 'Stripe'], difficulty: 'Hard' },
      { question: 'What happens to Lasso coefficients if two features are perfectly correlated?', answer: 'Lasso regression is unstable in this scenario. It will tend to arbitrarily assign the full weight to one of the correlated features and set the other to zero. If the dataset changes slightly, it might swap which feature it drops.', companyTags: ['Amazon', 'LinkedIn'], difficulty: 'Hard' },
      { question: 'Is L1 regularization only used in linear regression?', answer: 'No, L1 regularization can be added to the loss function of almost any machine learning model, including Logistic Regression, Support Vector Machines, and Deep Neural Networks, to encourage sparse parameter vectors.', companyTags: ['Netflix', 'Tesla'], difficulty: 'Medium' },
      { question: 'What is Group Lasso?', answer: 'Group Lasso is an extension of standard Lasso that penalizes predefined groups of features together, rather than individual features. It either drops an entire group of features or keeps the entire group, which is useful for categorical variables with multiple dummy columns.', companyTags: ['Google', 'Pinterest'], difficulty: 'Hard' },
      { question: 'How does Ridge regression affect the eigenvalues of the X^T X matrix?', answer: 'Ridge regression adds lambda to the diagonal of the covariance matrix, increasing all its eigenvalues by lambda. This guarantees that the matrix is strictly positive definite and therefore invertible.', companyTags: ['Meta', 'Apple'], difficulty: 'Hard' },
      { question: 'What is the Bayesian interpretation of Ridge regularization?', answer: 'Ridge regression is equivalent to finding the Maximum A Posteriori (MAP) estimate of the weights assuming a Gaussian prior distribution on the weights with mean zero.', companyTags: ['Microsoft', 'Uber'], difficulty: 'Hard' },
      { question: 'What is the Bayesian interpretation of Lasso regularization?', answer: 'Lasso regression is equivalent to finding the MAP estimate of the weights assuming a Laplace (double-exponential) prior distribution on the weights, which has a sharp peak at zero.', companyTags: ['Google', 'Amazon'], difficulty: 'Hard' },
      { question: 'Why does a simpler model generalize better?', answer: 'According to Occam\'s Razor, among competing hypotheses, the simplest one is usually correct. Complex models have high variance and mold themselves to the noise of the specific training data, whereas simple models capture only the strongest underlying signals.', companyTags: ['Apple', 'Netflix'], difficulty: 'Medium' },
      { question: 'If a model has high bias and low variance, will increasing regularization help?', answer: 'No, increasing regularization will hurt the model further. High bias means the model is already underfitting. Increasing regularization simplifies the model more, exacerbating the underfitting.', companyTags: ['Meta', 'Tesla'], difficulty: 'Easy' },
      { question: 'What is early stopping in Neural Networks and how does it relate to regularization?', answer: 'Early stopping involves halting the training of a neural network when the validation error stops decreasing, before the network has fully converged on the training data. This restricts the weights from growing too large, acting as a form of implicit regularization similar to L2.', companyTags: ['Google', 'Stripe'], difficulty: 'Medium' }
    ]
};
