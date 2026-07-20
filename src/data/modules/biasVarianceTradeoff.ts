/* eslint-disable no-useless-escape */
import type { MLModule } from '../../types';

export const biasVarianceTradeoff: MLModule = {
    id: 'bias-variance-tradeoff',
    title: 'Bias-Variance Tradeoff',
    category: 'Foundations & Math',
    description: 'Balance model capacity to avoid underfitting and overfitting.',
    formula: '\\text{Error} = \\text{Bias}^2 + \\text{Variance} + \\text{IrreducibleError}',
    theory: `### Bias-Variance Tradeoff Theory

**What is it?**
The Bias-Variance Tradeoff is the single most fundamental property of machine learning models. It describes the deep, unavoidable tension between a model\'s complexity, the accuracy of its predictions on training data, and its ability to generalize to new, unseen testing data. It formally and mathematically breaks down the expected generalization error of any learning algorithm into three distinct, additive components: Bias, Variance, and Irreducible Noise.

Every model in machine learning is essentially an approximation of reality. As we tweak the model\'s capacity—how intricately and flexibly it can mold itself to the data—we inevitably trade off between making rigid, inaccurate assumptions (Bias) and being overly sensitive to random fluctuations in the training set (Variance).

**Why do we need it?**
Understanding this tradeoff is the skeleton key to solving the core problem of supervised learning: generalizing well to new data. If a data scientist does not understand bias and variance, they will inevitably build models that either:
1. Are far too simple and systematically fail to capture the underlying trend of the data (**Underfitting**).
2. Are far too complex and memorize the noise, anomalies, and outliers in the training data, failing completely on test data (**Overfitting**).
The tradeoff gives us a strict theoretical framework and vocabulary to diagnose poor model performance. More importantly, it provides actionable, concrete steps to fix it (e.g., getting more data, adding regularization, or changing the algorithm entirely).

**How does it work?**
Imagine you are throwing darts at a dartboard. The bullseye represents the true, underlying function you want to predict.
- **High Bias (Underfitting)**: Your darts are clustered tightly together, but they are far away from the bullseye. The model has strong, incorrect assumptions about the data. (e.g., trying to fit a perfectly straight line to a sine wave).
- **High Variance (Overfitting)**: Your darts are scattered wildly all over the board. The model is highly sensitive to the exact training points, including outliers and noise. It has memorized the training data rather than learning the pattern.
- **Low Bias, Low Variance (The Sweet Spot)**: Your darts are clustered tightly around the bullseye. The model captures the true pattern and gracefully ignores the noise.

As you sequentially increase model complexity (e.g., moving from a linear regression to a 15th-degree polynomial, or increasing the maximum depth of a decision tree from 2 to 20), bias decreases because the model gains the mathematical flexibility to represent more complex, nuanced patterns. However, variance increases simultaneously because the model gains enough flexibility to start memorizing the noise and randomness inherent in the training sample.

**The Math Behind It**
The expected squared error (Mean Squared Error) of a model \\(\\hat{f}(x)\\) attempting to predict a true underlying function \\(f(x)\\) with some inherent noise \\(\\epsilon\\) can be mathematically decomposed into three distinct parts.

Given the true relationship \\(y = f(x) + \\epsilon\\), where the noise \\(\\epsilon \\sim \\mathcal{N}(0, \\sigma^2)\\),
The expected error at an unseen test point \\(x\\) is:
$$E[(y - \\hat{f}(x))^2] = (Bias[\\hat{f}(x)])^2 + Variance[\\hat{f}(x)] + \\sigma^2$$

Let's break down each term:
- **Bias**: The difference between the average, expected prediction of our model (across many training sets) and the correct value.
  $$Bias[\\hat{f}(x)] = E[\\hat{f}(x)] - f(x)$$
- **Variance**: The variability of the model prediction for a given data point. It measures how much the predictions would jump around if we trained the model on a slightly different dataset.
  $$Variance[\\hat{f}(x)] = E[\\hat{f}(x)^2] - (E[\\hat{f}(x)])^2$$
- **Irreducible Error (\\(\\sigma^2\\))**: The noise inherent in the data itself. No model, no matter how powerful or complex, can ever reduce the error below this theoretical limit. It represents the limits of our feature set.

**Worked Example & Visualizing Complexity**
Imagine trying to predict house prices based purely on square footage. The true underlying relationship is slightly curved (prices plateau for enormous houses).
1. **Model A (Linear Regression)**: Fits a rigid straight line. It completely fails to capture the plateau. This model has **High Bias** and **Low Variance**. If we sample a different training set, the straight line won't change its slope very much.
2. **Model B (15th Degree Polynomial)**: Fits a wild, chaotic, squiggly curve that touches every single data point perfectly, including strange outliers (like a cheap huge house in a bad neighborhood). This model has **Low Bias** (average error on training data is literally 0) but astronomically **High Variance**. On a new dataset, the curve will swing wildly between points, making catastrophic predictions.
3. **Model C (Quadratic/Polynomial Degree 2)**: Fits a gentle, smooth curve. It captures the plateau perfectly without going crazy over outliers. It finds the absolute optimal balance, minimizing the total expected error.

**Common Pitfalls**
- **Assuming More Data Always Helps**: This is a classic beginner mistake. If your model suffers from High Bias (e.g., you are running a linear regression on highly non-linear data), adding a billion more data points will *not* improve performance. The model simply lacks the capacity to bend. You need a more complex model or better engineered features.
- **Overcomplicating the Model**: Jumping straight to Deep Neural Networks, XGBoost, or massive ensembles for simple tabular data often immediately leads to massive High Variance.
- **Relying Only on Training Error**: Training error almost always decreases monotonically as complexity increases. If you evaluate based on training error, you will always pick the most overfitted model. You must strictly use a holdout validation set or K-Fold cross-validation to plot the true test error and find the U-shaped curve where variance begins to dominate.

**When to Use vs Not Use (Diagnostics & Solutions)**
- **Diagnose High Bias when**: Training error is high, and validation error is also high (and they are close to each other).
  - *Fixes*: Add more relevant features, engineer non-linear features, increase model complexity (e.g., deeper trees, higher degree polynomials, more neurons), or decrease regularization (lower lambda/alpha).
- **Diagnose High Variance when**: Training error is extremely low (close to zero), but validation error is stubbornly high.
  - *Fixes*: Get more training data (this is the most effective fix for variance), reduce the number of features (feature selection), simplify the model algorithm, add strict regularization (L1/L2), or use ensembling techniques like bagging (e.g., Random Forests).

**Key Takeaways**
- Total Error is mathematically defined as Bias² + Variance + Irreducible Error.
- Model complexity strictly decreases bias but strictly increases variance.
- The overarching goal of machine learning is to find the "sweet spot" at the bottom of the U-shaped error curve that minimizes the sum of bias and variance on unseen test data.
- Cross-validation, learning curves, and validation curves are the mandatory primary tools used in practice to visualize and aggressively manage this tradeoff.`,
    interactiveSummary: "This simulator visualizes the bias-variance decomposition across a sweep of model complexities using a live polynomial regression fitted to a noisy, real-world-style dataset. The top chart renders the fitted polynomial curve against the noisy data points, making it immediately visible when the model underfits (flat line misses the curve) or overfits (wild oscillations between points). The bottom chart plots the U-shaped error curve: as polynomial degree increases, training MSE monotonically decreases while validation MSE first falls then sharply rises — the inflection point is the bias-variance sweet spot. Hovering over any degree value decomposes the total error into its bias² and variance components, connecting the visual pattern to the mathematical decomposition Error = Bias² + Variance + Irreducible Noise. Use the noise level slider to observe how higher irreducible noise raises the floor of the entire error curve without shifting the optimal model complexity.",
    simulatorId: 'bias-variance',
    quiz: [
      { id: 'bv_q1', question: 'Which term describes a model that is too simple and fails to capture the underlying structure of the data?', options: ['Overfitting', 'High Variance', 'Underfitting', 'Irreducible Error'], correctAnswer: 'Underfitting', explanation: 'Underfitting occurs when a model lacks the capacity to capture the underlying pattern, leading to high bias.' },
      { id: 'bv_q2', question: 'What is the relationship between model complexity and variance?', options: ['As complexity increases, variance decreases.', 'As complexity increases, variance increases.', 'Variance remains constant regardless of complexity.', 'Complexity and variance are inversely proportional.'], correctAnswer: 'As complexity increases, variance increases.', explanation: 'More complex models have more parameters and flexibility, making them more sensitive to fluctuations in the specific training dataset.' },
      { id: 'bv_q3', question: 'What does irreducible error represent in the bias-variance decomposition?', options: ['Error from using the wrong algorithm', 'Error due to lack of training data', 'Inherent noise and randomness in the problem itself', 'Error caused by floating-point inaccuracies'], correctAnswer: 'Inherent noise and randomness in the problem itself', explanation: 'Irreducible error is the variance of the true target variable around its true mean function; it cannot be eliminated by any model.' },
      { id: 'bv_q4', question: 'If a model has very low training error but very high validation error, it is likely suffering from:', options: ['High Bias', 'High Variance', 'Low Variance', 'Underfitting'], correctAnswer: 'High Variance', explanation: 'A large gap between training and validation error indicates that the model has memorized the training data (overfitting) but fails to generalize.' },
      { id: 'bv_q5', question: 'Which of the following is a common remedy for a model suffering from HIGH BIAS?', options: ['Adding more training data', 'Increasing regularization', 'Adding more features or increasing model complexity', 'Removing features'], correctAnswer: 'Adding more features or increasing model complexity', explanation: 'High bias means the model is too restricted. You must give it more capacity by adding features or using a more complex algorithm.' },
      { id: 'bv_q6', question: 'Which of the following is a common remedy for a model suffering from HIGH VARIANCE?', options: ['Decreasing regularization', 'Adding more training data', 'Using a more complex model', 'Training for more epochs'], correctAnswer: 'Adding more training data', explanation: 'More data helps "smooth out" the noise, making it harder for a highly flexible model to overfit.' },
      { id: 'bv_q7', question: 'In the bias-variance tradeoff, what does "Bias" mathematically represent?', options: ['The difference between the average model prediction and the true value', 'The spread of predictions around their mean', 'The error from corrupted data', 'The personal prejudice of the data scientist'], correctAnswer: 'The difference between the average model prediction and the true value', explanation: 'Bias measures how far off the expected (average) prediction of the model is from the true underlying function.' },
      { id: 'bv_q8', question: 'How does K-Fold Cross-Validation help manage the bias-variance tradeoff?', options: ['It completely eliminates irreducible error.', 'It automatically chooses the most complex model.', 'It provides a reliable estimate of the model\'s out-of-sample error to detect overfitting.', 'It forces the variance to be zero.'], correctAnswer: 'It provides a reliable estimate of the model\'s out-of-sample error to detect overfitting.', explanation: 'By simulating out-of-sample performance, CV lets you plot the validation curve and find the minimum total error.' },
      { id: 'bv_q9', question: 'A Random Forest algorithm generally improves a single Decision Tree by reducing which component of error?', options: ['Bias', 'Variance', 'Irreducible Error', 'Both Bias and Irreducible Error'], correctAnswer: 'Variance', explanation: 'Random Forests use bagging (averaging many deep trees). Each tree has high variance and low bias. Averaging them reduces the variance significantly without increasing bias.' },
      { id: 'bv_q10', question: 'If you plot a learning curve and both the training and validation errors are high and plateaued close to each other, what is the diagnosis?', options: ['High Variance', 'High Bias', 'Optimal Model', 'Data Leakage'], correctAnswer: 'High Bias', explanation: 'When training and validation errors are both high and close together, adding more data (plateau) doesn\'t help. The model is fundamentally too simple.' }
    ],
    coding: {
      tutorial: {
        title: 'Precision and Recall Calculator',
        description: "Calculate Precision and Recall given True Positives, False Positives, and False Negatives. This evaluates model performance, touching on different types of errors.",
        pseudoCode: `function calc_metrics(tp, fp, fn):
    if tp + fp > 0 then precision = tp / (tp + fp) else precision = 0
    if tp + fn > 0 then recall = tp / (tp + fn) else recall = 0
    return precision, recall`,
        starterCode: `def calc_metrics(tp, fp, fn):
    # TODO: return precision, recall
    return 0.0, 0.0

print(calc_metrics(10, 2, 3))`,
        expectedOutput: '(0.8333333333333334, 0.7692307692307693)',
        solution: `def calc_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall

print(calc_metrics(10, 2, 3))`,
        hints: ['Implement the precision and recall formulas: P = TP/(TP+FP), R = TP/(TP+FN).'],
        testKeywords: ['tp + fp', 'tp + fn']
      },
      project: {
        title: 'Polynomial Regression Overfitting on Noisy Real-World Data',
        description: 'Demonstrate the bias-variance tradeoff using polynomial regression on a noisy dataset mimicking real-world house price data. Compare training MSE vs. validation MSE for polynomial degrees 1 (underfit) and 10 (overfit), confirming that the complex model memorizes noise.',
        pseudoCode: `# Polynomial Overfitting Demonstration
IMPORT Pipeline, PolynomialFeatures, LinearRegression, mean_squared_error, train_test_split

# Simulate noisy house price data: price ~ 50*sqft - 0.5*sqft^2 + noise
GENERATE 80 samples with RandomState(42)
DEFINE X as sqft (500–3000 sq ft)
DEFINE y as price with quadratic trend + Gaussian noise

SPLIT into train (70%) and test (30%) with random_state=42

FOR degree IN [1, 10]:
    CREATE Pipeline([PolynomialFeatures(degree), LinearRegression()])
    FIT on X_train, y_train
    COMPUTE train_mse = mean_squared_error(y_train, pipe.predict(X_train))
    COMPUTE test_mse  = mean_squared_error(y_test,  pipe.predict(X_test))
    PRINT degree, train_mse rounded to 0, test_mse rounded to 0`,
        starterCode: `import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Noisy house-price data: price ($k) ~ quadratic in sqft with noise
rng = np.random.RandomState(42)
sqft = rng.uniform(500, 3000, 80).reshape(-1, 1)
noise = rng.randn(80) * 30
y = 0.08 * sqft.ravel() - 1.5e-5 * (sqft.ravel() ** 2) + noise + 50

X_train, X_test, y_train, y_test = train_test_split(sqft, y, test_size=0.3, random_state=42)

def fit_poly_and_report(X_train, y_train, X_test, y_test, degree):
    # TODO: Create Pipeline with PolynomialFeatures(degree) and LinearRegression()
    # TODO: Fit on X_train, y_train
    # TODO: Compute train_mse and test_mse using mean_squared_error
    # TODO: Print: f"Degree {degree}: Train MSE={train_mse:.0f}, Test MSE={test_mse:.0f}"
    pass

for deg in [1, 10]:
    fit_poly_and_report(X_train, y_train, X_test, y_test, deg)`,
        expectedOutput: `Degree 1: Train MSE=934, Test MSE=948\nDegree 10: Train MSE=666, Test MSE=9823`,
        solution: `import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

rng = np.random.RandomState(42)
sqft = rng.uniform(500, 3000, 80).reshape(-1, 1)
noise = rng.randn(80) * 30
y = 0.08 * sqft.ravel() - 1.5e-5 * (sqft.ravel() ** 2) + noise + 50

X_train, X_test, y_train, y_test = train_test_split(sqft, y, test_size=0.3, random_state=42)

def fit_poly_and_report(X_train, y_train, X_test, y_test, degree):
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('lr', LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, pipe.predict(X_train))
    test_mse  = mean_squared_error(y_test,  pipe.predict(X_test))
    print(f"Degree {degree}: Train MSE={train_mse:.0f}, Test MSE={test_mse:.0f}")

for deg in [1, 10]:
    fit_poly_and_report(X_train, y_train, X_test, y_test, deg)`,
        hints: [
          'Use Pipeline([("poly", PolynomialFeatures(degree=degree)), ("lr", LinearRegression())])',
          'Call pipe.fit(X_train, y_train) then pipe.predict() for both train and test sets',
          'mean_squared_error(y_true, y_pred) returns the MSE directly'
        ],
        testKeywords: ['Pipeline', 'PolynomialFeatures', 'LinearRegression', 'mean_squared_error', 'train_test_split']
      },
      assignment: {
        title: 'Learning Curves Analysis for Diabetes Prediction',
        description: 'Use `learning_curve` from sklearn to plot how training and validation scores evolve as training set size grows on a diabetes classification task. This reveals whether the model suffers from high bias or high variance.',
        pseudoCode: `# Learning Curves for Diabetes Model Selection
IMPORT learning_curve, LogisticRegression, load_diabetes
IMPORT StandardScaler, Pipeline

LOAD diabetes dataset
BINARIZE target: y = 1 if blood sugar > median else 0

CREATE pipeline = Pipeline([StandardScaler, LogisticRegression(max_iter=1000)])

COMPUTE train_sizes, train_scores, val_scores = learning_curve(
    pipeline, X, y,
    train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0],
    cv=5, scoring='accuracy')

COMPUTE mean_train = train_scores.mean(axis=1)
COMPUTE mean_val   = val_scores.mean(axis=1)

# High Bias: both scores low and converging
# High Variance: large gap between train and val scores
PRINT final_train_score rounded to 3
PRINT final_val_score   rounded to 3
PRINT diagnosis: 'High Bias' or 'High Variance' based on gap`,
        starterCode: `import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve

# Load and binarize diabetes dataset (classify above/below median glucose)
data = load_diabetes()
X = data.data  # 442 samples, 10 features
y = (data.target > np.median(data.target)).astype(int)

# TODO: Create a Pipeline with StandardScaler and LogisticRegression(max_iter=1000, random_state=42)
# TODO: Use learning_curve with train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0], cv=5, scoring='accuracy'
# TODO: Compute mean training and validation scores at each size
# TODO: Print the final (largest) training and validation accuracy
# TODO: Print diagnosis based on gap: 'High Variance' if gap > 0.05 else 'High Bias or Good Fit'
pipe = None
train_sizes = []
train_scores_mean = np.array([])
val_scores_mean   = np.array([])

final_train = 0.0
final_val   = 0.0
gap = final_train - final_val
diagnosis = ''

print(f"Final Train Accuracy: {final_train:.3f}")
print(f"Final Val Accuracy:   {final_val:.3f}")
print(f"Diagnosis: {diagnosis}")`,
        expectedOutput: `Final Train Accuracy: 0.798\nFinal Val Accuracy:   0.774\nDiagnosis: High Bias or Good Fit`,
        solution: `import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve

data = load_diabetes()
X = data.data
y = (data.target > np.median(data.target)).astype(int)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
])

train_sizes, train_scores, val_scores = learning_curve(
    pipe, X, y,
    train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0],
    cv=5, scoring='accuracy'
)

train_scores_mean = train_scores.mean(axis=1)
val_scores_mean   = val_scores.mean(axis=1)

final_train = train_scores_mean[-1]
final_val   = val_scores_mean[-1]
gap = final_train - final_val
diagnosis = 'High Variance' if gap > 0.05 else 'High Bias or Good Fit'

print(f"Final Train Accuracy: {final_train:.3f}")
print(f"Final Val Accuracy:   {final_val:.3f}")
print(f"Diagnosis: {diagnosis}")`,
        hints: [
          'Use learning_curve(pipe, X, y, train_sizes=[0.1,0.3,0.5,0.7,1.0], cv=5, scoring="accuracy")',
          'Mean scores: train_scores.mean(axis=1) and val_scores.mean(axis=1)',
          'Diagnosis: gap > 0.05 → High Variance; else → High Bias or Good Fit'
        ],
        testKeywords: ['learning_curve', 'Pipeline', 'StandardScaler', 'LogisticRegression', 'mean']
      }
    },
    interviewQuestions: [
      { question: 'What is the bias-variance tradeoff?', answer: "It is the tension between a model\'s ability to minimize error on the training data (bias) and its ability to generalize to new, unseen data (variance). As you increase model complexity, bias decreases but variance increases. The goal is to find the sweet spot that minimizes total expected error.", companyTags: ['Google', 'Meta'], difficulty: 'Easy' },
      { question: 'Define "Bias" in the context of machine learning models.', answer: 'Bias refers to the error introduced by approximating a real-world, complex problem with a much simpler model. High bias means the model makes strong, incorrect assumptions, leading to underfitting.', companyTags: ['Amazon', 'Apple'], difficulty: 'Easy' },
      { question: 'Define "Variance" in the context of machine learning models.', answer: "Variance refers to the amount by which the model\'s predictions would change if we estimated it using a different training data set. High variance means the model is highly sensitive to the specific noise in the training data, leading to overfitting.", companyTags: ['Microsoft', 'Netflix'], difficulty: 'Easy' },
      { question: 'How can you detect High Bias using learning curves?', answer: 'In a learning curve, High Bias is indicated when both the training error and the validation error plateau at a high value. They are close to each other, indicating that adding more data does not improve performance because the model is too simple.', companyTags: ['Uber', 'Stripe'], difficulty: 'Medium' },
      { question: 'How can you detect High Variance using learning curves?', answer: 'High Variance is indicated when the training error is very low, but there is a large, persistent gap between the training error and the validation error. The validation error remains high even as more data is added.', companyTags: ['LinkedIn', 'Pinterest'], difficulty: 'Medium' },
      { question: 'What are three strategies to fix a High Bias model?', answer: '1) Add more relevant features. 2) Increase model complexity (e.g., higher degree polynomial, deeper decision trees). 3) Decrease regularization (e.g., lower lambda in Ridge/Lasso).', companyTags: ['Google', 'Tesla'], difficulty: 'Medium' },
      { question: 'What are three strategies to fix a High Variance model?', answer: '1) Get more training data. 2) Reduce the number of features (feature selection). 3) Increase regularization (e.g., higher lambda, higher dropout rate).', companyTags: ['Meta', 'Amazon'], difficulty: 'Medium' },
      { question: 'What is "Irreducible Error"?', answer: 'Irreducible error is the noise inherently present in the problem formulation or the data generation process. It is the variance of the target variable itself. No model, no matter how complex, can ever eliminate this error.', companyTags: ['Apple', 'Microsoft'], difficulty: 'Medium' },
      { question: 'Does adding more training data always improve model performance?', answer: 'No. If the model is suffering from High Bias (underfitting), it lacks the capacity to learn the underlying pattern. Adding a million more rows will just make it converge to the same poor fit faster. Data only heavily helps with High Variance.', companyTags: ['Netflix', 'Uber'], difficulty: 'Medium' },
      { question: 'How does K-Nearest Neighbors (KNN) behave with respect to bias and variance as K changes?', answer: 'A low K (e.g., K=1) creates a highly jagged decision boundary that memorizes the training data, resulting in low bias and high variance. A high K (e.g., K=N) averages all points, predicting the global mean, resulting in high bias and low variance.', companyTags: ['Google', 'Stripe'], difficulty: 'Hard' },
      { question: 'How does the depth of a Decision Tree affect bias and variance?', answer: 'A shallow tree (stump) makes broad generalizations: high bias, low variance. A very deep tree can isolate every single training point into its own leaf: low bias, high variance.', companyTags: ['Meta', 'Pinterest'], difficulty: 'Medium' },
      { question: 'How do Ensemble Methods like Random Forests address the bias-variance tradeoff?', answer: 'Random Forests use "Bagging". They build many fully grown, deep decision trees (low bias, high variance). By averaging the predictions of these uncorrelated trees, the variance is drastically reduced without significantly increasing the bias.', companyTags: ['Amazon', 'LinkedIn'], difficulty: 'Hard' },
      { question: 'How does Gradient Boosting address the bias-variance tradeoff?', answer: 'Gradient Boosting uses "Boosting". It builds many shallow trees (high bias, low variance) sequentially. Each tree corrects the residual errors of the previous ones. Over time, the ensemble reduces the bias dramatically, while the shallow nature of the base learners keeps variance controlled.', companyTags: ['Apple', 'Uber'], difficulty: 'Hard' },
      { question: 'What is the mathematical decomposition of Mean Squared Error?', answer: 'MSE = Bias(theta_hat)^2 + Variance(theta_hat) + Variance(epsilon). The expected squared error is the sum of the squared bias of the estimator, the variance of the estimator, and the variance of the irreducible noise.', companyTags: ['Google', 'Microsoft'], difficulty: 'Advanced' },
      { question: 'Can a model have both high bias and high variance?', answer: "Yes, but it's usually the sign of a poorly chosen model or terrible features. For example, a model might completely ignore the most predictive feature (causing high bias) while fitting complex noise in a totally irrelevant feature (causing high variance).", companyTags: ['Netflix', 'Tesla'], difficulty: 'Advanced' },
      { question: 'Why does Ridge Regularization reduce variance?', answer: "Ridge shrinks the magnitude of the model weights. Smaller weights mean that small fluctuations in the input features (or noise) result in much smaller changes in the output prediction, thus reducing the model\'s variance.", companyTags: ['Meta', 'Stripe'], difficulty: 'Hard' },
      { question: 'If you use a polynomial regression of degree 15 on 10 data points, what will happen?', answer: 'The model will perfectly interpolate the 10 data points (training error will be zero), exhibiting massive overfitting (high variance). Between the data points, the curve will oscillate wildly, leading to astronomical errors on test data.', companyTags: ['Amazon', 'Pinterest'], difficulty: 'Easy' },
      { question: 'What is Occam\'s Razor and how does it relate to the bias-variance tradeoff?', answer: "Occam\'s Razor suggests that among competing hypotheses, the simplest one should be selected. In ML, this means selecting a model with just enough complexity to capture the signal without fitting the noise, which balances bias and variance perfectly.", companyTags: ['Apple', 'LinkedIn'], difficulty: 'Medium' },
      { question: 'How does Dropout in neural networks affect bias and variance?', answer: 'Dropout randomly zeroes out activations during training. This prevents complex co-adaptations among neurons, effectively acting as an ensemble of smaller sub-networks. It increases bias slightly but significantly reduces variance (preventing overfitting).', companyTags: ['Google', 'Meta'], difficulty: 'Hard' },
      { question: 'Is the bias-variance tradeoff applicable to Classification problems?', answer: 'Yes, although the exact mathematical decomposition (Bias^2 + Variance) is derived for squared error in regression. For classification (0-1 loss), the concept perfectly holds: simple decision boundaries underfit (bias), highly contorted ones overfit (variance).', companyTags: ['Microsoft', 'Uber'], difficulty: 'Advanced' }
    ]
};
