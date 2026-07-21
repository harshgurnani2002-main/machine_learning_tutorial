import type { MLModule } from '../../types';

export const modelEvaluation: MLModule = {
  id: 'model-evaluation',
  title: 'Model Evaluation & Metrics',
  category: 'Foundations & Math',
  description: 'Master confusion matrices, ROC curves, cross-validation, and regression metrics to properly evaluate any ML model across classification and regression tasks.',
  formula: '\\text{ROC-AUC} = \\int_0^1 \\text{TPR}(\\text{FPR}^{-1}(x)) \\, dx',
  theory: `## Model Evaluation & Metrics — A Comprehensive Guide
  
  ### 1. The Confusion Matrix
  
  At the heart of classification evaluation lies the confusion matrix — a 2×2 table comparing predicted vs actual values:
  
  |              | Predicted Positive | Predicted Negative |
  |--------------|-------------------|-------------------|
  | **Actual Positive** | True Positive (TP) | False Negative (FN) |
  | **Actual Negative** | False Positive (FP) | True Negative (TN) |
  
  - **True Positive (TP)**: Correctly predicted positive class
  - **True Negative (TN)**: Correctly predicted negative class
  - **False Positive (FP)**: Type I error — incorrectly predicted positive
  - **False Negative (FN)**: Type II error — missed a positive
  
  ### 2. Key Classification Metrics
  
  Given the confusion matrix entries, we derive:
  
  **Accuracy:** \\(\\frac{TP + TN}{TP + TN + FP + FN}\\) — overall correctness. Misleading for imbalanced datasets (a 99% negative dataset yields 99% accuracy by guessing "negative" always).
  
  **Precision (Positive Predictive Value):** \\(\\frac{TP}{TP + FP}\\) — of all positive predictions, how many are correct? High precision matters when false positives are costly (e.g., spam filters).
  
  **Recall (Sensitivity / TPR):** \\(\\frac{TP}{TP + FN}\\) — of all actual positives, how many were found? High recall matters when false negatives are costly (e.g., cancer detection).
  
  **F1-Score:** \\(2 \\cdot \\frac{Precision \\cdot Recall}{Precision + Recall}\\) — harmonic mean of precision and recall. Balances both when class distribution is uneven.
  
  **Specificity (TNR):** \\(\\frac{TN}{TN + FP}\\) — of all actual negatives, how many were correctly identified?
  
  ### 3. ROC Curve and AUC
  
  The Receiver Operating Characteristic (ROC) curve plots True Positive Rate (Recall) against False Positive Rate (1 - Specificity) across all possible classification thresholds. Each threshold produces a different TPR/FPR pair tracing out the curve.
  
  - **Perfect classifier** (AUC = 1.0): curve hugs the top-left corner
  - **Random classifier** (AUC = 0.5): diagonal line from (0,0) to (1,1)
  - **Worse than random** (AUC < 0.5): curve dips below diagonal
  
  **AUC (Area Under the ROC Curve)** aggregates model performance across all thresholds into a single number: the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative one. AUC is computed via the trapezoidal rule.
  
  The **Precision-Recall (PR) curve** is preferred over ROC when the positive class is rare or when false positives are especially costly.
  
  ### 4. Cross-Validation
  
  Cross-validation estimates model performance on unseen data without requiring a separate validation set:
  
  - **k-Fold CV:** Split data into \\(k\\) folds. Train on \\(k-1\\) folds, validate on the held-out fold. Repeat \\(k\\) times. Report mean ± std of the metric. Common choices: \\(k = 5\\) or \\(k = 10\\). Higher \\(k\\) gives lower bias but higher variance.
  - **Stratified k-Fold:** Preserves class proportions in each fold — essential for imbalanced datasets.
  - **Leave-One-Out CV (LOOCV):** \\(k = n\\), where \\(n\\) is sample size. Each sample is its own validation fold. High variance, computationally expensive.
  - **Repeated k-Fold:** Run k-fold multiple times with different random splits for more stable estimates.
  
  ### 5. Learning Curves
  
  A learning curve plots training and validation error against training set size (or training iterations):
  
  - **High Bias (Underfitting):** Both training and validation error are high and converge quickly. Adding more data won't help — need a more complex model.
  - **High Variance (Overfitting):** Large gap between low training error and high validation error. Adding more data will likely close the gap.
  - **Good Fit:** Both curves converge to a low error with minimal gap.
  
  Learning curves are the primary diagnostic tool for determining whether to collect more data, increase model complexity, or add regularization.
  
  ### 6. Regression Metrics
  
  For continuous target variables:
  
  - **Mean Squared Error (MSE):** \\(\\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2\\) — penalizes large errors quadratically. Sensitive to outliers.
  - **Mean Absolute Error (MAE):** \\(\\frac{1}{n}\\sum_{i=1}^{n}|y_i - \\hat{y}_i|\\) — linear penalty, more robust to outliers.
  - **R² (Coefficient of Determination):** \\(1 - \\frac{\\sum (y_i - \\hat{y}_i)^2}{\\sum (y_i - \\bar{y})^2}\\) — proportion of variance explained by the model. Best possible: 1.0. Can be negative for terrible models.
  - **Adjusted R²:** \\(1 - \\frac{(1 - R²)(n - 1)}{n - p - 1}\\) — penalizes adding irrelevant features. Increases only if a new feature improves model more than expected by chance.
  
  ### 7. Overfitting Detection via Train/Test Gap
  
  The fundamental diagnostic for overfitting is the gap between training and test (or validation) performance:
  
  - **Small gap + high error** → underfitting (high bias)
  - **Small gap + low error** → good fit
  - **Large gap + low train error** → overfitting (high variance)
  
  Remedies for overfitting: more training data, feature selection, regularization (L1/L2), early stopping, dropout, simpler model.
  
  ### 8. Metric Selection Guide
  
  Choosing the right metric depends on problem characteristics:
  
  | Scenario | Primary Metric |
  |----------|---------------|
  | Balanced classes | Accuracy |
  | Imbalanced classes | F1-score, PR-AUC, Recall |
  | Rare positive class | Precision@K, Average Precision |
  | Ranking quality | ROC-AUC |
  | Regression | MSE (low noise), MAE (robust), R² (interpretable) |
  | Cost-sensitive | Custom loss / Precision-Recall tradeoff |
  
  **Class Imbalance:** When one class severely outnumbers another, accuracy becomes misleading. Precision, Recall, F1, and the PR curve give a more truthful picture. In extreme cases (fraud detection, rare disease diagnosis), consider resampling or cost-sensitive learning.
  
  ### Summary
  
  Effective model evaluation requires matching metrics to problem context. Always pair point estimates (accuracy F1) with diagnostic tools (confusion matrix ROC curve learning curve). Cross-validation produces more reliable performance estimates than a single train-test split.
  
  #### Python Implementation
  
  \`\`\`python
  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
  
  y_true = [1, 0, 1, 1, 0, 1]
  y_pred = [1, 0, 1, 0, 0, 1]
  print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
  print(f"Precision: {precision_score(y_true, y_pred):.2f}")
  print(f"Recall: {recall_score(y_true, y_pred):.2f}")
  print(f"F1: {f1_score(y_true, y_pred):.2f}")
  \`\`\``,


  interactiveSummary: 'Explore how classification thresholds affect the confusion matrix and derived metrics. The simulator lets you drag a threshold slider to trace an ROC curve in real time, toggle between balanced and imbalanced datasets, and inspect how precision, recall, F1, and accuracy change. You can also visualize learning curves to diagnose bias and variance.',
  simulatorId: 'model-eval',
  quiz: [
    {
      id: 'eval_q1',
      question: 'In a medical test for a rare disease, which metric is most important to minimize?',
      options: [
        'False Negatives (FN) — a missed diagnosis could be life-threatening',
        'False Positives (FP) — they cause unnecessary stress',
        'True Negatives (TN) — they confirm health',
        'Accuracy — overall correctness is all that matters'
      ],
      correctAnswer: 'False Negatives (FN) — a missed diagnosis could be life-threatening',
      explanation: 'In disease screening, the cost of missing a positive case (FN) is far higher than a false alarm (FP). Recall (sensitivity) is the priority metric.'
    },
    {
      id: 'eval_q2',
      question: 'A classifier achieves 99% accuracy on a dataset where 99% of samples are negative. Which statement is correct?',
      options: [
        'Accuracy is misleading due to class imbalance — examine precision/recall',
        '99% accuracy proves the model is excellent',
        'The model must have high F1 score too',
        'Accuracy is the only metric that matters for production'
      ],
      correctAnswer: 'Accuracy is misleading due to class imbalance — examine precision/recall',
      explanation: 'A naive classifier that always predicts "negative" achieves 99% accuracy on a 99% negative dataset. Always check precision, recall, and the confusion matrix.'
    },
    {
      id: 'eval_q3',
      question: 'What does an AUC of 0.5 indicate?',
      options: [
        'The classifier performs no better than random guessing',
        'The classifier is perfect',
        'The classifier is perfectly inverted',
        'The classifier has zero variance'
      ],
      correctAnswer: 'The classifier performs no better than random guessing',
      explanation: 'AUC = 0.5 means the ROC curve lies along the diagonal — the model has no discriminatory power. AUC < 0.5 means the classifier is worse than random.'
    },
    {
      id: 'eval_q4',
      question: 'Which cross-validation strategy is most appropriate for an imbalanced classification dataset?',
      options: [
        'Stratified k-Fold — preserves class proportions in each fold',
        'Standard k-Fold — no modification needed',
        'LOOCV — always the best choice',
        'Hold-out validation with a 70/30 split'
      ],
      correctAnswer: 'Stratified k-Fold — preserves class proportions in each fold',
      explanation: 'Standard k-Fold may create folds with zero positive samples. Stratified k-Fold ensures each fold maintains the original class distribution.'
    },
    {
      id: 'eval_q5',
      question: 'A learning curve shows low training error but high validation error. This is a classic sign of:',
      options: [
        'High variance (overfitting)',
        'High bias (underfitting)',
        'Optimal model complexity',
        'The model needs more features'
      ],
      correctAnswer: 'High variance (overfitting)',
      explanation: 'A large gap between low train error and high validation error indicates the model has memorized the training set and fails to generalize.'
    },
    {
      id: 'eval_q6',
      question: 'What is the key advantage of Adjusted R² over regular R²?',
      options: [
        'It penalizes adding irrelevant features',
        'It always gives a higher value than R²',
        'It works only for classification',
        'It eliminates the need for cross-validation'
      ],
      correctAnswer: 'It penalizes adding irrelevant features',
      explanation: 'Adjusted R² increases only if a new predictor improves the model more than expected by chance, preventing overfitting through feature proliferation.'
    },
    {
      id: 'eval_q7',
      question: 'Which metric is the most robust to outliers in regression tasks?',
      options: [
        'MAE (Mean Absolute Error)',
        'MSE (Mean Squared Error)',
        'RMSE (Root Mean Squared Error)',
        'Log Loss'
      ],
      correctAnswer: 'MAE (Mean Absolute Error)',
      explanation: 'MAE uses absolute differences, giving equal weight to all errors. MSE squares the errors, making it heavily influenced by outliers.'
    },
    {
      id: 'eval_q8',
      question: 'Given TP = 80, FP = 20, FN = 10, TN = 90, what is the F1 score?',
      options: [
        '0.842',
        '0.889',
        '0.727',
        '0.800'
      ],
      correctAnswer: '0.842',
      explanation: 'Precision = 80/(80+20) = 0.80. Recall = 80/(80+10) = 0.889. F1 = 2*(0.80*0.889)/(0.80+0.889) = 2*0.711/1.689 = 0.842.'
    }
  ],
  coding: {
    tutorial: {
      title: 'Confusion Matrix From Scratch',
      description: 'Implement a confusion matrix calculator without using scikit-learn. Given arrays of true labels and predicted labels, compute TP, TN, FP, FN, then derive accuracy, precision, recall, and F1.\n\n*Hint: In scikit-learn, the equivalent is `sklearn.metrics.confusion_matrix(y_true, y_pred)`.*',
      pseudoCode: '1. Initialize TP=TN=FP=FN=0\n2. Loop over each (true, pred) pair\n3. If true=1 and pred=1 → increment TP\n4. If true=0 and pred=0 → increment TN\n5. If true=0 and pred=1 → increment FP\n6. If true=1 and pred=0 → increment FN\n7. Compute accuracy, precision, recall, F1 from the counts\n8. Return all metrics',
      starterCode: `import numpy as np

def confusion_matrix_metrics(y_true, y_pred):
    tp = tn = fp = fn = 0
    # TODO: loop and count
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    return tp, tn, fp, fn, accuracy, precision, recall, f1

# Test case
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 0])
tp, tn, fp, fn, acc, prec, rec, f1 = confusion_matrix_metrics(y_true, y_pred)
print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
print(f"Accuracy={acc:.3f} Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")`,
      expectedOutput: 'TP=4 TN=3 FP=1 FN=2\nAccuracy=0.700 Precision=0.800 Recall=0.667 F1=0.727',
      solution: `import numpy as np

def confusion_matrix_metrics(y_true, y_pred):
    tp = tn = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1: tp += 1
        elif t == 0 and p == 0: tn += 1
        elif t == 0 and p == 1: fp += 1
        elif t == 1 and p == 0: fn += 1
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return tp, tn, fp, fn, accuracy, precision, recall, f1

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 0])
tp, tn, fp, fn, acc, prec, rec, f1 = confusion_matrix_metrics(y_true, y_pred)
print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
print(f"Accuracy={acc:.3f} Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")`,
      hints: [
        'Use a simple for loop with zip(y_true, y_pred) to compare each pair.',
        'Guard division by zero for precision and recall using an if-check.',
        'F1 is the harmonic mean: 2 * (P * R) / (P + R).'
      ],
      testKeywords: [
        'zip',
        'tp += 1',
        'accuracy ='
      ]
    },
    project: {
      title: 'ROC Curve with sklearn',
      description: 'Use sklearn to train a Logistic Regression model, generate predicted probabilities, compute the ROC curve points, and calculate AUC. Plot the ROC curve to visually assess model discrimination.\n\n*Dataset: Breast cancer diagnostic features with malignant/benign labels.*',
      pseudoCode: '1. Load and split data into train/test sets\n2. Scale features with StandardScaler\n3. Train LogisticRegression\n4. Get predicted probabilities for the positive class using predict_proba\n5. Compute FPR, TPR, thresholds using roc_curve\n6. Compute AUC using roc_auc_score\n7. Print the AUC score',
      starterCode: `import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

# Breast cancer features (8 samples, 4 features)
X = np.array([
    [17.99, 10.38, 122.8, 1001.0],
    [20.57, 17.77, 132.9, 1326.0],
    [11.42, 20.38, 77.58, 386.1],
    [11.29, 13.04, 73.26, 384.8],
    [12.45, 15.70, 82.57, 477.1],
    [18.25, 19.98, 119.6, 1040.0],
    [13.71, 20.83, 90.20, 577.9],
    [13.00, 21.82, 87.50, 519.8],
])
y = np.array([1, 1, 1, 0, 0, 1, 0, 0])

# TODO: Split (test_size=0.25, random_state=42)
# TODO: Scale features
# TODO: Fit LogisticRegression(max_iter=1000)
# TODO: Get positive-class probabilities
# TODO: Compute fpr, tpr, roc_auc

fpr = np.array([0.0, 0.5, 1.0])
tpr = np.array([0.0, 0.5, 1.0])
roc_auc = 0.5
print(f"ROC AUC: {roc_auc:.3f}")`,
      expectedOutput: 'ROC AUC: 1.000',
      solution: `import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

X = np.array([
    [17.99, 10.38, 122.8, 1001.0],
    [20.57, 17.77, 132.9, 1326.0],
    [11.42, 20.38, 77.58, 386.1],
    [11.29, 13.04, 73.26, 384.8],
    [12.45, 15.70, 82.57, 477.1],
    [18.25, 19.98, 119.6, 1040.0],
    [13.71, 20.83, 90.20, 577.9],
    [13.00, 21.82, 87.50, 519.8],
])
y = np.array([1, 1, 1, 0, 0, 1, 0, 0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
y_proba = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC AUC: {roc_auc:.3f}")`,
      hints: [
        'Use train_test_split with test_size=0.25 and random_state=42.',
        'Always scale features before fitting LogisticRegression.',
        'predict_proba returns a 2D array; use [:, 1] for positive class probabilities.'
      ],
      testKeywords: [
        'train_test_split',
        'StandardScaler',
        'predict_proba',
        'roc_auc_score'
      ]
    },
    assignment: {
      title: 'k-Fold Cross-Validation',
      description: 'Implement k-fold cross-validation manually without sklearn\'s cross_val_score. Split data into k folds, train on k-1 folds, evaluate on the held-out fold, and report mean ± std of accuracy across all folds.\n\n*In scikit-learn, this is done via sklearn.model_selection.cross_val_score.*',
      pseudoCode: '1. Shuffle data indices randomly\n2. Split indices into k equal folds\n3. For each fold:\n   a. Use fold i as validation set\n   b. Use remaining k-1 folds as training set\n   c. Train classifier on training set\n   d. Predict on validation set\n   e. Compute accuracy\n4. Return mean and standard deviation of accuracies',
      starterCode: `import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=5, random_state=42)

def manual_kfold_cv(X, y, k=5):
    n = len(X)
    indices = np.random.RandomState(42).permutation(n)
    fold_sizes = np.full(k, n // k)
    fold_sizes[:n % k] += 1
    current = 0
    accuracies = []
    
    # TODO: For each fold, split train/val, fit, predict, score
    
    return np.mean(accuracies), np.std(accuracies)

mean_acc, std_acc = manual_kfold_cv(X, y, k=5)
print(f"Mean accuracy: {mean_acc:.3f} +/- {std_acc:.3f}")`,
      expectedOutput: 'Mean accuracy: 0.840 +/- 0.066',
      solution: `import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=5, random_state=42)

def manual_kfold_cv(X, y, k=5):
    n = len(X)
    indices = np.random.RandomState(42).permutation(n)
    fold_sizes = np.full(k, n // k)
    fold_sizes[:n % k] += 1
    current = 0
    accuracies = []
    
    for fold_size in fold_sizes:
        val_idx = indices[current:current + fold_size]
        train_idx = np.concatenate([indices[:current], indices[current + fold_size:]])
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        acc = np.mean(model.predict(X_val) == y_val)
        accuracies.append(acc)
        current += fold_size
    
    return np.mean(accuracies), np.std(accuracies)

mean_acc, std_acc = manual_kfold_cv(X, y, k=5)
print(f"Mean accuracy: {mean_acc:.3f} +/- {std_acc:.3f}")`,
      hints: [
        'Use np.concatenate to merge the training fold indices.',
        'Compute accuracy per fold as mean(predictions == y_val).',
        'Use np.mean(accuracies) and np.std(accuracies) for the final report.'
      ],
      testKeywords: [
        'fold_size',
        'np.concatenate',
        'DecisionTreeClassifier',
        'np.mean'
      ]
    }
  },
  interviewQuestions: [
    {
      question: 'Explain the difference between ROC-AUC and PR-AUC. When would you use one over the other?',
      answer: 'ROC-AUC plots TPR vs FPR across thresholds and is insensitive to class imbalance — a model can show high ROC-AUC even on very imbalanced data. PR-AUC plots precision vs recall and is far more sensitive to performance on the positive (rare) class. Use ROC-AUC for balanced datasets or when you care equally about both classes. Use PR-AUC for highly imbalanced datasets, especially when the positive class is the focus (fraud detection, rare disease diagnosis).',
      companyTags: ['Google', 'Meta', 'Stripe'],
      difficulty: 'Medium'
    },
    {
      question: 'What is the difference between stratified k-fold and regular k-fold cross-validation?',
      answer: 'Regular k-fold splits data into k random folds without considering class distribution. In imbalanced datasets, some folds may end up with zero positive samples. Stratified k-fold ensures each fold preserves the original class proportion — if the full dataset is 80/20, each fold will also be approximately 80/20. This produces more reliable and less variable performance estimates.',
      companyTags: ['Amazon', 'Netflix'],
      difficulty: 'Easy'
    },
    {
      question: 'How would you diagnose whether a model is overfitting or underfitting using learning curves?',
      answer: 'Plot training error and validation error against training set size. Underfitting (high bias): both curves converge at a high error value — the model is too simple. Adding more data won\'t help; increase complexity or add features. Overfitting (high variance): large gap between low training error and high validation error. The model memorizes training data but fails to generalize. Solutions include more training data, regularization, feature selection, or simpler model architecture.',
      companyTags: ['Apple', 'Uber'],
      difficulty: 'Medium'
    },
    {
      question: 'What is the F1 score and why is it better than accuracy for imbalanced datasets?',
      answer: 'F1 score is the harmonic mean of precision and recall: 2 * (P * R) / (P + R). Unlike accuracy, it considers both false positives and false negatives equally. On a 99% negative dataset, a naive "always negative" classifier achieves 99% accuracy but has undefined F1 (zero denominator since TP = 0). F1 forces the model to actually identify positives, making it a far more honest metric for imbalanced problems.',
      companyTags: ['Google', 'Bloomberg'],
      difficulty: 'Easy'
    },
    {
      question: 'Explain the bias-variance tradeoff in the context of cross-validation results.',
      answer: 'When k is small (e.g., k=2), each training fold is small, so individual models have high bias (trained on less data) but the average estimate has low variance. When k is large (k=10 or LOOCV), each model trains on nearly all the data (low bias) but the estimate is more variable across runs. In practice, k=5 or k=10 balances this tradeoff. The standard deviation across folds also directly indicates model variance — high variance across folds suggests instability in the learning algorithm.',
      companyTags: ['Meta', 'Capital One', 'Airbnb'],
      difficulty: 'Hard'
    }
  ]
};
