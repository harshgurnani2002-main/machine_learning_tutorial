import type { MLModule } from '../../types';

export const kaggleCreditCardFraudProject: MLModule = {
  id: 'kaggle-credit-card-fraud-project',
  title: 'Kaggle Project: Credit Card Fraud Detection',
  category: 'Kaggle Real-World Projects',
  description: 'Detect rare fraud events with imbalanced-learning techniques, threshold tuning, and precision-recall evaluation.',
  formula: 'F_1 = 2 \\cdot \\frac{\\text{precision} \\cdot \\text{recall}}{\\text{precision} + \\text{recall}}',
  theory: `### Real-World Problem
Identify fraudulent card transactions while minimizing false alarms and missed fraud.

### Dataset
- Kaggle: Credit Card Fraud Detection
- Highly imbalanced classes with anonymized PCA-transformed features

### Notebook Coding Approach
1. Start with class ratio analysis and baseline metrics.
2. Split data with stratification.
3. Standardize Amount/Time where needed.
4. Train baseline Logistic Regression with class_weight='balanced'.
5. Compare with tree ensemble (RandomForest/XGBoost).
6. Evaluate using PR-AUC, recall, precision, F1.
7. Tune threshold for business target (for example, high recall).

### Real-world extension
Production fraud systems are threshold-driven and cost-sensitive. Model score alone is not enough without decision policy tuning.`,
  interactiveSummary: 'A realistic imbalanced classification project focused on precision-recall tradeoffs and threshold selection.',
  quiz: [
    {
      id: 'kcfp_q1',
      question: 'Why is accuracy misleading in fraud detection?',
      options: [
        'Because majority non-fraud dominates and can hide poor fraud recall.',
        'Because accuracy cannot be computed for binary tasks.',
        'Because fraud data is always noisy.',
        'Because trees do not output probabilities.'
      ],
      correctAnswer: 'Because majority non-fraud dominates and can hide poor fraud recall.',
      explanation: 'A model can be very accurate by predicting almost all transactions as non-fraud.'
    },
    {
      id: 'kcfp_q2',
      question: 'Which curve is most useful for severe class imbalance?',
      options: ['Precision-Recall curve', 'ROC only', 'Learning curve', 'Silhouette curve'],
      correctAnswer: 'Precision-Recall curve',
      explanation: 'PR curve focuses on positive class performance and is usually more informative in rare-event detection.'
    },
    {
      id: 'kcfp_q3',
      question: 'What does threshold tuning change?',
      options: [
        'The decision boundary for converting probabilities into class labels.',
        'The training data labels.',
        'The model architecture.',
        'The number of features.'
      ],
      correctAnswer: 'The decision boundary for converting probabilities into class labels.',
      explanation: 'The model stays fixed; only the decision cutoff changes, altering recall/precision tradeoff.'
    }
  ],
  notebookCells: [
    {
      id: 'kcfp_overview',
      type: 'markdown',
      title: 'Notebook overview',
      summary: 'Goal: detect rare fraud events using precision-recall metrics and threshold tuning.'
    },
    {
      id: 'kcfp_imports',
      type: 'code',
      title: 'Imports',
      summary: 'Load libraries for imbalanced classification.',
      code: `import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score`
    },
    {
      id: 'kcfp_load',
      type: 'code',
      title: 'Load data',
      summary: 'Read Kaggle CSV and confirm shape.',
      code: `df = pd.read_csv('creditcard.csv')
print(df.shape)`
    },
    {
      id: 'kcfp_ratio',
      type: 'code',
      title: 'Class imbalance check',
      summary: 'Compute fraud rate to understand imbalance.',
      code: `fraud_rate = df['Class'].mean()
print(round(fraud_rate, 4))`
    },
    {
      id: 'kcfp_split_scale',
      type: 'code',
      title: 'Split and scale',
      summary: 'Train/validation split with stratification and scale Amount/Time.',
      code: `X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = X_train.copy()
X_val = X_val.copy()

X_train['AmountScaled'] = scaler.fit_transform(X_train[['Amount']])
X_val['AmountScaled'] = scaler.transform(X_val[['Amount']])

X_train = X_train.drop(columns=['Amount', 'Time'])
X_val = X_val.drop(columns=['Amount', 'Time'])`
    },
    {
      id: 'kcfp_baseline',
      type: 'code',
      title: 'Balanced logistic baseline',
      summary: 'Fit Logistic Regression with class_weight and compute PR-AUC.',
      code: `logit = LogisticRegression(max_iter=1000, class_weight='balanced')
logit.fit(X_train, y_train)
proba = logit.predict_proba(X_val)[:, 1]
pr_auc = average_precision_score(y_val, proba)
print(round(pr_auc, 3))`,
      stageId: 'tutorial'
    },
    {
      id: 'kcfp_rf',
      type: 'code',
      title: 'Tree model comparison',
      summary: 'Compare Random Forest using PR-AUC.',
      code: `rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
proba_rf = rf.predict_proba(X_val)[:, 1]
pr_auc_rf = average_precision_score(y_val, proba_rf)
print(round(pr_auc_rf, 3))`,
      stageId: 'project'
    },
    {
      id: 'kcfp_threshold',
      type: 'code',
      title: 'Threshold tuning',
      summary: 'Sweep thresholds to reach 90% recall with best precision.',
      code: `best_threshold = 0.5
best_precision = 0.0

for t in np.linspace(0.1, 0.9, 17):
    y_pred = (proba >= t).astype(int)
    rec = recall_score(y_val, y_pred)
    if rec >= 0.9:
        prec = precision_score(y_val, y_pred, zero_division=0)
        if prec > best_precision:
            best_precision = prec
            best_threshold = float(t)

print(best_threshold, round(best_precision, 3))`,
      stageId: 'assignment'
    },
    {
      id: 'kcfp_submission',
      type: 'code',
      title: 'Submission export',
      summary: 'Generate predictions for the test set using the chosen threshold.',
      code: `test = pd.read_csv('test.csv')
test = test.copy()
test['AmountScaled'] = scaler.transform(test[['Amount']])
test = test.drop(columns=['Amount', 'Time'])

test_proba = logit.predict_proba(test)[:, 1]
test_pred = (test_proba >= best_threshold).astype(int)

submission = pd.DataFrame({
    'Time': np.arange(len(test_pred)),
    'Class': test_pred
})
submission.to_csv('submission.csv', index=False)
print(submission.head())`
    }
  ],
  coding: {
    tutorial: {
      title: 'Compute Precision, Recall, and F1',
      description: 'Evaluate imbalanced predictions with confusion-matrix-derived metrics.',
      pseudoCode: '1. Compute TP, FP, FN.\n2. Calculate precision and recall.\n3. Compute F1.',
      starterCode: `tp, fp, fn = 45, 15, 20
# TODO
precision = 0.0
recall = 0.0
f1 = 0.0
print(round(f1, 3))`,
      expectedOutput: '0.72',
      solution: `tp, fp, fn = 45, 15, 20
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
print(round(f1, 3))`,
      hints: ['Use safe denominator checks in real pipelines.'],
      testKeywords: ['precision', 'recall', 'f1']
    },
    project: {
      title: 'Train Balanced Logistic Baseline',
      description: 'Fit LogisticRegression with class_weight and report PR-AUC.',
      pseudoCode: '1. Train logistic model with class_weight="balanced".\n2. Predict probabilities.\n3. Compute average precision.',
      starterCode: `from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

# assume X_train, y_train, X_val, y_val
model = LogisticRegression(max_iter=1000)
# TODO
pr_auc = 0.0
print(round(pr_auc, 3))`,
      expectedOutput: '0.815',
      solution: `from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

# assume X_train, y_train, X_val, y_val
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)
proba = model.predict_proba(X_val)[:, 1]
pr_auc = average_precision_score(y_val, proba)
print(round(pr_auc, 3))`,
      hints: ['Use probability scores, not hard labels, for PR-AUC.'],
      testKeywords: ['class_weight', 'predict_proba', 'average_precision_score']
    },
    assignment: {
      title: 'Threshold Optimization for Recall Target',
      description: 'Find a threshold that reaches at least 90% recall with best possible precision.',
      pseudoCode: '1. Scan thresholds from 0.1 to 0.9.\n2. Compute recall and precision.\n3. Keep best precision where recall >= 0.9.',
      starterCode: `import numpy as np
from sklearn.metrics import precision_score, recall_score

# assume y_true and y_proba exist
best_threshold = 0.5
best_precision = 0.0
# TODO
print(best_threshold >= 0.1)`,
      expectedOutput: 'True',
      solution: `import numpy as np
from sklearn.metrics import precision_score, recall_score

# assume y_true and y_proba exist
best_threshold = 0.5
best_precision = 0.0

for t in np.linspace(0.1, 0.9, 17):
    y_pred = (y_proba >= t).astype(int)
    rec = recall_score(y_true, y_pred)
    if rec >= 0.9:
      prec = precision_score(y_true, y_pred, zero_division=0)
      if prec > best_precision:
          best_precision = prec
          best_threshold = float(t)

print(best_threshold >= 0.1)`,
      hints: ['This is policy tuning, not model retraining.', 'Use recall constraint first.'],
      testKeywords: ['linspace', 'precision_score', 'recall_score']
    }
  },
  interviewQuestions: [
    {
      question: 'How would you explain threshold selection to a fraud operations team?',
      answer: 'I would map thresholds to expected alert volume, missed-fraud rate, and manual review capacity, then choose an operating point that minimizes business loss under staffing constraints.',
      companyTags: ['Stripe', 'PayPal'],
      difficulty: 'Advanced'
    }
  ]
};
