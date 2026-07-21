import type { MLModule } from '../../types';

export const kaggleLoanDefaultRiskProject: MLModule = {
  id: 'kaggle-loan-default-risk-project',
  title: 'Kaggle Project: Loan Default Risk',
  category: 'Kaggle Real-World Projects',
  description: 'Predict loan default risk with imbalanced data, categorical encoding, and gradient boosting models.',
  formula: 'AUC = ∫ TPR(FPR) dFPR',
  theory: `### Real-World Problem
Estimate probability of loan default so credit teams can price risk appropriately.

### Dataset
- Kaggle: Home Credit Default Risk (or similar loan datasets)
- Mix of numeric and categorical attributes with missing values

### Notebook Coding Approach
1. Inspect class imbalance and baseline default rate.
2. Encode categoricals and handle missing values.
3. Train a balanced Logistic Regression baseline.
4. Compare Gradient Boosting / LightGBM-style models.
5. Evaluate with ROC-AUC and calibration checks.
6. Generate probability-based submission.

### Real-world extension
Probability calibration and thresholding are critical for credit decision policies.`,
  interactiveSummary: 'A credit-risk pipeline focused on AUC, class imbalance, and tree-based models.',
  quiz: [
    {
      id: 'kldrp_q1',
      question: 'Why is ROC-AUC preferred over accuracy for default risk?',
      options: [
        'It measures ranking quality across thresholds under class imbalance.',
        'It requires no validation set.',
        'It works only for regression models.',
        'It always equals F1.'
      ],
      correctAnswer: 'It measures ranking quality across thresholds under class imbalance.',
      explanation: 'AUC is robust to class imbalance and focuses on ranking probability scores.'
    },
    {
      id: 'kldrp_q2',
      question: 'What does class_weight="balanced" do?',
      options: [
        'Reweights classes inversely to their frequency.',
        'Removes all missing values.',
        'Forces all predictions to 50/50.',
        'Disables regularization.'
      ],
      correctAnswer: 'Reweights classes inversely to their frequency.',
      explanation: 'Balanced weights help minority class influence the loss more.'
    },
    {
      id: 'kldrp_q3',
      question: 'Which model family often performs best on tabular credit data?',
      options: ['Gradient boosted trees', 'CNNs', 'KMeans', 'PCA'],
      correctAnswer: 'Gradient boosted trees',
      explanation: 'Boosted trees excel on structured tabular data and handle non-linearities well.'
    }
  ],
  notebookCells: [
    {
      id: 'kldrp_overview',
      type: 'markdown',
      title: 'Notebook overview',
      summary: 'Goal: predict default risk with ROC-AUC and strong tree models.'
    },
    {
      id: 'kldrp_imports',
      type: 'code',
      title: 'Imports',
      summary: 'Load libraries for preprocessing and modeling.',
      code: `import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score`
    },
    {
      id: 'kldrp_load',
      type: 'code',
      title: 'Load data',
      summary: 'Read Kaggle train/test CSVs.',
      code: `train = pd.read_csv('application_train.csv')
test = pd.read_csv('application_test.csv')
print(train.shape, test.shape)`
    },
    {
      id: 'kldrp_missing',
      type: 'code',
      title: 'Missing value scan',
      summary: 'Rank columns by missing fraction.',
      code: `missing = train.isna().mean().sort_values(ascending=False)
print(missing.head(8))`
    },
    {
      id: 'kldrp_encode',
      type: 'code',
      title: 'Encode categoricals',
      summary: 'One-hot encode categorical columns and align train/test.',
      code: `y = train['TARGET']
train_ids = train['SK_ID_CURR']
test_ids = test['SK_ID_CURR']

train = train.drop(columns=['TARGET', 'SK_ID_CURR'])
test = test.drop(columns=['SK_ID_CURR'])

train = pd.get_dummies(train, drop_first=True)
test = pd.get_dummies(test, drop_first=True)

train, test = train.align(test, join='left', axis=1, fill_value=0)`
    },
    {
      id: 'kldrp_split',
      type: 'code',
      title: 'Train/validation split',
      summary: 'Split for evaluation with stratification.',
      code: `X_train, X_val, y_train, y_val = train_test_split(
    train, y, test_size=0.2, random_state=42, stratify=y
)`
    },
    {
      id: 'kldrp_balance',
      type: 'code',
      title: 'Balanced logistic baseline',
      summary: 'Train Logistic Regression with class weights.',
      code: `logit = LogisticRegression(max_iter=2000, class_weight='balanced')
logit.fit(X_train, y_train)
logit_proba = logit.predict_proba(X_val)[:, 1]
print(round(roc_auc_score(y_val, logit_proba), 3))`,
      stageId: 'tutorial'
    },
    {
      id: 'kldrp_gbm',
      type: 'code',
      title: 'Gradient boosting model',
      summary: 'Train a boosting model for stronger non-linear performance.',
      code: `gbm = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, random_state=42)
gbm.fit(X_train, y_train)
gbm_proba = gbm.predict_proba(X_val)[:, 1]
print(round(roc_auc_score(y_val, gbm_proba), 3))`,
      stageId: 'project'
    },
    {
      id: 'kldrp_submit',
      type: 'code',
      title: 'Submission probabilities',
      summary: 'Train on full data and export probabilities.',
      code: `gbm.fit(train, y)
proba = gbm.predict_proba(test)[:, 1]
submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': proba})
submission.to_csv('submission.csv', index=False)
print(submission.head())`,
      stageId: 'assignment'
    }
  ],
  coding: {
    tutorial: {
      title: 'Class-Weighted Logistic Baseline',
      description: 'Fit Logistic Regression using balanced class weights.',
      pseudoCode: '1. Instantiate LogisticRegression with class_weight="balanced".\n2. Fit on training data.\n3. Predict probabilities.',
      starterCode: `from sklearn.linear_model import LogisticRegression

# assume X_train, y_train, X_val, y_val
model = LogisticRegression(max_iter=2000)
# TODO
print(model is not None)`,
      expectedOutput: 'True',
      solution: `from sklearn.linear_model import LogisticRegression

# assume X_train, y_train, X_val, y_val
model = LogisticRegression(max_iter=2000, class_weight='balanced')
model.fit(X_train, y_train)
print(model is not None)`,
      hints: ['Balanced weights help with rare defaults.'],
      testKeywords: ['class_weight', 'LogisticRegression']
    },
    project: {
      title: 'Gradient Boosting Model',
      description: 'Train a histogram-based gradient boosting classifier.',
      pseudoCode: '1. Create HistGradientBoostingClassifier.\n2. Fit on training set.\n3. Predict validation probabilities.',
      starterCode: `from sklearn.ensemble import HistGradientBoostingClassifier

# assume X_train, y_train
model = HistGradientBoostingClassifier()
# TODO
print(model is not None)`,
      expectedOutput: 'True',
      solution: `from sklearn.ensemble import HistGradientBoostingClassifier

# assume X_train, y_train
model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)
print(model is not None)`,
      hints: ['Use moderate depth to reduce overfitting.'],
      testKeywords: ['HistGradientBoostingClassifier', 'fit']
    },
    assignment: {
      title: 'Export Default Probabilities',
      description: 'Generate probability predictions and create submission.csv.',
      pseudoCode: '1. Predict probabilities on test data.\n2. Build submission DataFrame.\n3. Save to CSV.',
      starterCode: `import pandas as pd

# assume model, X_test, test_ids
submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': []})
# TODO
print(list(submission.columns))`,
      expectedOutput: "['SK_ID_CURR', 'TARGET']",
      solution: `import pandas as pd

# assume model, X_test, test_ids
proba = model.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': proba})
submission.to_csv('submission.csv', index=False)
print(list(submission.columns))`,
      hints: ['Submit probabilities, not hard labels, for AUC competitions.'],
      testKeywords: ['predict_proba', 'to_csv', 'TARGET']
    }
  },
  interviewQuestions: [
    {
      question: 'How would you calibrate a default probability model?',
      answer: 'Use calibration curves or Platt scaling/isotonic regression on a holdout set and monitor Brier score.',
      companyTags: ['JPMorgan', 'Capital One'],
      difficulty: 'Advanced'
    }
  ]
};
