import type { MLModule } from '../../types';

export const kaggleTitanicProject: MLModule = {
  id: 'kaggle-titanic-project',
  title: 'Kaggle Project: Titanic Survival Prediction',
  category: 'Kaggle Real-World Projects',
  description: 'Build a full binary classification workflow on the Titanic dataset with feature engineering, validation, and submission generation.',
  formula: 'P(y=1 \\mid x) = \\sigma(w^T x + b)',
  theory: `### Real-World Problem
Predict whether a passenger survived the Titanic disaster using structured passenger data.

### Dataset
- Kaggle: Titanic - Machine Learning from Disaster
- Files: train.csv, test.csv, gender_submission.csv

### Why this project matters
This is a classic production-like tabular classification task:
- Missing values (Age, Cabin)
- Mixed feature types (numerical + categorical)
- Small data regime where validation discipline matters

### Coding Approach (Notebook Flow)
1. Load train/test CSV files.
2. Perform EDA: target balance, null map, simple distributions.
3. Build features:
   - Title from Name
   - FamilySize = SibSp + Parch + 1
   - IsAlone from FamilySize
4. Preprocess with ColumnTransformer:
   - Numeric: impute + scale
   - Categorical: impute + one-hot
5. Train baseline Logistic Regression.
6. Compare with RandomForestClassifier.
7. Evaluate via cross-validation (ROC-AUC and accuracy).
8. Fit best model on full training data and create Kaggle submission CSV.

### Real-world extension
Track experiment variants (feature set + model + CV score) in a simple table to avoid random trial-and-error.`,
  interactiveSummary: 'A complete beginner-friendly Kaggle workflow from messy tabular data to a submission-ready prediction file.',
  quiz: [
    {
      id: 'ktp_q1',
      question: 'Why should preprocessing be fit only on training folds during cross-validation?',
      options: [
        'To avoid data leakage from validation into training.',
        'To make training run faster.',
        'Because one-hot encoding only works on train data.',
        'To reduce model size.'
      ],
      correctAnswer: 'To avoid data leakage from validation into training.',
      explanation: 'Fitting imputers/scalers/encoders on all data leaks information and inflates validation metrics.'
    },
    {
      id: 'ktp_q2',
      question: 'Which metric is often more informative than raw accuracy for imbalanced binary problems?',
      options: ['ROC-AUC', 'MSE', 'R2', 'Silhouette Score'],
      correctAnswer: 'ROC-AUC',
      explanation: 'ROC-AUC measures ranking quality across thresholds, making it robust when class distribution is uneven.'
    },
    {
      id: 'ktp_q3',
      question: 'What is the purpose of feature FamilySize in Titanic?',
      options: [
        'To inject domain signal about social/evacuation context.',
        'To replace target labels.',
        'To normalize ticket prices.',
        'To reduce dataframe rows.'
      ],
      correctAnswer: 'To inject domain signal about social/evacuation context.',
      explanation: 'Domain-inspired features often improve tabular model performance significantly.'
    }
  ],
  notebookCells: [
    {
      id: 'ktp_overview',
      type: 'markdown',
      title: 'Notebook overview',
      summary: 'Goal: predict Survived (0/1) from passenger features using leakage-safe preprocessing and cross-validation.'
    },
    {
      id: 'ktp_imports',
      type: 'code',
      title: 'Imports',
      summary: 'Load core libraries for data prep, modeling, and evaluation.',
      code: `import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier`
    },
    {
      id: 'ktp_load_data',
      type: 'code',
      title: 'Load train/test data',
      summary: 'Read Kaggle CSVs and confirm dataset shapes.',
      code: `import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.shape, test.shape)`
    },
    {
      id: 'ktp_missing',
      type: 'code',
      title: 'Missing value scan',
      summary: 'Rank columns by missing fraction to guide imputation.',
      code: `missing = train.isna().mean().sort_values(ascending=False)
print(missing.head(8))`
    },
    {
      id: 'ktp_target_balance',
      type: 'code',
      title: 'Target balance check',
      summary: 'Compute survival rate to understand class balance.',
      code: `survival_rate = train['Survived'].mean()
print(round(survival_rate, 3))`
    },
    {
      id: 'ktp_features',
      type: 'code',
      title: 'Feature engineering',
      summary: 'Add Title, FamilySize, and IsAlone features for signal.',
      code: `def add_features(df):
    df = df.copy()
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\\.')
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    return df

train = add_features(train)
test = add_features(test)`
    },
    {
      id: 'ktp_split',
      type: 'code',
      title: 'Train/validation split',
      summary: 'Split features and target with stratification.',
      code: `y = train['Survived']
X = train.drop(columns=['Survived'])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)`
    },
    {
      id: 'ktp_columns',
      type: 'code',
      title: 'Column groups',
      summary: 'Define numeric vs categorical features for preprocessing.',
      code: `num_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'IsAlone']
cat_cols = ['Sex', 'Embarked', 'Pclass', 'Title']`
    },
    {
      id: 'ktp_preprocess',
      type: 'code',
      title: 'Preprocessing pipeline',
      summary: 'Impute/scale numeric features and one-hot encode categoricals.',
      code: `from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])`,
      stageId: 'tutorial'
    },
    {
      id: 'ktp_compare',
      type: 'code',
      title: 'Model comparison with CV',
      summary: 'Compare Logistic Regression vs Random Forest using accuracy and ROC-AUC.',
      code: `def cv_score(model, scoring):
    pipe = Pipeline([('prep', preprocessor), ('model', model)])
    scores = cross_val_score(pipe, X, y, cv=5, scoring=scoring)
    return float(scores.mean())

lr_acc = cv_score(LogisticRegression(max_iter=1000), 'accuracy')
lr_auc = cv_score(LogisticRegression(max_iter=1000), 'roc_auc')
rf_acc = cv_score(RandomForestClassifier(n_estimators=300, random_state=42), 'accuracy')
rf_auc = cv_score(RandomForestClassifier(n_estimators=300, random_state=42), 'roc_auc')

print('LR acc:', round(lr_acc, 3), 'AUC:', round(lr_auc, 3))
print('RF acc:', round(rf_acc, 3), 'AUC:', round(rf_auc, 3))`,
      stageId: 'project'
    },
    {
      id: 'ktp_submission',
      type: 'code',
      title: 'Submission export',
      summary: 'Fit the best model on full data and export submission.csv.',
      code: `best_model = RandomForestClassifier(n_estimators=300, random_state=42)
final_pipe = Pipeline([('prep', preprocessor), ('model', best_model)])
final_pipe.fit(X, y)

preds = final_pipe.predict(test)
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': preds.astype(int)
})
submission.to_csv('submission.csv', index=False)
print(submission.head())`,
      stageId: 'assignment'
    }
  ],
  coding: {
    tutorial: {
      title: 'Build Titanic Preprocessing Pipeline',
      description: 'Create a ColumnTransformer pipeline for numeric and categorical features.',
      pseudoCode: '1. Define numeric and categorical columns.\n2. Build numeric and categorical pipelines.\n3. Combine with ColumnTransformer.',
      starterCode: `from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num_cols = ['Age', 'Fare']
cat_cols = ['Sex', 'Embarked', 'Pclass']

# TODO: create preprocessor
preprocessor = None
print(preprocessor is not None)`,
      expectedOutput: 'True',
      solution: `from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num_cols = ['Age', 'Fare']
cat_cols = ['Sex', 'Embarked', 'Pclass']

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])
print(preprocessor is not None)`,
      hints: ['Use two Pipeline objects.', 'Combine with ColumnTransformer list tuples.'],
      testKeywords: ['ColumnTransformer', 'SimpleImputer', 'OneHotEncoder']
    },
    project: {
      title: 'Train and Compare Two Titanic Models',
      description: 'Fit LogisticRegression and RandomForest using a shared preprocessing pipeline and compare CV accuracy.',
      pseudoCode: '1. Build sklearn Pipeline(preprocessor + model).\n2. Run cross_val_score for each model.\n3. Print mean scores.',
      starterCode: `import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# assume X_train, y_train and preprocessor already exist

def model_score(model):
    # TODO
    return 0.0

print('LR:', round(model_score(LogisticRegression(max_iter=1000)), 3))
print('RF:', round(model_score(RandomForestClassifier(n_estimators=300, random_state=42)), 3))`,
      expectedOutput: 'LR: 0.80\nRF: 0.83',
      solution: `import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# assume X_train, y_train and preprocessor already exist

def model_score(model):
    pipe = Pipeline([
        ('prep', preprocessor),
        ('model', model)
    ])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
    return float(np.mean(scores))

print('LR:', round(model_score(LogisticRegression(max_iter=1000)), 3))
print('RF:', round(model_score(RandomForestClassifier(n_estimators=300, random_state=42)), 3))`,
      hints: ['Use Pipeline to avoid leakage.', 'Use cross_val_score(..., cv=5).'],
      testKeywords: ['Pipeline', 'cross_val_score', 'accuracy']
    },
    assignment: {
      title: 'Generate Kaggle Submission File',
      description: 'Train the final model on full train set and export predictions for test set to submission.csv.',
      pseudoCode: '1. Fit final pipeline.\n2. Predict test labels.\n3. Save PassengerId + Survived columns.',
      starterCode: `import pandas as pd

# assume final_model, X_train, y_train, X_test, test_df are available
# TODO: fit model, predict, create submission DataFrame
submission = pd.DataFrame({'PassengerId': [], 'Survived': []})
print(list(submission.columns))`,
      expectedOutput: "['PassengerId', 'Survived']",
      solution: `import pandas as pd

# assume final_model, X_train, y_train, X_test, test_df are available
final_model.fit(X_train, y_train)
preds = final_model.predict(X_test)
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': preds.astype(int)
})
submission.to_csv('submission.csv', index=False)
print(list(submission.columns))`,
      hints: ['Use model.predict on processed pipeline input.', 'Ensure Survived is integer values.'],
      testKeywords: ['to_csv', 'PassengerId', 'Survived']
    }
  },
  interviewQuestions: [
    {
      question: 'How do you prevent overfitting when leaderboard probing on Kaggle?',
      answer: 'Keep a strict local validation strategy, minimize manual leaderboard tuning, and track only changes that improve out-of-fold metrics. Prefer robust cross-validation over repeated test-set probing.',
      companyTags: ['Kaggle', 'Airbnb'],
      difficulty: 'Intermediate'
    }
  ]
};
