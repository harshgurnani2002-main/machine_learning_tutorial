import type { MLModule } from '../../types';

export const kaggleDisasterTweetsProject: MLModule = {
  id: 'kaggle-disaster-tweets-project',
  title: 'Kaggle Project: Disaster Tweets Classification',
  category: 'Kaggle Real-World Projects',
  description: 'Classify tweets as disaster-related using text preprocessing, TF-IDF, and linear models.',
  formula: 'F1 = 2 * (precision * recall) / (precision + recall)',
  theory: `### Real-World Problem
Detect disaster-related tweets to aid crisis response teams.

### Dataset
- Kaggle: Natural Language Processing with Disaster Tweets
- Text + keyword + location fields with binary labels

### Notebook Coding Approach
1. Clean text (lowercase, strip URLs/handles).
2. Vectorize with TF-IDF.
3. Train Logistic Regression baseline.
4. Compare with Linear SVM or Naive Bayes.
5. Evaluate with F1 score.
6. Generate submission predictions.

### Real-world extension
NLP pipelines often need quick retraining, so lightweight linear models remain highly competitive.`,
  interactiveSummary: 'A compact NLP workflow using TF-IDF + linear models with F1 evaluation.',
  quiz: [
    {
      id: 'kdtp_q1',
      question: 'Why is F1 used instead of accuracy for disaster tweet classification?',
      options: [
        'It balances precision and recall when class balance is uneven.',
        'It is faster to compute than accuracy.',
        'It ignores false positives.',
        'It is required only for neural networks.'
      ],
      correctAnswer: 'It balances precision and recall when class balance is uneven.',
      explanation: 'F1 is robust when positive/negative classes are imbalanced.'
    },
    {
      id: 'kdtp_q2',
      question: 'What does TF-IDF emphasize?',
      options: [
        'Words that are frequent in a document but rare across the corpus.',
        'Only the rarest words in the corpus.',
        'Only the most frequent words overall.',
        'Word order and grammar.'
      ],
      correctAnswer: 'Words that are frequent in a document but rare across the corpus.',
      explanation: 'TF-IDF downweights common words and highlights document-specific terms.'
    },
    {
      id: 'kdtp_q3',
      question: 'Which model is a strong baseline for sparse text features?',
      options: ['Logistic Regression', 'KMeans', 'KNN on raw text', 'Decision Tree without vectorization'],
      correctAnswer: 'Logistic Regression',
      explanation: 'Linear models perform very well on high-dimensional sparse TF-IDF features.'
    }
  ],
  notebookCells: [
    {
      id: 'kdtp_overview',
      type: 'markdown',
      title: 'Notebook overview',
      summary: 'Goal: classify tweets as disaster-related using TF-IDF and linear models with F1 evaluation.'
    },
    {
      id: 'kdtp_imports',
      type: 'code',
      title: 'Imports',
      summary: 'Load libraries for text processing and modeling.',
      code: `import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score`
    },
    {
      id: 'kdtp_load',
      type: 'code',
      title: 'Load data',
      summary: 'Read Kaggle train/test CSVs.',
      code: `train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.shape, test.shape)`
    },
    {
      id: 'kdtp_clean',
      type: 'code',
      title: 'Text cleaning',
      summary: 'Normalize case and remove URLs, mentions, and hashtags.',
      code: `def clean(text):
    text = text.lower()
    text = re.sub(r'http\\S+|@\\w+|#\\w+', '', text)
    text = re.sub(r'[^a-z0-9\\s]', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

train['clean'] = train['text'].astype(str).apply(clean)
test['clean'] = test['text'].astype(str).apply(clean)`
    },
    {
      id: 'kdtp_split',
      type: 'code',
      title: 'Train/validation split',
      summary: 'Split data for evaluation.',
      code: `y = train['target'].values
X_train_text, X_val_text, y_train, y_val = train_test_split(
    train['clean'].values, y, test_size=0.2, random_state=42, stratify=y
)`
    },
    {
      id: 'kdtp_tfidf',
      type: 'code',
      title: 'TF-IDF vectorization',
      summary: 'Convert raw text into sparse TF-IDF features.',
      code: `vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)
X_test = vectorizer.transform(test['clean'].values)`,
      stageId: 'tutorial'
    },
    {
      id: 'kdtp_logit',
      type: 'code',
      title: 'Logistic regression baseline',
      summary: 'Train a fast linear classifier and evaluate F1.',
      code: `logit = LogisticRegression(max_iter=1000)
logit.fit(X_train, y_train)
pred = logit.predict(X_val)
f1 = f1_score(y_val, pred)
print(round(f1, 3))`,
      stageId: 'project'
    },
    {
      id: 'kdtp_svm',
      type: 'code',
      title: 'Linear SVM comparison',
      summary: 'Compare LinearSVC for stronger margins.',
      code: `svm = LinearSVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_val)
svm_f1 = f1_score(y_val, svm_pred)
print(round(svm_f1, 3))`,
      stageId: 'assignment'
    },
    {
      id: 'kdtp_submission',
      type: 'code',
      title: 'Submission export',
      summary: 'Train on full data and submit predictions.',
      code: `final_vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
X_full = final_vectorizer.fit_transform(train['clean'].values)
X_test_full = final_vectorizer.transform(test['clean'].values)

final_model = LinearSVC()
final_model.fit(X_full, y)
test_pred = final_model.predict(X_test_full)

submission = pd.DataFrame({'id': test['id'], 'target': test_pred})
submission.to_csv('submission.csv', index=False)
print(submission.head())`
    }
  ],
  coding: {
    tutorial: {
      title: 'TF-IDF Vectorization',
      description: 'Create a TF-IDF matrix from raw tweets.',
      pseudoCode: '1. Instantiate TfidfVectorizer.\n2. Fit and transform text.\n3. Print shape.',
      starterCode: `from sklearn.feature_extraction.text import TfidfVectorizer

texts = ["storm coming", "loving the sunshine"]
# TODO
vectorizer = None
X = None
print(X.shape[0] if X is not None else 0)`,
      expectedOutput: '2',
      solution: `from sklearn.feature_extraction.text import TfidfVectorizer

texts = ["storm coming", "loving the sunshine"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
print(X.shape[0])`,
      hints: ['Use fit_transform to build the vocabulary.'],
      testKeywords: ['TfidfVectorizer', 'fit_transform']
    },
    project: {
      title: 'Logistic Regression with F1',
      description: 'Train a logistic baseline and compute F1 on validation data.',
      pseudoCode: '1. Fit LogisticRegression.\n2. Predict labels.\n3. Compute F1.',
      starterCode: `from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# assume X_train, y_train, X_val, y_val
model = LogisticRegression(max_iter=1000)
# TODO
f1 = 0.0
print(round(f1, 3))`,
      expectedOutput: '0.78',
      solution: `from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# assume X_train, y_train, X_val, y_val
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
pred = model.predict(X_val)
f1 = f1_score(y_val, pred)
print(round(f1, 3))`,
      hints: ['Use predicted class labels, not probabilities, for F1.'],
      testKeywords: ['f1_score', 'predict']
    },
    assignment: {
      title: 'Linear SVM Benchmark',
      description: 'Train a LinearSVC model and compare F1.',
      pseudoCode: '1. Fit LinearSVC.\n2. Predict on validation.\n3. Compute F1.',
      starterCode: `from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

# assume X_train, y_train, X_val, y_val
svm = LinearSVC()
# TODO
f1 = 0.0
print(f1 >= 0.0)`,
      expectedOutput: 'True',
      solution: `from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

# assume X_train, y_train, X_val, y_val
svm = LinearSVC()
svm.fit(X_train, y_train)
pred = svm.predict(X_val)
f1 = f1_score(y_val, pred)
print(f1 >= 0.0)`,
      hints: ['LinearSVC is fast on sparse data.'],
      testKeywords: ['LinearSVC', 'predict', 'f1_score']
    }
  },
  interviewQuestions: [
    {
      question: 'What is the main trade-off between Logistic Regression and Linear SVM in text classification?',
      answer: 'SVM can yield stronger margins and sometimes higher F1, while Logistic Regression provides calibrated probabilities for ranking and threshold tuning.',
      companyTags: ['Twitter', 'Reddit'],
      difficulty: 'Intermediate'
    }
  ]
};
