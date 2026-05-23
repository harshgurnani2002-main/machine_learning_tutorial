import type { MLModule } from '../../types';

export const kaggleDigitRecognizerProject: MLModule = {
  id: 'kaggle-digit-recognizer-project',
  title: 'Kaggle Project: Digit Recognizer (MNIST)',
  category: 'Kaggle Real-World Projects',
  description: 'Classify handwritten digits using strong baselines and a lightweight neural network.',
  formula: 'softmax(z_i) = exp(z_i) / Σ_j exp(z_j)',
  theory: `### Real-World Problem
Recognize handwritten digits (0-9) from grayscale pixel grids.

### Dataset
- Kaggle: Digit Recognizer (MNIST subset)
- 28x28 grayscale pixels flattened into 784 features

### Notebook Coding Approach
1. Visualize sample digits and confirm label distribution.
2. Normalize pixel values to [0, 1].
3. Train a baseline Logistic Regression model.
4. Compare with KNN for a non-linear baseline.
5. Train a small MLP/CNN for higher accuracy.
6. Evaluate accuracy and submit predictions.

### Real-world extension
Image classification workflows mirror OCR pipelines and quality checks used in document digitization.`,
  interactiveSummary: 'A compact vision pipeline from baseline models to a lightweight neural net.',
  quiz: [
    {
      id: 'kdrp_q1',
      question: 'Why normalize pixel values to the [0, 1] range?',
      options: [
        'It stabilizes optimization and makes gradients more consistent.',
        'It doubles the training data size.',
        'It removes all noise from images.',
        'It converts regression into classification.'
      ],
      correctAnswer: 'It stabilizes optimization and makes gradients more consistent.',
      explanation: 'Scaling pixels keeps gradients in a reasonable range and speeds up training.'
    },
    {
      id: 'kdrp_q2',
      question: 'Which metric is most common for MNIST-style digit recognition?',
      options: ['Accuracy', 'AUC', 'MAE', 'Silhouette Score'],
      correctAnswer: 'Accuracy',
      explanation: 'The competition evaluates the fraction of correct digit predictions.'
    },
    {
      id: 'kdrp_q3',
      question: 'Why keep a simple baseline before CNN/MLP?',
      options: [
        'It provides a sanity check for data and labels.',
        'Because deep networks cannot handle images.',
        'To avoid any hyperparameter tuning.',
        'To remove the need for validation.'
      ],
      correctAnswer: 'It provides a sanity check for data and labels.',
      explanation: 'Baselines verify preprocessing and highlight easy wins before complex models.'
    }
  ],
  notebookCells: [
    {
      id: 'kdrp_overview',
      type: 'markdown',
      title: 'Notebook overview',
      summary: 'Goal: classify handwritten digits (0-9) with strong baselines and a small neural net.'
    },
    {
      id: 'kdrp_imports',
      type: 'code',
      title: 'Imports',
      summary: 'Load libraries for image classification.',
      code: `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier`
    },
    {
      id: 'kdrp_load',
      type: 'code',
      title: 'Load data',
      summary: 'Read Kaggle train/test CSVs.',
      code: `train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.shape, test.shape)`
    },
    {
      id: 'kdrp_preview',
      type: 'code',
      title: 'Preview a digit',
      summary: 'Reshape a row into a 28x28 image for quick QA.',
      code: `sample = train.iloc[0, 1:].values.reshape(28, 28)
plt.imshow(sample, cmap='gray')`
    },
    {
      id: 'kdrp_scale',
      type: 'code',
      title: 'Normalize pixels',
      summary: 'Scale pixel values to [0, 1] for stable training.',
      code: `y = train['label'].values
X = train.drop(columns=['label']).values

X = X / 255.0
test_scaled = test.values / 255.0`,
      stageId: 'tutorial'
    },
    {
      id: 'kdrp_split',
      type: 'code',
      title: 'Train/validation split',
      summary: 'Split data for quick evaluation.',
      code: `X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)`
    },
    {
      id: 'kdrp_logit',
      type: 'code',
      title: 'Logistic regression baseline',
      summary: 'Train a fast linear classifier as a sanity check.',
      code: `logit = LogisticRegression(max_iter=500, n_jobs=-1)
logit.fit(X_train, y_train)
acc = logit.score(X_val, y_val)
print(round(acc, 3))`,
      stageId: 'project'
    },
    {
      id: 'kdrp_knn',
      type: 'code',
      title: 'KNN comparison',
      summary: 'Non-linear KNN baseline for accuracy boost.',
      code: `knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_acc = knn.score(X_val, y_val)
print(round(knn_acc, 3))`
    },
    {
      id: 'kdrp_mlp',
      type: 'code',
      title: 'MLP classifier',
      summary: 'Train a small neural network for a stronger baseline.',
      code: `mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=20, random_state=42)
mlp.fit(X_train, y_train)
mlp_acc = mlp.score(X_val, y_val)
print(round(mlp_acc, 3))`,
      stageId: 'assignment'
    },
    {
      id: 'kdrp_submission',
      type: 'code',
      title: 'Submission export',
      summary: 'Train on full data and submit predictions.',
      code: `final_model = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=20, random_state=42)
final_model.fit(X, y)
test_preds = final_model.predict(test_scaled)

submission = pd.DataFrame({'ImageId': np.arange(1, len(test_preds) + 1), 'Label': test_preds})
submission.to_csv('submission.csv', index=False)
print(submission.head())`
    }
  ],
  coding: {
    tutorial: {
      title: 'Normalize Pixel Values',
      description: 'Scale grayscale pixel values into the [0, 1] range.',
      pseudoCode: '1. Convert pixel array to float.\n2. Divide by 255.\n3. Confirm max <= 1.',
      starterCode: `import numpy as np

X = np.array([[0, 128, 255]], dtype=float)
# TODO
X_scaled = X
print(X_scaled.max() <= 1.0)`,
      expectedOutput: 'True',
      solution: `import numpy as np

X = np.array([[0, 128, 255]], dtype=float)
X_scaled = X / 255.0
print(X_scaled.max() <= 1.0)`,
      hints: ['Use float division to avoid integer truncation.'],
      testKeywords: ['255', 'float']
    },
    project: {
      title: 'Logistic Regression Baseline',
      description: 'Train a Logistic Regression classifier and report validation accuracy.',
      pseudoCode: '1. Fit LogisticRegression.\n2. Score on validation set.\n3. Print rounded accuracy.',
      starterCode: `from sklearn.linear_model import LogisticRegression

# assume X_train, y_train, X_val, y_val
model = LogisticRegression(max_iter=500)
# TODO
acc = 0.0
print(round(acc, 2))`,
      expectedOutput: '0.92',
      solution: `from sklearn.linear_model import LogisticRegression

# assume X_train, y_train, X_val, y_val
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
acc = model.score(X_val, y_val)
print(round(acc, 2))`,
      hints: ['Use model.score for accuracy.'],
      testKeywords: ['LogisticRegression', 'score']
    },
    assignment: {
      title: 'Train a Small MLP',
      description: 'Fit a lightweight neural network and compare accuracy.',
      pseudoCode: '1. Define MLPClassifier.\n2. Fit on training set.\n3. Score on validation set.',
      starterCode: `from sklearn.neural_network import MLPClassifier

# assume X_train, y_train, X_val, y_val
mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=20, random_state=42)
# TODO
acc = 0.0
print(acc >= 0.0)`,
      expectedOutput: 'True',
      solution: `from sklearn.neural_network import MLPClassifier

# assume X_train, y_train, X_val, y_val
mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=20, random_state=42)
mlp.fit(X_train, y_train)
acc = mlp.score(X_val, y_val)
print(acc >= 0.0)`,
      hints: ['Keep epochs low for quick iterations.'],
      testKeywords: ['MLPClassifier', 'fit', 'score']
    }
  },
  interviewQuestions: [
    {
      question: 'What is the biggest risk when moving from a baseline to a neural model on MNIST?',
      answer: 'Overfitting from over-parameterized networks without proper validation or regularization.',
      companyTags: ['Google', 'Microsoft'],
      difficulty: 'Intermediate'
    }
  ]
};
