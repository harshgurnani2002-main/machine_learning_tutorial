import type { MLModule } from '../../types';

export const introductionToMl: MLModule = {
  id: 'introduction-to-ml',
  title: 'Introduction to Machine Learning',
  category: 'Foundations & Math',
  description: 'Learn the core concepts, types of learning, the ML lifecycle, and how to choose the right algorithm for any problem.',
  formula: 'y = f(x) + \\epsilon',
  interactiveSummary: 'This module introduces the key paradigms of machine learning. There is no complex simulator — your goal is to internalize the mental model of how machines learn from data before diving into algorithms.',
  simulatorId: 'intro-simulator',
  theory: `### Introduction to Machine Learning

#### What is Machine Learning?
Machine Learning (ML) is a branch of Artificial Intelligence (AI) that enables computers to learn from data and improve their performance on tasks without being explicitly programmed for each step.

Think of it this way: instead of writing 10,000 rules to detect spam emails, you show the algorithm 10,000 labeled examples of spam and non-spam, and it derives the rules itself.

The relationship between AI, ML, and Deep Learning:
- **Artificial Intelligence**: The broadest field. Any technique that enables machines to mimic human intelligence.
- **Machine Learning**: A subset of AI. Systems that learn from data.
- **Deep Learning**: A subset of ML. Systems that use multi-layered neural networks.

#### Why Do We Need Machine Learning?
Traditional software requires humans to enumerate all rules explicitly. This breaks down when:
1. The rules are too numerous or complex to write manually (face recognition has millions of pixel combinations).
2. The rules change over time (fraud patterns evolve constantly).
3. The data is too high-dimensional for human intuition.

ML allows the machine to extract patterns from data at a scale and speed no human can match.

#### The Three Paradigms of Machine Learning

**1. Supervised Learning**
The algorithm learns from a labeled dataset — pairs of inputs $x$ and their correct outputs $y$.
The goal is to learn a mapping function $f$ such that $f(x) \\approx y$.
- Examples: Predicting house prices (regression), classifying emails as spam (classification).
- Algorithms: Linear Regression, Decision Trees, SVM, Neural Networks.

**2. Unsupervised Learning**
The algorithm is given only input data $X$ with no labels. It must discover the hidden structure on its own.
- Examples: Grouping customers by purchasing behavior (clustering), compressing images (dimensionality reduction).
- Algorithms: K-Means, DBSCAN, PCA, Autoencoders.

**3. Reinforcement Learning**
An agent learns by interacting with an environment. It receives rewards for good actions and penalties for bad ones. The goal is to maximize cumulative reward.
- Examples: Game-playing AI (AlphaGo), robotic control, recommendation systems.
- Key concept: Exploration vs Exploitation trade-off.

#### The Machine Learning Lifecycle
A real ML project is never just about training a model. It follows a structured pipeline:

1. **Problem Definition**: What are we predicting? What is the cost of a wrong prediction?
2. **Data Collection**: Gathering raw data from databases, APIs, sensors, or web scraping.
3. **Exploratory Data Analysis (EDA)**: Understanding the data — distributions, missing values, correlations.
4. **Data Preprocessing**: Cleaning, handling missing values, encoding categoricals, normalizing features.
5. **Feature Engineering**: Creating new features that capture domain knowledge.
6. **Model Selection**: Choosing the right algorithm class based on the problem type.
7. **Training**: Fitting the model to the training data.
8. **Evaluation**: Measuring performance on a held-out test set.
9. **Hyperparameter Tuning**: Optimizing the model's configuration.
10. **Deployment**: Serving the model in a production environment (API, batch job).
11. **Monitoring**: Tracking model performance over time as data distribution shifts.

#### Key Terminology Every ML Practitioner Must Know
- **Feature** ($x$): An input variable used to make a prediction. Also called a "predictor" or "independent variable".
- **Label / Target** ($y$): The output variable we want to predict. Also called the "dependent variable".
- **Training Set**: The data used to fit the model.
- **Validation Set**: Data used to tune hyperparameters without contaminating the test set.
- **Test Set**: Data used to report the final, unbiased model performance.
- **Overfitting**: The model performs well on training data but poorly on unseen data — it has memorized the training set.
- **Underfitting**: The model is too simple to capture the underlying pattern in either training or test data.
- **Hyperparameter**: A configuration choice set before training begins (e.g., number of trees, learning rate). Not learned from data.
- **Parameter**: A value learned by the model during training (e.g., weights in a neural network).
- **Generalization**: The ability of a model to perform well on data it has never seen.

#### The Math Behind Learning
At its core, ML is an optimization problem. We want to find parameters $\\theta$ that minimize a **loss function** $L$:

$$\\theta^* = \\arg\\min_{\\theta} L(y, f_{\\theta}(x))$$

For regression, the most common loss is **Mean Squared Error (MSE)**:

$$L_{MSE} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2$$

For classification, we commonly use **Cross-Entropy Loss**:

$$L_{CE} = -\\frac{1}{n} \\sum_{i=1}^{n} [y_i \\log(\\hat{p}_i) + (1-y_i) \\log(1 - \\hat{p}_i)]$$

The optimization is typically performed using **Gradient Descent** — repeatedly nudging parameters in the direction that reduces the loss.

#### How to Choose the Right Algorithm
There is no single "best" algorithm (the No Free Lunch theorem). Your choice depends on:
- **Problem type**: Classification, Regression, Clustering, Generation?
- **Data size**: Thousands of rows (classical ML) vs millions (deep learning)?
- **Interpretability**: Does a doctor/judge need to explain the decision? Use Decision Trees or Logistic Regression.
- **Training time**: Real-time requirements favor simpler models.
- **Feature types**: Text → NLP models. Images → CNNs. Tabular → Gradient Boosting.

A reliable general heuristic:
- Start with a simple **Logistic Regression** or **Linear Regression** as your baseline.
- Try **Gradient Boosting (XGBoost)** for structured/tabular data — it wins most Kaggle competitions.
- Use **Deep Learning** when you have massive data and the problem is images, text, or audio.
`,
  quiz: [
    {
      id: 'intro_q1',
      question: 'What is the key difference between traditional programming and machine learning?',
      options: [
        'ML is faster to execute than traditional programs',
        'Traditional programs require labeled data; ML does not',
        'ML derives rules from data; traditional programming requires rules to be written explicitly',
        'ML cannot handle numerical data'
      ],
      correctAnswer: 'ML derives rules from data; traditional programming requires rules to be written explicitly',
      explanation: 'In traditional programming, a human writes the rules. In ML, the algorithm learns the rules from labeled (or unlabeled) data examples.'
    },
    {
      id: 'intro_q2',
      question: 'Which type of machine learning is used when you have input data but NO corresponding labels?',
      options: [
        'Supervised Learning',
        'Reinforcement Learning',
        'Unsupervised Learning',
        'Semi-supervised Learning'
      ],
      correctAnswer: 'Unsupervised Learning',
      explanation: 'Unsupervised learning finds hidden patterns in data without any labeled output. Clustering (K-Means) and dimensionality reduction (PCA) are common examples.'
    },
    {
      id: 'intro_q3',
      question: 'A model scores 99% accuracy on training data but only 60% on the test set. What is the most likely problem?',
      options: [
        'Underfitting',
        'Overfitting',
        'Data leakage',
        'The model is too simple'
      ],
      correctAnswer: 'Overfitting',
      explanation: 'Overfitting means the model memorized the training data, including its noise, and failed to generalize to unseen data. The huge gap between train and test accuracy is the telltale sign.'
    },
    {
      id: 'intro_q4',
      question: 'What is a "hyperparameter" in machine learning?',
      options: [
        'A weight learned by the model during gradient descent',
        'A configuration set before training, not learned from data',
        'The final output prediction of the model',
        'A type of regularization technique'
      ],
      correctAnswer: 'A configuration set before training, not learned from data',
      explanation: 'Hyperparameters (like learning rate, number of trees, max depth) are set by the practitioner before training begins. Parameters (like weights) are learned from data during training.'
    },
    {
      id: 'intro_q5',
      question: 'What does the "No Free Lunch" theorem state in machine learning?',
      options: [
        'All ML algorithms have the same computational cost',
        'No single algorithm performs best on all possible problems',
        'Deep learning always outperforms classical ML',
        'You always need labeled data to train a model'
      ],
      correctAnswer: 'No single algorithm performs best on all possible problems',
      explanation: 'The No Free Lunch theorem proves that averaged across all possible problems, all optimization and learning algorithms perform equally. This is why choosing the right algorithm for your specific problem matters.'
    },
    {
      id: 'intro_q6',
      question: 'In the ML lifecycle, what is the purpose of a VALIDATION set (distinct from the test set)?',
      options: [
        'To train the model on more data',
        'To report the final, unbiased model performance',
        'To tune hyperparameters without contaminating the final test evaluation',
        'To generate synthetic data'
      ],
      correctAnswer: 'To tune hyperparameters without contaminating the final test evaluation',
      explanation: 'If you tune hyperparameters on the test set, you are effectively training on it and your reported performance will be optimistic. A separate validation set prevents this bias.'
    }
  ],
  coding: {
    tutorial: {
      title: 'Your First Scikit-learn Pipeline',
      description: 'Build a complete end-to-end classification pipeline using scikit-learn — from loading data to evaluating accuracy. This is the standard boilerplate for any supervised ML project.',
      pseudoCode: `1. Load the Iris dataset
2. Split into train (80%) and test (20%) sets
3. Build a pipeline: StandardScaler → LogisticRegression
4. Fit on training data
5. Evaluate on test data with accuracy score`,
      starterCode: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# 1. Load data
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. TODO: Build a Pipeline with StandardScaler and LogisticRegression
# pipeline = Pipeline([...])

# 4. TODO: Fit the pipeline on training data

# 5. TODO: Predict on test data and print accuracy
print("Accuracy:", 0.0)`,
      expectedOutput: `Accuracy: 1.0`,
      solution: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=200))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))`,
      hints: [
        'Pipeline takes a list of (name, estimator) tuples: Pipeline([(\"scaler\", StandardScaler()), ...])',
        'Call pipeline.fit(X_train, y_train) to train',
        'Use pipeline.predict(X_test) to generate predictions',
        'Pass y_test and y_pred to accuracy_score()'
      ],
      testKeywords: ['Pipeline', 'StandardScaler', 'LogisticRegression', 'accuracy_score', 'fit']
    },
    project: {
      title: 'EDA on the Iris Dataset',
      description: 'Perform a complete Exploratory Data Analysis on the Iris dataset. Understanding your data before modeling is the most critical step in the ML lifecycle. Compute statistics, check for missing values, and print feature names.',
      pseudoCode: `1. Load Iris into a pandas DataFrame
2. Print shape and first 5 rows
3. Print summary statistics with describe()
4. Check for missing values with isnull().sum()
5. Print the count of each target class`,
      starterCode: `import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

# TODO: Create a DataFrame from iris.data with iris.feature_names as columns
# df = pd.DataFrame(...)

# TODO: Add the target column
# df['target'] = ...

# TODO: Print the shape of the dataframe
# print("Shape:", ...)

# TODO: Print the first 5 rows
# print(df.head())

# TODO: Print summary statistics
# print(df.describe())

# TODO: Print missing value counts
# print(df.isnull().sum())

# TODO: Print value counts of the target column
# print(df['target'].value_counts())`,
      expectedOutput: `Shape: (150, 5)`,
      solution: `import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("Shape:", df.shape)
print(df.head())
print(df.describe())
print("Missing values:")
print(df.isnull().sum())
print("\\nClass distribution:")
print(df['target'].value_counts())`,
      hints: [
        'pd.DataFrame(iris.data, columns=iris.feature_names) creates the DataFrame',
        'df[\'target\'] = iris.target adds the labels',
        'df.isnull().sum() counts missing values per column',
        'df[\'target\'].value_counts() shows how many samples per class'
      ],
      testKeywords: ['DataFrame', 'describe', 'isnull', 'value_counts', 'shape']
    },
    assignment: {
      title: 'Full Classification Pipeline with Evaluation',
      description: 'Build a complete classification pipeline on the Wine dataset. Train a Random Forest, evaluate using accuracy and a detailed classification report, and identify the most important features. This mirrors a real-world ML workflow.',
      pseudoCode: `1. Load Wine dataset
2. Split 80/20 train/test (stratified)
3. Build Pipeline: StandardScaler → RandomForestClassifier(n_estimators=100)
4. Fit and predict
5. Print accuracy, classification_report
6. Print top 3 most important features`,
      starterCode: `from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

wine = load_wine()
X, y = wine.data, wine.target
feature_names = wine.feature_names

# TODO: Split with stratify=y and random_state=42

# TODO: Build Pipeline with StandardScaler and RandomForestClassifier(n_estimators=100, random_state=42)

# TODO: Fit and predict

# TODO: Print accuracy and classification_report

# TODO: Extract feature importances from the RF model inside the pipeline
# and print top 3 features`,
      expectedOutput: `Accuracy:`,
      solution: `from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

wine = load_wine()
X, y = wine.data, wine.target
feature_names = wine.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=wine.target_names))

importances = pipeline.named_steps['rf'].feature_importances_
top3_idx = np.argsort(importances)[::-1][:3]
print("Top 3 features:")
for i in top3_idx:
    print(f"  {feature_names[i]}: {importances[i]:.4f}")`,
      hints: [
        'train_test_split(..., stratify=y) ensures balanced class split',
        'Access the RF inside the pipeline with pipeline.named_steps[\'rf\']',
        'np.argsort(importances)[::-1][:3] gives indices of top 3 features',
        'classification_report gives precision, recall, F1 per class'
      ],
      testKeywords: ['RandomForestClassifier', 'classification_report', 'feature_importances_', 'stratify', 'Pipeline']
    }
  },
  interviewQuestions: [
    {
      question: 'Explain the bias-variance tradeoff in your own words.',
      answer: 'Bias is the error from wrong assumptions in the learning algorithm (e.g., assuming linear data when it is not). Variance is the error from sensitivity to small fluctuations in the training data. High bias → underfitting (model too simple). High variance → overfitting (model too complex). The tradeoff means reducing one tends to increase the other. Ensembles like Random Forests use many high-variance trees and average them to dramatically reduce variance while keeping bias low.',
      companyTags: ['Google', 'Meta', 'Amazon'],
      difficulty: 'Medium'
    },
    {
      question: 'When would you choose a simpler model over a more complex one even if the complex model has higher accuracy?',
      answer: 'When interpretability is required (medical diagnosis, legal decisions, financial risk). When the dataset is small and a complex model would overfit. When inference speed is critical (real-time systems). When you need to debug and trust the model\'s decisions. When the accuracy difference is marginal — the added complexity is rarely worth it for a 0.1% gain.',
      companyTags: ['McKinsey', 'JPMorgan', 'Stripe'],
      difficulty: 'Medium'
    },
    {
      question: 'What is data leakage, and how does it happen?',
      answer: 'Data leakage occurs when information from outside the training dataset is used to build the model, causing unrealistically high performance that collapses in production. Common sources: (1) Fitting a scaler or imputer on the entire dataset before splitting, so test data statistics pollute the training distribution. (2) Including a feature that is a proxy for the label (e.g., a timestamp that perfectly predicts churn because all churners in a certain period were added). (3) Target encoding without proper cross-validation folds. Prevention: always fit preprocessing only on training data, use scikit-learn Pipelines.',
      companyTags: ['Netflix', 'Airbnb', 'Uber'],
      difficulty: 'Hard'
    }
  ]
};
