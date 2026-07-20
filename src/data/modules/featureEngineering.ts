import type { MLModule } from '../../types';

export const featureEngineering: MLModule = {
  id: 'feature-engineering',
  title: 'Feature Engineering',
  category: 'Foundations & Math',
  description: 'Clean, transform, encode, and select features to maximize machine learning model performance.',
  formula: 'x_{\\text{scaled}} = \\frac{x - x_{\\min}}{x_{\\max} - x_{\\min}}',
  theory: `### Feature Engineering: The Kaggle & Industry Secret Weapon

#### What is it?
Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work. If machine learning is the engine, features are the fuel. No matter how complex your model is, it will perform poorly if the features do not represent the underlying structure of the problem.
Feature engineering includes cleaning data, handling missing values, scaling numerical attributes, encoding categorical variables, creating interaction terms, and selecting the most predictive features.

#### Why do we need it?
1. **Model Performance:** Better features allow you to use simpler models that are faster and require less tuning.
2. **Data Suitability:** Algorithms like Linear Regression or SVM cannot natively process text labels (categorical data) or handle missing values; data must be converted into clean, numerical formats.
3. **Information Extraction:** Creating compound features (e.g., density = mass / volume) extracts explicit relationships that tree-based or linear models might struggle to learn on their own.
4. **Preventing Leakage:** Standardizing features properly ensures that test set information does not leak into training, preserving model validation integrity.

#### How does it work?
Feature engineering is an iterative pipeline:
1. **Imputation:** Filling missing values using statistical measures (mean, median, mode) or models (K-NN, iterative imputer).
2. **Scaling:** Transforming continuous features to have similar ranges. Algorithms utilizing distance metrics (K-Means, SVM, K-NN) or gradient optimization (Neural Networks) are highly sensitive to scale.
   - **MinMax Scaling:** Compresses features into [0, 1] range.
   - **Standardization:** Transforms features to have $\\mu=0$ and $\\sigma=1$.
3. **Categorical Encoding:** Converting categories into numbers.
   - **One-Hot Encoding:** Creates binary columns for each category. (Best for low cardinality nominal features).
   - **Ordinal Encoding:** Assigns integers to ordered categories (e.g., Low=0, Medium=1, High=2).
   - **Target Encoding:** Replaces category labels with the mean target value of that category. (Powerful for Kaggle but prone to target leakage/overfitting).
4. **Feature Selection:** Removing redundant or noisy features to reduce dimensionality using variance thresholds, statistical tests (SelectKBest), or tree-based feature importances (SelectFromModel).

#### The Math Behind It

**MinMax Scaling (Normalization):**
$$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

**Standardization (Z-score Normalization):**
$$x_{std} = \frac{x - \\mu}{\\sigma}$$

**Target Encoding with Smoothing:**
To prevent overfitting on low-frequency categories, we smooth the category mean with the global target mean:
$$S_i = \\lambda(n_i) \\cdot \bar{y}_i + (1 - \\lambda(n_i)) \\cdot \bar{y}_{global}$$
where $n_i$ is count of category $i$, $\bar{y}_i$ is category mean, and $\\lambda(n_i) \\in [0, 1]$ is a weight function based on frequency.

#### Worked Example: Feature Scaling
Suppose we have a feature vector of housing sizes in sq ft: $X = [1000, 1500, 3000]$.
- $x_{min} = 1000$, $x_{max} = 3000$.
- For $x = 1500$, the MinMax scaled value is:
  $$x_{scaled} = \frac{1500 - 1000}{3000 - 1000} = \frac{500}{2000} = 0.25$$

#### Common Pitfalls
1. **Data Leakage:** Scaling or computing imputation statistics on the *entire* dataset before splitting into train/test. Test metrics must be computed strictly on training folds and applied to test.
2. **Dummy Variable Trap:** Failing to drop one column when one-hot encoding for linear models, leading to perfect multicollinearity.
3. **Overfitting in Target Encoding:** Standard target encoding leaks the target label directly into train folds. K-Fold target encoding must be applied.

#### When to Use vs Not Use
**Use when:**
- Working with tabular datasets.
- Distance-based models (SVM, KNN, KMeans) or gradient-based models are used.
- Improving baseline model scores in competitive data science (Kaggle).

**Not Use when:**
- Using pure Deep Learning on raw pixels or raw waveforms where end-to-end representation learning is preferred.
`,
  interactiveSummary: 'This interactive simulator demonstrates the effects of feature scaling. Plot points and observe how they compress under MinMax scaling vs. Standardization. Notice how extreme outliers distort standard scaling methods, showing the need for robust transformers.',
  simulatorId: 'feature-scaling',
  quiz: [
    {
      id: 'fe_q1',
      question: 'What is data leakage in feature engineering?',
      options: [
        'Allowing test set information to influence train set feature representations.',
        'Having missing values inside the database.',
        'Encrypting features incorrectly during export.',
        'Using too many features in a simple linear model.'
      ],
      correctAnswer: 'Allowing test set information to influence train set feature representations.',
      explanation: 'Data leakage happens when information from outside the training dataset is used to create features. For example, fitting a StandardScaler on the entire dataset instead of train-only leaks test statistics.'
    },
    {
      id: 'fe_q2',
      question: 'Which encoding method replaces a category with the average target label value for that category?',
      options: [
        'Target Encoding',
        'One-Hot Encoding',
        'Ordinal Encoding',
        'Binary Encoding'
      ],
      correctAnswer: 'Target Encoding',
      explanation: 'Target encoding maps categories directly to their mean target values. It is highly effective but highly prone to target leakage.'
    },
    {
      id: 'fe_q3',
      question: 'Why does K-Means clustering require feature scaling?',
      options: [
        'Because it computes Euclidean distances, and unscaled features with larger ranges will dominate calculations.',
        'To speed up tree splits.',
        'To convert categorical features into numerical ones.',
        'To make the data linearly separable.'
      ],
      correctAnswer: 'Because it computes Euclidean distances, and unscaled features with larger ranges will dominate calculations.',
      explanation: 'K-Means partitions data strictly based on geometric distance. Features with large scales (e.g. salary in $10k+) will dominate features with small scales (e.g. age in decades).'
    },
    {
      id: 'fe_q4',
      question: 'How do you handle the dummy variable trap in linear models?',
      options: [
        'Drop one of the one-hot encoded dummy columns (n-1 instead of n categories).',
        'Use ordinal encoding instead.',
        'Standardize all categorical features.',
        'Add a high L1 regularization penalty.'
      ],
      correctAnswer: 'Drop one of the one-hot encoded dummy columns (n-1 instead of n categories).',
      explanation: 'If a categorical variable has n categories, one-hot encoding all n categories creates a set of features that sum to 1, causing perfect multicollinearity with the intercept. Dropping one column resolves this.'
    },
    {
      id: 'fe_q5',
      question: 'What is the purpose of SelectFromModel in scikit-learn?',
      options: [
        'To select features based on their importances estimated by an underlying estimator (like a Random Forest).',
        'To pick which algorithm to run on the dataset.',
        'To tune the model\'s hyperparameters automatically.',
        'To create interaction terms.'
      ],
      correctAnswer: 'To select features based on their importances estimated by an underlying estimator (like a Random Forest).',
      explanation: 'SelectFromModel is a meta-transformer that selects features whose weight or importance exceeds a given threshold, as computed by a fitted model.'
    }
  ],
  coding: {
    tutorial: {
      title: 'Vectorized MinMax Scaling',
      description: 'Implement a vectorized NumPy function to perform MinMax scaling, squashing coordinates to the [0, 1] range.',
      pseudoCode: '1. Accept a numpy array X\n2. Compute the minimum and maximum values along axis 0\n3. Apply formula: (X - min) / (max - min)\n4. Return scaled array',
      starterCode: `import numpy as np

def minmax_scaler(X):
    # TODO: Implement MinMax scaling
    # Formula: (X - X_min) / (X_max - X_min)
    return X

X_val = np.array([[10.0, 100.0],
                  [20.0, 300.0],
                  [30.0, 200.0]])
print("Scaled:\n", np.round(minmax_scaler(X_val), 2))`,
      expectedOutput: 'Scaled:\n [[0.  0. ]\n [0.5 1. ]\n [1.  0.5]]',
      solution: `import numpy as np

def minmax_scaler(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min)

X_val = np.array([[10.0, 100.0],
                  [20.0, 300.0],
                  [30.0, 200.0]])
print("Scaled:\n", np.round(minmax_scaler(X_val), 2))`,
      hints: [
        'Use X.min(axis=0) and X.max(axis=0) to calculate bounds per feature column.',
        'Apply elements subtraction and division.'
      ],
      testKeywords: [
        'min(axis=0)',
        'max(axis=0)'
      ]
    },
    project: {
      title: 'Pre-processing Pipeline with ColumnTransformer',
      description: 'Create a scikit-learn preprocessing pipeline using ColumnTransformer. Standardize numerical columns and one-hot encode categorical columns.',
      pseudoCode: '1. Separate numerical and categorical columns.\n2. Build a ColumnTransformer.\n3. Apply StandardScaler to numeric columns.\n4. Apply OneHotEncoder to categorical columns.\n5. Fit and transform the data.',
      starterCode: `import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Tabular Dataset: [Age, Income, Department_Code]
# Department_Code is categorical ('HR'=0, 'ENG'=1, 'SALES'=2)
X = np.array([
    [25.0, 50000.0, 0.0],
    [30.0, 85000.0, 1.0],
    [45.0, 120000.0, 2.0],
    [22.0, 48000.0, 0.0]
])

# TODO: Create a ColumnTransformer named 'preprocessor'
# Apply StandardScaler to indices [0, 1]
# Apply OneHotEncoder(sparse_output=False) to index [2]
# Transform X and print final shape

preprocessor = None
X_trans = X
print("Shape:", X_trans.shape)`,
      expectedOutput: 'Shape: (4, 5)',
      solution: `import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

X = np.array([
    [25.0, 50000.0, 0.0],
    [30.0, 85000.0, 1.0],
    [45.0, 120000.0, 2.0],
    [22.0, 48000.0, 0.0]
])

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), [0, 1]),
    ('cat', OneHotEncoder(sparse_output=False), [2])
])
X_trans = preprocessor.fit_transform(X)
print("Shape:", X_trans.shape)`,
      hints: [
        'Pass transformers list: [(\'num\', StandardScaler(), [0, 1]), (\'cat\', OneHotEncoder(sparse_output=False), [2])].',
        'Use preprocessor.fit_transform(X) to execute pipeline.'
      ],
      testKeywords: [
        'ColumnTransformer',
        'StandardScaler',
        'OneHotEncoder',
        'fit_transform'
      ]
    },
    assignment: {
      title: 'Target Importance Selection',
      description: 'Kaggle challenge: Run feature selection using `SelectFromModel` with a `RandomForestClassifier` on a tabular classification dataset.',
      pseudoCode: '1. Fit a RandomForestClassifier on X, y.\n2. Instantiate SelectFromModel passing the classifier.\n3. Fit select model and transform X to X_selected.\n4. Print the shape of the selected features.',
      starterCode: `import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

np.random.seed(42)
X = np.random.rand(50, 10)  # 50 rows, 10 raw features
# Only features at index 1 and 3 are actually predictive:
y = (X[:, 1] + X[:, 3] > 1.0).astype(int)

# TODO: Fit a RandomForestClassifier(n_estimators=50, random_state=42)
# TODO: Use SelectFromModel to select features with importances > median
# TODO: Transform X and print final X_selected shape

selector = None
X_selected = X
print("Selected Shape:", X_selected.shape)`,
      expectedOutput: 'Selected Shape: (50, 5)',
      solution: `import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

np.random.seed(42)
X = np.random.rand(50, 10)
y = (X[:, 1] + X[:, 3] > 1.0).astype(int)

clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X, y)

selector = SelectFromModel(estimator=clf, threshold='median')
selector.fit(X, y)
X_selected = selector.transform(X)
print("Selected Shape:", X_selected.shape)`,
      hints: [
        'Set threshold=\'median\' inside SelectFromModel to keep top 50% features.',
        'Use selector.fit(X, y) followed by selector.transform(X).'
      ],
      testKeywords: [
        'RandomForestClassifier',
        'SelectFromModel',
        'threshold',
        'transform'
      ]
    }
  },
  interviewQuestions: [
    {
      question: 'What is Target Encoding, and what is its main drawback in competition environments?',
      answer: 'Target encoding replaces categorical labels with the average target label value for that category. Its main drawback is target leakage and overfitting. Because it maps categories to the target value directly, it can heavily overfit on rare categories unless smoothing or out-of-fold calculations (K-fold target encoding) are applied.',
      companyTags: ['Kaggle', 'Stripe'],
      difficulty: 'Advanced'
    },
    {
      question: 'Explain the difference between L1 (Lasso) and L2 (Ridge) regularization for feature selection.',
      answer: 'L1 regularization adds the absolute value of weights to the loss function, which causes some coefficients to shrink to exactly zero, serving as a built-in feature selection method. L2 regularization adds the squared weights, shrinking coefficients towards zero smoothly but keeping all features in the model, which is better for collinear features.',
      companyTags: ['Google', 'Amazon'],
      difficulty: 'Medium'
    },
    {
      question: 'What is multicollinearity, and how does standard scaling help or not help it?',
      answer: 'Multicollinearity is when multiple independent features are highly correlated, leading to unstable coefficients in linear models. Standard scaling does NOT solve multicollinearity (it only shifts and scales values). It must be fixed using L2 regularization (Ridge), feature drop, or dimensionality reduction (PCA).',
      companyTags: ['Uber', 'Capital One'],
      difficulty: 'Hard'
    }
  ]
};
