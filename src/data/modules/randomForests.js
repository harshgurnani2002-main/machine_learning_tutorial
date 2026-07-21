export const randomForests = {
    id: 'random-forests',
    title: 'Random Forests',
    category: 'Supervised Learning',
    description: 'Ensemble deep decision trees using bagging and random feature selection.',
    formula: 'f(x) = \frac{1}{B} \\sum_{b=1}^{B} f_b(x)',
    interactiveSummary: "This interactive simulator shows how a Random Forest creates an ensemble of decision trees, each trained on a different random bootstrap of your data. You can click the canvas to add new data points labeled as Class 0 or Class 1, then drag the 'Number of Trees' slider to see how the combined decision boundary shifts and stabilizes as more trees are added — notice how a single tree produces a jagged, unstable boundary that smooths out into a confident, robust region as the forest grows. Each tree independently votes for a class, and the shaded background regions show the majority-vote result across the entire feature space. This directly demonstrates the core strength of bagging: individual trees may disagree on edge cases, but their collective vote cancels out noise and produces a far more reliable prediction than any single model could.",
    theory: `### Random Forests & Ensemble Learning Theory

#### What is it?
Random Forest is a highly versatile, supervised machine learning algorithm. 
It is capable of performing both regression and classification tasks natively.
The algorithm operates by constructing a multitude of decision trees at training time.

For classification tasks, the output of the random forest is the class selected by the majority of the trees.
For regression tasks, the mean or average prediction of the individual trees is returned.

It belongs to the broader class of ensemble learning methods.
Ensemble learning combines multiple models to produce a single, more powerful model.
By relying on a "committee" of decision trees, Random Forests mitigate the weaknesses of individual trees.
This results in a highly robust, accurate, and stable algorithm that resists overfitting.

#### Why do we need it?
Single Decision Trees are notorious for overfitting. 
If allowed to grow deep without constraints, they will memorize the training dataset perfectly.
This captures noise and outliers, resulting in poor generalization to unseen data.

Conversely, shallow trees might underfit the data. 
We need Random Forests because they systematically reduce the variance of the model.
Crucially, they achieve this variance reduction without substantially increasing bias.

They combine the predictions of multiple weak learners (trees) into a single strong learner. 
This ensemble approach natively handles high-dimensional data.
It maintains accuracy even when a large proportion of data is missing.
It also provides a reliable measure of feature importance, making it highly valuable.

#### How does it work?
Random Forests build upon Decision Trees by introducing two key sources of randomness.
This randomness ensures that the individual trees are de-correlated:

1. **Bagging (Bootstrap Aggregation)**: 
Instead of training each tree on the entire dataset, each tree is trained on a random sample.
This sample is drawn *with replacement* from the training data. 
This means some samples may appear multiple times in a single tree's training set.
Other samples may not appear at all (these are the Out-Of-Bag samples).

2. **Random Feature Selection**: 
In a standard decision tree, the algorithm evaluates all possible features to find the best split.
In a Random Forest, at each node of a tree, only a random subset of features is considered.
This forces the trees to find alternative predictive pathways.

By injecting randomness in these two dimensions, the ensemble consists of uncorrelated trees. 
When these independent trees vote, the idiosyncratic errors cancel out.
This leads to a much more stable and accurate overall prediction.

#### The Math Behind It
The theoretical foundation of Random Forests relies on the variance of the average of random variables.
Let $\rho$ be the pairwise correlation between any two trees.
Let $\\sigma^2$ be the variance of a single tree. 
The variance of the ensemble average of $B$ trees is given by:

$$
\text{Var}\\left(\frac{1}{B} \\sum_{b=1}^{B} f_b(x)\right) = \rho \\sigma^2 + \frac{1 - \rho}{B} \\sigma^2
$$

As the number of trees $B \to \\infty$, the second term vanishes.
The overall variance is therefore bounded by $\rho \\sigma^2$. 
Random feature selection specifically aims to reduce the correlation $\rho$ between trees.
This directly drives down the overall ensemble variance.

1. **Bagging**: 
Given $N$ samples, a bootstrap sample selects $N$ elements with replacement.
The probability of a sample *not* being selected in a single draw is $1 - 1/N$. 
After $N$ draws, the probability is:

$$
\\lim_{N \to \\infty} \\left(1 - \frac{1}{N}\right)^N = \frac{1}{e} \\approx 0.368
$$

Thus, about $63.2\\%$ of unique records are used for training a given tree.
The remaining $36.8\\%$ form the **Out-Of-Bag (OOB)** samples.
These OOB samples act as a free validation set.

2. **Feature Subsampling**: 
To decorrelate the trees, we search only a random subset of features $m \\ll d$.
Here, $d$ is the total number of features. 
Typical heuristics used are:
- For Classification: 
$$m = \\sqrt{d}$$
- For Regression: 
$$m = \frac{d}{3}$$

3. **Aggregation**: 
- **Regression**: Average individual tree predictions: 
$$
\\hat{y} = \frac{1}{B} \\sum_{b=1}^{B} f_b(x)
$$
- **Classification**: Majority vote (mode) selection: 
$$
\\hat{y} = \\arg\\max_c \\sum_{b=1}^{B} I(f_b(x) = c)
$$
where $I$ is the indicator function.

#### Worked Example
Imagine you want to predict if a user will click on an advertisement (Click/No Click). 
You have 10,000 user profiles and 16 features (Age, Income, Browsing History, Device Type, etc.).

- You decide to train a Random Forest with 500 trees.
- **Tree 1** receives a bootstrapped sample of 10,000 rows.
  This sample contains ~6,320 unique users, with some duplicated. 
  At the root node, it randomly selects $\\sqrt{16} = 4$ features.
  For example: Age, Device Type, Time of Day, Location. 
  It finds that Device Type offers the best split and grows to its maximum depth.
- **Tree 2** receives a *different* bootstrapped sample. 
  At its root node, it randomly selects 4 *different* features.
  For example: Income, Browsing History, Gender, Ad Category. 
  It splits on Browsing History.
- This process repeats 500 times.
  The result is an ensemble of 500 highly diverse, specialized trees.
- During prediction for a new user, all 500 trees evaluate the user's profile. 
  If 350 trees predict "Click" and 150 predict "No Click".
  The final Random Forest prediction is the majority vote: "Click".

#### Common Pitfalls
1. **Memory and Disk Usage**: 
Random forests with hundreds or thousands of deep trees can consume a significant amount of RAM.
They also produce very large model files which can be difficult to deploy.

2. **Prediction Speed**: 
Because every single tree must be evaluated to make a prediction, inference can be slow.
This might be an issue in extreme low-latency environments (e.g., high-frequency trading).

3. **Extrapolation in Regression**: 
Tree-based models partition the feature space and average the targets in the leaves. 
Consequently, they cannot extrapolate or predict target values outside the training range.

4. **Suboptimal on Sparse Data**: 
On very high-dimensional, sparse datasets (like text TF-IDF matrices), Random Forests struggle.
The random feature subsets are overwhelmingly composed of uninformative zeros.

#### When to Use vs Not Use

**Use when:**
- You have tabular data with complex, non-linear relationships.
- There are significant interactions between features that linear models would miss.
- You want a strong baseline model that requires very little hyperparameter tuning.
- You need feature importance rankings out-of-the-box to explain the model.
- You have datasets with mixed data types and unscaled numerical features.

**Not Use when:**
- You need lightning-fast prediction speeds at inference time.
- You are working with high-dimensional sparse data like text or gene expressions.
- You are doing image, video, or audio recognition (Deep Learning is strictly better).
- You need to extrapolate trends into the future (e.g., time-series forecasting).

#### Key Takeaways
- **Ensemble Power**: Combines Bagging and Random Feature Subsampling.
- **De-correlation**: Heavily decorrelates decision trees, making the whole greater than the sum.
- **Bias-Variance Tradeoff**: Systematically reduces variance without significantly increasing bias.
- **Robustness**: Extremely robust to outliers and does not strictly require feature scaling.
- **OOB Evaluation**: The Out-Of-Bag (OOB) error gives an unbiased estimate of the test set error.

#### Python Implementation

\`\`\`python
from sklearn.ensemble import RandomForestClassifier

X = [[0, 0], [1, 1], [2, 2], [3, 3]]
y = [0, 0, 1, 1]
rf = RandomForestClassifier(n_estimators=100, max_depth=2)
rf.fit(X, y)
print(f"Feature importances: {rf.feature_importances_}")
\`\`\`
`,
    simulatorId: 'random-forest',
    quiz: [
        {
            id: 'rf_q1',
            question: 'Why does random feature selection (subsampling at splits) improve the ensemble performance compared to standard bagging?',
            options: [
                'It decorrelates individual trees, preventing dominant features from dictating every split.',
                'It forces the trees to learn linear boundaries instead of step boundaries.',
                'It speeds up prediction time significantly.',
                'It reduces model bias to zero.'
            ],
            correctAnswer: 'It decorrelates individual trees, preventing dominant features from dictating every split.',
            explanation: 'If a single feature is highly predictive, standard bagging trees will always pick it first, making trees highly correlated. Feature subsampling forces alternative pathways to be explored.'
        },
        {
            id: 'rf_q2',
            question: 'What percentage of training samples are typically omitted from a single bootstrapped dataset (Out-Of-Bag samples)?',
            options: ['~36.8%', '~63.2%', '~50%', '~10%'],
            correctAnswer: '~36.8%',
            explanation: 'The probability of a sample not being selected in N draws with replacement approaches 1/e ≈ 36.8% as N grows. These Out-Of-Bag (OOB) samples act as a built-in validation set.'
        },
        {
            id: 'rf_q3',
            question: 'How does increasing the number of trees (estimators) affect Random Forest performance?',
            options: [
                'It converges and stabilizes the variance without overfitting the ensemble.',
                'It increases the bias significantly.',
                'It causes severe overfitting.',
                'It reduces the feature dimensions.'
            ],
            correctAnswer: 'It converges and stabilizes the variance without overfitting the ensemble.',
            explanation: 'Adding more estimators does not cause overfitting in Random Forests; it simply averages out variance, converging toward a stable lower-bound error rate.'
        },
        {
            id: 'rf_q4',
            question: 'What is the default feature subsampling size (m) for classification tasks with d total features?',
            options: ['m = sqrt(d)', 'm = d/3', 'm = log2(d)', 'm = d'],
            correctAnswer: 'm = sqrt(d)',
            explanation: 'For classification, the typical rule of thumb and default in many libraries is to use the square root of the total number of features.'
        },
        {
            id: 'rf_q5',
            question: 'Which of the following is a disadvantage of Random Forests?',
            options: [
                'They cannot extrapolate beyond the range of training data in regression.',
                'They require extensive feature scaling and normalization.',
                'They are highly prone to overfitting on small datasets.',
                'They cannot handle categorical variables natively.'
            ],
            correctAnswer: 'They cannot extrapolate beyond the range of training data in regression.',
            explanation: 'Tree-based models predict by averaging leaf node values. They cannot produce a value larger than the maximum or smaller than the minimum value seen in the training data.'
        },
        {
            id: 'rf_q6',
            question: 'What is the primary way Random Forests handle missing values internally?',
            options: [
                'Proximity matrix-based imputation.',
                'Deleting rows with missing values.',
                'Using the mean of the entire dataset.',
                'Setting missing values to zero.'
            ],
            correctAnswer: 'Proximity matrix-based imputation.',
            explanation: 'Advanced implementations of RF use a proximity-based imputation method, guessing missing values based on similarities (proximities) between samples.'
        },
        {
            id: 'rf_q7',
            question: 'In a Random Forest, what is the role of an Out-Of-Bag (OOB) sample?',
            options: [
                'It is used to validate the model without needing a separate validation set.',
                'It is used to build the final tree in the ensemble.',
                'It is the subset of features not chosen during a split.',
                'It is the sample of data with missing labels.'
            ],
            correctAnswer: 'It is used to validate the model without needing a separate validation set.',
            explanation: 'Since OOB samples are not used to train a specific tree, they can be passed through that tree to get an unbiased estimate of the error.'
        },
        {
            id: 'rf_q8',
            question: 'True or False: Random Forests require feature scaling (like standardization or normalization).',
            options: ['True', 'False', 'Only for classification', 'Only for regression'],
            correctAnswer: 'False',
            explanation: 'Decision trees (and by extension, Random Forests) make splits based on relative ordering of values, not absolute magnitudes. Thus, they are invariant to monotonic transformations like scaling.'
        },
        {
            id: 'rf_q9',
            question: 'How is feature importance typically calculated in a Random Forest?',
            options: [
                'By measuring the mean decrease in impurity (e.g., Gini) across all trees for a feature.',
                'By looking at the p-value of the feature in a linear equation.',
                'By calculating the correlation matrix of the features.',
                'By counting how many times a feature is missing.'
            ],
            correctAnswer: 'By measuring the mean decrease in impurity (e.g., Gini) across all trees for a feature.',
            explanation: 'Feature importance is often computed as the total reduction of the criterion (e.g., Gini impurity or MSE) brought by that feature, averaged over all trees.'
        },
        {
            id: 'rf_q10',
            question: 'Which hyperparameter is typically MOST important to tune in a Random Forest?',
            options: [
                'max_features',
                'n_estimators',
                'random_state',
                'verbose'
            ],
            correctAnswer: 'max_features',
            explanation: 'While increasing n_estimators is always good (up to compute limits), max_features controls the diversity of the trees and has the largest impact on balancing bias and variance.'
        }
    ],
    coding: {
        tutorial: {
            title: 'Bootstrap Sampling Generator',
            description: 'Generate a bootstrap sample of indices with replacement from a given dataset length.\n\n**Sklearn Equivalent:**\n`from sklearn.utils import resample`\n`bootstrapped_data = resample(data, replace=True)`',
            pseudoCode: 'function generate_bootstrap(num_samples):\n  indices = empty_array\n  for i from 1 to num_samples:\n    random_idx = random_integer(0, num_samples-1)\n    indices.add(random_idx)\n  return indices',
            starterCode: `import numpy as np

def generate_bootstrap(num_samples, seed=42):
    np.random.seed(seed)
    # TODO: Select indices between 0 and num_samples-1 with replacement
    indices = np.array([])
    return indices

print("Bootstrapped Indices:", generate_bootstrap(5))`,
            expectedOutput: 'Bootstrapped Indices: [3 4 2 4 4]',
            solution: `import numpy as np

def generate_bootstrap(num_samples, seed=42):
    np.random.seed(seed)
    indices = np.random.choice(num_samples, size=num_samples, replace=True)
    return indices

print("Bootstrapped Indices:", generate_bootstrap(5))`,
            hints: ['Use np.random.choice to select indices.', 'Set the size parameter equal to num_samples.', 'Ensure replace=True is active.'],
            testKeywords: ['np.random.choice', 'replace=True', 'size=']
        },
        project: {
            title: 'Wine Quality Prediction with Random Forest',
            description: 'Use a Random Forest to predict whether a wine is high quality (1) or low quality (0) based on 11 physicochemical features from a real-world wine dataset (alcohol, pH, sulphates, residual sugar, citric acid, volatile acidity, fixed acidity, chlorides, free sulfur dioxide, total sulfur dioxide, density). This mirrors a real production use-case in food science and quality control.',
            pseudoCode: 'Load wine dataset with 11 physicochemical features\nInitialize RandomForestClassifier(n_estimators=100, random_state=42)\nSplit data into X (features) and y (quality label 0/1)\nscores = cross_val_score(clf, X, y, cv=3)\nprint(Mean CV Accuracy)',
            starterCode: `from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Wine Quality Dataset (11 features: fixed_acidity, volatile_acidity, citric_acid,
# residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
# density, pH, sulphates, alcohol -> quality: 0=low, 1=high)
X = np.array([
    [7.4, 0.70, 0.00, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4],
    [7.8, 0.88, 0.00, 2.6, 0.098, 25, 67, 0.9968, 3.20, 0.68, 9.8],
    [7.8, 0.76, 0.04, 2.3, 0.092, 15, 54, 0.9970, 3.26, 0.65, 9.8],
    [11.2, 0.28, 0.56, 1.9, 0.075, 17, 60, 0.9980, 3.16, 0.58, 9.8],
    [8.1, 0.28, 0.40, 6.9, 0.050, 30, 97, 0.9951, 3.26, 0.44, 10.1],
    [7.4, 0.66, 0.00, 1.8, 0.075, 13, 40, 0.9978, 3.51, 0.56, 9.4],
    [7.9, 0.60, 0.06, 1.6, 0.069,  5, 22, 0.9964, 3.30, 0.51, 9.4],
    [7.3, 0.65, 0.00, 1.2, 0.065, 15, 21, 0.9946, 3.39, 0.47, 10.0],
    [7.8, 0.58, 0.02, 2.0, 0.073,  9, 18, 0.9968, 3.36, 0.57, 9.5],
    [7.5, 0.50, 0.36, 6.1, 0.071, 17, 102, 0.9978, 3.35, 0.80, 10.5],
    [6.7, 0.58, 0.08, 1.8, 0.097, 15, 65, 0.9959, 3.28, 0.54, 9.2],
    [7.5, 0.50, 0.36, 6.1, 0.071, 17, 102, 0.9978, 3.35, 0.80, 10.5],
])
y = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1])

# TODO: Initialize a RandomForestClassifier with n_estimators=100, random_state=42
# TODO: Use cross_val_score with cv=3 to evaluate on this dataset
# TODO: Print the mean accuracy formatted to 3 decimal places

clf = None
scores = []
print(f"Mean CV Accuracy: 0.000")`,
            expectedOutput: 'Mean CV Accuracy: 0.583',
            solution: `from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

X = np.array([
    [7.4, 0.70, 0.00, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4],
    [7.8, 0.88, 0.00, 2.6, 0.098, 25, 67, 0.9968, 3.20, 0.68, 9.8],
    [7.8, 0.76, 0.04, 2.3, 0.092, 15, 54, 0.9970, 3.26, 0.65, 9.8],
    [11.2, 0.28, 0.56, 1.9, 0.075, 17, 60, 0.9980, 3.16, 0.58, 9.8],
    [8.1, 0.28, 0.40, 6.9, 0.050, 30, 97, 0.9951, 3.26, 0.44, 10.1],
    [7.4, 0.66, 0.00, 1.8, 0.075, 13, 40, 0.9978, 3.51, 0.56, 9.4],
    [7.9, 0.60, 0.06, 1.6, 0.069,  5, 22, 0.9964, 3.30, 0.51, 9.4],
    [7.3, 0.65, 0.00, 1.2, 0.065, 15, 21, 0.9946, 3.39, 0.47, 10.0],
    [7.8, 0.58, 0.02, 2.0, 0.073,  9, 18, 0.9968, 3.36, 0.57, 9.5],
    [7.5, 0.50, 0.36, 6.1, 0.071, 17, 102, 0.9978, 3.35, 0.80, 10.5],
    [6.7, 0.58, 0.08, 1.8, 0.097, 15, 65, 0.9959, 3.28, 0.54, 9.2],
    [7.5, 0.50, 0.36, 6.1, 0.071, 17, 102, 0.9978, 3.35, 0.80, 10.5],
])
y = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1])

clf = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(clf, X, y, cv=3)
print(f"Mean CV Accuracy: {scores.mean():.3f}")`,
            hints: ['Import RandomForestClassifier from sklearn.ensemble.', 'Set n_estimators=100 and random_state=42.', 'cross_val_score returns an array — call .mean() to compute average accuracy.'],
            testKeywords: ['RandomForestClassifier', 'n_estimators=100', 'cross_val_score', 'mean']
        },
        assignment: {
            title: 'Customer Churn Prediction & Feature Importance',
            description: 'A telecom company wants to identify customers likely to cancel their subscription (churn). Build a RandomForestClassifier on a customer churn dataset with features: age, tenure_months, monthly_charges, num_products, and has_support_contract (0/1). After fitting, extract the feature importances to reveal which factors drive churn the most — a critical insight for retention teams.',
            pseudoCode: 'Load churn dataset (age, tenure, monthly_charges, num_products, support_contract)\nInitialize RandomForestClassifier(n_estimators=50, random_state=0)\nrf.fit(X, y)\nfeature_names = ["age", "tenure_months", "monthly_charges", "num_products", "has_support_contract"]\nimportances = np.round(rf.feature_importances_, 3)\nfor name, imp in zip(feature_names, importances): print(name, imp)',
            starterCode: `from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Customer Churn Dataset
# Features: [age, tenure_months, monthly_charges, num_products, has_support_contract]
# Target: 1 = churned, 0 = retained
X = np.array([
    [22, 2,  85.5, 1, 0],
    [45, 36, 55.0, 3, 1],
    [30, 5,  95.0, 1, 0],
    [52, 60, 45.0, 4, 1],
    [25, 1,  105.0, 1, 0],
    [38, 24, 65.0, 2, 1],
    [29, 3,  90.0, 1, 0],
    [60, 48, 50.0, 3, 1],
    [34, 8,  80.0, 2, 0],
    [27, 2,  110.0, 1, 0],
])
y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 1])

feature_names = ["age", "tenure_months", "monthly_charges", "num_products", "has_support_contract"]

# TODO: Initialize RandomForestClassifier with n_estimators=50, random_state=0
# TODO: Fit the model on X, y
# TODO: Print each feature name alongside its importance (rounded to 3 decimal places)

rf = None
print("Feature Importances:")`,
            expectedOutput: 'Feature Importances: monthly_charges is most important',
            solution: `from sklearn.ensemble import RandomForestClassifier
import numpy as np

X = np.array([
    [22, 2,  85.5, 1, 0],
    [45, 36, 55.0, 3, 1],
    [30, 5,  95.0, 1, 0],
    [52, 60, 45.0, 4, 1],
    [25, 1,  105.0, 1, 0],
    [38, 24, 65.0, 2, 1],
    [29, 3,  90.0, 1, 0],
    [60, 48, 50.0, 3, 1],
    [34, 8,  80.0, 2, 0],
    [27, 2,  110.0, 1, 0],
])
y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 1])

feature_names = ["age", "tenure_months", "monthly_charges", "num_products", "has_support_contract"]

rf = RandomForestClassifier(n_estimators=50, random_state=0)
rf.fit(X, y)
importances = np.round(rf.feature_importances_, 3)
print("Feature Importances:")
for name, imp in zip(feature_names, importances):
    print(f"  {name}: {imp}")`,
            hints: ['Use RandomForestClassifier, not Regressor.', 'Call .fit(X, y) to train the model.', 'Access feature_importances_ and zip it with feature_names to print each pair.'],
            testKeywords: ['RandomForestClassifier', 'fit', 'feature_importances_', 'zip']
        }
    },
    interviewQuestions: [
        {
            question: 'Why do Random Forests not overfit when we increase the number of estimators, and what is the Out-Of-Bag (OOB) error?',
            answer: 'Adding trees reduces the variance of the ensemble because the law of large numbers guarantees that the average prediction error converges: the variance is bounded by (rho * sigma^2 + (1 - rho)/B * sigma^2). As B (trees) approaches infinity, the second term vanishes, leaving rho * sigma^2 (correlation among trees). Out-of-bag (OOB) error uses the ~36.8% samples omitted from training each tree. By predicting each sample using only trees that did not see it, we calculate an unbiased validation score without a separate validation split.',
            difficulty: 'Advanced',
            companyTags: ['Meta', 'Uber']
        },
        {
            question: 'What is bagging, and how does it relate to Random Forest?',
            answer: 'Bagging (Bootstrap Aggregating) is an ensemble technique where multiple models are trained on randomly sampled subsets of the training data (with replacement). Random Forest is a specific extension of bagging applied to decision trees, with the added twist of random feature selection at each split.',
            difficulty: 'Beginner',
            companyTags: ['Amazon', 'Google']
        },
        {
            question: 'How do you handle categorical variables in Random Forests?',
            answer: 'Most standard implementations (like sklearn) require categorical variables to be converted to numerical values using techniques like One-Hot Encoding or Ordinal Encoding. Some specific libraries (like H2O or R\'s randomForest) can handle categorical variables natively by evaluating splits across subsets of categories.',
            difficulty: 'Intermediate',
            companyTags: ['Netflix', 'Microsoft']
        },
        {
            question: 'Can Random Forests be used for unsupervised learning?',
            answer: 'Yes, they can be adapted to generate a proximity matrix, which measures the similarity between data points based on how frequently they end up in the same leaf node. This matrix can then be used for clustering, outlier detection, or missing value imputation.',
            difficulty: 'Advanced',
            companyTags: ['Apple', 'LinkedIn']
        },
        {
            question: 'What hyperparameters would you tune to combat overfitting in a Random Forest?',
            answer: 'I would decrease `max_depth` to constrain tree size, increase `min_samples_split` or `min_samples_leaf` to require more data for splits, or decrease `max_features` to force more diversity among the trees.',
            difficulty: 'Intermediate',
            companyTags: ['Google', 'Airbnb']
        },
        {
            question: 'How does Random Forest calculate feature importance?',
            answer: 'There are two main methods: Mean Decrease Impurity (Gini importance), which averages the reduction in impurity caused by a feature across all trees, and Permutation Importance, which measures the drop in model performance when a feature\'s values are randomly shuffled.',
            difficulty: 'Intermediate',
            companyTags: ['Stripe', 'Square']
        },
        {
            question: 'What is the impact of highly correlated features on Random Forest?',
            answer: 'Highly correlated features can mask each other\'s importance. The model might arbitrarily split importance between them, making the feature importance scores less reliable for interpretation. However, predictive performance usually remains robust.',
            difficulty: 'Advanced',
            companyTags: ['Snap', 'Pinterest']
        },
        {
            question: 'Why might a Random Forest perform poorly on sparse data (e.g., TF-IDF text features)?',
            answer: 'In sparse data, most feature values are zero. At any given node, a random subset of features is selected. If most selected features are uninformative zeros, the tree struggles to make meaningful splits, leading to poor and very deep trees.',
            difficulty: 'Intermediate',
            companyTags: ['Twitter', 'Amazon']
        },
        {
            question: 'What is the difference between Extra Trees and Random Forest?',
            answer: 'Extra Trees (Extremely Randomized Trees) also use random feature subsets, but unlike Random Forest, they do not bootstrap the data (they use the whole dataset) and they choose the splitting threshold randomly for each feature, rather than finding the optimal threshold.',
            difficulty: 'Advanced',
            companyTags: ['Uber', 'Databricks']
        },
        {
            question: 'How do you deal with imbalanced classes using Random Forests?',
            answer: 'You can use class weighting (giving higher weight to the minority class in the impurity calculation), or use Balanced Random Forest, where each bootstrap sample under-samples the majority class to achieve balance.',
            difficulty: 'Intermediate',
            companyTags: ['Meta', 'Capital One']
        },
        {
            question: 'Does Random Forest require feature scaling?',
            answer: 'No, tree-based models make decisions based on value orderings, not magnitudes or distances, so scaling (standardization/normalization) has no effect on the splits.',
            difficulty: 'Beginner',
            companyTags: ['Netflix', 'Spotify']
        },
        {
            question: 'How does Random Forest parallelize training?',
            answer: 'Because each tree is built independently on its own bootstrap sample, they can be trained in parallel across multiple CPU cores without any communication overhead.',
            difficulty: 'Intermediate',
            companyTags: ['Google', 'AWS']
        },
        {
            question: 'What is the time complexity of training a Random Forest?',
            answer: 'For a dataset with N samples, M features, and B trees, building one full tree is roughly O(M * N log N). Thus, training the forest is O(B * M * N log N). If max_features=sqrt(M), it becomes O(B * sqrt(M) * N log N).',
            difficulty: 'Advanced',
            companyTags: ['Microsoft', 'Jane Street']
        },
        {
            question: 'How is regression prediction handled at the leaf nodes?',
            answer: 'In regression trees, the prediction at a leaf node is typically the mean of the target values of all training samples that fall into that leaf.',
            difficulty: 'Beginner',
            companyTags: ['Airbnb', 'Lyft']
        },
        {
            question: 'What is Gini Impurity?',
            answer: 'Gini impurity measures the probability of incorrectly classifying a randomly chosen element in the dataset if it were randomly labeled according to the class distribution in the subset. It is used as a splitting criterion.',
            difficulty: 'Intermediate',
            companyTags: ['Apple', 'LinkedIn']
        },
        {
            question: 'Can a Random Forest overfit?',
            answer: 'While Random Forests are resistant to overfitting as n_estimators increases, they can still overfit if the individual trees are too deep (no max_depth constraint) and the dataset is noisy or very small.',
            difficulty: 'Intermediate',
            companyTags: ['Meta', 'Google']
        },
        {
            question: 'What happens if max_features = total_features?',
            answer: 'If max_features equals the total number of features, the algorithm just becomes standard bagging with decision trees. The trees will be highly correlated because they will all pick the same strong features at the top nodes.',
            difficulty: 'Intermediate',
            companyTags: ['Amazon', 'Stripe']
        },
        {
            question: 'How do you interpret a Random Forest model?',
            answer: 'While individual trees are interpretable, forests are "black boxes". You can use Feature Importance, Partial Dependence Plots (PDP), or SHAP (SHapley Additive exPlanations) values to interpret them.',
            difficulty: 'Advanced',
            companyTags: ['Netflix', 'Meta']
        },
        {
            question: 'Explain the concept of "variance reduction" in Random Forests.',
            answer: 'A single tree has high variance (changes significantly with different training data). By averaging the predictions of many loosely correlated trees, the overall variance of the prediction is reduced, leading to a more robust model.',
            difficulty: 'Intermediate',
            companyTags: ['Google', 'Spotify']
        },
        {
            question: 'Why might a Random Forest be preferred over Gradient Boosting?',
            answer: 'Random Forests are easier to tune, parallelize naturally, and are less prone to overfitting noise compared to un-tuned Gradient Boosting models. They serve as an excellent baseline.',
            difficulty: 'Intermediate',
            companyTags: ['Uber', 'Capital One']
        }
    ]
};
