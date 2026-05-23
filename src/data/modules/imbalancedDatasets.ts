import type { MLModule } from '../../types';

export const imbalancedDatasets: MLModule = {
  id: 'imbalanced-datasets',
  title: 'Imbalanced Datasets',
  category: 'Advanced & MLOps',
  description: 'Master techniques to handle skewed class distributions — SMOTE, class weighting, undersampling, and the right evaluation metrics for imbalanced problems.',
  formula: 'F_1 = \\frac{2 \\cdot \\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}}',
  interactiveSummary: 'Use the interactive SMOTE simulator to visualize how synthetic minority samples are generated through interpolation. Watch as the class distribution balances and the decision boundary shifts to properly classify the minority class.',
  simulatorId: 'imbalanced-simulator',
  theory: `### Handling Imbalanced Datasets

#### What is Class Imbalance?
A dataset is imbalanced when the number of samples in one class (the minority) is significantly smaller than another (the majority). Common real-world examples:
- **Fraud Detection**: 0.1% fraudulent, 99.9% legitimate transactions
- **Medical Diagnosis**: 1% cancer-positive, 99% healthy
- **Intrusion Detection**: Rare attacks among millions of normal packets

The imbalance ratio (IR) is defined as:

$$IR = \\frac{N_{majority}}{N_{minority}}$$

An IR above 10 is considered moderately imbalanced; above 100 is severely imbalanced.

#### Why Accuracy Fails on Imbalanced Data
This is the most critical concept: **accuracy is a misleading metric for imbalanced data**.

Consider a fraud dataset with 9,900 legitimate and 100 fraudulent transactions. A model that predicts "Legitimate" for **every single sample** achieves:

$$\\text{Accuracy} = \\frac{9900}{10000} = 99\\%$$

Despite being completely useless! This is called the **accuracy paradox**. You need better metrics.

#### Better Evaluation Metrics

**Confusion Matrix Terminology:**
- **True Positive (TP)**: Fraud correctly identified as fraud
- **True Negative (TN)**: Legitimate correctly identified as legitimate
- **False Positive (FP)**: Legitimate wrongly flagged as fraud (Type I Error)
- **False Negative (FN)**: Fraud missed and called legitimate (Type II Error — often the most costly!)

**Precision**: Of all predicted positives, how many are actually positive?
$$\\text{Precision} = \\frac{TP}{TP + FP}$$

**Recall (Sensitivity)**: Of all actual positives, how many did we catch?
$$\\text{Recall} = \\frac{TP}{TP + FN}$$

**F1 Score**: The harmonic mean of Precision and Recall. Balances both:
$$F_1 = \\frac{2 \\cdot \\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}}$$

**ROC-AUC**: Measures the model's ability to discriminate between classes at all thresholds. Robust to class imbalance.

**PR-AUC (Precision-Recall AUC)**: Preferred over ROC-AUC when the positive class is very rare. More sensitive to improvements in the minority class.

#### Resampling Strategies

**1. Random Undersampling**
Randomly remove samples from the majority class until the dataset is balanced.
- ✅ Fast; reduces training time on huge datasets
- ❌ Throws away potentially useful data; risky with small datasets

**2. Random Oversampling**
Randomly duplicate samples from the minority class.
- ✅ No information loss from the majority class
- ❌ Creates exact duplicates → model memorizes them → overfitting

**3. SMOTE (Synthetic Minority Over-sampling Technique)**
Generates **synthetic** minority samples by interpolating between existing ones.

The algorithm for each minority sample $x_i$:
1. Find the $k$ nearest neighbors of $x_i$ within the minority class
2. Randomly select one neighbor $x_{zi}$
3. Generate a new synthetic sample on the line connecting them:

$$x_{new} = x_i + \\lambda \\cdot (x_{zi} - x_i), \\quad \\lambda \\sim \\text{Uniform}(0, 1)$$

- ✅ Creates diverse, realistic synthetic samples (not exact duplicates)
- ✅ Proven to outperform random oversampling in most settings
- ❌ Can generate noisy samples if minority class is scattered among majority samples
- ❌ Does not respect the decision boundary — may generate samples in overlapping regions

**4. ADASYN (Adaptive Synthetic Sampling)**
An improvement over SMOTE. Generates more synthetic samples in regions where the minority class is harder to learn (near the decision boundary):

$$\\hat{G}_i = \\Delta_i / \\sum_{i=1}^{m_s} \\Delta_i$$

where $\\Delta_i$ is the ratio of majority samples in the $k$-neighborhood of minority sample $i$. Samples in boundary regions get more synthetic neighbors.

**5. Tomek Links**
A cleanup technique: remove majority class samples that are very close to minority class samples (forming "links"). This cleans the decision boundary.

**6. SMOTE + Tomek / SMOTE + ENN**
A combined strategy: oversample with SMOTE, then clean noisy samples with Tomek Links or Edited Nearest Neighbors. This is often the most effective combination.

#### Class Weighting — The Algebraic Approach
Instead of resampling, you can tell the model to **penalize misclassifications of the minority class more heavily** by modifying the loss function.

For Logistic Regression, the weighted cross-entropy loss is:

$$L = -\\sum_i w_{y_i} \\cdot [y_i \\log(\\hat{p}_i) + (1 - y_i) \\log(1 - \\hat{p}_i)]$$

With $w_{minority} \\gg w_{majority}$.

In scikit-learn, set **class_weight='balanced'** to automatically compute:
$$w_j = \\frac{n_{samples}}{n_{classes} \\cdot n_{samples_j}}$$

This is the simplest and most computationally efficient solution — no resampling needed.

#### Threshold Tuning
Most classifiers output a probability, not a hard class. The default threshold is 0.5. For imbalanced datasets, you can lower the threshold (e.g., to 0.3) to increase recall at the cost of precision. Use the **Precision-Recall curve** to find the optimal threshold for your business requirements.

#### When to Use Each Technique
| Technique | Best For |
|---|---|
| class_weight='balanced' | Quick fix for any sklearn model |
| SMOTE | Medium imbalance (IR < 100), tabular data |
| ADASYN | Imbalanced with complex boundaries |
| Undersampling | Very large majority class (millions of samples) |
| SMOTE + Tomek | Best overall for most tabular problems |
| Threshold tuning | When you have asymmetric misclassification costs |
`,
  quiz: [
    {
      id: 'imb_q1',
      question: 'A model predicts "Not Fraud" for every transaction on a 99:1 fraud dataset and achieves 99% accuracy. What is its Recall for the fraud class?',
      options: ['99%', '50%', '1%', '0%'],
      correctAnswer: '0%',
      explanation: 'If the model never predicts "Fraud", it has zero True Positives. Recall = TP / (TP + FN) = 0 / (0 + 100) = 0%. This is the accuracy paradox — high accuracy, completely useless recall.'
    },
    {
      id: 'imb_q2',
      question: 'What does SMOTE generate when oversampling the minority class?',
      options: [
        'Exact duplicates of existing minority samples',
        'Random noise samples in the feature space',
        'Synthetic samples interpolated between real minority neighbors',
        'Samples copied from the majority class with flipped labels'
      ],
      correctAnswer: 'Synthetic samples interpolated between real minority neighbors',
      explanation: 'SMOTE creates new, unique samples by picking a minority point, finding its k nearest minority neighbors, and generating a point on the line segment connecting them. This creates diverse, realistic samples rather than duplicates.'
    },
    {
      id: 'imb_q3',
      question: 'For a cancer detection model, which metric should you prioritize: Precision or Recall?',
      options: [
        'Precision — we want to minimize false diagnoses',
        'Recall — we must catch every actual cancer case',
        'Accuracy — it gives the full picture',
        'Specificity — we want to maximize true negatives'
      ],
      correctAnswer: 'Recall — we must catch every actual cancer case',
      explanation: 'A False Negative (missing a cancer case) is far more costly than a False Positive (unnecessary follow-up test). High Recall ensures we catch all actual positives. The PR curve helps you tune the precision-recall tradeoff.'
    },
    {
      id: 'imb_q4',
      question: 'What does setting class_weight="balanced" do in a scikit-learn classifier?',
      options: [
        'Resamples the training data using SMOTE',
        'Adjusts the loss function to penalize misclassifying the minority class more heavily',
        'Removes majority class samples until classes are equal size',
        'Duplicates minority class samples randomly'
      ],
      correctAnswer: 'Adjusts the loss function to penalize misclassifying the minority class more heavily',
      explanation: 'class_weight="balanced" computes weights inversely proportional to class frequencies and applies them to the loss function. It is computationally free (no extra samples) and applies to the model\'s training objective directly.'
    },
    {
      id: 'imb_q5',
      question: 'What is the main risk of using Random Oversampling (duplicating minority samples) compared to SMOTE?',
      options: [
        'It is too slow for large datasets',
        'It reduces the size of the training set',
        'The model may overfit by memorizing the duplicated samples',
        'It changes the feature distributions of majority samples'
      ],
      correctAnswer: 'The model may overfit by memorizing the duplicated samples',
      explanation: 'When you copy the exact same samples multiple times, the model can simply memorize those specific points rather than learning the underlying pattern. SMOTE avoids this by creating unique synthetic points through interpolation.'
    },
    {
      id: 'imb_q6',
      question: 'Why is PR-AUC often preferred over ROC-AUC for severely imbalanced datasets?',
      options: [
        'PR-AUC is faster to compute',
        'ROC-AUC can be artificially high because it accounts for the large number of true negatives',
        'PR-AUC only works with binary classification',
        'ROC-AUC penalizes false positives too heavily'
      ],
      correctAnswer: 'ROC-AUC can be artificially high because it accounts for the large number of true negatives',
      explanation: 'With extreme imbalance (e.g., 99.9% negatives), the ROC-AUC looks great even for mediocre models because the huge TN count inflates the metric. PR-AUC focuses only on the positive (minority) class and is much more sensitive to model improvements that matter.'
    }
  ],
  coding: {
    tutorial: {
      title: 'Apply SMOTE to Balance a Dataset',
      description: 'Use the imbalanced-learn library to apply SMOTE to a synthetically imbalanced dataset. Print class counts before and after resampling.',
      pseudoCode: `1. Create imbalanced dataset (class 0: 1000 samples, class 1: 50 samples)
2. Print original class distribution
3. Apply SMOTE(random_state=42)
4. Print new class distribution
5. Confirm balance`,
      starterCode: `from collections import Counter
import numpy as np
from sklearn.datasets import make_classification

# Create an imbalanced dataset
X, y = make_classification(
    n_samples=1050, n_features=5, n_informative=3,
    weights=[0.95, 0.05], flip_y=0, random_state=42
)

print("Before SMOTE:", Counter(y))

# TODO: Import SMOTE from imblearn.over_sampling
# TODO: Create a SMOTE instance with random_state=42
# TODO: Fit and resample: X_res, y_res = smote.fit_resample(X, y)
# TODO: Print "After SMOTE:", Counter(y_res)`,
      expectedOutput: `Before SMOTE: Counter({0: 997, 1: 53})
After SMOTE: Counter({0: 997, 1: 997})`,
      solution: `from collections import Counter
import numpy as np
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

X, y = make_classification(
    n_samples=1050, n_features=5, n_informative=3,
    weights=[0.95, 0.05], flip_y=0, random_state=42
)

print("Before SMOTE:", Counter(y))

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print("After SMOTE:", Counter(y_res))`,
      hints: [
        'from imblearn.over_sampling import SMOTE',
        'smote = SMOTE(random_state=42)',
        'X_res, y_res = smote.fit_resample(X, y)',
        'Counter(y_res) shows the class distribution after resampling'
      ],
      testKeywords: ['SMOTE', 'fit_resample', 'Counter']
    },
    project: {
      title: 'Credit Card Fraud Detection with SMOTE',
      description: 'Build a complete fraud detection pipeline. Compare model performance WITH and WITHOUT SMOTE using F1 score and classification report. This directly mirrors real-world imbalanced classification work.',
      pseudoCode: `1. Create a synthetic fraud dataset (1% positive class)
2. Split into train/test (stratified)
3. Train RandomForest WITHOUT SMOTE, evaluate F1
4. Apply SMOTE to training set only
5. Train RandomForest WITH SMOTE, evaluate F1
6. Compare results`,
      starterCode: `from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

# Synthetic fraud dataset: 1% fraud
X, y = make_classification(
    n_samples=5000, n_features=10, n_informative=5,
    weights=[0.99, 0.01], flip_y=0, random_state=42
)

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- WITHOUT SMOTE ---
# TODO: Train RandomForestClassifier(n_estimators=100, random_state=42) on X_train, y_train
# TODO: Predict on X_test
# TODO: Print F1 score (average='binary') and label

# --- WITH SMOTE ---
# TODO: Apply SMOTE to X_train, y_train → X_res, y_res
# TODO: Train RandomForestClassifier on X_res, y_res
# TODO: Predict on X_test
# TODO: Print F1 score and classification_report`,
      expectedOutput: `F1 Without SMOTE:`,
      solution: `from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

X, y = make_classification(
    n_samples=5000, n_features=10, n_informative=5,
    weights=[0.99, 0.01], flip_y=0, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Without SMOTE
rf1 = RandomForestClassifier(n_estimators=100, random_state=42)
rf1.fit(X_train, y_train)
pred1 = rf1.predict(X_test)
print("F1 Without SMOTE:", f1_score(y_test, pred1, average='binary'))

# With SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
print("Training set after SMOTE:", Counter(y_res))

rf2 = RandomForestClassifier(n_estimators=100, random_state=42)
rf2.fit(X_res, y_res)
pred2 = rf2.predict(X_test)
print("F1 With SMOTE:", f1_score(y_test, pred2, average='binary'))
print("\\nDetailed Report:")
print(classification_report(y_test, pred2, target_names=['Legit', 'Fraud']))`,
      hints: [
        'Apply SMOTE ONLY to training data — never to test data (that would be leakage)',
        'f1_score(y_test, pred, average=\'binary\') for binary classification',
        'classification_report shows precision, recall, F1 per class',
        'The F1 should improve significantly with SMOTE'
      ],
      testKeywords: ['SMOTE', 'f1_score', 'classification_report', 'fit_resample', 'RandomForestClassifier']
    },
    assignment: {
      title: 'Compare All Resampling Strategies',
      description: 'Systematically compare four strategies for handling class imbalance: No resampling, Random Oversampling, SMOTE, and class_weight="balanced". Report F1 score for each and determine the winner.',
      pseudoCode: `1. Create imbalanced dataset (5% positive)
2. Split train/test (stratified)
3. For each strategy: [None, RandomOverSampler, SMOTE, class_weight='balanced']
   a. Apply resampling (or set class_weight)
   b. Train LogisticRegression
   c. Record F1 score
4. Print comparison table`,
      starterCode: `from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE, RandomOverSampler

X, y = make_classification(
    n_samples=2000, n_features=8, n_informative=4,
    weights=[0.95, 0.05], flip_y=0, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

results = {}

# TODO 1: No resampling — train LogisticRegression(max_iter=500) and record F1
# TODO 2: RandomOverSampler — resample train, train model, record F1
# TODO 3: SMOTE — resample train, train model, record F1
# TODO 4: class_weight='balanced' — train LogisticRegression(class_weight='balanced'), record F1

# TODO: Print results
for strategy, score in results.items():
    print(f"{strategy:30s}: F1 = {score:.4f}")`,
      expectedOutput: `No Resampling`,
      solution: `from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE, RandomOverSampler

X, y = make_classification(
    n_samples=2000, n_features=8, n_informative=4,
    weights=[0.95, 0.05], flip_y=0, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

results = {}

# 1. No resampling
lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train, y_train)
results['No Resampling'] = f1_score(y_test, lr.predict(X_test))

# 2. Random Oversampling
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X_train, y_train)
lr2 = LogisticRegression(max_iter=500, random_state=42)
lr2.fit(X_ros, y_ros)
results['Random Oversampling'] = f1_score(y_test, lr2.predict(X_test))

# 3. SMOTE
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_train, y_train)
lr3 = LogisticRegression(max_iter=500, random_state=42)
lr3.fit(X_sm, y_sm)
results['SMOTE'] = f1_score(y_test, lr3.predict(X_test))

# 4. Class weights
lr4 = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
lr4.fit(X_train, y_train)
results['class_weight=balanced'] = f1_score(y_test, lr4.predict(X_test))

for strategy, score in results.items():
    print(f"{strategy:30s}: F1 = {score:.4f}")`,
      hints: [
        'RandomOverSampler is in imblearn.over_sampling',
        'Apply resampling only to training data',
        'For class_weight, pass it directly to LogisticRegression(class_weight=\'balanced\')',
        'All four models are evaluated on the same unmodified X_test, y_test'
      ],
      testKeywords: ['SMOTE', 'RandomOverSampler', 'class_weight', 'f1_score', 'fit_resample']
    }
  },
  interviewQuestions: [
    {
      question: 'Why should you NEVER apply SMOTE before splitting into train/test sets?',
      answer: 'Applying SMOTE before splitting causes data leakage. SMOTE generates synthetic points by interpolating between real minority samples. If those real samples appear in both the synthetic training data and the test set, the model has effectively "seen" the test set. The correct order is: split → apply SMOTE to training set only → train → evaluate on untouched test set. In scikit-learn pipelines, use imblearn\'s Pipeline to ensure SMOTE runs only during fit().',
      companyTags: ['Stripe', 'PayPal', 'Google'],
      difficulty: 'Hard'
    },
    {
      question: 'What is the precision-recall tradeoff and how do you choose the optimal classification threshold?',
      answer: 'By default, classifiers predict the positive class when P(y=1|x) > 0.5. Lowering this threshold increases Recall (fewer FNs) but reduces Precision (more FPs). The optimal threshold depends on the business cost of each error type. For fraud: a FN (missed fraud) costs $500, a FP (wrongly flagged) costs $5 in investigation time — you\'d lower the threshold significantly. Use the Precision-Recall curve to visualize all thresholds and pick the one that maximizes your business metric (e.g., F1 score or a custom cost function).',
      companyTags: ['Amazon', 'MasterCard', 'Visa'],
      difficulty: 'Medium'
    },
    {
      question: 'Explain when you would use class_weight="balanced" vs SMOTE.',
      answer: 'class_weight="balanced" modifies the loss function — it is computationally free, prevents data leakage by construction, and works inside standard sklearn Pipelines with no extra code. Use it as the first thing to try. SMOTE is a data-level technique that creates new training samples. It can improve performance further when class_weight is insufficient, especially with tree-based models that do not have a loss function (splitting criterion is not easily weighted). SMOTE also gives you more training samples, which matters for deep learning. For XGBoost, use the scale_pos_weight parameter instead of SMOTE.',
      companyTags: ['Uber', 'Lyft', 'DoorDash'],
      difficulty: 'Hard'
    }
  ]
};
