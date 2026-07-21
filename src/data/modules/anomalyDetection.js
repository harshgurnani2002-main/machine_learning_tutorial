export const anomalyDetection = {
    id: 'anomaly-detection',
    title: 'Anomaly & Outlier Detection',
    category: 'Unsupervised Learning',
    description: 'Detect rare events using Isolation Forest, Local Outlier Factor, and One-Class SVM — with deep dives into the math behind path-length scoring and density deviation.',
    formula: 's(x, n) = 2^{-\\frac{E[h(x)]}{c(n)}}',
    interactiveSummary: 'Use the interactive Isolation Forest simulator to see how random partitioning isolates outliers faster than normal points. Adjust the contamination parameter to dynamically flag anomalies and observe the anomaly score distribution.',
    simulatorId: 'anomaly-simulator',
    theory: `### Anomaly & Outlier Detection

#### What is Anomaly Detection?
Anomaly detection (also called outlier detection) identifies observations that deviate significantly from the expected behavior. Unlike classification, it is typically an **unsupervised** problem — you often have no labeled anomalies to train on.

Three types of anomalies exist:
1. **Point Anomaly**: A single data instance is anomalous (e.g., a transaction of $50,000 when average is $100).
2. **Contextual Anomaly**: Normal in one context, anomalous in another (e.g., a temperature of 35°C is normal in summer, anomalous in winter).
3. **Collective Anomaly**: A sequence of data points is anomalous together (e.g., a sudden pattern of clicks in web logs indicating a bot).

Real-world applications:
- Network intrusion detection
- Credit card fraud
- Manufacturing defect detection
- Medical monitoring (patient deterioration alerts)
- Predictive maintenance

#### Algorithm 1: Isolation Forest

**Intuition**: Anomalies are few and different. If you make random splits on feature values, you need very **few splits** to isolate an anomaly from the rest of the data. Normal points are surrounded by many others and require **many splits** to isolate.

**How it works**:
1. Build $T$ isolation trees. Each tree recursively randomly selects a feature $q$ and a random split value $p$ (between the feature's min and max).
2. A sample is isolated when it reaches a leaf node. Its **path length** $h(x)$ is the number of edges traversed.
3. Anomalies have short average path lengths across all trees.

**The Anomaly Score**:
The score is normalized by $c(n)$, the average path length of an unsuccessful Binary Search Tree search for sample size $n$:

$$c(n) = 2H(n-1) - \\frac{2(n-1)}{n}$$

where $H(i)$ is the harmonic number $\\sum_{k=1}^{i} \\frac{1}{k}$.

The final anomaly score for observation $x$ with sample size $n$:

$$s(x, n) = 2^{-\\frac{E[h(x)]}{c(n)}}$$

Interpretation:
- $s(x, n) \\to 1$: the instance is an anomaly (very short path length)
- $s(x, n) \\to 0.5$: the instance is normal
- $s(x, n) \\to 0$: the instance is deeply embedded in the data

**Advantages**:
- Linear time complexity $O(n \\log n)$ — extremely fast
- Works well in high dimensions (unlike density-based methods)
- No distance or density computation needed

**Key Parameter**: **contamination** — the expected proportion of anomalies in the dataset. Typical values: 0.01 to 0.1.

#### Algorithm 2: Local Outlier Factor (LOF)

LOF is a density-based method. The key insight: a point is an outlier if its local density is significantly lower than its neighbors' densities.

**Step 1 — k-distance**: The distance from point $p$ to its $k$-th nearest neighbor.

$$k\\text{-dist}(p) = \\text{distance to the } k\\text{-th nearest neighbor of } p$$

**Step 2 — Reachability Distance**: Smooths out statistical fluctuations:

$$\\text{reach-dist}_k(p, o) = \\max(k\\text{-dist}(o), d(p, o))$$

**Step 3 — Local Reachability Density (LRD)**:

$$\\text{lrd}_k(p) = \\left( \\frac{\\sum_{o \\in N_k(p)} \\text{reach-dist}_k(p, o)}{|N_k(p)|} \\right)^{-1}$$

**Step 4 — LOF Score**: The ratio of average neighbor density to the point's own density:

$$\\text{LOF}_k(p) = \\frac{\\sum_{o \\in N_k(p)} \\frac{\\text{lrd}_k(o)}{\\text{lrd}_k(p)}}{|N_k(p)|}$$

Interpretation:
- $\\text{LOF} \\approx 1$: similar density as neighbors → normal
- $\\text{LOF} \\gg 1$: much lower density than neighbors → outlier

**Advantages**:
- Detects local outliers that global methods miss (e.g., a dense cluster in a sparser region)
- The LOF score itself quantifies the degree of outlierness

**Disadvantages**:
- $O(n^2)$ complexity — slow on large datasets
- Sensitive to the choice of $k$

#### Algorithm 3: One-Class SVM
One-Class SVM learns a decision boundary around the "normal" class in kernel space. Any point outside the boundary is flagged as an anomaly. Useful when you have clean training data of only normal instances.

$$\\min_{w, \\xi, \\rho} \\frac{1}{2}\\|w\\|^2 + \\frac{1}{\\nu n} \\sum_i \\xi_i - \\rho$$

The $\\nu$ parameter controls the upper bound on the fraction of outliers and the lower bound on the fraction of support vectors.

#### Algorithm 4: Statistical Methods
For univariate data:
- **Z-Score**: Flag points where $|z| > 3$ (more than 3 standard deviations from mean)
$$z = \\frac{x - \\mu}{\\sigma}$$
- **IQR Method**: Flag points below $Q1 - 1.5 \\times IQR$ or above $Q3 + 1.5 \\times IQR$

These are fast and interpretable but fail in high dimensions.

#### Evaluation Without Labels
Anomaly detection is often unsupervised, making evaluation hard. If you have labeled anomalies:
- Use **Precision, Recall, F1** on the positive (anomaly) class
- Use **ROC-AUC** across contamination thresholds

Without labels:
- Expert review of flagged instances
- Inject known anomalies (synthetic validation set)
- Monitor drift in anomaly scores over time

#### Choosing the Right Algorithm
| Algorithm | Speed | High Dimensions | Interpretability | Best For |
|---|---|---|---|---|
| Isolation Forest | Fast | ✅ Excellent | Medium | General-purpose anomaly detection |
| LOF | Slow | ❌ Poor | High | Local anomalies, small datasets |
| One-Class SVM | Medium | ✅ Good | Low | Clean training data of normals |
| Z-Score / IQR | Very Fast | ❌ Univariate only | High | Quick EDA, univariate screening |

#### Python Implementation

\`\`\`python
from sklearn.ensemble import IsolationForest
import numpy as np

X = np.random.randn(200, 2)
clf = IsolationForest(contamination=0.1, random_state=42)
preds = clf.fit_predict(X)
print(f"Anomalies detected: {np.sum(preds == -1)}")
\`\`\`
`,
    quiz: [
        {
            id: 'anom_q1',
            question: 'How does Isolation Forest identify anomalies compared to normal points?',
            options: [
                'Anomalies have longer average path lengths in the isolation trees',
                'Anomalies have shorter average path lengths in the isolation trees',
                'Anomalies are identified by their high local density',
                'Anomalies are points outside the SVM decision boundary'
            ],
            correctAnswer: 'Anomalies have shorter average path lengths in the isolation trees',
            explanation: 'Anomalies are "few and different" — they are isolated from the bulk of data with very few random splits. Normal points are surrounded by many similar points and require many splits to isolate, giving them longer path lengths.'
        },
        {
            id: 'anom_q2',
            question: 'What does an Isolation Forest anomaly score of s ≈ 1.0 indicate?',
            options: [
                'The point is deeply embedded in the data and is definitely normal',
                'The point is highly anomalous — it was isolated very quickly',
                'The contamination parameter is set too high',
                'The model has not converged yet'
            ],
            correctAnswer: 'The point is highly anomalous — it was isolated very quickly',
            explanation: 'The score s(x,n) = 2^(-E[h(x)]/c(n)). When path length E[h(x)] is very small (anomaly), the exponent approaches 0, so s approaches 1. Normal points have s ≈ 0.5.'
        },
        {
            id: 'anom_q3',
            question: 'What does the Local Outlier Factor (LOF) score measure?',
            options: [
                'The absolute distance from the point to the dataset centroid',
                'The ratio of the average density of a point\'s neighbors to its own local density',
                'The number of random splits needed to isolate the point',
                'The probability of the point under a Gaussian distribution'
            ],
            correctAnswer: 'The ratio of the average density of a point\'s neighbors to its own local density',
            explanation: 'LOF compares a point\'s local density to its neighbors\'. If LOF >> 1, the point is in a sparse region relative to its neighbors — it is an outlier. LOF ≈ 1 means similar density, indicating a normal point.'
        },
        {
            id: 'anom_q4',
            question: 'What is a "contextual anomaly"?',
            options: [
                'An anomaly that only exists in the context of a single feature',
                'A point that is anomalous in one context (e.g., season) but normal in another',
                'An anomaly that is only detectable with labeled data',
                'A cluster of normal-looking points that is collectively anomalous'
            ],
            correctAnswer: 'A point that is anomalous in one context (e.g., season) but normal in another',
            explanation: 'A temperature of 40°C is anomalous in December (winter context) but perfectly normal in July (summer context). Contextual anomalies require domain context to detect — pure statistical methods often miss them.'
        },
        {
            id: 'anom_q5',
            question: 'What is the main computational disadvantage of the Local Outlier Factor algorithm?',
            options: [
                'It cannot handle high-dimensional data and scales as O(n²)',
                'It requires labeled training data',
                'It can only detect point anomalies, not collective anomalies',
                'It is not available in scikit-learn'
            ],
            correctAnswer: 'It cannot handle high-dimensional data and scales as O(n²)',
            explanation: 'LOF requires computing pairwise distances to find k-nearest neighbors, which is O(n²) in time and memory. It also suffers from the curse of dimensionality — distances become meaningless in high dimensions. Isolation Forest is preferred for large, high-dimensional datasets.'
        },
        {
            id: 'anom_q6',
            question: 'When is the Z-Score method sufficient for anomaly detection, and when does it fail?',
            options: [
                'Z-Score works for any dataset; Isolation Forest is just an approximation',
                'Z-Score works well for univariate, normally distributed data but fails in high dimensions and non-normal distributions',
                'Z-Score fails because it requires labeled anomalies',
                'Z-Score works for time series but not for tabular data'
            ],
            correctAnswer: 'Z-Score works well for univariate, normally distributed data but fails in high dimensions and non-normal distributions',
            explanation: 'Z-Score assumes data is approximately Gaussian and only works on one feature at a time. It cannot detect multivariate outliers (a point that is normal on each individual feature but anomalous when features are considered together). For real-world high-dimensional data, Isolation Forest or LOF are much more powerful.'
        }
    ],
    coding: {
        tutorial: {
            title: 'Isolation Forest on Synthetic Data',
            description: 'Apply Isolation Forest to detect outliers in a synthetic dataset containing a normal cluster and scattered anomalies. Visualize the predictions using scikit-learn.',
            pseudoCode: `1. Generate 200 normal points (Gaussian cluster) + 20 outliers
2. Stack into dataset X
3. Fit IsolationForest(contamination=0.1, random_state=42)
4. Predict: 1 = normal, -1 = anomaly
5. Count detected anomalies`,
            starterCode: `import numpy as np
from sklearn.ensemble import IsolationForest

np.random.seed(42)

# Normal cluster: 200 points around (0, 0)
X_normal = np.random.randn(200, 2)

# Outliers: 20 points scattered far from cluster
X_outliers = np.random.uniform(low=-6, high=6, size=(20, 2))

# Combine
X = np.vstack([X_normal, X_outliers])

# TODO: Create IsolationForest with contamination=0.1 and random_state=42
# model = ...

# TODO: Fit the model on X

# TODO: Get predictions (predict returns 1=normal, -1=anomaly)
# preds = ...

# TODO: Count anomalies
# n_anomalies = ...
# print("Detected anomalies:", n_anomalies)

# TODO: Print anomaly scores for first 5 outlier points
# scores = model.decision_function(X)
# print("Scores of first 5 outliers:", scores[200:205])`,
            expectedOutput: `Detected anomalies: 22`,
            solution: `import numpy as np
from sklearn.ensemble import IsolationForest

np.random.seed(42)
X_normal = np.random.randn(200, 2)
X_outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
X = np.vstack([X_normal, X_outliers])

model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X)
preds = model.predict(X)

n_anomalies = (preds == -1).sum()
print("Detected anomalies:", n_anomalies)

scores = model.decision_function(X)
print("Scores of first 5 outliers:", scores[200:205])`,
            hints: [
                'IsolationForest(contamination=0.1, random_state=42)',
                'model.predict() returns 1 for normal, -1 for anomaly',
                '(preds == -1).sum() counts anomalies',
                'model.decision_function() returns raw anomaly scores (negative = more anomalous)'
            ],
            testKeywords: ['IsolationForest', 'contamination', 'predict', 'decision_function']
        },
        project: {
            title: 'Network Intrusion Detection',
            description: 'Simulate a network intrusion detection scenario. Train an Isolation Forest on "normal" network traffic, then detect anomalous connection patterns. Compare with Local Outlier Factor.',
            pseudoCode: `1. Generate "normal" network traffic (duration, packet_size, n_connections)
2. Generate "attack" traffic with unusual patterns
3. Train IsolationForest on training data (normal only)
4. Evaluate on combined test set using precision/recall
5. Repeat with LocalOutlierFactor
6. Compare performance`,
            starterCode: `import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report

np.random.seed(42)

# Normal traffic: typical connections
n_normal = 500
normal_traffic = np.column_stack([
    np.random.normal(50, 10, n_normal),   # avg duration (ms)
    np.random.normal(1500, 200, n_normal), # packet size (bytes)
    np.random.normal(10, 3, n_normal)      # connections/sec
])

# Attack traffic: unusual patterns (high connections, weird packet sizes)
n_attacks = 50
attack_traffic = np.column_stack([
    np.random.uniform(100, 500, n_attacks),  # very long duration
    np.random.uniform(60, 100, n_attacks),   # tiny packets (port scan)
    np.random.uniform(100, 500, n_attacks)   # flood of connections
])

# Training data: only normal
X_train = normal_traffic[:400]

# Test data: mix of normal and attacks
X_test = np.vstack([normal_traffic[400:], attack_traffic])
y_test = np.array([1]*100 + [-1]*n_attacks)  # 1=normal, -1=attack

# --- Isolation Forest ---
# TODO: Fit IsolationForest(contamination=0.1, random_state=42) on X_train
# TODO: Predict on X_test
# TODO: Print classification_report

# --- Local Outlier Factor ---
# Note: LOF with novelty=True for out-of-sample prediction
# TODO: Fit LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True) on X_train
# TODO: Predict on X_test
# TODO: Print classification_report`,
            expectedOutput: `Isolation Forest Report:`,
            solution: `import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report

np.random.seed(42)
n_normal = 500
normal_traffic = np.column_stack([
    np.random.normal(50, 10, n_normal),
    np.random.normal(1500, 200, n_normal),
    np.random.normal(10, 3, n_normal)
])
n_attacks = 50
attack_traffic = np.column_stack([
    np.random.uniform(100, 500, n_attacks),
    np.random.uniform(60, 100, n_attacks),
    np.random.uniform(100, 500, n_attacks)
])

X_train = normal_traffic[:400]
X_test = np.vstack([normal_traffic[400:], attack_traffic])
y_test = np.array([1]*100 + [-1]*n_attacks)

# Isolation Forest
iso = IsolationForest(contamination=0.1, random_state=42)
iso.fit(X_train)
iso_preds = iso.predict(X_test)
print("Isolation Forest Report:")
print(classification_report(y_test, iso_preds, target_names=['Normal', 'Attack']))

# Local Outlier Factor (novelty=True for predict on new data)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
lof.fit(X_train)
lof_preds = lof.predict(X_test)
print("Local Outlier Factor Report:")
print(classification_report(y_test, lof_preds, target_names=['Normal', 'Attack']))`,
            hints: [
                'LOF requires novelty=True to predict on new test data after fitting',
                'Both models use 1=normal, -1=anomaly convention in sklearn',
                'classification_report works fine with these -1/1 labels',
                'The test set combines held-out normal + all attack samples'
            ],
            testKeywords: ['IsolationForest', 'LocalOutlierFactor', 'novelty', 'classification_report', 'contamination']
        },
        assignment: {
            title: 'Tune Isolation Forest Contamination Parameter',
            description: 'Systematically evaluate how different contamination values affect detection performance. Plot precision, recall, and F1 score as a function of contamination to find the optimal operating point for a given dataset.',
            pseudoCode: `1. Generate dataset: 900 normal + 100 anomalies
2. For contamination in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25]:
   a. Fit IsolationForest
   b. Compute F1, Precision, Recall for anomaly class
3. Print results table
4. Identify optimal contamination`,
            starterCode: `import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

np.random.seed(42)

# Dataset: normal cluster + outliers
X_normal = np.random.randn(900, 3)
X_outliers = np.random.uniform(low=-4, high=4, size=(100, 3)) + 3
X = np.vstack([X_normal, X_outliers])

# True labels: -1 = anomaly, 1 = normal (sklearn convention)
y_true = np.array([1]*900 + [-1]*100)

contamination_values = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25]

print(f"{'Contamination':>15} | {'Precision':>10} | {'Recall':>8} | {'F1':>8}")
print("-" * 50)

for c in contamination_values:
    # TODO: Fit IsolationForest with contamination=c, random_state=42
    # TODO: Predict on X
    # TODO: Compute precision, recall, F1 for the anomaly class (pos_label=-1)
    # TODO: Print formatted row
    pass

# TODO: Print which contamination gives the best F1`,
            expectedOutput: `Contamination`,
            solution: `import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

np.random.seed(42)
X_normal = np.random.randn(900, 3)
X_outliers = np.random.uniform(low=-4, high=4, size=(100, 3)) + 3
X = np.vstack([X_normal, X_outliers])
y_true = np.array([1]*900 + [-1]*100)

contamination_values = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25]

print(f"{'Contamination':>15} | {'Precision':>10} | {'Recall':>8} | {'F1':>8}")
print("-" * 50)

best_f1, best_c = 0, 0
for c in contamination_values:
    model = IsolationForest(contamination=c, random_state=42)
    model.fit(X)
    preds = model.predict(X)
    p = precision_score(y_true, preds, pos_label=-1, zero_division=0)
    r = recall_score(y_true, preds, pos_label=-1, zero_division=0)
    f = f1_score(y_true, preds, pos_label=-1, zero_division=0)
    print(f"{c:>15.2f} | {p:>10.4f} | {r:>8.4f} | {f:>8.4f}")
    if f > best_f1:
        best_f1, best_c = f, c

print(f"\nBest contamination: {best_c} with F1 = {best_f1:.4f}")`,
            hints: [
                'precision_score(..., pos_label=-1) tells sklearn the anomaly class is -1',
                'Loop over contamination values and store results',
                'Track best F1 with a running max variable',
                'The true contamination is 100/1000 = 0.10 — see if the search finds it!'
            ],
            testKeywords: ['IsolationForest', 'contamination', 'f1_score', 'precision_score', 'recall_score', 'pos_label']
        }
    },
    interviewQuestions: [
        {
            question: 'How does Isolation Forest work, and why is it particularly efficient?',
            answer: 'Isolation Forest builds an ensemble of random trees where each tree recursively partitions the data by randomly selecting a feature and a random split value. Anomalies are isolated in fewer steps because they are sparse and far from other points — their average path length across trees is short. The anomaly score normalizes path length against the expected path length of a BST, giving a score between 0 and 1. It is efficient because: (1) it uses subsampling (typically 256 points per tree), making each tree fast to build; (2) path length is cheap to compute; (3) it scales to millions of points. Time complexity is O(t * ψ * log ψ) where t=trees and ψ=subsample size.',
            companyTags: ['Google', 'Palantir', 'Databricks'],
            difficulty: 'Hard'
        },
        {
            question: 'What is the difference between a global outlier and a local outlier? Which algorithm handles each better?',
            answer: 'A global outlier is far from all data (e.g., a salary of $10M in a dataset where most are $50K-$100K). A local outlier is normal in the global context but unusual relative to its immediate neighbors (e.g., a data point that is slightly more sparse than its local cluster). Isolation Forest detects global outliers well due to path length. LOF detects local outliers by comparing local densities. In practice, use Isolation Forest first for speed, then LOF if you suspect the data has multi-density clusters with local anomalies.',
            companyTags: ['Microsoft', 'IBM', 'Cisco'],
            difficulty: 'Medium'
        },
        {
            question: 'How would you evaluate an anomaly detection model when you have no labeled anomalies?',
            answer: 'Several strategies exist: (1) Inject synthetic anomalies with known ground truth — measure precision/recall on those. (2) Expert review: have domain experts label a sample of flagged instances. (3) Holdout validation: if you have a small set of confirmed anomalies, hold them out for evaluation. (4) Compare anomaly scores over time — scores should be stable for normal periods and spike during known incidents. (5) Business metrics: measure downstream impact (fraud losses caught, manufacturing defects identified). Pure unsupervised evaluation is inherently limited — the contamination parameter choice reflects our prior knowledge about the expected anomaly rate.',
            companyTags: ['Splunk', 'Datadog', 'CrowdStrike'],
            difficulty: 'Hard'
        }
    ]
};
