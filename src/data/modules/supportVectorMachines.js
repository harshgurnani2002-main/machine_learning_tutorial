export const supportVectorMachines = {
    id: 'support-vector-machines',
    title: 'Support Vector Machines',
    category: 'Supervised Learning',
    description: 'Find optimal margin hyperplanes to strictly separate classes in high-dimensional space.',
    formula: '\\min \\frac{1}{2}\\|w\\|^2 + C \\sum \\xi_i',
    interactiveSummary: "This interactive simulator lets you explore the core mechanics of Support Vector Machines by placing data points directly on the canvas. The shaded band between the two dashed lines is the SVM margin — the 'empty street' the algorithm tries to maximize — and the solid points highlighted with a ring are the support vectors, the only training examples that actually determine where the boundary sits. Use the Kernel selector to switch between Linear (straight hyperplane), RBF (radially curved boundary), and Polynomial modes to see how the kernel trick implicitly warps the feature space, allowing SVM to carve out complex, non-linear regions. Drag the C slider to watch the tradeoff live: a high C enforces a narrow, rigid margin that strictly classifies every point, while a low C allows a wider, softer margin that ignores a few violations in exchange for better generalization. Understanding this margin-vs-misclassification tradeoff is the heart of why SVMs are so powerful on small, high-dimensional datasets like medical diagnostics and text classification.",
    theory: `### Support Vector Machines (SVM) Theory

#### What is it?
Support Vector Machines (SVM) are a set of highly powerful, robust supervised machine learning models.
They are used primarily for complex classification tasks.
However, they can also be adapted for regression (Support Vector Regression, SVR) and outlier detection. 

At its core, an SVM constructs a single hyperplane—or a set of hyperplanes.
This happens in a high-dimensional or infinite-dimensional feature space.
The goal is to strictly separate the data into distinct, non-overlapping classes. 

#### Why do we need it?
Many linear classification algorithms (like Perceptrons) just search for *any* boundary that separates classes. 
If the classes are easily separable, an infinite number of such boundaries exist. 

SVM is conceptually superior because it doesn\\'t just find *any* boundary.
It finds the *optimal* boundary. 
This optimal hyperplane is the one with the maximum margin (distance) between the classes. 

By focusing entirely on maximizing this margin, SVMs achieve mathematically proven bounds.
These bounds guarantee lower generalization error, making them incredibly resistant to overfitting.
This is especially true in high-dimensional spaces.

#### How does it work?
The operation of an SVM can be broken down into several geometric and mathematical concepts:

1. **The Hyperplane**: 
In a 2D space, a hyperplane is simply a 1D line. 
In a 3D space, it is a 2D flat plane. 
In $N$ dimensions, it is an $(N-1)$-dimensional subspace. 
SVM searches for the hyperplane that perfectly separates the classes.

2. **The Margin**: 
The margin is the "empty street" or clear distance between the separating hyperplane.
It is the distance to the closest training data points from either class. 
The SVM algorithm seeks to maximize the width of this street.

3. **Support Vectors**: 
These are the critical, pivotal data points that lie exactly on the edges of the margin. 
They "support" the entire structural boundary of the model. 
If you removed all other non-support training points, the hyperplane would not move. 
The model\\'s complexity is defined by the number of support vectors, not the number of features.

4. **The Kernel Trick**: 
Real-world data is rarely linearly separable. 
Instead of giving up, SVM implicitly maps the input features into a much higher-dimensional space.
In this new space, a linear hyperplane *can* separate them. 
It does this efficiently using a mathematical shortcut called a "Kernel".

#### The Math Behind It
The equation of a separating hyperplane is defined as:
$$w^T x + b = 0$$
where $w$ is the normal vector to the hyperplane.

The geometric distance between the hyperplanes that bound the margin is exactly $\\frac{2}{\\|w\\|}$. 
The bounding hyperplanes are defined as $w^T x + b = 1$ and $w^T x + b = -1$.
Maximizing this margin is mathematically equivalent to minimizing $\\frac{1}{2}\\|w\\|^2$.

**Hard Margin SVM**:
For perfectly separable data, we formulate a constrained optimization problem:

$$
\\min_{w,b} \\frac{1}{2}\\|w\\|^2
$$

Subject to the strict classification constraint: 
$$y_i(w^T x_i + b) \\ge 1 \\quad \\text{for all } i=1, \\dots, n$$

**Soft Margin SVM**:
Real data contains noise and overlapping classes. 
We introduce positive slack variables $\\xi_i$ to allow some points to violate the margin:

$$
\\min_{w,b} \\frac{1}{2}\\|w\\|^2 + C \\sum_{i=1}^n \\xi_i
$$

Subject to: 
$$y_i(w^T x_i + b) \\ge 1 - \\xi_i \\quad \\text{and} \\quad \\xi_i \\ge 0$$

- The hyperparameter $C$ dictates the degree of regularization. 
- **Large $C$**: Heavy penalty for misclassification. Results in a narrow "hard" margin (high variance).
- **Small $C$**: Low penalty. Results in a wide "soft" margin that ignores noisy points (high bias).

#### Worked Example
Imagine you are a factory quality inspector sorting apples (Class +1) and limes (Class -1).
They are traveling on a conveyor belt, measured by weight and color hue.

- A **Linear SVM** will mathematically draw a straight line directly through the middle.
  It aims for the empty space separating the heavy, red apples from the lighter, green limes.
  It ensures the line is as far away from the closest apple and closest lime as possible.

- If a few mutated, red limes (outliers) are mixed into the apple cluster, a strict line is impossible. 
  A **Soft Margin SVM** (with an appropriate $C$ value) will purposely ignore those mutated limes.
  It allows them to fall on the wrong side of the line.
  This maintains a wide, robust, general boundary for the remaining 99% of the fruits.

- If the apples form a ring completely surrounding the limes in the center, no line can separate them. 
  The SVM uses an **RBF Kernel** to project the data into 3D space.
  This effectively morphs the 2D plane into a bowl shape.
  The apples are pushed high on the rim, and the limes sink to the bottom. 
  A flat 2D plane can now slice horizontally through the bowl, perfectly separating them.

#### Common Pitfalls
1. **Feature Scaling is Mandatory**: 
SVMs are exceptionally sensitive to unscaled features because they strictly optimize geometric distance. 
If one feature ranges from 0-1 and another from 0-1,000,000, the smaller feature is ignored.
Data must be standardized before training.

2. **Computational Cost**: 
The time complexity for training an SVM with non-linear kernels is generally $O(n^2)$ to $O(n^3)$.
This scales poorly with respect to the number of samples. 
This makes SVMs computationally prohibitive for massive datasets (e.g., millions of rows).

3. **Interpretability**: 
While Linear SVMs provide interpretable weights, non-linear SVMs are effectively "black boxes." 
Extracting why a specific prediction was made in an infinite-dimensional space is exceedingly difficult.

4. **Tuning Sensitivity**: 
The performance of an RBF SVM is highly sensitive to the proper tuning of $C$ and $\\gamma$ (gamma). 
Without a thorough grid search, performance can be abysmal.

#### When to Use vs Not Use

**Use when:**
- There is a clear, identifiable margin of separation between classes.
- The dataset has extremely high dimensionality. 
- The number of dimensions exceeds the number of samples (e.g., genomics, text classification).
- You are working with small-to-medium sized datasets and need maximum accuracy.

**Not Use when:**
- The dataset is massive (millions of rows). The training time will be unacceptably long.
- The dataset is highly noisy with heavily overlapping target classes.
- You absolutely require probabilistic explanations for predictions. 
- Standard SVMs output geometric distances to the boundary, not true probabilities.

#### Key Takeaways
- **Margin Maximization**: SVMs find the mathematically widest boundary between classes.
- **Support Vectors**: The model discards the vast majority of training data after training.
- **The Kernel Trick**: Efficiently solves complex, non-linear classification boundaries.
- **Preprocessing**: Standardization and hyperparameter tuning are absolute prerequisites.

#### Python Implementation

\`\`\`python
from sklearn.svm import SVC
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=50, n_features=2, random_state=42)
clf = SVC(kernel="rbf", C=1.0)
clf.fit(X, y)
print(f"Support vectors: {len(clf.support_vectors_)}")
\`\`\`
`,
    simulatorId: 'svm',
    quiz: [
        {
            id: 'svm_q1',
            question: 'What is the role of the C hyperparameter in a Soft-Margin SVM?',
            options: [
                'It dictates the tradeoff between maximizing the margin and minimizing classification violations.',
                'It sets the degree of the polynomial kernel.',
                'It dictates the number of support vectors allowed.',
                'It determines the learning rate of the descent.'
            ],
            correctAnswer: 'It dictates the tradeoff between maximizing the margin and minimizing classification violations.',
            explanation: 'A low C allows a wider margin with more violations (high bias). A high C strictly enforces correct classifications, risking a narrow margin (high variance).'
        },
        {
            id: 'svm_q2',
            question: 'What mathematical property allows SVMs to compute non-linear boundaries in high-dimensional space efficiently?',
            options: [
                'The Kernel Trick',
                'Gradient Descent',
                'Singular Value Decomposition',
                'Bootstrap Aggregation'
            ],
            correctAnswer: 'The Kernel Trick',
            explanation: 'The Kernel trick computes dot products in a high-dimensional feature space without explicitly transforming the data, bypassing massive computational costs.'
        },
        {
            id: 'svm_q3',
            question: 'Why is SVM particularly memory efficient during inference?',
            options: [
                'The decision boundary relies entirely on a small subset of training points (support vectors).',
                'It quantizes weights into 8-bit integers.',
                'It always uses linear operations.',
                'It stores only the mean vector of each class.'
            ],
            correctAnswer: 'The decision boundary relies entirely on a small subset of training points (support vectors).',
            explanation: 'Once trained, the vast majority of training points inside the margin are discarded. Only the support vectors are retained to define the hyperplane.'
        },
        {
            id: 'svm_q4',
            question: 'Which Kernel is considered the default/most versatile for non-linear SVMs?',
            options: ['Radial Basis Function (RBF)', 'Linear', 'Polynomial', 'Sigmoid'],
            correctAnswer: 'Radial Basis Function (RBF)',
            explanation: 'The RBF (Gaussian) kernel maps data into an infinite-dimensional space, capable of wrapping complex, non-linear boundaries effectively.'
        },
        {
            id: 'svm_q5',
            question: 'What happens to a Hard-Margin SVM if the data is not linearly separable?',
            options: [
                'The optimization problem fails to find a valid solution.',
                'It automatically switches to a soft margin.',
                'It returns the mean of all points.',
                'It randomly assigns classes.'
            ],
            correctAnswer: 'The optimization problem fails to find a valid solution.',
            explanation: 'A Hard-Margin SVM strictly forbids misclassifications. If classes overlap, no separating hyperplane exists, and the solver cannot converge.'
        },
        {
            id: 'svm_q6',
            question: 'True or False: SVM requires feature scaling.',
            options: ['True', 'False', 'Only for the Linear kernel', 'Only for regression'],
            correctAnswer: 'True',
            explanation: 'SVM tries to maximize distance (margin). If features are on different scales, the feature with the larger scale will dominate the distance calculations.'
        },
        {
            id: 'svm_q7',
            question: 'What does the gamma parameter control in an RBF Kernel?',
            options: [
                'The radius of influence of a single training example.',
                'The polynomial degree.',
                'The number of support vectors.',
                'The penalty for misclassification.'
            ],
            correctAnswer: 'The radius of influence of a single training example.',
            explanation: 'A low gamma means "far reach" (smoother boundary). A high gamma means "close reach" (squiggly boundary, highly fitted to individual points).'
        },
        {
            id: 'svm_q8',
            question: 'What is the objective function being minimized in a linear SVM?',
            options: [
                '1/2 ||w||^2',
                'Mean Squared Error',
                'Cross-Entropy Loss',
                'Gini Impurity'
            ],
            correctAnswer: '1/2 ||w||^2',
            explanation: 'Minimizing the norm of the weight vector w (1/2 ||w||^2) is mathematically equivalent to maximizing the margin 2/||w||.'
        },
        {
            id: 'svm_q9',
            question: 'What are Support Vectors?',
            options: [
                'Data points that lie closest to the decision boundary and dictate its position.',
                'The weights assigned to each feature.',
                'The error terms in the loss function.',
                'The eigenvectors of the covariance matrix.'
            ],
            correctAnswer: 'Data points that lie closest to the decision boundary and dictate its position.',
            explanation: 'Support vectors are the critical points on the edge of the margin. Removing them changes the boundary; removing non-support vectors does not.'
        },
        {
            id: 'svm_q10',
            question: 'Why are SVMs generally not recommended for very large datasets (e.g., millions of rows)?',
            options: [
                'Training time complexity is O(N^2) to O(N^3).',
                'They require infinite memory.',
                'They cannot be regularized.',
                'They only support binary classification.'
            ],
            correctAnswer: 'Training time complexity is O(N^2) to O(N^3).',
            explanation: 'The quadratic programming problem required to solve non-linear SVMs scales terribly with the number of samples.'
        }
    ],
    coding: {
        tutorial: {
            title: 'Hinge Loss Calculation',
            description: 'Calculate the unregularized hinge loss for a set of predictions.\n\n**Sklearn Equivalent:**\n`from sklearn.metrics import hinge_loss`\n`loss = hinge_loss(y_true, decision_values)`',
            pseudoCode: 'function hinge_loss(y_true, y_pred):\n  total_loss = 0\n  for i in range(len(y_true)):\n    loss = max(0, 1 - y_true[i] * y_pred[i])\n    total_loss += loss\n  return total_loss / len(y_true)',
            starterCode: `import numpy as np

def compute_hinge_loss(y_true, y_pred_score):
    # y_true: array of +1 or -1
    # y_pred_score: array of continuous decision outputs
    # TODO: Calculate average hinge loss: max(0, 1 - y * f(x))
    loss = 0.0
    return loss

y = np.array([1, 1, -1])
scores = np.array([2.5, 0.2, -1.5])
print("Hinge Loss:", compute_hinge_loss(y, scores))`,
            expectedOutput: 'Hinge Loss: 0.26666666666666666',
            solution: `import numpy as np

def compute_hinge_loss(y_true, y_pred_score):
    margins = y_true * y_pred_score
    losses = np.maximum(0, 1 - margins)
    return np.mean(losses)

y = np.array([1, 1, -1])
scores = np.array([2.5, 0.2, -1.5])
print("Hinge Loss:", compute_hinge_loss(y, scores))`,
            hints: ['Multiply true labels by predicted scores to get margins.', 'Use np.maximum to clip negative values to 0.', 'Return the mean of the losses.'],
            testKeywords: ['np.maximum', 'mean']
        },
        project: {
            title: 'Breast Cancer Detection with SVM',
            description: 'Build a Support Vector Classifier to detect whether a tumor is malignant (1) or benign (0) using 6 cell-measurement features from a breast cancer biopsy dataset: mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, and mean_compactness. Because these features span very different numeric ranges (area ~300–2000 vs. smoothness ~0.05–0.16), we wrap the SVC in a Pipeline with StandardScaler — a mandatory step for SVM.',
            pseudoCode: 'Load breast cancer biopsy data (mean_radius, texture, perimeter, area, smoothness, compactness)\npipeline = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, random_state=42))\npipeline.fit(X_train, y_train)\ntest_cell = [[14.5, 18.2, 92.0, 640.0, 0.098, 0.105]]\npred = pipeline.predict(test_cell)\nprint(f"Tumor prediction (1=malignant, 0=benign): {pred}")',
            starterCode: `from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

# Breast Cancer Biopsy Dataset
# Features: [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness]
# Target: 1 = malignant, 0 = benign
X_train = np.array([
    [17.99, 10.38, 122.80, 1001.0, 0.1184, 0.2776],  # malignant
    [20.57, 17.77, 132.90, 1326.0, 0.0847, 0.0786],  # malignant
    [19.69, 21.25, 130.00, 1203.0, 0.1096, 0.1599],  # malignant
    [11.42, 20.38,  77.58,  386.1, 0.1425, 0.2839],  # malignant
    [12.45, 15.70,  82.57,  477.1, 0.1278, 0.1700],  # malignant
    [12.31, 11.22,  78.87,  466.4, 0.0942, 0.0497],  # benign
    [13.54, 14.36,  87.46,  566.3, 0.0977, 0.0818],  # benign
    [13.08, 15.71,  85.63,  520.0, 0.1075, 0.1270],  # benign
    [9.504, 12.44,  60.34,  273.9, 0.1024, 0.0649],  # benign
    [15.34, 14.26, 102.50,  704.4, 0.1073, 0.2135],  # benign
])
y_train = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

# Test: classify a new biopsy sample
test_cell = np.array([[14.5, 18.2, 92.0, 640.0, 0.098, 0.105]])

# TODO: Create a Pipeline with StandardScaler() and SVC(kernel='rbf', C=1.0, random_state=42)
# TODO: Fit the pipeline on X_train, y_train
# TODO: Predict the class for test_cell and print the result

pipeline = None
pred = []
print(f"Tumor prediction (1=malignant, 0=benign): {pred}")`,
            expectedOutput: 'Tumor prediction (1=malignant, 0=benign): [1]',
            solution: `from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

X_train = np.array([
    [17.99, 10.38, 122.80, 1001.0, 0.1184, 0.2776],
    [20.57, 17.77, 132.90, 1326.0, 0.0847, 0.0786],
    [19.69, 21.25, 130.00, 1203.0, 0.1096, 0.1599],
    [11.42, 20.38,  77.58,  386.1, 0.1425, 0.2839],
    [12.45, 15.70,  82.57,  477.1, 0.1278, 0.1700],
    [12.31, 11.22,  78.87,  466.4, 0.0942, 0.0497],
    [13.54, 14.36,  87.46,  566.3, 0.0977, 0.0818],
    [13.08, 15.71,  85.63,  520.0, 0.1075, 0.1270],
    [9.504, 12.44,  60.34,  273.9, 0.1024, 0.0649],
    [15.34, 14.26, 102.50,  704.4, 0.1073, 0.2135],
])
y_train = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

test_cell = np.array([[14.5, 18.2, 92.0, 640.0, 0.098, 0.105]])

pipeline = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, random_state=42))
pipeline.fit(X_train, y_train)
pred = pipeline.predict(test_cell)
print(f"Tumor prediction (1=malignant, 0=benign): {pred}")`,
            hints: ['Import make_pipeline from sklearn.pipeline.', 'Combine StandardScaler() and SVC(kernel=\'rbf\', C=1.0, random_state=42) in the pipeline.', 'Call .fit() then .predict() on the pipeline object.'],
            testKeywords: ['make_pipeline', 'StandardScaler', 'SVC', 'predict']
        },
        assignment: {
            title: 'Spam Email Classification with SVM',
            description: 'An email provider wants to filter spam. Each email is represented by 5 word-count features: count of words like "free", "winner", "click", "meeting", and "report". Build an SVM classifier in a Pipeline with StandardScaler to separate spam (1) from legitimate email (0). Then evaluate its accuracy on two held-out test emails.',
            pseudoCode: 'Load spam email word-count dataset (free, winner, click, meeting, report counts)\npipeline = make_pipeline(StandardScaler(), SVC(kernel="linear", C=10.0))\npipeline.fit(X_train, y_train)\nacc = pipeline.score(X_test, y_test)\nprint(f"Test Accuracy: {acc:.2f}")',
            starterCode: `from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

# Spam Email Word-Count Dataset
# Features: [free_count, winner_count, click_count, meeting_count, report_count]
# Target: 1 = spam, 0 = legitimate
X_train = np.array([
    [12,  8, 15,  0,  0],  # spam
    [20,  5, 18,  0,  0],  # spam
    [ 9, 11, 12,  1,  0],  # spam
    [18,  7, 20,  0,  0],  # spam
    [ 0,  0,  1, 14, 10],  # legitimate
    [ 0,  0,  0, 20, 15],  # legitimate
    [ 1,  0,  2, 18,  9],  # legitimate
    [ 0,  0,  0, 25, 12],  # legitimate
])
y_train = np.array([1, 1, 1, 1, 0, 0, 0, 0])

X_test = np.array([
    [14,  6, 17,  0,  0],  # should be spam
    [ 0,  0,  1, 22, 11],  # should be legitimate
])
y_test = np.array([1, 0])

# TODO: Create a pipeline with StandardScaler() and SVC(kernel='linear', C=10.0)
# TODO: Fit the pipeline on X_train, y_train
# TODO: Score on X_test, y_test and print the result

pipeline = None
acc = 0.0
print(f"Test Accuracy: {acc:.2f}")`,
            expectedOutput: 'Test Accuracy: 1.00',
            solution: `from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

X_train = np.array([
    [12,  8, 15,  0,  0],
    [20,  5, 18,  0,  0],
    [ 9, 11, 12,  1,  0],
    [18,  7, 20,  0,  0],
    [ 0,  0,  1, 14, 10],
    [ 0,  0,  0, 20, 15],
    [ 1,  0,  2, 18,  9],
    [ 0,  0,  0, 25, 12],
])
y_train = np.array([1, 1, 1, 1, 0, 0, 0, 0])

X_test = np.array([
    [14,  6, 17,  0,  0],
    [ 0,  0,  1, 22, 11],
])
y_test = np.array([1, 0])

pipeline = make_pipeline(StandardScaler(), SVC(kernel='linear', C=10.0))
pipeline.fit(X_train, y_train)
acc = pipeline.score(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")`,
            hints: ['Use make_pipeline with StandardScaler() and SVC(kernel=\'linear\', C=10.0).', 'Call .fit() on X_train and y_train.', 'Call .score(X_test, y_test) to get the test accuracy.'],
            testKeywords: ['make_pipeline', 'StandardScaler', 'SVC', 'score']
        }
    },
    interviewQuestions: [
        {
            question: 'What is the "Kernel Trick" in Support Vector Machines?',
            answer: 'The kernel trick allows SVMs to solve non-linear classification problems by implicitly mapping the input data into a high-dimensional feature space where it becomes linearly separable. It does this by computing the dot products between data points in the high-dimensional space without ever actually transforming the data, which saves massive computational resources.',
            difficulty: 'Advanced',
            companyTags: ['Google', 'Meta']
        },
        {
            question: 'What are Support Vectors?',
            answer: 'Support vectors are the subset of training data points that lie closest to the decision surface (the hyperplane). They are the most difficult points to classify. The entire decision boundary is determined solely by these points; if you removed all other training data, the boundary would not change.',
            difficulty: 'Beginner',
            companyTags: ['Amazon', 'Microsoft']
        },
        {
            question: 'Explain the C hyperparameter in SVM.',
            answer: 'C is the regularization parameter that controls the trade-off between achieving a low training error and a large margin. A small C creates a wider, softer margin that tolerates some misclassifications (higher bias, lower variance). A large C creates a narrow, hard margin that strictly penalizes misclassifications (lower bias, higher variance).',
            difficulty: 'Intermediate',
            companyTags: ['Uber', 'Lyft']
        },
        {
            question: 'Why is feature scaling crucial for SVMs?',
            answer: 'SVMs aim to maximize the geometric distance (margin) between the hyperplane and support vectors. If features are on completely different scales (e.g., Age 0-100 vs. Salary 0-100,000), the feature with the larger range will completely dominate the distance calculations, leading to a severely skewed boundary.',
            difficulty: 'Beginner',
            companyTags: ['Apple', 'Spotify']
        },
        {
            question: 'What is the Hinge Loss function?',
            answer: 'Hinge loss is the loss function used to train SVMs. For a prediction y_pred and true label y_true (+1 or -1), the hinge loss is max(0, 1 - y_true * y_pred). It penalizes predictions that are not only on the wrong side of the boundary, but also those on the right side but within the margin.',
            difficulty: 'Intermediate',
            companyTags: ['LinkedIn', 'Snap']
        },
        {
            question: 'What is the difference between Hard Margin and Soft Margin SVM?',
            answer: 'Hard Margin SVM strictly demands that all data points are correctly classified with a margin of at least 1 (fails if data isn\'t linearly separable). Soft Margin SVM introduces slack variables to allow some points to violate the margin or be misclassified, making it robust to noise and outliers.',
            difficulty: 'Intermediate',
            companyTags: ['Capital One', 'Stripe']
        },
        {
            question: 'What does the gamma parameter do in the RBF kernel?',
            answer: 'Gamma defines how far the influence of a single training example reaches. Low gamma means "far reach" (a smoother, more general decision boundary). High gamma means "close reach" (the boundary tightly wraps around individual data points, increasing the risk of overfitting).',
            difficulty: 'Advanced',
            companyTags: ['Netflix', 'Pinterest']
        },
        {
            question: 'Can SVM be used for multi-class classification?',
            answer: 'Yes, but SVM is inherently a binary classifier. Multi-class classification is achieved using strategies like One-vs-Rest (OvR), where N classifiers are trained to separate each class from all others, or One-vs-One (OvO), where N(N-1)/2 classifiers are trained for every pair of classes.',
            difficulty: 'Intermediate',
            companyTags: ['Google', 'Amazon']
        },
        {
            question: 'What is the time complexity of training an SVM?',
            answer: 'Training a non-linear SVM scales between O(N^2) and O(N^3) with respect to the number of training samples (N). This makes SVMs impractical for massive datasets with millions of rows.',
            difficulty: 'Advanced',
            companyTags: ['Meta', 'Jane Street']
        },
        {
            question: 'How does an SVM output probabilities?',
            answer: 'Standard SVMs do not output probabilities; they output a distance to the decision boundary. To get probabilities, Platt Scaling is used, which fits a logistic regression model on top of the SVM\'s distance scores using cross-validation.',
            difficulty: 'Advanced',
            companyTags: ['Microsoft', 'AWS']
        },
        {
            question: 'Name three common Kernels used in SVM.',
            answer: '1. Linear (no transformation), 2. Polynomial (maps to polynomial feature space), 3. Radial Basis Function (RBF or Gaussian, maps to an infinite-dimensional space).',
            difficulty: 'Beginner',
            companyTags: ['Apple', 'Airbnb']
        },
        {
            question: 'What happens to an SVM if you have more features than samples (D > N)?',
            answer: 'SVMs perform exceptionally well in high-dimensional spaces, even when D > N, because the optimization depends on the number of samples (support vectors) rather than the number of dimensions. A Linear kernel is usually best here to avoid overfitting.',
            difficulty: 'Intermediate',
            companyTags: ['Palantir', 'Kaggle']
        },
        {
            question: 'How do you handle imbalanced datasets in SVM?',
            answer: 'You can use class weights (e.g., `class_weight=\'balanced\'` in sklearn) to assign a higher penalty (higher C value) to misclassifications of the minority class, ensuring the hyperplane doesn\'t just ignore it.',
            difficulty: 'Intermediate',
            companyTags: ['Uber', 'Capital One']
        },
        {
            question: 'What is Support Vector Regression (SVR)?',
            answer: 'SVR is the adaptation of SVM for continuous regression tasks. Instead of finding a margin that separates classes, it finds a "tube" (epsilon-tube) around the prediction function that encompasses as many training points as possible, ignoring errors within the tube.',
            difficulty: 'Advanced',
            companyTags: ['Databricks', 'IBM']
        },
        {
            question: 'Why might an RBF kernel overfit, and how do you fix it?',
            answer: 'An RBF kernel overfits if gamma or C is too high, causing the boundary to tightly encircle individual training points. To fix it, you should decrease gamma (to widen the influence radius) and decrease C (to enforce a softer margin).',
            difficulty: 'Intermediate',
            companyTags: ['Google', 'Spotify']
        },
        {
            question: 'Is SVM sensitive to outliers?',
            answer: 'A Hard-Margin SVM is extremely sensitive to outliers because a single outlier can shift the entire boundary to maintain perfect separation. A Soft-Margin SVM (with a lower C) mitigates this by ignoring the outlier.',
            difficulty: 'Intermediate',
            companyTags: ['Meta', 'Snap']
        },
        {
            question: 'What is the primal vs. dual formulation of SVM?',
            answer: 'The primal problem directly optimizes the weights w and bias b. The dual formulation expresses the problem in terms of Lagrange multipliers (alpha), where the solution depends only on the dot products of the data. The dual form is what enables the Kernel trick.',
            difficulty: 'Advanced',
            companyTags: ['Jane Street', 'Two Sigma']
        },
        {
            question: 'Can you use a custom Kernel in SVM?',
            answer: 'Yes, as long as the custom kernel function satisfies Mercer\'s Theorem (it must be symmetric and positive semi-definite).',
            difficulty: 'Advanced',
            companyTags: ['OpenAI', 'Google']
        },
        {
            question: 'When would you prefer Logistic Regression over SVM?',
            answer: 'I would prefer Logistic Regression when I need native probability outputs, when I want a simpler interpretable model (using coefficients), or when working with massive datasets where SVM training would be computationally prohibitive.',
            difficulty: 'Intermediate',
            companyTags: ['Amazon', 'Microsoft']
        },
        {
            question: 'What is a slack variable in SVM?',
            answer: 'A slack variable (xi) allows a specific data point to violate the margin. It measures the distance by which a point is on the wrong side of its margin. The sum of all slack variables is minimized in the soft-margin objective function.',
            difficulty: 'Beginner',
            companyTags: ['Apple', 'Netflix']
        }
    ]
};
