import type { MLModule } from '../types';

export const modulesData: MLModule[] = [
  {
    id: 'linear-regression',
    title: 'Linear Regression',
    category: 'Foundations & Math',
    description: 'Fit lines to continuous variables using Least Squares or Gradient Descent.',
    formula: 'y = w^T x + b',
    theory: `### Linear Regression Derivation & Mechanics

Linear Regression assumes a linear relationship between input features $X \\in \\mathbb{R}^{m \\times d}$ and target outputs $y \\in \\mathbb{R}^m$. The hypothesis function is modeled as:
$$h_w(x) = w^T x + b$$
where $w$ is the weight vector and $b$ is the bias.

#### Mean Squared Error (MSE) Cost Function
To evaluate parameter fitness, we define the Cost Function $J(w, b)$ which measures residual variances:
$$J(w, b) = \\frac{1}{2m} \\sum_{i=1}^{m} \\left( h_w(x^{(i)}) - y^{(i)} \\right)^2$$

#### Parameter Optimization: Closed-Form Normal Equation
The optimal weights can be solved directly by taking the gradient of the cost function with respect to $w$, setting it to zero, and solving:
$$\\nabla_w J(w) = X^T X w - X^T y = 0 \\implies w = (X^T X)^{-1} X^T y$$

#### Parameter Optimization: Gradient Descent
Alternatively, we iteratively update parameters in the opposite direction of the gradient:
$$w_j \\leftarrow w_j - \\eta \\frac{\\partial J}{\\partial w_j} \\implies w_j \\leftarrow w_j - \\frac{\\eta}{m} \\sum_{i=1}^{m} \\left( h_w(x^{(i)}) - y^{(i)} \\right) x_j^{(i)}$$
$$b \\leftarrow b - \\frac{\\eta}{m} \\sum_{i=1}^{m} \\left( h_w(x^{(i)}) - y^{(i)} \\right)$$`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'lin-reg',
    quiz: [
      {
        id: 'lr_q1',
        question: 'What is the closed-form algebraic solution (Normal Equation) for Linear Regression weights?',
        options: [
          'w = (X^T X)^-1 X^T y',
          'w = X^T (X X^T)^-1 y',
          'w = (X X^T)^-1 X^T y',
          'w = (X^T X) X^T y'
        ],
        correctAnswer: 'w = (X^T X)^-1 X^T y',
        explanation: 'Minimizing the sum of squared residuals yields the Normal Equation w = (X^T X)^-1 X^T y, which is exact but computationally heavy for large feature spaces due to matrix inversion.'
      },
      {
        id: 'lr_q2',
        question: 'How does high model complexity affect linear regression coefficients if regularization is not applied?',
        options: [
          'Coefficients can become extremely large and sensitive to noise.',
          'Coefficients will always converge to exactly zero.',
          'The model bias will increase.',
          'The variance will decrease.'
        ],
        correctAnswer: 'Coefficients can become extremely large and sensitive to noise.',
        explanation: 'Without regularization, unconstrained fitting on highly correlated features or polynomial mappings causes coefficients to explode in magnitude to fit training points perfectly.'
      },
      {
        id: 'lr_q3',
        question: 'Which metric measures the proportion of variance in the dependent variable explained by the features?',
        options: [
          'R-squared (Coefficient of Determination)',
          'Mean Absolute Error (MAE)',
          'Mean Squared Error (MSE)',
          'Root Mean Squared Error (RMSE)'
        ],
        correctAnswer: 'R-squared (Coefficient of Determination)',
        explanation: 'R-squared (R^2) represents the fraction of total variance explained by the regression line model compared to a simple horizontal average line.'
      }
    ],
    coding: {
      tutorial: {
      title: 'Mean Squared Error Cost Function',
      description: 'Implement the Mean Squared Error (MSE) cost function J(w, b) in vectorized NumPy format.',
      pseudoCode: "",
        starterCode: `import numpy as np

def compute_mse(y_true, y_pred):
    m = len(y_true)
    # TODO: Calculate J = (1 / 2m) * sum((y_pred - y_true) ** 2)
    cost = 0.0
    return cost

y_t = np.array([1.5, 2.0, 3.1])
y_p = np.array([1.4, 2.2, 2.9])
print("Cost:", compute_mse(y_t, y_p))`,
        expectedOutput: 'Cost: 0.015',
        solution: `import numpy as np

def compute_mse(y_true, y_pred):
    m = len(y_true)
    cost = (1 / (2 * m)) * np.sum((y_pred - y_true) ** 2)
    return cost

y_t = np.array([1.5, 2.0, 3.1])
y_p = np.array([1.4, 2.2, 2.9])
print("Cost:", compute_mse(y_t, y_p))`,
        hints: ['Subtract y_true from y_pred.', 'Square the differences using **2.', 'Sum all values with np.sum() and scale by 1 / (2 * m).'],
        testKeywords: ['np.sum', 'np.square', '2 * m']
      },
      project: {
      title: 'Single-Step Gradient Descent',
      description: 'Implement a single iteration of gradient updates for weights w and bias b.',
      pseudoCode: "",
        starterCode: `import numpy as np

def gradient_step(X, y, w, b, eta):
    m = len(y)
    # TODO: Calculate gradients and update parameters
    # predictions: h = Xw + b
    # dw = (1/m) * X^T (h - y)
    # db = (1/m) * sum(h - y)
    dw = 0.0
    db = 0.0
    
    new_w = w - eta * dw
    new_b = b - eta * db
    return new_w, new_b

X = np.array([[1.0], [2.0], [3.0]])
y = np.array([2.0, 4.0, 6.0])
w = np.array([0.5])
b = 0.1
eta = 0.1
new_w, new_b = gradient_step(X, y, w, b, eta)
print("Updated weights:", np.round(new_w, 2), "Updated bias:", np.round(new_b, 2))`,
        expectedOutput: 'Updated weights: [1.27] Updated bias: 0.45',
        solution: `import numpy as np

def gradient_step(X, y, w, b, eta):
    m = len(y)
    h = np.dot(X, w) + b
    dw = (1 / m) * np.dot(X.T, (h - y))
    db = (1 / m) * np.sum(h - y)
    new_w = w - eta * dw
    new_b = b - eta * db
    return new_w, new_b

X = np.array([[1.0], [2.0], [3.0]])
y = np.array([2.0, 4.0, 6.0])
w = np.array([0.5])
b = 0.1
eta = 0.1
new_w, new_b = gradient_step(X, y, w, b, eta)
print("Updated weights:", np.round(new_w, 2), "Updated bias:", np.round(new_b, 2))`,
        hints: ['Calculate h = X * w + b first.', 'Use np.dot(X.T, h - y) for dw.', 'Make sure you subtract the learning rate multiplied by gradients.'],
        testKeywords: ['np.dot', 'np.sum', 'eta *']
      },
      assignment: {
      title: 'Normal Equation Solver',
      description: 'Write a closed-form Normal Equation solver to calculate exact regression weights directly.',
      pseudoCode: "",
        starterCode: `import numpy as np

def normal_equation(X, y):
    # TODO: Add column of ones to X, and compute w = (X^T X)^-1 X^T y
    # X_b = np.hstack([np.ones((X.shape[0], 1)), X])
    # w = np.dot(np.dot(np.linalg.inv(np.dot(X_b.T, X_b)), X_b.T), y)
    return None

X = np.array([[1.0], [2.0], [3.0]])
y = np.array([2.1, 3.9, 6.2])
print("Coefficients:", np.round(normal_equation(X, y), 2))`,
        expectedOutput: 'Coefficients: [0.03 2.05]',
        solution: `import numpy as np

def normal_equation(X, y):
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])
    w = np.dot(np.dot(np.linalg.inv(np.dot(X_b.T, X_b)), X_b.T), y)
    return w

X = np.array([[1.0], [2.0], [3.0]])
y = np.array([2.1, 3.9, 6.2])
print("Coefficients:", np.round(normal_equation(X, y), 2))`,
        hints: ['Use np.hstack to combine a bias vector of ones with X.', 'Calculate matrix inverse using np.linalg.inv.', 'Chain products using np.dot or the @ operator.'],
        testKeywords: ['np.linalg.inv', 'np.ones', 'hstack']
      }
    },
    interviewQuestions: [{
      question: 'Under what conditions does the Normal Equation fail or become mathematically unstable, and how do we resolve it?',
      answer: 'The Normal Equation w = (X^T X)^-1 X^T y fails when X^T X is non-invertible (singular matrix). This occurs if features are linearly dependent (perfect multicollinearity) or if the number of features d exceeds the number of samples m. We resolve this by removing redundant features, using pseudo-inverse (np.linalg.pinv), or applying L2 regularization (Ridge) which adds a diagonal lambda penalty (X^T X + lambda I)^-1, guaranteeing invertibility.',
      companyTags: ['Google', 'Amazon'],
      difficulty: 'Intermediate'
    }]
  },
  {
    id: 'logistic-regression',
    title: 'Logistic Regression',
    category: 'Foundations & Math',
    description: 'Model discrete class probabilities using Log-Odds and Sigmoid functions.',
    formula: 'p = \\sigma(w^T x + b)',
    theory: `### Logistic Regression Derivation & Binary Classification

Logistic Regression fits a decision boundary separating binary classes ($y \\in \\{0, 1\\}$).

#### Sigmoid (Logistic) Function
To map raw outputs $z = w^T x + b$ to a probability $p \\in (0, 1)$, we apply the Sigmoid function:
$$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$

#### The Log-Odds Representation
The log-odds (logit) is modeled as a linear combination of inputs:
$$\\log \\left( \\frac{p}{1 - p} \\right) = w^T x + b$$

#### Binary Cross-Entropy Loss (Log Loss)
Rather than Mean Squared Error (which results in a non-convex loss space when combined with Sigmoid), we minimize Log Loss via maximum likelihood:
$$J(w, b) = -\\frac{1}{m} \\sum_{i=1}^{m} \\left[ y^{(i)} \\log(p^{(i)}) + (1 - y^{(i)}) \\log(1 - p^{(i)}) \\right]$$
where $p^{(i)} = \\sigma(w^T x^{(i)} + b)$.`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'log-reg',
    quiz: [
      {
        id: 'log_q1',
        question: 'Why is Mean Squared Error (MSE) not ideal for optimizing Logistic Regression parameters?',
        options: [
          'The resulting loss landscape is non-convex, containing many local minima.',
          'MSE is mathematically impossible to differentiate with Sigmoids.',
          'MSE causes parameters to diverge immediately to infinity.',
          'Log loss always calculates faster than MSE.'
        ],
        correctAnswer: 'The resulting loss landscape is non-convex, containing many local minima.',
        explanation: 'Applying MSE to a Sigmoid prediction results in a non-convex cost surface with local traps. Binary Cross-Entropy yields a convex landscape, ensuring gradient descent reaches the global minimum.'
      },
      {
        id: 'log_q2',
        question: 'What is the derivative of the Sigmoid function g(z) with respect to z?',
        options: [
          'g(z)(1 - g(z))',
          'g(z) + g(-z)',
          '1 - g(z)^2',
          'g(z)(1 + g(z))'
        ],
        correctAnswer: 'g(z)(1 - g(z))',
        explanation: 'The Sigmoid function derivative is highly elegant: d/dz [sigma(z)] = sigma(z) * (1 - sigma(z)). This simplifies neural backpropagation equations.'
      },
      {
        id: 'log_q3',
        question: 'If a Logistic Regression model predicts p = 0.99 for y = 0, what is the approximate Log Loss contribution?',
        options: [
          'Extreme loss (near infinity)',
          '0.01 loss',
          '0.99 loss',
          '0.00 loss'
        ],
        correctAnswer: 'Extreme loss (near infinity)',
        explanation: 'Log Loss penalizes confident wrong predictions. The loss is -log(1 - 0.99) = -log(0.01) = 4.6, which spikes exponentially as probability approaches the incorrect extreme.'
      }
    ],
    coding: {
      tutorial: {
      title: 'Sigmoid Activation Function',
      description: 'Write a vectorized Sigmoid activation function that maps inputs to probabilities.',
      pseudoCode: "",
        starterCode: `import numpy as np

def sigmoid(z):
    # TODO: Implement 1 / (1 + e^-z)
    return 0.0

z_val = np.array([-10.0, 0.0, 10.0])
print("Probabilities:", np.round(sigmoid(z_val), 2))`,
        expectedOutput: 'Probabilities: [0.   0.5  1.  ]',
        solution: `import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z_val = np.array([-10.0, 0.0, 10.0])
print("Probabilities:", np.round(sigmoid(z_val), 2))`,
        hints: ['Use np.exp() for calculating exponents.', 'Ensure the formula maps correctly for vectors and matrices.'],
        testKeywords: ['np.exp', '1 /']
      },
      project: {
      title: 'Binary Cross-Entropy Loss',
      description: 'Implement Binary Cross-Entropy (Log Loss) function to grade probabilities.',
      pseudoCode: "",
        starterCode: `import numpy as np

def log_loss(y_true, y_pred):
    m = len(y_true)
    # Clip predictions to prevent log(0) errors
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    # TODO: Compute -1/m * sum(y*log(p) + (1-y)*log(1-p))
    loss = 0.0
    return loss

y_t = np.array([0, 1, 1, 0])
y_p = np.array([0.1, 0.9, 0.8, 0.2])
print("Log Loss:", np.round(log_loss(y_t, y_p), 4))`,
        expectedOutput: 'Log Loss: 0.1643',
        solution: `import numpy as np

def log_loss(y_true, y_pred):
    m = len(y_true)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = - (1 / m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

y_t = np.array([0, 1, 1, 0])
y_p = np.array([0.1, 0.9, 0.8, 0.2])
print("Log Loss:", np.round(log_loss(y_t, y_p), 4))`,
        hints: ['Use np.log() for the natural log.', 'Multiply element-wise and sum values.'],
        testKeywords: ['np.log', 'np.clip', 'np.sum']
      },
      assignment: {
      title: 'Logistic Parameter Gradient Step',
      description: 'Calculate gradients and perform an update step for logistic weights.',
      pseudoCode: "",
        starterCode: `import numpy as np

def logistic_update(X, y, w, b, eta):
    m = len(y)
    # predictions: p = sigmoid(Xw + b)
    # dw = (1/m) * X^T (p - y)
    # db = (1/m) * sum(p - y)
    # TODO: Implement update
    new_w = w
    new_b = b
    return new_w, new_b

X = np.array([[1.0], [2.0], [3.0]])
y = np.array([0.0, 0.0, 1.0])
w = np.array([0.1])
b = -0.5
eta = 0.5
nw, nb = logistic_update(X, y, w, b, eta)
print("nw:", np.round(nw, 3), "nb:", np.round(nb, 3))`,
        expectedOutput: 'nw: [0.187] nb: -0.449',
        solution: `import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_update(X, y, w, b, eta):
    m = len(y)
    z = np.dot(X, w) + b
    p = sigmoid(z)
    dw = (1 / m) * np.dot(X.T, (p - y))
    db = (1 / m) * np.sum(p - y)
    new_w = w - eta * dw
    new_b = b - eta * db
    return new_w, new_b

X = np.array([[1.0], [2.0], [3.0]])
y = np.array([0.0, 0.0, 1.0])
w = np.array([0.1])
b = -0.5
eta = 0.5
nw, nb = logistic_update(X, y, w, b, eta)
print("nw:", np.round(nw, 3), "nb:", np.round(nb, 3))`,
        hints: ['Remember to import or use a helper sigmoid function.', 'Compute raw score z, then probability p.', 'Ensure shape of X.T matches prediction error.'],
        testKeywords: ['np.dot', 'sigmoid', 'eta *']
      }
    },
    interviewQuestions: [{
      question: 'Explain the concept of odds ratio and log-odds. Why do we model the log-odds linearly in logistic regression instead of probability?',
      answer: 'Probabilities are bounded between 0 and 1, making linear modeling impossible (a line extends to infinity). The odds ratio p/(1-p) maps [0, 1] to [0, +inf]. Taking the logarithm gives the Log-Odds (logit), which spans (-inf, +inf). This matches the range of linear equations w^T x + b, allowing coefficients to represent a linear change in log-odds. Exp(w_j) represents the multiplicative change in the odds ratio for each unit increase in feature x_j.',
      companyTags: ['Meta', 'Netflix'],
      difficulty: 'Advanced'
    }]
  },
  {
    id: 'decision-trees',
    title: 'Decision Trees',
    category: 'Supervised Learning',
    description: 'Split datasets recursively to isolate target variables based on impurity reductions.',
    formula: 'IG(D, A) = H(D) - H(D|A)',
    theory: `### Decision Tree Partitioning & Information Theory

Decision Trees segment the feature space recursively using greedy threshold boundaries.

#### Entropy & Information Content
Entropy $H(S)$ measures classification impurity:
$$H(S) = -\\sum_{c \\in C} p_c \\log_2(p_c)$$
where $p_c$ is the proportion of samples belonging to class $c$.

#### Gini Impurity (alternative metric)
Used in CART trees:
$$Gini(S) = 1 - \\sum_{c \\in C} p_c^2$$

#### Information Gain (ID3)
Select the feature split that maximizes the reduction in entropy:
$$IG(T, a) = H(T) - H(T | a) = H(T) - \\sum_{v \\in values(a)} \\frac{|T_v|}{|T|} H(T_v)$$`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'decision-tree',
    quiz: [
      {
        id: 'dt_q1',
        question: 'What is the entropy of a collection with 5 positive and 5 negative binary class samples?',
        options: ['1.0', '0.0', '0.5', '2.0'],
        correctAnswer: '1.0',
        explanation: 'At 50/50 split, impurity is maximal. Entropy = - (0.5 * log2(0.5) + 0.5 * log2(0.5)) = - (-0.5 - 0.5) = 1.0.'
      },
      {
        id: 'dt_q2',
        question: 'Which method helps prevent decision trees from overestimating noise in training data?',
        options: ['Pruning (Pre- and Post-pruning)', 'Increasing max depth', 'Removing categorical features', 'Bagging only'],
        correctAnswer: 'Pruning (Pre- and Post-pruning)',
        explanation: 'Pruning collapses weak branches that fit minor noise variations, capping tree depth or minimum leaf counts to encourage generalization.'
      },
      {
        id: 'dt_q3',
        question: 'What type of decision boundaries are produced by a standard decision tree?',
        options: [
          'Axis-aligned orthogonal boundaries',
          'Curved polynomial boundaries',
          'Hyperplanes in kernel space',
          'Circular radial regions'
        ],
        correctAnswer: 'Axis-aligned orthogonal boundaries',
        explanation: 'Since trees split on one variable at a time (e.g. x_1 > threshold), boundaries are step-like and perpendicular to coordinate axes.'
      }
    ],
    coding: {
      tutorial: {
      title: 'Calculate Label Entropy',
      description: 'Write a function to calculate the Shannon Entropy of a list of binary class labels.',
      pseudoCode: "",
        starterCode: `import numpy as np

def calculate_entropy(labels):
    # TODO: Calculate probability of each class and return entropy
    entropy = 0.0
    return entropy

lbls = [1, 0, 1, 1, 1, 0]
print("Entropy:", np.round(calculate_entropy(lbls), 3))`,
        expectedOutput: 'Entropy: 0.918',
        solution: `import numpy as np

def calculate_entropy(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = np.bincount(labels)
    probs = counts / n
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

lbls = [1, 0, 1, 1, 1, 0]
print("Entropy:", np.round(calculate_entropy(lbls), 3))`,
        hints: ['Find the counts of each unique class.', 'Divide by total size to get probabilities.', 'Calculate sum of -p * log2(p) for positive probabilities.'],
        testKeywords: ['np.log2', 'probs', 'entropy']
      },
      project: {
      title: 'Compute Split Information Gain',
      description: 'Calculate Information Gain given parent labels and two partitioned child label arrays.',
      pseudoCode: "",
        starterCode: `import numpy as np

def entropy(y):
    counts = np.bincount(y)
    probs = counts / len(y)
    return -sum(p * np.log2(p) for p in probs if p > 0)

def info_gain(parent, left_child, right_child):
    # TODO: Compute entropy reduction: H(parent) - (w_l * H(left) + w_r * H(right))
    gain = 0.0
    return gain

parent = [1, 1, 0, 0, 0, 1]
left = [1, 1, 1]
right = [0, 0, 0]
print("Information Gain:", np.round(info_gain(parent, left, right), 3))`,
        expectedOutput: 'Information Gain: 1.0',
        solution: `import numpy as np

def entropy(y):
    counts = np.bincount(y)
    probs = counts / len(y)
    return -sum(p * np.log2(p) for p in probs if p > 0)

def info_gain(parent, left_child, right_child):
    w_left = len(left_child) / len(parent)
    w_right = len(right_child) / len(parent)
    gain = entropy(parent) - (w_left * entropy(left_child) + w_right * entropy(right_child))
    return gain

parent = [1, 1, 0, 0, 0, 1]
left = [1, 1, 1]
right = [0, 0, 0]
print("Information Gain:", np.round(info_gain(parent, left, right), 3))`,
        hints: ['Compute total parent entropy H(parent).', 'Weight each child entropy by child_size / parent_size.', 'Subtract weighted sums from H(parent).'],
        testKeywords: ['entropy(left_child)', 'len(parent)', 'gain']
      },
      assignment: {
      title: 'Determine Best Splitting Threshold',
      description: 'Search a continuous variable feature array to locate the threshold yielding max Information Gain.',
      pseudoCode: "",
        starterCode: `import numpy as np

def find_best_split(X_feat, y):
    best_gain = -1.0
    best_threshold = None
    # TODO: For each unique value in X_feat, partition y into left (<= val) and right (> val)
    # Calculate info_gain, track max gain.
    return best_threshold

X_feat = np.array([1.5, 2.2, 3.8, 4.5, 5.0])
y = np.array([0, 0, 1, 1, 1])
print("Best split threshold:", find_best_split(X_feat, y))`,
        expectedOutput: 'Best split threshold: 2.2',
        solution: `import numpy as np

def entropy(y):
    if len(y) == 0: return 0
    counts = np.bincount(y)
    probs = counts / len(y)
    return -sum(p * np.log2(p) for p in probs if p > 0)

def info_gain(parent, left, right):
    w_l = len(left) / len(parent)
    w_r = len(right) / len(parent)
    return entropy(parent) - (w_l * entropy(left) + w_r * entropy(right))

def find_best_split(X_feat, y):
    best_gain = -1.0
    best_threshold = None
    sorted_idx = np.argsort(X_feat)
    
    # Check split positions between sorted values
    for i in range(len(X_feat) - 1):
        threshold = X_feat[sorted_idx[i]]
        left_mask = X_feat <= threshold
        right_mask = X_feat > threshold
        
        gain = info_gain(y, y[left_mask], y[right_mask])
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
            
    return best_threshold

X_feat = np.array([1.5, 2.2, 3.8, 4.5, 5.0])
y = np.array([0, 0, 1, 1, 1])
print("Best split threshold:", find_best_split(X_feat, y))`,
        hints: ['Sort the feature array to evaluate sequential cuts.', 'Slice labels using boolean masks like X <= threshold.', 'Return threshold representing the partition line.'],
        testKeywords: ['left_mask', 'info_gain', 'best_threshold']
      }
    },
    interviewQuestions: [{
      question: 'How do you prevent a decision tree from overfitting, and why are deep trees highly prone to high variance?',
      answer: 'Deep trees are prone to high variance because they continue creating complex sub-branches to fit every random noise item in training. We prevent overfitting by: 1) Pre-pruning: setting limits on max depth, min samples per split, or min samples per leaf; 2) Post-pruning: growing the tree fully, then pruning back branches that contribute less than a cost-complexity threshold; 3) Ensemble methods like Random Forests that average multiple deep trees to cancel individual variances.',
      companyTags: ['Amazon', 'Apple'],
      difficulty: 'Intermediate'
    }]
  },
  {
    id: 'random-forests',
    title: 'Random Forests',
    category: 'Supervised Learning',
    description: 'Ensemble deep decision trees using bagging and random feature selection.',
    formula: 'f(x) = \\frac{1}{B} \\sum_{b=1}^{B} f_b(x)',
    theory: `### Random Forests & Ensemble Learning Theory

Random Forests reduce variance by creating an ensemble of diverse, uncorrelated Decision Trees.

#### Bootstrap Aggregation (Bagging)
Each tree $f_b(x)$ is trained on a bootstrap sample $D_b$ drawn with replacement from the training set $D$.
Given $N$ samples, a bootstrap sample selects $N$ elements with replacement, resulting in approximately $63.2\\%$ unique records.

#### Random Feature Subsampling
To decorrelate the trees, at each split node, the tree builder searches only a random subset of features $m \\ll d$. Typically:
$$m = \\sqrt{d} \\quad \\text{(for classification)}$$
$$m = \\frac{d}{3} \\quad \\text{(for regression)}$$

#### Prediction Aggregation
- **Regression**: Average individual predictions:
$$\\hat{y} = \\frac{1}{B} \\sum_{b=1}^{B} f_b(x)$$
- **Classification**: Majority vote selection.`,
    interactiveSummary: "Interactive demonstration of this concept.",
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
      }
    ],
    coding: {
      tutorial: {
      title: 'Bootstrap Sampling Generator',
      description: 'Generate a bootstrap sample of indices with replacement from a given dataset length.',
      pseudoCode: "",
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
      title: 'Random Feature Masking',
      description: 'Generate a random subset of feature indices to search at split nodes.',
      pseudoCode: "",
        starterCode: `import numpy as np

def feature_subsample(num_features, seed=42):
    np.random.seed(seed)
    # TODO: Calculate k = sqrt(num_features) rounded down, choose k unique indices
    indices = np.array([])
    return indices

print("Subsampled Features:", feature_subsample(10))`,
        expectedOutput: 'Subsampled Features: [8 1 5]',
        solution: `import numpy as np

def feature_subsample(num_features, seed=42):
    np.random.seed(seed)
    k = int(np.sqrt(num_features))
    indices = np.random.choice(num_features, size=k, replace=False)
    return sorted(indices)

print("Subsampled Features:", feature_subsample(10))`,
        hints: ['Calculate square root with np.sqrt() and cast to int.', 'Choose features without replacement (replace=False).'],
        testKeywords: ['sqrt', 'replace=False', 'choice']
      },
      assignment: {
      title: 'Ensemble Majority Vote Aggregator',
      description: 'Combine class predictions from multiple trees to output final majority vote labels.',
      pseudoCode: "",
        starterCode: `import numpy as np

def ensemble_vote(tree_predictions):
    # tree_predictions: 2D array of shape (num_trees, num_samples)
    # TODO: Perform majority vote for each sample
    votes = np.array([])
    return votes

# 3 trees, 5 test samples
preds = np.array([
    [0, 1, 1, 0, 1],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 0, 0]
])
print("Final predictions:", ensemble_vote(preds))`,
        expectedOutput: 'Final predictions: [0 1 1 0 0]',
        solution: `import numpy as np

def ensemble_vote(tree_predictions):
    num_samples = tree_predictions.shape[1]
    votes = []
    for i in range(num_samples):
        sample_preds = tree_predictions[:, i]
        counts = np.bincount(sample_preds)
        votes.append(np.argmax(counts))
    return np.array(votes)

preds = np.array([
    [0, 1, 1, 0, 1],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 0, 0]
])
print("Final predictions:", ensemble_vote(preds))`,
        hints: ['Iterate through each sample coordinate (each column).', 'Use np.bincount to count class occurrences.', 'Select argmax to yield the majority winner.'],
        testKeywords: ['bincount', 'argmax', 'shape']
      }
    },
    interviewQuestions: [{
      question: 'Why do Random Forests not overfit when we increase the number of estimators, and what is the Out-Of-Bag (OOB) error?',
      answer: 'Adding trees reduces the variance of the ensemble because the law of large numbers guarantees that the average prediction error converges: the variance is bounded by (rho * sigma^2 + (1 - rho)/B * sigma^2). As B (trees) approaches infinity, the second term vanishes, leaving rho * sigma^2 (correlation among trees). Out-of-bag (OOB) error uses the ~36.8% samples omitted from training each tree. By predicting each sample using only trees that did not see it, we calculate an unbiased validation score without a separate validation split.',
      companyTags: ['Meta', 'Uber'],
      difficulty: 'Advanced'
    }]
  },
  {
    id: 'kmeans-clustering',
    title: 'K-Means Clustering',
    category: 'Unsupervised Learning',
    description: 'Group unlabeled data into K dense geometric clusters using distance minimizations.',
    formula: 'J = \\sum \\|x_i - \\mu_k\\|^2',
    theory: `### K-Means Clustering Mechanics & Lloyd's Algorithm

K-Means partitioning finds geometric cluster centers (centroids) in unlabeled datasets.

#### Optimization Goal: Within-Cluster Sum of Squares (WCSS)
We minimize the inertia metric:
$$J = \\sum_{k=1}^{K} \\sum_{x \\in C_k} \\left\\| x - \\mu_k \\right\\|^2$$
where $\\mu_k$ is the mean centroid coordinate of cluster $C_k$.

#### Lloyd's Iterative Algorithm
1. **Centroid Initialization**: Select $K$ start points.
2. **Assignment Step**: Assign each sample to the nearest centroid:
$$c^{(i)} \\leftarrow \\arg\\min_{k} \\left\\| x^{(i)} - \\mu_k \\right\\|^2$$
3. **Update Step**: Re-calculate centroid coordinates:
$$\\mu_k \\leftarrow \\frac{1}{|C_k|} \\sum_{i \\in C_k} x^{(i)}$$
Steps 2 and 3 repeat until centroids stabilize.`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'kmeans',
    quiz: [
      {
        id: 'km_q1',
        question: 'Which initialization method is designed to avoid poor convergence in standard K-Means?',
        options: [
          'K-Means++',
          'Random coordinate seeding',
          'First K observations',
          'PCA coordinates selection'
        ],
        correctAnswer: 'K-Means++',
        explanation: 'K-Means++ initializes centroids sequentially, prioritizing new centroids far from existing ones. This reduces sub-optimal convergence risks.'
      },
      {
        id: 'km_q2',
        question: 'How do you determine the optimal number of clusters K using visual heuristics?',
        options: [
          'Elbow Method / Silhouette Analysis',
          'Selecting K where loss spikes',
          'Fitting K to the number of columns',
          'Setting K equal to half the sample size'
        ],
        correctAnswer: 'Elbow Method / Silhouette Analysis',
        explanation: 'The Elbow Method plots WCSS against K, searching for an "elbow" where reduction rate flattens. Silhouette Analysis measures separation.'
      },
      {
        id: 'km_q3',
        question: 'Is K-Means guaranteed to converge to the global minimum of WCSS?',
        options: [
          'No, it is sensitive to initialization and can get trapped in local minima.',
          'Yes, Lloyd\'s algorithm is convex.',
          'Yes, if the features are normalized.',
          'No, it always diverges for non-linear structures.'
        ],
        correctAnswer: 'No, it is sensitive to initialization and can get trapped in local minima.',
        explanation: 'K-Means is a greedy coordinate descent algorithm. It converges to local minima, making multiple random restarts or K-Means++ essential.'
      }
    ],
    coding: {
      tutorial: {
      title: 'Euclidean Distance Matrices',
      description: 'Calculate Euclidean distance between a single point and multiple centroids.',
      pseudoCode: "",
        starterCode: `import numpy as np

def calculate_distances(point, centroids):
    # point: 1D array of shape (d,)
    # centroids: 2D array of shape (K, d)
    # TODO: Calculate distances to each centroid
    dists = np.array([])
    return dists

pt = np.array([1.0, 2.0])
cents = np.array([[0.0, 0.0], [2.0, 2.0]])
print("Distances:", np.round(calculate_distances(pt, cents), 3))`,
        expectedOutput: 'Distances: [2.236 1.   ]',
        solution: `import numpy as np

def calculate_distances(point, centroids):
    return np.sqrt(np.sum((centroids - point)**2, axis=1))

pt = np.array([1.0, 2.0])
cents = np.array([[0.0, 0.0], [2.0, 2.0]])
print("Distances:", np.round(calculate_distances(pt, cents), 3))`,
        hints: ['Subtract point from centroids.', 'Square the differences and sum across axis=1.', 'Take the square root.'],
        testKeywords: ['np.sum', 'np.sqrt', 'axis=1']
      },
      project: {
      title: 'Cluster Assignment Step',
      description: 'Assign each data point to its closest centroid index.',
      pseudoCode: "",
        starterCode: `import numpy as np

def assign_clusters(X, centroids):
    # X: (N, d), centroids: (K, d)
    # TODO: Assign each row in X to the nearest centroid index (0 to K-1)
    assignments = np.array([])
    return assignments

X = np.array([[1.0, 1.0], [0.5, 0.2], [4.0, 4.0]])
cents = np.array([[0.0, 0.0], [3.0, 3.0]])
print("Assignments:", assign_clusters(X, cents))`,
        expectedOutput: 'Assignments: [0 0 1]',
        solution: `import numpy as np

def assign_clusters(X, centroids):
    assignments = []
    for x in X:
        dists = np.sqrt(np.sum((centroids - x)**2, axis=1))
        assignments.append(np.argmin(dists))
    return np.array(assignments)

X = np.array([[1.0, 1.0], [0.5, 0.2], [4.0, 4.0]])
cents = np.array([[0.0, 0.0], [3.0, 3.0]])
print("Assignments:", assign_clusters(X, cents))`,
        hints: ['Iterate through each point in X.', 'Calculate distance vector to centroids.', 'Find index of minimum distance with np.argmin.'],
        testKeywords: ['argmin', 'np.sum', 'axis=1']
      },
      assignment: {
      title: 'Centroid Update Step',
      description: 'Recompute centroids by averaging coordinates of assigned data points.',
      pseudoCode: "",
        starterCode: `import numpy as np

def update_centroids(X, assignments, K):
    # X: (N, d), assignments: (N,) containing cluster indices
    # TODO: Compute mean of points assigned to each cluster index
    centroids = np.zeros((K, X.shape[1]))
    return centroids

X = np.array([[1.0, 2.0], [2.0, 4.0], [0.0, 0.0]])
assigns = np.array([1, 1, 0])
print("Updated centroids:", update_centroids(X, assigns, 2))`,
        expectedOutput: 'Updated centroids: [[0.  0. ]\n [1.5 3. ]]',
        solution: `import numpy as np

def update_centroids(X, assignments, K):
    d = X.shape[1]
    centroids = np.zeros((K, d))
    for k in range(K):
        pts = X[assignments == k]
        if len(pts) > 0:
            centroids[k] = np.mean(pts, axis=0)
        else:
            centroids[k] = np.zeros(d)
    return centroids

X = np.array([[1.0, 2.0], [2.0, 4.0], [0.0, 0.0]])
assigns = np.array([1, 1, 0])
print("Updated centroids:", update_centroids(X, assigns, 2))`,
        hints: ['Use boolean masking (assignments == k) to filter rows.', 'Apply np.mean with axis=0.', 'Handle cases where a cluster has zero points.'],
        testKeywords: ['np.mean', 'assignments ==', 'axis=0']
      }
    },
    interviewQuestions: [{
      question: 'What are the core limitations of the K-Means algorithm, and when would you choose DBSCAN instead?',
      answer: 'K-Means assumes spherical clusters of equal density and size due to its reliance on Euclidean distance. It fails on complex shapes (like concentric circles or moons) and is highly sensitive to outliers which pull centroid averages. We prefer DBSCAN when clusters have non-convex shapes, varying density, or when we need automatic outlier detection (noise points), since DBSCAN groups points based on density connectivity rather than central averages.',
      companyTags: ['Google', 'Lyft'],
      difficulty: 'Intermediate'
    }]
  },
  {
    id: 'support-vector-machines',
    title: 'Support Vector Machines',
    category: 'Supervised Learning',
    description: 'Find the optimal hyperplanes maximizing class boundary margins.',
    formula: 'y_i(w^T x_i + b) \\ge 1',
    theory: `### Support Vector Machines (SVM) & Margin Maximization

SVM models class splits by seeking the widest boundary corridor (margin).

#### Primal Optimization Formula
For linearly separable sets, we maximize the margin $2 / \\|w\\|$ which simplifies to:
$$\\min_{w, b} \\frac{1}{2} \\|w\\|^2 \\quad \\text{subject to } y_i(w^T x_i + b) \\ge 1, \\quad \\forall i$$

#### Soft Margin Optimization (C-SVM)
To handle overlapping noise, slack variables $\\xi_i \\ge 0$ allow minor margin violations:
$$\\min_{w, b, \\xi} \\frac{1}{2} \\|w\\|^2 + C \\sum_{i=1}^{m} \\xi_i \\quad \\text{subject to } y_i(w^T x_i + b) \\ge 1 - \\xi_i$$
where $C$ controls the violation penalty.

#### The Dual Formulation & Kernel Trick
Representing optimization using Lagrange duals yields:
$$\\max_{\\alpha} \\sum_{i=1}^{m} \\alpha_i - \\frac{1}{2} \\sum_{i,j} \\alpha_i \\alpha_j y_i y_j K(x_i, x_j)$$
where $K(x_i, x_j) = \\Phi(x_i)^T \\Phi(x_j)$ is a Kernel function (e.g. RBF/Radial Basis Kernel) mapping inputs to higher dimensional spaces.`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'svm',
    quiz: [
      {
        id: 'svm_q1',
        question: 'What happens if we increase the regularization constant C in a Soft Margin SVM classifier?',
        options: [
          'The model allows fewer margin violations, leading to a narrower margin.',
          'The model allows more margin violations, leading to a wider margin.',
          'The model becomes strictly linear.',
          'The support vectors are completely ignored.'
        ],
        correctAnswer: 'The model allows fewer margin violations, leading to a narrower margin.',
        explanation: 'C scales the penalty on violations. High C forces strict fitting of data points, producing narrow margins. Low C allows soft corridors with better generalization.'
      },
      {
        id: 'svm_q2',
        question: 'What is the "Kernel Trick" in SVM design?',
        options: [
          'Computing dot products in high-dimensional space without explicitly mapping features.',
          'Approximating margins using decision trees.',
          'Using multiple learning rates in SGD.',
          'Removing support vectors to speed up predictions.'
        ],
        correctAnswer: 'Computing dot products in high-dimensional space without explicitly mapping features.',
        explanation: 'The kernel trick replaces costly high-dimensional mappings Phi(x) with a similarity function K(x,y), avoiding the "curse of dimensionality" computations.'
      },
      {
        id: 'svm_q3',
        question: 'Which observations directly dictate the location and slope of the SVM decision boundary?',
        options: [
          'Support Vectors',
          'Centroids',
          'Out-Of-Bag instances',
          'Mean vectors'
        ],
        correctAnswer: 'Support Vectors',
        explanation: 'Only points lying on the margin boundaries (or violating them) have non-zero Lagrange multipliers alpha_i. These are "Support Vectors".'
      }
    ],
    coding: {
      tutorial: {
      title: 'Hinge Loss Function',
      description: 'Calculate Hinge Loss for a single prediction score: max(0, 1 - y * raw_score).',
      pseudoCode: "",
        starterCode: `import numpy as np

def hinge_loss(y, score):
    # TODO: Implement max(0, 1 - y * score)
    loss = 0.0
    return loss

print("Loss (Correct):", hinge_loss(1, 1.5))
print("Loss (Wrong):", hinge_loss(1, -0.5))`,
        expectedOutput: 'Loss (Correct): 0.0\nLoss (Wrong): 1.5',
        solution: `import numpy as np

def hinge_loss(y, score):
    return max(0.0, 1.0 - y * score)

print("Loss (Correct):", hinge_loss(1, 1.5))
print("Loss (Wrong):", hinge_loss(1, -0.5))`,
        hints: ['Compare 0.0 and 1.0 - y * score.', 'Use python\'s built-in max function.'],
        testKeywords: ['max', '1.0 -']
      },
      project: {
      title: 'Linear SVM Decision Function',
      description: 'Compute predicted raw margins for input vectors using weights and bias.',
      pseudoCode: "",
        starterCode: `import numpy as np

def svm_predict(X, w, b):
    # TODO: Calculate raw scores z = Xw + b
    # Return signs: +1 if z >= 0 else -1
    return np.array([])

X = np.array([[1.0, 2.0], [-1.0, -2.0]])
w = np.array([0.5, 0.5])
b = -0.1
print("Predictions:", svm_predict(X, w, b))`,
        expectedOutput: 'Predictions: [ 1 -1]',
        solution: `import numpy as np

def svm_predict(X, w, b):
    scores = np.dot(X, w) + b
    return np.where(scores >= 0, 1, -1)

X = np.array([[1.0, 2.0], [-1.0, -2.0]])
w = np.array([0.5, 0.5])
b = -0.1
print("Predictions:", svm_predict(X, w, b))`,
        hints: ['Use np.dot to compute matrix products.', 'Apply np.where to assign labels based on sign threshold.'],
        testKeywords: ['np.dot', 'np.where', '>= 0']
      },
      assignment: {
      title: 'SVM Gradient Step',
      description: 'Implement a gradient descent update step for C-SVM parameters with L2 regularization.',
      pseudoCode: "",
        starterCode: `import numpy as np

def svm_step(X, y, w, b, C, eta):
    # w_grad = w - C * y * X (if violation) else w
    # b_grad = - C * y (if violation) else 0
    # For one sample (X: (d,), y: scalar)
    violation = y * (np.dot(X, w) + b) < 1
    # TODO: Calculate gradients and return updated w, b
    new_w = w
    new_b = b
    return new_w, new_b

X = np.array([1.5, 0.5])
y = 1
w = np.array([0.2, 0.1])
b = 0.0
C = 1.0
eta = 0.1
nw, nb = svm_step(X, y, w, b, C, eta)
print("Updated weights:", np.round(nw, 2), "bias:", np.round(nb, 2))`,
        expectedOutput: 'Updated weights: [0.33 0.14] bias: 0.1',
        solution: `import numpy as np

def svm_step(X, y, w, b, C, eta):
    violation = y * (np.dot(X, w) + b) < 1
    if violation:
        dw = w - C * y * X
        db = - C * y
    else:
        dw = w
        db = 0.0
    new_w = w - eta * dw
    new_b = b - eta * db
    return new_w, new_b

X = np.array([1.5, 0.5])
y = 1
w = np.array([0.2, 0.1])
b = 0.0
C = 1.0
eta = 0.1
nw, nb = svm_step(X, y, w, b, C, eta)
print("Updated weights:", np.round(nw, 2), "bias:", np.round(nb, 2))`,
        hints: ['Evaluate the condition y * (X w + b) < 1.', 'Compute gradients based on violation state.', 'Subtract gradients scaled by learning rate eta.'],
        testKeywords: ['violation', 'C * y *', 'eta *']
      }
    },
    interviewQuestions: [{
      question: 'Compare SVM to Logistic Regression. Under what circumstances would you choose one over the other?',
      answer: 'SVM is a geometric margin maximizer, whereas Logistic Regression is a probabilistic likelihood maximizer. Logistic Regression is ideal when you need class probability scores, outputs online gradients easily, and works well on noise-heavy datasets. SVM is selected when boundaries are clean but highly non-linear, as the kernel trick handles complex partitions efficiently. Furthermore, SVM is less sensitive to extreme outliers because the boundary is dictated solely by support vectors.',
      companyTags: ['Microsoft', 'Amazon'],
      difficulty: 'Advanced'
    }]
  },
  {
    id: 'naive-bayes',
    title: 'Naive Bayes Classifier',
    category: 'Supervised Learning',
    description: 'Predict classes using conditional probability likelihoods and Bayes Theorem.',
    formula: 'P(y|x) \\propto P(y) \\prod P(x_i|y)',
    theory: `### Naive Bayes & Probabilistic Classification

Naive Bayes classifiers apply Bayes theorem under the assumption of feature independence.

#### Bayes Theorem
The posterior probability of class $y$ given features $x$ is:
$$P(y | x) = \\frac{P(x | y) P(y)}{P(x)}$$

#### The "Naive" Assumption
We assume features $x_i$ are conditionally independent given the class label $y$:
$$P(x | y) = P(x_1, x_2, \\dots, x_d | y) = \\prod_{i=1}^{d} P(x_i | y)$$
This simplifies the posterior calculation to:
$$P(y | x) \\propto P(y) \\prod_{i=1}^{d} P(x_i | y)$$

#### Laplace Smoothing
To avoid zero probability issues for unseen categorical features:
$$P(x_i | y) = \\frac{count(x_i, y) + \\alpha}{count(y) + \\alpha \\cdot |V|}$$
where $\\alpha$ is the smoothing parameter (usually 1.0) and $|V|$ is vocabulary size.`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'naive-bayes',
    quiz: [
      {
        id: 'nb_q1',
        question: 'What is the "Naive" assumption in the Naive Bayes classifier?',
        options: [
          'All features are conditionally independent of each other given the class label.',
          'The features always follow a Gaussian normal distribution.',
          'Prior probabilities are equal for all classes.',
          'The data is linearly separable.'
        ],
        correctAnswer: 'All features are conditionally independent of each other given the class label.',
        explanation: 'It assumes that the presence of one feature does not influence the probability of another feature, given the class label, which is rarely true but simplifies calculations.'
      },
      {
        id: 'nb_q2',
        question: 'Why is Laplace smoothing applied during Naive Bayes training?',
        options: [
          'To prevent zero probabilities for unseen features from canceling the entire product.',
          'To scale features between 0 and 1.',
          'To reduce variance of support vectors.',
          'To make the decision boundary continuous.'
        ],
        correctAnswer: 'To prevent zero probabilities for unseen features from canceling the entire product.',
        explanation: 'If a feature never appeared with class y, P(x_i|y) = 0, which zeroes out the cumulative product. Laplace smoothing adds dummy counts.'
      },
      {
        id: 'nb_q3',
        question: 'Which variant of Naive Bayes is designed for continuous variables?',
        options: ['Gaussian Naive Bayes', 'Multinomial Naive Bayes', 'Bernoulli Naive Bayes', 'Bivariate Naive Bayes'],
        correctAnswer: 'Gaussian Naive Bayes',
        explanation: 'Gaussian Naive Bayes assumes continuous features are normally distributed within each class, calculating likelihoods via Gaussian probability density function.'
      }
    ],
    coding: {
      tutorial: {
      title: 'Calculate Prior Probabilities',
      description: 'Calculate prior probabilities P(y) for each class in a label array.',
      pseudoCode: "",
        starterCode: `import numpy as np

def calculate_priors(labels):
    # TODO: Return prior probability array [P(y=0), P(y=1)]
    return np.array([0.0, 0.0])

lbls = np.array([0, 1, 1, 1, 0])
print("Priors:", calculate_priors(lbls))`,
        expectedOutput: 'Priors: [0.4 0.6]',
        solution: `import numpy as np

def calculate_priors(labels):
    counts = np.bincount(labels)
    return counts / len(labels)

lbls = np.array([0, 1, 1, 1, 0])
print("Priors:", calculate_priors(lbls))`,
        hints: ['Use np.bincount to count labels.', 'Divide counts by total length of labels array.'],
        testKeywords: ['bincount', 'len']
      },
      project: {
      title: 'Gaussian Likelihood Probability Density',
      description: 'Calculate the Gaussian probability density P(x_i | y) given mean, variance, and feature value.',
      pseudoCode: "",
        starterCode: `import numpy as np

def gaussian_likelihood(x, mean, var):
    # Formula: (1 / sqrt(2 * pi * var)) * exp(- (x - mean)^2 / (2 * var))
    # TODO: Calculate likelihood
    return 0.0

print("Likelihood:", np.round(gaussian_likelihood(2.0, 1.5, 0.5), 3))`,
        expectedOutput: 'Likelihood: 0.438',
        solution: `import numpy as np

def gaussian_likelihood(x, mean, var):
    eps = 1e-9
    denom = np.sqrt(2 * np.pi * var + eps)
    num = np.exp(-((x - mean) ** 2) / (2 * var + eps))
    return num / denom

print("Likelihood:", np.round(gaussian_likelihood(2.0, 1.5, 0.5), 3))`,
        hints: ['Use np.pi for mathematical pi constant.', 'Use np.exp for Euler exponentiation.', 'Add a small epsilon term to prevent division by zero.'],
        testKeywords: ['np.pi', 'np.exp', 'np.sqrt']
      },
      assignment: {
      title: 'Posterior Classifier Predictor',
      description: 'Predict the final class by multiplying priors and Gaussian feature likelihoods.',
      pseudoCode: "",
        starterCode: `import numpy as np

def predict_class(X, priors, means, vars):
    # X: (d,) features
    # priors: [P(y=0), P(y=1)]
    # means: (2, d) - mean features per class
    # vars: (2, d) - variance features per class
    # TODO: Calculate posterior numerators, return class with max probability
    return 0

priors = np.array([0.4, 0.6])
means = np.array([[1.0], [3.0]])
vars = np.array([[0.5], [0.5]])
X = np.array([2.5])
print("Predicted Class:", predict_class(X, priors, means, vars))`,
        expectedOutput: 'Predicted Class: 1',
        solution: `import numpy as np

def gaussian_pdf(x, m, v):
    return (1.0 / np.sqrt(2 * np.pi * v)) * np.exp(-((x - m)**2)/(2*v))

def predict_class(X, priors, means, vars):
    posteriors = []
    for c in range(2):
        prior = np.log(priors[c])
        # Log-likelihood addition to prevent numeric underflow
        likelihood = np.sum(np.log(gaussian_pdf(X, means[c], vars[c])))
        posteriors.append(prior + likelihood)
    return np.argmax(posteriors)

priors = np.array([0.4, 0.6])
means = np.array([[1.0], [3.0]])
vars = np.array([[0.5], [0.5]])
X = np.array([2.5])
print("Predicted Class:", predict_class(X, priors, means, vars))`,
        hints: ['To avoid floating underflow, sum log-likelihoods instead of multiplying raw probabilities.', 'Use np.argmax to select final class index.'],
        testKeywords: ['argmax', 'np.log', 'priors[']
      }
    },
    interviewQuestions: [{
      question: 'Explain what numeric underflow is in Naive Bayes and how to solve it. What is the impact of violating the independence assumption?',
      answer: 'Numeric underflow occurs in Naive Bayes when multiplying many probabilities (value range [0, 1]) together, causing the product to decay to zero, exceeding float capacity. We resolve this by converting equations to Log-Space, transforming products into sums of log probabilities: log P(y|X) = log P(y) + sum(log P(x_i|y)). Violating the independence assumption typically results in overconfident posterior estimates, pulling probabilities closer to 0 or 1, although classification boundaries often remain highly accurate.',
      companyTags: ['Google', 'Bloomberg'],
      difficulty: 'Advanced'
    }]
  },
  {
    id: 'knn-classification',
    title: 'K-Nearest Neighbors',
    category: 'Supervised Learning',
    description: 'Classify points based on class majorities of closest neighbors.',
    formula: 'd(p, q) = \\sqrt{\\sum (p_i - q_i)^2}',
    theory: `### K-Nearest Neighbors (KNN) & Distance Metrics

KNN is a non-parametric instance-based lazy learner. It stores all training samples and delays generalization calculations until querying.

#### Distance Calculation
Typically, distance is computed using Euclidean Distance:
$$d(x, x') = \\sqrt{\\sum_{j=1}^{d} (x_j - x'_j)^2}$$
Alternative distance metrics include:
- Manhattan Distance ($L_1$ Norm):
$$d(x, x') = \\sum_{j=1}^{d} |x_j - x'_j|$$
- Minkowski Distance (generalized norm):
$$d(x, x') = \\left( \\sum_{j=1}^{d} |x_j - x'_j|^p \\right)^{1/p}$$

#### Classification Decision Rule
Locate the $K$ points in the training set closest to test point $x_{test}$ (represented by $N_K(x_{test})$). The output class is selected via majority vote:
$$\\hat{y} = \\arg\\max_{c} \\sum_{i \\in N_K(x_{test})} \\mathbb{I}(y^{(i)} = c)$$
where $\\mathbb{I}$ is the indicator function.`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'knn',
    quiz: [
      {
        id: 'knn_q1',
        question: 'Why is KNN referred to as a "lazy learner"?',
        options: [
          'It bypasses parameter training, storing datasets and delaying all computations until prediction.',
          'It converges extremely slowly during training cycles.',
          'It is written in low-performance script code.',
          'It performs poor classification on linear data.'
        ],
        correctAnswer: 'It bypasses parameter training, storing datasets and delaying all computations until prediction.',
        explanation: 'KNN requires no training phase. It simply stores vectors, deferring distance calculations until a test sample is evaluated.'
      },
      {
        id: 'knn_q2',
        question: 'How does selecting a very small value for neighbor count K affect bias and variance?',
        options: [
          'High variance, low bias (prone to overfitting noisy structures).',
          'High bias, low variance (smooth boundaries).',
          'Low variance, low bias.',
          'No impact on variance.'
        ],
        correctAnswer: 'High variance, low bias (prone to overfitting noisy structures).',
        explanation: 'At K=1, predictions match the single nearest point. Any outlier outlier flips predictions, creating noisy, high-variance decision boundaries.'
      },
      {
        id: 'knn_q3',
        question: 'Why is feature scaling (normalization) critical before using KNN?',
        options: [
          'Features with large numeric ranges will dominate distance calculations.',
          'It transforms non-linear relationships to linear relationships.',
          'Scaling removes duplicate data points.',
          'It is required to compute log likelihoods.'
        ],
        correctAnswer: 'Features with large numeric ranges will dominate distance calculations.',
        explanation: 'Because KNN uses spatial distance metrics, unscaled features (like salary: $100,000 vs age: 25) dominate distance, rendering smaller range variables irrelevant.'
      }
    ],
    coding: {
      tutorial: {
      title: 'Minkowski Distance Calculator',
      description: 'Implement Minkowski distance formula for vectors p and q with scale p_norm.',
      pseudoCode: "",
        starterCode: `import numpy as np

def minkowski_distance(a, b, p_norm):
    # TODO: Calculate (sum(|a - b| ** p_norm)) ** (1 / p_norm)
    return 0.0

a = np.array([1.0, 3.0])
b = np.array([2.0, 1.0])
print("Distance:", np.round(minkowski_distance(a, b, 2), 3))`,
        expectedOutput: 'Distance: 2.236',
        solution: `import numpy as np

def minkowski_distance(a, b, p_norm):
    return np.sum(np.abs(a - b) ** p_norm) ** (1 / p_norm)

a = np.array([1.0, 3.0])
b = np.array([2.0, 1.0])
print("Distance:", np.round(minkowski_distance(a, b, 2), 3))`,
        hints: ['Compute absolute differences with np.abs.', 'Exponentiate by p_norm, sum all elements, and take root.'],
        testKeywords: ['np.abs', 'np.sum', '** (1 /']
      },
      project: {
      title: 'Find K-Nearest Neighbors',
      description: 'Compute distances between a test point and all rows in X, returning the indices of the closest K items.',
      pseudoCode: "",
        starterCode: `import numpy as np

def get_k_neighbors(X, test_point, k):
    # TODO: Compute distance array, sort indices, return first K indices
    indices = np.array([])
    return indices

X = np.array([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [1.5, 1.5]])
test_point = np.array([1.2, 1.2])
print("Indices:", get_k_neighbors(X, test_point, 2))`,
        expectedOutput: 'Indices: [0 3]',
        solution: `import numpy as np

def get_k_neighbors(X, test_point, k):
    dists = np.sqrt(np.sum((X - test_point) ** 2, axis=1))
    return np.argsort(dists)[:k]

X = np.array([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [1.5, 1.5]])
test_point = np.array([1.2, 1.2])
print("Indices:", get_k_neighbors(X, test_point, 2))`,
        hints: ['Calculate distance array to each row.', 'Use np.argsort to sort indices by distance value.', 'Slice the first k elements.'],
        testKeywords: ['argsort', 'np.sum', 'axis=1']
      },
      assignment: {
      title: 'KNN Majority Voter',
      description: 'Predict the class of a query vector by gathering closest neighbor labels and performing voting.',
      pseudoCode: "",
        starterCode: `import numpy as np

def knn_predict(X_train, y_train, test_point, k):
    # TODO: Find closest K neighbor indices. Get their labels from y_train, return most common label
    return 0

X_train = np.array([[1.0, 2.0], [0.0, 1.0], [2.0, 3.0], [1.0, 1.0]])
y_train = np.array([0, 1, 0, 1])
test_point = np.array([1.2, 1.8])
print("Class Prediction:", knn_predict(X_train, y_train, test_point, 3))`,
        expectedOutput: 'Class Prediction: 0',
        solution: `import numpy as np

def knn_predict(X_train, y_train, test_point, k):
    dists = np.sqrt(np.sum((X_train - test_point) ** 2, axis=1))
    k_indices = np.argsort(dists)[:k]
    neighbor_labels = y_train[k_indices]
    counts = np.bincount(neighbor_labels)
    return np.argmax(counts)

X_train = np.array([[1.0, 2.0], [0.0, 1.0], [2.0, 3.0], [1.0, 1.0]])
y_train = np.array([0, 1, 0, 1])
test_point = np.array([1.2, 1.8])
print("Class Prediction:", knn_predict(X_train, y_train, test_point, 3))`,
        hints: ['Calculate spatial distances first.', 'Retrieve target values using the argsort index slice.', 'Use np.bincount and argmax to pick the winner.'],
        testKeywords: ['np.bincount', 'np.argmax', 'np.argsort']
      }
    },
    interviewQuestions: [{
      question: 'Explain the "Curse of Dimensionality" in relation to K-Nearest Neighbors, and how it impacts distance comparisons.',
      answer: 'In high-dimensional spaces, the volume of coordinate space increases exponentially, making data points extremely sparse. Consequently, the distance between any two points converges to the same value (the ratio of the difference between minimum and maximum distance and minimum distance approaches 0). This causes KNN to fail because neighbors are no longer locally relevant; all points appear almost equally distant from the query vector.',
      companyTags: ['Amazon', 'Google'],
      difficulty: 'Advanced'
    }]
  },
  {
    id: 'pca-reduction',
    title: 'Principal Component Analysis',
    category: 'Unsupervised Learning',
    description: 'Project high-dimensional vectors onto orthogonal axes of maximal variance.',
    formula: 'Sigma = \\frac{1}{m} X^T X',
    theory: `### Principal Component Analysis (PCA) Math & Projections

PCA projects high-dimensional matrices onto low-dimensional spaces while preserving maximum variance.

#### Steps to Derive Principal Components
1. **Data Normalization**: Center inputs to mean zero:
$$X_{centered} = X - \\mu$$
2. **Covariance Matrix Calculation**: Compute the covariance matrix $\\Sigma$:
$$\\Sigma = \\frac{1}{m} X_{centered}^T X_{centered}$$
3. **Eigendecomposition**: Calculate eigenvectors $W$ and eigenvalues $\\lambda$ of $\\Sigma$:
$$\\Sigma v = \\lambda v$$
4. **Projection Step**: Project centered data matrix onto the top $k$ eigenvectors (principal components):
$$Z = X_{centered} W_k$$
where $W_k$ represents the $k$ eigenvectors with the largest eigenvalues.`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'pca',
    quiz: [
      {
        id: 'pca_q1',
        question: 'Why must dataset columns be mean-centered before performing PCA calculations?',
        options: [
          'To ensure the covariance matrix represents variance around the origin.',
          'To scale features to exactly 1.0 maximum value.',
          'To avoid matrix singularity constraints.',
          'To convert categorical labels into numeric integers.'
        ],
        correctAnswer: 'To ensure the covariance matrix represents variance around the origin.',
        explanation: 'If data is not centered, the first principal component will pull toward the dataset mean offset rather than aligning with the axis of maximum variance.'
      },
      {
        id: 'pca_q2',
        question: 'What mathematical properties describe the relationship between principal components in PCA?',
        options: [
          'They are orthogonal (independent) vectors.',
          'They are parallel vectors.',
          'They are non-linear mappings.',
          'They always match training labels.'
        ],
        correctAnswer: 'They are orthogonal (independent) vectors.',
        explanation: 'Principal components are eigenvectors of a symmetric covariance matrix, making them mutually perpendicular (orthogonal) and preventing redundant overlap.'
      },
      {
        id: 'pca_q3',
        question: 'How do you measure how much information is retained after dropping low-eigenvalue components?',
        options: [
          'Explained Variance Ratio',
          'Mean Squared Reconstruction Loss',
          'Silhouette scores',
          'L1 weight sparsity'
        ],
        correctAnswer: 'Explained Variance Ratio',
        explanation: 'The Explained Variance Ratio divides an eigenvalue by the sum of all eigenvalues. It represents the proportion of variance captured by each principal component.'
      }
    ],
    coding: {
      tutorial: {
      title: 'Mean Centering Data',
      description: 'Subtract the mean of each column from the data matrix.',
      pseudoCode: "",
        starterCode: `import numpy as np

def center_data(X):
    # TODO: Subtract column means from X
    return X

X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
print("Centered:\n", center_data(X))`,
        expectedOutput: 'Centered:\n [[-2. -2.]\n [ 0.  0.]\n [ 2.  2.]]',
        solution: `import numpy as np

def center_data(X):
    return X - np.mean(X, axis=0)

X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
print("Centered:\n", center_data(X))`,
        hints: ['Calculate mean using np.mean.', 'Set axis=0 to calculate means per column.', 'Subtract result directly from X.'],
        testKeywords: ['np.mean', 'axis=0', 'X -']
      },
      project: {
      title: 'Compute Covariance Matrix',
      description: 'Calculate the covariance matrix of centered feature observations.',
      pseudoCode: "",
        starterCode: `import numpy as np

def compute_covariance(X_centered):
    m = X_centered.shape[0]
    # TODO: Compute 1/m * X_centered.T * X_centered
    cov = np.zeros((X_centered.shape[1], X_centered.shape[1]))
    return cov

X_c = np.array([[-1.0, -1.0], [1.0, 1.0]])
print("Covariance:\n", compute_covariance(X_c))`,
        expectedOutput: 'Covariance:\n [[1. 1.]\n [1. 1.]]',
        solution: `import numpy as np

def compute_covariance(X_centered):
    m = X_centered.shape[0]
    return (1 / m) * np.dot(X_centered.T, X_centered)

X_c = np.array([[-1.0, -1.0], [1.0, 1.0]])
print("Covariance:\n", compute_covariance(X_c))`,
        hints: ['Use np.dot to multiply the transpose X_centered.T with X_centered.', 'Divide the result by the sample size m.'],
        testKeywords: ['np.dot', 'X_centered.T', 'm']
      },
      assignment: {
      title: 'Project Data onto Components',
      description: 'Perform PCA dimensionality reduction by projecting centered data onto top K eigenvectors.',
      pseudoCode: "",
        starterCode: `import numpy as np

def pca_project(X, k):
    # 1. Mean-center data
    X_c = X - np.mean(X, axis=0)
    # 2. Covariance matrix
    cov = np.dot(X_c.T, X_c) / X.shape[0]
    # 3. Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # TODO: Sort eigenvectors descending by eigenvalues, pick top k, project X_c
    return np.array([])

X = np.array([[1.0, 2.0], [3.0, 3.2], [5.0, 4.0]])
print("Projected:\n", np.round(pca_project(X, 1), 2))`,
        expectedOutput: 'Projected:\n [[-2.18]\n [ 0.05]\n [ 2.12]]',
        solution: `import numpy as np

def pca_project(X, k):
    X_c = X - np.mean(X, axis=0)
    cov = np.dot(X_c.T, X_c) / X.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort indices in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_k_vectors = eigenvectors[:, sorted_indices[:k]]
    
    # Project data
    return np.dot(X_c, top_k_vectors)

X = np.array([[1.0, 2.0], [3.0, 3.2], [5.0, 4.0]])
print("Projected:\n", np.round(pca_project(X, 1), 2))`,
        hints: ['Use np.argsort to get indices of eigenvalues.', 'Reverse order with [::-1] to get descending order.', 'Multiply X_c with the selected top eigenvectors.'],
        testKeywords: ['np.dot', 'argsort', '[::-1]']
      }
    },
    interviewQuestions: [{
      question: 'Under what circumstances does PCA fail to capture structure, and how do Kernel PCA and t-SNE resolve this?',
      answer: 'PCA fails when the underlying data manifold is non-linear (like a Swiss Roll or concentric rings), because linear projections collapse non-linear curves together. Kernel PCA resolves this by using the kernel trick to implicitly project data to a high-dimensional space where it becomes linearly separable before applying PCA. t-SNE preserves local probability neighborhoods in a low-dimensional manifold instead of global variances, making it superior for cluster visualization.',
      companyTags: ['Netflix', 'Google'],
      difficulty: 'Advanced'
    }]
  },
  {
    id: 'gradient-descent',
    title: 'Gradient Descent & Optimizers',
    category: 'Foundations & Math',
    description: 'Navigate loss landscapes iteratively using derivatives, momentum, and adaptive scales.',
    formula: 'w_{t+1} = w_t - \\eta \\nabla L(w_t)',
    theory: `### Gradient Descent Mechanics & Optimizers

Optimizers adjust weights to minimize a loss function $L(w)$ by stepping along its negative gradient.

#### Vanilla Gradient Descent
Parameters are updated using a fixed learning rate $\\eta$:
$$w_{t+1} = w_t - \\eta \\nabla L(w_t)$$

#### Gradient Descent with Momentum
Adds a fraction $\\beta$ of the past update velocity $v_t$ to accelerate through flat regions and dampen oscillations:
$$v_t = \\beta v_{t-1} + \\eta \\nabla L(w_t)$$
$$w_{t+1} = w_t - v_t$$

#### Adaptive Momentum Estimation (Adam)
Adam combines momentum (first moments $m_t$) and adaptive learning rates (second moments $v_t$) with bias correction:
$$m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t, \\quad v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2$$
$$\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\quad \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}$$
$$w_{t+1} = w_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t$$`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'gradient-descent',
    quiz: [
      {
        id: 'gd_q1',
        question: 'Why does adding Momentum help gradient descent optimization converge faster in steep valleys?',
        options: [
          'It dampens oscillations perpendicular to the valley path while accelerating updates along the valley.',
          'It changes the learning rate dynamically based on gradient size.',
          'It makes the loss function convex.',
          'It reduces parameter sizes.'
        ],
        correctAnswer: 'It dampens oscillations perpendicular to the valley path while accelerating updates along the valley.',
        explanation: 'By averaging velocity vectors, orthogonal components cancel out while parallel components accumulate, guiding updates down the center of the valley.'
      },
      {
        id: 'gd_q2',
        question: 'What is the function of the epsilon term in adaptive optimizers like RMSprop and Adam?',
        options: [
          'To prevent division by zero when the rolling second-moment gradient approaches zero.',
          'To scale up learning rates.',
          'To calculate gradients.',
          'To control momentum decay.'
        ],
        correctAnswer: 'To prevent division by zero when the rolling second-moment gradient approaches zero.',
        explanation: 'The epsilon constant (typically 1e-8) is added to the denominator before division, preventing math errors when features have flat gradients.'
      },
      {
        id: 'gd_q3',
        question: 'Which optimizer utilizes rolling averages of squared gradients without using momentum?',
        options: ['RMSprop', 'SGD', 'AdaGrad', 'Vanilla Gradient Descent'],
        correctAnswer: 'RMSprop',
        explanation: 'RMSprop addresses AdaGrad\'s decaying learning rate by using an exponentially decaying average of squared gradients, stabilizing updates.'
      }
    ],
    coding: {
      tutorial: {
      title: 'Vanilla Gradient Update',
      description: 'Update parameter w using gradient g and learning rate eta.',
      pseudoCode: "",
        starterCode: `import numpy as np

def van_update(w, g, eta):
    # TODO: Update w
    return w

print("Updated:", van_update(1.5, 0.2, 0.1))`,
        expectedOutput: 'Updated: 1.48',
        solution: `import numpy as np

def van_update(w, g, eta):
    return w - eta * g

print("Updated:", van_update(1.5, 0.2, 0.1))`,
        hints: ['Subtract learning rate eta * gradient g from w.'],
        testKeywords: ['w -', 'eta * g']
      },
      project: {
      title: 'Momentum Update Rule',
      description: 'Implement a momentum-assisted update step, returning the new velocity and updated parameter.',
      pseudoCode: "",
        starterCode: `import numpy as np

def momentum_update(w, g, v, beta, eta):
    # TODO: Update velocity: v = beta * v + eta * g
    # Update weight: w = w - v
    new_v = 0.0
    new_w = w
    return new_w, new_v

print("nw, nv:", momentum_update(1.0, 0.1, 0.5, 0.9, 0.1))`,
        expectedOutput: 'nw, nv: (0.54, 0.46)',
        solution: `import numpy as np

def momentum_update(w, g, v, beta, eta):
    new_v = beta * v + eta * g
    new_w = w - new_v
    return new_w, new_v

print("nw, nv:", momentum_update(1.0, 0.1, 0.5, 0.9, 0.1))`,
        hints: ['Calculate updated velocity vector v first.', 'Subtract updated v from w.', 'Return both values.'],
        testKeywords: ['beta * v', 'w - new_v']
      },
      assignment: {
      title: 'Adam Adaptive Optimizer Update',
      description: 'Calculate updated parameter coordinates using Adam updates (including first and second moments and bias correction).',
      pseudoCode: "",
        starterCode: `import numpy as np

def adam_update(w, g, m, v, t, beta1=0.9, beta2=0.999, eta=0.01, eps=1e-8):
    # TODO: Update m and v. Calculate bias-corrected m_hat and v_hat. Update w.
    # Return w, m, v
    return w, m, v

w, m, v = 1.0, 0.0, 0.0
w, m, v = adam_update(w, 0.5, m, v, 1)
print("Updated w:", np.round(w, 4))`,
        expectedOutput: 'Updated w: 0.99',
        solution: `import numpy as np

def adam_update(w, g, m, v, t, beta1=0.9, beta2=0.999, eta=0.01, eps=1e-8):
    m_new = beta1 * m + (1 - beta1) * g
    v_new = beta2 * v + (1 - beta2) * (g ** 2)
    
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)
    
    w_new = w - (eta / (np.sqrt(v_hat) + eps)) * m_hat
    return w_new, m_new, v_new

w, m, v = 1.0, 0.0, 0.0
w, m, v = adam_update(w, 0.5, m, v, 1)
print("Updated w:", np.round(w, 4))`,
        hints: ['Calculate moment updates m and v.', 'Apply bias correction (1 - beta^t) using epoch step t.', 'Use np.sqrt to compute the square root of v_hat.'],
        testKeywords: ['beta1 * m', 'beta2 * v', 'np.sqrt', 'beta1 ** t']
      }
    },
    interviewQuestions: [{
      question: 'Why do we need bias correction in Adam optimization, and what occurs during early training steps without it?',
      answer: 'Moments m and v are initialized as zeros. During early steps (especially when beta2 is near 1, e.g. 0.999), updates are heavily biased toward zero. The bias correction terms (1 - beta1^t) and (1 - beta2^t) scale up early updates. As t increases, these terms approach 1, phasing out the correction once moments stabilize.',
      companyTags: ['Meta', 'OpenAI'],
      difficulty: 'Advanced'
    }]
  },
  {
    id: 'decision-tree-algorithm', // Dummy ID for matching the 25 modules
    title: 'Decision Tree Splits',
    category: 'Supervised Learning',
    description: 'Explore recursive feature splitting in tree nodes.',
    formula: 'IG(D, A) = H(D) - H(D|A)',
    theory: `### Decision Tree Splits Theory
This module expands on the recursive binary partitioning algorithm. At each level of the tree, we greedily select features to split dataset samples into left and right nodes.`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'tree-splits',
    quiz: [
      { id: 'dta_q1', question: 'How is split quality evaluated?', options: ['Entropy/Gini', 'MSE only', 'Silhouette', 'L1 norm'], correctAnswer: 'Entropy/Gini', explanation: 'Decision trees evaluate split quality using impurity reduction metrics like Gini or Entropy.' }
    ],
    coding: {
      tutorial: {
      title: 'Split Check',
      description: 'Verify if labels are pure (all 0 or all 1).',
      pseudoCode: "",
        starterCode: `def is_pure(y):
    # TODO: Return True if all elements in y are identical
    return False

print(is_pure([1, 1, 1]))`,
        expectedOutput: 'True',
        solution: `def is_pure(y):
    return len(set(y)) <= 1

print(is_pure([1, 1, 1]))`,
        hints: ['Convert y to a set and check its length.'],
        testKeywords: ['set']
      },
      project: {
      title: 'Binary Partitioning helper',
      description: 'Split data array into two groups based on a feature threshold.',
      pseudoCode: "",
        starterCode: `import numpy as np
def partition(X, feature_idx, threshold):
    # TODO: return left_indices, right_indices
    return [], []

X = np.array([[1.5], [2.5], [3.5]])
print(partition(X, 0, 2.0))`,
        expectedOutput: '(array([0]), array([1, 2]))',
        solution: `import numpy as np
def partition(X, feature_idx, threshold):
    left = np.where(X[:, feature_idx] <= threshold)[0]
    right = np.where(X[:, feature_idx] > threshold)[0]
    return left, right

X = np.array([[1.5], [2.5], [3.5]])
print(partition(X, 0, 2.0))`,
        hints: ['Use np.where with feature columns and thresholds.'],
        testKeywords: ['np.where']
      },
      assignment: {
      title: 'Tree Depth Calculator',
      description: 'Predict depth of a binary tree given tree height parameter.',
      pseudoCode: "",
        starterCode: `def max_nodes(depth):
    # TODO: return 2^depth - 1
    return 0

print(max_nodes(3))`,
        expectedOutput: '7',
        solution: `def max_nodes(depth):
    return (2 ** depth) - 1

print(max_nodes(3))`,
        hints: ['Use exponentiation **.'],
        testKeywords: ['2 **']
      }
    },
    interviewQuestions: [{
      question: 'What is cost-complexity pruning and how does it balance tree complexity and performance?',
      answer: 'Cost-complexity pruning minimizes R(T) + alpha * |T|, where R(T) is the training error and |T| is the leaf count. This penalizes large trees, pruning branches that do not reduce error enough to justify the complexity penalty alpha.',
      companyTags: ['Google'],
      difficulty: 'Advanced'
    }]
  },
  {
    id: 'gradient-boosting',
    title: 'Gradient Boosting & XGBoost',
    category: 'Supervised Learning',
    description: 'Ensemble weak learners sequentially by fitting residuals to correct past errors.',
    formula: 'F_m(x) = F_{m-1}(x) + \\gamma_m h_m(x)',
    theory: `### Gradient Boosting Theory & XGBoost Mechanics

Gradient Boosting models are trained sequentially, with each new model fitting the residuals (errors) of the previous ensemble.

#### General Boosting Equation
An ensemble of $M$ estimators predicts outputs as:
$$F_M(x) = \\sum_{m=1}^{M} \\gamma_m h_m(x)$$

#### Residual Fitting
Each step fits a weak learner $h_m(x)$ to the pseudo-residuals, defined as the negative gradient of the loss function:
$$r_{im} = -\\left[ \\frac{\\partial L(y_i, F(x_i))}{\\partial F(x_i)} \\right]_{F(x) = F_{m-1}(x)}$$

#### XGBoost Optimizations
XGBoost minimizes a regularized objective function using a second-order Taylor expansion:
$$\\mathcal{L}^{(t)} \\approx \\sum_{i=1}^{n} \\left[ g_i f_t(x_i) + \\frac{1}{2} h_i f_t^2(x_i) \\right] + \\gamma T + \\frac{1}{2} \\lambda \\sum_{j=1}^{T} w_j^2$$
where $g_i$ and $h_i$ are the first and second-order derivatives (gradients and Hessians) of the loss function.`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'gradient-boosting',
    quiz: [
      {
        id: 'gb_q1',
        question: 'What do subsequent weak learners fit in a gradient boosting architecture?',
        options: [
          'The negative gradient (residuals) of the loss function.',
          'A random bootstrap sample of training labels.',
          'Only the misclassified support vectors.',
          'The original target values directly.'
        ],
        correctAnswer: 'The negative gradient (residuals) of the loss function.',
        explanation: 'Instead of reweighting samples (like AdaBoost), Gradient Boosting trains trees on the remaining residuals (errors) of the current ensemble.'
      },
      {
        id: 'gb_q2',
        question: 'What optimization does XGBoost utilize that separates it from standard Gradient Boosting?',
        options: [
          'Second-order Taylor expansion (Hessians) and regularization constraints.',
          'Deep neural networks as weak learners.',
          'Linear regression initializations.',
          'Closed-form normal equations.'
        ],
        correctAnswer: 'Second-order Taylor expansion (Hessians) and regularization constraints.',
        explanation: 'XGBoost uses both first-order gradients and second-order Hessians along with tree complexity penalties to optimize splits.'
      },
      {
        id: 'gb_q3',
        question: 'What is the function of the learning rate (shrinkage) in Gradient Boosting?',
        options: [
          'It scales the contribution of each new tree to reduce variance.',
          'It sets the maximum tree depth.',
          'It controls bootstrap sample sizes.',
          'It scales features to prevent gradients from vanishing.'
        ],
        correctAnswer: 'It scales the contribution of each new tree to reduce variance.',
        explanation: 'Scaling tree outputs by a shrinkage factor (eta < 0.1) requires more trees to fit the training set, which prevents overfitting and improves generalization.'
      }
    ],
    coding: {
      tutorial: {
      title: 'Calculate Pseudo-Residuals',
      description: 'Calculate the pseudo-residuals (y_true - y_pred) for a regression task.',
      pseudoCode: "",
        starterCode: `import numpy as np

def get_residuals(y_true, y_pred):
    # TODO: Calculate residuals
    return np.array([])

y_t = np.array([10.0, 15.0, 12.0])
y_p = np.array([9.5, 15.5, 11.0])
print("Residuals:", get_residuals(y_t, y_p))`,
        expectedOutput: 'Residuals: [ 0.5 -0.5  1. ]',
        solution: `import numpy as np

def get_residuals(y_true, y_pred):
    return y_true - y_pred

y_t = np.array([10.0, 15.0, 12.0])
y_p = np.array([9.5, 15.5, 11.0])
print("Residuals:", get_residuals(y_t, y_p))`,
        hints: ['Subtract y_pred from y_true.'],
        testKeywords: ['y_true - y_pred']
      },
      project: {
      title: 'XGBoost Gradient & Hessian Calc',
      description: 'Compute first-order gradient (g) and second-order Hessian (h) for Squared Error loss.',
      pseudoCode: "",
        starterCode: `import numpy as np

def grad_hess_se(y_true, y_pred):
    # Loss = 0.5 * (y_pred - y_true)^2
    # g = df/dypred = (y_pred - y_true)
    # h = d^2f/dypred^2 = 1.0
    # TODO: Return g, h
    g = np.array([])
    h = np.array([])
    return g, h

y_t = np.array([2.0, 4.0])
y_p = np.array([1.5, 4.5])
g, h = grad_hess_se(y_t, y_p)
print("g:", g, "h:", h)`,
        expectedOutput: 'g: [-0.5  0.5] h: [1. 1.]',
        solution: `import numpy as np

def grad_hess_se(y_true, y_pred):
    g = y_pred - y_true
    h = np.ones_like(y_true, dtype=float)
    return g, h

y_t = np.array([2.0, 4.0])
y_p = np.array([1.5, 4.5])
g, h = grad_hess_se(y_t, y_p)
print("g:", g, "h:", h)`,
        hints: ['Subtract y_true from y_pred for g.', 'Return a vector of ones matching y_true length for h.'],
        testKeywords: ['y_pred - y_true', 'ones_like']
      },
      assignment: {
      title: 'GBM Prediction Aggregation',
      description: 'Calculate predictions from a sequential ensemble, applying a learning rate (shrinkage).',
      pseudoCode: "",
        starterCode: `import numpy as np

def gbm_predict(base_pred, trees_preds, lr):
    # base_pred: scalar initial prediction
    # trees_preds: list of prediction vectors, one per tree
    # TODO: Add scaled tree predictions to base prediction
    # F(x) = base + lr * sum(trees)
    return np.array([])

base = 10.0
trees = [np.array([1.0, -0.5]), np.array([0.5, 0.5])]
lr = 0.1
print("Final predictions:", gbm_predict(base, trees, lr))`,
        expectedOutput: 'Final predictions: [10.15  9.99]',
        solution: `import numpy as np

def gbm_predict(base_pred, trees_preds, lr):
    ensemble_sum = np.sum(trees_preds, axis=0)
    return base_pred + lr * ensemble_sum

base = 10.0
trees = [np.array([1.0, -0.5]), np.array([0.5, 0.5])]
lr = 0.1
print("Final predictions:", gbm_predict(base, trees, lr))`,
        hints: ['Sum the tree predictions column-wise.', 'Scale the sum by the learning rate.', 'Add the scaled sum to base_pred.'],
        testKeywords: ['np.sum', 'axis=0', 'lr *']
      }
    },
    interviewQuestions: [{
      question: 'Compare Random Forests and Gradient Boosting in terms of bias-variance trade-off and speed.',
      answer: 'Random Forests use deep bagging trees to reduce variance, meaning they do not overfit as more estimators are added, and they can be trained in parallel. Gradient Boosting trains shallow trees sequentially to reduce bias, meaning it is prone to overfitting if learning rate or estimator count is too high, and training cannot be parallelized easily since each tree depends on the residuals of the previous one.',
      companyTags: ['Microsoft', 'Lyft'],
      difficulty: 'Advanced'
    }]
  },
  {
    id: 'single-perceptron',
    title: 'Single-Layer Perceptron',
    category: 'Deep Learning',
    description: 'Learn linear decision boundaries using step activation gates.',
    formula: 'y = f(w^T x + b)',
    theory: `### Perceptron Learning Rule

The Perceptron is a simple model for binary classification of linearly separable patterns.

#### Activation Function (Heaviside Step Function)
$$f(z) = \\begin{cases} 1 & z \\ge 0 \\ 0 & z < 0 \\end{cases}$$

#### The Perceptron Convergence Theorem
If the training dataset is linearly separable, the Perceptron learning rule is guaranteed to find a separating hyperplane in a finite number of steps.

#### Parameter Update Rule
For each training example $(x^{(i)}, y^{(i)})$, we compute predictions $\\hat{y}^{(i)} = f(w^T x^{(i)} + b)$. If $\\hat{y}^{(i)} \\neq y^{(i)}$, we update weights:
$$w_j \\leftarrow w_j + \\eta (y^{(i)} - \\hat{y}^{(i)}) x_j^{(i)}$$
$$b \\leftarrow b + \\eta (y^{(i)} - \\hat{y}^{(i)})$$`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'perceptron',
    quiz: [
      { id: 'slp_q1', question: 'What classification limitation does a single perceptron have?', options: ['Cannot solve XOR (non-linear splits)', 'Cannot solve OR', 'Cannot classify negative numbers', 'Slow predictions'], correctAnswer: 'Cannot solve XOR (non-linear splits)', explanation: 'Single perceptrons can only form linear boundaries, meaning they cannot solve non-linear problems like XOR.' }
    ],
    coding: {
      tutorial: {
      title: 'Heaviside Step Function',
      description: 'Implement the threshold activation: 1 if z >= 0 else 0.',
      pseudoCode: "",
        starterCode: `def step_func(z):
    # TODO: Implement step function
    return 0

print([step_func(-2), step_func(1.5)])`,
        expectedOutput: '[0, 1]',
        solution: `def step_func(z):
    return 1 if z >= 0 else 0

print([step_func(-2), step_func(1.5)])`,
        hints: ['Check if z is greater than or equal to 0.'],
        testKeywords: ['>= 0']
      },
      project: {
      title: 'Perceptron Predictor',
      description: 'Predict the class label given inputs, weights, and bias.',
      pseudoCode: "",
        starterCode: `import numpy as np
def predict(X, w, b):
    # TODO: compute sign of Xw+b
    return 0

X = np.array([1.5, -0.5])
w = np.array([0.5, 1.0])
b = -0.1
print(predict(X, w, b))`,
        expectedOutput: '1',
        solution: `import numpy as np
def predict(X, w, b):
    return 1 if np.dot(X, w) + b >= 0 else 0

X = np.array([1.5, -0.5])
w = np.array([0.5, 1.0])
b = -0.1
print(predict(X, w, b))`,
        hints: ['Use np.dot to compute the dot product.'],
        testKeywords: ['np.dot', '>= 0']
      },
      assignment: {
      title: 'Perceptron Parameter Update',
      description: 'Update perceptron parameters using training errors.',
      pseudoCode: "",
        starterCode: `import numpy as np
def update(w, b, x, y, y_pred, eta):
    # TODO: return updated w, b
    return w, b

print(update(np.array([0.5, 0.5]), 0.0, np.array([1.0, 1.0]), 1, 0, 0.1))`,
        expectedOutput: '(array([0.6, 0.6]), 0.1)',
        solution: `import numpy as np
def update(w, b, x, y, y_pred, eta):
    error = y - y_pred
    w_new = w + eta * error * x
    b_new = b + eta * error
    return w_new, b_new

print(update(np.array([0.5, 0.5]), 0.0, np.array([1.0, 1.0]), 1, 0, 0.1))`,
        hints: ['Calculate error = y - y_pred first.'],
        testKeywords: ['error', 'eta *']
      }
    },
    interviewQuestions: [{
      question: 'Explain the XOR problem and why it marked a major milestone in neural network history.',
      answer: 'The XOR problem represents a non-linearly separable truth table. A single-layer perceptron can only form linear boundaries, so it cannot solve XOR. This limitation, highlighted by Minsky and Papert in 1969, temporarily halted neural network research until multi-layer networks and backpropagation were introduced.',
      companyTags: ['Meta'],
      difficulty: 'Intermediate'
    }]
  },
  {
    id: 'multi-layer-perceptron',
    title: 'Multi-Layer Perceptron',
    category: 'Deep Learning',
    description: 'Build neural layers and train weights using backpropagation and the chain rule.',
    formula: 'a^{[l]} = \\sigma(W^{[l]} a^{[l-1]} + b^{[l]})',
    theory: `### Multi-Layer Perceptron (MLP) & Backpropagation

MLPs resolve non-linear splits by chaining layers together with non-linear activation functions.

#### Feedforward Equations
For layer $l$:
$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$
$$a^{[l]} = g^{[l]}(z^{[l]})$$
where $a^{[0]} = x$ and $g^{[l]}$ is the activation function.

#### Backpropagation: Calculating Gradients
Using the chain rule, we compute derivatives of loss $L$ with respect to parameters:
$$\\delta^{[l]} = \\frac{\\partial L}{\\partial z^{[l]}} = \\left( W^{[l+1]T} \\delta^{[l+1]} \\right) \\odot g^{[l]\\prime}(z^{[l]})$$
$$\\frac{\\partial L}{\\partial W^{[l]}} = \\delta^{[l]} a^{[l-1]T}$$
$$\\frac{\\partial L}{\\partial b^{[l]}} = \\delta^{[l]}$$`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'neural-network',
    quiz: [
      {
        id: 'mlp_q1',
        question: 'Why are non-linear activation functions necessary in hidden layers of neural networks?',
        options: [
          'Without them, the network collapses to a single linear transformation, regardless of depth.',
          'They prevent numeric overflow.',
          'They scale gradients to 1.0.',
          'They ensure all weights are positive.'
        ],
        correctAnswer: 'Without them, the network collapses to a single linear transformation, regardless of depth.',
        explanation: 'Compounding linear layers yields another linear layer. Non-linear activations allow networks to learn non-linear manifolds.'
      },
      {
        id: 'mlp_q2',
        question: 'What is the purpose of backpropagation in deep neural networks?',
        options: [
          'To calculate gradients of the loss function with respect to weights using the chain rule.',
          'To generate training predictions.',
          'To select hidden layer node sizes.',
          'To normalize input matrices.'
        ],
        correctAnswer: 'To calculate gradients of the loss function with respect to weights using the chain rule.',
        explanation: 'Backpropagation propagates prediction errors backward from the output layer to calculate parameter gradients for weight updates.'
      },
      {
        id: 'mlp_q3',
        question: 'Which activation function is defined as g(z) = max(0, z)?',
        options: ['ReLU', 'Sigmoid', 'Tanh', 'Leaky ReLU'],
        correctAnswer: 'ReLU',
        explanation: 'ReLU (Rectified Linear Unit) is max(0, z). Its constant derivative of 1.0 for positive inputs helps mitigate the vanishing gradient problem.'
      }
    ],
    coding: {
      tutorial: {
      title: 'Feedforward Single Layer',
      description: 'Compute raw scores z and activations a for a single feedforward layer.',
      pseudoCode: "",
        starterCode: `import numpy as np

def forward_layer(a_prev, W, b):
    # TODO: Compute z = W * a_prev + b, and a = ReLU(z)
    z = np.zeros(2)
    a = np.zeros(2)
    return z, a

a_prev = np.array([1.0, 0.5])
W = np.array([[0.2, 0.8], [-0.5, 1.0]])
b = np.array([0.1, -0.2])
z, a = forward_layer(a_prev, W, b)
print("z:", z, "a:", a)`,
        expectedOutput: 'z: [0.7 0. ] a: [0.7 0. ]',
        solution: `import numpy as np

def forward_layer(a_prev, W, b):
    z = np.dot(W, a_prev) + b
    a = np.maximum(0, z)
    return z, a

a_prev = np.array([1.0, 0.5])
W = np.array([[0.2, 0.8], [-0.5, 1.0]])
b = np.array([0.1, -0.2])
z, a = forward_layer(a_prev, W, b)
print("z:", z, "a:", a)`,
        hints: ['Use np.dot to multiply weights and activations.', 'Use np.maximum(0, z) to apply ReLU activation.'],
        testKeywords: ['np.dot', 'np.maximum', '+ b']
      },
      project: {
      title: 'ReLU Gradient Derivative',
      description: 'Calculate the derivative of the ReLU activation function for an array.',
      pseudoCode: "",
        starterCode: `import numpy as np

def relu_derivative(z):
    # TODO: Return 1.0 if z > 0 else 0.0
    return np.zeros_like(z)

z = np.array([-2.5, 0.0, 3.5])
print("dReLU:", relu_derivative(z))`,
        expectedOutput: 'dReLU: [0. 0. 1.]',
        solution: `import numpy as np

def relu_derivative(z):
    return np.where(z > 0, 1.0, 0.0)

z = np.array([-2.5, 0.0, 3.5])
print("dReLU:", relu_derivative(z))`,
        hints: ['Use np.where to assign 1.0 or 0.0.', 'The derivative is undefined or zero at exactly z=0.'],
        testKeywords: ['np.where', 'z > 0']
      },
      assignment: {
      title: 'Output Layer Delta Error',
      description: 'Compute delta (error gradient) at the output layer for MSE loss with Sigmoid activation.',
      pseudoCode: "",
        starterCode: `import numpy as np

def output_delta(y_true, y_pred, z_out):
    # Sigmoid derivative: s * (1 - s)
    # Loss gradient: y_pred - y_true
    # delta = (y_pred - y_true) * sigmoid_derivative(z_out)
    # TODO: Calculate delta
    return np.array([])

y_t = np.array([0.0, 1.0])
y_p = np.array([0.1, 0.8])
z_o = np.array([-2.0, 1.5])
print("Delta:", np.round(output_delta(y_t, y_p, z_o), 3))`,
        expectedOutput: 'Delta: [0.010 0.021]',
        solution: `import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def output_delta(y_true, y_pred, z_out):
    s = sigmoid(z_out)
    ds = s * (1.0 - s)
    return (y_pred - y_true) * ds

y_t = np.array([0.0, 1.0])
y_p = np.array([0.1, 0.8])
z_o = np.array([-2.0, 1.5])
print("Delta:", np.round(output_delta(y_t, y_p, z_o), 3))`,
        hints: ['Calculate Sigmoid of z_out.', 'Compute Sigmoid derivative s * (1 - s).', 'Multiply element-wise by (y_pred - y_true).'],
        testKeywords: ['sigmoid', '1.0 - s', '* ds']
      }
    },
    interviewQuestions: [{
      question: 'Detail the vanishing gradient problem. What are its root causes, and how do modern architectures solve it?',
      answer: 'The vanishing gradient problem occurs in deep networks when early layer gradients decay exponentially during backpropagation. This is caused by chaining activation derivatives smaller than 1.0 (such as Sigmoid: max 0.25, or Tanh: max 1.0 but usually smaller). We resolve this using ReLU (derivative of 1.0 for positive inputs), residual skip connections (allowing gradients to bypass layers directly), and Batch Normalization (which keeps inputs in active regions).',
      companyTags: ['Meta', 'Google'],
      difficulty: 'Advanced'
    }]
  },
  {
    id: 'convolutional-networks',
    title: 'Convolutional Neural Networks',
    category: 'Deep Learning',
    description: 'Extract spatial patterns from grids using convolution kernels and pooling.',
    formula: 'S(i,j) = (I * K)(i,j)',
    theory: `### CNN Architecture & Operations

Convolutional Neural Networks extract spatial features using weight-sharing kernels.

#### 2D Convolution Operation
The convolution of an input grid image $I$ with a kernel $K$ is computed as:
$$S(i, j) = (I * K)(i, j) = \\sum_{m} \\sum_{n} I(i-m, j-n) K(m, n)$$

#### Output Map Dimension Calculator
Given input dimension $W$, filter size $F$, padding $P$, and stride $S$:
$$W_{out} = \\left\\lfloor \\frac{W - F + 2P}{S} \\right\\rfloor + 1$$

#### Feature Pooling
Pooling reduces spatial dimensions to enforce translation invariance. Max Pooling extracts the maximum value within a window.`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'cnn',
    quiz: [
      { id: 'cnn_q1', question: 'How is the output dimension computed for an input of width 32, filter size 5, stride 1, and no padding?', options: ['28', '30', '32', '27'], correctAnswer: '28', explanation: 'Output = ((32 - 5 + 0) / 1) + 1 = 28.' }
    ],
    coding: {
      tutorial: {
      title: 'Single 2D Filter Step',
      description: 'Perform 2D convolution of a 3x3 patch with a 3x3 kernel.',
      pseudoCode: "",
        starterCode: `import numpy as np
def convolve_patch(patch, kernel):
    # TODO: return elementwise product sum
    return 0.0

p = np.ones((3, 3))
k = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]])
print(convolve_patch(p, k))`,
        expectedOutput: '6.0',
        solution: `import numpy as np
def convolve_patch(patch, kernel):
    return float(np.sum(patch * kernel))

p = np.ones((3, 3))
k = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]])
print(convolve_patch(p, k))`,
        hints: ['Multiply arrays element-wise and sum.'],
        testKeywords: ['np.sum']
      },
      project: {
      title: 'Max Pooling 2D operation',
      description: 'Compute max pooling output for a 2x2 patch.',
      pseudoCode: "",
        starterCode: `import numpy as np
def max_pool_patch(patch):
    # TODO: return max value of patch
    return 0.0

p = np.array([[1.0, 4.0], [2.0, 3.0]])
print(max_pool_patch(p))`,
        expectedOutput: '4.0',
        solution: `import numpy as np
def max_pool_patch(patch):
    return float(np.max(patch))

p = np.array([[1.0, 4.0], [2.0, 3.0]])
print(max_pool_patch(p))`,
        hints: ['Use np.max to find the maximum value.'],
        testKeywords: ['np.max']
      },
      assignment: {
      title: 'Flatten Layer Feature Map',
      description: 'Flatten a 3D feature tensor into a 1D vector.',
      pseudoCode: "",
        starterCode: `import numpy as np
def flatten(tensor):
    # TODO: return flattened 1D array
    return tensor

t = np.ones((2, 2, 3))
print(flatten(t).shape)`,
        expectedOutput: '(12,)',
        solution: `import numpy as np
def flatten(tensor):
    return tensor.flatten()

t = np.ones((2, 2, 3))
print(flatten(t).shape)`,
        hints: ['Use the .flatten() method.'],
        testKeywords: ['flatten()']
      }
    },
    interviewQuestions: [{
      question: 'What is translational invariance in CNNs, and how does pooling achieve it?',
      answer: 'Translational invariance means the network detects features regardless of their spatial shift in the image. Max pooling achieves this by selecting the maximum value in a window, meaning small shifts in input coordinates do not change the pooled output.',
      companyTags: ['Apple'],
      difficulty: 'Advanced'
    }]
  },
  {
    id: 'recurrent-networks',
    title: 'Recurrent Networks & LSTMs',
    category: 'Deep Learning',
    description: 'Process sequential observations using recurrent states and gating corridors.',
    formula: 'h_t = \\tanh(W x_t + U h_{t-1} + b)',
    theory: `### Recurrent Networks & Gated Sequences

RNNs process variable-length sequences by passing a hidden state vector $h_t$ over time.

#### RNN Cell Equation
$$h_t = \\tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$

#### The Vanishing Gradient in RNNs
Standard RNNs struggle to learn long-term dependencies because backpropagating through time (BPTT) repeatedly multiplies hidden weight matrices, causing gradients to vanish or explode.

#### LSTM gating solution
Long Short-Term Memory networks solve this by routing information through a constant-flow cell state $C_t$ controlled by three gates:
- **Forget Gate**: $f_t = \\sigma(W_f [h_{t-1}, x_t] + b_f)$
- **Input Gate**: $i_t = \\sigma(W_i [h_{t-1}, x_t] + b_i)$
- **Output Gate**: $o_t = \\sigma(W_o [h_{t-1}, x_t] + b_o)$`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'rnn',
    quiz: [
      { id: 'rnn_q1', question: 'Which gate controls how much of the past cell state memory is discarded in an LSTM cell?', options: ['Forget gate', 'Input gate', 'Output gate', 'Update gate'], correctAnswer: 'Forget gate', explanation: 'The forget gate outputs a value between 0 and 1, scaling the past cell state to discard or retain historical inputs.' }
    ],
    coding: {
      tutorial: {
      title: 'RNN Hidden Update Step',
      description: 'Update the hidden state vector using inputs, weights, and tanh activation.',
      pseudoCode: "",
        starterCode: `import numpy as np
def rnn_step(x, h_prev, Wx, Wh, b):
    # TODO: return tanh(Wx*x + Wh*h_prev + b)
    return h_prev

x = 1.0
h_prev = 0.5
print(np.round(rnn_step(x, h_prev, 0.5, 0.8, -0.1), 3))`,
        expectedOutput: '0.664',
        solution: `import numpy as np
def rnn_step(x, h_prev, Wx, Wh, b):
    return np.tanh(Wx * x + Wh * h_prev + b)

x = 1.0
h_prev = 0.5
print(np.round(rnn_step(x, h_prev, 0.5, 0.8, -0.1), 3))`,
        hints: ['Use np.tanh for the hyperbolic tangent activation.'],
        testKeywords: ['np.tanh']
      },
      project: {
      title: 'LSTM Forget Gate Activation',
      description: 'Compute the activation output of an LSTM forget gate.',
      pseudoCode: "",
        starterCode: `import numpy as np
def forget_gate(x, h_prev, Wf, bf):
    # sigmoid = 1 / (1 + exp(-z))
    # z = Wf * [h_prev, x] + bf
    # TODO: compute gate output
    return 0.0

print(np.round(forget_gate(1.0, 0.5, np.array([0.5, 0.2]), -0.1), 3))`,
        expectedOutput: '0.646',
        solution: `import numpy as np
def forget_gate(x, h_prev, Wf, bf):
    inputs = np.array([h_prev, x])
    z = np.dot(Wf, inputs) + bf
    return 1 / (1 + np.exp(-z))

print(np.round(forget_gate(1.0, 0.5, np.array([0.5, 0.2]), -0.1), 3))`,
        hints: ['Combine inputs, compute dot product, and apply sigmoid.'],
        testKeywords: ['np.dot', 'np.exp']
      },
      assignment: {
      title: 'LSTM Cell State Update',
      description: 'Update the LSTM cell state C_t using gate activations.',
      pseudoCode: "",
        starterCode: `def update_cell_state(C_prev, f_t, i_t, C_tilde):
    # TODO: C_t = f_t * C_prev + i_t * C_tilde
    return 0.0

print(update_cell_state(2.0, 0.9, 0.1, 0.5))`,
        expectedOutput: '1.85',
        solution: `def update_cell_state(C_prev, f_t, i_t, C_tilde):
    return f_t * C_prev + i_t * C_tilde

print(update_cell_state(2.0, 0.9, 0.1, 0.5))`,
        hints: ['Implement the weighted sum of past cell state and candidate state.'],
        testKeywords: ['f_t *', 'i_t *']
      }
    },
    interviewQuestions: [{
      question: 'How do LSTMs prevent vanishing gradients over long sequences compared to standard RNNs?',
      answer: 'LSTMs introduce a cell state pathway with constant error carousel updates. The forget gate determines how much information is retained. Because the cell state update is linear (C_t = f_t * C_prev + i_t * C_tilde), derivatives can flow backward without being repeatedly multiplied by weights, mitigating vanishing gradients.',
      companyTags: ['Google'],
      difficulty: 'Advanced'
    }]
  },
  {
    id: 'transformer-attention',
    title: 'Transformer Self-Attention',
    category: 'Deep Learning',
    description: 'Link tokens in context using Query, Key, and Value projection matrix scores.',
    formula: 'Attention(Q,K,V) = Softmax(\\frac{QK^T}{\\sqrt{d_k}}) V',
    theory: `### Transformer Self-Attention & QKV Projection

Self-Attention allows models to score relationships between words in a sequence simultaneously.

#### Attention Equation
$$Attention(Q, K, V) = Softmax \\left( \\frac{Q K^T}{\\sqrt{d_k}} \\right) V$$
where $Q, K, V$ are Query, Key, and Value matrices projected from inputs:
$$Q = X W_Q, \\quad K = X W_K, \\quad V = X W_V$$

#### Scaled Dot Product
Dividing by $\\sqrt{d_k}$ balances variance. At large values of dimension $d_k$, dot products grow large, pushing Softmax into regions with vanishingly small gradients. Scaling prevents this.`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'attention',
    quiz: [
      { id: 'tf_q1', question: 'Why is the dot product scaled by 1/sqrt(d_k) in self-attention?', options: ['To prevent dot products from growing too large, which flattens the Softmax gradient.', 'To speed up matrix multiplication.', 'To reduce the number of features.', 'To normalize embeddings between 0 and 1.'], correctAnswer: 'To prevent dot products from growing too large, which flattens the Softmax gradient.', explanation: 'Large dimensions yield high dot products, which pushes Softmax into flat regions where derivatives approach zero. Scaling prevents this gradient flattening.' }
    ],
    coding: {
      tutorial: {
      title: 'Project QKV Matrices',
      description: 'Compute Query projection matrix Q = X * W_Q.',
      pseudoCode: "",
        starterCode: `import numpy as np
def project_queries(X, W_q):
    # TODO: return dot product
    return np.array([])

X = np.array([[1.0, 0.5]])
W_q = np.array([[0.2, 0.1], [0.5, 0.8]])
print(project_queries(X, W_q))`,
        expectedOutput: '[[0.45 0.5 ]]',
        solution: `import numpy as np
def project_queries(X, W_q):
    return np.dot(X, W_q)

X = np.array([[1.0, 0.5]])
W_q = np.array([[0.2, 0.1], [0.5, 0.8]])
print(project_queries(X, W_q))`,
        hints: ['Use np.dot to project inputs.'],
        testKeywords: ['np.dot']
      },
      project: {
      title: 'Compute Scaled Attention Scores',
      description: 'Calculate scaled dot-product scores: Q * K^T / sqrt(d_k).',
      pseudoCode: "",
        starterCode: `import numpy as np
def attention_scores(Q, K, d_k):
    # TODO: return dot(Q, K.T) / sqrt(d_k)
    return np.array([])

Q = np.array([[1.0, 0.0]])
K = np.array([[1.0, 1.0]])
print(np.round(attention_scores(Q, K, 4), 2))`,
        expectedOutput: '[[0.5]]',
        solution: `import numpy as np
def attention_scores(Q, K, d_k):
    return np.dot(Q, K.T) / np.sqrt(d_k)

Q = np.array([[1.0, 0.0]])
K = np.array([[1.0, 1.0]])
print(np.round(attention_scores(Q, K, 4), 2))`,
        hints: ['Multiply Q and transpose of K, then divide by the square root of d_k.'],
        testKeywords: ['np.dot', 'np.sqrt']
      },
      assignment: {
      title: 'Softmax Row Normalizer',
      description: 'Apply the Softmax function along the rows of an attention score grid.',
      pseudoCode: "",
        starterCode: `import numpy as np
def softmax(scores):
    # TODO: compute exponential normalizer along axis=-1
    return scores

s = np.array([[1.0, 2.0]])
print(np.round(softmax(s), 3))`,
        expectedOutput: '[[0.269 0.731]]',
        solution: `import numpy as np
def softmax(scores):
    exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    return exp_s / np.sum(exp_s, axis=-1, keepdims=True)

s = np.array([[1.0, 2.0]])
print(np.round(softmax(s), 3))`,
        hints: ['Subtract maximum for numerical stability before applying np.exp.'],
        testKeywords: ['np.exp', 'np.sum']
      }
    },
    interviewQuestions: [{
      question: 'Explain the difference between Self-Attention and Multi-Head Attention.',
      answer: 'Self-attention computes attention scores once across the entire hidden dimension. Multi-head attention splits queries, keys, and values into multiple subsets of lower dimensionality, calculates attention for each head in parallel, and concatenates the outputs. This allows the model to attend to information from different representation subspaces simultaneously (e.g. tracking both syntax and semantics in different heads).',
      companyTags: ['Meta', 'Google'],
      difficulty: 'Advanced'
    }]
  },
  {
    id: 'word-embeddings',
    title: 'Word Embeddings & Word2Vec',
    category: 'Deep Learning',
    description: 'Learn dense semantic vector coordinates for words using local contexts.',
    formula: 'cos(\\theta) = \\frac{u \\cdot v}{\\|u\\| \\|v\\|}',
    theory: `### Word Embeddings & Word2Vec Mechanics

Word embeddings map words to dense vector spaces where geometric distance represents semantic similarity.

#### Word2Vec Architectures
- **Continuous Bag of Words (CBOW)**: Predicts the target word using surrounding context.
- **Skip-Gram**: Predicts context words using the target word.

#### Cosine Similarity Metric
Measures the directional alignment (cosine of the angle) between word vectors:
$$CosineSimilarity(u, v) = \\frac{u \\cdot v}{\\|u\\| \\|v\\|} = \\frac{\\sum u_i v_i}{\\sqrt{\\sum u_i^2} \\sqrt{\\sum v_i^2}}$$`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'word-embeddings',
    quiz: [
      { id: 'w2v_q1', question: 'Which Word2Vec architecture predicts context words given a target input word?', options: ['Skip-Gram', 'CBOW', 'Transformer', 'PCA'], correctAnswer: 'Skip-Gram', explanation: 'Skip-gram inputs a single target word and outputs probabilities for surrounding context words.' }
    ],
    coding: {
      tutorial: {
      title: 'Generate Skip-Gram Pairs',
      description: 'Generate context pairs given a list of tokens and a context window size.',
      pseudoCode: "",
        starterCode: `def get_pairs(tokens, window):
    pairs = []
    # TODO: Return list of (target, context) token pairs
    return pairs

print(get_pairs(['the', 'cat', 'sat'], 1))`,
        expectedOutput: "[('cat', 'the'), ('cat', 'sat')]",
        solution: `def get_pairs(tokens, window):
    pairs = []
    for i, target in enumerate(tokens):
        for w in range(-window, window + 1):
            context_idx = i + w
            if w != 0 and 0 <= context_idx < len(tokens):
                pairs.append((target, tokens[context_idx]))
    return pairs

print(get_pairs(['the', 'cat', 'sat'], 1))`,
        hints: ['Iterate through tokens and collect indices within window offsets.'],
        testKeywords: ['enumerate', 'window']
      },
      project: {
      title: 'Cosine Similarity Calculator',
      description: 'Calculate cosine similarity between word vectors u and v.',
      pseudoCode: "",
        starterCode: `import numpy as np
def cosine_sim(u, v):
    # TODO: return dot(u,v) / (norm(u)*norm(v))
    return 0.0

u = np.array([1.0, 0.0])
v = np.array([1.0, 1.0])
print(np.round(cosine_sim(u, v), 3))`,
        expectedOutput: '0.707',
        solution: `import numpy as np
def cosine_sim(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return np.dot(u, v) / (norm_u * norm_v)

u = np.array([1.0, 0.0])
v = np.array([1.0, 1.0])
print(np.round(cosine_sim(u, v), 3))`,
        hints: ['Use np.linalg.norm to calculate vector magnitudes.'],
        testKeywords: ['np.dot', 'np.linalg.norm']
      },
      assignment: {
      title: 'Negative Sampling Loss',
      description: 'Calculate negative sampling loss for target, positive, and negative context vectors.',
      pseudoCode: "",
        starterCode: `import numpy as np
def neg_loss(target, pos, neg):
    # Loss = -log(sigmoid(t.pos)) - sum(log(sigmoid(-t.neg)))
    # TODO: compute loss
    return 0.0

t = np.array([1.0, 0.0])
p = np.array([1.0, 0.0])
n = np.array([-1.0, 0.0])
print(np.round(neg_loss(t, p, n), 3))`,
        expectedOutput: '0.627',
        solution: `import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def neg_loss(target, pos, neg):
    p_score = sigmoid(np.dot(target, pos))
    n_score = sigmoid(-np.dot(target, neg))
    return -np.log(p_score) - np.log(n_score)

t = np.array([1.0, 0.0])
p = np.array([1.0, 0.0])
n = np.array([-1.0, 0.0])
print(np.round(neg_loss(t, p, n), 3))`,
        hints: ['Implement the sigmoid function helper.', 'Compute dot products, apply sigmoid, and sum log values.'],
        testKeywords: ['sigmoid', 'np.log']
      }
    },
    interviewQuestions: [{
      question: 'Why is standard softmax calculation inefficient in Word2Vec, and how do Hierarchical Softmax or Negative Sampling resolve this?',
      answer: 'Standard Softmax requires calculating denominator exponents across the entire vocabulary size |V| (often 100k+ words), which is computationally expensive. Hierarchical Softmax replaces this with a binary Huffman tree, reducing calculations to O(log |V|). Negative sampling simplifies this to a binary logistic regression task, updating only the target word and a few randomly selected negative words (O(k)).',
      companyTags: ['Google'],
      difficulty: 'Advanced'
    }]
  },
  {
    id: 'q-learning-rl',
    title: 'Q-Learning',
    category: 'Advanced & MLOps',
    description: 'Train agents to navigate environments using state-action value iterations.',
    formula: 'Q(s,a) \\leftarrow Q(s,a) + \\alpha [r + \\gamma \\max Q(s\', a\') - Q(s,a)]',
    theory: `### Q-Learning & Temporal Difference Updates

Q-learning is a model-free reinforcement learning algorithm that computes optimal state-action values.

#### Bellman Equation
The optimal action-value function satisfies the Bellman Optimality Equation:
$$Q^*(s, a) = R(s, a) + \\gamma \\max_{a'} Q^*(s', a')$$

#### Q-Value Update Rule
We update the Q-table iteratively using the Temporal Difference error:
$$Q(s, a) \\leftarrow Q(s, a) + \\alpha \\left[ r + \\gamma \\max_{a'} Q(s', a') - Q(s, a) \\right]$$
where $\\alpha$ is the learning rate, $\\gamma$ is the discount factor, and $r + \\gamma \\max Q(s', a')$ is the TD Target.`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'q-learning',
    quiz: [
      { id: 'ql_q1', question: 'What is the purpose of the discount factor gamma in the Q-learning update equation?', options: ['To balance the value of immediate rewards against future rewards.', 'To set the step size of parameter updates.', 'To scale down feature matrices.', 'To regularize weights.'], correctAnswer: 'To balance the value of immediate rewards against future rewards.', explanation: 'Gamma (value between 0 and 1) discounts future rewards, prioritizing immediate rewards (low gamma) or long-term goals (high gamma).' }
    ],
    coding: {
      tutorial: {
      title: 'Temporal Difference Target',
      description: 'Calculate the Temporal Difference target: reward + gamma * maxQ(s\', a\').',
      pseudoCode: "",
        starterCode: `import numpy as np
def td_target(reward, gamma, next_state_q):
    # TODO: return reward + gamma * max(next_state_q)
    return 0.0

next_q = np.array([1.5, 3.0, 0.5])
print(td_target(1.0, 0.9, next_q))`,
        expectedOutput: '3.7',
        solution: `import numpy as np
def td_target(reward, gamma, next_state_q):
    return reward + gamma * np.max(next_state_q)

next_q = np.array([1.5, 3.0, 0.5])
print(td_target(1.0, 0.9, next_q))`,
        hints: ['Find max value of next_state_q with np.max and scale by gamma.'],
        testKeywords: ['np.max']
      },
      project: {
      title: 'Epsilon-Greedy Action Selection',
      description: 'Select an action index using epsilon-greedy exploration.',
      pseudoCode: "",
        starterCode: `import numpy as np
def select_action(q_values, epsilon, seed=42):
    np.random.seed(seed)
    # TODO: If rand < epsilon, choose random index (explore)
    # Else choose argmax of q_values (exploit)
    return 0

q_vals = np.array([1.2, 3.5, 0.8])
print(select_action(q_vals, 0.1))`,
        expectedOutput: '1',
        solution: `import numpy as np
def select_action(q_values, epsilon, seed=42):
    np.random.seed(seed)
    if np.random.rand() < epsilon:
        return np.random.choice(len(q_values))
    else:
        return np.argmax(q_values)

q_vals = np.array([1.2, 3.5, 0.8])
print(select_action(q_vals, 0.1))`,
        hints: ['Compare a random float against epsilon.', 'Use np.argmax or np.random.choice.'],
        testKeywords: ['np.random.rand', 'np.argmax']
      },
      assignment: {
      title: 'Q-Value Table Update',
      description: 'Update a cell in the Q-table using the temporal difference formula.',
      pseudoCode: "",
        starterCode: `def update_q_value(current_q, td_target, alpha):
    # TODO: return Q_updated = Q + alpha * (td_target - Q)
    return current_q

print(update_q_value(2.0, 3.5, 0.1))`,
        expectedOutput: '2.15',
        solution: `def update_q_value(current_q, td_target, alpha):
    return current_q + alpha * (td_target - current_q)

print(update_q_value(2.0, 3.5, 0.1))`,
        hints: ['Calculate difference, scale by alpha, and add to current_q.'],
        testKeywords: ['current_q +']
      }
    },
    interviewQuestions: [{
      question: 'What is the exploration-exploitation dilemma in Reinforcement Learning, and how does epsilon-greedy decay address it?',
      answer: 'Exploration means seeking new environment paths to discover higher rewards. Exploitation means taking known paths with high rewards. Epsilon-greedy balances this by exploring with probability epsilon. Decaying epsilon over training steps allows the agent to explore heavily early on and shift to exploitation as it converges.',
      companyTags: ['Tesla', 'Google'],
      difficulty: 'Advanced'
    }]
  },
  {
    id: 'generative-adversarial-networks',
    title: 'Generative Adversarial Networks',
    category: 'Advanced & MLOps',
    description: 'Generate synthetic distributions by contesting Generator and Discriminator networks.',
    formula: '\\min_G \\max_D V(D,G)',
    theory: `### Generative Adversarial Networks (GANs)

GANs train two competing neural networks in a minimax game framework.

#### The Minimax Objective Function
$$\\min_{G} \\max_{D} V(D, G) = \\mathbb{E}_{x \\sim p_{data}} [\\log D(x)] + \\mathbb{E}_{z \\sim p_z} [\\log (1 - D(G(z)))]$$
where:
- $D(x)$ is the probability that real sample $x$ is real.
- $G(z)$ generates synthetic samples from noise vector $z$.

#### Dynamic Optimization
- **Discriminator**: Maximizes the probability of assigning correct labels to both real and generated samples.
- **Generator**: Minimizes $\\log(1 - D(G(z)))$ (or maximizes $\\log D(G(z))$ to prevent vanishing gradients early on).`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'gan',
    quiz: [
      { id: 'gan_q1', question: 'What is the optimization objective of the Generator network in a standard GAN framework?', options: ['Maximize the probability of the Discriminator predicting synthetic samples are real.', 'Minimize Discriminator parameters.', 'Maximize prior distribution variance.', 'Fit target labels directly.'], correctAnswer: 'Maximize the probability of the Discriminator predicting synthetic samples are real.', explanation: 'The generator tries to create samples that fool the discriminator, maximizing D(G(z)).' }
    ],
    coding: {
      tutorial: {
      title: 'Generator Loss Function',
      description: 'Calculate binary cross-entropy loss for fake samples: -log(D(G(z))).',
      pseudoCode: "",
        starterCode: `import numpy as np
def generator_loss(d_fake_pred):
    # TODO: return -sum(log(p))
    return 0.0

preds = np.array([0.8, 0.9])
print(np.round(generator_loss(preds), 3))`,
        expectedOutput: '0.164',
        solution: `import numpy as np
def generator_loss(d_fake_pred):
    return -np.mean(np.log(d_fake_pred))

preds = np.array([0.8, 0.9])
print(np.round(generator_loss(preds), 3))`,
        hints: ['Apply natural log np.log, compute mean, and negate.'],
        testKeywords: ['np.log', 'np.mean']
      },
      project: {
      title: 'Generator Noise Vector Generator',
      description: 'Generate a matrix of random Gaussian noise vectors of shape (batch_size, latent_dim).',
      pseudoCode: "",
        starterCode: `import numpy as np
def generate_noise(batch_size, latent_dim, seed=42):
    np.random.seed(seed)
    # TODO: return random normal matrix
    return np.array([])

print(generate_noise(2, 3).shape)`,
        expectedOutput: '(2, 3)',
        solution: `import numpy as np
def generate_noise(batch_size, latent_dim, seed=42):
    np.random.seed(seed)
    return np.random.normal(0, 1, size=(batch_size, latent_dim))

print(generate_noise(2, 3).shape)`,
        hints: ['Use np.random.normal with size tuple.'],
        testKeywords: ['np.random.normal']
      },
      assignment: {
      title: 'Discriminator Loss Function',
      description: 'Calculate Discriminator loss for real and fake predictions.',
      pseudoCode: "",
        starterCode: `import numpy as np
def discriminator_loss(d_real, d_fake):
    # Loss = -mean(log(d_real)) - mean(log(1 - d_fake))
    # TODO: compute loss
    return 0.0

real = np.array([0.9, 0.95])
fake = np.array([0.1, 0.15])
print(np.round(discriminator_loss(real, fake), 3))`,
        expectedOutput: '0.118',
        solution: `import numpy as np
def discriminator_loss(d_real, d_fake):
    real_loss = -np.mean(np.log(d_real))
    fake_loss = -np.mean(np.log(1.0 - d_fake))
    return real_loss + fake_loss

real = np.array([0.9, 0.95])
fake = np.array([0.1, 0.15])
print(np.round(discriminator_loss(real, fake), 3))`,
        hints: ['Compute average log loss for real predictions, and average log(1 - fake) for fake predictions.'],
        testKeywords: ['np.log', 'np.mean']
      }
    },
    interviewQuestions: [{
      question: 'What is mode collapse in GAN training, and how do Wasserstein GANs (WGAN) mitigate this issue?',
      answer: 'Mode collapse occurs when the generator learns to produce samples from only a single distribution mode (e.g. generating only one digit class in MNIST), ignoring other target classes because they successfully fool the discriminator. Wasserstein GANs resolve this by replacing the binary classifier discriminator with a critic that outputs a continuous score, using the Earth Mover\'s distance to provide smoother gradients that guide the generator across all distribution modes.',
      companyTags: ['OpenAI'],
      difficulty: 'Advanced'
    }]
  },
  {
    id: 'variational-autoencoders',
    title: 'Variational Autoencoders',
    category: 'Advanced & MLOps',
    description: 'Map datasets to continuous latent spaces using reconstruction and KL divergence penalties.',
    formula: 'L = E[log P(x|z)] - D_{KL}(Q(z|x) || P(z))',
    theory: `### Variational Autoencoders (VAEs) Theory

VAEs learn continuous, structured latent spaces that allow generating new samples.

#### Loss Function: Evidence Lower Bound (ELBO)
The objective function balances reconstruction loss and latent space organization:
$$\\mathcal{L}_{VAE}(\\theta, \\phi) = \\mathbb{E}_{q_\\phi(z|x)} [\\log p_\\theta(x|z)] - D_{KL}(q_\\phi(z|x) \\parallel p(z))$$

#### Kullback-Leibler (KL) Divergence Penalty
Forces the encoder\\'s predicted latent distributions $q(z|x)$ to align with a standard normal distribution prior $P(z) = \\mathcal{N}(0, I)$, preventing coordinate grouping separation:
$$D_{KL} = -\\frac{1}{2} \\sum_{j=1}^{J} \\left( 1 + \\log(\\sigma_j^2) - \\mu_j^2 - \\sigma_j^2 \\right)$$

#### The Reparameterization Trick
Allows backpropagation through random sampling steps by expressing the latent vector $z$ as:
$$z = \\mu + \\sigma \\odot \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0, I)$$`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'vae',
    quiz: [
      { id: 'vae_q1', question: 'What is the purpose of the reparameterization trick in VAEs?', options: ['To allow backpropagation gradients to flow through stochastic sampling nodes.', 'To compress latent dimensions.', 'To calculate reconstruction errors.', 'To normalize latent coordinates.'], correctAnswer: 'To allow backpropagation gradients to flow through stochastic sampling nodes.', explanation: 'Since random sampling lacks derivatives, the reparameterization trick projects randomness into a constant external input, making parameters differentiable.' }
    ],
    coding: {
      tutorial: {
      title: 'KL Divergence Penalty',
      description: 'Calculate the KL divergence penalty given mean and variance vectors.',
      pseudoCode: "",
        starterCode: `import numpy as np
def kl_divergence(mu, var):
    # TODO: return -0.5 * sum(1 + log(var) - mu^2 - var)
    return 0.0

mu = np.array([0.0])
var = np.array([1.0])
print(kl_divergence(mu, var))`,
        expectedOutput: '-0.0',
        solution: `import numpy as np
def kl_divergence(mu, var):
    return -0.5 * np.sum(1.0 + np.log(var) - mu**2 - var)

mu = np.array([0.0])
var = np.array([1.0])
print(kl_divergence(mu, var))`,
        hints: ['Use np.log and np.sum to calculate the penalty.'],
        testKeywords: ['np.log', 'np.sum']
      },
      project: {
      title: 'Reparameterization Trick',
      description: 'Compute latent vector z = mu + std * epsilon.',
      pseudoCode: "",
        starterCode: `import numpy as np
def reparameterize(mu, log_var, seed=42):
    np.random.seed(seed)
    # std = exp(0.5 * log_var)
    # epsilon = random normal matching std shape
    # TODO: return mu + std * epsilon
    return mu

m = np.array([0.5, -0.2])
lv = np.array([0.0, 0.0])
print(np.round(reparameterize(m, lv), 3))`,
        expectedOutput: '[ 0.997 -0.338]',
        solution: `import numpy as np
def reparameterize(mu, log_var, seed=42):
    np.random.seed(seed)
    std = np.exp(0.5 * log_var)
    eps = np.random.normal(0, 1, size=mu.shape)
    return mu + std * eps

m = np.array([0.5, -0.2])
lv = np.array([0.0, 0.0])
print(np.round(reparameterize(m, lv), 3))`,
        hints: ['Calculate std using exponential of 0.5 * log_var, sample epsilon, and combine.'],
        testKeywords: ['np.exp', 'np.random.normal']
      },
      assignment: {
      title: 'Total VAE Loss',
      description: 'Calculate the total VAE loss as the sum of Reconstruction Loss and KL Divergence.',
      pseudoCode: "",
        starterCode: `def total_vae_loss(recon_loss, kl_div):
    # TODO: return recon_loss + kl_div
    return 0.0

print(total_vae_loss(12.5, 2.3))`,
        expectedOutput: '14.8',
        solution: `def total_vae_loss(recon_loss, kl_div):
    return recon_loss + kl_div

print(total_vae_loss(12.5, 2.3))`,
        hints: ['Sum the two loss terms.'],
        testKeywords: ['recon_loss +']
      }
    },
    interviewQuestions: [{
      question: 'Explain why standard Autoencoders are not suitable as generative models, and how VAEs resolve this.',
      answer: 'Standard autoencoders map inputs to discrete latent coordinates. This leads to gaps in the latent space where generated outputs are nonsensical. VAEs resolve this by mapping inputs to probability distributions (mean and variance vectors). The KL divergence penalty forces these distributions to overlap around the origin, creating a continuous, structured latent space that allows smooth interpolation.',
      companyTags: ['Google'],
      difficulty: 'Advanced'
    }]
  },
  {
    id: 'tsne-visualization',
    title: 't-SNE Dimensionality Reduction',
    category: 'Unsupervised Learning',
    description: 'Visualize high-dimensional clusters by matching local probability distributions.',
    formula: 'p_{j|i} = \\frac{\\exp(-\\|x_i-x_j\\|^2 / 2\\sigma_i^2)}{\\sum \\exp(-\\|x_i-x_k\\|^2 / 2\\sigma_i^2)}',
    theory: `### t-SNE Clustering Visualization

t-SNE projects high-dimensional vectors to 2D/3D spaces for visualization by matching local probability distributions.

#### Probability Mappings
- **High-Dimensional Space**: The probability of point $x_j$ being a neighbor of $x_i$ is modeled as a Gaussian distribution:
$$p_{j|i} = \\frac{\\exp(-\\|x_i - x_j\\|^2 / 2\\sigma_i^2)}{\\sum_{k \\neq i} \\exp(-\\|x_i - x_k\\|^2 / 2\\sigma_i^2)}$$
- **Low-Dimensional Space**: Neighborhood probabilities are modeled using a Student-t distribution (with 1 degree of freedom) to address crowding issues:
$$q_{ij} = \\frac{(1 + \\|y_i - y_j\\|^2)^{-1}}{\\sum_{k} \\sum_{l \\neq k} (1 + \\|y_k - y_l\\|^2)^{-1}}$$

#### Objective Function Optimization
We minimize the difference between high- and low-dimensional probability distributions using Kullback-Leibler divergence:
$$KL(P \\parallel Q) = \\sum_{i} \\sum_{j} p_{ij} \\log \\frac{p_{ij}}{q_{ij}}$$`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'tsne',
    quiz: [
      { id: 'tsne_q1', question: 'Why does t-SNE use a Student-t distribution in the low-dimensional projection space?', options: ['To resolve the crowding problem by allowing points to spread out further.', 'To speed up training gradients.', 'To force features to be orthogonal.', 'To normalize coordinates.'], correctAnswer: 'To resolve the crowding problem by allowing points to spread out further.', explanation: 'The Student-t distribution has heavier tails than a Gaussian distribution. This addresses the crowding problem by allowing distant points to remain far apart in low-dimensional space.' }
    ],
    coding: {
      tutorial: {
      title: 'Calculate Squared Distances',
      description: 'Calculate the squared Euclidean distance between two vectors.',
      pseudoCode: "",
        starterCode: `import numpy as np
def squared_distance(u, v):
    # TODO: return sum((u - v)^2)
    return 0.0

u = np.array([1.0, 2.0])
v = np.array([4.0, 6.0])
print(squared_distance(u, v))`,
        expectedOutput: '25.0',
        solution: `import numpy as np
def squared_distance(u, v):
    return float(np.sum((u - v) ** 2))

u = np.array([1.0, 2.0])
v = np.array([4.0, 6.0])
print(squared_distance(u, v))`,
        hints: ['Subtract vectors, square differences, and sum.'],
        testKeywords: ['np.sum']
      },
      project: {
      title: 'Low-Dimensional Probabilities',
      description: 'Compute Student-t distribution similarity score for a distance value.',
      pseudoCode: "",
        starterCode: `def student_t_similarity(d_squared):
    # TODO: return 1 / (1 + d_squared)
    return 0.0

print(student_t_similarity(4.0))`,
        expectedOutput: '0.2',
        solution: `def student_t_similarity(d_squared):
    return 1.0 / (1.0 + d_squared)

print(student_t_similarity(4.0))`,
        hints: ['Divide 1.0 by (1.0 + d_squared).'],
        testKeywords: ['1.0 /']
      },
      assignment: {
      title: 'KL Divergence Objective Loss',
      description: 'Calculate KL divergence between two probability matrices.',
      pseudoCode: "",
        starterCode: `import numpy as np
def tsne_kl_loss(P, Q):
    # Clip to prevent log(0)
    P = np.clip(P, 1e-15, 1.0)
    Q = np.clip(Q, 1e-15, 1.0)
    # TODO: return sum(P * log(P / Q))
    return 0.0

P = np.array([[0.5, 0.5]])
Q = np.array([[0.4, 0.6]])
print(np.round(tsne_kl_loss(P, Q), 3))`,
        expectedOutput: '0.02',
        solution: `import numpy as np
def tsne_kl_loss(P, Q):
    P = np.clip(P, 1e-15, 1.0)
    Q = np.clip(Q, 1e-15, 1.0)
    return float(np.sum(P * np.log(P / Q)))

P = np.array([[0.5, 0.5]])
Q = np.array([[0.4, 0.6]])
print(np.round(tsne_kl_loss(P, Q), 3))`,
        hints: ['Perform element-wise multiplication of P and np.log(P / Q), then sum.'],
        testKeywords: ['np.sum', 'np.log', 'clip']
      }
    },
    interviewQuestions: [{
      question: 'What is the perplexity parameter in t-SNE, and how does it balance global and local dataset structure?',
      answer: 'Perplexity is a measure of the effective number of neighbors to consider when calculating high-dimensional probabilities, serving as a variance normalizer. Low perplexity values prioritize local clustering structure, whereas high perplexity values capture global dataset relationships.',
      companyTags: ['Netflix'],
      difficulty: 'Advanced'
    }]
  },
  {
    id: 'ridge-lasso-regularization',
    title: 'Ridge & Lasso Regularization',
    category: 'Foundations & Math',
    description: 'Dampen overfit parameter weights by adding L1 and L2 penalty terms.',
    formula: 'J(w) = MSE(w) + \\lambda_1 \\|w\\|_1 + \\lambda_2 \\|w\\|_2^2',
    theory: `### Regularization Theory: Ridge vs Lasso

Regularization prevents overfitting by adding a penalty on weight magnitudes to the loss function.

#### Ridge Regression ($L_2$ Regularization)
Adds a squared magnitude penalty to the loss function, shrinking weights toward zero:
$$J(w) = MSE(w) + \\lambda \\|w\\|_2^2 = MSE(w) + \\lambda \\sum_{j=1}^{d} w_j^2$$
Solving the gradient yields:
$$w = (X^T X + \\lambda I)^{-1} X^T y$$

#### Lasso Regression ($L_1$ Regularization)
Adds an absolute magnitude penalty, driving weights to exactly zero and producing sparse models:
$$J(w) = MSE(w) + \\lambda \\|w\\|_1 = MSE(w) + \\lambda \\sum_{j=1}^{d} |w_j|$$`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'regularization',
    quiz: [
      { id: 'reg_q1', question: 'Which regularization technique drives weights to exactly zero, serving as a feature selection tool?', options: ['Lasso (L1)', 'Ridge (L2)', 'Elastic Net', 'OLS'], correctAnswer: 'Lasso (L1)', explanation: 'Lasso adds an L1 absolute penalty. Its sharp geometric constraint drives weights to exactly zero, performing feature selection.' }
    ],
    coding: {
      tutorial: {
      title: 'Calculate L1 Norm penalty',
      description: 'Calculate Lasso weight penalty: sum(|w|).',
      pseudoCode: "",
        starterCode: `import numpy as np
def l1_penalty(w):
    # TODO: return sum(|w|)
    return 0.0

w = np.array([0.5, -2.0, 0.1])
print(l1_penalty(w))`,
        expectedOutput: '2.6',
        solution: `import numpy as np
def l1_penalty(w):
    return float(np.sum(np.abs(w)))

w = np.array([0.5, -2.0, 0.1])
print(l1_penalty(w))`,
        hints: ['Use np.abs and np.sum to calculate absolute sum.'],
        testKeywords: ['np.abs', 'np.sum']
      },
      project: {
      title: 'Calculate L2 Norm penalty',
      description: 'Calculate Ridge weight penalty: sum(w^2).',
      pseudoCode: "",
        starterCode: `import numpy as np
def l2_penalty(w):
    # TODO: return sum(w^2)
    return 0.0

w = np.array([0.5, -2.0, 0.1])
print(l2_penalty(w))`,
        expectedOutput: '4.26',
        solution: `import numpy as np
def l2_penalty(w):
    return float(np.sum(w ** 2))

w = np.array([0.5, -2.0, 0.1])
print(l2_penalty(w))`,
        hints: ['Square the weights and sum them.'],
        testKeywords: ['np.sum']
      },
      assignment: {
      title: 'Lasso Subgradient helper',
      description: 'Compute the sign gradient for Lasso regularization.',
      pseudoCode: "",
        starterCode: `import numpy as np
def lasso_subgradient(w):
    # TODO: return sign vector (-1 for negative, +1 for positive, 0 for zero)
    return np.array([])

w = np.array([-0.5, 0.0, 2.5])
print(lasso_subgradient(w))`,
        expectedOutput: '[-1.  0.  1.]',
        solution: `import numpy as np
def lasso_subgradient(w):
    return np.sign(w)

w = np.array([-0.5, 0.0, 2.5])
print(lasso_subgradient(w))`,
        hints: ['Use the np.sign function.'],
        testKeywords: ['np.sign']
      }
    },
    interviewQuestions: [{
      question: 'Explain why Lasso (L1) produces sparse weight matrices, whereas Ridge (L2) only shrinks weights without setting them to zero.',
      answer: 'The L1 constraint region is diamond-shaped with sharp corners along the axes, whereas the L2 region is spherical. When optimization contours touch these boundaries, the contact point is highly likely to occur at a corner on an axis in L1 space, driving that feature\'s weight to exactly zero.',
      companyTags: ['Google'],
      difficulty: 'Advanced'
    }]
  },
  {
    id: 'dbscan-clustering',
    title: 'DBSCAN Clustering',
    category: 'Unsupervised Learning',
    description: 'Group spatial coordinates by density connectivity, identifying noise points automatically.',
    formula: 'N_\\epsilon(p) = \\{q \\in D \\mid dist(p,q) \\le \\epsilon\\}',
    theory: `### DBSCAN density clustering Theory

DBSCAN groups points based on density connectivity, bypassing K-Means centroid assumptions.

#### Density Parameters
- **Epsilon ($\\epsilon$)**: Maximum radius to search for neighbors:
$$N_\\epsilon(p) = \\{q \\in D \\mid dist(p, q) \\le \\epsilon\\}$$
- **MinSamples**: Minimum neighbors required to form a dense region.

#### Point Classification
1. **Core Points**: Points with $\\ge$ MinSamples neighbors in their $\\epsilon$-neighborhood.
2. **Border Points**: Points within $\\epsilon$-distance of a core point but lacking enough neighbors.
3. **Noise Points**: All other points.`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'dbscan',
    quiz: [
      { id: 'dbs_q1', question: 'Which point class represents an outlier that does not belong to any cluster in DBSCAN?', options: ['Noise Point', 'Core Point', 'Border Point', 'Centroid'], correctAnswer: 'Noise Point', explanation: 'Outliers that fail to meet density requirements are labeled as Noise Points.' }
    ],
    coding: {
      tutorial: {
      title: 'Locate Neighbors',
      description: 'Find indices of points within epsilon distance of a target point.',
      pseudoCode: "",
        starterCode: `import numpy as np
def find_neighbors(X, target_idx, eps):
    # TODO: return indices of points where distance <= eps
    return []

X = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0]])
print(find_neighbors(X, 0, 0.5))`,
        expectedOutput: '[0, 1]',
        solution: `import numpy as np
def find_neighbors(X, target_idx, eps):
    target = X[target_idx]
    dists = np.sqrt(np.sum((X - target) ** 2, axis=1))
    return list(np.where(dists <= eps)[0])

X = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0]])
print(find_neighbors(X, 0, 0.5))`,
        hints: ['Calculate distance array to target, filter indices using np.where.'],
        testKeywords: ['np.sum', 'np.where']
      },
      project: {
      title: 'Core Point Classifier',
      description: 'Label a point as a core point if it has enough neighbors.',
      pseudoCode: "",
        starterCode: `def is_core_point(neighbors_count, min_samples):
    # TODO: return True if count >= min_samples
    return False

print(is_core_point(5, 4))`,
        expectedOutput: 'True',
        solution: `def is_core_point(neighbors_count, min_samples):
    return neighbors_count >= min_samples

print(is_core_point(5, 4))`,
        hints: ['Check if neighbor count is greater than or equal to min_samples.'],
        testKeywords: ['>=']
      },
      assignment: {
      title: 'Label Dense Clusters',
      description: 'Differentiate core points and noise points based on neighbor counts.',
      pseudoCode: "",
        starterCode: `import numpy as np
def label_points(X, eps, min_samples):
    # TODO: For each point, find neighbors. If count >= min_samples, label core (1), else noise (0)
    return np.array([])

X = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0]])
print(label_points(X, 0.5, 2))`,
        expectedOutput: '[1 1 0]',
        solution: `import numpy as np
def label_points(X, eps, min_samples):
    labels = []
    for i in range(len(X)):
        dists = np.sqrt(np.sum((X - X[i]) ** 2, axis=1))
        neighbors = np.where(dists <= eps)[0]
        if len(neighbors) >= min_samples:
            labels.append(1)
        else:
            labels.append(0)
    return np.array(labels)

X = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0]])
print(label_points(X, 0.5, 2))`,
        hints: ['Iterate through coordinates, calculate neighbor count, and assign labels.'],
        testKeywords: ['np.sum', 'np.where']
      }
    },
    interviewQuestions: [{
      question: 'Compare DBSCAN and K-Means. In what scenarios does K-Means fail where DBSCAN excels?',
      answer: 'K-Means assumes spherical clusters and requires specifying K beforehand. It fails on complex shapes (like moons) and is sensitive to outliers. DBSCAN identifies clusters of arbitrary shapes, automatically detects noise, and does not require pre-specifying K.',
      companyTags: ['Uber'],
      difficulty: 'Advanced'
    }]
  },
  {
    id: 'neural-activations',
    title: 'Neural Activation Functions',
    category: 'Deep Learning',
    description: 'Explore ReLU, Sigmoid, Tanh, and GeLU mathematical responses.',
    formula: 'ReLU(z) = \\max(0,z)',
    theory: `### Neural Activation Functions Theory

Activation functions introduce non-linearities into neural networks, enabling them to learn complex patterns.

#### Common Activations
- **Sigmoid**: Maps inputs to (0, 1):
$$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$
- **Hyperbolic Tangent (Tanh)**: Maps inputs to (-1, 1):
$$\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}$$
- **Rectified Linear Unit (ReLU)**: Maps negative inputs to zero:
$$ReLU(z) = \\max(0, z)$$
- **Gaussian Error Linear Unit (GeLU)**: Scales inputs by the cumulative Gaussian distribution:
$$GeLU(z) = z \\cdot \\Phi(z)$$`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'activations',
    quiz: [
      { id: 'act_q1', question: 'Which activation function maps real inputs to the range (-1.0, 1.0)?', options: ['Tanh', 'Sigmoid', 'ReLU', 'GeLU'], correctAnswer: 'Tanh', explanation: 'Tanh maps inputs to (-1, 1), centering outputs at zero.' }
    ],
    coding: {
      tutorial: {
      title: 'Exponential Linear Unit (ELU)',
      description: 'Implement ELU: z if z > 0 else alpha * (exp(z) - 1).',
      pseudoCode: "",
        starterCode: `import numpy as np
def elu(z, alpha=1.0):
    # TODO: implement ELU
    return z

print(np.round(elu(-1.0), 3))`,
        expectedOutput: '-0.632',
        solution: `import numpy as np
def elu(z, alpha=1.0):
    return float(z if z > 0 else alpha * (np.exp(z) - 1.0))

print(np.round(elu(-1.0), 3))`,
        hints: ['Check the sign of z, using np.exp for negative values.'],
        testKeywords: ['np.exp']
      },
      project: {
      title: 'Softmax Vector Calculator',
      description: 'Calculate Softmax probabilities for a vector.',
      pseudoCode: "",
        starterCode: `import numpy as np
def softmax_vector(z):
    # TODO: return exp(z) / sum(exp(z))
    return z

z = np.array([1.0, 2.0])
print(np.round(softmax_vector(z), 3))`,
        expectedOutput: '[0.269 0.731]',
        solution: `import numpy as np
def softmax_vector(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

z = np.array([1.0, 2.0])
print(np.round(softmax_vector(z), 3))`,
        hints: ['Subtract the maximum value before exponentiating for numerical stability.'],
        testKeywords: ['np.exp', 'np.sum']
      },
      assignment: {
      title: 'Sigmoid Derivative calculator',
      description: 'Compute the derivative of the Sigmoid function: s * (1 - s).',
      pseudoCode: "",
        starterCode: `def sigmoid_derivative(s):
    # TODO: return s * (1 - s)
    return s

print(sigmoid_derivative(0.5))`,
        expectedOutput: '0.25',
        solution: `def sigmoid_derivative(s):
    return s * (1.0 - s)

print(sigmoid_derivative(0.5))`,
        hints: ['Multiply the sigmoid output by 1.0 - s.'],
        testKeywords: ['* (1.0 -']
      }
    },
    interviewQuestions: [{
      question: 'Why does ReLU suffer from the "dying ReLU" problem, and how do Leaky ReLU or GeLU address it?',
      answer: 'Dying ReLU occurs when weights update such that a neuron outputs zero for all training inputs, resulting in a gradient of zero that prevents further learning. Leaky ReLU resolves this by adding a small negative slope (e.g. 0.01 * z). GeLU uses a stochastic gating mechanism scaled by the Gaussian CDF, ensuring small non-zero gradients for negative inputs.',
      companyTags: ['Meta'],
      difficulty: 'Advanced'
    }]
  },
  {
    id: 'optimization-algorithms',
    title: 'Optimization Algorithms',
    category: 'Foundations & Math',
    description: 'Compare learning behaviors of Adam, RMSprop, and Momentum optimizers.',
    formula: '\\theta_{t+1} = \\theta_t - \\eta \\cdot \\text{update}',
    theory: `### Optimization Algorithms Theory

Optimizers use gradient calculations to update parameters and minimize loss functions.

#### Common Optimizers
- **SGD**: Classic stochastic gradient descent:
$$w_{t+1} = w_t - \\eta g_t$$
- **Momentum**: Accelerates updates by adding a fraction of past velocities:
$$v_t = \\beta v_{t-1} + \\eta g_t$$
- **RMSprop**: Scales updates by a running average of squared gradients:
$$v_t = \\beta v_{t-1} + (1 - \\beta) g_t^2$$
$$w_{t+1} = w_t - \\frac{\\eta}{\\sqrt{v_t} + \\epsilon} g_t$$`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'optimizers',
    quiz: [
      { id: 'opt_q1', question: 'Which optimizer scales learning rates using running averages of squared gradients?', options: ['RMSprop', 'Momentum', 'SGD', 'Lasso'], correctAnswer: 'RMSprop', explanation: 'RMSprop scales the gradient updates by the square root of the running average of squared gradients.' }
    ],
    coding: {
      tutorial: {
      title: 'RMSprop Squared gradient update',
      description: 'Update the running average of squared gradients: v = beta * v + (1 - beta) * g^2.',
      pseudoCode: "",
        starterCode: `def rmsprop_v(v, g, beta):
    # TODO: return updated v
    return v

print(rmsprop_v(0.5, 0.2, 0.9))`,
        expectedOutput: '0.454',
        solution: `def rmsprop_v(v, g, beta):
    return beta * v + (1.0 - beta) * (g ** 2)

print(rmsprop_v(0.5, 0.2, 0.9))`,
        hints: ['Multiply beta by v, and add 1 - beta multiplied by squared gradient.'],
        testKeywords: ['beta * v', '1.0 - beta']
      },
      project: {
      title: 'Adam First Moment Update',
      description: 'Compute the updated first moment: m = beta1 * m + (1 - beta1) * g.',
      pseudoCode: "",
        starterCode: `def adam_m(m, g, beta1):
    # TODO: return updated m
    return m

print(adam_m(0.2, 0.1, 0.9))`,
        expectedOutput: '0.19',
        solution: `def adam_m(m, g, beta1):
    return beta1 * m + (1.0 - beta1) * g

print(adam_m(0.2, 0.1, 0.9))`,
        hints: ['Combine beta1 * m and (1.0 - beta1) * g.'],
        testKeywords: ['beta1 * m']
      },
      assignment: {
      title: 'RMSprop Parameter Step',
      description: 'Compute the parameter update step for RMSprop.',
      pseudoCode: "",
        starterCode: `import numpy as np
def rmsprop_step(w, g, v, eta, eps=1e-8):
    # TODO: return w - (eta / (sqrt(v) + eps)) * g
    return w

print(np.round(rmsprop_step(1.0, 0.1, 0.04, 0.01), 3))`,
        expectedOutput: '0.995',
        solution: `import numpy as np
def rmsprop_step(w, g, v, eta, eps=1e-8):
    return w - (eta / (np.sqrt(v) + eps)) * g

print(np.round(rmsprop_step(1.0, 0.1, 0.04, 0.01), 3))`,
        hints: ['Divide learning rate by (sqrt(v) + eps) and multiply by g.'],
        testKeywords: ['np.sqrt', 'w -']
      }
    },
    interviewQuestions: [{
      question: 'Compare SGD, Momentum, and Adam. What are the key trade-offs in convergence speed and generalization?',
      answer: 'SGD is simple and generalizes well but converges slowly. Momentum accelerates updates through flat regions. Adam converges fastest using adaptive learning rates, but can generalize poorly compared to SGD on complex surfaces.',
      companyTags: ['Google'],
      difficulty: 'Advanced'
    }]
  },
  {
    id: 'bias-variance-tradeoff',
    title: 'Bias-Variance Tradeoff',
    category: 'Foundations & Math',
    description: 'Balance model capacity to avoid underfitting and overfitting.',
    formula: 'Error = Bias^2 + Variance + IrreducibleError',
    theory: `### Bias-Variance Tradeoff Theory

The bias-variance tradeoff decomposition breaks down expected prediction error:
$$ExpectedError = Bias^2 + Variance + \\sigma^2_{noise}$$

#### Core Metrics
- **Bias**: Error from incorrect assumptions in the model (underfitting).
- **Variance**: Sensitivity to small fluctuations in training data (overfitting).
- **Precision**: Proportion of true positive predictions among all positive predictions:
$$Precision = \\frac{TP}{TP + FP}$$
- **Recall**: Proportion of true positive predictions detected among all actual positive samples:
$$Recall = \\frac{TP}{TP + FN}$$
- **F1 Score**: Harmonic mean of Precision and Recall:
$$F1 = 2 \\cdot \\frac{Precision \\cdot Recall}{Precision + Recall}$$`,
    interactiveSummary: "Interactive demonstration of this concept.",
    simulatorId: 'bias-variance',
    quiz: [
      { id: 'bv_q1', question: 'Which metric calculates the harmonic mean of Precision and Recall?', options: ['F1 Score', 'Accuracy', 'MSE', 'ROC-AUC'], correctAnswer: 'F1 Score', explanation: 'The F1 Score is the harmonic mean of precision and recall, balancing both metrics.' }
    ],
    coding: {
      tutorial: {
      title: 'Precision and Recall Calculator',
      description: 'Calculate Precision and Recall given TP, FP, and FN.',
      pseudoCode: "",
        starterCode: `def calc_metrics(tp, fp, fn):
    # TODO: return precision, recall
    return 0.0, 0.0

print(calc_metrics(10, 2, 3))`,
        expectedOutput: '(0.8333333333333334, 0.7692307692307693)',
        solution: `def calc_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall

print(calc_metrics(10, 2, 3))`,
        hints: ['Implement the precision and recall formulas.'],
        testKeywords: ['tp + fp', 'tp + fn']
      },
      project: {
      title: 'Compute F1 Score',
      description: 'Compute F1 score from precision and recall values.',
      pseudoCode: "",
        starterCode: `def calc_f1(precision, recall):
    # TODO: return 2 * (p * r) / (p + r)
    return 0.0

print(np.round(calc_f1(0.8, 0.6), 3))`,
        expectedOutput: '0.686', // note: actually 2 * 0.48 / 1.4 = 0.6857...
        solution: `def calc_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)

print(round(calc_f1(0.8, 0.6), 3))`,
        hints: ['Multiply precision and recall, scale by 2, and divide by their sum.'],
        testKeywords: ['precision * recall']
      },
      assignment: {
      title: 'TPR and FPR Calculator',
      description: 'Calculate True Positive Rate and False Positive Rate.',
      pseudoCode: "",
        starterCode: `def calc_rates(tp, fn, fp, tn):
        # tpr = tp / (tp + fn)
        # fpr = fp / (fp + tn)
        # TODO: return tpr, fpr
        return 0.0, 0.0

print(calc_rates(10, 2, 2, 10))`,
        expectedOutput: '(0.8333333333333334, 0.16666666666666666)',
        solution: `def calc_rates(tp, fn, fp, tn):
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tpr, fpr

print(calc_rates(10, 2, 2, 10))`,
        hints: ['Divide TP by actual positives, and FP by actual negatives.'],
        testKeywords: ['tp + fn', 'fp + tn']
      }
    },
    interviewQuestions: [{
      question: 'Explain the bias-variance trade-off. How do model complexity, training error, and validation error behave?',
      answer: 'High bias models underfit, yielding high training and validation error. High variance models overfit, yielding low training error but high validation error. Optimal models balance complexity to minimize expected prediction error.',
      companyTags: ['Google'],
      difficulty: 'Intermediate'
    }]
  }
];


