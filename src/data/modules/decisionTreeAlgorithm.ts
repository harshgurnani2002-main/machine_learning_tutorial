import type { MLModule } from '../../types';

export const decisionTreeAlgorithm: MLModule = {
  "id": "decision-tree-algorithm",
  "title": "Decision Tree Splits",
  "category": "Supervised Learning",
  "description": "Explore recursive feature splitting in tree nodes.",
  "formula": "IG(D, A) = H(D) - H(D|A)",
  "theory": `### What are Decision Trees?

A Decision Tree is a powerful, non-parametric supervised learning algorithm that is utilized for both classification and regression tasks. At its core, it builds a predictive model in the form of a tree structure, systematically breaking down a complex dataset into progressively smaller, more homogenous subsets while simultaneously developing an associated decision tree incrementally. The culmination of this process is a tree with decision nodes (where data is split) and leaf nodes (where final predictions are made).

### Why do we need it?

Decision trees are highly prized for their unparalleled interpretability. Unlike "black box" models (e.g., deep neural networks), a decision tree mimics human decision-making processes. You can visually trace the path from the root to a leaf to understand exactly why a specific prediction was made. They require minimal data preparation—there is no need to normalize or scale features—and they seamlessly handle a mixture of numerical and categorical data. Furthermore, they are robust to outliers and form the essential building blocks for highly sophisticated ensemble methods like Random Forests and Gradient Boosting Machines (GBM).

### How does it work?

The foundational algorithm for building decision trees (such as ID3, C4.5, or CART) employs a top-down, greedy search through the vast space of possible branches without backtracking. 
1. **Initialization:** The algorithm starts at the root node, which encompasses the entire training dataset.
2. **Feature Evaluation:** It systematically evaluates all possible splits across all features to find the one that best separates the data into distinct classes (for classification) or minimizes variance (for regression).
3. **Partitioning:** The dataset is split into two (CART algorithm) or more subsets based on the selected feature and its corresponding threshold.
4. **Recursion:** This splitting process is repeated recursively on each generated child node.
5. **Termination:** The recursion halts when a stopping criterion is met—such as reaching a maximum depth limit, falling below a minimum number of samples per leaf, or achieving complete purity in a node.

### The Math Behind It

The "best" split is determined using an impurity metric. The objective at each node is to select a split that results in the purest possible child nodes. For classification, the prevailing metrics are Gini Impurity and Information Gain (based on Entropy).

#### 1. Entropy
Entropy is a concept borrowed from information theory that measures the amount of uncertainty, unpredictability, or impurity in a set of examples $D$. For a binary classification task with proportions of positive $p_+$ and negative $p_-$ examples:
$$H(D) = -p_+ \\log_2(p_+) - p_- \\log_2(p_-)$$

#### 2. Information Gain
Information Gain is the expected reduction in entropy achieved by partitioning the examples according to a given attribute $A$. It is essentially the entropy of the parent node minus the weighted sum of the entropies of the child nodes:
$$IG(D, A) = H(D) - \\sum_{v \\in Values(A)} \frac{|D_v|}{|D|} H(D_v)$$
Where $D_v$ is the subset of $D$ for which attribute $A$ has value $v$. The algorithm greedily chooses the split that maximizes $IG$.

#### 3. Gini Impurity
Predominantly used by the CART (Classification and Regression Trees) algorithm, Gini measures how often a randomly chosen element from the set would be incorrectly labeled if it were randomly labeled according to the distribution of labels in the subset. It is computationally faster than entropy because it avoids logarithmic calculations.
$$Gini(D) = 1 - \\sum_{i=1}^C (p_i)^2$$
Where $C$ is the total number of classes and $p_i$ is the probability of an item belonging to class $i$.

### Worked Example

Imagine you possess a dataset of 10 people (6 enjoy Apples, 4 enjoy Oranges).
The initial Gini impurity of the parent node is:
$$Gini_{parent} = 1 - \\left(\\left(\frac{6}{10}\right)^2 + \\left(\frac{4}{10}\right)^2\right) = 1 - (0.36 + 0.16) = 1 - 0.52 = 0.48$$

You evaluate a split based on the feature "Gender":
- **Left Node (Male):** Contains 5 people (4 like Apples, 1 likes Oranges).
  $$Gini_L = 1 - \\left(\\left(\frac{4}{5}\right)^2 + \\left(\frac{1}{5}\right)^2\right) = 1 - (0.64 + 0.04) = 1 - 0.68 = 0.32$$
- **Right Node (Female):** Contains 5 people (2 like Apples, 3 like Oranges).
  $$Gini_R = 1 - \\left(\\left(\frac{2}{5}\right)^2 + \\left(\frac{3}{5}\right)^2\right) = 1 - (0.16 + 0.36) = 1 - 0.52 = 0.48$$

The weighted average Gini of the child nodes after the split is:
$$Gini_{split} = \\left(\frac{5}{10}\right) \times 0.32 + \\left(\frac{5}{10}\right) \times 0.48 = 0.16 + 0.24 = 0.40$$

Because $0.40 < 0.48$, this specific split effectively reduces impurity. The tree evaluates all features and thresholds, choosing the one that yields the lowest split impurity.

### Common Pitfalls

1. **Overfitting:** This is the achilles heel of decision trees. Unconstrained trees will grow arbitrarily deep, creating a leaf for almost every single training instance. They memorize the training data perfectly but fail spectacularly to generalize to unseen data. This is mitigated through regularization techniques like pruning, setting a \`max_depth\`, or enforcing \`min_samples_split\`.
2. **Instability (High Variance):** Decision trees are notoriously sensitive to minor perturbations in the training data. A slightly different training set can result in a radically different tree structure. This instability is the primary reason why ensemble methods (Random Forests) are preferred in robust production environments.
3. **Bias towards dominant classes:** If the dataset is highly imbalanced, the tree will be heavily biased toward predicting the majority class. Techniques like class weighting or SMOTE are often necessary.
4. **Greedy Nature:** The greedy approach makes locally optimal decisions at each node. This does not guarantee finding the globally optimal tree structure.

### When to Use vs Not Use

**When to Use:**
- When model transparency, explainability, and interpretability are paramount (e.g., healthcare, finance compliance).
- When dealing with mixed data types (categorical and continuous) without wanting to extensively preprocess, scale, or one-hot encode.
- As a fast, reliable baseline model for both classification and regression tasks.

**When Not to Use:**
- For highly complex, non-linear relationships where slight variations in data are frequent (ensembles or neural networks are vastly superior here).
- When the dataset contains an enormous number of features with sparse representations (like text classification with TF-IDF or genomic data), where linear models or SVMs often perform significantly better.
- Extrapolation in regression: Decision trees cannot predict values outside the range of the training data.

### Key Takeaways
- Decision Trees partition data recursively to maximize purity (minimize impurity) using orthogonal splits.
- They utilize Gini Impurity or Information Gain for classification, and Mean Squared Error (variance reduction) for regression.
- They are remarkably easy to understand and visualize but are highly susceptible to overfitting if not properly regularized or pruned.
- They form the foundational weak learners for state-of-the-art ensemble models like Random Forests and XGBoost.`,
  "interactiveSummary": "In this interactive simulator, you can build a decision tree step-by-step. Adjust hyperparameters like 'max_depth' and 'min_samples_split' to see how they directly impact the tree's size and shape. Observe the decision boundaries changing in real-time as the tree depth increases, illustrating the concept of overfitting.",
  "simulatorId": "tree-splits",
  "quiz": [
    {
      "id": "dt_q1",
      "question": "What is the main goal of a split in a Decision Tree?",
      "options": [
        "To increase the depth of the tree",
        "To increase the Gini Impurity",
        "To maximize Information Gain or minimize impurity",
        "To reduce the number of features"
      ],
      "correctAnswer": "To maximize Information Gain or minimize impurity",
      "explanation": "At each node, the Decision Tree evaluates splits and chooses the one that results in the purest child nodes, which corresponds to minimizing impurity or maximizing information gain."
    },
    {
      "id": "dt_q2",
      "question": "Which impurity metric is primarily used by the CART algorithm?",
      "options": [
        "Information Gain",
        "Gini Impurity",
        "Mean Absolute Error",
        "Log Loss"
      ],
      "correctAnswer": "Gini Impurity",
      "explanation": "CART (Classification and Regression Trees) primarily uses Gini Impurity for classification splits because it's computationally faster than calculating the logarithm required for Entropy."
    },
    {
      "id": "dt_q3",
      "question": "What is a major disadvantage of a fully grown, unpruned Decision Tree?",
      "options": [
        "It requires too much data preparation",
        "It suffers from severe underfitting",
        "It is highly prone to overfitting the training data",
        "It cannot handle numerical features"
      ],
      "correctAnswer": "It is highly prone to overfitting the training data",
      "explanation": "If allowed to grow without constraints, a tree will create a leaf for almost every training sample, fitting noise and leading to poor generalization (overfitting)."
    },
    {
      "id": "dt_q4",
      "question": "How do Decision Trees handle non-linear relationships?",
      "options": [
        "By using polynomial kernels",
        "By applying a sigmoid activation function",
        "By partitioning the feature space into hyper-rectangles",
        "They cannot handle non-linear relationships"
      ],
      "correctAnswer": "By partitioning the feature space into hyper-rectangles",
      "explanation": "Decision trees make orthogonal (axis-aligned) splits, dividing the feature space into distinct rectangular regions, which allows them to easily model non-linear boundaries."
    },
    {
      "id": "dt_q5",
      "question": "What is Cost-Complexity Pruning?",
      "options": [
        "A method to add more nodes to a tree",
        "A regularization technique that adds a penalty for the number of terminal nodes (leaves)",
        "A technique to scale features",
        "A way to handle missing values in the tree"
      ],
      "correctAnswer": "A regularization technique that adds a penalty for the number of terminal nodes (leaves)",
      "explanation": "Cost-complexity pruning (or weakest link pruning) finds the optimal tree size by minimizing a cost function that penalizes both training error and tree size (number of leaves)."
    },
    {
      "id": "dt_q6",
      "question": "Which of these is NOT a common stopping criterion for splitting a node?",
      "options": [
        "Maximum tree depth reached",
        "Minimum number of samples required to split a node",
        "Node becomes completely pure",
        "Learning rate decays to zero"
      ],
      "correctAnswer": "Learning rate decays to zero",
      "explanation": "Learning rate is a concept used in gradient-based optimization and boosting, not in the standard greedy building of a single Decision Tree."
    },
    {
      "id": "dt_q7",
      "question": "If a dataset has 10 samples, 5 positive and 5 negative, what is its Gini impurity?",
      "options": [
        "0.0",
        "0.5",
        "1.0",
        "0.25"
      ],
      "correctAnswer": "0.5",
      "explanation": "Gini = 1 - ((5/10)^2 + (5/10)^2) = 1 - (0.25 + 0.25) = 1 - 0.5 = 0.5. This is the maximum possible Gini impurity for a binary classification, representing total uncertainty."
    },
    {
      "id": "dt_q8",
      "question": "What metric do Decision Trees use to evaluate splits for a Regression problem?",
      "options": [
        "Gini Impurity",
        "Information Gain",
        "Mean Squared Error (MSE) or Variance Reduction",
        "Cross-Entropy Loss"
      ],
      "correctAnswer": "Mean Squared Error (MSE) or Variance Reduction",
      "explanation": "For regression, trees predict the mean value of the samples in a leaf. They choose splits that minimize the variance (or Mean Squared Error) within the child nodes."
    },
    {
      "id": "dt_q9",
      "question": "Why are Decision Trees considered 'greedy' algorithms?",
      "options": [
        "They use up all available memory",
        "They make the locally optimal choice at each node without considering global tree structure",
        "They require large amounts of data to train",
        "They always find the global minimum error"
      ],
      "correctAnswer": "They make the locally optimal choice at each node without considering global tree structure",
      "explanation": "A greedy algorithm makes the best choice at the current step. Decision Trees find the best immediate split, but this does not guarantee the optimal tree overall."
    },
    {
      "id": "dt_q10",
      "question": "How do Decision Trees inherently handle feature scaling?",
      "options": [
        "They require standard scaling (mean 0, variance 1)",
        "They require Min-Max scaling",
        "They do not require feature scaling because splits are invariant to monotonic transformations",
        "They scale features internally during the split computation"
      ],
      "correctAnswer": "They do not require feature scaling because splits are invariant to monotonic transformations",
      "explanation": "A split relies only on the ordering of values (e.g., is Feature A > 10?). Scaling or shifting the feature values preserves the order, so the same splits will be found."
    }
  ],
  "coding": {
    "tutorial": {
      "title": "Split Check",
      "description": "Verify if labels are pure (all 0 or all 1).\n\n*Scikit-Learn Equivalent:* When using `DecisionTreeClassifier`, sklearn handles leaf purity checks internally based on the `criterion` (gini or entropy).",
      "pseudoCode": "function is_pure(y):\n    return length(unique_elements(y)) <= 1",
      "starterCode": "def is_pure(y):\n    # TODO: Return True if all elements in y are identical\n    return False\n\nprint(is_pure([1, 1, 1]))",
      "expectedOutput": "True",
      "solution": "def is_pure(y):\n    return len(set(y)) <= 1\n\nprint(is_pure([1, 1, 1]))",
      "hints": [
        "Convert y to a set and check its length."
      ],
      "testKeywords": [
        "set"
      ]
    },
    "project": {
      "title": "Decision Tree Classifier Pipeline",
      "description": "Use sklearn's `DecisionTreeClassifier` to classify a toy dataset. We will restrict the `max_depth` to prevent overfitting.",
      "starterCode": "import numpy as np\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.metrics import accuracy_score\n\n# Toy dataset (e.g., predicting fruit type: 0=Apple, 1=Orange)\nX = np.array([[150, 1], [170, 0], [140, 1], [180, 0], [130, 1], [190, 0]])\ny = np.array([0, 1, 0, 1, 0, 1])\n\n# TODO: Create a pipeline with DecisionTreeClassifier(max_depth=2, random_state=42)\n# TODO: Fit the model and predict on X\nclf = None\npredictions = []\nacc = 0.0\n\nprint(\"Predictions:\", predictions)\nprint(\"Accuracy:\", acc)",
      "expectedOutput": "Predictions: [0 1 0 1 0 1]\nAccuracy: 1.0",
      "solution": "import numpy as np\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.metrics import accuracy_score\n\nX = np.array([[150, 1], [170, 0], [140, 1], [180, 0], [130, 1], [190, 0]])\ny = np.array([0, 1, 0, 1, 0, 1])\n\nclf = make_pipeline(DecisionTreeClassifier(max_depth=2, random_state=42))\nclf.fit(X, y)\npredictions = clf.predict(X)\nacc = accuracy_score(y, predictions)\n\nprint(\"Predictions:\", predictions)\nprint(\"Accuracy:\", acc)",
      "hints": [
        "Use make_pipeline(DecisionTreeClassifier(max_depth=2, random_state=42))",
        "Call fit() and predict()",
        "Compute accuracy_score"
      ],
      "testKeywords": [
        "make_pipeline",
        "DecisionTreeClassifier",
        "fit",
        "predict"
      ]
    },
    "assignment": {
      "title": "Evaluate Tree Pruning with Cross Validation",
      "description": "Decision Trees overfit if they go too deep. Train a `DecisionTreeClassifier` and evaluate its performance using `cross_val_score`.",
      "starterCode": "import numpy as np\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.model_selection import cross_val_score\n\n# Sample Dataset\nX = np.random.RandomState(42).rand(100, 5)\ny = (X[:, 0] + X[:, 1] > 1.0).astype(int)  # Class depends on first two features\n\n# TODO: Create a DecisionTreeClassifier with max_depth=3, random_state=42\n# TODO: Evaluate with cross_val_score using cv=5\nmodel = None\nscores = []\n\nprint(\"Mean Accuracy:\", np.round(np.mean(scores), 2))",
      "expectedOutput": "Mean Accuracy: 0.94",
      "solution": "import numpy as np\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.model_selection import cross_val_score\n\nX = np.random.RandomState(42).rand(100, 5)\ny = (X[:, 0] + X[:, 1] > 1.0).astype(int)\n\nmodel = DecisionTreeClassifier(max_depth=3, random_state=42)\nscores = cross_val_score(model, X, y, cv=5)\n\nprint(\"Mean Accuracy:\", np.round(np.mean(scores), 2))",
      "hints": [
        "Initialize DecisionTreeClassifier(max_depth=3, random_state=42)",
        "Pass the model, X, and y into cross_val_score(model, X, y, cv=5)"
      ],
      "testKeywords": [
        "cross_val_score",
        "DecisionTreeClassifier",
        "max_depth=3"
      ]
    }
  },
  "interviewQuestions": [
    {
      "question": "Explain how a Decision Tree algorithm decides where to split.",
      "answer": "It iterates over all features and all possible split points for those features. For each split point, it calculates the impurity (like Gini or Entropy) of the resulting child nodes. It selects the split that maximizes information gain or minimizes impurity.",
      "companyTags": [
        "Amazon",
        "Microsoft"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What is Information Gain and how is it calculated?",
      "answer": "Information Gain is the reduction in entropy achieved by splitting a dataset. It is calculated as the entropy of the parent node minus the weighted average of the entropies of the child nodes.",
      "companyTags": [
        "Google",
        "Meta"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What is the difference between Gini Impurity and Entropy?",
      "answer": "Both measure node impurity. Entropy is computationally more expensive because it uses logarithms. Gini impurity uses squared probabilities, making it slightly faster to compute. In practice, they usually produce very similar trees.",
      "companyTags": [
        "Apple",
        "Netflix"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "Why are Decision Trees prone to overfitting?",
      "answer": "Because they are non-parametric and can grow indefinitely to perfectly fit the training data, creating a leaf for every single training point. This memorizes the noise in the training set instead of learning general patterns.",
      "companyTags": [
        "Meta",
        "Amazon"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "How can you prevent a Decision Tree from overfitting?",
      "answer": "You can use pre-pruning (early stopping) by setting hyperparameters like max_depth, min_samples_split, or min_samples_leaf. Alternatively, you can use post-pruning like Cost-Complexity Pruning to remove branches that contribute little to predictive power.",
      "companyTags": [
        "Google",
        "Microsoft"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What is Cost-Complexity Pruning?",
      "answer": "It\'s a post-pruning technique that minimizes an objective function balancing the tree's training error against its size (number of leaves). An alpha parameter controls the penalty applied to the tree size. Branches that don\'t reduce error enough to justify their complexity are pruned.",
      "companyTags": [
        "Apple",
        "Meta"
      ],
      "difficulty": "Hard"
    },
    {
      "question": "Do Decision Trees require feature scaling?",
      "answer": "No, because the splits are determined based on thresholds that separate the data, which depend purely on the ordering of the feature values. Monotonic transformations like scaling or standardizing do not change the split points.",
      "companyTags": [
        "Netflix",
        "Amazon"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "Can Decision Trees handle missing values?",
      "answer": "Yes, some implementations (like C4.5 and XGBoost) handle missing values inherently by assigning them to the branch that minimizes loss, or by using surrogate splits. However, sklearn's basic CART implementation does not support missing values out-of-the-box and requires imputation.",
      "companyTags": [
        "Google",
        "OpenAI"
      ],
      "difficulty": "Hard"
    },
    {
      "question": "Are Decision Trees affected by outliers?",
      "answer": "They are relatively robust to outliers in the features because the splits are based on ordering, not absolute values. However, outliers in the target variable (in regression) can heavily influence the MSE, leading to deeper trees that try to fit those outliers.",
      "companyTags": [
        "Microsoft",
        "Meta"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What is the time complexity of building a Decision Tree?",
      "answer": "At each node, sorting the data takes O(n log n). There are d features, and the tree depth is typically O(log n). Thus, the training complexity is roughly O(n * d * log n). Prediction complexity is O(depth) = O(log n).",
      "companyTags": [
        "Google",
        "Amazon"
      ],
      "difficulty": "Hard"
    },
    {
      "question": "Why use Random Forest instead of a single Decision Tree?",
      "answer": "A single Decision Tree has high variance and easily overfits. Random Forest aggregates the predictions of many trees (trained on different bootstrap samples and using random feature subsets) to significantly reduce variance and improve generalization.",
      "companyTags": [
        "Apple",
        "Netflix"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "What is Information Gain Ratio?",
      "answer": "Information Gain Ratio is used by the C4.5 algorithm to bias against features with a large number of distinct values (like ID columns), which artificially inflate standard Information Gain. It divides the IG by the Intrinsic Information (split entropy).",
      "companyTags": [
        "Meta",
        "Microsoft"
      ],
      "difficulty": "Hard"
    },
    {
      "question": "How is a Decision Tree used for Regression?",
      "answer": "Instead of predicting a class, a regression tree predicts the mean value of the target variable for the samples in a leaf. It selects splits that minimize the Mean Squared Error (MSE) or variance in the child nodes.",
      "companyTags": [
        "Google",
        "Amazon"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What does a Decision Tree decision boundary look like?",
      "answer": "It consists of axis-aligned, orthogonal lines (or planes/hyperplanes in higher dimensions), partitioning the feature space into hyper-rectangles.",
      "companyTags": [
        "Apple",
        "OpenAI"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "Is a Decision Tree a parametric or non-parametric model?",
      "answer": "It is a non-parametric model. It does not assume a fixed set of parameters or a specific functional form for the relationship between features and target; its structure grows with the complexity of the data.",
      "companyTags": [
        "Meta",
        "Netflix"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "What is 'greediness' in the context of building a Decision Tree?",
      "answer": "It refers to the algorithm making the locally optimal choice at each step (finding the best immediate split) without looking ahead to see if a sub-optimal immediate split might lead to a better overall tree later.",
      "companyTags": [
        "Google",
        "Amazon"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "How do you interpret feature importance from a Decision Tree?",
      "answer": "Feature importance is calculated as the normalized total reduction of the criterion (e.g., Gini impurity) brought by that feature. Features placed higher in the tree and used in more splits usually have higher importance.",
      "companyTags": [
        "Microsoft",
        "Meta"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What is the limitation of evaluating feature importance in a single Decision Tree?",
      "answer": "In the presence of highly correlated features, the tree might arbitrarily choose one of them for a split, assigning it high importance while assigning the other zero, masking the true importance of the correlated feature.",
      "companyTags": [
        "Apple",
        "Amazon"
      ],
      "difficulty": "Hard"
    },
    {
      "question": "What are surrogate splits?",
      "answer": "Surrogate splits are alternative splits closely matching the primary split on a node. If a primary feature is missing for a sample during inference, the tree uses the surrogate split to determine which path to follow.",
      "companyTags": [
        "Google",
        "Netflix"
      ],
      "difficulty": "Hard"
    },
    {
      "question": "Explain the difference between ID3, C4.5, and CART algorithms.",
      "answer": "ID3 uses Information Gain and only handles categorical data. C4.5 improves ID3 by handling continuous data and using Gain Ratio. CART (Classification and Regression Trees) uses Gini Impurity, produces strictly binary trees, and supports regression.",
      "companyTags": [
        "Microsoft",
        "Meta"
      ],
      "difficulty": "Hard"
    }
  ]
};
