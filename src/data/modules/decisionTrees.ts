import type { MLModule } from '../../types';

export const decisionTrees: MLModule = {
  "id": "decision-trees",
  "title": "Decision Trees",
  "category": "Supervised Learning",
  "description": "Split datasets recursively to isolate target variables based on impurity reductions.",
  "formula": "IG(D, A) = H(D) - H(D|A)",
  "theory": "### Decision Trees: A Comprehensive Guide\n\n#### What is it?\nA Decision Tree is a versatile, interpretable supervised machine learning algorithm used for both classification and regression tasks. It builds a flow-chart-like tree structure where each internal node represents a \"test\" or \"split\" on a feature (e.g., is age > 25?), each branch represents the outcome of the test, and each leaf node represents a class label (classification) or a continuous value (regression).\n\n#### Why do we need it?\nDecision Trees are needed because they mimic human decision-making, making them highly intuitive and explainable. You can easily visualize the tree and explain to stakeholders exactly why a specific prediction was made.\nFurthermore, unlike linear models (Linear/Logistic Regression), Decision Trees can automatically capture complex non-linear relationships and interactions between features without requiring manual feature transformations.\n\n#### How does it work?\nThe tree is built top-down recursively. It starts with the entire dataset at the root node.\n1. The algorithm searches through all features and all possible split points to find the \"best\" split.\n2. The \"best\" split is the one that maximally separates the target variables, essentially making the resulting child nodes as pure as possible.\n3. It splits the dataset into two subsets (left and right branches).\n4. This process repeats recursively on each child node until a stopping criterion is met (e.g., maximum depth reached, or a node is 100% pure).\nThe impurity is measured using metrics like Gini Impurity or Entropy.\n\n#### The Math Behind It\nLet a dataset $D$ contain samples belonging to $C$ different classes.\n\n**Entropy $H(D)$:**\nEntropy is a concept from information theory that measures the impurity or chaos of a dataset. If a dataset is 50/50 split between two classes, entropy is at its maximum (1.0). If it contains only one class, entropy is 0.\n$$H(D) = -\\sum_{c=1}^{C} p_c \\log_2(p_c)$$\nwhere $p_c$ is the proportion of samples belonging to class $c$ in dataset $D$.\n\n**Gini Impurity $Gini(D)$:**\nAn alternative to Entropy used by the CART (Classification and Regression Trees) algorithm. It measures the probability of incorrectly classifying a randomly chosen element.\n$$Gini(D) = 1 - \\sum_{c=1}^{C} p_c^2$$\n\n**Information Gain $IG(D, A)$:**\nTo choose a split on feature $A$, we calculate the Information Gain, which is the difference between the parent node's entropy and the weighted average entropy of the child nodes. The algorithm picks the split that maximizes Information Gain.\n$$IG(D, A) = H(D) - \\sum_{v \\in values(A)} \\frac{|D_v|}{|D|} H(D_v)$$\nwhere $D_v$ is the subset of $D$ for which feature $A$ has value $v$.\n\n#### Worked Example\nImagine classifying whether to \"Play Tennis\" (Yes/No) based on feature \"Outlook\" (Sunny/Overcast/Rain).\n- The root node has 9 Yes, 5 No. Entropy $H(D) = 0.940$.\n- Splitting by \"Outlook\" creates 3 child nodes:\n  - Overcast: 4 Yes, 0 No. Entropy = 0.0 (Pure!).\n  - Sunny: 2 Yes, 3 No. Entropy = 0.970.\n  - Rain: 3 Yes, 2 No. Entropy = 0.970.\n- Weighted average of child entropy = $(4/14)*0 + (5/14)*0.970 + (5/14)*0.970 = 0.693$.\n- Information Gain = $0.940 - 0.693 = 0.247$.\nThe algorithm computes this for all features (Humidity, Wind, etc.) and picks the one with the highest gain for the root split.\n\n#### Common Pitfalls\n1. **Overfitting:** Trees have a massive tendency to overfit. If grown unconstrained, they will create a leaf node for every single outlier, achieving 100% training accuracy but failing on test data (high variance).\n2. **Instability:** A small change in the training data can cause a completely different tree to be generated.\n3. **Orthogonal Boundaries:** Decision trees can only draw axis-aligned boundaries (step functions). They struggle to model smooth diagonal lines.\n\n#### When to Use vs Not Use\n**When to Use:**\n- Interpretability is critical.\n- Data has non-linear relationships and you don\'t want to engineer features.\n- You have a mix of categorical and numerical data.\n- As a building block for powerful ensemble models like Random Forests or Gradient Boosting Machines (XGBoost).\n\n**When Not to Use:**\n- You have highly smooth, linear datasets (linear regression is better).\n- You are using it completely standalone without limits (it will overfit).\n- You are predicting time-series trends requiring extrapolation (Trees cannot predict values outside the range of training data).\n\n#### Key Takeaways\n- Decision Trees split data recursively based on Information Gain (Entropy reduction) or Gini Impurity.\n- They are intuitive, interpretable, and handle non-linear data well.\n- They are highly prone to overfitting, so hyperparameters like `max_depth` and `min_samples_split` must be tuned, or ensemble methods should be used.\n",
  "interactiveSummary": "In this simulator, you can interact with a plotted dataset to see how a Decision Tree algorithm creates step-like partitions. Adjust the 'Max Depth' slider to observe how deeper trees create smaller, highly customized bounding boxes that fit the training data perfectly (demonstrating overfitting), while shallower trees maintain robust generalized boundaries.",
  "simulatorId": "decision-tree",
  "quiz": [
    {
      "id": "dt_q1",
      "question": "What is the entropy of a collection with 5 positive and 5 negative binary class samples?",
      "options": [
        "1.0",
        "0.0",
        "0.5",
        "2.0"
      ],
      "correctAnswer": "1.0",
      "explanation": "At 50/50 split, impurity is maximal. Entropy = - (0.5 * log2(0.5) + 0.5 * log2(0.5)) = - (-0.5 - 0.5) = 1.0."
    },
    {
      "id": "dt_q2",
      "question": "Which method helps prevent decision trees from overestimating noise in training data?",
      "options": [
        "Pruning (Pre- and Post-pruning)",
        "Increasing max depth",
        "Removing categorical features",
        "Bagging only"
      ],
      "correctAnswer": "Pruning (Pre- and Post-pruning)",
      "explanation": "Pruning collapses weak branches that fit minor noise variations, capping tree depth or minimum leaf counts to encourage generalization."
    },
    {
      "id": "dt_q3",
      "question": "What type of decision boundaries are produced by a standard decision tree?",
      "options": [
        "Axis-aligned orthogonal boundaries",
        "Curved polynomial boundaries",
        "Hyperplanes in kernel space",
        "Circular radial regions"
      ],
      "correctAnswer": "Axis-aligned orthogonal boundaries",
      "explanation": "Since trees split on one variable at a time (e.g. x_1 > threshold), boundaries are step-like and perpendicular to coordinate axes."
    },
    {
      "id": "dt_q4",
      "question": "How does a Decision Tree handle regression problems?",
      "options": [
        "By predicting the mean (average) of the target values in a leaf node.",
        "By converting the target into categorical bins.",
        "By running linear regression in every node.",
        "Decision Trees cannot be used for regression."
      ],
      "correctAnswer": "By predicting the mean (average) of the target values in a leaf node.",
      "explanation": "In Regression Trees, the split criterion minimizes Mean Squared Error (variance), and the final prediction is simply the average value of all samples in that specific leaf."
    },
    {
      "id": "dt_q5",
      "question": "What does Gini Impurity measure?",
      "options": [
        "The probability of incorrectly classifying a randomly chosen element in the node.",
        "The depth of the tree.",
        "The information content in bits.",
        "The correlation between features."
      ],
      "correctAnswer": "The probability of incorrectly classifying a randomly chosen element in the node.",
      "explanation": "Gini Impurity (1 - sum(p^2)) calculates the likelihood that an item would be mislabeled if randomly assigned a label based on the distribution in the node."
    },
    {
      "id": "dt_q6",
      "question": "Why are Decision Trees highly sensitive to small changes in training data?",
      "options": [
        "Because a small change might alter the very first split at the root, leading to a completely different tree topology.",
        "Because they require feature scaling.",
        "Because they use gradient descent.",
        "They are actually very robust and not sensitive to data changes."
      ],
      "correctAnswer": "Because a small change might alter the very first split at the root, leading to a completely different tree topology.",
      "explanation": "Trees are greedy algorithms. A minor data tweak can make feature B slightly better than feature A for the root split, completely changing every subsequent branch down the tree."
    },
    {
      "id": "dt_q7",
      "question": "Which of the following is a hyperparameter used to restrict tree size and prevent overfitting?",
      "options": [
        "min_samples_split",
        "learning_rate",
        "epochs",
        "num_clusters"
      ],
      "correctAnswer": "min_samples_split",
      "explanation": "Hyperparameters like min_samples_split, max_depth, and min_samples_leaf are used in pre-pruning to stop the tree from growing too large."
    },
    {
      "id": "dt_q8",
      "question": "What is Information Gain?",
      "options": [
        "The reduction in entropy achieved by partitioning data on a feature.",
        "The increase in classification accuracy on the test set.",
        "The amount of memory required to store the tree.",
        "A measure of computational efficiency."
      ],
      "correctAnswer": "The reduction in entropy achieved by partitioning data on a feature.",
      "explanation": "Information Gain = Entropy(Parent) - Weighted_Average_Entropy(Children). The algorithm chooses the feature that maximizes this gain."
    },
    {
      "id": "dt_q9",
      "question": "Do Decision Trees require feature scaling (normalization/standardization)?",
      "options": [
        "No, they are invariant to monotonic transformations of features.",
        "Yes, otherwise gradient descent will fail.",
        "Yes, but only for categorical features.",
        "Yes, because distance metrics are used."
      ],
      "correctAnswer": "No, they are invariant to monotonic transformations of features.",
      "explanation": "Since splits are based on thresholds (e.g. x > 5) and not on Euclidean distances or gradients, scaling features does not change the splits or the resulting tree."
    },
    {
      "id": "dt_q10",
      "question": "Can a Decision Tree extrapolate predictions outside the bounding box of its training data?",
      "options": [
        "No, it will just predict the value of the nearest leaf node.",
        "Yes, by fitting a linear trend.",
        "Yes, if it is a regression tree.",
        "Yes, but only if polynomial features are added."
      ],
      "correctAnswer": "No, it will just predict the value of the nearest leaf node.",
      "explanation": "Unlike linear regression which can project a line infinitely, a tree's prediction is bounded by the mean/majority class of the terminal leaf. It cannot extrapolate trends beyond the min/max of training data."
    }
  ],
  "coding": {
    "tutorial": {
      "title": "Calculate Label Entropy",
      "description": "Write a function to calculate the Shannon Entropy of a list of binary class labels.\n\n*Note: In scikit-learn, entropy calculation is handled internally when you set `criterion='entropy'` in DecisionTreeClassifier.*",
      "pseudoCode": "1. If list is empty, return 0\n2. Count occurrences of each class\n3. Divide counts by total length to get probabilities\n4. For each prob > 0, subtract (prob * log2(prob)) from total entropy\n5. Return total entropy",
      "starterCode": "import numpy as np\n\ndef calculate_entropy(labels):\n    # TODO: Calculate probability of each class and return entropy\n    entropy = 0.0\n    return entropy\n\nlbls = [1, 0, 1, 1, 1, 0]\nprint(\"Entropy:\", np.round(calculate_entropy(lbls), 3))",
      "expectedOutput": "Entropy: 0.918",
      "solution": "import numpy as np\n\ndef calculate_entropy(labels):\n    n = len(labels)\n    if n == 0:\n        return 0.0\n    counts = np.bincount(labels)\n    probs = counts / n\n    entropy = 0.0\n    for p in probs:\n        if p > 0:\n            entropy -= p * np.log2(p)\n    return entropy\n\nlbls = [1, 0, 1, 1, 1, 0]\nprint(\"Entropy:\", np.round(calculate_entropy(lbls), 3))",
      "hints": [
        "Find the counts of each unique class.",
        "Divide by total size to get probabilities.",
        "Calculate sum of -p * log2(p) for positive probabilities."
      ],
      "testKeywords": [
        "np.log2",
        "probs",
        "entropy"
      ]
    },
    "project": {
      "title": "Building a Decision Tree Classifier",
      "description": "Use Scikit-Learn\'s `DecisionTreeClassifier` to classify flowers (similar to the Iris dataset) based on petal width and length.",
      "pseudoCode": "1. Import DecisionTreeClassifier from sklearn.tree\n2. Create an instance of DecisionTreeClassifier with max_depth=3 to prevent overfitting\n3. Fit the model to the training data\n4. Predict the test data and return predictions",
      "starterCode": "import numpy as np\nfrom sklearn.tree import DecisionTreeClassifier\n\n# Dataset: [Petal Length, Petal Width], Labels: 0=Setosa, 1=Versicolor, 2=Virginica\nX_train = np.array([[1.4, 0.2], [1.5, 0.2], [4.5, 1.5], [4.7, 1.4], [6.0, 2.5], [5.1, 1.9]])\ny_train = np.array([0, 0, 1, 1, 2, 2])\nX_test = np.array([[1.3, 0.2], [4.6, 1.3], [5.9, 2.1]])\n\ndef train_tree(X_train, y_train, X_test):\n    # TODO: Initialize DecisionTreeClassifier (max_depth=3), fit, and predict\n    # model = ...\n    predictions = np.zeros(len(X_test))\n    return predictions\n\nprint(\"Predictions:\", train_tree(X_train, y_train, X_test))",
      "expectedOutput": "Predictions: [0 1 2]",
      "solution": "import numpy as np\nfrom sklearn.tree import DecisionTreeClassifier\n\nX_train = np.array([[1.4, 0.2], [1.5, 0.2], [4.5, 1.5], [4.7, 1.4], [6.0, 2.5], [5.1, 1.9]])\ny_train = np.array([0, 0, 1, 1, 2, 2])\nX_test = np.array([[1.3, 0.2], [4.6, 1.3], [5.9, 2.1]])\n\ndef train_tree(X_train, y_train, X_test):\n    model = DecisionTreeClassifier(max_depth=3, random_state=42)\n    model.fit(X_train, y_train)\n    predictions = model.predict(X_test)\n    return predictions\n\nprint(\"Predictions:\", train_tree(X_train, y_train, X_test))",
      "hints": [
        "Initialize with `model = DecisionTreeClassifier(max_depth=3)`.",
        "Train using `model.fit(X_train, y_train)`.",
        "Get outputs with `model.predict(X_test)`."
      ],
      "testKeywords": [
        "DecisionTreeClassifier",
        "fit",
        "predict"
      ]
    },
    "assignment": {
      "title": "Hyperparameter Tuning with Cross-Validation",
      "description": "Decision trees easily overfit. Construct a model and use `cross_val_score` to evaluate how a constrained tree (e.g. `max_depth=2`) performs across different folds.",
      "pseudoCode": "1. Import DecisionTreeClassifier and cross_val_score\n2. Initialize a DecisionTreeClassifier with max_depth=2 and min_samples_leaf=2\n3. Use cross_val_score with cv=3 on X, y\n4. Return the mean accuracy",
      "starterCode": "import numpy as np\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.model_selection import cross_val_score\n\n# Dataset: Medical patient vitals -> Label: 0=Healthy, 1=Sick\nX = np.array([[120, 80], [130, 85], [180, 110], [190, 120], [115, 75], [175, 105]])\ny = np.array([0, 0, 1, 1, 0, 1])\n\ndef eval_tree(X, y):\n    # TODO: Create DecisionTreeClassifier(max_depth=2, random_state=42)\n    # TODO: Run cross_val_score with cv=3 and return the mean score\n    mean_accuracy = 0.0\n    return mean_accuracy\n\nprint(\"Mean CV Accuracy:\", eval_tree(X, y))",
      "expectedOutput": "Mean CV Accuracy: 1.0",
      "solution": "import numpy as np\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.model_selection import cross_val_score\n\nX = np.array([[120, 80], [130, 85], [180, 110], [190, 120], [115, 75], [175, 105]])\ny = np.array([0, 0, 1, 1, 0, 1])\n\ndef eval_tree(X, y):\n    model = DecisionTreeClassifier(max_depth=2, random_state=42)\n    scores = cross_val_score(model, X, y, cv=3)\n    return np.mean(scores)\n\nprint(\"Mean CV Accuracy:\", eval_tree(X, y))",
      "hints": [
        "Instantiate `DecisionTreeClassifier` with hyperparameters like `max_depth=2`.",
        "Call `cross_val_score(model, X, y, cv=3)`.",
        "Compute the mean of the returned array."
      ],
      "testKeywords": [
        "DecisionTreeClassifier",
        "cross_val_score",
        "np.mean"
      ]
    }
  },
  "interviewQuestions": [
    {
      "question": "How do you prevent a decision tree from overfitting, and why are deep trees highly prone to high variance?",
      "answer": "Deep trees have high variance because they create complex rules to isolate every outlier (memorizing noise). Overfitting is prevented via Pre-pruning (setting max_depth, min_samples_split, min_samples_leaf) or Post-pruning (growing the tree fully and pruning branches with minimal cost-complexity benefit). Ensembles like Random Forest also solve this.",
      "companyTags": [
        "Amazon",
        "Apple"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "What is the difference between Entropy and Gini Impurity?",
      "answer": "Both measure node impurity. Entropy relies on logarithms (-sum(p*log2(p))), making it slightly slower to compute. Gini relies on squared probabilities (1 - sum(p^2)) and is faster. They usually yield very similar trees, but Gini is the default in algorithms like CART (Scikit-Learn).",
      "companyTags": [
        "Google",
        "Meta"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "How are continuous variables handled when creating splits in a Decision Tree?",
      "answer": "The continuous feature is sorted. The algorithm evaluates the midpoints between every adjacent pair of values as a potential split threshold. It calculates the Information Gain for each midpoint and chooses the threshold that yields the best split.",
      "companyTags": [
        "Uber",
        "Netflix"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "Why do Decision Trees not require feature scaling?",
      "answer": "Trees make splits based on hard thresholds (e.g., is X > 10?), not on distance metrics (like KNN) or gradient updates (like Neural Networks). A monotonic transformation like standardizing doesn\'t change the order of values, so the optimal split index remains identical.",
      "companyTags": [
        "LinkedIn",
        "Airbnb"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "Explain Information Gain and how it is calculated.",
      "answer": "Information Gain measures how much uncertainty is reduced after a split. It is calculated as the Entropy of the parent node minus the weighted average Entropy of the child nodes. The algorithm greedily selects the feature split that maximizes Information Gain.",
      "companyTags": [
        "Amazon",
        "Bloomberg"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What is a 'Greedy Algorithm' and why is a Decision Tree considered one?",
      "answer": "A greedy algorithm makes the locally optimal choice at each step with the hope of finding a global optimum. A Decision Tree evaluates splits only for the current node and picks the best one immediately; it does not look ahead to see if a suboptimal current split might lead to a better tree further down.",
      "companyTags": [
        "Apple",
        "Stripe"
      ],
      "difficulty": "Hard"
    },
    {
      "question": "What is the difference between Classification Trees and Regression Trees?",
      "answer": "Classification Trees predict discrete classes using Entropy/Gini to measure split quality, and the leaf outputs the majority class. Regression Trees predict continuous values, use Mean Squared Error (MSE) or Mean Absolute Error (MAE) to measure variance reduction for splits, and the leaf outputs the average of the samples.",
      "companyTags": [
        "Meta",
        "Capital One"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "Can Decision Trees model the XOR logic gate?",
      "answer": "Yes. While a single linear classifier cannot model XOR, a Decision Tree easily can by making two sequential splits: first splitting on feature X1, and then splitting on feature X2 within the branches, creating 4 quadrants.",
      "companyTags": [
        "Google",
        "Spotify"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What are the advantages of Decision Trees?",
      "answer": "They are highly interpretable (white-box model), require minimal data preparation (no scaling, handles missing values sometimes), can handle both numerical and categorical data, and automatically model non-linear relationships and feature interactions.",
      "companyTags": [
        "Netflix",
        "Pinterest"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "What are the main disadvantages of Decision Trees?",
      "answer": "They are extremely prone to overfitting, they are highly unstable (high variance) where small data changes cause massive structural changes, and they cannot extrapolate beyond the training data range.",
      "companyTags": [
        "Amazon",
        "Twitter"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "How does Random Forest fix the shortcomings of a single Decision Tree?",
      "answer": "Random Forest builds hundreds of deep decision trees on bootstrapped subsets of the data and uses random subsets of features for each split. By averaging the predictions (or majority voting), it drastically reduces the variance and instability of single trees without increasing bias.",
      "companyTags": [
        "Google",
        "Uber"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What is Cost-Complexity Pruning?",
      "answer": "Also known as Weakest Link Pruning. It\'s a post-pruning technique that minimizes an objective function: Error(Tree) + alpha * |Leaves|. As alpha increases, the penalty for having a large tree increases, leading the algorithm to prune branches that don\'t significantly reduce error.",
      "companyTags": [
        "Meta",
        "Bloomberg"
      ],
      "difficulty": "Hard"
    },
    {
      "question": "If a Decision Tree's depth is not restricted, what will the training error be?",
      "answer": "Assuming there are no conflicting identical samples with different labels, an unconstrained tree will eventually create a leaf node for every single sample. The training error will be exactly 0 (perfect memorization), which represents massive overfitting.",
      "companyTags": [
        "Apple",
        "Airbnb"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "How do Decision Trees handle missing values?",
      "answer": "While scikit-learn's CART implementation historically required imputation, algorithms like XGBoost or standard C4.5 can handle missing values natively by distributing instances with missing values to both branches using probabilistic weights, or learning a 'default' direction for missing data during training.",
      "companyTags": [
        "LinkedIn",
        "Uber"
      ],
      "difficulty": "Hard"
    },
    {
      "question": "What is the time complexity of building a Decision Tree?",
      "answer": "For a dataset with n samples and f features, at each node, it takes O(n log n) to sort the continuous features and O(n) to find the split. Since the tree depth is typically O(log n), the overall training complexity is roughly O(f * n * log n^2).",
      "companyTags": [
        "Google",
        "Meta"
      ],
      "difficulty": "Hard"
    },
    {
      "question": "How does setting `min_samples_leaf` prevent overfitting?",
      "answer": "It forces every terminal leaf node to contain at least a specified number of samples. This stops the tree from creating hyper-specific branches that isolate 1 or 2 outlier data points, smoothing out the decision boundary.",
      "companyTags": [
        "Amazon",
        "Stripe"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "Can a feature be used more than once in a Decision Tree?",
      "answer": "Yes. A continuous feature can be split at threshold X > 50 at the root, and later down the tree, that same feature can be split again at X > 75 to create finer granular partitions.",
      "companyTags": [
        "Netflix",
        "Spotify"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "Why might a Decision Tree perform poorly on image data?",
      "answer": "Images have high dimensionality and pixel values hold spatial relationships. A standard Decision Tree looks at pixels independently and tries to draw axis-aligned cuts, ignoring spatial locality, leading to massive, poor-performing trees. CNNs are far better.",
      "companyTags": [
        "Apple",
        "Pinterest"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What happens if you have highly correlated features in a Decision Tree?",
      "answer": "Unlike Linear Regression where multicollinearity ruins coefficient estimates, Decision Trees are robust to it. The tree will simply pick one of the correlated features to split on. The other feature will likely be ignored, but the model\'s accuracy won't degrade.",
      "companyTags": [
        "Bloomberg",
        "Capital One"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "Why is 'Information Gain Ratio' sometimes preferred over 'Information Gain'?",
      "answer": "Information Gain is biased towards features with many unique categorical levels (e.g., an ID column will have max gain because it perfectly separates data, but it's useless for generalization). Gain Ratio penalizes features with high intrinsic cardinality to correct this bias.",
      "companyTags": [
        "Google",
        "Meta"
      ],
      "difficulty": "Hard"
    }
  ]
};
