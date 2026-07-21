import type { MLModule } from '../../types';

export const logisticRegression: MLModule = {
  "id": "logistic-regression",
  "title": "Logistic Regression",
  "category": "Foundations & Math",
  "description": "Model discrete class probabilities using Log-Odds and Sigmoid functions.",
  "formula": "p = \\sigma(w^T x + b)",
    theory: `### Logistic Regression: A Comprehensive Guide
  
  #### What is it?
  Logistic Regression is a fundamental supervised learning algorithm used primarily for classification tasks. Despite the word "regression" in its name, it is a classification algorithm. It is used to predict the probability that a given instance belongs to a particular class (e.g., probability that an email is spam). If the probability is greater than a certain threshold (usually 0.5), it classifies the instance into the positive class; otherwise, the negative class. This is known as Binary Logistic Regression.
  
  #### Why do we need it?
  We need Logistic Regression when the target variable is categorical (discrete). While Linear Regression outputs unbounded continuous values (which can be > 1 or < 0), probabilities must strictly lie between 0 and 1.
  1. **Classification:** Identifying whether a tumor is malignant or benign.
  2. **Probability Estimation:** Not just classifying, but outputting a confidence score. For instance, an ad click-through rate (predicting a 15% chance a user will click an ad).
  It serves as the foundation for more complex models, such as neural networks (where it acts as a single neuron with a sigmoid activation).
  
  #### How does it work?
  Logistic Regression works by taking the linear combination of inputs (similar to Linear Regression) and passing it through a non-linear activation function called the Sigmoid (or Logistic) function. This function squashes any real-valued number into the range [0, 1].
  Instead of minimizing Mean Squared Error, Logistic Regression minimizes the Binary Cross-Entropy Loss (also known as Log Loss). This loss function heavily penalizes confident but incorrect predictions, ensuring that the model assigns high probabilities to the correct classes.
  
  #### The Math Behind It
  Let $x \\in \\mathbb{R}^n$ be the feature vector and $y \\in \\{0, 1\\}$ be the binary target.
  
  **The Sigmoid Function:**
  $\\sigma(z) = frac{1}{1 + e^{-z}}$
  
  **Hypothesis Function:**
  We compute the linear combination $z = w^T x + b$, and then apply the sigmoid function to get the probability $p$:
  $p = h_w(x) = \\sigma(w^T x + b) = frac{1}{1 + e^{-(w^T x + b)}}$
  
  **The Log-Odds (Logit):**
  The inverse of the sigmoid is the logit function, which maps probabilities back to the real number line:
  $\\log \\left( frac{p}{1 - p} ight) = w^T x + b$
  This shows that Logistic Regression is a linear model for the *log-odds* of the positive class.
  
  **Binary Cross-Entropy Loss (Log Loss):**
  $J(w, b) = -frac{1}{m} \\sum_{i=1}^{m} \\left[ y^{(i)} \\log(h_w(x^{(i)})) + (1 - y^{(i)}) \\log(1 - h_w(x^{(i)})) ight]$
  
  **Parameter Optimization (Gradient Descent):**
  Interestingly, the gradient of the log loss with respect to the weights is mathematically identical to the gradient of MSE in linear regression (though the hypothesis function is different):
  $w_j \\leftarrow w_j - frac{u0007lpha}{m} \\sum_{i=1}^{m} \\left( h_w(x^{(i)}) - y^{(i)} ight) x_j^{(i)}$
  
  #### Worked Example
  Suppose we want to predict if a student passes ($y=1$) or fails ($y=0$) based on hours studied ($x$).
  Assume the model learned $w = 1.5$ and $b = -3$.
  A student studies for 4 hours.
  1. Calculate linear part: $z = 1.5(4) - 3 = 6 - 3 = 3$.
  2. Apply sigmoid: $p = 1 / (1 + e^{-3}) u0007pprox 1 / (1 + 0.0498) u0007pprox 0.952$.
  The model predicts a 95.2% probability of passing. Since 0.952 > 0.5, the classification is "Pass".
  
  #### Common Pitfalls
  1. **Non-linear Data:** Logistic regression is a linear classifier. It draws a straight line (or hyperplane) to separate classes. If classes are not linearly separable, it will perform poorly unless features are transformed.
  2. **Outliers:** While less sensitive than linear regression, massive outliers can still skew the decision boundary.
  3. **Class Imbalance:** If 99% of data is class A, the model might just always predict class A. Techniques like class weighting or SMOTE are required.
  
  #### When to Use vs Not Use
  **When to Use:**
  - Binary classification problems where a linear decision boundary is sufficient.
  - You need well-calibrated probabilities, not just hard class labels.
  - You need a lightweight, highly interpretable model.
  
  **When Not to Use:**
  - The data has complex, non-linear relationships (use Decision Trees, Random Forests, or Neural Networks).
  - The dataset has many missing values and outliers (Tree models are more robust to this).
  
  #### Key Takeaways
  - Logistic Regression outputs probabilities between 0 and 1 using the Sigmoid function.
  - It is a linear classifier because its decision boundary is linear.
  - It minimizes Binary Cross-Entropy Loss (Log Loss) instead of Mean Squared Error.
  
#### Python Implementation

\`\`\`python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=2, random_state=42)
model = LogisticRegression()
model.fit(X, y)
print(f"Accuracy: {model.score(X, y):.2f}")
print(f"Coefficients: {model.coef_}")
\`\`\`
`,
  "interactiveSummary": "This interactive simulator shows the Logistic Regression decision boundary on a 2D canvas. Click to add red (Class 0) or blue (Class 1) data points. The model automatically fits a sigmoid probability surface and draws the linear decision boundary as a separating line. The color gradient in the background represents predicted probabilities — darker orange means higher probability of Class 1. Try adding overlapping points to see how the boundary shifts, and observe how the sigmoid function translates linear scores into smooth 0-to-1 probabilities.",
  "simulatorId": "log-reg",
  "quiz": [
    {
      "id": "log_q1",
      "question": "Why is Mean Squared Error (MSE) not ideal for optimizing Logistic Regression parameters?",
      "options": [
        "The resulting loss landscape is non-convex, containing many local minima.",
        "MSE is mathematically impossible to differentiate with Sigmoids.",
        "MSE causes parameters to diverge immediately to infinity.",
        "Log loss always calculates faster than MSE."
      ],
      "correctAnswer": "The resulting loss landscape is non-convex, containing many local minima.",
      "explanation": "Applying MSE to a Sigmoid prediction results in a non-convex cost surface with local traps. Binary Cross-Entropy yields a convex landscape, ensuring gradient descent reaches the global minimum."
    },
    {
      "id": "log_q2",
      "question": "What is the derivative of the Sigmoid function g(z) with respect to z?",
      "options": [
        "g(z)(1 - g(z))",
        "g(z) + g(-z)",
        "1 - g(z)^2",
        "g(z)(1 + g(z))"
      ],
      "correctAnswer": "g(z)(1 - g(z))",
      "explanation": "The Sigmoid function derivative is highly elegant: d/dz [sigma(z)] = sigma(z) * (1 - sigma(z)). This simplifies neural backpropagation equations."
    },
    {
      "id": "log_q3",
      "question": "If a Logistic Regression model predicts p = 0.99 for y = 0, what is the approximate Log Loss contribution?",
      "options": [
        "Extreme loss (near infinity)",
        "0.01 loss",
        "0.99 loss",
        "0.00 loss"
      ],
      "correctAnswer": "Extreme loss (near infinity)",
      "explanation": "Log Loss penalizes confident wrong predictions. The loss is -log(1 - 0.99) = -log(0.01) = 4.6, which spikes exponentially as probability approaches the incorrect extreme."
    },
    {
      "id": "log_q4",
      "question": "What is the range of output values for the Sigmoid function?",
      "options": [
        "(0, 1)",
        "[-1, 1]",
        "(-infinity, +infinity)",
        "[0, infinity)"
      ],
      "correctAnswer": "(0, 1)",
      "explanation": "The Sigmoid function squashes any real-valued number into a strict range between 0 and 1, making it perfect for representing probabilities."
    },
    {
      "id": "log_q5",
      "question": "Logistic Regression is considered what kind of model?",
      "options": [
        "A Linear Classifier",
        "A Non-Linear Classifier",
        "A Regression Model for continuous output",
        "An Unsupervised Learning Model"
      ],
      "correctAnswer": "A Linear Classifier",
      "explanation": "Even though it applies a non-linear sigmoid activation, the decision boundary itself is determined by the linear equation w^T x + b = 0, making it a linear classifier."
    },
    {
      "id": "log_q6",
      "question": "What happens to the decision boundary if we change the classification threshold from 0.5 to 0.8?",
      "options": [
        "It shifts, requiring a stronger signal (higher z) to classify as the positive class.",
        "It remains exactly the same, but probabilities change.",
        "It becomes non-linear.",
        "The model becomes completely invalid."
      ],
      "correctAnswer": "It shifts, requiring a stronger signal (higher z) to classify as the positive class.",
      "explanation": "Increasing the threshold means the model needs to be more confident to predict class 1. This effectively shifts the decision boundary towards the region of class 1."
    },
    {
      "id": "log_q7",
      "question": "In Logistic Regression, what does the logit function represent?",
      "options": [
        "The log-odds of the positive class.",
        "The probability of the negative class.",
        "The error between predicted and actual values.",
        "The regularized loss."
      ],
      "correctAnswer": "The log-odds of the positive class.",
      "explanation": "The logit is the inverse of the sigmoid. It represents the logarithm of the odds (p / (1-p)), and it maps probabilities back to the real number line."
    },
    {
      "id": "log_q8",
      "question": "How is multiclass classification handled by Logistic Regression?",
      "options": [
        "Using the Softmax function (Multinomial Logistic Regression) or One-vs-Rest (OvR).",
        "By simply adding more bias terms.",
        "It cannot handle multiclass problems.",
        "By using Mean Squared Error instead of Log Loss."
      ],
      "correctAnswer": "Using the Softmax function (Multinomial Logistic Regression) or One-vs-Rest (OvR).",
      "explanation": "Logistic regression can be extended to multiple classes natively using the Softmax function, or by training multiple binary OvR classifiers."
    },
    {
      "id": "log_q9",
      "question": "Why do we use the logarithm in Binary Cross-Entropy (Log Loss)?",
      "options": [
        "It heavily penalizes confident but incorrect predictions.",
        "It speeds up calculation time.",
        "It limits the maximum error to 1.",
        "It prevents the model from predicting exactly 0 or 1."
      ],
      "correctAnswer": "It heavily penalizes confident but incorrect predictions.",
      "explanation": "The logarithm approaches negative infinity as the probability approaches 0. If the model confidently predicts 0 for a true label of 1, the penalty is astronomically high."
    },
    {
      "id": "log_q10",
      "question": "Which regularization techniques are commonly used with Logistic Regression?",
      "options": [
        "L1 (Lasso) and L2 (Ridge) Regularization.",
        "Dropout and Early Stopping.",
        "Batch Normalization.",
        "Max Pooling."
      ],
      "correctAnswer": "L1 (Lasso) and L2 (Ridge) Regularization.",
      "explanation": "L1 and L2 penalties are commonly added to the Log Loss to prevent weights from becoming too large, thus preventing overfitting."
    }
  ],
  "coding": {
    "tutorial": {
      "title": "Sigmoid Activation Function",
      "description": "Write a vectorized Sigmoid activation function that maps inputs to probabilities.\n\n*Note: In scikit-learn, probabilities are returned automatically using `model.predict_proba(X)`.*",
      "pseudoCode": "1. Accept a numpy array or scalar z\n2. Compute the exponential of -z\n3. Add 1 to the result\n4. Divide 1 by that sum\n5. Return the probability array",
      "starterCode": "import numpy as np\n\ndef sigmoid(z):\n    # TODO: Implement 1 / (1 + e^-z)\n    return 0.0\n\nz_val = np.array([-10.0, 0.0, 10.0])\nprint(\"Probabilities:\", np.round(sigmoid(z_val), 2))",
      "expectedOutput": "Probabilities: [0.   0.5  1.  ]",
      "solution": "import numpy as np\n\ndef sigmoid(z):\n    return 1 / (1 + np.exp(-z))\n\nz_val = np.array([-10.0, 0.0, 10.0])\nprint(\"Probabilities:\", np.round(sigmoid(z_val), 2))",
      "hints": [
        "Use np.exp() for calculating exponents.",
        "Ensure the formula maps correctly for vectors and matrices."
      ],
      "testKeywords": [
        "np.exp",
        "1 /"
      ]
    },
    "project": {
      "title": "Breast Cancer Wisconsin Detection",
      "description": "Use the Breast Cancer Wisconsin dataset features to train a LogisticRegression classifier to detect malignant vs. benign tumors. This is a classic real-world binary classification benchmark used by medical researchers.",
      "pseudoCode": "1. Define 8 features: radius_mean, texture_mean, perimeter_mean, area_mean,\n   smoothness_mean, compactness_mean, concavity_mean, symmetry_mean\n2. Target: 0=Benign, 1=Malignant\n3. Scale features with StandardScaler\n4. Fit LogisticRegression(max_iter=1000)\n5. Predict on test set and compute accuracy + F1 score",
      "starterCode": "import numpy as np\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import accuracy_score, f1_score\n\n# Breast Cancer Wisconsin Dataset (subset of real values)\n# Features: radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry\nX = np.array([\n    [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.2419],  # Malignant\n    [20.57, 17.77, 132.9, 1326.0, 0.0847, 0.0786, 0.0869, 0.1752],  # Malignant\n    [11.42, 20.38, 77.58, 386.1, 0.1425, 0.2839, 0.2414, 0.2597],   # Malignant\n    [11.29, 13.04, 73.26, 384.8, 0.1148, 0.0603, 0.0285, 0.1700],   # Benign\n    [12.45, 15.70, 82.57, 477.1, 0.1278, 0.1700, 0.1578, 0.1560],   # Benign\n    [18.25, 19.98, 119.6, 1040.0, 0.0946, 0.1099, 0.1827, 0.1803],  # Malignant\n    [13.71, 20.83, 90.20, 577.9, 0.1189, 0.1645, 0.0935, 0.1955],   # Benign\n    [13.00, 21.82, 87.50, 519.8, 0.1273, 0.1932, 0.1859, 0.2350],   # Benign\n])\ny = np.array([1, 1, 1, 0, 0, 1, 0, 0])  # 1=Malignant, 0=Benign\n\n# TODO: Split data (test_size=0.25, random_state=42)\n# TODO: Scale with StandardScaler\n# TODO: Fit LogisticRegression(max_iter=1000)\n# TODO: Predict and print accuracy\n\nX_train, X_test = X[:6], X[6:]\ny_train, y_test = y[:6], y[6:]\nprint(\"Accuracy: 0.0\")\nprint(\"F1 Score: 0.0\")",
      "expectedOutput": "Accuracy: 1.0\nF1 Score: 1.0",
      "solution": "import numpy as np\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import accuracy_score, f1_score\n\nX = np.array([\n    [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.2419],\n    [20.57, 17.77, 132.9, 1326.0, 0.0847, 0.0786, 0.0869, 0.1752],\n    [11.42, 20.38, 77.58, 386.1, 0.1425, 0.2839, 0.2414, 0.2597],\n    [11.29, 13.04, 73.26, 384.8, 0.1148, 0.0603, 0.0285, 0.1700],\n    [12.45, 15.70, 82.57, 477.1, 0.1278, 0.1700, 0.1578, 0.1560],\n    [18.25, 19.98, 119.6, 1040.0, 0.0946, 0.1099, 0.1827, 0.1803],\n    [13.71, 20.83, 90.20, 577.9, 0.1189, 0.1645, 0.0935, 0.1955],\n    [13.00, 21.82, 87.50, 519.8, 0.1273, 0.1932, 0.1859, 0.2350],\n])\ny = np.array([1, 1, 1, 0, 0, 1, 0, 0])\n\nX_train, X_test, y_train, y_test = X[:6], X[6:], y[:6], y[6:]\n\nscaler = StandardScaler()\nX_train_scaled = scaler.fit_transform(X_train)\nX_test_scaled = scaler.transform(X_test)\n\nmodel = LogisticRegression(max_iter=1000)\nmodel.fit(X_train_scaled, y_train)\npredictions = model.predict(X_test_scaled)\n\nprint(\"Accuracy:\", accuracy_score(y_test, predictions))\nprint(\"F1 Score:\", f1_score(y_test, predictions))",
      "hints": [
        "Use StandardScaler before fitting: scaler.fit_transform(X_train) and scaler.transform(X_test).",
        "LogisticRegression(max_iter=1000) avoids convergence warnings on small datasets.",
        "Use accuracy_score and f1_score from sklearn.metrics."
      ],
      "testKeywords": [
        "StandardScaler",
        "LogisticRegression",
        "fit",
        "accuracy_score"
      ]
    },
    "assignment": {
      "title": "Credit Card Default Detection — Class Imbalance Handling",
      "description": "Real-world challenge: Detect credit card defaults from financial features. Use class_weight='balanced' to handle class imbalance. Evaluate with ROC AUC score — the metric used by banks to measure classifier performance.",
      "pseudoCode": "1. Features: credit_limit, payment_ratio, num_late_payments, age, credit_utilization\n2. Target: 0=No Default, 1=Default (imbalanced — fewer defaults)\n3. Scale features with StandardScaler\n4. Fit LogisticRegression(class_weight='balanced', max_iter=1000)\n5. Predict probabilities with predict_proba[:, 1]\n6. Compute ROC AUC score",
      "starterCode": "import numpy as np\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.metrics import roc_auc_score\n\n# Credit Card Dataset (simulated from real Taiwan credit card data distributions)\n# Features: [credit_limit, payment_ratio, num_late_payments, age, utilization_rate]\nX = np.array([\n    [50000, 0.85, 0, 35, 0.12],   # Reliable customer\n    [20000, 0.60, 1, 28, 0.45],   # Somewhat risky\n    [80000, 0.95, 0, 52, 0.08],   # Very reliable\n    [15000, 0.20, 4, 22, 0.92],   # High risk — default\n    [30000, 0.10, 5, 25, 0.88],   # High risk — default\n    [60000, 0.75, 0, 44, 0.15],   # Reliable\n    [10000, 0.15, 6, 31, 0.95],   # Default\n    [45000, 0.70, 1, 38, 0.30],   # Mostly reliable\n])\ny = np.array([0, 0, 0, 1, 1, 0, 1, 0])  # 1=Default\n\n# TODO: Scale X with StandardScaler\n# TODO: Fit LogisticRegression(class_weight='balanced', max_iter=1000)\n# TODO: Get predicted probabilities for class 1 using predict_proba[:, 1]\n# TODO: Compute roc_auc_score\n\nscaler = None\nmodel = None\nproba = np.zeros(len(y))\nauc = 0.0\nprint(f\"ROC AUC: {auc:.2f}\")",
      "expectedOutput": "ROC AUC: 0.95",
      "solution": "import numpy as np\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.metrics import roc_auc_score\n\nX = np.array([\n    [50000, 0.85, 0, 35, 0.12],\n    [20000, 0.60, 1, 28, 0.45],\n    [80000, 0.95, 0, 52, 0.08],\n    [15000, 0.20, 4, 22, 0.92],\n    [30000, 0.10, 5, 25, 0.88],\n    [60000, 0.75, 0, 44, 0.15],\n    [10000, 0.15, 6, 31, 0.95],\n    [45000, 0.70, 1, 38, 0.30],\n])\ny = np.array([0, 0, 0, 1, 1, 0, 1, 0])\n\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\n\nmodel = LogisticRegression(class_weight='balanced', max_iter=1000)\nmodel.fit(X_scaled, y)\nproba = model.predict_proba(X_scaled)[:, 1]\nauc = roc_auc_score(y, proba)\nprint(f\"ROC AUC: {auc:.2f}\")",
      "hints": [
        "Use class_weight='balanced' to handle the class imbalance.",
        "predict_proba returns a 2D array — take [:, 1] for positive class probabilities.",
        "roc_auc_score(y_true, y_proba) measures the discrimination ability."
      ],
      "testKeywords": [
        "StandardScaler",
        "LogisticRegression",
        "predict_proba",
        "roc_auc_score"
      ]
    }
  },
  "interviewQuestions": [
    {
      "question": "What is Logistic Regression and why is it called regression if it's a classification algorithm?",
      "answer": "Logistic Regression is a supervised learning algorithm for binary classification. It\'s called 'regression' because its underlying technique relies on linear regression. It fits a linear model to the log-odds of the positive class, effectively 'regressing' continuous log-odds before squashing them into probabilities.",
      "companyTags": [
        "Google",
        "Microsoft"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "Explain the concept of odds ratio and log-odds. Why do we model the log-odds linearly?",
      "answer": "Probabilities are bounded between [0, 1]. The odds ratio p/(1-p) maps this to [0, +inf]. Taking the log gives Log-Odds (logit), spanning (-inf, +inf). This matches the range of linear equations (w^T x + b), allowing coefficients to represent a linear change in log-odds.",
      "companyTags": [
        "Meta",
        "Netflix"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "Why don\'t we use Mean Squared Error (MSE) as a cost function for Logistic Regression?",
      "answer": "Using MSE with the non-linear sigmoid activation creates a non-convex cost function with multiple local minima. Gradient descent is not guaranteed to find the global minimum. Log Loss (Binary Cross-Entropy) is mathematically proven to be convex for this setup.",
      "companyTags": [
        "Amazon",
        "Uber"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What is the difference between Generative and Discriminative models? Which one is Logistic Regression?",
      "answer": "Generative models (like Naive Bayes) learn the joint probability P(X, Y) and how the data was generated. Discriminative models learn the conditional probability P(Y | X) directly to draw a boundary between classes. Logistic Regression is a discriminative model.",
      "companyTags": [
        "Apple",
        "LinkedIn"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "How does Logistic Regression handle linear separability and perfect classification?",
      "answer": "If data is perfectly linearly separable, standard Logistic Regression weights will diverge to infinity because the model tries to achieve a log-loss of exactly 0 (probabilities of exactly 1 or 0). This is known as complete separation. Regularization (L1/L2) must be applied to constrain the weights.",
      "companyTags": [
        "Google",
        "Bloomberg"
      ],
      "difficulty": "Hard"
    },
    {
      "question": "What are the assumptions of Logistic Regression?",
      "answer": "1. Binary/ordinal target variable. 2. Observations are independent. 3. No multicollinearity among features. 4. Linearity of independent variables and log-odds (log(p/(1-p))). It does NOT require a linear relationship between features and the target directly, nor does it require normally distributed residuals.",
      "companyTags": [
        "Capital One",
        "Stripe"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "How can you extend Logistic Regression for multiclass classification?",
      "answer": "1. One-vs-Rest (OvR): Train K separate binary classifiers, one for each class against all others. 2. Multinomial Logistic Regression (Softmax Regression): Replace the sigmoid function with the softmax function to directly output a probability distribution across K classes.",
      "companyTags": [
        "Meta",
        "Spotify"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What is the interpretation of a coefficient in Logistic Regression?",
      "answer": "A coefficient represents the expected change in the log-odds of the outcome for a one-unit increase in the predictor variable. Equivalently, taking the exponent of the coefficient (e^w) gives the odds ratio\u2014the multiplicative factor by which the odds of the outcome increase.",
      "companyTags": [
        "Netflix",
        "Airbnb"
      ],
      "difficulty": "Hard"
    },
    {
      "question": "How do you evaluate a Logistic Regression model?",
      "answer": "Since it's a classifier, we use metrics like Accuracy, Precision, Recall, F1-Score, and the Confusion Matrix. We also evaluate the quality of probabilities using Log Loss, and measure discriminatory power using the ROC Curve and AUC (Area Under the Curve).",
      "companyTags": [
        "Amazon",
        "Twitter"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "What happens if your classes are highly imbalanced (e.g., 99% Negative, 1% Positive)?",
      "answer": "The model will likely become biased towards the majority class and achieve 99% accuracy by predicting only Negative. Solutions include: adjusting the classification threshold, using class weights (giving higher penalty for missing the 1%), or resampling techniques like SMOTE.",
      "companyTags": [
        "Uber",
        "Dropbox"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "Can Logistic Regression capture non-linear relationships?",
      "answer": "Inherently, no. It creates a linear decision boundary. However, you can manually capture non-linear relationships by engineering non-linear features (like polynomial terms x^2, x1*x2), applying the model in the expanded feature space.",
      "companyTags": [
        "Apple",
        "Pinterest"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What is L1 vs L2 regularization in the context of Logistic Regression?",
      "answer": "L2 (Ridge) shrinks all weights towards zero smoothly, preventing any single feature from dominating. L1 (Lasso) shrinks some weights exactly to zero, effectively performing feature selection. L1 is preferred when we suspect only a few features are actually important.",
      "companyTags": [
        "Google",
        "DoorDash"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "Explain the tradeoff between Precision and Recall. How do you adjust it in Logistic Regression?",
      "answer": "Precision is accuracy of positive predictions; Recall is ability to find all positive instances. They are inversely related. In Logistic Regression, you adjust the tradeoff by changing the decision threshold (default 0.5). Increasing it boosts Precision; decreasing it boosts Recall.",
      "companyTags": [
        "Meta",
        "LinkedIn"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "Why is scaling features important for Logistic Regression?",
      "answer": "If regularization is used (which is standard in libraries like scikit-learn), features must be on the same scale so the penalty applies uniformly. Additionally, feature scaling helps Gradient Descent converge much faster by making the loss landscape spherical.",
      "companyTags": [
        "Amazon",
        "Bloomberg"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "Derive the update rule for Logistic Regression using Gradient Descent.",
      "answer": "The update rule is w_j = w_j - alpha * (1/m) * sum((h(x) - y) * x_j). Interestingly, this is the exact same formula as Linear Regression, but the hypothesis h(x) here is the sigmoid function sigma(w^T x), not the linear function.",
      "companyTags": [
        "Google",
        "Netflix"
      ],
      "difficulty": "Hard"
    },
    {
      "question": "What is the ROC curve and what does AUC represent?",
      "answer": "The ROC curve plots True Positive Rate (Recall) against False Positive Rate across all possible classification thresholds. AUC (Area Under Curve) aggregates this into a single metric from 0 to 1, representing the probability that the model ranks a random positive example higher than a random negative one.",
      "companyTags": [
        "Spotify",
        "Stripe"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What is the difference between Softmax and Sigmoid?",
      "answer": "Sigmoid is used for binary classification and outputs an independent probability for one class. Softmax is a generalization of Sigmoid for multiclass classification; it takes a vector of scores and outputs a normalized probability distribution that sums to 1 across all classes.",
      "companyTags": [
        "Apple",
        "Uber"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "How do you detect overfitting in Logistic Regression?",
      "answer": "Overfitting is detected when the model has high accuracy and low loss on the training set, but poor accuracy and high loss on the validation/test set. The learned coefficients are usually extremely large in magnitude.",
      "companyTags": [
        "Amazon",
        "Meta"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "What is maximum likelihood estimation (MLE) and how does it relate to Logistic Regression?",
      "answer": "MLE is a method to estimate parameters that maximize the likelihood of observing the training data given the model. Logistic regression minimizes Log Loss, which is mathematically equivalent to maximizing the log-likelihood of the Bernoulli distribution representing the data.",
      "companyTags": [
        "Google",
        "Airbnb"
      ],
      "difficulty": "Hard"
    },
    {
      "question": "If two features are perfectly correlated, what happens to their coefficients in Logistic Regression?",
      "answer": "Without regularization, the model suffers from perfect multicollinearity; the matrix X^T X is non-invertible, and there are infinite combinations of weights for those two features that yield the same prediction. With L2 regularization, the weight will be split evenly between the two features.",
      "companyTags": [
        "Meta",
        "Capital One"
      ],
      "difficulty": "Hard"
    }
  ]
};
