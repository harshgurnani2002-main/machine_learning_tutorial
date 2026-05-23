import type { MLModule } from '../../types';

export const linearRegression: MLModule = {
  "id": "linear-regression",
  "title": "Linear Regression",
  "category": "Foundations & Math",
  "description": "Fit lines to continuous variables using Least Squares or Gradient Descent.",
  "formula": "y = w^T x + b",
  "theory": "### Linear Regression: A Comprehensive Guide\n\n#### What is it?\nLinear Regression is one of the most fundamental and widely used machine learning algorithms. It is a supervised learning algorithm used for predictive analysis, specifically for predicting a continuous target variable based on one or more input features. In its simplest form (Simple Linear Regression), it attempts to draw a straight line that best fits the relationship between a single feature and the target. When there are multiple features, it fits a hyperplane in a multi-dimensional space (Multiple Linear Regression).\n\n#### Why do we need it?\nWe need Linear Regression to model and analyze the relationship between variables. It serves two primary purposes:\n1. **Forecasting/Prediction:** Given a set of features (e.g., house size, number of bedrooms), we can predict a continuous outcome (e.g., house price).\n2. **Inference:** It helps us understand the strength and direction of the relationship between variables. For example, quantifying how much the sales of a product increase for every additional dollar spent on marketing.\nIts simplicity and interpretability make it the perfect baseline model for almost any regression problem.\n\n#### How does it work?\nLinear Regression works by assuming a linear relationship between the input features ($X$) and the target variable ($y$). The model assigns a weight (coefficient) to each feature and adds a bias term (intercept). The prediction is simply the weighted sum of the inputs plus the bias.\nTo find the \"best-fitting\" line, the algorithm minimizes a loss function, typically the Mean Squared Error (MSE), which measures the average squared difference between the predicted values and the actual target values. The minimization can be done analytically using the Normal Equation or iteratively using Gradient Descent.\n\n#### The Math Behind It\nLet $m$ be the number of training examples and $n$ be the number of features.\nLet $X \\in \\mathbb{R}^{m \times n}$ be the feature matrix and $y \\in \\mathbb{R}^m$ be the target vector.\n\n**Hypothesis Function:**\n$$h_w(x) = w_0 + w_1 x_1 + w_2 x_2 + \\dots + w_n x_n = w^T x$$\nwhere $w \\in \\mathbb{R}^{n+1}$ is the weight vector (including the bias $w_0$) and $x$ is the feature vector (with $x_0 = 1$).\n\n**Cost Function (Mean Squared Error):**\n$$J(w) = \\frac{1}{2m} \\sum_{i=1}^{m} \\left( h_w(x^{(i)}) - y^{(i)} \\right)^2$$\n\n**Parameter Optimization (Normal Equation):**\nTo find the global minimum, we can take the derivative of the cost function with respect to $w$ and set it to zero:\n$$\\nabla_w J(w) = X^T X w - X^T y = 0 \\implies w = (X^T X)^{-1} X^T y$$\n\n**Parameter Optimization (Gradient Descent):**\nFor large datasets, we iteratively update the weights to minimize the cost function:\n$$w_j \\leftarrow w_j - \u0007lpha \\frac{\\partial J}{\\partial w_j} = w_j - \\frac{\u0007lpha}{m} \\sum_{i=1}^{m} \\left( h_w(x^{(i)}) - y^{(i)} \\right) x_j^{(i)}$$\nwhere $\u0007lpha$ is the learning rate.\n\n#### Worked Example\nImagine predicting house prices ($y$ in $1000s) based on size ($x$ in 1000 sq ft).\nDataset: $(x_1, y_1) = (1, 150), (x_2, y_2) = (2, 250), (x_3, y_3) = (3, 350)$\nAssume $h_w(x) = w_1 x + w_0$.\nUsing the Normal Equation, we find $w_1 = 100$ and $w_0 = 50$.\nSo, $y = 100x + 50$.\nFor a new house of size 2.5k sq ft, the predicted price is $100(2.5) + 50 = 300$ ($300,000).\n\n#### Common Pitfalls\n1. **Outliers:** Linear regression minimizes squared errors, making it highly sensitive to outliers. A single extreme value can significantly skew the best-fit line.\n2. **Multicollinearity:** When features are highly correlated, the matrix $X^T X$ becomes nearly singular, leading to unstable coefficient estimates.\n3. **Non-linearity:** Applying linear regression to non-linear data without polynomial transformations leads to underfitting.\n4. **Heteroscedasticity:** The assumption of constant variance of residuals might be violated, making confidence intervals unreliable.\n\n#### When to Use vs Not Use\n**When to Use:**\n- The relationship between features and target is known or suspected to be linear.\n- Interpretability is a high priority (you need to explain the model to stakeholders).\n- You need a simple, fast baseline model before trying complex algorithms.\n\n**When Not to Use:**\n- The data is highly non-linear and complex.\n- There are many significant outliers that cannot be removed.\n- You are dealing with categorical targets (use Logistic Regression instead).\n\n#### Key Takeaways\n- Linear Regression is a fundamental supervised learning technique for continuous targets.\n- It finds the best-fit hyperplane by minimizing the Mean Squared Error (MSE).\n- It can be solved analytically via the Normal Equation or iteratively via Gradient Descent.\n- While simple and highly interpretable, it assumes linearity and is sensitive to outliers.\n",
  "interactiveSummary": "In this interactive simulator, you can plot data points on a 2D plane and visualize the best-fit line. Adjust the slope and intercept sliders manually to see how the Mean Squared Error (MSE) changes in real-time, or click 'Run Gradient Descent' to watch the line automatically converge to the optimal parameters. Use this to intuitively understand how the line minimizes residual distances.",
  "simulatorId": "lin-reg",
  "quiz": [
    {
      "id": "lr_q1",
      "question": "What is the closed-form algebraic solution (Normal Equation) for Linear Regression weights?",
      "options": [
        "w = (X^T X)^-1 X^T y",
        "w = X^T (X X^T)^-1 y",
        "w = (X X^T)^-1 X^T y",
        "w = (X^T X) X^T y"
      ],
      "correctAnswer": "w = (X^T X)^-1 X^T y",
      "explanation": "Minimizing the sum of squared residuals yields the Normal Equation w = (X^T X)^-1 X^T y, which is exact but computationally heavy for large feature spaces due to matrix inversion."
    },
    {
      "id": "lr_q2",
      "question": "How does high model complexity affect linear regression coefficients if regularization is not applied?",
      "options": [
        "Coefficients can become extremely large and sensitive to noise.",
        "Coefficients will always converge to exactly zero.",
        "The model bias will increase.",
        "The variance will decrease."
      ],
      "correctAnswer": "Coefficients can become extremely large and sensitive to noise.",
      "explanation": "Without regularization, unconstrained fitting on highly correlated features or polynomial mappings causes coefficients to explode in magnitude to fit training points perfectly."
    },
    {
      "id": "lr_q3",
      "question": "Which metric measures the proportion of variance in the dependent variable explained by the features?",
      "options": [
        "R-squared (Coefficient of Determination)",
        "Mean Absolute Error (MAE)",
        "Mean Squared Error (MSE)",
        "Root Mean Squared Error (RMSE)"
      ],
      "correctAnswer": "R-squared (Coefficient of Determination)",
      "explanation": "R-squared (R^2) represents the fraction of total variance explained by the regression line model compared to a simple horizontal average line."
    },
    {
      "id": "lr_q4",
      "question": "Why might Gradient Descent be preferred over the Normal Equation?",
      "options": [
        "When the number of features is very large, making matrix inversion computationally expensive.",
        "When the data is strictly linear without any noise.",
        "When we want an exact analytical solution.",
        "When there are very few features but many training examples."
      ],
      "correctAnswer": "When the number of features is very large, making matrix inversion computationally expensive.",
      "explanation": "Inverting a matrix is O(n^3). When the number of features n is in the tens of thousands, Gradient Descent is much faster than computing the inverse of (X^T X)."
    },
    {
      "id": "lr_q5",
      "question": "What is the purpose of the bias term (intercept) in Linear Regression?",
      "options": [
        "To allow the best-fit line to not pass through the origin (0,0).",
        "To prevent overfitting by reducing coefficient sizes.",
        "To increase the learning rate dynamically.",
        "To handle non-linear relationships."
      ],
      "correctAnswer": "To allow the best-fit line to not pass through the origin (0,0).",
      "explanation": "Without the bias term, the regression line is forced to pass through the origin, which often leads to a poor fit if the true relationship has a non-zero intercept."
    },
    {
      "id": "lr_q6",
      "question": "What happens if the learning rate in Gradient Descent is too large?",
      "options": [
        "The algorithm may overshoot the minimum and diverge.",
        "The algorithm will converge extremely slowly.",
        "The loss function will become zero immediately.",
        "The model will automatically apply L2 regularization."
      ],
      "correctAnswer": "The algorithm may overshoot the minimum and diverge.",
      "explanation": "A high learning rate causes the weight updates to be too large, leading the cost function to oscillate or diverge instead of converging to the minimum."
    },
    {
      "id": "lr_q7",
      "question": "What is the impact of outliers on standard Linear Regression?",
      "options": [
        "It heavily influences the best-fit line because MSE penalizes large errors quadratically.",
        "It has no impact since Linear Regression is robust to outliers.",
        "It only affects the bias term, leaving the slope unchanged.",
        "It automatically transforms the line into a curve."
      ],
      "correctAnswer": "It heavily influences the best-fit line because MSE penalizes large errors quadratically.",
      "explanation": "Because the MSE cost function squares the errors, points far away from the line (outliers) incur a massive penalty, dragging the line towards them."
    },
    {
      "id": "lr_q8",
      "question": "Which assumption is NOT required for basic Linear Regression?",
      "options": [
        "The features must be perfectly normally distributed.",
        "Linear relationship between features and target.",
        "Homoscedasticity (constant variance of residuals).",
        "Independence of errors."
      ],
      "correctAnswer": "The features must be perfectly normally distributed.",
      "explanation": "While residuals are often assumed to be normally distributed for statistical significance testing, the features themselves do not need to be normally distributed."
    },
    {
      "id": "lr_q9",
      "question": "What is multicollinearity in the context of Linear Regression?",
      "options": [
        "When two or more independent variables are highly correlated with each other.",
        "When the target variable is highly correlated with an independent variable.",
        "When the residuals are correlated with the predictions.",
        "When the features are measured on different scales."
      ],
      "correctAnswer": "When two or more independent variables are highly correlated with each other.",
      "explanation": "Multicollinearity means features are redundant, making the matrix X^T X close to singular and causing the coefficient estimates to become highly unstable."
    },
    {
      "id": "lr_q10",
      "question": "How does L1 Regularization (Lasso) affect Linear Regression differently than L2 (Ridge)?",
      "options": [
        "Lasso tends to drive some coefficients exactly to zero, performing feature selection.",
        "Lasso only shrinks coefficients but never makes them zero.",
        "Lasso adds the squared magnitude of coefficients to the loss.",
        "Lasso makes the optimization function strictly differentiable everywhere."
      ],
      "correctAnswer": "Lasso tends to drive some coefficients exactly to zero, performing feature selection.",
      "explanation": "Because of the absolute value penalty, L1 regularization often results in sparse weight vectors, effectively eliminating less important features."
    }
  ],
  "coding": {
    "tutorial": {
      "title": "Mean Squared Error Cost Function",
      "description": "Implement the Mean Squared Error (MSE) cost function J(w, b) in vectorized NumPy format.\n\n*Note: In scikit-learn, MSE is computed using `from sklearn.metrics import mean_squared_error; mse = mean_squared_error(y_true, y_pred)`*",
      "pseudoCode": "1. Get number of samples m\n2. Compute differences between y_pred and y_true\n3. Square the differences\n4. Sum them up and divide by (2 * m)\n5. Return the cost",
      "starterCode": "import numpy as np\n\ndef compute_mse(y_true, y_pred):\n    m = len(y_true)\n    # TODO: Calculate J = (1 / 2m) * sum((y_pred - y_true) ** 2)\n    cost = 0.0\n    return cost\n\ny_t = np.array([1.5, 2.0, 3.1])\ny_p = np.array([1.4, 2.2, 2.9])\nprint(\"Cost:\", compute_mse(y_t, y_p))",
      "expectedOutput": "Cost: 0.015",
      "solution": "import numpy as np\n\ndef compute_mse(y_true, y_pred):\n    m = len(y_true)\n    cost = (1 / (2 * m)) * np.sum((y_pred - y_true) ** 2)\n    return cost\n\ny_t = np.array([1.5, 2.0, 3.1])\ny_p = np.array([1.4, 2.2, 2.9])\nprint(\"Cost:\", compute_mse(y_t, y_p))",
      "hints": [
        "Subtract y_true from y_pred.",
        "Square the differences using **2.",
        "Sum all values with np.sum() and scale by 1 / (2 * m)."
      ],
      "testKeywords": [
        "np.sum",
        "2 * m"
      ]
    },
    "project": {
      "title": "California Housing Price Prediction",
      "description": "Use Scikit-Learn's `LinearRegression` on a real-world housing dataset. The California Housing dataset (subset) uses features like median income, average rooms, house age, and population per household to predict median house value. This is the standard workflow for regression in industry.",
      "pseudoCode": "1. Load features: median_income, avg_rooms, house_age, population\n2. Split into X (features) and y (house value in $100k)\n3. Scale features with StandardScaler\n4. Fit LinearRegression model\n5. Predict and compute R^2 score",
      "starterCode": "import numpy as np\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.metrics import r2_score\n\n# California Housing Dataset (subset — real feature distributions)\n# Features: [median_income, avg_rooms, house_age, population]\nX = np.array([\n    [8.3, 6.98, 41.0, 322],\n    [8.3, 7.0, 21.0, 2401],\n    [5.64, 4.94, 52.0, 496],\n    [4.58, 6.05, 52.0, 558],\n    [3.85, 4.88, 52.0, 565],\n    [6.13, 5.94, 34.0, 1200],\n    [3.12, 4.76, 42.0, 890],\n    [7.65, 6.35, 18.0, 1100],\n])\n# Target: median house value ($100k)\ny = np.array([4.526, 3.585, 3.521, 3.413, 3.422, 2.697, 2.992, 4.100])\n\n# TODO: Scale X with StandardScaler\n# TODO: Fit a LinearRegression model\n# TODO: Predict on the same X_scaled\n# TODO: Print R^2 score rounded to 2 decimal places\n\nscaler = None\nmodel = None\npredictions = np.zeros(len(y))\nprint(\"R^2 Score:\", round(r2_score(y, predictions), 2))",
      "expectedOutput": "R^2 Score: 0.89",
      "solution": "import numpy as np\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.metrics import r2_score\n\nX = np.array([\n    [8.3, 6.98, 41.0, 322],\n    [8.3, 7.0, 21.0, 2401],\n    [5.64, 4.94, 52.0, 496],\n    [4.58, 6.05, 52.0, 558],\n    [3.85, 4.88, 52.0, 565],\n    [6.13, 5.94, 34.0, 1200],\n    [3.12, 4.76, 42.0, 890],\n    [7.65, 6.35, 18.0, 1100],\n])\ny = np.array([4.526, 3.585, 3.521, 3.413, 3.422, 2.697, 2.992, 4.100])\n\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\nmodel = LinearRegression()\nmodel.fit(X_scaled, y)\npredictions = model.predict(X_scaled)\nprint(\"R^2 Score:\", round(r2_score(y, predictions), 2))",
      "hints": [
        "Use scaler = StandardScaler() and X_scaled = scaler.fit_transform(X).",
        "Fit LinearRegression with model.fit(X_scaled, y).",
        "Use r2_score(y, predictions) to evaluate performance."
      ],
      "testKeywords": [
        "StandardScaler",
        "LinearRegression",
        "fit",
        "r2_score"
      ]
    },
    "assignment": {
      "title": "Ames House Price Prediction — Full Pipeline",
      "description": "Real-world challenge: Predict house sale prices using Ames Housing data features. Build a complete sklearn pipeline with feature scaling, cross-validation, and model evaluation. Report RMSE and R² score.",
      "pseudoCode": "1. Define features: GrLivArea, OverallQual, YearBuilt, TotalBsmtSF, GarageArea\n2. Build a Pipeline with StandardScaler → Ridge(alpha=1.0)\n3. Split data using train_test_split (test_size=0.25)\n4. Fit the pipeline on training data\n5. Predict on test set\n6. Print RMSE and R² score",
      "starterCode": "import numpy as np\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import Ridge\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import mean_squared_error, r2_score\n\n# Ames Housing Dataset (subset of real features)\n# Features: [GrLivArea, OverallQual, YearBuilt, TotalBsmtSF, GarageArea]\nX = np.array([\n    [1656, 7, 2003, 856, 548],\n    [896,  6, 1976, 1262, 460],\n    [1329, 7, 2001, 920, 608],\n    [2110, 7, 1915, 756, 642],\n    [1629, 8, 2000, 1145, 836],\n    [1629, 5, 1993, 796, 480],\n    [1710, 8, 2004, 1686, 636],\n    [1040, 7, 1973, 1107, 484],\n])\n# Target: Sale Price ($1000s)\ny = np.array([208.5, 181.5, 223.5, 140.0, 250.0, 143.0, 307.0, 200.0])\n\n# TODO: Split data with train_test_split(test_size=0.25, random_state=42)\n# TODO: Build Pipeline with StandardScaler and Ridge(alpha=1.0)\n# TODO: Fit, predict, compute RMSE and R^2\n\nrmse = 0.0\nr2 = 0.0\nprint(f\"RMSE: {rmse:.1f}\")\nprint(f\"R2: {r2:.2f}\")",
      "expectedOutput": "RMSE: 28.4\nR2: 0.82",
      "solution": "import numpy as np\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import Ridge\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import mean_squared_error, r2_score\n\nX = np.array([\n    [1656, 7, 2003, 856, 548],\n    [896,  6, 1976, 1262, 460],\n    [1329, 7, 2001, 920, 608],\n    [2110, 7, 1915, 756, 642],\n    [1629, 8, 2000, 1145, 836],\n    [1629, 5, 1993, 796, 480],\n    [1710, 8, 2004, 1686, 636],\n    [1040, 7, 1973, 1107, 484],\n])\ny = np.array([208.5, 181.5, 223.5, 140.0, 250.0, 143.0, 307.0, 200.0])\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)\n\npipeline = Pipeline([\n    ('scaler', StandardScaler()),\n    ('ridge', Ridge(alpha=1.0))\n])\npipeline.fit(X_train, y_train)\npredictions = pipeline.predict(X_test)\n\nrmse = np.sqrt(mean_squared_error(y_test, predictions))\nr2 = r2_score(y_test, predictions)\nprint(f\"RMSE: {rmse:.1f}\")\nprint(f\"R2: {r2:.2f}\")",
      "hints": [
        "Use train_test_split(X, y, test_size=0.25, random_state=42).",
        "Pipeline steps: [('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))].",
        "RMSE = np.sqrt(mean_squared_error(y_test, predictions))."
      ],
      "testKeywords": [
        "Pipeline",
        "StandardScaler",
        "Ridge",
        "train_test_split",
        "mean_squared_error"
      ]
    }
  },
  "interviewQuestions": [
    {
      "question": "What is the difference between Gradient Descent and the Normal Equation?",
      "answer": "The Normal Equation is an analytical solution that computes weights directly via matrix inversion O(n^3). Gradient Descent is an iterative optimization algorithm that updates weights step-by-step. Gradient Descent is preferred for very large feature sets (n > 10,000) where matrix inversion becomes intractable.",
      "companyTags": [
        "Google",
        "Amazon"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "How do you handle categorical variables in Linear Regression?",
      "answer": "Categorical variables must be converted to numerical values using techniques like One-Hot Encoding (for nominal data) or Ordinal Encoding (for ordinal data). To avoid the 'dummy variable trap' (perfect multicollinearity), one category is often dropped if an intercept term is included.",
      "companyTags": [
        "Meta",
        "Uber"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "Explain the assumptions of Linear Regression.",
      "answer": "1. Linearity: linear relationship between X and Y. 2. Independence of errors. 3. Homoscedasticity: constant variance of errors. 4. No perfect multicollinearity. 5. Normality of residuals (required for confidence intervals, not strictly for fitting).",
      "companyTags": [
        "Netflix",
        "LinkedIn"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What is multicollinearity, why is it a problem, and how do you fix it?",
      "answer": "Multicollinearity occurs when independent variables are highly correlated. It makes coefficient estimates highly sensitive to small changes in the model and inflates standard errors. It can be fixed by removing highly correlated features, using PCA, or applying Ridge regularization.",
      "companyTags": [
        "Apple",
        "Airbnb"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "How do outliers affect Linear Regression and how do you handle them?",
      "answer": "Linear Regression uses MSE, which squares errors, making it highly sensitive to outliers. Outliers can be handled by removing them, transforming the data (e.g., log transformation), or using robust regression techniques like Huber Loss or RANSAC.",
      "companyTags": [
        "Amazon",
        "Spotify"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What is the difference between R-squared and Adjusted R-squared?",
      "answer": "R-squared measures the proportion of variance explained by the model, but it always increases when a new feature is added, even if it's noise. Adjusted R-squared penalizes the addition of non-useful features, providing a more accurate measure of model quality for multiple regression.",
      "companyTags": [
        "Google",
        "Dropbox"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "Under what conditions does the Normal Equation fail or become mathematically unstable?",
      "answer": "The Normal Equation fails when X^T X is non-invertible (singular). This occurs if features are perfectly correlated (multicollinearity) or if the number of features d exceeds the number of samples m. This can be resolved using pseudo-inverse or Ridge regularization.",
      "companyTags": [
        "Google",
        "Amazon"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "What is Heteroscedasticity?",
      "answer": "Heteroscedasticity refers to the non-constant variance of residuals. It violates the homoscedasticity assumption, meaning the spread of errors changes across the range of predictions. It doesn\'t bias the coefficients, but it makes standard errors and p-values unreliable. It can be detected using residual plots.",
      "companyTags": [
        "Stripe",
        "Bloomberg"
      ],
      "difficulty": "Hard"
    },
    {
      "question": "Why do we scale features before applying Gradient Descent in Linear Regression?",
      "answer": "Feature scaling (e.g., standardization) ensures all features have a similar scale. This makes the cost function's contour more spherical rather than elliptical, allowing Gradient Descent to converge much faster and preventing it from oscillating.",
      "companyTags": [
        "Meta",
        "DoorDash"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "Can Linear Regression model non-linear relationships?",
      "answer": "Yes, standard Linear Regression can model non-linear relationships by applying feature transformations. For example, Polynomial Regression simply creates new features (e.g., x^2, x^3) and applies Linear Regression to them. The model remains 'linear in the parameters'.",
      "companyTags": [
        "Apple",
        "Netflix"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What is the difference between Ridge and Lasso Regression?",
      "answer": "Ridge (L2) adds a penalty equal to the sum of squared weights, shrinking coefficients towards zero but rarely making them exactly zero. Lasso (L1) adds a penalty equal to the sum of absolute weights, which can shrink coefficients exactly to zero, effectively performing feature selection.",
      "companyTags": [
        "Amazon",
        "Meta"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "How would you assess the goodness of fit for a Linear Regression model?",
      "answer": "Common metrics include R-squared, Adjusted R-squared, Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE). Residual analysis (plotting residuals vs. predictions) is also crucial to verify assumptions.",
      "companyTags": [
        "LinkedIn",
        "Twitter"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "Why is Mean Absolute Error (MAE) more robust to outliers than Mean Squared Error (MSE)?",
      "answer": "MSE squares the error terms, meaning a large error generated by an outlier gets amplified exponentially. MAE takes the absolute difference, treating all errors proportionally, making it less sensitive to extreme values.",
      "companyTags": [
        "Spotify",
        "Uber"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "Explain the Bias-Variance tradeoff in the context of Linear Regression.",
      "answer": "A simple linear model (few features) has high bias (underfitting) and low variance. Adding more features or polynomial terms reduces bias but increases variance (overfitting). Regularization techniques (Ridge/Lasso) add a penalty to reduce variance at the cost of introducing some bias.",
      "companyTags": [
        "Google",
        "Apple"
      ],
      "difficulty": "Hard"
    },
    {
      "question": "If your R-squared is very high on training data but low on test data, what is happening?",
      "answer": "The model is overfitting. It has memorized the training data\'s noise rather than learning the underlying signal. This usually happens when the model is too complex (e.g., high-degree polynomial) or has too many features compared to the number of samples.",
      "companyTags": [
        "Netflix",
        "Airbnb"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "How do you interpret a coefficient in Multiple Linear Regression?",
      "answer": "A coefficient represents the expected change in the target variable for a one-unit increase in its corresponding feature, holding all other features constant (ceteris paribus).",
      "companyTags": [
        "Bloomberg",
        "Stripe"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What is the 'Dummy Variable Trap'?",
      "answer": "It is a scenario where independent variables are multicollinear due to one-hot encoding categorical variables. If a category has 'n' levels, creating 'n' dummy variables causes perfect multicollinearity. The solution is to create 'n-1' dummy variables.",
      "companyTags": [
        "Amazon",
        "Capital One"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "Can you use Linear Regression for classification?",
      "answer": "While possible, it is not recommended. Linear regression predicts continuous values, which can fall outside the [0, 1] probability range. It is also highly sensitive to outliers, which can easily shift the decision boundary. Logistic Regression is better suited for classification.",
      "companyTags": [
        "Google",
        "Meta"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What does a negative R-squared value mean?",
      "answer": "A negative R-squared indicates that the model fits the data worse than a simple horizontal line that constantly predicts the mean of the target. This usually happens if the model is forced through an incorrect intercept or is completely inappropriate for the data.",
      "companyTags": [
        "Uber",
        "Pinterest"
      ],
      "difficulty": "Hard"
    },
    {
      "question": "What are interaction terms in Linear Regression?",
      "answer": "Interaction terms are created by multiplying two or more features together (e.g., x1 * x2). They are used when the effect of one feature on the target variable depends on the value of another feature. For example, the effect of house age on price might depend on the neighborhood.",
      "companyTags": [
        "Meta",
        "Apple"
      ],
      "difficulty": "Medium"
    }
  ]
};
