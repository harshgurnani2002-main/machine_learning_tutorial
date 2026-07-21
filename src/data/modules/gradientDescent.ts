import type { MLModule } from '../../types';

export const gradientDescent: MLModule = {
  "id": "gradient-descent",
  "title": "Gradient Descent & Optimizers",
  "category": "Foundations & Math",
  "description": "Navigate loss landscapes iteratively using derivatives, momentum, and adaptive scales.",
  "formula": "w_{t+1} = w_t - \\eta \nabla L(w_t)",
  "theory": `### What is Gradient Descent?

Gradient Descent is a fundamental, first-order iterative optimization algorithm used universally across machine learning and deep learning to find the local or global minimum of a differentiable function. In the context of machine learning, this function is the loss (or cost) function, which measures the discrepancy between the model\'s predictions and the actual target values. By iteratively adjusting the model\'s parameters (weights and biases) in the direction of the steepest descent—given by the negative of the gradient—the algorithm systematically reduces the prediction error until it converges to an optimal or near-optimal state.

### Why do we need it?

When training complex models, especially those with high dimensionality like deep neural networks, finding an analytical or closed-form solution (such as the Normal Equation for Linear Regression) is computationally prohibitive or mathematically intractable. Neural networks can have billions of parameters, and the cost function is highly non-convex. Gradient descent provides a scalable, computationally efficient, and highly versatile numerical approach. It relies purely on the first derivatives of the function, making it memory-efficient and adaptable to massive datasets via stochastic approximations (e.g., Stochastic Gradient Descent), solving optimization problems where traditional matrix inversions would fail or take centuries to compute.

### How does it work?

Imagine you are blindfolded near the peak of a rugged mountain and your goal is to reach the lowest point in the valley. You cannot see the global landscape, so you probe the ground beneath your feet to determine the steepest downward slope. You take a step of a specific size in that direction. You repeat this process—probing, stepping, probing, stepping—until the ground levels out, indicating you have reached the bottom (a minimum).

In mathematical optimization terms, the algorithm follows these precise steps:
1. **Initialization:** Start with random values for the parameters (weights $w$ and biases $b$).
2. **Forward Pass:** Compute the model\'s predictions using the current parameters and evaluate the overall loss $L$.
3. **Backward Pass (Gradient Computation):** Calculate the gradient of the loss function with respect to each parameter, $\nabla L(w)$. This gradient vector points in the direction of steepest ascent.
4. **Parameter Update:** Move the parameters in the opposite direction of the gradient, scaled by a hyperparameter called the learning rate ($\\eta$).
5. **Iteration:** Repeat steps 2-4 for multiple epochs or until the gradients approach zero and the loss plateaus (convergence).

### The Math Behind It

#### 1. Vanilla Gradient Descent Update Rule
The fundamental update rule for a parameter vector $w$ at iteration $t$ with learning rate $\\eta$ and loss $L(w)$ is:
$$w_{t+1} = w_t - \\eta \nabla L(w_t)$$
The learning rate $\\eta$ controls the step size. If $\\eta$ is too large, the updates can overshoot the minimum; if too small, convergence is agonizingly slow.

#### 2. Momentum (Accelerated Gradient)
In regions of the loss landscape that resemble long, narrow ravines, vanilla gradient descent oscillates inefficiently across the ravine walls. Momentum introduces a velocity term $v_t$ that acts like physical momentum, accumulating the gradient of past steps with a decay factor $\beta$ (typically 0.9):
$$v_t = \beta v_{t-1} + \\eta \nabla L(w_t)$$
$$w_{t+1} = w_t - v_t$$
This dampens perpendicular oscillations and accelerates progress along the ravine floor.

#### 3. Adaptive Moment Estimation (Adam)
Adam is a sophisticated optimizer that combines the advantages of AdaGrad and RMSProp. It computes individual adaptive learning rates for different parameters by maintaining moving averages of both the gradients (first moment, $m_t$) and the squared gradients (second moment, $v_t$).

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Because $m_0$ and $v_0$ are initialized to zeros, they are biased toward zero during the initial steps. Adam applies a bias correction:
$$\\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \\quad \\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Finally, the parameter update utilizes these corrected moments:
$$w_{t+1} = w_t - \frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t$$
where $\\epsilon$ is a small smoothing term (e.g., $10^{-8}$) to prevent division by zero.

### Worked Example

Consider a simple quadratic loss function representing a single weight $w$: $L(w) = w^2$. 
The derivative (gradient) is $\nabla L(w) = 2w$.
Let's initialize $w_0 = 10$ and set the learning rate $\\eta = 0.1$.

- **Iteration 1:**
  - Gradient $g_0 = 2 \times 10 = 20$
  - Update: $w_1 = 10 - 0.1 \times 20 = 10 - 2 = 8$
- **Iteration 2:**
  - Gradient $g_1 = 2 \times 8 = 16$
  - Update: $w_2 = 8 - 0.1 \times 16 = 8 - 1.6 = 6.4$
- **Iteration 3:**
  - Gradient $g_2 = 2 \times 6.4 = 12.8$
  - Update: $w_3 = 6.4 - 0.1 \times 12.8 = 6.4 - 1.28 = 5.12$
- **Iteration 4:**
  - Gradient $g_3 = 2 \times 5.12 = 10.24$
  - Update: $w_4 = 5.12 - 0.1 \times 10.24 = 4.096$

With each iteration, the gradient decreases, and the step size naturally shrinks as $w$ approaches the optimal minimum at $w = 0$.

### Common Pitfalls

1. **Learning Rate Sensitivity:** Choosing the wrong learning rate is catastrophic. Too high leads to divergence (loss exploding to infinity), while too low leads to agonizingly slow training. Learning rate schedulers are often required.
2. **Vanishing and Exploding Gradients:** In deep architectures (like RNNs), repeated multiplication of gradients can cause them to exponentially shrink to zero (vanishing) or grow infinitely (exploding), paralyzing the update process.
3. **Saddle Points:** In highly multi-dimensional spaces, points where the gradient is zero are rarely local minima; they are usually saddle points (minimum in some dimensions, maximum in others). Standard SGD can stagnate here, making momentum essential.
4. **Poor Initialization:** Starting weights at zero or highly symmetric values can cause neurons to learn identical features, preventing the network from breaking symmetry.
5. **Feature Scaling:** Unscaled inputs lead to elongated, elliptical loss contours, causing gradient descent to oscillate heavily. Normalizing features (e.g., standard scaler) makes the contours spherical, drastically speeding up convergence.

### When to Use vs Not Use

**When to Use:**
- When the objective function is smooth and differentiable.
- In almost all Deep Learning architectures (CNNs, RNNs, Transformers).
- For large-scale datasets where batch processing is impossible; Stochastic Gradient Descent (SGD) or Mini-batch GD is indispensable here.
- When memory is constrained, as GD scales $O(N)$ with parameters.

**When Not to Use:**
- When a fast, exact analytical solution exists (e.g., Ordinary Least Squares for Linear Regression on small datasets).
- When the problem is discrete or the loss function is highly non-differentiable (requires evolutionary algorithms or decision trees).
- For very small datasets where second-order methods (like L-BFGS or Newton-Raphson) can converge much faster by computing the Hessian matrix.

### Key Takeaways
- Gradient descent iteratively minimizes a loss function by taking steps proportional to the negative gradient.
- The Learning Rate ($\\eta$) dictates step size and is the most critical hyperparameter to tune.
- Mini-batch GD balances the noisy, rapid updates of SGD with the stable, slow updates of Batch GD.
- Advanced optimizers (Momentum, RMSProp, Adam) solve the inherent flaws of standard SGD by dampening oscillations and independently adapting learning rates for each parameter.
#### Python Implementation

\`\`\`python
import numpy as np

X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([2, 4, 6])
lr = 0.01
w = np.zeros(2)
for _ in range(100):
    grad = (2 / len(X)) * X.T @ (X @ w - y)
    w -= lr * grad
print(f"Weights: {w}")
\`\`\`
`,
  "interactiveSummary": "In this interactive simulator, you can visualize the loss landscape and observe how changing the learning rate, momentum, and optimizer type (SGD vs Adam) affects the trajectory of the weights towards the global minimum. Experiment with different starting points to see how optimizers handle local minima and saddle points.",
  "simulatorId": "gradient-descent",
  "quiz": [
    {
      "id": "gd_q1",
      "question": "What is the primary purpose of Gradient Descent in Machine Learning?",
      "options": [
        "To maximize the objective function",
        "To minimize the loss function by iteratively updating parameters",
        "To reduce the dimensionality of the dataset",
        "To augment the training data"
      ],
      "correctAnswer": "To minimize the loss function by iteratively updating parameters",
      "explanation": "Gradient Descent is an optimization algorithm used to minimize the loss (or cost) function by iteratively moving in the direction of steepest descent."
    },
    {
      "id": "gd_q2",
      "question": "What happens if the learning rate in Gradient Descent is set too high?",
      "options": [
        "The model converges very slowly",
        "The model finds the global minimum immediately",
        "The algorithm may overshoot the minimum and diverge",
        "The gradients become exactly zero in the first step"
      ],
      "correctAnswer": "The algorithm may overshoot the minimum and diverge",
      "explanation": "A high learning rate causes the updates to be too large, which can lead to overshooting the minimum, oscillating, or diverging completely."
    },
    {
      "id": "gd_q3",
      "question": "Which of the following describes Stochastic Gradient Descent (SGD)?",
      "options": [
        "It uses the entire dataset to compute the gradient at each step",
        "It uses a single randomly chosen training example to compute the gradient at each step",
        "It uses an analytical formula to find the exact minimum",
        "It doesn\'t use gradients for optimization"
      ],
      "correctAnswer": "It uses a single randomly chosen training example to compute the gradient at each step",
      "explanation": "Stochastic Gradient Descent approximates the true gradient by calculating the gradient on a single training example per iteration."
    },
    {
      "id": "gd_q4",
      "question": "Why does adding Momentum help gradient descent optimization?",
      "options": [
        "It dampens oscillations perpendicular to the valley path while accelerating updates along the valley.",
        "It changes the learning rate dynamically based on gradient size.",
        "It makes the loss function convex.",
        "It guarantees finding the global minimum."
      ],
      "correctAnswer": "It dampens oscillations perpendicular to the valley path while accelerating updates along the valley.",
      "explanation": "Momentum accumulates a velocity vector in directions of persistent gradients and cancels out oscillations, guiding updates down the center of the valley."
    },
    {
      "id": "gd_q5",
      "question": "What is the function of the epsilon term in adaptive optimizers like RMSprop and Adam?",
      "options": [
        "To scale up learning rates.",
        "To prevent division by zero when the rolling second-moment gradient approaches zero.",
        "To calculate gradients.",
        "To control momentum decay."
      ],
      "correctAnswer": "To prevent division by zero when the rolling second-moment gradient approaches zero.",
      "explanation": "The epsilon constant (typically 1e-8) is added to the denominator before division, preventing math errors when features have flat gradients."
    },
    {
      "id": "gd_q6",
      "question": "Which optimizer utilizes rolling averages of squared gradients without using momentum?",
      "options": [
        "RMSprop",
        "SGD",
        "AdaGrad",
        "Adam"
      ],
      "correctAnswer": "RMSprop",
      "explanation": "RMSprop addresses AdaGrad's decaying learning rate by using an exponentially decaying average of squared gradients, stabilizing updates."
    },
    {
      "id": "gd_q7",
      "question": "In Adam optimizer, what is the purpose of bias correction?",
      "options": [
        "To increase the learning rate over time",
        "To prevent the moments from being biased towards zero in early iterations",
        "To add noise to the gradients to escape local minima",
        "To regularize the model parameters"
      ],
      "correctAnswer": "To prevent the moments from being biased towards zero in early iterations",
      "explanation": "Since the moving averages are initialized to 0, they are biased toward 0 initially. Bias correction counteracts this, especially in early steps."
    },
    {
      "id": "gd_q8",
      "question": "What is a saddle point?",
      "options": [
        "A point where the gradient is zero and it is a minimum in all directions",
        "A point where the gradient is zero, but it is a minimum in one direction and a maximum in another",
        "A point where the gradient is undefined",
        "The absolute lowest point of a loss function"
      ],
      "correctAnswer": "A point where the gradient is zero, but it is a minimum in one direction and a maximum in another",
      "explanation": "At a saddle point, the gradient is zero, but the surface curves upwards in some dimensions and downwards in others, resembling a horse's saddle."
    },
    {
      "id": "gd_q9",
      "question": "What is Mini-batch Gradient Descent?",
      "options": [
        "Updating parameters using the entire dataset",
        "Updating parameters using a single sample",
        "Updating parameters using a small, randomly sampled subset of the data",
        "Updating parameters without using any data"
      ],
      "correctAnswer": "Updating parameters using a small, randomly sampled subset of the data",
      "explanation": "Mini-batch GD strikes a balance between Batch GD and SGD by using a subset (e.g., 32 or 64 samples) to compute the gradient, allowing for vectorized operations."
    },
    {
      "id": "gd_q10",
      "question": "How do you know when Gradient Descent has converged?",
      "options": [
        "The learning rate becomes zero",
        "The parameters become zero",
        "The gradients become very close to zero, meaning the loss function is no longer decreasing significantly",
        "After exactly 100 epochs"
      ],
      "correctAnswer": "The gradients become very close to zero, meaning the loss function is no longer decreasing significantly",
      "explanation": "Convergence occurs when the gradient approaches zero, indicating that the algorithm has reached a local or global minimum and updates are negligible."
    }
  ],
  "coding": {
    "tutorial": {
      "title": "Vanilla Gradient Update",
      "description": "Update parameter w using gradient g and learning rate eta.\n\n*Scikit-Learn Equivalent:* `from sklearn.linear_model import SGDRegressor`\nSGDRegressor applies this update iteratively over the dataset.",
      "pseudoCode": "function van_update(w, g, eta):\n    return w - eta * g",
      "starterCode": "import numpy as np\n\ndef van_update(w, g, eta):\n    # TODO: Update w\n    return w\n\nprint(\"Updated:\", van_update(1.5, 0.2, 0.1))",
      "expectedOutput": "Updated: 1.48",
      "solution": "import numpy as np\n\ndef van_update(w, g, eta):\n    return w - eta * g\n\nprint(\"Updated:\", van_update(1.5, 0.2, 0.1))",
      "hints": [
        "Subtract learning rate eta * gradient g from w."
      ],
      "testKeywords": [
        "w -",
        "eta * g"
      ]
    },
    "project": {
      "title": "Stochastic Gradient Descent Classifier",
      "description": "Use sklearn's SGDClassifier to train a linear classifier on a toy dataset. We will use a pipeline with StandardScaler.",
      "starterCode": "import numpy as np\nfrom sklearn.linear_model import SGDClassifier\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.metrics import accuracy_score\n\n# Sample Dataset\nX = np.array([[1, 2], [1, 4], [5, 2], [5, 4], [2, 1], [3, 1]])\ny = np.array([0, 0, 1, 1, 0, 0])\n\n# TODO: Create a pipeline with StandardScaler and SGDClassifier\n# TODO: Fit the model and predict on X\nclf = None\npredictions = []\nacc = 0.0\n\nprint(\"Predictions:\", predictions)\nprint(\"Accuracy:\", acc)",
      "expectedOutput": "Predictions: [0 0 1 1 0 0]\nAccuracy: 1.0",
      "solution": "import numpy as np\nfrom sklearn.linear_model import SGDClassifier\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.metrics import accuracy_score\n\nX = np.array([[1, 2], [1, 4], [5, 2], [5, 4], [2, 1], [3, 1]])\ny = np.array([0, 0, 1, 1, 0, 0])\n\nclf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, random_state=42))\nclf.fit(X, y)\npredictions = clf.predict(X)\nacc = accuracy_score(y, predictions)\n\nprint(\"Predictions:\", predictions)\nprint(\"Accuracy:\", acc)",
      "hints": [
        "Use make_pipeline(StandardScaler(), SGDClassifier(random_state=42))",
        "Call clf.fit(X, y)",
        "Use accuracy_score(y, predictions)"
      ],
      "testKeywords": [
        "make_pipeline",
        "StandardScaler",
        "SGDClassifier",
        "fit",
        "predict"
      ]
    },
    "assignment": {
      "title": "SGD Regressor with Cross-Validation",
      "description": "Implement a real-world scenario using `SGDRegressor`. Use `cross_val_score` to evaluate the model on a small regression dataset.",
      "starterCode": "import numpy as np\nfrom sklearn.linear_model import SGDRegressor\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.model_selection import cross_val_score\n\n# Toy Dataset (e.g., predicting house prices based on size and rooms)\nX = np.array([[100, 2], [150, 3], [200, 3], [250, 4], [300, 4], [50, 1], [80, 2], [120, 2]])\ny = np.array([150, 220, 280, 350, 400, 80, 130, 180])\n\n# TODO: Create a pipeline with StandardScaler and SGDRegressor(random_state=42)\n# TODO: Use cross_val_score with cv=3 and scoring='r2'\nscores = []\n\nprint(\"Mean R2:\", np.round(np.mean(scores), 2))",
      "expectedOutput": "Mean R2: 0.81",
      "solution": "import numpy as np\nfrom sklearn.linear_model import SGDRegressor\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.model_selection import cross_val_score\n\nX = np.array([[100, 2], [150, 3], [200, 3], [250, 4], [300, 4], [50, 1], [80, 2], [120, 2]])\ny = np.array([150, 220, 280, 350, 400, 80, 130, 180])\n\nmodel = make_pipeline(StandardScaler(), SGDRegressor(random_state=42))\nscores = cross_val_score(model, X, y, cv=3, scoring='r2')\n\nprint(\"Mean R2:\", np.round(np.mean(scores), 2))",
      "hints": [
        "Create model using make_pipeline with StandardScaler and SGDRegressor",
        "Call cross_val_score(model, X, y, cv=3, scoring='r2')"
      ],
      "testKeywords": [
        "cross_val_score",
        "SGDRegressor",
        "StandardScaler"
      ]
    }
  },
  "interviewQuestions": [
    {
      "question": "What is Gradient Descent and how does it work?",
      "answer": "Gradient descent is an iterative optimization algorithm for finding the minimum of a function. It works by calculating the gradient of the loss function with respect to its parameters, and moving the parameters in the opposite direction of the gradient, scaled by a learning rate.",
      "companyTags": [
        "Amazon",
        "Google"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "What is the difference between Batch GD, Stochastic GD (SGD), and Mini-batch GD?",
      "answer": "Batch GD computes the gradient using the entire dataset. SGD computes it using a single sample per step, making it noisy but fast. Mini-batch GD uses a small random subset (e.g., 32 samples) per step, balancing computational efficiency and convergence stability.",
      "companyTags": [
        "Meta",
        "Microsoft"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What happens if the learning rate is too large or too small?",
      "answer": "If too small, the convergence will be extremely slow and it may get stuck in local minima. If too large, the algorithm might overshoot the minimum, oscillating endlessly or even diverging.",
      "companyTags": [
        "Apple",
        "Netflix"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "How does Momentum help Gradient Descent?",
      "answer": "Momentum accumulates past gradients to determine the current direction, acting like a ball rolling down a hill. It speeds up convergence in flat regions and dampens oscillations in narrow ravines.",
      "companyTags": [
        "Google",
        "Meta"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What is the purpose of Adam optimizer?",
      "answer": "Adam (Adaptive Moment Estimation) combines the benefits of Momentum and RMSProp. It uses moving averages of both the gradients and their squared values to adapt the learning rate for each parameter individually.",
      "companyTags": [
        "Amazon",
        "OpenAI"
      ],
      "difficulty": "Hard"
    },
    {
      "question": "Why do we need bias correction in Adam optimizer?",
      "answer": "Because the moving averages of the first and second moments are initialized to zero, they are heavily biased towards zero in the early steps of training. Bias correction scales them up to counteract this initial drag.",
      "companyTags": [
        "Meta",
        "DeepMind"
      ],
      "difficulty": "Hard"
    },
    {
      "question": "What are vanishing and exploding gradients?",
      "answer": "In deep networks, gradients can multiply through many layers. If they are <1, they shrink exponentially (vanish), halting learning in early layers. If >1, they grow exponentially (explode), causing numeric instability.",
      "companyTags": [
        "Google",
        "Microsoft"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "How do you deal with exploding gradients?",
      "answer": "Exploding gradients can be managed using gradient clipping (capping the gradient norm), careful weight initialization (e.g., He or Xavier), and using architectures like ResNets or LSTMs.",
      "companyTags": [
        "Apple",
        "Amazon"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "Can Gradient Descent guarantee finding the global minimum?",
      "answer": "Only for strictly convex functions (like linear regression or logistic regression). For non-convex functions (like deep neural networks), it only guarantees finding a local minimum or a saddle point.",
      "companyTags": [
        "Netflix",
        "Meta"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What is a saddle point, and why is it a problem?",
      "answer": "A saddle point is a point where the gradient is zero, but it is a minimum along one axis and a maximum along another. In high-dimensional spaces, saddle points are more common than local minima and can slow down training, though SGD and Momentum usually escape them.",
      "companyTags": [
        "Google",
        "OpenAI"
      ],
      "difficulty": "Hard"
    },
    {
      "question": "What is AdaGrad, and what is its main limitation?",
      "answer": "AdaGrad scales the learning rate inversely proportional to the square root of the sum of all historical squared gradients. Its limitation is that the accumulated sum keeps growing, causing the learning rate to monotonically decrease to zero, eventually stopping learning.",
      "companyTags": [
        "Microsoft",
        "Amazon"
      ],
      "difficulty": "Hard"
    },
    {
      "question": "How does RMSProp improve upon AdaGrad?",
      "answer": "RMSProp replaces AdaGrad's accumulating sum of squared gradients with an exponentially decaying moving average. This prevents the learning rate from decaying to zero and allows the model to continue learning.",
      "companyTags": [
        "Google",
        "Meta"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "Why is the epsilon parameter used in Adam and RMSProp?",
      "answer": "Epsilon is a very small number (e.g., 1e-8) added to the denominator of the learning rate scaling factor to prevent division by zero when the moving average of squared gradients is exactly zero.",
      "companyTags": [
        "Apple",
        "Amazon"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "How does Nesterov Accelerated Gradient (NAG) differ from standard Momentum?",
      "answer": "Standard Momentum computes the gradient at the current position and adds the momentum step. NAG first takes a step in the direction of the accumulated momentum, then computes the gradient at that 'lookahead' position, which provides a more accurate update.",
      "companyTags": [
        "DeepMind",
        "Google"
      ],
      "difficulty": "Hard"
    },
    {
      "question": "What are learning rate schedules?",
      "answer": "Learning rate schedules intentionally decay or change the learning rate during training. Common strategies include step decay, exponential decay, and cosine annealing. This allows fast early learning and fine-tuning later.",
      "companyTags": [
        "Netflix",
        "Meta"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What is Weight Decay?",
      "answer": "Weight decay is a regularization technique (equivalent to L2 regularization for standard SGD) that adds a penalty proportional to the magnitude of the weights to the loss function, encouraging the model to use smaller weights and reducing overfitting.",
      "companyTags": [
        "Amazon",
        "Microsoft"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "Does Weight Decay behave the same as L2 regularization in Adam?",
      "answer": "No, in adaptive optimizers like Adam, L2 regularization gets scaled by the adaptive learning rates, leading to sub-optimal regularization. AdamW was introduced to decouple weight decay from the gradient updates.",
      "companyTags": [
        "OpenAI",
        "Google"
      ],
      "difficulty": "Hard"
    },
    {
      "question": "How do you choose between SGD and Adam?",
      "answer": "Adam often converges faster and requires less learning rate tuning, making it great for prototyping. However, SGD with Momentum often generalizes better and achieves a lower final loss if the learning rate schedule is carefully tuned.",
      "companyTags": [
        "Meta",
        "Apple"
      ],
      "difficulty": "Hard"
    },
    {
      "question": "What is Gradient Accumulation?",
      "answer": "When memory constraints prevent using a large batch size, gradient accumulation computes the gradients on small micro-batches and accumulates them over several steps before updating the weights, simulating a larger batch size.",
      "companyTags": [
        "Amazon",
        "Microsoft"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "Why do neural networks often use random initialization instead of initializing with zeros?",
      "answer": "If all weights are initialized to zero, all neurons in a layer compute the same output and receive the same gradient during backpropagation. Random initialization breaks this symmetry, allowing different neurons to learn different features.",
      "companyTags": [
        "Google",
        "Netflix"
      ],
      "difficulty": "Medium"
    }
  ]
};
