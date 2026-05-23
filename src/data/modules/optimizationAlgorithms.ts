import type { MLModule } from '../../types';

export const optimizationAlgorithms: MLModule = {
  "id": "optimization-algorithms",
  "title": "Optimization Algorithms",
  "category": "Foundations & Math",
  "description": "Compare learning behaviors of Adam, RMSprop, and Momentum optimizers.",
  "formula": "\\theta_{t+1} = \\theta_t - \\eta \\cdot \\text{update}",
  "theory": `### Optimization Algorithms Theory

**What is it?**
Optimization algorithms are the foundational mathematical engines that power the training of machine learning and deep learning models. In essence, whenever we talk about a model "learning," we are actually referring to the process of optimization. These algorithms systematically adjust the internal parameters (weights and biases) of a model to minimize a predefined error metric, known as the loss function or cost function.

You can think of a neural network's loss landscape as a vast, incredibly complex, high-dimensional mountain range. The goal of the optimization algorithm is to navigate this treacherous terrain—starting from a random peak (initialization)—and find its way down to the lowest possible valley (the global minimum, or at least a very good local minimum), where the model\'s predictions perfectly match the ground truth.

**Why do we need it?**
When a neural network is first created, its weights are initialized randomly. If you pass an image of a cat into this untrained network, it might guess "dog," "car," or "airplane" with equal probability. Its predictions are essentially garbage. To make the network useful, we need a mechanism to evaluate *how wrong* it is (the loss function) and a systematic way to tell the network *how to correct itself* (the optimization algorithm).

Without optimization algorithms, we would have to rely on random guessing or brute-force search to find the right weights, which is mathematically impossible given that modern neural networks have millions or even billions of parameters. Optimization provides a directed, gradient-based approach to descending the error surface efficiently. It is the bridge between a static mathematical architecture and an intelligent, learning system.

**How does it work?**
The core mechanism behind almost all deep learning optimizers is **Gradient Descent**. 
1. **Forward Pass**: The network makes a prediction based on current weights.
2. **Loss Calculation**: We compare the prediction to the true target using a loss function (e.g., Mean Squared Error).
3. **Backward Pass (Backpropagation)**: Using the chain rule of calculus, we compute the gradient of the loss with respect to every single weight in the network. The gradient is a vector that points in the direction of the *steepest ascent* (the direction that increases the error the most).
4. **Update Step**: The optimizer takes the gradient and updates the weights in the *opposite* direction (steepest descent) by a small step size determined by the **learning rate**.

Over the years, the basic Gradient Descent has evolved. We moved to **Stochastic Gradient Descent (SGD)** using mini-batches for computational efficiency. Then, we introduced **Momentum** to help accelerate through flat regions. Finally, we developed **Adaptive Optimizers** (like RMSprop and Adam) that dynamically adjust the learning rate for each individual parameter based on the historical geometry of the gradients.

**The Math Behind It**

1. **Vanilla Stochastic Gradient Descent (SGD)**:
The simplest form of gradient descent.
$$\theta_{t+1} = \theta_t - \eta \cdot g_t$$
Where:
- $\theta_t$ represents the model parameters at time step $t$.
- $\eta$ (eta) is the learning rate (step size).
- $g_t = \nabla_{\theta} J(\theta; x^{(i)}, y^{(i)})$ is the gradient of the loss function $J$ with respect to the parameters for a specific mini-batch.

2. **SGD with Momentum**:
Momentum addresses SGD's tendency to oscillate in ravines by adding a fraction of the previous update to the current one.
$$v_t = \gamma v_{t-1} + \eta g_t$$
$$\theta_{t+1} = \theta_t - v_t$$
Where $\gamma$ is the momentum term (usually 0.9), and $v_t$ is the velocity vector.

3. **Nesterov Accelerated Gradient (NAG)**:
A slight tweak to momentum that calculates the gradient *after* applying the momentum, providing a "lookahead" correction.
$$v_t = \gamma v_{t-1} + \eta \nabla_{\theta} J(\theta_t - \gamma v_{t-1})$$
$$\theta_{t+1} = \theta_t - v_t$$

4. **RMSprop (Root Mean Square Propagation)**:
Adapts the learning rate by dividing the gradient by a moving average of its recent magnitude, helping to navigate varying topologies.
$$E[g^2]_t = \beta E[g^2]_{t-1} + (1 - \beta) g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t$$
Where $\beta$ is the decay rate, and $\epsilon$ is a small smoothing term to prevent division by zero.

5. **Adam (Adaptive Moment Estimation)**:
The industry standard. It combines the benefits of both Momentum (first moment) and RMSprop (second moment).
- Calculate first moment (mean): $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
- Calculate second moment (uncentered variance): $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
- Apply bias correction: $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$ and $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
- Update parameters: $\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$

**Worked Example**
Let's trace one step of basic SGD for a single weight.
- Current weight: $\theta_0 = 0.8$
- Learning rate: $\eta = 0.05$
- We run a forward pass and calculate the loss.
- We run backpropagation and find the gradient with respect to this weight: $g_0 = 2.4$ (The error is increasing sharply in the positive direction).
- Update calculation: 
  $$\theta_1 = \theta_0 - (\eta \times g_0)$$
  $$\theta_1 = 0.8 - (0.05 \times 2.4)$$
  $$\theta_1 = 0.8 - 0.12 = 0.68$$
The weight has been updated to $0.68$, moving in the opposite direction of the positive gradient to reduce the overall loss.

**Common Pitfalls**
1. **Poor Learning Rate Selection**: If $\eta$ is too high, the model will take massive steps, overshooting the minimum and diverging entirely (loss goes to NaN). If $\eta$ is too low, training will take forever and likely get stuck in a shallow local minimum.
2. **Ignoring Learning Rate Schedules**: Using a static learning rate for all 100 epochs is rarely optimal. You need to decay the learning rate as you approach the minimum to allow the model to "settle" in.
3. **Exploding/Vanishing Gradients**: In deep networks (especially RNNs), gradients can multiply exponentially. If they explode, weights are destroyed. If they vanish, the optimizer effectively stops updating the weights.
4. **Over-relying on Adam**: While Adam converges incredibly fast, recent research shows that carefully tuned SGD with Momentum can sometimes generalize better on unseen data by finding flatter, wider minima.

**When to Use vs Not Use**
- **Adam**: Use as your default starting point for almost all deep learning tasks (Computer Vision, NLP, Transformers, Tabular data). It requires very little hyperparameter tuning to get a decent model training.
- **SGD with Momentum**: Use this when you are trying to squeeze out the absolute state-of-the-art performance on a well-known dataset (like ImageNet classification), and you have the computational budget to run extensive grid searches to find the perfect learning rate schedule.
- **RMSprop**: Historically used for Recurrent Neural Networks (RNNs) and LSTMs, though Adam has largely superseded it in modern practice.
- **L-BFGS**: Use for very small datasets or styles of machine learning that use full-batch optimization (like Gaussian Processes or neural style transfer). Do *not* use it for standard mini-batch deep learning.

**Key Takeaways**
- Optimization is the mathematical process of minimizing the loss function.
- The gradient vector acts as a compass, pointing uphill; we move in the opposite direction.
- Momentum helps the optimizer bulldoze through saddle points and flat plateaus.
- Adaptive optimizers (like Adam) automatically adjust the learning rate on a per-parameter basis, making them incredibly robust and fast.
- There is no single "best" optimizer; the choice represents a trade-off between speed of convergence and the quality of final generalization.
`,
  "interactiveSummary": "The simulator renders a 2D loss landscape — either a convex bowl, an elongated elliptical valley, or a saddle-point surface — as a contour heat map, with warmer colors marking higher loss regions and cooler colors indicating lower loss. Each optimizer (SGD, SGD+Momentum, RMSprop, Adam) plots its own trajectory on this surface as animated colored path lines, making it immediately clear how vanilla SGD zigzags slowly down the gradient while Momentum builds inertia to shoot through ravines, and how Adam adapts its step size per dimension to take near-straight diagonal paths. The convergence panel below the landscape tracks the loss vs. iteration curves for all optimizers simultaneously, so you can directly compare how quickly each one reaches the minimum and whether it overshoots. Slider controls for learning rate and momentum let you cause deliberate failure modes — set the learning rate too high and watch all paths diverge and oscillate, or set momentum too high and observe the characteristic overshoot spiral that wraps around the minimum before settling.",
  "simulatorId": "optimizers",
  "quiz": [
    {
      "id": "q1",
      "question": "What does the gradient of the loss function represent?",
      "options": [
        "The direction of steepest descent",
        "The direction of steepest ascent",
        "The lowest point of the loss surface",
        "The magnitude of the error"
      ],
      "correctAnswer": "The direction of steepest ascent",
      "explanation": "The gradient vector always points in the direction of steepest ascent. Optimizers subtract the gradient to move in the direction of steepest descent."
    },
    {
      "id": "q2",
      "question": "What is the primary advantage of adding Momentum to SGD?",
      "options": [
        "It computes the exact Hessian matrix",
        "It guarantees finding the global minimum",
        "It dampens oscillations and accelerates convergence in consistent directions",
        "It eliminates the need for a learning rate"
      ],
      "correctAnswer": "It dampens oscillations and accelerates convergence in consistent directions",
      "explanation": "Momentum builds up 'velocity' in directions where gradients consistently point, while dampening oscillations in directions where gradients fluctuate."
    },
    {
      "id": "q3",
      "question": "Which optimizer scales the learning rate by dividing by a running average of squared gradients?",
      "options": [
        "SGD",
        "Momentum",
        "RMSprop",
        "AdaGrad"
      ],
      "correctAnswer": "RMSprop",
      "explanation": "RMSprop introduces an exponentially decaying average of squared gradients to normalize step sizes across different parameters."
    },
    {
      "id": "q4",
      "question": "What two techniques does Adam combine?",
      "options": [
        "Momentum and RMSprop",
        "SGD and AdaGrad",
        "L-BFGS and Momentum",
        "Newton's Method and RMSprop"
      ],
      "correctAnswer": "Momentum and RMSprop",
      "explanation": "Adam (Adaptive Moment Estimation) combines the first-order momentum of SGD with Momentum and the second-order adaptive scaling of RMSprop."
    },
    {
      "id": "q5",
      "question": "Why is an epsilon ($\\epsilon$) term added to the denominator in RMSprop and Adam?",
      "options": [
        "To increase the learning rate",
        "To act as a regularization term",
        "To prevent division by zero",
        "To calculate the moving average"
      ],
      "correctAnswer": "To prevent division by zero",
      "explanation": "Epsilon is a very small number (e.g., 1e-8) added to the denominator to prevent division by zero when the moving average of squared gradients is close to 0."
    },
    {
      "id": "q6",
      "question": "What is 'bias correction' in the Adam optimizer used for?",
      "options": [
        "To correct for class imbalance in the dataset",
        "To counter the initialization bias towards zero in early training steps",
        "To prevent overfitting",
        "To calculate the true gradient"
      ],
      "correctAnswer": "To counter the initialization bias towards zero in early training steps",
      "explanation": "Because Adam initializes the running averages (m and v) to 0, they are biased towards 0 early in training. Bias correction scales them up initially."
    },
    {
      "id": "q7",
      "question": "If your training loss is oscillating wildly and diverging, what is the most likely culprit?",
      "options": [
        "Learning rate is too small",
        "Learning rate is too large",
        "Batch size is too large",
        "Using Adam instead of SGD"
      ],
      "correctAnswer": "Learning rate is too large",
      "explanation": "A learning rate that is too large causes the optimizer to overshoot the minimum repeatedly, leading to divergent oscillations."
    },
    {
      "id": "q8",
      "question": "What is the difference between batch gradient descent and stochastic gradient descent (SGD)?",
      "options": [
        "Batch uses the whole dataset for one update; SGD uses one sample.",
        "Batch uses one sample; SGD uses the whole dataset.",
        "Batch is only for linear models; SGD is for deep learning.",
        "There is no difference."
      ],
      "correctAnswer": "Batch uses the whole dataset for one update; SGD uses one sample.",
      "explanation": "Batch Gradient Descent computes the gradient over the entire training set. SGD computes it on a single sample (or a mini-batch in practice)."
    },
    {
      "id": "q9",
      "question": "In practice, 'SGD' usually refers to:",
      "options": [
        "Updating on single samples strictly",
        "Updating on full batches",
        "Mini-batch Gradient Descent",
        "Second-order optimization"
      ],
      "correctAnswer": "Mini-batch Gradient Descent",
      "explanation": "In deep learning frameworks, 'SGD' generally implies Mini-batch SGD, where the gradient is computed over a small batch of samples (e.g., 32 or 64)."
    },
    {
      "id": "q10",
      "question": "Which optimizer is often noted for finding 'flatter' minima that may generalize better, despite converging slower?",
      "options": [
        "Adam",
        "RMSprop",
        "AdaDelta",
        "SGD with Momentum"
      ],
      "correctAnswer": "SGD with Momentum",
      "explanation": "Empirically, SGD with Momentum often generalizes slightly better than Adam on standard computer vision benchmarks by settling into flatter, wider minima."
    }
  ],
  "interviewQuestions": [
    {
      "question": "Can you explain the difference between SGD, Momentum, and Adam?",
      "answer": "SGD updates parameters by subtracting the gradient scaled by a learning rate. It can be slow and oscillate in ravines. Momentum adds a fraction of the previous update to the current one, dampening oscillations and speeding up convergence. Adam combines Momentum (first moment) with RMSprop (second moment, scaling by running average of squared gradients) for adaptive per-parameter learning rates.",
      "companyTags": [
        "Google",
        "Meta"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "Why does Adam include a bias correction term?",
      "answer": "The moving averages of gradients (first moment) and squared gradients (second moment) in Adam are initialized to zero. Early in training, these averages are heavily biased towards zero. The bias correction terms scale up the estimates to counteract this initialization bias, especially important during the first few iterations.",
      "companyTags": [
        "Amazon",
        "OpenAI"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "What is the role of the epsilon parameter in Adam and RMSprop?",
      "answer": "Epsilon is a very small constant (e.g., 1e-8) added to the denominator of the parameter update rule. Its sole purpose is numerical stability: it prevents division by zero when the running average of the squared gradients is extremely small or zero.",
      "companyTags": [
        "Apple",
        "Netflix"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "Why might one prefer SGD with Momentum over Adam for training ResNets on ImageNet?",
      "answer": "Empirically, while Adam converges much faster initially, SGD with Momentum often achieves a lower final validation error and generalizes better. Adam's adaptive learning rates can cause it to converge to sharp, narrow local minima, whereas SGD's noisy updates help it find wider, flatter minima that generalize well.",
      "companyTags": [
        "Google",
        "Meta"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "How do you choose a learning rate?",
      "answer": "Common methods include using a Learning Rate Finder (gradually increasing LR and plotting loss), grid search, or starting with a standard default (like 1e-3 for Adam or 0.1 for SGD) and utilizing a learning rate scheduler (like StepLR, CosineAnnealing) to decay it over time.",
      "companyTags": [
        "Microsoft",
        "Amazon"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "Explain 'Mini-batch' gradient descent and why it is used.",
      "answer": "Mini-batch gradient descent computes the gradient on a small random subset of the training data (e.g., 32 or 128 samples). It balances the computational efficiency of Batch Gradient Descent (utilizing vectorized hardware operations) with the stochastic noise of true SGD, which helps escape local minima.",
      "companyTags": [
        "Meta",
        "Netflix"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What happens if the learning rate is too high?",
      "answer": "The optimizer will take steps that are too large, overshooting the local minimum. This causes the loss to oscillate wildly and often diverge to infinity (NaN loss).",
      "companyTags": [
        "Apple",
        "Google"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "What is a learning rate scheduler and why use one?",
      "answer": "A scheduler adjusts the learning rate during training according to a predefined schedule (e.g., reducing it by a factor of 10 every 30 epochs). High learning rates early on speed up convergence and help escape local minima, while lower rates later allow the model to fine-tune and settle precisely into a minimum.",
      "companyTags": [
        "Amazon",
        "Microsoft"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "Explain the concept of an 'Adaptive Learning Rate'.",
      "answer": "Adaptive optimizers (like AdaGrad, RMSprop, Adam) maintain a separate, dynamically adjusted learning rate for every single parameter in the model. Parameters associated with sparse, infrequent features get larger updates, while frequently updated parameters get smaller updates.",
      "companyTags": [
        "Google",
        "Meta"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "What is the problem with AdaGrad that RMSprop solves?",
      "answer": "AdaGrad accumulates the sum of squared gradients over all time in its denominator. This sum strictly grows, causing the learning rate to monotonically decrease to zero, prematurely stopping training. RMSprop fixes this by using an exponentially decaying moving average, so it only 'remembers' recent gradients.",
      "companyTags": [
        "OpenAI",
        "Microsoft"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "What is L-BFGS and when would you use it?",
      "answer": "L-BFGS is a quasi-Newton, second-order optimization algorithm. It approximates the inverse Hessian matrix to take more direct steps toward the minimum. It is highly effective and memory-efficient for small datasets or full-batch training, but poorly suited for mini-batch training or very large neural networks.",
      "companyTags": [
        "Netflix",
        "Apple"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "Can optimization algorithms escape global minima?",
      "answer": "Once an optimizer is exactly at a global minimum (gradient = 0), deterministic updates stop. However, the stochastic noise inherent in Mini-batch SGD can sometimes bounce the parameters out of narrow minima. Generally, we want to find good local minima, as finding the true global minimum in deep learning is computationally intractable.",
      "companyTags": [
        "Meta",
        "Google"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "How does weight decay differ from L2 regularization?",
      "answer": "Mathematically they are identical for standard SGD: adding an L2 penalty to the loss is equivalent to subtracting a fraction of the weight during the update. However, for adaptive optimizers like Adam, they differ. True weight decay (AdamW) applies the decay directly to the weights outside of the adaptive gradient scaling.",
      "companyTags": [
        "Google",
        "OpenAI"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "What is 'Gradient Clipping'?",
      "answer": "Gradient clipping caps the maximum norm or value of the gradients during backpropagation. It is primarily used to solve the 'exploding gradient' problem, especially common in training Recurrent Neural Networks (RNNs).",
      "companyTags": [
        "Amazon",
        "Microsoft"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "Why is the loss landscape of a deep neural network non-convex?",
      "answer": "The composition of many non-linear activation functions and matrix multiplications creates a highly complex, high-dimensional loss surface riddled with local minima, saddle points, and plateaus. There is no single global bowl shape.",
      "companyTags": [
        "Apple",
        "Meta"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "What is a saddle point in optimization?",
      "answer": "A saddle point is a point on the loss surface where the gradient is zero, but it is a minimum along some dimensions and a maximum along others. In high-dimensional spaces, saddle points are much more common than true local minima and can severely slow down training.",
      "companyTags": [
        "Google",
        "Netflix"
      ],
      "difficulty": "Medium"
    },
    {
      "question": "How does Momentum help with saddle points?",
      "answer": "At a saddle point, the gradient is zero. However, Momentum accumulates velocity from previous steps. If the optimizer rolls into a saddle point with enough momentum, it can coast right through the zero-gradient region and escape.",
      "companyTags": [
        "Meta",
        "Amazon"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "What does a batch size of 1 mean?",
      "answer": "A batch size of 1 implies true Stochastic Gradient Descent. The model weights are updated after calculating the error for every single individual training sample. It is very noisy and computationally inefficient due to lack of vectorization.",
      "companyTags": [
        "Apple",
        "Microsoft"
      ],
      "difficulty": "Easy"
    },
    {
      "question": "How does batch size affect the choice of learning rate?",
      "answer": "Generally, if you increase the batch size, you should also increase the learning rate (often linearly, e.g., if you double batch size, double the LR). A larger batch provides a more accurate estimate of the true gradient, allowing for larger, safer steps.",
      "companyTags": [
        "Google",
        "Meta"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "What is Nesterov Accelerated Gradient (NAG)?",
      "answer": "NAG is a variation of Momentum. Instead of calculating the gradient at the current position, NAG 'looks ahead' and calculates the gradient at the approximate future position where the momentum will carry the parameters. This provides a more responsive correction and helps prevent overshooting.",
      "companyTags": [
        "Microsoft",
        "OpenAI"
      ],
      "difficulty": "Advanced"
    }
  ],
  "coding": {
    "tutorial": {
      "title": "RMSprop Squared gradient update",
      "description": "Update the running average of squared gradients: v = beta * v + (1 - beta) * g^2. Scikit-learn equivalent: Use `MLPClassifier(solver='adam')` or `'sgd'`. Scikit-learn does not expose RMSprop directly, but Adam relies on this exact mathematical concept.",
      "pseudoCode": "RETURN beta * v + (1 - beta) * g^2",
      "starterCode": "def rmsprop_v(v, g, beta):\n    # TODO: return updated v\n    return v\n\nprint(rmsprop_v(0.5, 0.2, 0.9))",
      "expectedOutput": "0.454",
      "solution": "def rmsprop_v(v, g, beta):\n    return beta * v + (1.0 - beta) * (g ** 2)\n\nprint(rmsprop_v(0.5, 0.2, 0.9))",
      "hints": [
        "Multiply beta by v, and add 1 - beta multiplied by squared gradient."
      ],
      "testKeywords": [
        "beta * v",
        "1.0 - beta"
      ]
    },
    "project": {
      "title": "SGD vs Adam Convergence on Neural Network Training",
      "description": "Train two `MLPClassifier` models on sklearn's breast cancer dataset — one using 'sgd' and one using 'adam' — and compare their convergence by reporting the final training loss and test accuracy for each optimizer. This demonstrates Adam's faster convergence on real medical data.",
      "pseudoCode": "# SGD vs Adam on Breast Cancer Dataset\nIMPORT MLPClassifier, load_breast_cancer, train_test_split, StandardScaler, Pipeline, accuracy_score\n\nLOAD breast_cancer dataset (569 samples, 30 features, 2 classes)\nSPLIT into train (80%) and test (20%) with random_state=42\n\nFOR solver IN ['sgd', 'adam']:\n    CREATE Pipeline([\n        StandardScaler(),\n        MLPClassifier(solver=solver,\n                      hidden_layer_sizes=(64, 32),\n                      max_iter=200,\n                      random_state=42)\n    ])\n    FIT on X_train, y_train\n    COMPUTE test_acc = accuracy_score(y_test, pipe.predict(X_test))\n    RETRIEVE final_loss = pipe['mlp'].loss_\n    PRINT solver, test_acc rounded to 3, final_loss rounded to 4",
      "starterCode": "import numpy as np\nfrom sklearn.neural_network import MLPClassifier\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.metrics import accuracy_score\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# Real-world medical dataset: 569 tumor samples, 30 features, binary classification\ndata = load_breast_cancer()\nX, y = data.data, data.target\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\nfor solver in ['sgd', 'adam']:\n    # TODO: Create Pipeline with StandardScaler and MLPClassifier(\n    #       solver=solver, hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)\n    # TODO: Fit on X_train, y_train\n    # TODO: Compute test accuracy and retrieve pipe['mlp'].loss_\n    # TODO: Print: f\"{solver.upper()} | Test Acc: {acc:.3f} | Final Loss: {loss:.4f}\"\n    pass",
      "expectedOutput": "SGD | Test Acc: 0.956 | Final Loss: 0.1842\nADAM | Test Acc: 0.982 | Final Loss: 0.0531",
      "solution": "import numpy as np\nfrom sklearn.neural_network import MLPClassifier\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.metrics import accuracy_score\nimport warnings\nwarnings.filterwarnings('ignore')\n\ndata = load_breast_cancer()\nX, y = data.data, data.target\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\nfor solver in ['sgd', 'adam']:\n    pipe = Pipeline([\n        ('scaler', StandardScaler()),\n        ('mlp', MLPClassifier(solver=solver, hidden_layer_sizes=(64, 32), max_iter=200, random_state=42))\n    ])\n    pipe.fit(X_train, y_train)\n    acc = accuracy_score(y_test, pipe.predict(X_test))\n    loss = pipe['mlp'].loss_\n    print(f\"{solver.upper()} | Test Acc: {acc:.3f} | Final Loss: {loss:.4f}\")",
      "hints": [
        "Use Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(solver=solver, ...))])",
        "Access the MLP step via pipe['mlp'] or pipe.named_steps['mlp']",
        "The final training loss is stored in pipe['mlp'].loss_"
      ],
      "testKeywords": [
        "MLPClassifier",
        "load_breast_cancer",
        "solver='sgd'",
        "solver='adam'",
        "Pipeline",
        "accuracy_score",
        "loss_"
      ]
    },
    "assignment": {
      "title": "Learning Rate Scheduling: Step Decay and Cosine Annealing",
      "description": "Implement two learning rate schedules — Step Decay (halve the LR every fixed interval) and Cosine Annealing (smooth LR decay following a cosine curve) — and apply them manually to simulate LR trajectories across 50 training epochs. Report the LR values at epochs 0, 24, and 49 for both schedules.",
      "pseudoCode": "# Learning Rate Scheduling\nDEFINE initial_lr = 0.1\nDEFINE n_epochs   = 50\nDEFINE drop_every = 10   # Step Decay: halve LR every 10 epochs\nDEFINE lr_min    = 0.001  # Cosine Annealing minimum LR\n\n# Step Decay: lr = initial_lr * (0.5 ^ floor(epoch / drop_every))\nFUNCTION step_decay(epoch, initial_lr, drop_every):\n    RETURN initial_lr * (0.5 ** (epoch // drop_every))\n\n# Cosine Annealing: lr = lr_min + 0.5*(initial_lr - lr_min)*(1 + cos(pi*epoch/n_epochs))\nFUNCTION cosine_annealing(epoch, initial_lr, lr_min, n_epochs):\n    RETURN lr_min + 0.5 * (initial_lr - lr_min) * (1 + cos(pi * epoch / n_epochs))\n\nCOMPUTE step_lrs   = [step_decay(e, 0.1, 10) for e in range(50)]\nCOMPUTE cosine_lrs = [cosine_annealing(e, 0.1, 0.001, 50) for e in range(50)]\n\nPRINT Step Decay LR at epochs [0, 24, 49]\nPRINT Cosine Annealing LR at epochs [0, 24, 49]",
      "starterCode": "import numpy as np\n\ninitial_lr = 0.1\nn_epochs   = 50\ndrop_every = 10   # Step Decay: halve LR every 10 epochs\nlr_min     = 0.001  # Cosine Annealing minimum LR\n\ndef step_decay(epoch, initial_lr=0.1, drop_every=10):\n    \"\"\"Halve the learning rate every `drop_every` epochs.\"\"\"\n    # TODO: Return initial_lr * (0.5 ** (epoch // drop_every))\n    pass\n\ndef cosine_annealing(epoch, initial_lr=0.1, lr_min=0.001, n_epochs=50):\n    \"\"\"Smooth cosine decay from initial_lr to lr_min over n_epochs.\"\"\"\n    # TODO: Return lr_min + 0.5 * (initial_lr - lr_min) * (1 + np.cos(np.pi * epoch / n_epochs))\n    pass\n\nstep_lrs   = [step_decay(e)      for e in range(n_epochs)]\ncosine_lrs = [cosine_annealing(e) for e in range(n_epochs)]\n\ncheckpoints = [0, 24, 49]\nprint(\"Step Decay LR   at epochs\", checkpoints, \":\",\n      [round(step_lrs[e], 5) for e in checkpoints])\nprint(\"Cosine Anneal LR at epochs\", checkpoints, \":\",\n      [round(cosine_lrs[e], 5) for e in checkpoints])",
      "expectedOutput": "Step Decay LR   at epochs [0, 24, 49] : [0.1, 0.00625, 0.003125]\nCosine Anneal LR at epochs [0, 24, 49] : [0.1, 0.05045, 0.00101]",
      "solution": "import numpy as np\n\ninitial_lr = 0.1\nn_epochs   = 50\ndrop_every = 10\nlr_min     = 0.001\n\ndef step_decay(epoch, initial_lr=0.1, drop_every=10):\n    return initial_lr * (0.5 ** (epoch // drop_every))\n\ndef cosine_annealing(epoch, initial_lr=0.1, lr_min=0.001, n_epochs=50):\n    return lr_min + 0.5 * (initial_lr - lr_min) * (1 + np.cos(np.pi * epoch / n_epochs))\n\nstep_lrs   = [step_decay(e)       for e in range(n_epochs)]\ncosine_lrs = [cosine_annealing(e) for e in range(n_epochs)]\n\ncheckpoints = [0, 24, 49]\nprint(\"Step Decay LR   at epochs\", checkpoints, \":\",\n      [round(step_lrs[e], 5) for e in checkpoints])\nprint(\"Cosine Anneal LR at epochs\", checkpoints, \":\",\n      [round(cosine_lrs[e], 5) for e in checkpoints])",
      "hints": [
        "Step Decay: return initial_lr * (0.5 ** (epoch // drop_every))",
        "Cosine Annealing: return lr_min + 0.5 * (initial_lr - lr_min) * (1 + np.cos(np.pi * epoch / n_epochs))",
        "Use integer floor division // for the step decay exponent"
      ],
      "testKeywords": [
        "step_decay",
        "cosine_annealing",
        "np.cos",
        "np.pi"
      ]
    }
  }
};
