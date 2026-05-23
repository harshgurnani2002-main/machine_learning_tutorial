import json

data = {
    "id": "optimization-algorithms",
    "title": "Optimization Algorithms",
    "category": "Foundations & Math",
    "description": "Compare learning behaviors of Adam, RMSprop, and Momentum optimizers.",
    "formula": "\\theta_{t+1} = \\theta_t - \\eta \\cdot \\text{update}",
    "theory": r"""### Optimization Algorithms Theory

**What is it?**
Optimization algorithms are the engines that power machine learning and deep learning training. Their primary goal is to minimize a loss function (like Mean Squared Error or Cross-Entropy) by iteratively updating the model's parameters (weights and biases).

**Why do we need it?**
When we initialize a neural network, its predictions are basically random. We need a systematic way to tell the network how to adjust its internal parameters so that its predictions get closer to the ground truth. Optimization algorithms provide this systematic, mathematically sound approach to descending the error surface towards a minimum.

**How does it work?**
Optimizers calculate the gradient of the loss function with respect to the model's parameters using a technique called backpropagation. The gradient tells us the direction of the steepest ascent of the loss surface. By taking a step in the opposite direction (descent), we can reduce the loss. Advanced optimizers modify this basic step using concepts like momentum (building up speed in consistent directions) and adaptive learning rates (adjusting step sizes for each parameter individually).

**The Math Behind It**

1. **Stochastic Gradient Descent (SGD)**: The baseline optimizer.
   $$\theta_{t+1} = \theta_t - \eta g_t$$
   Where $\theta$ is the parameter, $\eta$ is the learning rate, and $g_t$ is the gradient at time $t$.

2. **SGD with Momentum**: Adds a fraction of the previous update to the current one to accelerate learning and dampen oscillations.
   $$v_t = \beta v_{t-1} + (1 - \beta) g_t$$
   $$\theta_{t+1} = \theta_t - \eta v_t$$

3. **RMSprop (Root Mean Square Propagation)**: Adapts the learning rate for each parameter by dividing the gradient by a running average of its recent magnitude.
   $$s_t = \beta s_{t-1} + (1 - \beta) g_t^2$$
   $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t} + \epsilon} g_t$$

4. **Adam (Adaptive Moment Estimation)**: Combines the benefits of Momentum (first moment) and RMSprop (second moment).
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
   Bias correction: $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$, $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
   $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**Worked Example**
Let's trace one step of basic SGD.
- Current weight $\theta_0 = 0.5$
- Learning rate $\eta = 0.1$
- Calculated gradient $g_0 = 2.0$
- Update: $\theta_1 = 0.5 - (0.1 \times 2.0) = 0.5 - 0.2 = 0.3$
The weight moves in the opposite direction of the gradient to reduce the loss.

**Common Pitfalls**
- **Learning Rate Too High**: The optimizer might overshoot the minimum and diverge, causing loss to explode.
- **Learning Rate Too Low**: The optimizer will take tiny steps, leading to very slow convergence and potentially getting stuck in local minima.
- **Dying Gradients**: If gradients become extremely small, parameters barely update.

**When to Use vs Not Use**
- **Adam**: Use as the default for most deep learning tasks, especially NLP and Computer Vision. It's robust and fast.
- **SGD with Momentum**: Often used for fine-tuning or when you want the absolute best generalization on standard datasets (like ImageNet). It requires more careful tuning of the learning rate.
- **RMSprop**: Useful in Recurrent Neural Networks (RNNs).

**Key Takeaways**
- Optimizers are crucial for navigating the loss landscape.
- Adaptive optimizers like Adam generally require less tuning of the learning rate compared to standard SGD.
- The choice of optimizer affects both the speed of convergence and the final generalization ability of the model.""",
    "interactiveSummary": "This interactive simulator allows you to visualize a 2D loss landscape (like a bowl or saddle point) and trace the path taken by SGD, Momentum, and Adam. You can tweak learning rates and momentum parameters to see how they influence convergence speed and oscillations.",
    "simulatorId": "optimizers",
    "quiz": [
        {"id": "q1", "question": "What does the gradient of the loss function represent?", "options": ["The direction of steepest descent", "The direction of steepest ascent", "The lowest point of the loss surface", "The magnitude of the error"], "correctAnswer": "The direction of steepest ascent", "explanation": "The gradient vector always points in the direction of steepest ascent. Optimizers subtract the gradient to move in the direction of steepest descent."},
        {"id": "q2", "question": "What is the primary advantage of adding Momentum to SGD?", "options": ["It computes the exact Hessian matrix", "It guarantees finding the global minimum", "It dampens oscillations and accelerates convergence in consistent directions", "It eliminates the need for a learning rate"], "correctAnswer": "It dampens oscillations and accelerates convergence in consistent directions", "explanation": "Momentum builds up 'velocity' in directions where gradients consistently point, while dampening oscillations in directions where gradients fluctuate."},
        {"id": "q3", "question": "Which optimizer scales the learning rate by dividing by a running average of squared gradients?", "options": ["SGD", "Momentum", "RMSprop", "AdaGrad"], "correctAnswer": "RMSprop", "explanation": "RMSprop introduces an exponentially decaying average of squared gradients to normalize step sizes across different parameters."},
        {"id": "q4", "question": "What two techniques does Adam combine?", "options": ["Momentum and RMSprop", "SGD and AdaGrad", "L-BFGS and Momentum", "Newton's Method and RMSprop"], "correctAnswer": "Momentum and RMSprop", "explanation": "Adam (Adaptive Moment Estimation) combines the first-order momentum of SGD with Momentum and the second-order adaptive scaling of RMSprop."},
        {"id": "q5", "question": "Why is an epsilon ($\epsilon$) term added to the denominator in RMSprop and Adam?", "options": ["To increase the learning rate", "To act as a regularization term", "To prevent division by zero", "To calculate the moving average"], "correctAnswer": "To prevent division by zero", "explanation": "Epsilon is a very small number (e.g., 1e-8) added to the denominator to prevent division by zero when the moving average of squared gradients is close to 0."},
        {"id": "q6", "question": "What is 'bias correction' in the Adam optimizer used for?", "options": ["To correct for class imbalance in the dataset", "To counter the initialization bias towards zero in early training steps", "To prevent overfitting", "To calculate the true gradient"], "correctAnswer": "To counter the initialization bias towards zero in early training steps", "explanation": "Because Adam initializes the running averages (m and v) to 0, they are biased towards 0 early in training. Bias correction scales them up initially."},
        {"id": "q7", "question": "If your training loss is oscillating wildly and diverging, what is the most likely culprit?", "options": ["Learning rate is too small", "Learning rate is too large", "Batch size is too large", "Using Adam instead of SGD"], "correctAnswer": "Learning rate is too large", "explanation": "A learning rate that is too large causes the optimizer to overshoot the minimum repeatedly, leading to divergent oscillations."},
        {"id": "q8", "question": "What is the difference between batch gradient descent and stochastic gradient descent (SGD)?", "options": ["Batch uses the whole dataset for one update; SGD uses one sample.", "Batch uses one sample; SGD uses the whole dataset.", "Batch is only for linear models; SGD is for deep learning.", "There is no difference."], "correctAnswer": "Batch uses the whole dataset for one update; SGD uses one sample.", "explanation": "Batch Gradient Descent computes the gradient over the entire training set. SGD computes it on a single sample (or a mini-batch in practice)."},
        {"id": "q9", "question": "In practice, 'SGD' usually refers to:", "options": ["Updating on single samples strictly", "Updating on full batches", "Mini-batch Gradient Descent", "Second-order optimization"], "correctAnswer": "Mini-batch Gradient Descent", "explanation": "In deep learning frameworks, 'SGD' generally implies Mini-batch SGD, where the gradient is computed over a small batch of samples (e.g., 32 or 64)."},
        {"id": "q10", "question": "Which optimizer is often noted for finding 'flatter' minima that may generalize better, despite converging slower?", "options": ["Adam", "RMSprop", "AdaDelta", "SGD with Momentum"], "correctAnswer": "SGD with Momentum", "explanation": "Empirically, SGD with Momentum often generalizes slightly better than Adam on standard computer vision benchmarks by settling into flatter, wider minima."}
    ],
    "interviewQuestions": [
        {"question": "Can you explain the difference between SGD, Momentum, and Adam?", "answer": "SGD updates parameters by subtracting the gradient scaled by a learning rate. It can be slow and oscillate in ravines. Momentum adds a fraction of the previous update to the current one, dampening oscillations and speeding up convergence. Adam combines Momentum (first moment) with RMSprop (second moment, scaling by running average of squared gradients) for adaptive per-parameter learning rates.", "companyTags": ["Google", "Meta"], "difficulty": "Medium"},
        {"question": "Why does Adam include a bias correction term?", "answer": "The moving averages of gradients (first moment) and squared gradients (second moment) in Adam are initialized to zero. Early in training, these averages are heavily biased towards zero. The bias correction terms scale up the estimates to counteract this initialization bias, especially important during the first few iterations.", "companyTags": ["Amazon", "OpenAI"], "difficulty": "Advanced"},
        {"question": "What is the role of the epsilon parameter in Adam and RMSprop?", "answer": "Epsilon is a very small constant (e.g., 1e-8) added to the denominator of the parameter update rule. Its sole purpose is numerical stability: it prevents division by zero when the running average of the squared gradients is extremely small or zero.", "companyTags": ["Apple", "Netflix"], "difficulty": "Easy"},
        {"question": "Why might one prefer SGD with Momentum over Adam for training ResNets on ImageNet?", "answer": "Empirically, while Adam converges much faster initially, SGD with Momentum often achieves a lower final validation error and generalizes better. Adam's adaptive learning rates can cause it to converge to sharp, narrow local minima, whereas SGD's noisy updates help it find wider, flatter minima that generalize well.", "companyTags": ["Google", "Meta"], "difficulty": "Advanced"},
        {"question": "How do you choose a learning rate?", "answer": "Common methods include using a Learning Rate Finder (gradually increasing LR and plotting loss), grid search, or starting with a standard default (like 1e-3 for Adam or 0.1 for SGD) and utilizing a learning rate scheduler (like StepLR, CosineAnnealing) to decay it over time.", "companyTags": ["Microsoft", "Amazon"], "difficulty": "Medium"},
        {"question": "Explain 'Mini-batch' gradient descent and why it is used.", "answer": "Mini-batch gradient descent computes the gradient on a small random subset of the training data (e.g., 32 or 128 samples). It balances the computational efficiency of Batch Gradient Descent (utilizing vectorized hardware operations) with the stochastic noise of true SGD, which helps escape local minima.", "companyTags": ["Meta", "Netflix"], "difficulty": "Medium"},
        {"question": "What happens if the learning rate is too high?", "answer": "The optimizer will take steps that are too large, overshooting the local minimum. This causes the loss to oscillate wildly and often diverge to infinity (NaN loss).", "companyTags": ["Apple", "Google"], "difficulty": "Easy"},
        {"question": "What is a learning rate scheduler and why use one?", "answer": "A scheduler adjusts the learning rate during training according to a predefined schedule (e.g., reducing it by a factor of 10 every 30 epochs). High learning rates early on speed up convergence and help escape local minima, while lower rates later allow the model to fine-tune and settle precisely into a minimum.", "companyTags": ["Amazon", "Microsoft"], "difficulty": "Medium"},
        {"question": "Explain the concept of an 'Adaptive Learning Rate'.", "answer": "Adaptive optimizers (like AdaGrad, RMSprop, Adam) maintain a separate, dynamically adjusted learning rate for every single parameter in the model. Parameters associated with sparse, infrequent features get larger updates, while frequently updated parameters get smaller updates.", "companyTags": ["Google", "Meta"], "difficulty": "Medium"},
        {"question": "What is the problem with AdaGrad that RMSprop solves?", "answer": "AdaGrad accumulates the sum of squared gradients over all time in its denominator. This sum strictly grows, causing the learning rate to monotonically decrease to zero, prematurely stopping training. RMSprop fixes this by using an exponentially decaying moving average, so it only 'remembers' recent gradients.", "companyTags": ["OpenAI", "Microsoft"], "difficulty": "Advanced"},
        {"question": "What is L-BFGS and when would you use it?", "answer": "L-BFGS is a quasi-Newton, second-order optimization algorithm. It approximates the inverse Hessian matrix to take more direct steps toward the minimum. It is highly effective and memory-efficient for small datasets or full-batch training, but poorly suited for mini-batch training or very large neural networks.", "companyTags": ["Netflix", "Apple"], "difficulty": "Advanced"},
        {"question": "Can optimization algorithms escape global minima?", "answer": "Once an optimizer is exactly at a global minimum (gradient = 0), deterministic updates stop. However, the stochastic noise inherent in Mini-batch SGD can sometimes bounce the parameters out of narrow minima. Generally, we want to find good local minima, as finding the true global minimum in deep learning is computationally intractable.", "companyTags": ["Meta", "Google"], "difficulty": "Medium"},
        {"question": "How does weight decay differ from L2 regularization?", "answer": "Mathematically they are identical for standard SGD: adding an L2 penalty to the loss is equivalent to subtracting a fraction of the weight during the update. However, for adaptive optimizers like Adam, they differ. True weight decay (AdamW) applies the decay directly to the weights outside of the adaptive gradient scaling.", "companyTags": ["Google", "OpenAI"], "difficulty": "Advanced"},
        {"question": "What is 'Gradient Clipping'?", "answer": "Gradient clipping caps the maximum norm or value of the gradients during backpropagation. It is primarily used to solve the 'exploding gradient' problem, especially common in training Recurrent Neural Networks (RNNs).", "companyTags": ["Amazon", "Microsoft"], "difficulty": "Medium"},
        {"question": "Why is the loss landscape of a deep neural network non-convex?", "answer": "The composition of many non-linear activation functions and matrix multiplications creates a highly complex, high-dimensional loss surface riddled with local minima, saddle points, and plateaus. There is no single global bowl shape.", "companyTags": ["Apple", "Meta"], "difficulty": "Advanced"},
        {"question": "What is a saddle point in optimization?", "answer": "A saddle point is a point on the loss surface where the gradient is zero, but it is a minimum along some dimensions and a maximum along others. In high-dimensional spaces, saddle points are much more common than true local minima and can severely slow down training.", "companyTags": ["Google", "Netflix"], "difficulty": "Medium"},
        {"question": "How does Momentum help with saddle points?", "answer": "At a saddle point, the gradient is zero. However, Momentum accumulates velocity from previous steps. If the optimizer rolls into a saddle point with enough momentum, it can coast right through the zero-gradient region and escape.", "companyTags": ["Meta", "Amazon"], "difficulty": "Advanced"},
        {"question": "What does a batch size of 1 mean?", "answer": "A batch size of 1 implies true Stochastic Gradient Descent. The model weights are updated after calculating the error for every single individual training sample. It is very noisy and computationally inefficient due to lack of vectorization.", "companyTags": ["Apple", "Microsoft"], "difficulty": "Easy"},
        {"question": "How does batch size affect the choice of learning rate?", "answer": "Generally, if you increase the batch size, you should also increase the learning rate (often linearly, e.g., if you double batch size, double the LR). A larger batch provides a more accurate estimate of the true gradient, allowing for larger, safer steps.", "companyTags": ["Google", "Meta"], "difficulty": "Advanced"},
        {"question": "What is Nesterov Accelerated Gradient (NAG)?", "answer": "NAG is a variation of Momentum. Instead of calculating the gradient at the current position, NAG 'looks ahead' and calculates the gradient at the approximate future position where the momentum will carry the parameters. This provides a more responsive correction and helps prevent overshooting.", "companyTags": ["Microsoft", "OpenAI"], "difficulty": "Advanced"}
    ],
    "coding": {
        "tutorial": {
            "title": "RMSprop Squared gradient update",
            "description": "Update the running average of squared gradients: v = beta * v + (1 - beta) * g^2. Scikit-learn equivalent: Use `MLPClassifier(solver='adam')` or `'sgd'`. Scikit-learn does not expose RMSprop directly, but Adam relies on this exact mathematical concept.",
            "pseudoCode": "RETURN beta * v + (1 - beta) * g^2",
            "starterCode": "def rmsprop_v(v, g, beta):\n    # TODO: return updated v\n    return v\n\nprint(rmsprop_v(0.5, 0.2, 0.9))",
            "expectedOutput": "0.454",
            "solution": "def rmsprop_v(v, g, beta):\n    return beta * v + (1.0 - beta) * (g ** 2)\n\nprint(rmsprop_v(0.5, 0.2, 0.9))",
            "hints": ["Multiply beta by v, and add 1 - beta multiplied by squared gradient."],
            "testKeywords": ["beta * v", "1.0 - beta"]
        },
        "project": {
            "title": "Compare Optimizers using Scikit-learn",
            "description": "Train two `MLPClassifier` models on the same toy dataset, one using 'sgd' and one using 'adam', and compare their training progress.",
            "pseudoCode": "CREATE MLP with solver='sgd'. FIT it. CREATE MLP with solver='adam'. FIT it. COMPARE loss curves or final predictions.",
            "starterCode": "import numpy as np\nfrom sklearn.neural_network import MLPClassifier\nimport warnings\nwarnings.filterwarnings('ignore')\n\nX = np.random.RandomState(42).randn(200, 2)\ny = (X[:, 0] + X[:, 1] > 0).astype(int)\n\n# TODO: Create MLPClassifier with solver='sgd', max_iter=20, random_state=42\n# TODO: Create MLPClassifier with solver='adam', max_iter=20, random_state=42\n# TODO: Fit both models on X, y\n# TODO: Print the final loss attribute (model.loss_) for both models rounded to 4 decimals\n",
            "expectedOutput": "SGD Loss: 0.6974\nAdam Loss: 0.6690",
            "solution": "import numpy as np\nfrom sklearn.neural_network import MLPClassifier\nimport warnings\nwarnings.filterwarnings('ignore')\n\nX = np.random.RandomState(42).randn(200, 2)\ny = (X[:, 0] + X[:, 1] > 0).astype(int)\n\nmlp_sgd = MLPClassifier(solver='sgd', max_iter=20, random_state=42)\nmlp_adam = MLPClassifier(solver='adam', max_iter=20, random_state=42)\n\nmlp_sgd.fit(X, y)\nmlp_adam.fit(X, y)\n\nprint(f\"SGD Loss: {mlp_sgd.loss_:.4f}\")\nprint(f\"Adam Loss: {mlp_adam.loss_:.4f}\")",
            "hints": ["Use the `solver` parameter in MLPClassifier.", "The final loss is stored in the `.loss_` attribute of the fitted model."],
            "testKeywords": ["MLPClassifier", "solver='sgd'", "solver='adam'", ".loss_"]
        },
        "assignment": {
            "title": "SGD with Momentum via Scikit-learn",
            "description": "Configure an `MLPClassifier` to use SGD with Nesterov momentum and evaluate it using cross-validation.",
            "pseudoCode": "CREATE MLPClassifier with solver='sgd', momentum=0.9, nesterovs_momentum=True. RUN cross_val_score. PRINT mean.",
            "starterCode": "import numpy as np\nfrom sklearn.neural_network import MLPClassifier\nfrom sklearn.model_selection import cross_val_score\n\nX = np.random.RandomState(42).randn(150, 4)\ny = (X[:, 0] ** 2 + X[:, 1] > 1).astype(int)\n\n# TODO: Create MLPClassifier with solver='sgd', momentum=0.9, nesterovs_momentum=True, max_iter=500, random_state=42\n# TODO: Run cross_val_score with cv=3\n# TODO: Print the mean score rounded to 2 decimal places\n",
            "expectedOutput": "0.93",
            "solution": "import numpy as np\nfrom sklearn.neural_network import MLPClassifier\nfrom sklearn.model_selection import cross_val_score\n\nX = np.random.RandomState(42).randn(150, 4)\ny = (X[:, 0] ** 2 + X[:, 1] > 1).astype(int)\n\nmlp = MLPClassifier(solver='sgd', momentum=0.9, nesterovs_momentum=True, max_iter=500, random_state=42)\nscores = cross_val_score(mlp, X, y, cv=3)\nprint(f\"{scores.mean():.2f}\")",
            "hints": ["Pass `momentum=0.9` and `nesterovs_momentum=True` to the classifier.", "Use cross_val_score and take the mean."],
            "testKeywords": ["solver='sgd'", "momentum=0.9", "cross_val_score"]
        }
    }
}

ts_file_content = f"""import type {{ MLModule }} from '../../types';

export const optimizationAlgorithms: MLModule = {json.dumps(data, indent=2)};
"""
with open("/home/legion/fastapi/machine_learning_tutorial/src/data/modules/optimizationAlgorithms.ts", "w") as f:
    f.write(ts_file_content)

print("optimizationAlgorithms.ts generated")
