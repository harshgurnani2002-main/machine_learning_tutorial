import type { MLModule } from '../../types';

export const singlePerceptron: MLModule = {
    id: 'single-perceptron',
    title: 'Single-Layer Perceptron',
    category: 'Deep Learning',
    description: 'Learn linear decision boundaries using step activation gates.',
    formula: 'y = f(w^T x + b)',
    interactiveSummary: 'In this interactive demo, you can adjust the weights and bias of a single perceptron to see how the decision boundary shifts in a 2D feature space. Try to classify different linearly separable datasets by tuning the parameters manually.',
    theory: `### What is it?
The Single-Layer Perceptron is the simplest type of artificial neural network and forms the foundation of deep learning. It is a linear binary classifier that maps its input $x$ (a real-valued vector) to an output value $f(x)$ (a single binary value) across a single layer of weights. Inspired by biological neurons, it receives incoming signals, applies a weighted sum, and "fires" if the sum exceeds a certain threshold.

### Why do we need it?
Before modern deep learning architectures like CNNs and Transformers, the perceptron was introduced in the late 1950s as a foundational model for supervised learning. We need it to understand the fundamental building blocks of neural networks: weighted sums, biases, and activation functions. It serves as the baseline for classifying linearly separable patterns, mimicking the basic all-or-nothing firing of biological neurons. Understanding the perceptron is essential before diving into Multi-Layer Perceptrons (MLPs) and backpropagation.

### How does it work?
A perceptron receives multiple input signals (features), multiplies them by learned weights, and sums them up along with a bias term. This weighted sum is then passed through an activation function (typically the Heaviside step function). If the sum is greater than or equal to zero, the perceptron outputs a 1 (positive class); otherwise, it outputs a 0 (negative class). During training, it uses a simple error-correction rule: it updates its weights whenever it makes a mistake, incrementally shifting the decision boundary until the classes are cleanly separated.

### The Math Behind It
Given an input vector $x \\in \\mathbb{R}^n$, weight vector $w \\in \\mathbb{R}^n$, and bias $b \\in \\mathbb{R}$:
1. Compute the linear combination (activation potential):
$$z = w^T x + b = \\sum_{i=1}^n w_i x_i + b$$
2. Apply the step activation function:
$$f(z) = \begin{cases} 1 & z \\ge 0 \\ 0 & z < 0 \\end{cases}$$
3. The learning rule updates parameters for each training example $(x^{(i)}, y^{(i)})$ with prediction $\\hat{y}^{(i)}$ and learning rate $\\eta$:
$$w \\leftarrow w + \\eta (y^{(i)} - \\hat{y}^{(i)}) x^{(i)}$$
$$b \\leftarrow b + \\eta (y^{(i)} - \\hat{y}^{(i)})$$

### Worked Example
Suppose we have a 2D input $x = [1, 0.5]^T$, weights $w = [0.4, -0.2]^T$, and bias $b = -0.1$.
1. Compute $z = (0.4)(1) + (-0.2)(0.5) - 0.1 = 0.4 - 0.1 - 0.1 = 0.2$.
2. Apply step function: since $0.2 \\ge 0$, output is $1$.
If the true label was $0$, the error is $0 - 1 = -1$.
With $\\eta = 0.1$, the new weights become:
$w_1 = 0.4 + 0.1(-1)(1) = 0.3$
$w_2 = -0.2 + 0.1(-1)(0.5) = -0.25$
$b = -0.1 + 0.1(-1) = -0.2$.

### Common Pitfalls
1. **Non-Linearly Separable Data:** The perceptron will never converge if the classes cannot be separated by a straight line (e.g., the XOR problem). It will oscillate infinitely.
2. **Learning Rate Too High:** It might cause the weights to oscillate wildly and never settle on a valid decision boundary.
3. **No Activation Function:** Without an activation function, it is just a standard linear regression model, failing to produce binary class outputs.
4. **Zero Initialization Trap:** While a single perceptron can learn with zero-initialized weights, it is bad practice as it scales poorly to deeper networks.

### When to Use vs Not Use
**When to Use:**
- When the data is guaranteed to be linearly separable.
- For simple binary classification baselines.
- For educational purposes to introduce neural network mechanics.
- In online learning settings where features stream in one by one and linear separation is sufficient.

**When Not to Use:**
- When data is non-linear (most real-world tasks like image or text classification).
- When multi-class classification is needed without One-vs-Rest wrappers.
- When you need class probabilities (use Logistic Regression instead).
- When optimization requires gradient descent (the step function is non-differentiable).

### Key Takeaways
- The Single-Layer Perceptron is a linear classifier acting as the precursor to deep learning.
- It uses a step function and simple error-correction weight updates.
- Guaranteed to converge only on linearly separable data (Perceptron Convergence Theorem).
- The inability to solve the XOR problem mathematically proved by Minsky and Papert led to the first "AI Winter" until MLPs were popularized.

#### Python Implementation

\`\`\`python
import numpy as np

X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([0, 0, 0, 1])
w = np.random.randn(3)
for _ in range(10):
    for xi, yi in zip(X, y):
        pred = 1 if xi @ w > 0 else 0
        w += 0.1 * (yi - pred) * xi
print(f"Trained weights: {w}")
\`\`\`
`,
    simulatorId: 'perceptron',
    quiz: [
      { id: 'slp_q1', question: 'What classification limitation does a single perceptron have?', options: ['Cannot solve XOR (non-linear splits)', 'Cannot solve OR', 'Cannot classify negative numbers', 'Slow predictions'], correctAnswer: 'Cannot solve XOR (non-linear splits)', explanation: 'Single perceptrons can only form linear boundaries, meaning they cannot solve non-linear problems like XOR.' },
      { id: 'slp_q2', question: 'What activation function does a standard perceptron use?', options: ['Sigmoid', 'Heaviside Step Function', 'ReLU', 'Tanh'], correctAnswer: 'Heaviside Step Function', explanation: 'A standard perceptron uses the Heaviside step function which outputs 1 if z >= 0 else 0.' },
      { id: 'slp_q3', question: 'Under what condition is the perceptron learning rule guaranteed to converge?', options: ['If learning rate is large', 'If data is linearly separable', 'If data has more than 1000 samples', 'If bias is 0'], correctAnswer: 'If data is linearly separable', explanation: 'The Perceptron Convergence Theorem states it will find a solution in finite steps if data is linearly separable.' },
      { id: 'slp_q4', question: 'What happens if a perceptron is trained on non-linearly separable data?', options: ['It outputs probabilities', 'It transforms the data', 'It oscillates and never converges', 'It drops the bias term'], correctAnswer: 'It oscillates and never converges', explanation: 'Without a margin to separate the classes, the error-correction rule keeps updating weights infinitely.' },
      { id: 'slp_q5', question: 'In the weight update rule w = w + \\eta (y - \\hat{y}) x, what is \\eta?', options: ['Error', 'Learning rate', 'Momentum', 'Bias'], correctAnswer: 'Learning rate', explanation: '\\eta scales the magnitude of the weight updates.' },
      { id: 'slp_q6', question: 'If prediction \\hat{y}=1 and true label y=0, what is the value of (y - \\hat{y})?', options: ['1', '-1', '0', 'None of the above'], correctAnswer: '-1', explanation: '0 - 1 = -1, meaning the perceptron over-predicted and must reduce weights.' },
      { id: 'slp_q7', question: 'What represents the decision boundary of a perceptron with 2 inputs?', options: ['A point', 'A curve', 'A straight line', 'A hyperplane'], correctAnswer: 'A straight line', explanation: 'In 2D space, the linear equation w1*x1 + w2*x2 + b = 0 forms a straight line.' },
      { id: 'slp_q8', question: 'Why is the step function not used in modern backpropagation?', options: ['It is too slow', 'It is non-differentiable at zero and derivative is zero elsewhere', 'It only outputs positive numbers', 'It requires too much memory'], correctAnswer: 'It is non-differentiable at zero and derivative is zero elsewhere', explanation: 'Gradient descent requires informative derivatives to update weights, but the step function derivative is almost always zero.' },
      { id: 'slp_q9', question: 'Which logical operation can a single perceptron compute?', options: ['XOR', 'XNOR', 'AND', 'None'], correctAnswer: 'AND', explanation: 'AND is linearly separable, while XOR and XNOR are not.' },
      { id: 'slp_q10', question: 'Who highlighted the XOR limitation of the perceptron?', options: ['Rosenblatt', 'Minsky and Papert', 'Turing', 'Hinton'], correctAnswer: 'Minsky and Papert', explanation: 'Their 1969 book "Perceptrons" mathematically proved the limitations of single-layer perceptrons.' }
    ],
    coding: {
      tutorial: {
        title: 'Heaviside Step Function',
        description: 'Implement the threshold activation: 1 if z >= 0 else 0. Note: In sklearn, you would just use `from sklearn.linear_model import Perceptron`.',
        pseudoCode: 'function step_func(z): return 1 if z >= 0 else 0',
        starterCode: `def step_func(z):\n    # TODO: Implement step function\n    return 0\n\nprint([step_func(-2), step_func(1.5)])`,
        expectedOutput: '[0, 1]',
        solution: `def step_func(z):\n    return 1 if z >= 0 else 0\n\nprint([step_func(-2), step_func(1.5)])`,
        hints: ['Check if z is greater than or equal to 0.'],
        testKeywords: ['>= 0']
      },
      project: {
        title: 'PyTorch Perceptron Basics',
        description: 'Use a PyTorch pseudo framework to build a Single-Layer Perceptron and predict an unseen sample using an inline synthetic dataset.',
        pseudoCode: 'Initialize a linear layer with 1 output. Apply a sigmoid or threshold. Pass the input tensor.',
        starterCode: `import torch\nimport torch.nn as nn\n\n# AND gate data (PyTorch tensors)\nX = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])\ny = torch.tensor([[0.0], [0.0], [0.0], [1.0]])\n\n# TODO: Create a simple nn.Linear layer (2 inputs, 1 output)\n# Initialize model, pass [[1.0, 1.0]] and apply a threshold of 0.5\nmodel = None\npred = [0]\nprint(pred)`,
        expectedOutput: 'tensor([[1.]])',
        solution: `import torch\nimport torch.nn as nn\n\nX = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])\ny = torch.tensor([[0.0], [0.0], [0.0], [1.0]])\n\nmodel = nn.Linear(2, 1)\n# Manually set weights for AND gate to avoid training loop in this basic demo\nwith torch.no_grad():\n    model.weight.fill_(1.0)\n    model.bias.fill_(-1.5)\n\ntest_input = torch.tensor([[1.0, 1.0]])\noutput = model(test_input)\npred = (output >= 0).float()\nprint(pred)`,
        hints: ['Use nn.Linear(2, 1)', 'Set weights manually or use a simple threshold', 'Convert boolean to float'],
        testKeywords: ['nn.Linear', 'tensor']
      },
      assignment: {
        title: 'Training a Perceptron in PyTorch',
        description: 'Implement a basic training loop for a perceptron using PyTorch BCEWithLogitsLoss to classify synthetic data.',
        pseudoCode: 'Create model, optimizer (SGD), and loss function. Loop epochs, zero_grad, forward, backward, step.',
        starterCode: `import torch\nimport torch.nn as nn\nimport torch.optim as optim\n\n# Synthetic linearly separable data\nX = torch.tensor([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [-1.0, -1.0]])\ny = torch.tensor([[1.0], [1.0], [1.0], [0.0]])\n\nmodel = nn.Linear(2, 1)\n# TODO: Setup BCEWithLogitsLoss and SGD optimizer\n# Run a small training loop (100 epochs)\n\nprint("Training complete.")`,
        expectedOutput: 'Training complete.',
        solution: `import torch\nimport torch.nn as nn\nimport torch.optim as optim\n\nX = torch.tensor([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [-1.0, -1.0]])\ny = torch.tensor([[1.0], [1.0], [1.0], [0.0]])\n\nmodel = nn.Linear(2, 1)\ncriterion = nn.BCEWithLogitsLoss()\noptimizer = optim.SGD(model.parameters(), lr=0.1)\n\nfor epoch in range(100):\n    optimizer.zero_grad()\n    outputs = model(X)\n    loss = criterion(outputs, y)\n    loss.backward()\n    optimizer.step()\n\nprint("Training complete.")`,
        hints: ['criterion = nn.BCEWithLogitsLoss()', 'optimizer = optim.SGD(model.parameters(), lr=0.1)', 'optimizer.zero_grad() -> backward() -> step()'],
        testKeywords: ['criterion', 'backward', 'step', 'BCEWithLogitsLoss']
      }
    },
    interviewQuestions: [
      { question: 'What is a single-layer perceptron?', answer: 'A linear binary classifier that uses a step function for its activation. It forms a single hyperplane to separate two classes.', companyTags: ['Amazon', 'Meta'], difficulty: 'Easy' },
      { question: 'Explain the perceptron update rule.', answer: 'If prediction matches target, no change. If target is 1 and prediction 0, add input features scaled by learning rate to weights. If target is 0 and prediction 1, subtract them.', companyTags: ['Google'], difficulty: 'Medium' },
      { question: 'What is the XOR problem?', answer: 'XOR requires a non-linear decision boundary, which a single perceptron cannot create, highlighting its limitation.', companyTags: ['Apple', 'Microsoft'], difficulty: 'Easy' },
      { question: 'What happens if perceptron data is not linearly separable?', answer: 'The training loop will oscillate indefinitely because no single line can separate the classes with zero error.', companyTags: ['Netflix'], difficulty: 'Medium' },
      { question: 'What is the role of the bias term?', answer: 'It shifts the decision boundary away from the origin, allowing the perceptron to fit datasets where the ideal boundary does not pass through (0,0).', companyTags: ['Meta'], difficulty: 'Easy' },
      { question: 'Why don\'t we use the step function in deep neural networks?', answer: 'Its derivative is zero everywhere except at the origin where it\'s undefined, meaning gradient descent cannot backpropagate error signals.', companyTags: ['Google', 'Meta'], difficulty: 'Medium' },
      { question: 'Can a perceptron do regression?', answer: 'No, because its output is constrained by a step function to discrete values (0 or 1).', companyTags: ['Amazon'], difficulty: 'Easy' },
      { question: 'How is Logistic Regression different from a Perceptron?', answer: 'Logistic Regression uses a sigmoid activation and optimizes a continuous cross-entropy loss, producing probabilities, whereas a Perceptron uses a step function and optimizes a discrete error.', companyTags: ['Google', 'Apple'], difficulty: 'Medium' },
      { question: 'What is the Perceptron Convergence Theorem?', answer: 'It guarantees that if a dataset is linearly separable, the perceptron learning algorithm will find a separating hyperplane in a finite number of iterations.', companyTags: ['Microsoft'], difficulty: 'Hard' },
      { question: 'Does a perceptron find the OPTIMAL decision boundary?', answer: 'No, it stops as soon as it finds ANY boundary that perfectly separates the data, unlike SVMs which maximize the margin.', companyTags: ['Meta'], difficulty: 'Medium' },
      { question: 'What is the time complexity of one perceptron update?', answer: 'O(N), where N is the number of input features, since it requires a dot product between the weight vector and the input vector.', companyTags: ['Amazon'], difficulty: 'Easy' },
      { question: 'Can a perceptron handle multi-class classification natively?', answer: 'No, it is strictly a binary classifier. Multi-class requires strategies like One-vs-Rest (OvR).', companyTags: ['Apple'], difficulty: 'Medium' },
      { question: 'What happens if we initialize perceptron weights to zero?', answer: 'It will still learn successfully. Symmetry breaking is only required in multi-layer architectures, not single-layer perceptrons.', companyTags: ['Google'], difficulty: 'Hard' },
      { question: 'Is feature scaling necessary for perceptrons?', answer: 'Yes, if features have drastically different scales, the perceptron will update weights unevenly, causing slower convergence.', companyTags: ['Netflix'], difficulty: 'Medium' },
      { question: 'How does learning rate affect the perceptron algorithm?', answer: 'It scales the magnitude of the weight updates. In the standard perceptron, it merely scales the final weight vector uniformly and doesn\'t change the decision boundary shape, but it affects convergence speed.', companyTags: ['Meta'], difficulty: 'Hard' },
      { question: 'Describe the hinge loss function. Is it used in perceptrons?', answer: 'Hinge loss is used in SVMs. The perceptron uses a zero-one loss mathematically represented as a piecewise linear error, not a true hinge loss.', companyTags: ['Google'], difficulty: 'Hard' },
      { question: 'If you have 100 features and 1,000,000 samples, is a perceptron computationally efficient?', answer: 'Yes, because updates are simple vector additions, making it highly scalable and capable of online learning.', companyTags: ['Amazon'], difficulty: 'Medium' },
      { question: 'What does the term "epoch" mean in perceptron training?', answer: 'One complete pass through the entire training dataset.', companyTags: ['Apple'], difficulty: 'Easy' },
      { question: 'Can a single perceptron learn the identity function?', answer: 'Yes, the identity function f(x) = x for binary input is linearly separable.', companyTags: ['Microsoft'], difficulty: 'Medium' },
      { question: 'Why was the perceptron criticized in the late 1960s?', answer: 'Because Minsky and Papert proved it could not learn the XOR function, leading to a decline in neural network funding.', companyTags: ['IBM'], difficulty: 'Easy' }
    ]
};
