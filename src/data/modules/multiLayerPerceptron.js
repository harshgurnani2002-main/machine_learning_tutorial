export const multiLayerPerceptron = {
    id: 'multi-layer-perceptron',
    title: 'Multi-Layer Perceptron (MLP)',
    category: 'Deep Learning',
    description: 'Stack multiple layers of perceptrons with non-linear activations to solve complex tasks.',
    formula: 'y = \\sigma(W_2 \\cdot \\text{ReLU}(W_1 x + b_1) + b_2)',
    interactiveSummary: 'This simulator visualizes a live Multi-Layer Perceptron architecture, letting you add or remove hidden layers and tune the neuron count per layer in real time. Watch as forward-pass activations light up neuron by neuron across the network, and observe how ReLU vs. Sigmoid activation functions change the gradient flow during backpropagation. The loss curve updates dynamically after each training epoch, showing how deeper architectures converge faster on complex non-linear tasks. Toggle different weight initialization schemes to see how symmetry-breaking affects which features each hidden layer learns to detect.',
    theory: `### What is it?
A Multi-Layer Perceptron (MLP) is a fully connected class of feedforward artificial neural network (ANN). It consists of at least three layers of nodes: an input layer, one or more hidden layers, and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function.

### Why do we need it?
Single-layer perceptrons can only classify linearly separable data, famously failing on the XOR problem. We need MLPs to model complex, non-linear relationships. By stacking layers and introducing non-linear activation functions (like ReLU or Sigmoid), MLPs can approximate any continuous function (Universal Approximation Theorem), making them capable of solving sophisticated pattern recognition tasks in computer vision, natural language processing, and tabular data analysis.

### How does it work?
Data flows forward from the input layer through the hidden layers to the output layer. At each layer, the inputs are multiplied by a weight matrix, a bias vector is added, and the result is passed through a non-linear activation function. During training, the network predicts an output, computes the error using a loss function, and then uses an algorithm called **Backpropagation** to compute the gradient of the loss with respect to each weight by applying the chain rule of calculus backwards from the output to the input. Finally, an optimizer like Gradient Descent or Adam updates the weights to minimize the error.

### The Math Behind It
For a network with one hidden layer:
1. **Forward Pass:**
   Hidden layer output: $h = \\sigma(W_1 x + b_1)$
   Final output: $\\hat{y} = \text{softmax}(W_2 h + b_2)$
   Where $W_1, W_2$ are weight matrices and $b_1, b_2$ are bias vectors. $\\sigma$ is a non-linear activation like ReLU.

2. **Loss Function (e.g., Cross-Entropy):**
   $L = -\\sum_{i} y_i \\log(\\hat{y}_i)$

3. **Backward Pass (Backpropagation):**
   Compute gradients using the chain rule. For the output layer weights:
   $\frac{\\partial L}{\\partial W_2} = \frac{\\partial L}{\\partial \\hat{y}} \frac{\\partial \\hat{y}}{\\partial (W_2 h + b_2)} \frac{\\partial (W_2 h + b_2)}{\\partial W_2}$
   These gradients are then used to update weights: $W_2 \\leftarrow W_2 - \\eta \frac{\\partial L}{\\partial W_2}$.

### Worked Example
Suppose we have a 1D input $x = 2$, a hidden layer with 2 neurons, and a 1D output.
- $W_1 = [0.5, -0.5]^T$, $b_1 = [0, 1]^T$
- $W_2 = [1, 2]$, $b_2 = 0$
- Activation: ReLU ($f(z) = \\max(0, z)$)

**Forward Pass:**
1. $z_1 = W_1 x + b_1 = [0.5(2)+0, -0.5(2)+1]^T = [1, 0]^T$
2. $h = \text{ReLU}(z_1) = [1, 0]^T$
3. $z_2 = W_2 h + b_2 = 1(1) + 2(0) + 0 = 1$
The output of the network is 1. If the target was 2, we would backpropagate the error (MSE) back through the weights to update them.

### Common Pitfalls
1. **Vanishing/Exploding Gradients:** Deep MLPs with sigmoid or tanh activations suffer from vanishing gradients. Using ReLU and proper weight initialization (like He initialization) mitigates this.
2. **Overfitting:** MLPs with too many parameters can easily memorize the training data. Use regularization techniques like Dropout, L2 Weight Decay, or Early Stopping.
3. **Dead ReLUs:** A large learning rate can push a ReLU neuron into a state where it always outputs 0 and never recovers.
4. **Symmetric Initialization:** Initializing all weights to zero will cause all neurons to learn the exact same features. Always use random initialization to break symmetry.

### When to Use vs Not Use
**When to Use:**
- For tabular or structured data with complex non-linear relationships.
- As the final fully-connected classifier layers in CNNs or feature extractors.
- When you need a highly flexible universal approximator and have sufficient training data.

**When Not to Use:**
- For image data where spatial hierarchies matter (use CNNs).
- For sequential or time-series data where temporal context is crucial (use RNNs, LSTMs, or Transformers).
- For very small datasets where simple linear models or tree-based models (XGBoost) are less prone to overfitting.

### Key Takeaways
- MLPs solve the XOR problem and act as universal function approximators.
- They rely on non-linear activation functions and the backpropagation algorithm.
- Regularization and proper initialization are critical to training deep MLPs successfully.
- While powerful for unstructured vectors, they are often superseded by specialized architectures for images and sequences.

#### Python Implementation

\`\`\`python
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=500, random_state=42)
mlp.fit(X, y)
print(f"Accuracy: {mlp.score(X, y):.3f}")
\`\`\`
`,
    simulatorId: 'mlp',
    quiz: [
        { id: 'mlp_q1', question: 'What problem does an MLP solve that a single perceptron cannot?', options: ['Linear Regression', 'The XOR Problem', 'Image Compression', 'Audio Generation'], correctAnswer: 'The XOR Problem', explanation: 'MLPs can model non-linear boundaries, allowing them to solve the XOR problem.' },
        { id: 'mlp_q2', question: 'Which algorithm is used to train Multi-Layer Perceptrons?', options: ['K-Means', 'Backpropagation', 'Apriori', 'PageRank'], correctAnswer: 'Backpropagation', explanation: 'Backpropagation computes the gradients of the loss with respect to the network weights using the chain rule.' },
        { id: 'mlp_q3', question: 'What is the purpose of non-linear activation functions in an MLP?', options: ['To speed up computation', 'To allow the network to learn non-linear representations', 'To prevent overfitting', 'To reduce memory usage'], correctAnswer: 'To allow the network to learn non-linear representations', explanation: 'Without non-linear activations, stacking multiple linear layers is mathematically equivalent to a single linear layer.' },
        { id: 'mlp_q4', question: 'What happens if you initialize all weights in an MLP to zero?', options: ['It learns faster', 'Neurons fail to break symmetry and learn identical features', 'It perfectly avoids overfitting', 'Loss becomes infinite'], correctAnswer: 'Neurons fail to break symmetry and learn identical features', explanation: 'Zero initialization causes all neurons in a hidden layer to compute identical gradients and undergo identical updates.' },
        { id: 'mlp_q5', question: 'What is a common symptom of the vanishing gradient problem?', options: ['Loss goes to NaN', 'Weights update too rapidly', 'Early layers train extremely slowly', 'The network outputs only 1s'], correctAnswer: 'Early layers train extremely slowly', explanation: 'Gradients become vanishingly small as they propagate backwards, causing early layers to barely update.' },
        { id: 'mlp_q6', question: 'Which activation function is most commonly used in hidden layers to avoid vanishing gradients?', options: ['Sigmoid', 'Tanh', 'ReLU', 'Step Function'], correctAnswer: 'ReLU', explanation: 'ReLU (Rectified Linear Unit) does not saturate in the positive domain, preventing vanishing gradients.' },
        { id: 'mlp_q7', question: 'What is the Universal Approximation Theorem?', options: ['MLPs can compute any function perfectly in 1 epoch', 'A neural network with one hidden layer can approximate any continuous function', 'Every dataset is linearly separable in high dimensions', 'MLPs always converge to a global minimum'], correctAnswer: 'A neural network with one hidden layer can approximate any continuous function', explanation: 'It proves the theoretical capability of MLPs to model any continuous mathematical function given enough neurons.' },
        { id: 'mlp_q8', question: 'What technique is used to prevent an MLP from overfitting?', options: ['Gradient Ascent', 'Dropout', 'Zero Initialization', 'Increasing learning rate'], correctAnswer: 'Dropout', explanation: 'Dropout randomly deactivates neurons during training, forcing the network to learn robust, distributed representations.' },
        { id: 'mlp_q9', question: 'In PyTorch, which loss function is best suited for a multi-class classification MLP?', options: ['MSELoss', 'L1Loss', 'CrossEntropyLoss', 'HuberLoss'], correctAnswer: 'CrossEntropyLoss', explanation: 'CrossEntropyLoss combines LogSoftmax and NLLLoss, ideal for multi-class categorization problems.' },
        { id: 'mlp_q10', question: 'What does the term "forward pass" mean in an MLP?', options: ['Updating the weights', 'Passing data from inputs to outputs to generate a prediction', 'Calculating the loss gradient', 'Initializing the network'], correctAnswer: 'Passing data from inputs to outputs to generate a prediction', explanation: 'The forward pass computes the activations of all neurons layer by layer to produce the final output.' }
    ],
    coding: {
        tutorial: {
            title: 'Sigmoid Activation Function',
            description: 'Implement the sigmoid activation function used in older MLP architectures. Note: In PyTorch/sklearn, this is typically handled via `nn.Sigmoid()` or implicit parameters.',
            pseudoCode: 'function sigmoid(z): return 1 / (1 + exp(-z))',
            starterCode: `import numpy as np\n\ndef sigmoid(z):\n    # TODO: Implement the sigmoid formula\n    return 0.0\n\nprint(np.round(sigmoid(np.array([0, 2, -2])), 2))`,
            expectedOutput: '[0.5  0.88 0.12]',
            solution: `import numpy as np\n\ndef sigmoid(z):\n    return 1 / (1 + np.exp(-z))\n\nprint(np.round(sigmoid(np.array([0, 2, -2])), 2))`,
            hints: ['Use np.exp(-z)'],
            testKeywords: ['np.exp']
        },
        project: {
            title: 'Handwritten Digit Classifier (MLP)',
            description: 'Build a multi-layer perceptron in PyTorch to classify handwritten digit patterns. Each sample is a simplified 64-dimensional pixel-intensity feature vector (8×8 image flattened) representing digits 0–9. Use two hidden layers with ReLU activations and a 10-class softmax output.',
            pseudoCode: `# Handwritten Digit MLP\nclass DigitMLP(nn.Module):\n    __init__:\n        layer1 = Linear(64 → 128)   # input: 64 pixel features\n        layer2 = Linear(128 → 64)   # hidden layer\n        output = Linear(64 → 10)    # 10 digit classes\n        relu   = ReLU()\n    forward(x):\n        x = relu(layer1(x))\n        x = relu(layer2(x))\n        return output(x)            # logits; CrossEntropyLoss applies softmax\n\nmodel = DigitMLP()\noptimizer = Adam(model.parameters(), lr=0.001)\nfor epoch in range(300):\n    loss = CrossEntropyLoss(model(X_train), y_train)\n    loss.backward(); optimizer.step()`,
            starterCode: `import torch\nimport torch.nn as nn\nimport torch.optim as optim\n\n# Simulated 8x8 digit pixel features (64 inputs) -> 10 classes (digits 0-9)\ntorch.manual_seed(42)\nX = torch.randn(100, 64)   # 100 digit samples, 64 pixel features each\ny = torch.randint(0, 10, (100,))  # labels: 0 through 9\n\n# TODO: Build a DigitMLP class with two hidden layers (64->128->64->10)\nclass DigitMLP(nn.Module):\n    def __init__(self):\n        super().__init__()\n        # TODO: Define layer1 (64->128), layer2 (128->64), output (64->10), relu\n        pass\n    def forward(self, x):\n        # TODO: Apply relu(layer1), relu(layer2), then output\n        return x\n\nmodel = DigitMLP()\nprint(model.__class__.__name__)`,
            expectedOutput: 'DigitMLP',
            solution: `import torch\nimport torch.nn as nn\nimport torch.optim as optim\n\ntorch.manual_seed(42)\nX = torch.randn(100, 64)\ny = torch.randint(0, 10, (100,))\n\nclass DigitMLP(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layer1 = nn.Linear(64, 128)\n        self.layer2 = nn.Linear(128, 64)\n        self.output = nn.Linear(64, 10)\n        self.relu = nn.ReLU()\n\n    def forward(self, x):\n        x = self.relu(self.layer1(x))\n        x = self.relu(self.layer2(x))\n        return self.output(x)\n\nmodel = DigitMLP()\nprint(model.__class__.__name__)`,
            hints: ['self.layer1 = nn.Linear(64, 128)', 'self.output = nn.Linear(64, 10)', 'Apply relu after each hidden layer'],
            testKeywords: ['nn.Linear', 'ReLU', 'DigitMLP']
        },
        assignment: {
            title: 'Heart Disease Detection MLP',
            description: 'Build and train an MLP for binary heart disease detection using real clinical features: age, cholesterol, blood_pressure, max_heart_rate, chest_pain_type, and fasting_blood_sugar. Train with Adam and BCEWithLogitsLoss for 200 epochs and print the final training accuracy.',
            pseudoCode: `# Heart Disease MLP Pipeline\nfeatures = [age, cholesterol, blood_pressure, max_heart_rate,\n            chest_pain_type, fasting_blood_sugar]  # 6 inputs\n\nclass HeartDiseaseNet(nn.Module):\n    __init__:\n        fc1 = Linear(6 → 32)\n        fc2 = Linear(32 → 16)\n        fc3 = Linear(16 → 1)   # sigmoid output -> probability of disease\n        relu = ReLU()\n    forward(x):\n        x = relu(fc1(x))\n        x = relu(fc2(x))\n        return fc3(x)           # raw logit; loss applies sigmoid\n\nmodel = HeartDiseaseNet()\ncriterion = BCEWithLogitsLoss()\noptimizer = Adam(model.parameters(), lr=0.01)\nfor epoch in range(200):\n    optimizer.zero_grad()\n    preds = model(X)\n    loss  = criterion(preds, y)\n    loss.backward()\n    optimizer.step()\nprint accuracy`,
            starterCode: `import torch\nimport torch.nn as nn\nimport torch.optim as optim\n\n# Clinical features: age, cholesterol, blood_pressure,\n#                    max_heart_rate, chest_pain_type, fasting_blood_sugar\ntorch.manual_seed(7)\nX = torch.tensor([\n    [63.0, 233.0, 145.0, 150.0, 3.0, 1.0],\n    [37.0, 250.0, 130.0, 187.0, 2.0, 0.0],\n    [41.0, 204.0, 130.0, 172.0, 1.0, 0.0],\n    [56.0, 236.0, 120.0, 178.0, 1.0, 0.0],\n    [57.0, 354.0, 140.0, 163.0, 0.0, 1.0],\n    [57.0, 192.0, 140.0, 148.0, 0.0, 0.0],\n    [56.0, 294.0, 130.0, 256.0, 1.0, 0.0],\n    [44.0, 263.0, 120.0, 173.0, 3.0, 0.0],\n], dtype=torch.float32)\ny = torch.tensor([[1.],[1.],[0.],[0.],[1.],[0.],[1.],[0.]])  # 1=disease, 0=healthy\n\n# TODO: Define HeartDiseaseNet with layers (6->32->16->1)\nclass HeartDiseaseNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        # TODO: fc1 (6->32), fc2 (32->16), fc3 (16->1), relu\n        pass\n    def forward(self, x):\n        # TODO: apply relu(fc1), relu(fc2), fc3\n        return x\n\nmodel = HeartDiseaseNet()\n# TODO: Use BCEWithLogitsLoss and Adam(lr=0.01). Train 200 epochs.\n\npreds = (torch.sigmoid(model(X)) > 0.5).float()\nacc = (preds == y).float().mean().item()\nprint(f"Training Accuracy: {acc:.2f}")`,
            expectedOutput: 'Training Accuracy: 1.00',
            solution: `import torch\nimport torch.nn as nn\nimport torch.optim as optim\n\ntorch.manual_seed(7)\nX = torch.tensor([\n    [63.0, 233.0, 145.0, 150.0, 3.0, 1.0],\n    [37.0, 250.0, 130.0, 187.0, 2.0, 0.0],\n    [41.0, 204.0, 130.0, 172.0, 1.0, 0.0],\n    [56.0, 236.0, 120.0, 178.0, 1.0, 0.0],\n    [57.0, 354.0, 140.0, 163.0, 0.0, 1.0],\n    [57.0, 192.0, 140.0, 148.0, 0.0, 0.0],\n    [56.0, 294.0, 130.0, 256.0, 1.0, 0.0],\n    [44.0, 263.0, 120.0, 173.0, 3.0, 0.0],\n], dtype=torch.float32)\ny = torch.tensor([[1.],[1.],[0.],[0.],[1.],[0.],[1.],[0.]])\n\nclass HeartDiseaseNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear(6, 32)\n        self.fc2 = nn.Linear(32, 16)\n        self.fc3 = nn.Linear(16, 1)\n        self.relu = nn.ReLU()\n\n    def forward(self, x):\n        x = self.relu(self.fc1(x))\n        x = self.relu(self.fc2(x))\n        return self.fc3(x)\n\nmodel = HeartDiseaseNet()\ncriterion = nn.BCEWithLogitsLoss()\noptimizer = optim.Adam(model.parameters(), lr=0.01)\n\nfor epoch in range(200):\n    optimizer.zero_grad()\n    preds = model(X)\n    loss = criterion(preds, y)\n    loss.backward()\n    optimizer.step()\n\npreds = (torch.sigmoid(model(X)) > 0.5).float()\nacc = (preds == y).float().mean().item()\nprint(f"Training Accuracy: {acc:.2f}")`,
            hints: ['self.fc1 = nn.Linear(6, 32)', 'Use BCEWithLogitsLoss()', 'optimizer.zero_grad() then loss.backward() then optimizer.step()'],
            testKeywords: ['HeartDiseaseNet', 'BCEWithLogitsLoss', 'backward', 'Adam']
        }
    },
    interviewQuestions: [
        { question: 'What is backpropagation?', answer: 'An algorithm that computes the gradient of the loss function with respect to the network weights by applying the chain rule of calculus backwards from the output layer to the input layer.', companyTags: ['Google', 'Meta'], difficulty: 'Medium' },
        { question: 'Explain the vanishing gradient problem. How do modern architectures solve it?', answer: 'In deep networks, gradients can decay exponentially during backpropagation because of compounding small derivatives (e.g. from Sigmoid). Modern architectures solve this using ReLU activations, skip connections (ResNets), and Batch Normalization.', companyTags: ['Meta', 'Amazon'], difficulty: 'Hard' },
        { question: 'Why do we initialize weights randomly instead of to zero?', answer: 'If all weights are zero, all neurons in a hidden layer will compute the exact same gradients and update identically, failing to break symmetry and learn diverse features.', companyTags: ['Apple', 'Microsoft'], difficulty: 'Medium' },
        { question: 'What is Dropout and how does it work during inference?', answer: 'Dropout randomly disables neurons during training to prevent overfitting. During inference, dropout is turned off, and the weights are scaled down by the dropout probability to compensate for the active neurons.', companyTags: ['Netflix', 'Google'], difficulty: 'Medium' },
        { question: 'Compare Sigmoid, Tanh, and ReLU.', answer: 'Sigmoid maps to [0,1], suffers from vanishing gradients. Tanh maps to [-1,1], centered around zero, better than sigmoid but still vanishes. ReLU maps negative inputs to 0 and positive to themselves, non-saturating gradient prevents vanishing, but can suffer from dead neurons.', companyTags: ['Meta', 'Amazon'], difficulty: 'Medium' },
        { question: 'What is the role of the Softmax function in an MLP?', answer: 'It is used in the output layer for multi-class classification to convert raw scores (logits) into a normalized probability distribution that sums to 1.', companyTags: ['Google', 'Apple'], difficulty: 'Easy' },
        { question: 'How does Batch Normalization help in training MLPs?', answer: 'It normalizes the inputs of each layer to have a mean of 0 and variance of 1 across the mini-batch, which stabilizes learning, allows higher learning rates, and reduces dependence on careful weight initialization.', companyTags: ['Meta', 'Microsoft'], difficulty: 'Hard' },
        { question: 'What is the difference between an epoch and a mini-batch?', answer: 'A mini-batch is a small subset of the training data used for a single gradient update. An epoch is one full pass through the entire training dataset.', companyTags: ['Amazon'], difficulty: 'Easy' },
        { question: 'How do you choose the number of hidden layers and neurons?', answer: 'It\'s a hyperparameter tuned empirically. Generally, we start small and increase capacity until the network starts overfitting, then apply regularization. Common practice is using a funnel shape (decreasing neurons per layer).', companyTags: ['Netflix'], difficulty: 'Medium' },
        { question: 'What optimizer is commonly used instead of vanilla Gradient Descent, and why?', answer: 'Adam (Adaptive Moment Estimation) is commonly used because it computes adaptive learning rates for each parameter using moving averages of both the gradients and their squared values, speeding up convergence.', companyTags: ['Google', 'Meta'], difficulty: 'Medium' },
        { question: 'What is Early Stopping?', answer: 'A regularization technique where training is halted when performance on a validation set starts to degrade, preventing the model from overfitting to the training set.', companyTags: ['Apple'], difficulty: 'Easy' },
        { question: 'What causes the "Exploding Gradient" problem and how do you fix it?', answer: 'It occurs when large weights or derivatives compound during backpropagation, causing numeric overflow. It\'s fixed using Gradient Clipping, careful initialization, and Batch Normalization.', companyTags: ['Meta', 'Microsoft'], difficulty: 'Hard' },
        { question: 'Explain L1 vs L2 Regularization in MLPs.', answer: 'L1 adds the absolute value of weights to the loss, promoting sparsity (zeroing out weights). L2 adds the squared value, keeping weights small and distributed. L2 (Weight Decay) is more common in MLPs.', companyTags: ['Amazon', 'Google'], difficulty: 'Medium' },
        { question: 'What is a "Dead ReLU" and how do you prevent it?', answer: 'A ReLU neuron that consistently outputs zero because its pre-activation is always negative. Prevent it by lowering the learning rate or using variants like Leaky ReLU or ELU.', companyTags: ['Netflix', 'Meta'], difficulty: 'Medium' },
        { question: 'Why is Cross-Entropy Loss preferred over Mean Squared Error (MSE) for classification?', answer: 'MSE with Sigmoid/Softmax leads to non-convex loss surfaces and very slow learning when predictions are confidently wrong (gradient approaches 0). Cross-entropy cancels out the activation derivative, ensuring strong gradients for wrong predictions.', companyTags: ['Google', 'Apple'], difficulty: 'Hard' },
        { question: 'Can an MLP learn a circular decision boundary?', answer: 'Yes, with at least one hidden layer and non-linear activations, an MLP can approximate closed geometric decision boundaries.', companyTags: ['Microsoft'], difficulty: 'Medium' },
        { question: 'What is the "Curse of Dimensionality" in the context of MLPs?', answer: 'As input dimensions increase, the volume of the feature space grows exponentially, requiring exponentially more data to train the massive number of weights effectively without overfitting.', companyTags: ['Amazon'], difficulty: 'Hard' },
        { question: 'What is the Universal Approximation Theorem?', answer: 'It states that a feed-forward network with a single hidden layer containing a finite number of neurons can approximate continuous functions on compact subsets of R^n, under mild assumptions on the activation function.', companyTags: ['Meta'], difficulty: 'Hard' },
        { question: 'Is the loss function of an MLP convex?', answer: 'No, due to hidden layers and non-linearities, the loss landscape is highly non-convex with many local minima, saddle points, and plateaus.', companyTags: ['Google'], difficulty: 'Medium' },
        { question: 'How do skip connections affect the loss landscape?', answer: 'They make the loss landscape smoother and alleviate the vanishing gradient problem, making it significantly easier for optimization algorithms to find good minima.', companyTags: ['Netflix', 'Apple'], difficulty: 'Hard' }
    ]
};
