export const neuralActivations = {
    "id": "neural-activations",
    "title": "Neural Activation Functions",
    "category": "Deep Learning",
    "description": "Explore ReLU, Sigmoid, Tanh, and GeLU mathematical responses.",
    "formula": "ReLU(z) = \\max(0,z)",
    "theory": `### Neural Activation Functions Theory

**What is it?**
Activation functions are mathematical equations that determine the output of a neural network node (neuron). In biological terms, they decide whether a neuron should "fire" or "activate" based on the signals it receives. In artificial neural networks, they sit at the end of a node's linear calculation and apply a transformation before passing the output to the next layer. 

**Why do we need it?**
A neural network without activation functions is mathematically equivalent to a single-layer linear regression model, regardless of how many hidden layers it has. This is because the composition of multiple linear functions is simply another linear function. The real world is rarely linear; relationships in images, text, and sound are highly complex and non-linear. Activation functions introduce this crucial **non-linearity** into the network, allowing it to learn and approximate complex functional mappings (as stated by the Universal Approximation Theorem).

**How does it work?**
A neuron first calculates a linear combination of its inputs. If the inputs are $x$, the weights are $w$, and the bias is $b$, it computes $z = \\sum w_i x_i + b$. Then, an activation function $f$ is applied to this raw sum $z$ to produce the final output (or "activation") of the neuron: $a = f(z)$. This output $a$ is then passed forward to the neurons in the subsequent layer.

**The Math Behind It**

1. **Sigmoid (Logistic) Function**:
Squashes any real-valued number into a range between 0 and 1.
$$\\sigma(z) = \frac{1}{1 + e^{-z}}$$
- **Derivative**: $\\sigma'(z) = \\sigma(z)(1 - \\sigma(z))$
- **Pros**: Clear probabilistic interpretation.
- **Cons**: Suffers heavily from vanishing gradients. Outputs are not zero-centered.

2. **Hyperbolic Tangent (Tanh)**:
A rescaled version of Sigmoid that maps inputs to a range between -1 and 1.
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
- **Derivative**: $\tanh'(z) = 1 - \tanh^2(z)$
- **Pros**: Zero-centered, which makes optimization easier compared to Sigmoid.
- **Cons**: Still suffers from vanishing gradients for extreme values of $z$.

3. **Rectified Linear Unit (ReLU)**:
The breakthrough activation function that enabled deep learning. It returns the input if positive, otherwise zero.
$$\text{ReLU}(z) = \\max(0, z)$$
- **Derivative**: $1$ if $z > 0$, else $0$.
- **Pros**: Solves the vanishing gradient problem for positive values. Extremely computationally efficient.
- **Cons**: The "Dying ReLU" problem, where neurons permanently output 0 and stop learning.

4. **Leaky ReLU**:
A fix for the Dying ReLU problem. It allows a small, non-zero gradient when the unit is inactive.
$$\text{Leaky ReLU}(z) = \\max(0.01z, z)$$
- **Derivative**: $1$ if $z > 0$, else $0.01$.

5. **Gaussian Error Linear Unit (GeLU)**:
The activation function of choice for modern Transformer architectures (like BERT and GPT). It weights inputs by their value, multiplied by the cumulative distribution function of the standard normal distribution.
$$\text{GeLU}(z) = z \\cdot \\Phi(z)$$
where $\\Phi(z) = \frac{1}{2} \\left[ 1 + \text{erf}\\left(\frac{z}{\\sqrt{2}}\right) \right]$.
- **Pros**: Smooth, non-monotonic curve that empirically performs better in deep NLP models.

**Worked Example**
Imagine a neuron in a hidden layer receives inputs $x = [2.0, -1.0, 3.0]$ with corresponding weights $w = [0.5, -2.0, 0.1]$ and a bias $b = -0.5$.
1. **Linear combination**: 
   $z = (2.0 \\cdot 0.5) + (-1.0 \\cdot -2.0) + (3.0 \\cdot 0.1) - 0.5$
   $z = 1.0 + 2.0 + 0.3 - 0.5 = 2.8$
2. **Apply Activation**:
   - If using **ReLU**: $a = \\max(0, 2.8) = 2.8$
   - If using **Sigmoid**: $a = \frac{1}{1 + e^{-2.8}} \\approx 0.942$
   - If using **Tanh**: $a = \tanh(2.8) \\approx 0.992$
The resulting value $a$ is passed to the next layer.

**Common Pitfalls**
1. **Vanishing Gradients with Deep Networks**: If you stack 20 layers of Sigmoid or Tanh, the gradients will diminish to zero as they propagate backward during backpropagation. The early layers will never update their weights.
2. **Exploding Gradients**: Unbounded activations like ReLU can theoretically contribute to exploding activations if weights are initialized poorly or learning rates are too high.
3. **Using ReLU in Output Layers**: ReLU only outputs positive numbers. If your target variable can be negative (e.g., predicting temperature in Celsius), ReLU will fail entirely in the output layer. Use a Linear activation instead.
4. **Dying ReLU**: If you set the learning rate too high, a massive gradient update might push a neuron's weights so far negative that its input $z$ is always negative for all training examples. That neuron will always output 0, and its gradient will always be 0. It is dead forever.

**When to Use vs Not Use**
- **ReLU**: Use it as the default choice for hidden layers in almost all Convolutional Neural Networks (CNNs) and Multi-Layer Perceptrons (MLPs). It is fast, reliable, and practically standard.
- **Sigmoid**: Only use it in the output layer of a binary classification problem (where you need a probability between 0 and 1). Never use it in hidden layers of deep networks.
- **Tanh**: Often used in the hidden layers of Recurrent Neural Networks (RNNs) and LSTMs. It is generally preferred over Sigmoid in hidden layers because it is zero-centered.
- **GeLU**: Use it if you are building or fine-tuning Transformer models. It is the modern standard for NLP architectures.
- **Softmax**: Use exclusively in the output layer of multi-class classification problems to generate a valid probability distribution over $N$ mutually exclusive classes.

**Key Takeaways**
- Activation functions allow neural networks to learn non-linear decision boundaries.
- The choice of activation function directly dictates how effectively gradients can flow backwards through the network.
- ReLU is the undisputed king of hidden layers for general-purpose deep learning, solving the vanishing gradient problem for positive values.
- Understanding the derivative of these functions is crucial, as the derivative is what actually updates the weights during backpropagation.

#### Python Implementation

\`\`\`python
import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))
def relu(x): return np.maximum(0, x)
def tanh(x): return np.tanh(x)

z = np.array([-2, -1, 0, 1, 2])
print(f"Sigmoid: {sigmoid(z)}")
print(f"ReLU:    {relu(z)}")
print(f"Tanh:    {tanh(z)}")
\`\`\`
`,
    "interactiveSummary": "The simulator renders four side-by-side activation function curves — Sigmoid, Tanh, ReLU, and GeLU — plotted over the full input range from −6 to +6, giving you an immediate visual feel for their shape, saturation zones, and output ranges. Drag the interactive input slider (z-value) and watch a live marker traverse each curve simultaneously, with the exact output value and its derivative displayed in real time; this makes it viscerally clear how the Sigmoid derivative collapses to near-zero at extreme values (vanishing gradient) while ReLU's derivative is a crisp constant 1 in the positive region. A dedicated 'Derivative Mode' overlays the gradient curves, revealing at a glance why deep stacking of Sigmoid layers starves backpropagation while ReLU preserves gradient flow. Toggle the 'Dying Neuron' scenario to simulate a neuron stuck at large negative z and observe how Leaky ReLU maintains a small non-zero gradient whereas standard ReLU flatlines, directly motivating the fix.",
    "simulatorId": "activations",
    "quiz": [
        {
            "id": "q1",
            "question": "What is the primary purpose of an activation function in a neural network?",
            "options": [
                "To normalize the input data",
                "To introduce non-linearity",
                "To calculate the loss",
                "To update the weights"
            ],
            "correctAnswer": "To introduce non-linearity",
            "explanation": "Activation functions introduce non-linearities, allowing networks to learn complex mappings. Without them, the network would just be a linear regression model."
        },
        {
            "id": "q2",
            "question": "Which activation function is most likely to suffer from the 'dying' problem?",
            "options": [
                "Sigmoid",
                "Tanh",
                "ReLU",
                "GeLU"
            ],
            "correctAnswer": "ReLU",
            "explanation": "ReLU outputs zero for all negative inputs, which means the gradient becomes zero, and the neuron can permanently stop updating ('die')."
        },
        {
            "id": "q3",
            "question": "What is the output range of the Tanh activation function?",
            "options": [
                "[0, 1]",
                "[-1, 1]",
                "[0, infinity)",
                "(-infinity, infinity)"
            ],
            "correctAnswer": "[-1, 1]",
            "explanation": "Tanh maps input values to a range between -1 and 1, making it zero-centered."
        },
        {
            "id": "q4",
            "question": "Which activation function is typically used in the output layer for a binary classification problem?",
            "options": [
                "ReLU",
                "Tanh",
                "Sigmoid",
                "Linear"
            ],
            "correctAnswer": "Sigmoid",
            "explanation": "Sigmoid squashes the output to a range between 0 and 1, which can be interpreted as a probability for binary classification."
        },
        {
            "id": "q5",
            "question": "What is the derivative of the ReLU function for positive inputs?",
            "options": [
                "0",
                "1",
                "z",
                "e^z"
            ],
            "correctAnswer": "1",
            "explanation": "For z > 0, ReLU(z) = z, so the derivative with respect to z is 1."
        },
        {
            "id": "q6",
            "question": "Why is vanishing gradient a problem with Sigmoid?",
            "options": [
                "It outputs negative values",
                "Its derivative approaches 0 for large positive/negative inputs",
                "It is non-differentiable at 0",
                "It is computationally expensive"
            ],
            "correctAnswer": "Its derivative approaches 0 for large positive/negative inputs",
            "explanation": "For very large or very small inputs, the Sigmoid curve is nearly flat, meaning the gradient is close to 0. This stops the network from learning."
        },
        {
            "id": "q7",
            "question": "How does Leaky ReLU address the dying ReLU problem?",
            "options": [
                "By outputting 1 for negative inputs",
                "By applying a small positive slope for negative inputs",
                "By using a Gaussian cumulative distribution",
                "By capping the maximum output"
            ],
            "correctAnswer": "By applying a small positive slope for negative inputs",
            "explanation": "Leaky ReLU allows a small, non-zero gradient when the input is negative (e.g., 0.01z), preventing neurons from dying completely."
        },
        {
            "id": "q8",
            "question": "Which activation function is most commonly associated with Transformer architectures like BERT and GPT?",
            "options": [
                "Sigmoid",
                "Tanh",
                "GeLU",
                "Step Function"
            ],
            "correctAnswer": "GeLU",
            "explanation": "GeLU (Gaussian Error Linear Unit) provides a smooth, non-monotonic nonlinearity and is widely used in modern Transformers."
        },
        {
            "id": "q9",
            "question": "If a neural network only uses linear activation functions, what kind of function does the entire network represent?",
            "options": [
                "A highly non-linear function",
                "A polynomial of degree N",
                "A single linear function",
                "A logarithmic function"
            ],
            "correctAnswer": "A single linear function",
            "explanation": "The composition of multiple linear functions is simply another linear function. The network would collapse into the equivalent of a single layer without non-linear activations."
        },
        {
            "id": "q10",
            "question": "What is the maximum value of the derivative of the Sigmoid function?",
            "options": [
                "1.0",
                "0.5",
                "0.25",
                "Infinity"
            ],
            "correctAnswer": "0.25",
            "explanation": "The derivative of Sigmoid is s(1-s). The maximum of this quadratic function occurs at s=0.5, yielding 0.5 * 0.5 = 0.25."
        }
    ],
    "interviewQuestions": [
        {
            "question": "Can you explain the vanishing gradient problem and how to mitigate it?",
            "answer": "The vanishing gradient problem occurs in deep networks using squashing activations like Sigmoid or Tanh, where gradients become exponentially small as they propagate backward. This halts learning in earlier layers. It\'s mitigated by using ReLU or its variants, employing Batch Normalization, using residual connections (ResNets), or careful weight initialization (e.g., He or Xavier).",
            "companyTags": [
                "Google",
                "Meta"
            ],
            "difficulty": "Medium"
        },
        {
            "question": "What is the 'Dying ReLU' problem and how can it be fixed?",
            "answer": "Dying ReLU happens when a large number of neurons output 0 and their gradients become 0 (since ReLU is 0 for negative inputs). This can be caused by high learning rates or large negative bias initializations. It\'s fixed by using Leaky ReLU, Parametric ReLU (PReLU), or GeLU, which have small non-zero gradients for negative inputs.",
            "companyTags": [
                "Amazon",
                "Apple"
            ],
            "difficulty": "Medium"
        },
        {
            "question": "Why is Tanh generally preferred over Sigmoid in hidden layers?",
            "answer": "Tanh is zero-centered (outputs range from -1 to 1), whereas Sigmoid is not (outputs 0 to 1). Zero-centered activations ensure that gradients are not restricted to being all positive or all negative, which prevents zig-zagging optimization paths and leads to faster convergence.",
            "companyTags": [
                "Netflix",
                "Meta"
            ],
            "difficulty": "Medium"
        },
        {
            "question": "What makes GeLU different from ReLU?",
            "answer": "GeLU smoothly weights inputs by their probability under a standard Gaussian distribution, whereas ReLU is a harsh threshold at 0. GeLU is differentiable everywhere (mostly) and can output small negative values, allowing for a non-monotonic curvature that empirically works better in NLP architectures.",
            "companyTags": [
                "OpenAI",
                "Google"
            ],
            "difficulty": "Advanced"
        },
        {
            "question": "Why do we even need non-linear activation functions?",
            "answer": "Without non-linear activations, a multi-layer neural network collapses mathematically into a single-layer linear model (since a linear combination of linear combinations is just a linear combination). Non-linearities allow the network to approximate complex, non-linear functions (Universal Approximation Theorem).",
            "companyTags": [
                "Amazon",
                "Microsoft"
            ],
            "difficulty": "Easy"
        },
        {
            "question": "How do you choose between Softmax and Sigmoid in the output layer?",
            "answer": "Use Sigmoid for binary classification or multi-label classification (where each class is independent). Use Softmax for multi-class classification (where classes are mutually exclusive) because it converts logits into a normalized probability distribution summing to 1.",
            "companyTags": [
                "Meta",
                "Google"
            ],
            "difficulty": "Easy"
        },
        {
            "question": "Is ReLU differentiable everywhere?",
            "answer": "No, ReLU is not differentiable exactly at z = 0. In practice, software implementations subgradient methods and arbitrarily assign the gradient at 0 to be either 0 or 1, which works perfectly fine during backpropagation.",
            "companyTags": [
                "Apple",
                "Netflix"
            ],
            "difficulty": "Easy"
        },
        {
            "question": "Explain Swish activation.",
            "answer": "Swish, proposed by Google, is defined as f(x) = x * sigmoid(beta * x). It is smooth, non-monotonic, and bounded below but unbounded above. Empirically, it tends to work better than ReLU in deeper networks as it preserves small negative values without harsh cutoffs.",
            "companyTags": [
                "Google",
                "Meta"
            ],
            "difficulty": "Advanced"
        },
        {
            "question": "What is the computational complexity of evaluating ReLU vs Sigmoid?",
            "answer": "ReLU is highly computationally efficient since it requires a simple max(0, x) operation, essentially a sign check. Sigmoid is more expensive as it requires computing the exponential function (e^-x) and division.",
            "companyTags": [
                "Microsoft",
                "Amazon"
            ],
            "difficulty": "Easy"
        },
        {
            "question": "Why is the derivative of an activation function important?",
            "answer": "The derivative is crucial for backpropagation. The network learns by updating weights in proportion to the gradient of the loss with respect to the weights. The chain rule dictates that we multiply the gradients of the activation functions. If the derivative is zero, learning stops.",
            "companyTags": [
                "Google",
                "Apple"
            ],
            "difficulty": "Medium"
        },
        {
            "question": "What is Parametric ReLU (PReLU)?",
            "answer": "PReLU is similar to Leaky ReLU, but instead of having a fixed small slope (like 0.01) for negative inputs, the slope parameter is learned jointly with the neural network's weights during training.",
            "companyTags": [
                "Meta",
                "Netflix"
            ],
            "difficulty": "Medium"
        },
        {
            "question": "Why not use linear activations in hidden layers and a non-linear activation only at the end?",
            "answer": "Using linear activations in hidden layers still collapses the entire sequence of hidden layers into a single linear transformation. The network would not have deep hierarchical feature extraction capabilities, fundamentally limiting its expressivity.",
            "companyTags": [
                "Amazon",
                "Google"
            ],
            "difficulty": "Medium"
        },
        {
            "question": "Can activation functions be asymmetric?",
            "answer": "Yes, ReLU is a prime example of a highly asymmetric activation function. Asymmetry can be beneficial because it breaks symmetry in weight updates and allows for sparse representations.",
            "companyTags": [
                "Microsoft"
            ],
            "difficulty": "Medium"
        },
        {
            "question": "How does batch normalization interact with activation functions?",
            "answer": "Batch normalization is usually applied before the activation function (though sometimes after). It normalizes the inputs to have zero mean and unit variance. For activations like Sigmoid or Tanh, this keeps the inputs in the active, linear regime (around 0) where gradients are largest, mitigating vanishing gradients.",
            "companyTags": [
                "Google",
                "Meta"
            ],
            "difficulty": "Advanced"
        },
        {
            "question": "What is the purpose of the ELU (Exponential Linear Unit) activation?",
            "answer": "ELU aims to combine the benefits of ReLU and zero-centered activations. It behaves like ReLU for positive inputs but smoothly exponentially decays to a negative constant (-alpha) for negative inputs. This pushes the mean of activations closer to zero, speeding up learning.",
            "companyTags": [
                "Apple",
                "Amazon"
            ],
            "difficulty": "Advanced"
        },
        {
            "question": "Are there any unbounded activation functions?",
            "answer": "Yes, ReLU, Leaky ReLU, and ELU are unbounded for positive inputs. This avoids vanishing gradients for large positive numbers, unlike Sigmoid and Tanh which are bounded on both sides.",
            "companyTags": [
                "Netflix",
                "Meta"
            ],
            "difficulty": "Easy"
        },
        {
            "question": "Why is the Softplus function sometimes used instead of ReLU?",
            "answer": "Softplus, defined as log(1 + exp(x)), is a smooth approximation of ReLU. Its derivative is the Sigmoid function. While smoother, it is computationally more expensive than ReLU and often performs similarly or slightly worse in deep networks.",
            "companyTags": [
                "Google",
                "Microsoft"
            ],
            "difficulty": "Advanced"
        },
        {
            "question": "In what scenario would you use a linear activation function?",
            "answer": "Linear activation is used in the output layer of a neural network when the task is regression, where the target variable can be any continuous real number.",
            "companyTags": [
                "Amazon",
                "Apple"
            ],
            "difficulty": "Easy"
        },
        {
            "question": "What is the issue with non-zero-centered activations?",
            "answer": "When activations are strictly positive (like Sigmoid), all gradients passed to the weights of a given neuron will have the same sign. This forces the weight updates to occur in a zig-zag manner, leading to inefficient and slow optimization dynamics.",
            "companyTags": [
                "Meta",
                "Google"
            ],
            "difficulty": "Medium"
        },
        {
            "question": "Explain the concept of 'sparse activation' in relation to ReLU.",
            "answer": "Because ReLU outputs exactly zero for negative inputs, many neurons in a network will have a zero output for a given input sample. This creates a sparse representation, which can be computationally efficient and theoretically provides better disentangling of features.",
            "companyTags": [
                "Microsoft",
                "OpenAI"
            ],
            "difficulty": "Advanced"
        }
    ],
    "coding": {
        "tutorial": {
            "title": "Exponential Linear Unit (ELU)",
            "description": "Implement ELU: z if z > 0 else alpha * (exp(z) - 1). Scikit-learn equivalent typically uses `MLPClassifier` with standard activations (relu, tanh, logistic). Note: sklearn's MLP does not natively support ELU, but PyTorch and TensorFlow do.",
            "pseudoCode": "IF z > 0 THEN return z ELSE return alpha * (exp(z) - 1)",
            "starterCode": "import numpy as np\ndef elu(z, alpha=1.0):\n    # TODO: implement ELU\n    return z\n\nprint(np.round(elu(-1.0), 3))",
            "expectedOutput": "-0.632",
            "solution": "import numpy as np\ndef elu(z, alpha=1.0):\n    return float(z if z > 0 else alpha * (np.exp(z) - 1.0))\n\nprint(np.round(elu(-1.0), 3))",
            "hints": [
                "Check the sign of z, using np.exp for negative values."
            ],
            "testKeywords": [
                "np.exp"
            ]
        },
        "project": {
            "title": "Activation Function Impact on MNIST-like Classification",
            "description": "Compare the effect of different activation functions ('relu', 'tanh', 'logistic') on an MLPClassifier trained on sklearn's digits dataset (MNIST-like 8×8 images). Report test accuracy and training loss for each activation to demonstrate how activation choice affects classification performance.",
            "pseudoCode": "# Activation Functions on Digits Dataset\nIMPORT MLPClassifier, load_digits, train_test_split, accuracy_score, StandardScaler, Pipeline\n\nLOAD digits dataset (1797 samples, 64 features, 10 classes)\nSPLIT into train (80%) and test (20%) with random_state=42\n\nFOR activation IN ['relu', 'tanh', 'logistic']:\n    CREATE Pipeline([\n        StandardScaler(),\n        MLPClassifier(hidden_layer_sizes=(64, 32),\n                      activation=activation,\n                      max_iter=300,\n                      random_state=42)\n    ])\n    FIT on X_train, y_train\n    COMPUTE test_acc = accuracy_score(y_test, pipe.predict(X_test))\n    PRINT activation, test_acc rounded to 3, loss_ rounded to 4",
            "starterCode": "import numpy as np\nfrom sklearn.neural_network import MLPClassifier\nfrom sklearn.datasets import load_digits\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.metrics import accuracy_score\n\n# MNIST-like dataset: 1797 samples of 8x8 handwritten digits (10 classes)\ndigits = load_digits()\nX, y = digits.data, digits.target\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\nactivations = ['relu', 'tanh', 'logistic']\n\nfor activation in activations:\n    # TODO: Create a Pipeline with StandardScaler and MLPClassifier(\n    #       hidden_layer_sizes=(64, 32), activation=activation,\n    #       max_iter=300, random_state=42)\n    # TODO: Fit on X_train, y_train\n    # TODO: Compute test accuracy and retrieve pipe[-1].loss_\n    # TODO: Print: f\"Activation: {activation} | Test Acc: {acc:.3f} | Loss: {loss:.4f}\"\n    pass",
            "expectedOutput": "Activation: relu | Test Acc: 0.978 | Loss: 0.0121\nActivation: tanh | Test Acc: 0.975 | Loss: 0.0180\nActivation: logistic | Test Acc: 0.964 | Loss: 0.0301",
            "solution": "import numpy as np\nfrom sklearn.neural_network import MLPClassifier\nfrom sklearn.datasets import load_digits\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.metrics import accuracy_score\n\ndigits = load_digits()\nX, y = digits.data, digits.target\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\nactivations = ['relu', 'tanh', 'logistic']\n\nfor activation in activations:\n    pipe = Pipeline([\n        ('scaler', StandardScaler()),\n        ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), activation=activation, max_iter=300, random_state=42))\n    ])\n    pipe.fit(X_train, y_train)\n    acc = accuracy_score(y_test, pipe.predict(X_test))\n    loss = pipe.named_steps['mlp'].loss_\n    print(f\"Activation: {activation} | Test Acc: {acc:.3f} | Loss: {loss:.4f}\")",
            "hints": [
                "Loop over activations = ['relu', 'tanh', 'logistic']",
                "Use Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(...))])",
                "Access loss via pipe.named_steps['mlp'].loss_ after fitting"
            ],
            "testKeywords": [
                "MLPClassifier",
                "load_digits",
                "Pipeline",
                "StandardScaler",
                "accuracy_score",
                "loss_"
            ]
        },
        "assignment": {
            "title": "Dying ReLU Detection and Leaky ReLU Solution",
            "description": "Simulate the Dying ReLU problem by initializing neuron weights so that pre-activations are always negative, then compare the gradient flow of standard ReLU (zero gradient) vs. Leaky ReLU (non-zero gradient). Demonstrate that Leaky ReLU rescues dead neurons by computing the gradient magnitude for both activation functions.",
            "pseudoCode": "# Dying ReLU Detection & Leaky ReLU Fix\nDEFINE relu(z): return max(0, z)\nDEFINE leaky_relu(z, alpha=0.01): return z if z > 0 else alpha * z\nDEFINE relu_grad(z): return 1.0 if z > 0 else 0.0\nDEFINE leaky_relu_grad(z, alpha=0.01): return 1.0 if z > 0 else alpha\n\n# Simulate neuron stuck in negative region (dying ReLU scenario)\nDEFINE pre_activations = [-3.2, -1.8, -0.5, -4.1, -2.2]\n\nCOMPUTE relu_grads    = [relu_grad(z)       for z in pre_activations]\nCOMPUTE leaky_grads   = [leaky_relu_grad(z) for z in pre_activations]\nCOMPUTE relu_outputs  = [relu(z)            for z in pre_activations]\nCOMPUTE leaky_outputs = [leaky_relu(z)      for z in pre_activations]\n\nPRINT Pre-activations\nPRINT ReLU outputs (all zeros showing dead neurons)\nPRINT ReLU gradients (all zeros = learning stopped)\nPRINT Leaky ReLU outputs (small negatives, neuron alive)\nPRINT Leaky ReLU gradients (all 0.01 = gradient flows)",
            "starterCode": "import numpy as np\n\n# Activation functions\ndef relu(z):\n    return max(0.0, z)\n\ndef leaky_relu(z, alpha=0.01):\n    # TODO: Return z if z > 0, else alpha * z\n    pass\n\n# Gradient (derivative) functions\ndef relu_grad(z):\n    return 1.0 if z > 0 else 0.0\n\ndef leaky_relu_grad(z, alpha=0.01):\n    # TODO: Return 1.0 if z > 0, else alpha\n    pass\n\n# Simulate dead neuron scenario: all pre-activations are negative\npre_activations = [-3.2, -1.8, -0.5, -4.1, -2.2]\n\nrelu_outputs  = [relu(z)            for z in pre_activations]\nleaky_outputs = [leaky_relu(z)      for z in pre_activations]\nrelu_grads    = [relu_grad(z)       for z in pre_activations]\nleaky_grads   = [leaky_relu_grad(z) for z in pre_activations]\n\nprint(\"Pre-activations:      \", pre_activations)\nprint(\"ReLU outputs:         \", relu_outputs)\nprint(\"ReLU gradients:       \", relu_grads)\nprint(\"Leaky ReLU outputs:   \", np.round(leaky_outputs, 3).tolist())\nprint(\"Leaky ReLU gradients: \", leaky_grads)",
            "expectedOutput": "Pre-activations:       [-3.2, -1.8, -0.5, -4.1, -2.2]\nReLU outputs:          [0.0, 0.0, 0.0, 0.0, 0.0]\nReLU gradients:        [0.0, 0.0, 0.0, 0.0, 0.0]\nLeaky ReLU outputs:    [-0.032, -0.018, -0.005, -0.041, -0.022]\nLeaky ReLU gradients:  [0.01, 0.01, 0.01, 0.01, 0.01]",
            "solution": "import numpy as np\n\ndef relu(z):\n    return max(0.0, z)\n\ndef leaky_relu(z, alpha=0.01):\n    return z if z > 0 else alpha * z\n\ndef relu_grad(z):\n    return 1.0 if z > 0 else 0.0\n\ndef leaky_relu_grad(z, alpha=0.01):\n    return 1.0 if z > 0 else alpha\n\npre_activations = [-3.2, -1.8, -0.5, -4.1, -2.2]\n\nrelu_outputs  = [relu(z)            for z in pre_activations]\nleaky_outputs = [leaky_relu(z)      for z in pre_activations]\nrelu_grads    = [relu_grad(z)       for z in pre_activations]\nleaky_grads   = [leaky_relu_grad(z) for z in pre_activations]\n\nprint(\"Pre-activations:      \", pre_activations)\nprint(\"ReLU outputs:         \", relu_outputs)\nprint(\"ReLU gradients:       \", relu_grads)\nprint(\"Leaky ReLU outputs:   \", np.round(leaky_outputs, 3).tolist())\nprint(\"Leaky ReLU gradients: \", leaky_grads)",
            "hints": [
                "leaky_relu(z, alpha=0.01): return z if z > 0 else alpha * z",
                "leaky_relu_grad(z, alpha=0.01): return 1.0 if z > 0 else alpha",
                "ReLU gradients will all be 0.0 for negative pre-activations; Leaky ReLU gradients will all be 0.01"
            ],
            "testKeywords": [
                "leaky_relu",
                "leaky_relu_grad",
                "relu_grad",
                "alpha"
            ]
        }
    }
};
