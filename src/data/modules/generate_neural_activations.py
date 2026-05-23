import json

data = {
    "id": "neural-activations",
    "title": "Neural Activation Functions",
    "category": "Deep Learning",
    "description": "Explore ReLU, Sigmoid, Tanh, and GeLU mathematical responses.",
    "formula": "ReLU(z) = \\max(0,z)",
    "theory": r"""### Neural Activation Functions Theory

**What is it?**
Activation functions are mathematical equations attached to each neuron in a neural network. They determine whether a neuron should be activated or not, based on whether the neuron's input is relevant for the model's prediction. Essentially, they transform the summed weighted input from the node into the activation of the node or output for that input.

**Why do we need it?**
Without activation functions, a neural network, no matter how many layers it has, would behave just like a single-layer perceptron. This is because summing linear functions gives another linear function. The real world is highly non-linear, and we need neural networks to learn complex, non-linear mappings from inputs to outputs (like image recognition, language translation, etc.). Activation functions introduce this crucial non-linearity.

**How does it work?**
A neuron calculates a weighted sum of its inputs and adds a bias: $z = \sum w_i x_i + b$. The activation function $f$ is then applied to this sum to produce the neuron's output: $a = f(z)$. The output $a$ then becomes the input to the next layer. Different activation functions have different shapes and properties, affecting how the network learns during backpropagation.

**The Math Behind It**
Here are the formulas for some of the most common activation functions:

1. **Sigmoid**: Maps inputs to a range between 0 and 1.
   $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
   Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

2. **Hyperbolic Tangent (Tanh)**: Maps inputs to a range between -1 and 1.
   $$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
   Derivative: $\tanh'(z) = 1 - \tanh^2(z)$

3. **Rectified Linear Unit (ReLU)**: Outputs the input directly if it is positive, otherwise, it outputs zero.
   $$\text{ReLU}(z) = \max(0, z)$$
   Derivative: $\text{ReLU}'(z) = 1$ if $z > 0$, else $0$

4. **Gaussian Error Linear Unit (GeLU)**: Weights inputs by their value, rather than gating inputs by their sign as in ReLU.
   $$\text{GeLU}(z) = z \cdot \Phi(z)$$
   where $\Phi(z)$ is the cumulative distribution function of the standard normal distribution.

**Worked Example**
Imagine a neuron receives inputs $x = [2, -1, 3]$ with weights $w = [0.5, -2, 0.1]$ and bias $b = -0.5$.
1. Calculate the linear combination $z$:
   $z = (2 \cdot 0.5) + (-1 \cdot -2) + (3 \cdot 0.1) - 0.5 = 1.0 + 2.0 + 0.3 - 0.5 = 2.8$
2. Apply an activation function:
   - **ReLU**: $\max(0, 2.8) = 2.8$
   - **Sigmoid**: $1 / (1 + e^{-2.8}) \approx 0.942$
   - **Tanh**: $(e^{2.8} - e^{-2.8}) / (e^{2.8} + e^{-2.8}) \approx 0.992$

**Common Pitfalls**
- **Vanishing Gradients**: Sigmoid and Tanh functions flatten out for very high or very low values of $z$. The derivative approaches zero, meaning the network stops learning.
- **Dying ReLU**: If a large number of neurons output 0 (because their $z < 0$), they "die" and stop updating their weights during backpropagation because the gradient of ReLU is 0 for negative inputs.
- **Not Zero-Centered**: Sigmoid outputs are strictly positive, which can lead to zig-zagging dynamics in gradient updates.

**When to Use vs Not Use**
- **ReLU**: Use it as the default for hidden layers. It's fast and mitigates vanishing gradients. Don't use it for output layers (unless predicting positive continuous values).
- **Sigmoid**: Use for binary classification output layers. Avoid in hidden layers due to vanishing gradients.
- **Tanh**: Sometimes preferred over Sigmoid in hidden layers because it is zero-centered, but still suffers from vanishing gradients. Use in RNNs.
- **GeLU**: Often used in state-of-the-art Transformer models (like BERT, GPT) as a smoother alternative to ReLU.

**Key Takeaways**
- Activation functions are essential for learning non-linear patterns.
- The choice of activation function significantly impacts training speed and final model performance.
- ReLU and its variants are the go-to choices for modern deep learning architectures.""",
    "interactiveSummary": "The simulator allows you to input various continuous values and visualize the exact output mapped by Sigmoid, Tanh, ReLU, and GeLU. You can adjust inputs dynamically and see how the functions squash or clip values.",
    "simulatorId": "activations",
    "quiz": [
        {"id": "q1", "question": "What is the primary purpose of an activation function in a neural network?", "options": ["To normalize the input data", "To introduce non-linearity", "To calculate the loss", "To update the weights"], "correctAnswer": "To introduce non-linearity", "explanation": "Activation functions introduce non-linearities, allowing networks to learn complex mappings. Without them, the network would just be a linear regression model."},
        {"id": "q2", "question": "Which activation function is most likely to suffer from the 'dying' problem?", "options": ["Sigmoid", "Tanh", "ReLU", "GeLU"], "correctAnswer": "ReLU", "explanation": "ReLU outputs zero for all negative inputs, which means the gradient becomes zero, and the neuron can permanently stop updating ('die')."},
        {"id": "q3", "question": "What is the output range of the Tanh activation function?", "options": ["[0, 1]", "[-1, 1]", "[0, infinity)", "(-infinity, infinity)"], "correctAnswer": "[-1, 1]", "explanation": "Tanh maps input values to a range between -1 and 1, making it zero-centered."},
        {"id": "q4", "question": "Which activation function is typically used in the output layer for a binary classification problem?", "options": ["ReLU", "Tanh", "Sigmoid", "Linear"], "correctAnswer": "Sigmoid", "explanation": "Sigmoid squashes the output to a range between 0 and 1, which can be interpreted as a probability for binary classification."},
        {"id": "q5", "question": "What is the derivative of the ReLU function for positive inputs?", "options": ["0", "1", "z", "e^z"], "correctAnswer": "1", "explanation": "For z > 0, ReLU(z) = z, so the derivative with respect to z is 1."},
        {"id": "q6", "question": "Why is vanishing gradient a problem with Sigmoid?", "options": ["It outputs negative values", "Its derivative approaches 0 for large positive/negative inputs", "It is non-differentiable at 0", "It is computationally expensive"], "correctAnswer": "Its derivative approaches 0 for large positive/negative inputs", "explanation": "For very large or very small inputs, the Sigmoid curve is nearly flat, meaning the gradient is close to 0. This stops the network from learning."},
        {"id": "q7", "question": "How does Leaky ReLU address the dying ReLU problem?", "options": ["By outputting 1 for negative inputs", "By applying a small positive slope for negative inputs", "By using a Gaussian cumulative distribution", "By capping the maximum output"], "correctAnswer": "By applying a small positive slope for negative inputs", "explanation": "Leaky ReLU allows a small, non-zero gradient when the input is negative (e.g., 0.01z), preventing neurons from dying completely."},
        {"id": "q8", "question": "Which activation function is most commonly associated with Transformer architectures like BERT and GPT?", "options": ["Sigmoid", "Tanh", "GeLU", "Step Function"], "correctAnswer": "GeLU", "explanation": "GeLU (Gaussian Error Linear Unit) provides a smooth, non-monotonic nonlinearity and is widely used in modern Transformers."},
        {"id": "q9", "question": "If a neural network only uses linear activation functions, what kind of function does the entire network represent?", "options": ["A highly non-linear function", "A polynomial of degree N", "A single linear function", "A logarithmic function"], "correctAnswer": "A single linear function", "explanation": "The composition of multiple linear functions is simply another linear function. The network would collapse into the equivalent of a single layer without non-linear activations."},
        {"id": "q10", "question": "What is the maximum value of the derivative of the Sigmoid function?", "options": ["1.0", "0.5", "0.25", "Infinity"], "correctAnswer": "0.25", "explanation": "The derivative of Sigmoid is s(1-s). The maximum of this quadratic function occurs at s=0.5, yielding 0.5 * 0.5 = 0.25."}
    ],
    "interviewQuestions": [
        {"question": "Can you explain the vanishing gradient problem and how to mitigate it?", "answer": "The vanishing gradient problem occurs in deep networks using squashing activations like Sigmoid or Tanh, where gradients become exponentially small as they propagate backward. This halts learning in earlier layers. It's mitigated by using ReLU or its variants, employing Batch Normalization, using residual connections (ResNets), or careful weight initialization (e.g., He or Xavier).", "companyTags": ["Google", "Meta"], "difficulty": "Medium"},
        {"question": "What is the 'Dying ReLU' problem and how can it be fixed?", "answer": "Dying ReLU happens when a large number of neurons output 0 and their gradients become 0 (since ReLU is 0 for negative inputs). This can be caused by high learning rates or large negative bias initializations. It's fixed by using Leaky ReLU, Parametric ReLU (PReLU), or GeLU, which have small non-zero gradients for negative inputs.", "companyTags": ["Amazon", "Apple"], "difficulty": "Medium"},
        {"question": "Why is Tanh generally preferred over Sigmoid in hidden layers?", "answer": "Tanh is zero-centered (outputs range from -1 to 1), whereas Sigmoid is not (outputs 0 to 1). Zero-centered activations ensure that gradients are not restricted to being all positive or all negative, which prevents zig-zagging optimization paths and leads to faster convergence.", "companyTags": ["Netflix", "Meta"], "difficulty": "Medium"},
        {"question": "What makes GeLU different from ReLU?", "answer": "GeLU smoothly weights inputs by their probability under a standard Gaussian distribution, whereas ReLU is a harsh threshold at 0. GeLU is differentiable everywhere (mostly) and can output small negative values, allowing for a non-monotonic curvature that empirically works better in NLP architectures.", "companyTags": ["OpenAI", "Google"], "difficulty": "Advanced"},
        {"question": "Why do we even need non-linear activation functions?", "answer": "Without non-linear activations, a multi-layer neural network collapses mathematically into a single-layer linear model (since a linear combination of linear combinations is just a linear combination). Non-linearities allow the network to approximate complex, non-linear functions (Universal Approximation Theorem).", "companyTags": ["Amazon", "Microsoft"], "difficulty": "Easy"},
        {"question": "How do you choose between Softmax and Sigmoid in the output layer?", "answer": "Use Sigmoid for binary classification or multi-label classification (where each class is independent). Use Softmax for multi-class classification (where classes are mutually exclusive) because it converts logits into a normalized probability distribution summing to 1.", "companyTags": ["Meta", "Google"], "difficulty": "Easy"},
        {"question": "Is ReLU differentiable everywhere?", "answer": "No, ReLU is not differentiable exactly at z = 0. In practice, software implementations subgradient methods and arbitrarily assign the gradient at 0 to be either 0 or 1, which works perfectly fine during backpropagation.", "companyTags": ["Apple", "Netflix"], "difficulty": "Easy"},
        {"question": "Explain Swish activation.", "answer": "Swish, proposed by Google, is defined as f(x) = x * sigmoid(beta * x). It is smooth, non-monotonic, and bounded below but unbounded above. Empirically, it tends to work better than ReLU in deeper networks as it preserves small negative values without harsh cutoffs.", "companyTags": ["Google", "Meta"], "difficulty": "Advanced"},
        {"question": "What is the computational complexity of evaluating ReLU vs Sigmoid?", "answer": "ReLU is highly computationally efficient since it requires a simple max(0, x) operation, essentially a sign check. Sigmoid is more expensive as it requires computing the exponential function (e^-x) and division.", "companyTags": ["Microsoft", "Amazon"], "difficulty": "Easy"},
        {"question": "Why is the derivative of an activation function important?", "answer": "The derivative is crucial for backpropagation. The network learns by updating weights in proportion to the gradient of the loss with respect to the weights. The chain rule dictates that we multiply the gradients of the activation functions. If the derivative is zero, learning stops.", "companyTags": ["Google", "Apple"], "difficulty": "Medium"},
        {"question": "What is Parametric ReLU (PReLU)?", "answer": "PReLU is similar to Leaky ReLU, but instead of having a fixed small slope (like 0.01) for negative inputs, the slope parameter is learned jointly with the neural network's weights during training.", "companyTags": ["Meta", "Netflix"], "difficulty": "Medium"},
        {"question": "Why not use linear activations in hidden layers and a non-linear activation only at the end?", "answer": "Using linear activations in hidden layers still collapses the entire sequence of hidden layers into a single linear transformation. The network would not have deep hierarchical feature extraction capabilities, fundamentally limiting its expressivity.", "companyTags": ["Amazon", "Google"], "difficulty": "Medium"},
        {"question": "Can activation functions be asymmetric?", "answer": "Yes, ReLU is a prime example of a highly asymmetric activation function. Asymmetry can be beneficial because it breaks symmetry in weight updates and allows for sparse representations.", "companyTags": ["Microsoft"], "difficulty": "Medium"},
        {"question": "How does batch normalization interact with activation functions?", "answer": "Batch normalization is usually applied before the activation function (though sometimes after). It normalizes the inputs to have zero mean and unit variance. For activations like Sigmoid or Tanh, this keeps the inputs in the active, linear regime (around 0) where gradients are largest, mitigating vanishing gradients.", "companyTags": ["Google", "Meta"], "difficulty": "Advanced"},
        {"question": "What is the purpose of the ELU (Exponential Linear Unit) activation?", "answer": "ELU aims to combine the benefits of ReLU and zero-centered activations. It behaves like ReLU for positive inputs but smoothly exponentially decays to a negative constant (-alpha) for negative inputs. This pushes the mean of activations closer to zero, speeding up learning.", "companyTags": ["Apple", "Amazon"], "difficulty": "Advanced"},
        {"question": "Are there any unbounded activation functions?", "answer": "Yes, ReLU, Leaky ReLU, and ELU are unbounded for positive inputs. This avoids vanishing gradients for large positive numbers, unlike Sigmoid and Tanh which are bounded on both sides.", "companyTags": ["Netflix", "Meta"], "difficulty": "Easy"},
        {"question": "Why is the Softplus function sometimes used instead of ReLU?", "answer": "Softplus, defined as log(1 + exp(x)), is a smooth approximation of ReLU. Its derivative is the Sigmoid function. While smoother, it is computationally more expensive than ReLU and often performs similarly or slightly worse in deep networks.", "companyTags": ["Google", "Microsoft"], "difficulty": "Advanced"},
        {"question": "In what scenario would you use a linear activation function?", "answer": "Linear activation is used in the output layer of a neural network when the task is regression, where the target variable can be any continuous real number.", "companyTags": ["Amazon", "Apple"], "difficulty": "Easy"},
        {"question": "What is the issue with non-zero-centered activations?", "answer": "When activations are strictly positive (like Sigmoid), all gradients passed to the weights of a given neuron will have the same sign. This forces the weight updates to occur in a zig-zag manner, leading to inefficient and slow optimization dynamics.", "companyTags": ["Meta", "Google"], "difficulty": "Medium"},
        {"question": "Explain the concept of 'sparse activation' in relation to ReLU.", "answer": "Because ReLU outputs exactly zero for negative inputs, many neurons in a network will have a zero output for a given input sample. This creates a sparse representation, which can be computationally efficient and theoretically provides better disentangling of features.", "companyTags": ["Microsoft", "OpenAI"], "difficulty": "Advanced"}
    ],
    "coding": {
        "tutorial": {
            "title": "Exponential Linear Unit (ELU)",
            "description": "Implement ELU: z if z > 0 else alpha * (exp(z) - 1). Scikit-learn equivalent typically uses `MLPClassifier` with standard activations (relu, tanh, logistic). Note: sklearn's MLP does not natively support ELU, but PyTorch and TensorFlow do.",
            "pseudoCode": "IF z > 0 THEN return z ELSE return alpha * (exp(z) - 1)",
            "starterCode": "import numpy as np\ndef elu(z, alpha=1.0):\n    # TODO: implement ELU\n    return z\n\nprint(np.round(elu(-1.0), 3))",
            "expectedOutput": "-0.632",
            "solution": "import numpy as np\ndef elu(z, alpha=1.0):\n    return float(z if z > 0 else alpha * (np.exp(z) - 1.0))\n\nprint(np.round(elu(-1.0), 3))",
            "hints": ["Check the sign of z, using np.exp for negative values."],
            "testKeywords": ["np.exp"]
        },
        "project": {
            "title": "Build a Simple Neural Network with Scikit-learn",
            "description": "Use `sklearn.neural_network.MLPClassifier` in a Pipeline to train a model with a specific activation function (ReLU) on a toy dataset.",
            "pseudoCode": "CREATE Pipeline with StandardScaler and MLPClassifier(activation='relu'). FIT Pipeline. PREDICT classes.",
            "starterCode": "import numpy as np\nfrom sklearn.neural_network import MLPClassifier\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\n\n# Toy dataset: XOR-like\nX = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])\ny = np.array([0, 0, 1, 1])\n\n# TODO: Create a Pipeline with StandardScaler and MLPClassifier(activation='relu', hidden_layer_sizes=(4,), max_iter=1000, random_state=42)\n# TODO: Fit the pipeline on X, y\n# TODO: Print the predictions for X\n",
            "expectedOutput": "[0 0 1 1]",
            "solution": "import numpy as np\nfrom sklearn.neural_network import MLPClassifier\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\n\nX = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])\ny = np.array([0, 0, 1, 1])\n\npipe = Pipeline([\n    ('scaler', StandardScaler()),\n    ('mlp', MLPClassifier(activation='relu', hidden_layer_sizes=(4,), max_iter=1000, random_state=42))\n])\n\npipe.fit(X, y)\npreds = pipe.predict(X)\nprint(preds)",
            "hints": ["Use Pipeline() with two steps: 'scaler' and 'mlp'.", "Don't forget to call fit and predict."],
            "testKeywords": ["Pipeline", "StandardScaler", "MLPClassifier", "fit", "predict"]
        },
        "assignment": {
            "title": "Evaluate Activations with Cross-Validation",
            "description": "Evaluate how different activations (e.g. 'tanh' vs 'relu') perform on a small classification task using `cross_val_score`.",
            "pseudoCode": "CREATE MLPClassifier. RUN cross_val_score for 'tanh'. PRINT mean score.",
            "starterCode": "import numpy as np\nfrom sklearn.neural_network import MLPClassifier\nfrom sklearn.model_selection import cross_val_score\n\n# Toy dataset\nX = np.random.RandomState(42).randn(100, 5)\ny = (X[:, 0] * X[:, 1] > 0).astype(int)\n\n# TODO: Instantiate MLPClassifier with activation='tanh', hidden_layer_sizes=(5,), max_iter=500, random_state=42\n# TODO: Use cross_val_score with cv=3\n# TODO: Print the mean score rounded to 2 decimal places\n",
            "expectedOutput": "0.96",
            "solution": "import numpy as np\nfrom sklearn.neural_network import MLPClassifier\nfrom sklearn.model_selection import cross_val_score\n\nX = np.random.RandomState(42).randn(100, 5)\ny = (X[:, 0] * X[:, 1] > 0).astype(int)\n\nmlp = MLPClassifier(activation='tanh', hidden_layer_sizes=(5,), max_iter=500, random_state=42)\nscores = cross_val_score(mlp, X, y, cv=3)\nprint(f\"{scores.mean():.2f}\")",
            "hints": ["Use cross_val_score(estimator, X, y, cv=3).", "Use the .mean() method on the result array."],
            "testKeywords": ["cross_val_score", "mean", "MLPClassifier"]
        }
    }
}

ts_file_content = f"""import type {{ MLModule }} from '../../types';

export const neuralActivations: MLModule = {json.dumps(data, indent=2)};
"""
with open("/home/legion/fastapi/machine_learning_tutorial/src/data/modules/neuralActivations.ts", "w") as f:
    f.write(ts_file_content)

print("neuralActivations.ts generated")
