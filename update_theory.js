const fs = require('fs');

const optimTheory = `### Optimization Algorithms Theory

**What is it?**
Optimization algorithms are the foundational mathematical engines that power the training of machine learning and deep learning models. In essence, whenever we talk about a model "learning," we are actually referring to the process of optimization. These algorithms systematically adjust the internal parameters (weights and biases) of a model to minimize a predefined error metric, known as the loss function or cost function. 

You can think of a neural network's loss landscape as a vast, incredibly complex, high-dimensional mountain range. The goal of the optimization algorithm is to navigate this treacherous terrain—starting from a random peak (initialization)—and find its way down to the lowest possible valley (the global minimum, or at least a very good local minimum), where the model's predictions perfectly match the ground truth.

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
$$\\theta_{t+1} = \\theta_t - \\eta \\cdot g_t$$
Where:
- $\\theta_t$ represents the model parameters at time step $t$.
- $\\eta$ (eta) is the learning rate (step size).
- $g_t = \\nabla_{\\theta} J(\\theta; x^{(i)}, y^{(i)})$ is the gradient of the loss function $J$ with respect to the parameters for a specific mini-batch.

2. **SGD with Momentum**:
Momentum addresses SGD's tendency to oscillate in ravines by adding a fraction of the previous update to the current one.
$$v_t = \\gamma v_{t-1} + \\eta g_t$$
$$\\theta_{t+1} = \\theta_t - v_t$$
Where $\\gamma$ is the momentum term (usually 0.9), and $v_t$ is the velocity vector.

3. **Nesterov Accelerated Gradient (NAG)**:
A slight tweak to momentum that calculates the gradient *after* applying the momentum, providing a "lookahead" correction.
$$v_t = \\gamma v_{t-1} + \\eta \\nabla_{\\theta} J(\\theta_t - \\gamma v_{t-1})$$
$$\\theta_{t+1} = \\theta_t - v_t$$

4. **RMSprop (Root Mean Square Propagation)**:
Adapts the learning rate by dividing the gradient by a moving average of its recent magnitude, helping to navigate varying topologies.
$$E[g^2]_t = \\beta E[g^2]_{t-1} + (1 - \\beta) g_t^2$$
$$\\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{E[g^2]_t + \\epsilon}} g_t$$
Where $\\beta$ is the decay rate, and $\\epsilon$ is a small smoothing term to prevent division by zero.

5. **Adam (Adaptive Moment Estimation)**:
The industry standard. It combines the benefits of both Momentum (first moment) and RMSprop (second moment).
- Calculate first moment (mean): $m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t$
- Calculate second moment (uncentered variance): $v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2$
- Apply bias correction: $\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}$ and $\\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}$
- Update parameters: $\\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t$

**Worked Example**
Let's trace one step of basic SGD for a single weight.
- Current weight: $\\theta_0 = 0.8$
- Learning rate: $\\eta = 0.05$
- We run a forward pass and calculate the loss.
- We run backpropagation and find the gradient with respect to this weight: $g_0 = 2.4$ (The error is increasing sharply in the positive direction).
- Update calculation: 
  $$\\theta_1 = \\theta_0 - (\\eta \\times g_0)$$
  $$\\theta_1 = 0.8 - (0.05 \\times 2.4)$$
  $$\\theta_1 = 0.8 - 0.12 = 0.68$$
The weight has been updated to $0.68$, moving in the opposite direction of the positive gradient to reduce the overall loss.

**Common Pitfalls**
1. **Poor Learning Rate Selection**: If $\\eta$ is too high, the model will take massive steps, overshooting the minimum and diverging entirely (loss goes to NaN). If $\\eta$ is too low, training will take forever and likely get stuck in a shallow local minimum.
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
- There is no single "best" optimizer; the choice represents a trade-off between speed of convergence and the quality of final generalization.`;

const replaceTheory = (filePath, newTheory) => {
    let content = fs.readFileSync(filePath, 'utf8');
    // Using string replacement. Find the theory property
    // For optimizationAlgorithms.ts, it might use double quotes or backticks.
    
    // Simple regex to match theory field regardless of quotes or backticks, until interactiveSummary
    const regex = /"theory":\s*"(?:[^"\\]|\\.)*",/s;
    if (regex.test(content)) {
        content = content.replace(regex, 'theory: `' + newTheory.replace(/`/g, '\\`') + '`,');
        fs.writeFileSync(filePath, content);
        console.log('Updated ' + filePath);
    } else {
        const regex2 = /theory:\s*`(?:[^`\\]|\\.)*`,/s;
        if (regex2.test(content)) {
            content = content.replace(regex2, 'theory: `' + newTheory.replace(/`/g, '\\`') + '`,');
            fs.writeFileSync(filePath, content);
            console.log('Updated ' + filePath + ' (backticks)');
        } else {
            console.log('Could not find theory in ' + filePath);
        }
    }
}

replaceTheory('src/data/modules/optimizationAlgorithms.ts', optimTheory);
