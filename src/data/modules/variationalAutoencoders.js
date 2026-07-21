export const variationalAutoencoders = {
    id: 'variational-autoencoders',
    title: 'Variational Autoencoders',
    category: 'Advanced & MLOps',
    description: 'Map datasets to continuous latent spaces using reconstruction and KL divergence penalties.',
    formula: 'L = E[log P(x|z)] - D_{KL}(Q(z|x) || P(z))',
    theory: `### Variational Autoencoders (VAEs) Theory

#### What is it?
Variational Autoencoders (VAEs) are a class of powerful generative models heavily rooted in Bayesian inference. Unlike standard Autoencoders that compress data into a fixed, discrete vector, VAEs compress data into a probability distribution. 
By learning the parameters of a distribution (mean and variance) rather than fixed points, VAEs ensure that the latent space is continuous and highly structured. 
This continuous mapping allows the model to generate entirely new, original data points simply by sampling from this continuous distribution and decoding the sampled vector.

#### Why do we need it?
Standard Autoencoders are excellent for dimensionality reduction or denoising, but they fail catastrophically as generative models. Their latent spaces are disjoint and discontinuous. If you sample a random point from the latent space of a standard autoencoder, the decoder will likely produce meaningless noise or garbage.
VAEs were designed to solve this exact problem. By enforcing a continuous, regularized latent space, VAEs enable:
1. **Smooth Interpolation:** You can mathematically blend two different inputs (e.g., transforming a picture of a cat smoothly into a dog) by drawing a line between their latent representations.
2. **Robust Data Generation:** Sampling any random point near the origin of the latent space yields a highly plausible data generation.
3. **Anomaly Detection:** VAEs can identify out-of-distribution or anomalous data points by evaluating their reconstruction probabilities.

#### How does it work?
A VAE architecture consists of two symmetric neural networks:
1. **Encoder (Inference Network, $q_\\phi(z|x)$):** It maps the input data $x$ to a latent distribution. Instead of outputting a single vector, the Encoder outputs two vectors: a mean vector $\\mu$ and a variance vector $\\sigma^2$ (typically parameterized as $\\log(\\sigma^2)$ for numerical stability).
2. **Decoder (Generative Network, $p_\theta(x|z)$):** During training, we sample a random point $z$ from the distribution defined by $\\mu$ and $\\sigma^2$. The Decoder then takes this sampled vector and reconstructs the original input data $x'$.
To generate brand new data after training, we simply discard the Encoder, sample a random point $z$ from a Standard Normal distribution $\\mathcal{N}(0, I)$, and pass it through the Decoder.

#### The Math Behind It
The VAE loss function, known as the Evidence Lower Bound (ELBO), balances two competing objectives. We want to maximize the ELBO, which is equivalent to minimizing the negative ELBO:
$$ \\mathcal{L}_{VAE}(\theta, \\phi) = -\\mathbb{E}_{q_\\phi(z|x)} [\\log p_\theta(x|z)] + D_{KL}(q_\\phi(z|x) \\parallel p(z)) $$

Breaking it down:
- **Reconstruction Loss ($-\\mathbb{E}[\\log p_\theta(x|z)]$):** Measures how well the decoder reconstructed the input. For images, this is often implemented as Mean Squared Error (MSE) or Binary Cross Entropy (BCE). It forces the network to retain essential information.
- **KL Divergence ($D_{KL}$):** This acts as a regularizer. It penalizes the encoder if its learned distributions $\\mathcal{N}(\\mu, \\sigma^2)$ deviate too far from a standard normal prior $\\mathcal{N}(0, I)$. This forces the distributions to heavily overlap near the origin, ensuring continuity.
$$ D_{KL} = -\frac{1}{2} \\sum_{j=1}^{J} \\left( 1 + \\log(\\sigma_j^2) - \\mu_j^2 - \\sigma_j^2 \right) $$

**The Reparameterization Trick:**
To train the network using backpropagation, we must take derivatives through the sampling process. However, random sampling is a non-differentiable operation. VAEs elegantly solve this using the reparameterization trick:
$$ z = \\mu + \\sigma \\odot \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0, I) $$
By shifting the randomness to a constant external noise variable $\\epsilon$, the network can backpropagate gradients smoothly through $\\mu$ and $\\sigma$.

#### Worked Example
Let's walk through an image of a handwritten digit '7' passing through a VAE:
1. The Encoder receives the '7' and outputs $\\mu = [0.5, -0.2]$ and $\\log(\\sigma^2)$ mapping to $\\sigma = [0.1, 0.5]$.
2. The network generates a random noise vector $\\epsilon = [0.05, -0.01]$ from a standard normal distribution.
3. We compute the latent vector: $z = [0.5, -0.2] + [0.1, 0.5] \times [0.05, -0.01] = [0.505, -0.205]$.
4. The Decoder receives $z$ and attempts to recreate the digit '7'.
5. The loss is calculated: MSE tells it how close the output pixels are to '7', and KL Divergence penalizes $\\mu$ and $\\sigma$ for being different from 0 and 1, respectively.
6. The weights are updated via backpropagation.

#### Common Pitfalls
- **Posterior Collapse:** If the KL divergence penalty is too strong relative to the reconstruction loss, the encoder simply ignores the input entirely. All latent distributions collapse exactly to the standard normal prior $\\mathcal{N}(0, I)$. The decoder then ignores the latent variable and just outputs the dataset average, learning nothing.
- **Blurry Outputs:** Because VAEs use explicit density estimation and pixel-wise loss functions (like MSE), generated images tend to be blurrier compared to the sharp outputs of GANs. MSE penalizes variations equally, so the model learns to output a "safe average" of possible features.

#### When to Use vs Not Use
- **Use VAEs when:** 
  - You need a highly structured, smooth continuous latent space for interpolation or representation learning.
  - You want stable, reliable training that is mathematically grounded, avoiding the wild instabilities of GANs.
  - You are doing anomaly detection or want to extract meaningful latent features from complex data.
- **Do Not Use VAEs when:**
  - You need extremely sharp, high-fidelity photorealistic images. Generative Adversarial Networks (GANs) or Diffusion models are significantly better suited for peak perceptual quality.

#### Key Takeaways
1. VAEs encode inputs as probability distributions (mean and variance), not fixed distinct points.
2. The ELBO loss function carefully balances accurate reconstruction with strict latent space regularization.
3. The Reparameterization Trick is a critical mathematical maneuver enabling backpropagation through stochastic nodes.
4. They provide excellent latent space interpolation and stable training, but typically yield slightly blurry generations.
#### Python Implementation

\`\`\`python
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(784, 20)
        self.decoder = nn.Linear(10, 784)
    
def vae_loss(recon, x, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(recon, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss
\`\`\`
`,
    interactiveSummary: 'This interactive tool visualizes the 2D latent space of a trained VAE. You can click anywhere on the coordinate grid, which represents a sampled vector $z$, and watch the Decoder reconstruct an image in real-time. Dragging your cursor across the space demonstrates smooth interpolation, morphing one digit/image seamlessly into another, showing the continuous nature of the learned distributions.',
    simulatorId: 'vae',
    quiz: [
        { id: 'vae_q1', question: 'What is the purpose of the reparameterization trick in VAEs?', options: ['To allow backpropagation gradients to flow through stochastic sampling nodes.', 'To compress latent dimensions.', 'To calculate reconstruction errors.', 'To normalize latent coordinates.'], correctAnswer: 'To allow backpropagation gradients to flow through stochastic sampling nodes.', explanation: 'Since random sampling lacks derivatives, the reparameterization trick projects randomness into a constant external input, making parameters differentiable.' },
        { id: 'vae_q2', question: 'What does the Encoder network in a VAE output?', options: ['A fixed single point in latent space.', 'A mean vector and a variance/standard deviation vector.', 'A reconstructed image.', 'A binary classification probability.'], correctAnswer: 'A mean vector and a variance/standard deviation vector.', explanation: 'Instead of a discrete point, the VAE encoder outputs the parameters of a probability distribution.' },
        { id: 'vae_q3', question: 'What is the purpose of the KL Divergence term in the VAE loss function?', options: ['To ensure the reconstructed image matches the original pixel-by-pixel.', 'To force the learned latent distributions to approximate a standard normal distribution.', 'To prevent vanishing gradients.', 'To increase the capacity of the decoder.'], correctAnswer: 'To force the learned latent distributions to approximate a standard normal distribution.', explanation: 'The KL divergence penalizes the network if its latent distributions deviate too far from a standard normal prior, ensuring the latent space is dense and continuous.' },
        { id: 'vae_q4', question: 'What is posterior collapse in a VAE?', options: ['The encoder ignores the input data and maps everything to the exact prior distribution.', 'The decoder outputs completely random noise.', 'The latent space becomes completely disjoint.', 'The network stops learning due to exploding gradients.'], correctAnswer: 'The encoder ignores the input data and maps everything to the exact prior distribution.', explanation: 'If the KL penalty is too strong, the encoder stops using the input, and the latent distributions collapse to the prior, resulting in generic, average decoder outputs.' },
        { id: 'vae_q5', question: 'Why do VAEs often produce blurrier images compared to GANs?', options: ['Because they have fewer layers.', 'Because they use random noise.', 'Because they rely on a pixel-wise reconstruction loss (like MSE) that averages out uncertainty.', 'Because the latent space is continuous.'], correctAnswer: 'Because they rely on a pixel-wise reconstruction loss (like MSE) that averages out uncertainty.', explanation: 'MSE loss penalizes being slightly off in any direction, so the model learns to output the "average" of possible predictions, resulting in a blurry effect.' },
        { id: 'vae_q6', question: 'What does ELBO stand for?', options: ['Estimated Latent Base Output', 'Evidence Lower Bound', 'Encoder Loss Base Optimization', 'Error Logarithmic Base Objective'], correctAnswer: 'Evidence Lower Bound', explanation: 'ELBO represents the mathematical lower bound on the log-likelihood of the data, which VAEs maximize during training.' },
        { id: 'vae_q7', question: 'How do you generate a completely new, fake image using a trained VAE?', options: ['Pass a real image through the encoder.', 'Pass random noise through the encoder.', 'Sample a vector from a standard normal distribution and pass it to the decoder.', 'Average two real images and pass them to the decoder.'], correctAnswer: 'Sample a vector from a standard normal distribution and pass it to the decoder.', explanation: 'Because the latent space was forced to resemble a standard normal distribution, any random sample from N(0,I) will generate a plausible output.' },
        { id: 'vae_q8', question: 'In the reparameterization trick z = mu + sigma * epsilon, what is epsilon?', options: ['The learning rate.', 'A sample from a standard normal distribution N(0, 1).', 'The reconstruction error.', 'The bias term of the neural network.'], correctAnswer: 'A sample from a standard normal distribution N(0, 1).', explanation: 'Epsilon is the external source of randomness injected into the sampling process.' },
        { id: 'vae_q9', question: 'What happens if you remove the KL Divergence term from the VAE loss function?', options: ['It turns into a standard Autoencoder with a disjoint, meaningless latent space.', 'It turns into a GAN.', 'It generates much sharper images.', 'The network trains faster with better results.'], correctAnswer: 'It turns into a standard Autoencoder with a disjoint, meaningless latent space.', explanation: 'Without the KL regularization, the encoder has no incentive to organize the distributions, and it just learns discrete point mappings.' },
        { id: 'vae_q10', question: 'What is latent space interpolation in a VAE?', options: ['Training the model on missing data.', 'Moving gradually between two points in the latent space to watch the generated outputs smoothly morph into each other.', 'Adding new layers to the latent space.', 'Discretizing the continuous variables.'], correctAnswer: 'Moving gradually between two points in the latent space to watch the generated outputs smoothly morph into each other.', explanation: 'Because the VAE latent space is continuous, drawing a line between two points and decoding the steps produces a smooth morphing effect.' }
    ],
    coding: {
        tutorial: {
            title: 'KL Divergence Penalty',
            description: 'Calculate the KL divergence penalty given mean and variance vectors. Note: While `sklearn` does not natively support VAEs, methods like `sklearn.decomposition.PCA` perform linear dimensionality reduction, a simpler precursor to autoencoders.',
            pseudoCode: `function kl_divergence(mu, var):
    return -0.5 * sum(1 + log(var) - mu^2 - var)`,
            starterCode: `import numpy as np

def kl_divergence(mu, var):
    # TODO: return -0.5 * sum(1 + log(var) - mu^2 - var)
    return 0.0

mu = np.array([0.0])
var = np.array([1.0])
print(kl_divergence(mu, var))`,
            expectedOutput: '-0.0',
            solution: `import numpy as np

def kl_divergence(mu, var):
    return -0.5 * np.sum(1.0 + np.log(var) - mu**2 - var)

mu = np.array([0.0])
var = np.array([1.0])
print(kl_divergence(mu, var))`,
            hints: ['Use np.log and np.sum to calculate the penalty.'],
            testKeywords: ['np.log', 'np.sum']
        },
        project: {
            title: 'Building a VAE Encoder in PyTorch',
            description: 'Create the Inference Network (Encoder) for a VAE using PyTorch, returning both the mean and log-variance vectors.',
            pseudoCode: `function build_encoder(input_dim, latent_dim):
    fc = Linear(input_dim, 2, bias=False)
    fc_mu = Linear(2, latent_dim, bias=False)
    fc_logvar = Linear(2, latent_dim, bias=False)
    
    function forward(x):
        h = ReLU(fc(x))
        return fc_mu(h), fc_logvar(h)`,
            starterCode: `import torch
import torch.nn as nn

# TODO: Create a VAEEncoder class inheriting from nn.Module
# It needs a shared dense layer (fc), and two separate dense layers (fc_mu, fc_logvar).
# Input dimension is 2, hidden is 2, latent is 2. Do not use biases.

class VAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers
        
    def forward(self, x):
        # Define forward pass
        return None, None

enc = VAEEncoder()
# Set weights to ones for deterministic output
with torch.no_grad():
    for param in enc.parameters():
        param.fill_(1.0)

x = torch.tensor([[0.5, 0.5]])
mu, logvar = enc(x)
print(f"mu: {mu}")
print(f"logvar: {logvar}")`,
            expectedOutput: 'mu: tensor([[2., 2.]]), logvar: tensor([[2., 2.]])',
            solution: `import torch
import torch.nn as nn

class VAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2, bias=False)
        self.fc_mu = nn.Linear(2, 2, bias=False)
        self.fc_logvar = nn.Linear(2, 2, bias=False)
        
    def forward(self, x):
        h = torch.relu(self.fc(x))
        return self.fc_mu(h), self.fc_logvar(h)

enc = VAEEncoder()

with torch.no_grad():
    for param in enc.parameters():
        param.fill_(1.0)

x = torch.tensor([[0.5, 0.5]])
mu, logvar = enc(x)

# Print clean output format
print(f"mu: {mu}, logvar: {logvar}")`,
            hints: ['Inherit from nn.Module', 'Use torch.relu on the shared layer output before passing to mu and logvar layers'],
            testKeywords: ['nn.Linear', 'torch.relu']
        },
        assignment: {
            title: 'Implementing the Reparameterization Trick',
            description: 'Implement the reparameterization trick mathematically in PyTorch so gradients can flow through the random sampling step.',
            pseudoCode: `function reparameterize(mu, logvar):
    std = exp(0.5 * logvar)
    eps = generate_random_tensor_like(std)
    return mu + eps * std`,
            starterCode: `import torch

def reparameterize(mu, logvar):
    # TODO: Calculate standard deviation (std) from logvar
    # TODO: Generate an epsilon tensor of the same shape as std
    # For this exercise, assume epsilon is fixed at 0.5 for deterministic output
    # TODO: Return mu + eps * std
    pass

mu = torch.tensor([[1.0, 1.0]])
logvar = torch.tensor([[0.0, 0.0]]) # Note: exp(0) = 1

z = reparameterize(mu, logvar)
print(z)`,
            expectedOutput: 'tensor([[1.5000, 1.5000]])',
            solution: `import torch

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    # Using fixed epsilon = 0.5 for stable expected output matching
    eps = torch.ones_like(std) * 0.5
    return mu + eps * std

mu = torch.tensor([[1.0, 1.0]])
logvar = torch.tensor([[0.0, 0.0]])

z = reparameterize(mu, logvar)
print(z)`,
            hints: ['std is torch.exp(0.5 * logvar)', 'multiply eps with std and add mu'],
            testKeywords: ['torch.exp', 'ones_like']
        }
    },
    interviewQuestions: [
        { question: 'What is a Variational Autoencoder (VAE)?', answer: 'A VAE is a generative model that learns to map input data to a continuous, probabilistic latent space, allowing for the generation of new data by sampling from this distribution.', companyTags: ['Google', 'Meta'], difficulty: 'Medium' },
        { question: 'How does a VAE differ from a standard Autoencoder?', answer: 'A standard Autoencoder compresses data into discrete, fixed-point vectors, leading to disjoint latent spaces. VAEs compress data into probability distributions (mean and variance), creating a continuous space.', companyTags: ['Tesla', 'Amazon'], difficulty: 'Medium' },
        { question: 'What is the mathematical loss function of a VAE called?', answer: 'The Evidence Lower Bound (ELBO), which is composed of a Reconstruction Loss term and a Kullback-Leibler (KL) Divergence penalty term.', companyTags: ['OpenAI', 'DeepMind'], difficulty: 'Hard' },
        { question: 'What role does KL Divergence play in VAEs?', answer: 'It acts as a regularizer, penalizing the network if the learned latent distributions deviate too far from a standard normal prior distribution N(0,I). This ensures overlapping, continuous clusters.', companyTags: ['Google', 'Netflix'], difficulty: 'Advanced' },
        { question: 'What is the Reparameterization Trick, and why is it necessary?', answer: 'Backpropagation cannot flow through a random sampling node. The trick rewrites the random sample z as mu + sigma * epsilon, moving the randomness to an external constant epsilon, making mu and sigma differentiable.', companyTags: ['Meta', 'Apple'], difficulty: 'Hard' },
        { question: 'What is posterior collapse in VAEs?', answer: 'It happens when the KL divergence penalty dominates the loss. The encoder ignores the input, forcing all latent distributions exactly to the prior. The decoder then only generates an average representation of the dataset.', companyTags: ['OpenAI', 'Google'], difficulty: 'Advanced' },
        { question: 'How can you mitigate posterior collapse?', answer: 'Strategies include KL Annealing (gradually increasing the weight of the KL term during training), using more expressive decoders, or modifying the architecture (like VQ-VAE).', companyTags: ['DeepMind', 'Meta'], difficulty: 'Advanced' },
        { question: 'Why do VAEs generally produce blurrier images compared to GANs?', answer: 'VAEs explicitly optimize for reconstruction loss like Mean Squared Error (MSE), which penalizes average pixel differences. This causes the model to output conservative, "average" blurred predictions instead of sharp, high-frequency details.', companyTags: ['Tesla', 'Amazon'], difficulty: 'Medium' },
        { question: 'What is the prior distribution in a standard VAE?', answer: 'It is a standard Multivariate Normal (Gaussian) distribution with a mean of 0 and an identity covariance matrix: N(0, I).', companyTags: ['Microsoft', 'Uber'], difficulty: 'Easy' },
        { question: 'What does the Encoder network output in a VAE?', answer: 'It outputs two vectors: the mean (mu) and the logarithm of the variance (log_var) for the normal distribution describing the latent space mapping of the input.', companyTags: ['Apple', 'Netflix'], difficulty: 'Medium' },
        { question: 'Why do VAEs output log variance instead of variance directly?', answer: 'Neural networks can easily output negative numbers, but variance must be positive. By predicting log_var, we can use exponential (exp) to guarantee a positive variance while letting the network output unconstrained values.', companyTags: ['Google', 'Meta'], difficulty: 'Advanced' },
        { question: 'What is latent space interpolation?', answer: 'Because VAEs have continuous latent spaces, you can take the latent vector of Image A, draw a straight line to the vector of Image B, and decode the points along the line to watch A smoothly morph into B.', companyTags: ['OpenAI', 'Tesla'], difficulty: 'Medium' },
        { question: 'What is a Disentangled VAE (beta-VAE)?', answer: 'A beta-VAE adds a hyperparameter beta > 1 to heavily weight the KL divergence term. This encourages the latent variables to become statistically independent, meaning single latent dimensions control distinct semantic features (e.g., hair color, smile).', companyTags: ['DeepMind', 'Google'], difficulty: 'Hard' },
        { question: 'What is the difference between explicit and implicit density models?', answer: 'VAEs are explicit density models; they maximize an explicit lower bound on the data log-likelihood. GANs are implicit models; they learn to sample from the distribution without explicitly modeling the probability density function.', companyTags: ['Meta', 'Amazon'], difficulty: 'Advanced' },
        { question: 'How do you generate a new sample using a trained VAE?', answer: 'You bypass the encoder, sample a random noise vector from the standard normal prior distribution N(0, I), and pass it directly through the trained decoder.', companyTags: ['Apple', 'Microsoft'], difficulty: 'Easy' },
        { question: 'What happens to the latent space if you remove the KL penalty?', answer: 'The VAE degenerates into a standard autoencoder. The network maps inputs to tightly compressed, non-overlapping points, destroying the continuous properties required for generation.', companyTags: ['Uber', 'Google'], difficulty: 'Medium' },
        { question: 'Can VAEs be used for anomaly detection?', answer: 'Yes. Anomaly detection can be performed by computing the reconstruction error of a new sample. High error implies the model hasn\'t seen similar data, flagging it as an anomaly.', companyTags: ['Amazon', 'Tesla'], difficulty: 'Medium' },
        { question: 'What is a Vector Quantized VAE (VQ-VAE)?', answer: 'A VQ-VAE uses a discrete latent space rather than continuous, mapping encoder outputs to the nearest vector in a learned codebook. It resolves issues like posterior collapse and generates much sharper images.', companyTags: ['DeepMind', 'OpenAI'], difficulty: 'Hard' },
        { question: 'In the ELBO equation, what does "Evidence" refer to?', answer: 'Evidence refers to the marginal likelihood of the observed data, P(x). The ELBO establishes a lower bound on the log of this evidence.', companyTags: ['Meta', 'Google'], difficulty: 'Advanced' },
        { question: 'Why is Mean Squared Error (MSE) often used as the reconstruction loss for images in VAEs?', answer: 'MSE corresponds to the assumption that the decoder\'s output models the mean of a Gaussian distribution for the pixel values, which mathematically aligns with the maximum likelihood estimation framework of the VAE.', companyTags: ['Microsoft', 'Apple'], difficulty: 'Advanced' }
    ]
};
