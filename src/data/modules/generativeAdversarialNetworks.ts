import type { MLModule } from '../../types';

export const generativeAdversarialNetworks: MLModule = {
    id: 'generative-adversarial-networks',
    title: 'Generative Adversarial Networks',
    category: 'Advanced & MLOps',
    description: 'Generate synthetic distributions by contesting Generator and Discriminator networks.',
    formula: '\\min_G \\max_D V(D,G)',
    theory: `### Generative Adversarial Networks (GANs)

#### What is it?
Generative Adversarial Networks (GANs) represent a monumental leap in generative modeling, introduced by Ian Goodfellow et al. in 2014. At its core, a GAN is a machine learning framework that pits two artificial neural networks against each other in a zero-sum game. 
The two networks are called the Generator and the Discriminator. 
By contesting with one another, the Generator learns to map from a latent space to a data distribution of interest, while the Discriminator distinguishes candidates produced by the generator from the true data distribution.
This framework can generate stunningly realistic images, videos, audio, and even tabular data that perfectly mirror the statistical properties of the original dataset.
Unlike previous models that relied heavily on explicit density estimation, GANs operate implicitly, focusing purely on the generative process.

#### Why do we need it?
Before GANs, generative models like Restricted Boltzmann Machines or standard Autoencoders struggled to produce high-fidelity, sharp outputs. 
They often resulted in blurry images because they optimized for average pixel distances (e.g., Mean Squared Error).
GANs solve this by not using a traditional loss function like MSE for the Generator. Instead, the Generator's "loss function" is dynamically provided by the Discriminator.
We need GANs in domains requiring extreme realism:
1. **Data Augmentation:** Generating synthetic medical scans to train cancer-detecting models when real data is scarce or restricted by privacy laws.
2. **Creative Arts and Media:** High-resolution image generation, style transfer, and music composition.
3. **Super-resolution and Inpainting:** Upscaling low-resolution images or filling in missing parts of a picture.
4. **Drug Discovery:** Generating novel molecular structures with desired chemical properties.

#### How does it work?
The GAN architecture relies on the interplay of its two components:
- **Generator Network ($G$):** This network takes a random noise vector $z$ (sampled from a simple prior distribution, like a standard Gaussian or Uniform distribution) and transforms it into a synthetic data sample $G(z)$. Its sole objective is to create outputs so realistic that the Discriminator classifies them as real.
- **Discriminator Network ($D$):** This is a binary classifier. It receives alternating inputs: real samples $x$ from the true training dataset, and fake samples $G(z)$ from the Generator. It outputs a single scalar probability $D(x)$ representing the likelihood that the input is real.

**The Training Loop:**
1. Sample a batch of random noise vectors and pass them through $G$ to create fake samples.
2. Sample a batch of real data from the training set.
3. Train $D$ for one or more steps to correctly classify the real samples as 1 and the fake samples as 0.
4. Train $G$ for one step by passing noise through $G$, then evaluating it with $D$, and updating $G$'s weights to maximize $D$'s output (i.e., making $D$ predict 1 for fakes).
As training progresses, $D$ becomes a sharper critic, forcing $G$ to produce increasingly indistinguishable fakes.

#### The Math Behind It
The GAN framework is formalized as a minimax optimization problem where $G$ minimizes the objective while $D$ maximizes it:

$$ \\min_{G} \\max_{D} V(D, G) = \\mathbb{E}_{x \\sim p_{data}(x)} [\\log D(x)] + \\mathbb{E}_{z \\sim p_z(z)} [\\log (1 - D(G(z)))] $$

Breaking this down:
- $D(x)$ is the Discriminator's estimate of the probability that real data instance $x$ is real.
- $\\mathbb{E}_{x \\sim p_{data}(x)} [\\log D(x)]$ represents the Discriminator's success in correctly identifying real data.
- $G(z)$ is the Generator's output given noise $z$.
- $D(G(z))$ is the Discriminator's estimate of the probability that a fake instance is real.
- $\\mathbb{E}_{z \\sim p_z(z)} [\\log (1 - D(G(z)))]$ represents the Discriminator's success in correctly identifying fake data.

**The Non-Saturating Heuristic:**
In practice, the theoretical minimax loss can cause vanishing gradients for the Generator early in training. If $D$ is perfect, $D(G(z)) \\approx 0$, making $\\log(1 - D(G(z)))$ saturate with a gradient near 0.
Instead of minimizing $\\log(1 - D(G(z)))$, the Generator is typically trained to maximize $\\log D(G(z))$. This provides strong gradients precisely when $G$ is performing poorly.

#### Worked Example
Imagine training a GAN to generate human faces.
1. **Initialization:** The Generator $G$ produces matrices of pure static noise. The Discriminator $D$ randomly guesses.
2. **Early Training:** $D$ quickly learns that faces have eyes, noses, and distinct colors, while $G$'s outputs look like television static. $D$'s accuracy hits 99%.
3. **Generator Adaptation:** Because we train $G$ to maximize $\\log D(G(z))$, $G$ receives gradients telling it to add skin tones and eye-like shapes to its static.
4. **Mid Training:** $G$ produces blobs that vaguely resemble faces. $D$ must now look for finer details, like pupils or hair textures, to tell fakes apart.
5. **Convergence:** Ideally, $G$ produces photorealistic faces. $D$ is completely stumped, predicting $D(x) = 0.5$ for both real and fake images, meaning it is randomly guessing because the distributions match perfectly.

#### Common Pitfalls
GANs are notoriously difficult to train due to the delicate balance required between the two networks.
1. **Mode Collapse:** The Generator discovers a specific output (or a small set of outputs) that consistently fools the Discriminator. It stops exploring the latent space and only generates variations of this single "mode" (e.g., generating only pictures of white cats, ignoring all other animals).
2. **Vanishing Gradients:** If the Discriminator learns much faster than the Generator, it achieves perfect classification. The gradients passed back to the Generator become infinitesimally small, and learning halts.
3. **Oscillation and Non-convergence:** Because the loss landscape constantly shifts (the target moves as the other network updates), the parameters can oscillate wildly without ever settling into a Nash Equilibrium.

#### When to Use vs Not Use
**Use GANs when:**
- You require exceptionally sharp, high-fidelity, photorealistic generations (e.g., style transfer, image synthesis).
- You want an implicit generative model without needing to explicitly define a complex probability distribution.
- You are working in domains where perceptual quality is paramount over strictly covering the entire underlying distribution.

**Do Not Use GANs when:**
- You are working with discrete sequences (like Natural Language). The discrete nature makes backpropagation from the Discriminator to Generator extremely difficult without RL techniques. Transformers/LLMs are vastly superior.
- You need a perfectly mapped, dense latent space with exact log-likelihoods (use VAEs or Normalizing Flows).
- Training stability and strict reproducibility are critical, as GAN training is highly sensitive to hyperparameters and random initialization. Diffusion models are increasingly preferred for stable high-quality generation.

#### Key Takeaways
1. GANs utilize a two-player zero-sum game between a Generator and a Discriminator.
2. The Generator maps random noise to data space; the Discriminator acts as a binary critic.
3. The objective relies on the minimax formulation: $\\min_G \\max_D V(D,G)$.
4. While GANs produce incredibly sharp outputs, they suffer from training instabilities like Mode Collapse and Vanishing Gradients.
5. Architectural improvements like DCGAN and loss formulation changes like Wasserstein GAN (WGAN) are standard practices to stabilize training.`,
    interactiveSummary: 'This interactive simulation demonstrates the delicate adversarial dance between the Generator and the Discriminator. Use the controls to adjust the learning rates for each network. You will observe how the Generator attempts to warp the red data points (synthetic data) to perfectly overlap the blue data points (real data distribution), while the Discriminator dynamically shifts its confidence boundary (the background color map) to tell them apart. Watch out for mode collapse if the generator focuses on a single dense cluster!',
    simulatorId: 'gan',
    quiz: [
      { id: 'gan_q1', question: 'What is the optimization objective of the Generator network in a standard GAN framework?', options: ['Maximize the probability of the Discriminator predicting synthetic samples are real.', 'Minimize Discriminator parameters.', 'Maximize prior distribution variance.', 'Fit target labels directly.'], correctAnswer: 'Maximize the probability of the Discriminator predicting synthetic samples are real.', explanation: 'The generator tries to create samples that fool the discriminator, maximizing D(G(z)).' },
      { id: 'gan_q2', question: 'What is mode collapse in a GAN?', options: ['The generator produces a very limited variety of outputs.', 'The discriminator loss drops to zero.', 'The network architecture becomes too deep.', 'The discriminator forgets past distributions.'], correctAnswer: 'The generator produces a very limited variety of outputs.', explanation: 'Mode collapse happens when the generator learns a few specific patterns that fool the discriminator and repeats them.' },
      { id: 'gan_q3', question: 'What input does the Generator usually take?', options: ['Random noise vectors.', 'Real images.', 'Class labels.', 'Discriminator weights.'], correctAnswer: 'Random noise vectors.', explanation: 'The generator maps random noise (latent space) to a complex data distribution.' },
      { id: 'gan_q4', question: 'In the minimax game, what is the ideal final state for the Discriminator?', options: ['It guesses randomly (0.5 probability) because fakes are indistinguishable from real.', 'It predicts fakes perfectly (0 probability).', 'It predicts reals perfectly (1 probability).', 'Its loss goes to infinity.'], correctAnswer: 'It guesses randomly (0.5 probability) because fakes are indistinguishable from real.', explanation: 'If the generator is perfect, the discriminator cannot tell the difference, resulting in a 50/50 guess.' },
      { id: 'gan_q5', question: 'Why is the standard GAN generator loss often changed from minimizing log(1 - D(G(z))) to maximizing log(D(G(z)))?', options: ['To prevent vanishing gradients early in training.', 'To prevent mode collapse.', 'To reduce computational complexity.', 'To make the discriminator stronger.'], correctAnswer: 'To prevent vanishing gradients early in training.', explanation: 'Early on, when fakes are bad, D is very confident. log(1 - D) is flat, causing vanishing gradients. log(D) provides stronger gradients.' },
      { id: 'gan_q6', question: 'What kind of network is the Discriminator typically?', options: ['A binary classifier.', 'A linear regression model.', 'A recurrent neural network.', 'An autoencoder.'], correctAnswer: 'A binary classifier.', explanation: 'It outputs a single probability representing whether the input is real or fake.' },
      { id: 'gan_q7', question: 'What does a Wasserstein GAN (WGAN) change compared to a standard GAN?', options: ['It replaces the discriminator with a critic that outputs a real number instead of a probability.', 'It uses multiple generators.', 'It removes the generator.', 'It trains using supervised learning.'], correctAnswer: 'It replaces the discriminator with a critic that outputs a real number instead of a probability.', explanation: 'WGAN uses Earth Mover\'s distance and a critic outputting unconstrained scores, leading to smoother gradients.' },
      { id: 'gan_q8', question: 'Which architectural variation uses Convolutional Neural Networks for image generation?', options: ['DCGAN', 'RNN-GAN', 'MLP-GAN', 'Transformer-GAN'], correctAnswer: 'DCGAN', explanation: 'Deep Convolutional GANs (DCGAN) popularized the use of CNNs for both generator and discriminator.' },
      { id: 'gan_q9', question: 'What is a Conditional GAN (cGAN)?', options: ['A GAN where both generator and discriminator are conditioned on extra information like class labels.', 'A GAN that only trains under certain conditions.', 'A GAN that generates conditional probabilities.', 'A GAN with conditional statements in its code.'], correctAnswer: 'A GAN where both generator and discriminator are conditioned on extra information like class labels.', explanation: 'cGANs allow you to specify what class of image to generate by feeding the label to both networks.' },
      { id: 'gan_q10', question: 'Compared to VAEs, what is a characteristic of GAN-generated images?', options: ['They are typically sharper but harder to train.', 'They are blurrier but easier to train.', 'They are lower resolution.', 'They have perfectly accurate continuous latent spaces.'], correctAnswer: 'They are typically sharper but harder to train.', explanation: 'GANs do not rely on pixel-wise reconstruction loss like VAEs, resulting in sharper images, but suffer from training instability.' }
    ],
    coding: {
      tutorial: {
        title: 'Generator Loss Function',
        description: 'Calculate binary cross-entropy loss for fake samples. Note: In `sklearn`, we can simulate the discriminator using `sklearn.neural_network.MLPClassifier` and evaluate its probability output.',
        pseudoCode: `function generator_loss(discriminator_fake_predictions):
    return -mean(log(discriminator_fake_predictions))`,
        starterCode: `import numpy as np

def generator_loss(d_fake_pred):
    # TODO: return -mean(log(p))
    return 0.0

preds = np.array([0.8, 0.9])
print(np.round(generator_loss(preds), 3))`,
        expectedOutput: '0.164',
        solution: `import numpy as np

def generator_loss(d_fake_pred):
    return -np.mean(np.log(d_fake_pred))

preds = np.array([0.8, 0.9])
print(np.round(generator_loss(preds), 3))`,
        hints: ['Apply natural log np.log, compute mean, and negate.'],
        testKeywords: ['np.log', 'np.mean']
      },
      project: {
        title: 'Building a Simple Generator in PyTorch',
        description: 'Construct a multi-layer perceptron (MLP) Generator using PyTorch to map random noise into a 2D data distribution.',
        pseudoCode: `function build_generator(latent_dim, out_dim):
    model = Sequential(
        Linear(latent_dim, 2, bias=False),
        ReLU(),
        Linear(2, out_dim, bias=False)
    )
    return model`,
        starterCode: `import torch
import torch.nn as nn

# TODO: Create a simple Generator class inheriting from nn.Module
# It should have a latent_dim of 2 and an output_dim of 2
# Use a single hidden layer with 2 units, no biases, and a ReLU activation

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers here
        
    def forward(self, x):
        # Define forward pass
        return x

gen = Generator()
noise = torch.tensor([[0.5, 0.5]])
# Assume weights are initialized to 1 for this test
with torch.no_grad():
    for param in gen.parameters():
        param.fill_(1.0)
output = gen(noise)

print(output)`,
        expectedOutput: 'tensor([[2., 2.]], grad_fn=<MmBackward0>)',
        solution: `import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 2, bias=False),
            nn.ReLU(),
            nn.Linear(2, 2, bias=False)
        )
        
    def forward(self, x):
        return self.net(x)

gen = Generator()
noise = torch.tensor([[0.5, 0.5]])

# Ensure deterministic output by filling weights with 1.0
with torch.no_grad():
    for param in gen.parameters():
        param.fill_(1.0)

output = gen(noise)
print(output)`,
        hints: ['Inherit from nn.Module', 'Use nn.Sequential with nn.Linear and nn.ReLU'],
        testKeywords: ['nn.Module', 'nn.Linear', 'nn.ReLU']
      },
      assignment: {
        title: 'Training Step of a Discriminator in PyTorch',
        description: 'Implement a single training step for the Discriminator using Binary Cross Entropy (BCE) loss on real and fake data.',
        pseudoCode: `function train_discriminator_step(real_preds, fake_preds, criterion):
    real_loss = criterion(real_preds, ones)
    fake_loss = criterion(fake_preds, zeros)
    return real_loss + fake_loss`,
        starterCode: `import torch
import torch.nn as nn

criterion = nn.BCELoss()

# Simulated predictions from a Discriminator model
real_preds = torch.tensor([[0.9], [0.8]])
fake_preds = torch.tensor([[0.1], [0.2]])

# TODO: Create target tensors (1s for real, 0s for fake)
# TODO: Compute loss for real data
# TODO: Compute loss for fake data
# TODO: Print the rounded sum of both losses

loss_total = 0.0
print(round(loss_total, 2))`,
        expectedOutput: '0.33',
        solution: `import torch
import torch.nn as nn

criterion = nn.BCELoss()
real_preds = torch.tensor([[0.9], [0.8]])
fake_preds = torch.tensor([[0.1], [0.2]])

real_targets = torch.ones_like(real_preds)
fake_targets = torch.zeros_like(fake_preds)

loss_real = criterion(real_preds, real_targets)
loss_fake = criterion(fake_preds, fake_targets)

loss_total = loss_real + loss_fake
print(round(loss_total.item(), 2))`,
        hints: ['Use torch.ones_like and torch.zeros_like for targets', 'Call criterion(predictions, targets)'],
        testKeywords: ['ones_like', 'zeros_like', 'BCELoss']
      }
    },
    interviewQuestions: [
      { question: 'What is a Generative Adversarial Network (GAN)?', answer: 'A GAN is a deep learning architecture consisting of two neural networks, a Generator and a Discriminator, which compete against each other to generate highly realistic synthetic data.', companyTags: ['OpenAI', 'Google'], difficulty: 'Medium' },
      { question: 'What are the distinct roles of the Generator and the Discriminator?', answer: 'The Generator takes random noise and generates fake data samples. The Discriminator takes both real and fake data and tries to classify them correctly. The Generator learns to fool the Discriminator.', companyTags: ['Meta', 'Apple'], difficulty: 'Easy' },
      { question: 'What is the minimax loss function in a GAN?', answer: 'It is a zero-sum game formulation where the Generator tries to minimize the objective while the Discriminator tries to maximize it: min_G max_D V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))].', companyTags: ['Tesla', 'Amazon'], difficulty: 'Hard' },
      { question: 'What is Mode Collapse?', answer: 'Mode collapse is a failure case where the generator produces a very limited variety of outputs that successfully fool the discriminator, ignoring the full diversity of the real data distribution.', companyTags: ['OpenAI', 'DeepMind'], difficulty: 'Advanced' },
      { question: 'How do Wasserstein GANs (WGAN) address mode collapse?', answer: 'WGANs replace the binary classifier Discriminator with a Critic that outputs a real-valued score based on the Earth Mover’s distance, providing smoother, non-vanishing gradients across all distribution modes.', companyTags: ['Google', 'Meta'], difficulty: 'Advanced' },
      { question: 'Why is the heuristic non-saturating loss preferred over the minimax loss for the Generator?', answer: 'Early in training, the Discriminator rejects fakes easily, causing log(1-D(G(z))) to have vanishing gradients. Maximizing log(D(G(z))) instead provides stronger gradients for the Generator when it needs them most.', companyTags: ['Amazon', 'Tesla'], difficulty: 'Hard' },
      { question: 'What is the vanishing gradient problem in GANs?', answer: 'If the Discriminator becomes too accurate too quickly, it assigns a probability of 0 to fakes. The derivative of the loss becomes near zero, preventing the Generator from updating its weights.', companyTags: ['Microsoft', 'Apple'], difficulty: 'Medium' },
      { question: 'What is an unrolled GAN?', answer: 'An unrolled GAN simulates the discriminator\'s updates a few steps into the future before updating the generator. This helps stabilize training and reduces mode collapse.', companyTags: ['DeepMind', 'OpenAI'], difficulty: 'Advanced' },
      { question: 'How is a Conditional GAN (cGAN) different from a standard GAN?', answer: 'A cGAN feeds auxiliary information (like class labels) into both the Generator and Discriminator, allowing the model to generate data specific to a chosen condition.', companyTags: ['Google', 'Uber'], difficulty: 'Medium' },
      { question: 'What is Fréchet Inception Distance (FID)?', answer: 'FID is a widely used metric to evaluate GAN quality. It measures the distance between the feature vectors of real and generated images extracted from a pre-trained Inception network.', companyTags: ['Meta', 'Netflix'], difficulty: 'Hard' },
      { question: 'Why are GANs difficult to evaluate?', answer: 'Unlike traditional supervised learning, GANs lack an explicit objective function (like accuracy or MSE) that directly indicates the quality and diversity of the generated images.', companyTags: ['Apple', 'Amazon'], difficulty: 'Medium' },
      { question: 'Can GANs generate discrete data like text?', answer: 'Generating discrete data is difficult because the sampling step is non-differentiable, making backpropagation from the Discriminator to the Generator challenging. Reinforcement learning (e.g., policy gradients) or Gumbel-Softmax tricks are often needed.', companyTags: ['OpenAI', 'Google'], difficulty: 'Hard' },
      { question: 'What is DCGAN?', answer: 'Deep Convolutional GAN (DCGAN) is a foundational GAN variant that successfully integrated Convolutional Neural Networks (CNNs) by replacing pooling layers with strided convolutions and applying batch normalization.', companyTags: ['Tesla', 'Meta'], difficulty: 'Medium' },
      { question: 'What is the role of Batch Normalization in GANs?', answer: 'Batch Normalization stabilizes learning by normalizing the input to each layer, helping to prevent the generator from collapsing to a single point. However, applying it to all layers can cause sample correlation within batches.', companyTags: ['Microsoft', 'Uber'], difficulty: 'Medium' },
      { question: 'What does the latent space (z-space) represent in a GAN?', answer: 'It is a lower-dimensional vector space of random variables (noise). The generator learns to map points from this continuous latent space to points in the complex data distribution space.', companyTags: ['Amazon', 'DeepMind'], difficulty: 'Easy' },
      { question: 'How can you perform arithmetic in the latent space of a well-trained GAN?', answer: 'By finding the latent vectors representing specific features (e.g., man with glasses), you can add/subtract them to generate new features (e.g., vector(man with glasses) - vector(man) + vector(woman) = vector(woman with glasses)).', companyTags: ['Google', 'Meta'], difficulty: 'Advanced' },
      { question: 'What is CycleGAN?', answer: 'CycleGAN is an architecture that performs unpaired image-to-image translation. It uses cycle-consistency loss to ensure that translating an image from domain A to B and back to A yields the original image.', companyTags: ['OpenAI', 'Apple'], difficulty: 'Hard' },
      { question: 'Why do we use LeakyReLU instead of ReLU in Discriminators?', answer: 'LeakyReLU allows a small positive gradient when the unit is not active, preventing "dying ReLUs" and ensuring gradients flow backwards to the Generator continuously.', companyTags: ['Tesla', 'Netflix'], difficulty: 'Medium' },
      { question: 'What is Spectral Normalization in GANs?', answer: 'Spectral Normalization stabilizes discriminator training by constraining the Lipschitz constant of the network, preventing the discriminator gradients from exploding without the heavy tuning required by WGAN-GP.', companyTags: ['Google', 'DeepMind'], difficulty: 'Advanced' },
      { question: 'What are deepfakes?', answer: 'Deepfakes are highly realistic manipulated videos or images, often created using GAN architectures to superimpose faces or alter voices, raising significant ethical and security concerns.', companyTags: ['Microsoft', 'Amazon'], difficulty: 'Easy' }
    ]
};
