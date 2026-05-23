import type { MLModule } from '../../types';

export const convolutionalNetworks: MLModule = {
    id: 'cnn',
    title: 'Convolutional Neural Networks',
    category: 'Deep Learning',
    description: 'Learn spatial hierarchies of features automatically using convolutions and pooling.',
    formula: 'S(i, j) = (I * K)(i, j) = \\sum_m \\sum_n I(i-m, j-n) K(m, n)',
    interactiveSummary: 'This interactive demo lets you visualize individual convolution filters sliding across an input image in real time, generating feature maps that highlight edges, textures, and shapes detected at each layer. You can swap between hand-crafted kernels (edge detector, blur, sharpen) and learned filters to see how low-level features compose into high-level representations. The pooling section shows side-by-side max vs. average pooling on the feature maps, illustrating how spatial downsampling preserves the strongest activations while shrinking dimensionality. Observe how stacking Conv → ReLU → Pool blocks progressively compresses the spatial resolution while increasing the channel depth, ultimately producing compact feature vectors that feed into the classifier.',
    theory: `### What is it?
A Convolutional Neural Network (CNN or ConvNet) is a specialized class of deep neural networks designed to process data that has a known grid-like topology, most notably images (2D grids of pixels). Unlike MLPs where every input is connected to every neuron, CNNs use sparsely connected layers, weight sharing, and pooling to automatically and adaptively learn spatial hierarchies of features from low- to high-level patterns.

### Why do we need it?
When applying standard MLPs to high-resolution images, the number of parameters explodes. A fully connected layer taking a 1000x1000 RGB image would require 3 million weights per neuron, leading to unmanageable compute requirements and severe overfitting. Furthermore, MLPs ignore the spatial structure of an image; they treat pixels as an unordered list. CNNs were invented to exploit this spatial structure: pixels close together are more heavily related than those far apart. We need them to efficiently and effectively solve computer vision tasks like image classification, object detection, and semantic segmentation.

### How does it work?
A CNN consists of three main types of layers:
1. **Convolutional Layers:** These layers slide (convolve) small filters (kernels) across the input image to detect features like edges, corners, and textures. Because the same filter is applied across the whole image, the network learns translation-invariant features (a cat is a cat no matter where it is in the image).
2. **Pooling Layers:** These layers downsample the spatial dimensions of the feature maps, reducing the number of parameters and computation in the network, while simultaneously making the detected features more robust to slight spatial shifts (translational invariance).
3. **Fully Connected Layers:** At the end of the network, the 3D feature maps are flattened into a 1D vector and passed through standard dense layers to output the final classification probabilities.

### The Math Behind It
1. **Convolution Operation:**
For a 2D image $I$ and a 2D kernel $K$ of size $m \\times n$, the discrete convolution is defined as:
$$S(i, j) = (I * K)(i, j) = \\sum_{m} \\sum_{n} I(i-m, j-n) K(m, n)$$
In deep learning, we typically compute the cross-correlation (no kernel flipping) but call it convolution.

2. **Output Size Calculation:**
Given input size $W$, filter size $F$, padding $P$, and stride $S$, the output dimension is:
$$O = \\lfloor \\frac{W - F + 2P}{S} \\rfloor + 1$$

3. **Max Pooling:**
Given a region $R_{i,j}$ of the feature map, max pooling outputs:
$$P(i, j) = \\max_{(x,y) \\in R_{i,j}} I(x, y)$$

### Worked Example
Suppose we have a 5x5 input image and a 3x3 filter (kernel) with stride 1 and no padding.
- Input size $W = 5$
- Filter size $F = 3$
- Padding $P = 0$
- Stride $S = 1$
The output size will be $\\lfloor (5 - 3 + 0)/1 \\rfloor + 1 = 3$. So the result is a 3x3 feature map.
If the top-left 3x3 patch of the image is all 1s, and the filter is all 1s, the top-left pixel of the feature map will be the sum of $1 \\times 1$ nine times, which is 9. This sliding window process repeats for all 9 valid positions.

### Common Pitfalls
1. **Ignoring Receptive Field:** If a CNN is not deep enough or lacks pooling, the neurons in the final layer might only "see" a small portion of the input image, missing the global context needed for classification.
2. **Aggressive Pooling:** Too much pooling too early drops critical spatial information, crippling performance on tasks requiring high resolution (like segmentation).
3. **Improper Padding:** Using 'valid' padding continuously shrinks the feature maps rapidly in deep networks. 'Same' padding should be used to maintain spatial dimensions.
4. **Parameter Explosion in FC Layers:** Flattening large feature maps into dense layers introduces millions of parameters. Use Global Average Pooling (GAP) instead to mitigate this.

### When to Use vs Not Use
**When to Use:**
- For any image-based tasks (classification, detection, generation).
- For sequential data with strong local patterns using 1D CNNs (e.g., audio processing, certain NLP tasks, time-series forecasting).
- For volumetric data using 3D CNNs (e.g., MRI scans, video action recognition).

**When Not to Use:**
- For tabular data where features have no spatial relationship (use MLPs, XGBoost).
- When global temporal dependencies in sequences are more critical than local patterns (use Transformers or LSTMs).
- On extremely small datasets where transfer learning is unavailable.

### Key Takeaways
- CNNs drastically reduce parameters via weight sharing and sparse connectivity.
- They are built by stacking Conv, ReLU, and Pooling layers, culminating in Dense or GAP layers.
- They automatically learn a hierarchy of features: edges $\\rightarrow$ shapes $\\rightarrow$ objects.
- ResNets (with skip connections) and MobileNets (with depthwise separable convolutions) represent significant modern advancements in CNN architecture.
`,
    simulatorId: 'cnn',
    quiz: [
      { id: 'cnn_q1', question: 'What is the primary advantage of CNNs over MLPs for image data?', options: ['They use sigmoid activations', 'They reduce parameters via weight sharing and spatial awareness', 'They never overfit', 'They do not require labeled data'], correctAnswer: 'They reduce parameters via weight sharing and spatial awareness', explanation: 'CNNs slide filters across the image, reusing the same weights and reducing the total parameter count drastically.' },
      { id: 'cnn_q2', question: 'What does a Pooling layer do?', options: ['Increases image resolution', 'Downsamples spatial dimensions to reduce parameters', 'Adds color channels', 'Performs backpropagation'], correctAnswer: 'Downsamples spatial dimensions to reduce parameters', explanation: 'Pooling (e.g., Max Pooling) summarizes local regions, reducing spatial size and conferring translational invariance.' },
      { id: 'cnn_q3', question: 'If you have a 32x32 input, a 3x3 filter, stride 1, and 0 padding, what is the output size?', options: ['32x32', '30x30', '34x34', '16x16'], correctAnswer: '30x30', explanation: 'Output = (32 - 3 + 2*0)/1 + 1 = 30.' },
      { id: 'cnn_q4', question: 'Which activation function is most commonly used after a convolutional layer?', options: ['Tanh', 'Sigmoid', 'Softmax', 'ReLU'], correctAnswer: 'ReLU', explanation: 'ReLU introduces non-linearity efficiently without suffering from vanishing gradients early in the network.' },
      { id: 'cnn_q5', question: 'What is "Same" padding?', options: ['Padding with the same values as the image pixels', 'Adding padding so the output spatial dimensions equal the input spatial dimensions', 'Using the same padding across all layers', 'Padding only the top and bottom'], correctAnswer: 'Adding padding so the output spatial dimensions equal the input spatial dimensions', explanation: 'Same padding ensures the feature map size does not shrink after convolution (assuming stride 1).' },
      { id: 'cnn_q6', question: 'What is a 1x1 convolution commonly used for?', options: ['Increasing spatial resolution', 'Edge detection', 'Changing the number of channels (dimensionality reduction/expansion)', 'Adding non-linearity without weights'], correctAnswer: 'Changing the number of channels (dimensionality reduction/expansion)', explanation: 'A 1x1 convolution acts as a pixel-wise fully connected layer across channels, useful for controlling channel depth.' },
      { id: 'cnn_q7', question: 'What is the Receptive Field?', options: ['The area of the image that a filter is currently processing', 'The region in the original input image that affects a specific neuron in a deeper layer', 'The number of channels in a layer', 'The final output probability distribution'], correctAnswer: 'The region in the original input image that affects a specific neuron in a deeper layer', explanation: 'As you go deeper, each neuron aggregates information from a larger patch of the original image.' },
      { id: 'cnn_q8', question: 'Why is Global Average Pooling (GAP) often used instead of a Fully Connected layer at the end of a CNN?', options: ['To increase parameter count', 'To map arbitrary image sizes to a fixed-length vector and prevent overfitting', 'To add more non-linearity', 'To increase resolution'], correctAnswer: 'To map arbitrary image sizes to a fixed-length vector and prevent overfitting', explanation: 'GAP averages each feature map into a single number, replacing millions of dense parameters with zero parameters.' },
      { id: 'cnn_q9', question: 'What problem did the ResNet architecture solve?', options: ['Vanishing gradients in extremely deep networks', 'The need for max pooling', 'High memory usage of 1x1 convolutions', 'The inability to process color images'], correctAnswer: 'Vanishing gradients in extremely deep networks', explanation: 'ResNet introduced skip connections, allowing gradients to bypass layers and flow easily back to early layers.' },
      { id: 'cnn_q10', question: 'What type of features do the early layers (closer to the input) of a CNN typically learn?', options: ['Complete objects like faces or cars', 'Class probabilities', 'Low-level features like edges, lines, and colors', 'Complex textures'], correctAnswer: 'Low-level features like edges, lines, and colors', explanation: 'Early layers act as Gabor filters detecting basic structural elements, while deeper layers assemble them into complex concepts.' }
    ],
    coding: {
      tutorial: {
        title: 'Flattening Images',
        description: 'Before PyTorch or Keras introduced easy functional layers, flattening a batch of 2D images into 1D vectors was a common manual step using numpy. Note: Modern frameworks use `nn.Flatten()`.',
        pseudoCode: 'Use numpy reshape to collapse spatial dimensions while keeping batch dimension intact.',
        starterCode: `import numpy as np\n\n# 5 images, 28x28 resolution, 3 color channels\nX_batch = np.random.rand(5, 28, 28, 3)\n\n# TODO: Flatten X_batch to shape (5, 2352)\nX_flat = None\nprint(X_flat.shape)`,
        expectedOutput: '(5, 2352)',
        solution: `import numpy as np\n\nX_batch = np.random.rand(5, 28, 28, 3)\nX_flat = X_batch.reshape(X_batch.shape[0], -1)\nprint(X_flat.shape)`,
        hints: ['Use X_batch.reshape(X_batch.shape[0], -1)'],
        testKeywords: ['reshape']
      },
      project: {
        title: 'Traffic Sign Image Classification',
        description: 'Use `sklearn.neural_network.MLPClassifier` to classify simulated flattened image vectors — this mirrors the Dense head of a CNN after spatial features have been extracted and flattened. sklearn has no Conv2d layers; the project stage demonstrates the classification component using sklearn, while the assignment upgrades the full pipeline to PyTorch.',
        pseudoCode: '1. Scale features with StandardScaler\n2. Split into train and test sets (75/25)\n3. Initialize MLPClassifier with hidden_layer_sizes=(128, 64), activation=\'relu\', max_iter=500\n4. Fit model and predict on test set\n5. Print accuracy rounded to 2 decimal places',
        starterCode: `import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Simulated flattened 16x16 grayscale images (sklearn cannot use 2D convolutions,
# so we flatten first — this is exactly what a CNN does before its Dense head)
np.random.seed(42)
X = np.random.rand(80, 256)   # 80 images, 16x16 = 256 features each
y = np.array([0]*20 + [1]*20 + [2]*20 + [3]*20)  # 4 classes

# TODO: Scale X with StandardScaler
# TODO: Split 75/25 with train_test_split(random_state=42)
# TODO: Fit MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=500, random_state=42) on training data
# TODO: Predict on X_test and print accuracy rounded to 2 decimal places

scaler = None
model = None
print("Accuracy:", 0.0)`,
        expectedOutput: 'Accuracy: 0.2',
        solution: `import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X = np.random.rand(80, 256)
y = np.array([0]*20 + [1]*20 + [2]*20 + [3]*20)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=500, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, preds), 2))`,
        hints: ['Import and use StandardScaler, train_test_split, and MLPClassifier.', 'Make sure to set hidden_layer_sizes=(128, 64) and random_state=42 for reproducibility.'],
        testKeywords: ['MLPClassifier', 'StandardScaler', 'train_test_split', 'accuracy_score']
      },
      assignment: {
        title: 'Plant Disease Detection CNN',
        description: 'Build and train a CNN to detect plant diseases from leaf images. Samples are simulated as 1-channel (grayscale) 16×16 tensors with 3 disease classes: Healthy (0), Leaf Blight (1), and Rust (2). Train using CrossEntropyLoss and Adam for 150 epochs. Print the final training accuracy.',
        pseudoCode: `# Plant Disease Detection CNN\nclass PlantDiseaseCNN(nn.Module):\n    __init__:\n        conv1 = Conv2d(1 in, 8 out, kernel=3, padding=1)  # grayscale -> 8 maps\n        pool  = MaxPool2d(2)                                # 16x16 -> 8x8\n        conv2 = Conv2d(8 in, 16 out, kernel=3, padding=1) # 8x8 -> 4x4 after pool\n        fc1   = Linear(16*4*4, 32)\n        fc2   = Linear(32, 3)                              # 3 disease classes\n        relu  = ReLU()\n    forward(x):\n        x = pool(relu(conv1(x)))   # -> (B, 8, 8, 8)\n        x = pool(relu(conv2(x)))   # -> (B, 16, 4, 4)\n        x = flatten(x, 1)          # -> (B, 256)\n        x = relu(fc1(x))\n        return fc2(x)              # logits for 3 classes\n\nmodel = PlantDiseaseCNN()\ncriterion = CrossEntropyLoss()\noptimizer = Adam(model.parameters(), lr=0.01)\nfor epoch in range(150):\n    optimizer.zero_grad()\n    loss = criterion(model(X), y)\n    loss.backward(); optimizer.step()\nprint accuracy`,
        starterCode: `import torch\nimport torch.nn as nn\nimport torch.optim as optim\n\n# Simulated grayscale leaf images\ntorch.manual_seed(21)\nX = torch.randn(24, 1, 16, 16)   # 24 leaf images, 1 channel (grayscale), 16x16\ny = torch.tensor([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2])  # 3 classes\n\n# TODO: Define PlantDiseaseCNN\n# conv1: 1 -> 8 channels, kernel=3, padding=1\n# MaxPool2d(2) -> 8x8\n# conv2: 8 -> 16 channels, kernel=3, padding=1\n# MaxPool2d(2) -> 4x4\n# fc1: 16*4*4 -> 32, fc2: 32 -> 3\nclass PlantDiseaseCNN(nn.Module):\n    def __init__(self):\n        super().__init__()\n        # TODO: define conv1, conv2, pool, fc1, fc2, relu\n        pass\n    def forward(self, x):\n        # TODO: forward pass\n        return x\n\nmodel = PlantDiseaseCNN()\n# TODO: CrossEntropyLoss and Adam(lr=0.01). Train 150 epochs.\n\nwith torch.no_grad():\n    preds = model(X).argmax(dim=1)\n    acc = (preds == y).float().mean().item()\nprint(f"Accuracy: {acc:.2f}")`,
        expectedOutput: 'Accuracy: 1.00',
        solution: `import torch\nimport torch.nn as nn\nimport torch.optim as optim\n\ntorch.manual_seed(21)\nX = torch.randn(24, 1, 16, 16)\ny = torch.tensor([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2])\n\nclass PlantDiseaseCNN(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)\n        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)\n        self.pool = nn.MaxPool2d(2)\n        self.fc1 = nn.Linear(16 * 4 * 4, 32)\n        self.fc2 = nn.Linear(32, 3)\n        self.relu = nn.ReLU()\n\n    def forward(self, x):\n        x = self.pool(self.relu(self.conv1(x)))\n        x = self.pool(self.relu(self.conv2(x)))\n        x = torch.flatten(x, 1)\n        x = self.relu(self.fc1(x))\n        return self.fc2(x)\n\nmodel = PlantDiseaseCNN()\ncriterion = nn.CrossEntropyLoss()\noptimizer = optim.Adam(model.parameters(), lr=0.01)\n\nfor epoch in range(150):\n    optimizer.zero_grad()\n    loss = criterion(model(X), y)\n    loss.backward()\n    optimizer.step()\n\nwith torch.no_grad():\n    preds = model(X).argmax(dim=1)\n    acc = (preds == y).float().mean().item()\nprint(f"Accuracy: {acc:.2f}")`,
        hints: ['self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)', 'Use CrossEntropyLoss()', 'torch.flatten(x, 1) before fc layers'],
        testKeywords: ['PlantDiseaseCNN', 'CrossEntropyLoss', 'nn.Conv2d', 'backward']
      }
    },
    interviewQuestions: [
      { question: 'What is translational invariance in CNNs, and how does pooling achieve it?', answer: 'Translational invariance means recognizing an object regardless of its position. Max pooling achieves this by selecting the highest activation in a local window, so minor spatial shifts of the input do not change the pooled output.', companyTags: ['Apple', 'Meta'], difficulty: 'Advanced' },
      { question: 'Explain the difference between Valid and Same padding.', answer: 'Valid padding means no padding, so the spatial dimensions shrink after convolution. Same padding pads the input with zeros such that the output spatial dimension matches the input dimension (assuming stride 1).', companyTags: ['Google', 'Amazon'], difficulty: 'Easy' },
      { question: 'Why do we use 3x3 filters instead of larger ones like 7x7?', answer: 'Stacking multiple 3x3 filters achieves the same receptive field as a larger filter but with significantly fewer parameters and more non-linearities, making the network deeper and more efficient.', companyTags: ['Meta', 'Microsoft'], difficulty: 'Medium' },
      { question: 'What is a 1x1 convolution and why is it useful?', answer: 'It computes a linear combination of the channels for each spatial pixel. It acts as a cross-channel parametric pooling, used to reduce or expand the channel dimension (dimensionality reduction), as seen in Inception and ResNet architectures.', companyTags: ['Google', 'Apple'], difficulty: 'Hard' },
      { question: 'How do you calculate the number of parameters in a convolutional layer?', answer: 'Number of parameters = ((Filter_width * Filter_height * Input_channels) + 1_for_bias) * Output_channels.', companyTags: ['Netflix', 'Amazon'], difficulty: 'Medium' },
      { question: 'What is the Receptive Field?', answer: 'The specific region in the original input image that influences a particular neuron in a deep layer. As you stack convolutions and pooling, the receptive field grows exponentially.', companyTags: ['Meta'], difficulty: 'Medium' },
      { question: 'Why are CNNs better than MLPs for image classification?', answer: 'Images have massive input dimensions (e.g., 224x224x3). MLPs require a weight for every pixel connection, leading to explosion of parameters. CNNs use weight sharing and sparse connectivity, preserving spatial structure and preventing overfitting.', companyTags: ['Google'], difficulty: 'Easy' },
      { question: 'What is Dilated (Atrous) Convolution?', answer: 'A convolution where the filter is expanded by inserting spaces between its elements. It exponentially expands the receptive field without increasing the number of parameters or losing spatial resolution.', companyTags: ['Apple', 'Meta'], difficulty: 'Hard' },
      { question: 'Explain the ResNet architecture and why it was revolutionary.', answer: 'ResNet introduced skip (residual) connections that bypass one or more layers. This allows gradients to flow directly during backpropagation, solving the vanishing gradient problem and enabling networks with hundreds of layers.', companyTags: ['Microsoft', 'Amazon'], difficulty: 'Hard' },
      { question: 'What is Depthwise Separable Convolution?', answer: 'It splits a standard convolution into two steps: a spatial depthwise convolution (applied independently per channel) and a 1x1 pointwise convolution (combining channels). It drastically reduces parameters and computation, used heavily in MobileNets.', companyTags: ['Google', 'Apple'], difficulty: 'Advanced' },
      { question: 'What is the purpose of Global Average Pooling (GAP)?', answer: 'GAP averages the entire spatial map of each channel into a single value, replacing massive fully connected layers at the end of CNNs. It heavily reduces parameters and prevents overfitting.', companyTags: ['Meta', 'Netflix'], difficulty: 'Medium' },
      { question: 'How does Batch Normalization work in a CNN?', answer: 'In CNNs, Batch Norm normalizes across the batch and spatial dimensions for each channel independently, meaning it learns one mean and variance per feature map (channel).', companyTags: ['Google', 'Amazon'], difficulty: 'Medium' },
      { question: 'Can CNNs be used for text or time-series data?', answer: 'Yes, 1D CNNs can slide over sequences to detect local patterns like n-grams in text or local trends in time series, often much faster than RNNs.', companyTags: ['Microsoft'], difficulty: 'Easy' },
      { question: 'What is Transposed Convolution (Deconvolution)?', answer: 'It is used to upsample spatial dimensions, often used in autoencoders, GANs, and semantic segmentation (like U-Net) to project low-resolution feature maps back to high-resolution images.', companyTags: ['Meta', 'Apple'], difficulty: 'Hard' },
      { question: 'What happens if you set the stride greater than the filter size?', answer: 'The filter skips portions of the input image, leading to a loss of information because some pixels will never be analyzed.', companyTags: ['Amazon'], difficulty: 'Easy' },
      { question: 'What is Semantic Segmentation?', answer: 'A computer vision task where every pixel in an image is classified into a category, often solved using Fully Convolutional Networks (FCNs) like U-Net.', companyTags: ['Google'], difficulty: 'Medium' },
      { question: 'Why might Average Pooling be preferred over Max Pooling?', answer: 'Average pooling considers all elements in a window, providing a smoother representation, whereas max pooling only propagates the strongest feature. Average is often used at the very end (GAP) to summarize the whole feature map.', companyTags: ['Microsoft'], difficulty: 'Medium' },
      { question: 'Explain the concept of Feature Maps.', answer: 'The output tensor of a convolutional layer. Each slice (channel) of the feature map represents the activation of a specific filter across the spatial extent of the input.', companyTags: ['Netflix'], difficulty: 'Easy' },
      { question: 'What is Data Augmentation and why is it crucial for CNNs?', answer: 'Applying random transformations (rotations, flips, crops) to training images. It artificially increases dataset size and forces the CNN to learn robust, invariant features rather than memorizing exact pixel layouts.', companyTags: ['Amazon', 'Meta'], difficulty: 'Medium' },
      { question: 'How do you address varying image sizes in CNNs?', answer: 'Traditionally, images are resized/cropped to a fixed dimension before training. Alternatively, Global Average Pooling or Spatial Pyramid Pooling can be used to accept arbitrary spatial dimensions and output a fixed-size vector for dense layers.', companyTags: ['Apple', 'Google'], difficulty: 'Hard' }
    ]
};
