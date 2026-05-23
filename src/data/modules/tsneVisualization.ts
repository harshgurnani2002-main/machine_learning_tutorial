import type { MLModule } from '../../types';

export const tsneVisualization: MLModule = {
    id: 'tsne-visualization',
    title: 't-SNE Dimensionality Reduction',
    category: 'Unsupervised Learning',
    description: 'Visualize high-dimensional clusters by matching local probability distributions.',
    formula: 'p_{j|i} = \\frac{\\exp(-\\|x_i-x_j\\|^2 / 2\\sigma_i^2)}{\\sum \\exp(-\\|x_i-x_k\\|^2 / 2\\sigma_i^2)}',
    theory: `### t-SNE Dimensionality Reduction Theory

**What is it?**
t-Distributed Stochastic Neighbor Embedding (t-SNE) is an advanced, non-linear machine learning algorithm primarily used for dimensionality reduction and data visualization. Developed by Laurens van der Maaten and Geoffrey Hinton, t-SNE excels at taking incredibly high-dimensional data (like 1024-dimensional neural network embeddings, high-resolution images, or dense genetic profiles) and squashing it down into an intuitive 2D or 3D scatter plot. 

**Why do we need it?**
The human brain simply cannot comprehend geometry beyond three dimensions. When data scientists train complex models, they often produce high-dimensional feature vectors that represent the data. To understand if the model has successfully separated different classes (e.g., dogs vs. cats in an embedding space), we need to visualize it.

Traditional techniques like Principal Component Analysis (PCA) are linear. PCA is great at preserving the global variance and large-scale distances in a dataset, but it often completely fails to preserve the intricate, non-linear, local neighborhood structures. t-SNE was specifically designed to solve this: it prioritizes keeping similar data points close together in the low-dimensional map, making it the ultimate tool for finding and visualizing clusters.

**How does it work?**
t-SNE works by converting distances between data points into probabilities. It does this in two separate spaces (the high-dimensional original space, and the low-dimensional target space) and then tries to make the two probability distributions match.

1. **Calculate High-Dimensional Affinities**: For every point $x_i$, t-SNE calculates the probability that it would pick another point $x_j$ as its "neighbor." This probability is proportional to the probability density under a Gaussian distribution centered at $x_i$. Close points get high probabilities; distant points get near-zero probabilities.
2. **Calculate Low-Dimensional Affinities**: t-SNE randomly initializes the points in the 2D/3D space. It calculates the neighborhood probabilities again, but this time it uses a Student-t distribution (which has heavy, "fat" tails) instead of a Gaussian.
3. **Minimize Divergence**: t-SNE uses Gradient Descent to physically move the points around in the 2D/3D space to minimize the Kullback-Leibler (KL) Divergence between the high-dimensional Gaussian probabilities and the low-dimensional Student-t probabilities.

**The Math Behind It**

- **High-Dimensional Probabilities ($P$)**:
The conditional probability $p_{j|i}$ that $x_i$ picks $x_j$ as its neighbor is:
$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$
The variance $\sigma_i$ is determined dynamically based on a hyperparameter called **Perplexity**, which acts as a smooth measure of the effective number of neighbors. The joint probability is symmetrized: $p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$.

- **Low-Dimensional Probabilities ($Q$)**:
The joint probability $q_{ij}$ for points $y_i$ and $y_j$ in the low-dimensional map uses a Student-t distribution with one degree of freedom:
$$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k} \sum_{l \neq k} (1 + \|y_k - y_l\|^2)^{-1}}$$
The use of the heavy-tailed t-distribution solves the **"Crowding Problem"**, allowing points that are moderately far apart in high dimensions to be placed much further apart in the 2D map, preventing everything from collapsing into the center.

- **Objective Function (KL Divergence)**:
The cost function minimized using gradient descent is:
$$C = KL(P \parallel Q) = \sum_{i} \sum_{j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$
Because $p_{ij}$ is in the numerator, the algorithm heavily penalizes placing points far apart in 2D if they were close in high dimensions (large $p$, small $q$). However, it barely penalizes placing points close together in 2D if they were far in high dimensions (small $p$, large $q$). This fundamentally drives t-SNE to prioritize local cluster preservation.

**Worked Example**
Imagine applying t-SNE to the famous MNIST dataset of handwritten digits (784 dimensions per image).
1. We choose a perplexity of 30.
2. For an image of a '7', t-SNE calculates the Euclidean distance to all other images. Other '7's are closer, so they get higher probabilities ($p_{ij}$). '2's are far, getting near-zero probabilities.
3. We initialize 2D points randomly on a canvas.
4. We calculate the 2D probabilities ($q_{ij}$) using the Student-t distribution.
5. We calculate the KL divergence gradient. Because the '7's have a high $p_{ij}$ but are currently scattered randomly (low $q_{ij}$), the gradient exerts an attractive force, pulling the '7's together.
6. After 1000 iterations of gradient descent, the '7's have formed a distinct cluster, far away from the '2's, creating a beautiful visualization.

**Common Pitfalls**
1. **Misinterpreting Global Geometry**: The distance between distinct clusters in a t-SNE plot is largely meaningless. Just because cluster A is closer to cluster B than cluster C does NOT mean A and B are more similar in high-dimensional space.
2. **Misinterpreting Cluster Size**: Because t-SNE dynamically adjusts its local variance ($\sigma_i$) based on data density, dense clusters expand and sparse clusters contract. The visual size of a cluster in a t-SNE plot does not reflect its true variance or dispersion.
3. **Ignoring Hyperparameters**: Perplexity makes or breaks the algorithm. If it's too low (e.g., 2), the plot will splinter into meaningless micro-clusters. If it's too high (e.g., equal to the dataset size), the plot will collapse into a single blob. Standard values are between 5 and 50.
4. **Stochasticity**: The KL divergence cost function is highly non-convex. Two runs of t-SNE on the exact same data will produce different plots (clusters might be rotated or placed in different corners). 

**When to Use vs Not Use**
- **Use when**: You want to intuitively explore a complex dataset, verify that deep learning embeddings (like BERT or ResNet vectors) have successfully separated classes, or present beautiful visualizations of data manifolds.
- **Not Use when**: You are performing feature extraction for a downstream machine learning pipeline. t-SNE does not learn a parametric mapping function $f(x)$, meaning you cannot apply it to new, unseen test data easily. Use PCA or Autoencoders for actual dimensionality reduction pipelines.

**Key Takeaways**
- t-SNE is a non-linear dimensionality reduction technique optimized solely for visualization.
- It operates by matching probability distributions of pairwise distances.
- It beautifully preserves local neighborhoods (clusters) but destroys global distances and topology.
- It utilizes the heavy-tailed Student-t distribution to solve the crowding problem of high-dimensional spaces.
`,
    interactiveSummary: 'This interactive simulator lets you adjust the perplexity and learning rate hyperparameters of the t-SNE algorithm in real-time. You can observe how the high-dimensional points are projected onto the 2D canvas, and how the algorithm iterations minimize the KL divergence loss. Use this to understand the impact of perplexity on the formation and separation of local clusters.',
    simulatorId: 'tsne',
    quiz: [
      { id: 'tsne_q1', question: 'Why does t-SNE use a Student-t distribution in the low-dimensional projection space?', options: ['To resolve the crowding problem by allowing points to spread out further.', 'To speed up training gradients.', 'To force features to be orthogonal.', 'To normalize coordinates.'], correctAnswer: 'To resolve the crowding problem by allowing points to spread out further.', explanation: 'The Student-t distribution has heavier tails than a Gaussian distribution. This addresses the crowding problem by allowing distant points to remain far apart in low-dimensional space.' },
      { id: 'tsne_q2', question: 'What does the perplexity parameter in t-SNE control?', options: ['The balance between local and global attention.', 'The number of output dimensions.', 'The learning rate of the gradient descent.', 'The total number of iterations.'], correctAnswer: 'The balance between local and global attention.', explanation: 'Perplexity can be interpreted as a smooth measure of the effective number of neighbors. It dictates how to balance attention between local and global aspects of the dataset.' },
      { id: 'tsne_q3', question: 'Which distance metric is typically used to compute high-dimensional affinities in standard t-SNE?', options: ['Euclidean distance', 'Manhattan distance', 'Cosine similarity', 'Jaccard index'], correctAnswer: 'Euclidean distance', explanation: 'Standard t-SNE uses squared Euclidean distances to compute the Gaussian probabilities in the high-dimensional space.' },
      { id: 'tsne_q4', question: 'Is the cost function of t-SNE convex?', options: ['No', 'Yes', 'Only for small datasets', 'Only when perplexity is large'], correctAnswer: 'No', explanation: 'The KL divergence objective in t-SNE is non-convex. This means gradient descent can get stuck in local minima, and different runs can yield different results.' },
      { id: 'tsne_q5', question: 'Which algorithm is t-SNE based on?', options: ['Stochastic Neighbor Embedding (SNE)', 'Principal Component Analysis (PCA)', 'Isomap', 'Locally Linear Embedding (LLE)'], correctAnswer: 'Stochastic Neighbor Embedding (SNE)', explanation: 't-SNE is a variation of SNE that introduces a symmetric cost function and uses a t-distribution in the low-dimensional space.' },
      { id: 'tsne_q6', question: 'Can the distances between clusters in a t-SNE plot be reliably interpreted as the true distance between them in high-dimensional space?', options: ['No', 'Yes', 'Only if PCA is applied first', 'Only if perplexity is high'], correctAnswer: 'No', explanation: 't-SNE is designed to preserve local structures. The distances between distinct clusters (global structure) are not reliably preserved and can be misleading.' },
      { id: 'tsne_q7', question: 'What is a common preprocessing step before applying t-SNE to very high-dimensional data (e.g., > 50 dimensions)?', options: ['Applying PCA to reduce dimensions to around 50', 'Applying L1 regularization', 'Applying K-Means clustering', 'Scaling features to follow a power-law distribution'], correctAnswer: 'Applying PCA to reduce dimensions to around 50', explanation: 'It is highly recommended to use another dimensionality reduction method (like PCA) to reduce the number of dimensions to a reasonable amount before running t-SNE, to speed up computation and suppress noise.' },
      { id: 'tsne_q8', question: 'What does the KL divergence measure in the context of t-SNE?', options: ['The difference between the high-dimensional and low-dimensional probability distributions', 'The distance between a point and its nearest neighbor', 'The variance of the Gaussian distribution', 'The number of effective neighbors'], correctAnswer: 'The difference between the high-dimensional and low-dimensional probability distributions', explanation: 'KL divergence quantifies how much the low-dimensional probability distribution Q deviates from the high-dimensional probability distribution P.' },
      { id: 'tsne_q9', question: 'If you want to focus more on global structure in t-SNE, how should you adjust the perplexity?', options: ['Increase it', 'Decrease it', 'Set it to 0', 'Perplexity does not affect this'], correctAnswer: 'Increase it', explanation: 'Larger perplexity values cause the algorithm to consider more points as neighbors, increasing the influence of global structure, whereas lower values focus strictly on immediate local neighborhoods.' },
      { id: 'tsne_q10', question: 'What problem does the "crowding problem" refer to in SNE?', options: ['There is not enough room in 2D space to accommodate all neighbors from a high-dimensional space at their true distances', 'The algorithm taking too long to run on large datasets', 'Multiple points having the exact same coordinates in high-dimensional space', 'The gradient becoming too large during optimization'], correctAnswer: 'There is not enough room in 2D space to accommodate all neighbors from a high-dimensional space at their true distances', explanation: 'In high dimensions, a point can have many equidistant neighbors. When mapped to 2D, they cannot all be equidistant without crowding the center, hence the t-distribution is used to allow them to spread out.' }
    ],
    coding: {
      tutorial: {
        title: 'Calculate Squared Distances',
        description: 'Calculate the squared Euclidean distance between two vectors. Note: the sklearn equivalent is heavily encapsulated within TSNE.fit_transform(X).',
        pseudoCode: `function squared_distance(u, v):
    difference = u - v
    squared_difference = difference ^ 2
    return sum(squared_difference)`,
        starterCode: `import numpy as np
def squared_distance(u, v):
    # TODO: return sum((u - v)^2)
    return 0.0

u = np.array([1.0, 2.0])
v = np.array([4.0, 6.0])
print(squared_distance(u, v))`,
        expectedOutput: '25.0',
        solution: `import numpy as np
def squared_distance(u, v):
    return float(np.sum((u - v) ** 2))

u = np.array([1.0, 2.0])
v = np.array([4.0, 6.0])
print(squared_distance(u, v))`,
        hints: ['Subtract vectors, square differences, and sum.'],
        testKeywords: ['np.sum']
      },
      project: {
        title: 'Sklearn t-SNE on Digits',
        description: 'Use sklearn to apply t-SNE for dimensionality reduction on a sample dataset.',
        pseudoCode: `Import TSNE from sklearn.manifold
Initialize TSNE with n_components=2, perplexity=30, random_state=42
Fit and transform the dataset X
Return the transformed data shape and the first point's coordinates`,
        starterCode: `import numpy as np
from sklearn.manifold import TSNE

# Hardcoded small dataset representing 4 clusters
X = np.array([
    [1, 1, 1, 1], [1, 2, 1, 2],
    [10, 10, 10, 10], [10, 11, 10, 11],
    [-10, -10, -10, -10], [-10, -11, -10, -11],
    [5, -5, 5, -5], [6, -6, 6, -6]
])

def apply_tsne(data):
    # TODO: Initialize TSNE (n_components=2, perplexity=2, random_state=42)
    # TODO: fit_transform data
    # TODO: return transformed data
    pass

X_embedded = apply_tsne(X)
print("Shape:", X_embedded.shape)
print("First point:", np.round(X_embedded[0], 2))`,
        expectedOutput: `Shape: (8, 2)\nFirst point: [-162.24   85.64]`,
        solution: `import numpy as np
from sklearn.manifold import TSNE

X = np.array([
    [1, 1, 1, 1], [1, 2, 1, 2],
    [10, 10, 10, 10], [10, 11, 10, 11],
    [-10, -10, -10, -10], [-10, -11, -10, -11],
    [5, -5, 5, -5], [6, -6, 6, -6]
])

def apply_tsne(data):
    tsne = TSNE(n_components=2, perplexity=2, random_state=42)
    return tsne.fit_transform(data)

X_embedded = apply_tsne(X)
print("Shape:", X_embedded.shape)
print("First point:", np.round(X_embedded[0], 2))`,
        hints: ['Initialize TSNE with perplexity=2 for this tiny dataset', 'Call fit_transform on the TSNE object'],
        testKeywords: ['TSNE', 'fit_transform', 'perplexity=2']
      },
      assignment: {
        title: 'Sklearn t-SNE KL Divergence Loss',
        description: 'Run t-SNE using sklearn and retrieve the final KL divergence loss from the fitted model.',
        pseudoCode: `Import TSNE
Initialize TSNE with specific hyperparameters
Fit the dataset
Extract the .kl_divergence_ attribute from the model
Return it`,
        starterCode: `import numpy as np
from sklearn.manifold import TSNE

# Toy dataset
X = np.array([
    [1.5, 2.5], [1.6, 2.6], [1.4, 2.4],
    [8.5, 8.5], [8.6, 8.6], [8.4, 8.4]
])

def get_tsne_kl_loss(data):
    # TODO: Initialize TSNE(n_components=2, perplexity=2, random_state=42, n_iter=250)
    # TODO: fit data
    # TODO: return the kl_divergence_ attribute
    pass

loss = get_tsne_kl_loss(X)
print("KL Divergence:", round(loss, 3))`,
        expectedOutput: `KL Divergence: 0.165`,
        solution: `import numpy as np
from sklearn.manifold import TSNE

X = np.array([
    [1.5, 2.5], [1.6, 2.6], [1.4, 2.4],
    [8.5, 8.5], [8.6, 8.6], [8.4, 8.4]
])

def get_tsne_kl_loss(data):
    tsne = TSNE(n_components=2, perplexity=2, random_state=42, n_iter=250)
    tsne.fit(data)
    return tsne.kl_divergence_

loss = get_tsne_kl_loss(X)
print("KL Divergence:", round(loss, 3))`,
        hints: ['Use tsne.fit() instead of fit_transform() if you only need the attribute', 'Access the kl_divergence_ property on the fitted tsne object'],
        testKeywords: ['TSNE', 'fit', 'kl_divergence_']
      }
    },
    interviewQuestions: [
      { question: 'What is the perplexity parameter in t-SNE, and how does it balance global and local dataset structure?', answer: 'Perplexity is a measure of the effective number of neighbors to consider when calculating high-dimensional probabilities, serving as a variance normalizer. Low perplexity values prioritize local clustering structure, whereas high perplexity values capture global dataset relationships.', companyTags: ['Netflix', 'Meta'], difficulty: 'Advanced' },
      { question: 'How does t-SNE solve the crowding problem?', answer: 'It uses a Student-t distribution with one degree of freedom (which has heavy tails) in the low-dimensional space. This requires points that are moderately distant in high-dimensional space to be placed much further apart in low-dimensional space to achieve the same probability density, thus preventing points from crowding in the center.', companyTags: ['Google', 'Amazon'], difficulty: 'Medium' },
      { question: 'Why is t-SNE generally preferred over PCA for visualization?', answer: 'PCA is a linear technique that preserves global variance, which often fails to separate non-linear data manifolds. t-SNE focuses on preserving local neighborhood structures, which is much more effective at revealing clusters and complex non-linear relationships in low dimensions.', companyTags: ['Apple', 'Microsoft'], difficulty: 'Medium' },
      { question: 'What is the time complexity of standard t-SNE?', answer: 'The standard t-SNE algorithm has a time complexity of O(N^2), where N is the number of data points, because it computes pairwise distances. This makes it unscalable for large datasets unless approximations like Barnes-Hut (O(N log N)) are used.', companyTags: ['Meta', 'Uber'], difficulty: 'Hard' },
      { question: 'What does the Kullback-Leibler (KL) divergence measure in t-SNE?', answer: 'The KL divergence measures the asymmetry between the high-dimensional probability distribution P and the low-dimensional probability distribution Q. t-SNE minimizes this divergence to ensure that Q is as close to P as possible.', companyTags: ['Netflix', 'LinkedIn'], difficulty: 'Medium' },
      { question: 'Why is the KL divergence cost function in t-SNE asymmetric?', answer: 'The KL divergence heavily penalizes modeling close high-dimensional points as distant in low dimensions (p_ij is large, q_ij is small). However, it barely penalizes modeling distant high-dimensional points as close in low dimensions (p_ij is small, q_ij is large). This asymmetry enforces the preservation of local structure.', companyTags: ['Google', 'Stripe'], difficulty: 'Hard' },
      { question: 'Can you use t-SNE for dimensionality reduction prior to training a classifier?', answer: 'It is generally not recommended. t-SNE is highly stochastic, doesn\'t preserve global geometry, and doesn\'t provide a mapping function f(x) to apply to new, unseen test data. PCA or Autoencoders are better suited for feature extraction.', companyTags: ['Amazon', 'Tesla'], difficulty: 'Medium' },
      { question: 'How do you handle new test data with t-SNE?', answer: 'Standard t-SNE does not learn a parametric mapping function, so you cannot directly transform new test data. You either have to run the algorithm again with the new data included, or use a parametric version of t-SNE (like a neural network trained to minimize t-SNE loss).', companyTags: ['Apple', 'Google'], difficulty: 'Medium' },
      { question: 'What is Barnes-Hut t-SNE?', answer: 'It is an optimization technique that reduces the time complexity of t-SNE from O(N^2) to O(N log N). It uses a quadtree (or octree) to approximate the interactions between distant points, instead of calculating exact pairwise probabilities for all points.', companyTags: ['Meta', 'Pinterest'], difficulty: 'Hard' },
      { question: 'Why do we often run PCA before t-SNE?', answer: 'Running PCA to reduce the data to a moderate number of dimensions (e.g., 50) before t-SNE helps suppress noise, reduce computational cost, and improve the quality of the final visualization by eliminating highly correlated or uninformative features.', companyTags: ['Microsoft', 'Spotify'], difficulty: 'Easy' },
      { question: 'Is t-SNE deterministic?', answer: 'No, t-SNE is stochastic. Its objective function is non-convex, meaning gradient descent can get trapped in different local minima depending on the random initialization. Different runs will yield differently oriented or shaped plots.', companyTags: ['Netflix', 'Amazon'], difficulty: 'Easy' },
      { question: 'What happens if perplexity is set too low?', answer: 'If perplexity is too low, the algorithm considers very few neighbors. The resulting visualization will be extremely fragmented into many small, meaningless clusters, failing to capture any broader structure.', companyTags: ['Uber', 'Airbnb'], difficulty: 'Medium' },
      { question: 'What happens if perplexity is set too high?', answer: 'If perplexity is close to the number of data points, the algorithm considers almost all points as neighbors. The resulting visualization will likely resemble a massive, structureless blob, losing all local cluster definitions.', companyTags: ['Google', 'Meta'], difficulty: 'Medium' },
      { question: 'How do cluster sizes in a t-SNE plot correlate with variance?', answer: 'They don\'t correlate well. Because t-SNE adapts its density metric locally based on perplexity, it expands dense clusters and contracts sparse ones in the low-dimensional plot. Visual cluster size does not imply true variance.', companyTags: ['Apple', 'Netflix'], difficulty: 'Hard' },
      { question: 'How do distances between clusters in a t-SNE plot correlate with high-dimensional distances?', answer: 'They are largely meaningless. The KL divergence primarily penalizes misrepresenting local neighborhoods. As a result, the distance between two separate clusters in the 2D plot does not accurately reflect their relative distance in the original high-dimensional space.', companyTags: ['Amazon', 'Microsoft'], difficulty: 'Hard' },
      { question: 'What is the role of Early Exaggeration in t-SNE?', answer: 'Early exaggeration artificially multiplies the high-dimensional probabilities (P) by a constant factor for the first few iterations. This forces clusters to stay tight and move further apart from each other initially, allowing the algorithm to find a better global arrangement before focusing on local details.', companyTags: ['Meta', 'Google'], difficulty: 'Hard' },
      { question: 'Explain the "Student-t" part of t-SNE.', answer: 'In the low-dimensional mapping, t-SNE uses a Student-t distribution with 1 degree of freedom (equivalent to a Cauchy distribution). Its heavy tails allow points that are far apart in high dimensions to be placed even further apart in low dimensions, solving the crowding problem of regular SNE.', companyTags: ['Netflix', 'Apple'], difficulty: 'Medium' },
      { question: 'What is the difference between SNE and t-SNE?', answer: 'SNE uses Gaussian distributions in both high and low dimensions and has an asymmetric cost function. t-SNE introduced a symmetric cost function (simplifying gradients) and uses a heavy-tailed Student-t distribution in the low-dimensional space to solve the crowding problem.', companyTags: ['Uber', 'Amazon'], difficulty: 'Medium' },
      { question: 'Can t-SNE be used for 3D visualizations?', answer: 'Yes, t-SNE can map data to 3D spaces as well as 2D. However, 2D is more common because it is easier to render and interpret on standard screens, whereas 3D plots often suffer from occlusion and require interactive rotation to interpret fully.', companyTags: ['Microsoft', 'Meta'], difficulty: 'Easy' },
      { question: 'How does learning rate affect t-SNE optimization?', answer: 'If the learning rate is too low, the algorithm will get stuck in a bad local minimum (often looking like a dense ball). If it is too high, the points will repel each other too strongly and the clusters will fail to form or become overly spread out.', companyTags: ['Google', 'Stripe'], difficulty: 'Medium' }
    ]
};
