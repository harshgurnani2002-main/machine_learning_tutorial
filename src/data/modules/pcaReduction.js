export const pcaReduction = {
    id: 'pca-reduction',
    title: 'Principal Component Analysis',
    category: 'Unsupervised Learning',
    description: 'Project high-dimensional vectors onto orthogonal axes of maximal variance.',
    formula: 'Sigma = \frac{1}{m} X^T X',
    theory: `### Principal Component Analysis (PCA)

#### What is it?
Principal Component Analysis (PCA) is an unsupervised, linear dimensionality reduction technique. It transforms a dataset with many correlated variables into a smaller set of uncorrelated variables called Principal Components, while retaining as much of the original variance (information) as possible.

#### Why do we need it?
In the era of big data, datasets often have hundreds or thousands of features. High dimensionality leads to the "Curse of Dimensionality," causing models to overfit, drastically increasing training times, and making visualization impossible. PCA solves this by:
- **Dimensionality Reduction**: Shrinking the feature space.
- **Visualization**: Allowing projection of 100D data into 2D or 3D for human analysis.
- **Noise Reduction**: Dropping lower-variance components often drops random noise.
- **Feature Extraction**: Uncovering hidden structural relationships between variables.

#### How does it work?
PCA searches for the directions (axes) in the feature space where the data varies the most.
1. It finds the line of maximum variance through the data (1st Principal Component).
2. It finds a second line, orthogonal (perpendicular) to the first, that captures the second most variance.
3. It repeats this process until it has as many principal components as original dimensions.
Because the components are ranked by the variance they capture, we can drop the lowest-ranked components with minimal information loss.

#### The Math Behind It
1. **Data Normalization**: Center inputs to mean zero, ensuring the first component doesn\'t just point to the data\'s mean location:
$$X_{centered} = X - \\mu$$
*(Note: It\'s also highly recommended to standardize the variance to 1).*

2. **Covariance Matrix Calculation**: Compute the covariance matrix $\\Sigma$ to understand how variables change together:
$$\\Sigma = \frac{1}{m} X_{centered}^T X_{centered}$$

3. **Eigendecomposition**: Calculate the eigenvectors $W$ and eigenvalues $\\lambda$ of the covariance matrix $\\Sigma$:
$$\\Sigma v = \\lambda v$$
- **Eigenvectors** indicate the direction of the new axes (Principal Components).
- **Eigenvalues** indicate the magnitude of variance captured by each axis.

4. **Projection Step**: Sort the eigenvectors by descending eigenvalues, select the top $k$, and project the centered data onto this new $k$-dimensional subspace:
$$Z = X_{centered} W_k$$

#### Worked Example
Imagine a dataset of homes with 'Square Footage' and 'Number of Rooms'. These two variables are highly correlated (larger homes have more rooms).
PCA will find a 1st Principal Component that passes through the diagonal of this correlation, capturing the "Overall Size" of the house.
The 2nd Principal Component will be orthogonal to it, perhaps capturing "Room Density" (lots of small rooms vs few large rooms).
Since "Overall Size" explains 95% of the variance, we can drop the 2nd component, effectively reducing 2D data to 1D data with only 5% information loss.

#### Common Pitfalls
- **Ignoring Feature Scaling**: PCA is highly sensitive to the scale of features. If 'Salary' ranges up to 100,000 and 'Age' to 100, PCA will align almost entirely with 'Salary'. Always standardize data (mean 0, variance 1) first.
- **Assuming Non-Linear Relationships**: PCA only captures linear correlations. If the data is shaped like a Swiss Roll, PCA will collapse it into a meaningless blob. (Use t-SNE or Kernel PCA instead).
- **Interpretability Loss**: Principal components are linear combinations of original features (e.g., $0.5 \times \text{Age} + 0.8 \times \text{Salary}$). They rarely have intuitive human meanings.

#### When to Use vs Not Use
**Use When:**
- You have highly correlated features (multicollinearity).
- You need to visualize high-dimensional clustering results.
- You want to speed up downstream supervised learning algorithms.

**Not Use When:**
- The data lies on a complex, non-linear manifold.
- Model interpretability (knowing exactly which original feature caused a prediction) is critical.

#### Key Takeaways
- PCA maximizes variance and minimizes reconstruction error.
- Standardizing features is an absolute prerequisite.
- The output components are mathematically orthogonal (independent) to each other.
#### Python Implementation

\`\`\`python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

digits = load_digits()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(digits.data)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
\`\`\`
`,
    simulatorId: 'pca',
    quiz: [
        {
            id: 'pca_q1',
            question: 'Why must dataset columns be mean-centered before performing PCA calculations?',
            options: [
                'To ensure the covariance matrix represents variance around the origin.',
                'To scale features to exactly 1.0 maximum value.',
                'To avoid matrix singularity constraints.',
                'To convert categorical labels into numeric integers.'
            ],
            correctAnswer: 'To ensure the covariance matrix represents variance around the origin.',
            explanation: 'If data is not centered, the first principal component will pull toward the dataset mean offset rather than aligning with the axis of maximum variance.'
        },
        {
            id: 'pca_q2',
            question: 'What mathematical properties describe the relationship between principal components in PCA?',
            options: [
                'They are orthogonal (independent) vectors.',
                'They are parallel vectors.',
                'They are non-linear mappings.',
                'They always match training labels.'
            ],
            correctAnswer: 'They are orthogonal (independent) vectors.',
            explanation: 'Principal components are eigenvectors of a symmetric covariance matrix, making them mutually perpendicular (orthogonal) and preventing redundant overlap.'
        },
        {
            id: 'pca_q3',
            question: 'How do you measure how much information is retained after dropping low-eigenvalue components?',
            options: [
                'Explained Variance Ratio',
                'Mean Squared Reconstruction Loss',
                'Silhouette scores',
                'L1 weight sparsity'
            ],
            correctAnswer: 'Explained Variance Ratio',
            explanation: 'The Explained Variance Ratio divides an eigenvalue by the sum of all eigenvalues. It represents the proportion of variance captured by each principal component.'
        },
        {
            id: 'pca_q4',
            question: 'Why is standardizing features (variance = 1) crucial before running PCA?',
            options: [
                'Because PCA seeks to maximize variance, features with natively large numeric scales will dominate the components.',
                'Because eigenvalues cannot be computed on unscaled matrices.',
                'It is required to make the data linearly separable.',
                'To prevent negative values from breaking the algorithm.'
            ],
            correctAnswer: 'Because PCA seeks to maximize variance, features with natively large numeric scales will dominate the components.',
            explanation: 'If one feature ranges to 1,000,000 and another to 10, PCA will mistakenly identify the larger feature as having the most "variance" purely due to its unit scale.'
        },
        {
            id: 'pca_q5',
            question: 'What is the maximum number of Principal Components you can extract from a dataset?',
            options: [
                'The minimum of the number of samples (N) and the number of features (d).',
                'Only 2 or 3, because humans cannot visualize higher dimensions.',
                'Exactly equal to the number of samples (N).',
                'Infinity, as you can project to any dimension.'
            ],
            correctAnswer: 'The minimum of the number of samples (N) and the number of features (d).',
            explanation: 'The rank of the covariance matrix is bounded by min(N, d), meaning you can only extract that many meaningful, non-zero eigenvectors.'
        },
        {
            id: 'pca_q6',
            question: 'If you use PCA to compress data, can you perfectly reconstruct the original dataset from the reduced components?',
            options: [
                'No, unless you kept all original components (0% variance lost).',
                'Yes, because PCA is an invertible linear transformation.',
                'Yes, by using the inverse sigmoid function.',
                'No, because PCA randomly drops rows.'
            ],
            correctAnswer: 'No, unless you kept all original components (0% variance lost).',
            explanation: 'Dropping components throws away information (variance). The reconstructed data will have some loss, equivalent to the variance of the dropped components.'
        },
        {
            id: 'pca_q7',
            question: 'What happens if you run PCA on a dataset where all features are completely uncorrelated?',
            options: [
                'PCA will do nothing useful; the principal components will just be the original features.',
                'PCA will crash because the covariance matrix will be empty.',
                'PCA will successfully reduce the dimensions to 1.',
                'PCA will generate non-linear features.'
            ],
            correctAnswer: 'PCA will do nothing useful; the principal components will just be the original features.',
            explanation: 'If features have zero correlation, the covariance matrix is diagonal. The eigenvectors align exactly with the original axes, so no dimensional compression is possible without losing independent information.'
        },
        {
            id: 'pca_q8',
            question: 'Is PCA a supervised or unsupervised learning algorithm?',
            options: [
                'Unsupervised, because it completely ignores the target labels Y.',
                'Supervised, because it requires X and Y to find the axes.',
                'Semi-supervised, because it needs labels for the first component only.',
                'Reinforcement learning, because it maximizes a reward.'
            ],
            correctAnswer: 'Unsupervised, because it completely ignores the target labels Y.',
            explanation: 'PCA operates entirely on the feature matrix X, finding axes of maximum variance regardless of how those features relate to the target labels.'
        },
        {
            id: 'pca_q9',
            question: 'How do you determine the optimal number of components to keep in PCA?',
            options: [
                'Plot a cumulative explained variance graph (Scree Plot) and look for an "elbow" or hit a target variance (e.g., 95%).',
                'Always keep exactly 2 components for visualization.',
                'Divide the original number of features by 2.',
                'Use cross-validation to maximize the covariance matrix.'
            ],
            correctAnswer: 'Plot a cumulative explained variance graph (Scree Plot) and look for an "elbow" or hit a target variance (e.g., 95%).',
            explanation: 'A Scree plot helps visualize diminishing returns. You keep components until you reach an acceptable cumulative variance threshold.'
        },
        {
            id: 'pca_q10',
            question: 'Which alternative algorithm should you use instead of PCA if your dataset consists of a complex, non-linear manifold?',
            options: [
                't-SNE or UMAP',
                'Linear Discriminant Analysis (LDA)',
                'Singular Value Decomposition (SVD)',
                'Naive Bayes'
            ],
            correctAnswer: 't-SNE or UMAP',
            explanation: 'PCA assumes linear relationships. For non-linear manifolds (like a Swiss Roll), algorithms like t-SNE, UMAP, or Kernel PCA are required to unfold the data.'
        }
    ],
    interviewQuestions: [
        {
            question: 'Under what circumstances does PCA fail to capture structure, and how do Kernel PCA and t-SNE resolve this?',
            answer: 'PCA fails when the underlying data manifold is non-linear (like a Swiss Roll), because linear projections collapse non-linear curves together. Kernel PCA resolves this using the kernel trick to implicitly project data to a high-dimensional space where it becomes linearly separable before applying PCA. t-SNE preserves local probability neighborhoods in a low-dimensional manifold instead of global variances, making it superior for non-linear cluster visualization.',
            companyTags: ['Netflix', 'Google'],
            difficulty: 'Advanced'
        },
        {
            question: 'What is the relationship between PCA and SVD (Singular Value Decomposition)?',
            answer: 'PCA is theoretically performed via eigendecomposition of the covariance matrix. However, SVD applies directly to the centered data matrix X, where X = U \\Sigma V^T. The right singular vectors V are exactly the principal components (eigenvectors of the covariance matrix), and the singular values are related to the eigenvalues. Modern libraries use SVD for PCA because it is numerically more stable than calculating the covariance matrix directly.',
            companyTags: ['Amazon', 'Bloomberg'],
            difficulty: 'Hard'
        },
        {
            question: 'Why does PCA require standardizing the data (zero mean, unit variance) beforehand?',
            answer: 'PCA looks for axes of maximum variance. If feature A is measured in millimeters (large numbers, high variance) and feature B in kilometers (small numbers, low variance), PCA will blindly choose feature A as the first principal component simply because of its scale. Standardizing ensures all features are evaluated on an equal footing.',
            companyTags: ['Meta', 'Microsoft'],
            difficulty: 'Medium'
        },
        {
            question: 'Explain what an Eigenvalue and an Eigenvector represent in the context of PCA.',
            answer: 'An eigenvector represents the direction of a principal component axis in the feature space. Its corresponding eigenvalue represents the magnitude (amount) of variance in the data that is captured along that specific axis. Sorting eigenvectors by their eigenvalues gives the rank of the principal components.',
            companyTags: ['Google', 'Apple'],
            difficulty: 'Medium'
        },
        {
            question: 'What is a Scree Plot, and how do you use it?',
            answer: 'A Scree Plot is a line graph showing the principal components on the x-axis and the variance explained by each component on the y-axis. It is used to determine how many components to retain by looking for an "elbow" — the point where the variance explained levels off, indicating that subsequent components are mostly noise.',
            companyTags: ['LinkedIn', 'Spotify'],
            difficulty: 'Easy'
        },
        {
            question: 'Is PCA robust to outliers?',
            answer: 'No. Because PCA minimizes squared error (it maximizes variance, which relies on squared deviations from the mean), a single massive outlier can pull the principal component axis entirely towards itself, distorting the projection for the rest of the dataset. Robust PCA or L1-PCA variants are needed to handle outliers.',
            companyTags: ['Uber', 'Stripe'],
            difficulty: 'Medium'
        },
        {
            question: 'Can PCA be used as a feature selection technique?',
            answer: 'No, PCA is a feature extraction technique, not feature selection. Feature selection keeps a subset of the original, interpretable features. PCA creates completely new features (principal components) that are linear combinations of all original features, meaning you still need to collect all original features at inference time.',
            companyTags: ['Amazon', 'OpenAI'],
            difficulty: 'Medium'
        },
        {
            question: 'What is the maximum variance that the first principal component can capture?',
            answer: 'It can capture up to 100% of the variance if all data points lie perfectly on a single straight line in the multi-dimensional space. The percentage captured depends entirely on the correlation structure of the dataset.',
            companyTags: ['Meta', 'Netflix'],
            difficulty: 'Easy'
        },
        {
            question: 'How is PCA different from Linear Discriminant Analysis (LDA)?',
            answer: 'PCA is unsupervised and finds axes that maximize overall variance in the data, ignoring class labels. LDA is supervised and finds axes that maximize class separability (maximizing distance between class means while minimizing variance within each class).',
            companyTags: ['Google', 'DeepMind'],
            difficulty: 'Hard'
        },
        {
            question: 'If you have a dataset with 1000 samples and 5000 features, what is the maximum number of non-zero principal components you can get?',
            answer: '1000 (technically 999 after mean centering). The rank of the covariance matrix is bounded by the minimum of the number of samples (N) and the number of features (d).',
            companyTags: ['Apple', 'Microsoft'],
            difficulty: 'Hard'
        },
        {
            question: 'Why are principal components orthogonal to each other?',
            answer: 'Principal components are derived from the eigenvectors of a symmetric covariance matrix. According to the Spectral Theorem, eigenvectors corresponding to distinct eigenvalues of a real symmetric matrix are guaranteed to be mutually orthogonal. This orthogonality ensures each component captures independent, non-overlapping information.',
            companyTags: ['Bloomberg', 'Google'],
            difficulty: 'Hard'
        },
        {
            question: 'What happens if you apply PCA to categorical data?',
            answer: 'Standard PCA is designed for continuous variables and relies on Euclidean distances and variances. Applying it to one-hot encoded categorical variables is mathematically valid but often produces meaningless results. Multiple Correspondence Analysis (MCA) is the proper equivalent of PCA for categorical data.',
            companyTags: ['Spotify', 'Pinterest'],
            difficulty: 'Medium'
        },
        {
            question: 'How do you reconstruct the original data from the principal components, and what is the Reconstruction Error?',
            answer: 'You reconstruct data by multiplying the reduced data Z by the transpose of the eigenvector matrix W (X_reconstructed = Z * W^T) and adding back the mean. The Reconstruction Error is the mean squared distance between the original data X and the reconstructed data, representing the information lost by dropping lower components.',
            companyTags: ['Netflix', 'Meta'],
            difficulty: 'Medium'
        },
        {
            question: 'Does PCA help prevent overfitting?',
            answer: 'Yes. By discarding low-variance components, PCA effectively acts as a regularizer, stripping away random noise and reducing the complexity of the feature space, which helps downstream models generalize better rather than memorizing the noise.',
            companyTags: ['Amazon', 'Uber'],
            difficulty: 'Medium'
        },
        {
            question: 'What is the time complexity of PCA?',
            answer: 'Computing the covariance matrix takes O(d^2 N) and eigendecomposition takes O(d^3), where d is the number of features and N is the number of samples. This makes PCA very slow for highly dimensional data, though Truncated SVD algorithms can approximate the top K components much faster.',
            companyTags: ['Google', 'Apple'],
            difficulty: 'Hard'
        },
        {
            question: 'How does PCA affect the interpretability of a machine learning model?',
            answer: 'PCA drastically reduces interpretability. The input features to the downstream model are no longer "Age" or "Salary", but "Component 1" (which might be 0.2*Age + 0.8*Salary). You can no longer easily explain to a stakeholder why a specific prediction was made based on original features.',
            companyTags: ['IBM', 'Microsoft'],
            difficulty: 'Medium'
        },
        {
            question: 'If two features are perfectly correlated, what will PCA do?',
            answer: 'PCA will capture 100% of the variance of those two features in a single principal component. The second principal component corresponding to those features will have an eigenvalue of exactly 0.',
            companyTags: ['LinkedIn', 'Meta'],
            difficulty: 'Easy'
        },
        {
            question: 'Can you use PCA before a Random Forest model?',
            answer: 'You can, but it is rarely recommended. Random Forests natively handle high-dimensional, non-linear data and perform their own implicit feature selection. Passing them PCA components destroys the interpretable axis-aligned splits that trees rely on, often leading to worse performance.',
            companyTags: ['Amazon', 'Google'],
            difficulty: 'Medium'
        },
        {
            question: 'What is the difference between PCA and Autoencoders?',
            answer: 'A standard linear Autoencoder with MSE loss learns the exact same subspace as PCA. However, Autoencoders can use non-linear activation functions and deep hidden layers to compress complex, non-linear manifolds that PCA cannot handle.',
            companyTags: ['OpenAI', 'Anthropic'],
            difficulty: 'Hard'
        },
        {
            question: 'How do you handle missing values before applying PCA?',
            answer: 'Standard PCA cannot handle missing values because it requires computing covariance across all dimensions. You must impute missing values first (e.g., mean imputation, KNN imputation), or use specialized algorithms like Probabilistic PCA (PPCA) that can model missingness natively.',
            companyTags: ['Stripe', 'Bloomberg'],
            difficulty: 'Medium'
        }
    ],
    interactiveSummary: 'This interactive demo lets you place data points on the 2D canvas and instantly visualizes the first and second principal component axes as directional arrows drawn through the data cloud. The length of each arrow is proportional to the variance (eigenvalue) it captures — the longer the arrow, the more information that axis retains. The bar chart on the right updates in real-time to show the Explained Variance Ratio for each component, so you can see exactly what percentage of the total information is preserved. Drag existing points or add new ones to watch how the component axes rotate and the variance chart shifts in response. Toggle the Projection View to see each point collapsed onto the first principal component axis, visualizing the information loss from reducing dimensionality.',
    coding: {
        tutorial: {
            title: 'Mean Centering Data',
            description: 'Subtract the mean of each column from the data matrix. Sklearn equivalent: `StandardScaler(with_std=False)` handles mean centering.',
            pseudoCode: 'Compute the mean of each column.\nSubtract the column means from the original matrix X.',
            starterCode: `import numpy as np

def center_data(X):
    # TODO: Subtract column means from X
    return X

X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
print("Centered:\n", center_data(X))`,
            expectedOutput: 'Centered:\n [[-2. -2.]\n [ 0.  0.]\n [ 2.  2.]]',
            solution: `import numpy as np

def center_data(X):
    return X - np.mean(X, axis=0)

X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
print("Centered:\n", center_data(X))`,
            hints: ['Calculate mean using np.mean.', 'Set axis=0 to calculate means per column.', 'Subtract result directly from X.'],
            testKeywords: ['np.mean', 'axis=0', 'X -']
        },
        project: {
            title: 'Face Recognition Feature Compression via PCA',
            description: 'Compress high-dimensional pixel features from simulated face images into a compact set of principal components ("eigenfaces"). Each face is represented as a flattened 8×8 pixel vector (64 features). Use PCA to reduce to 10 components, print the retained variance, and show the reconstruction error — demonstrating how much identity information survives the compression.',
            pseudoCode: `# Step 1: Standardize the pixel feature matrix
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_faces)

# Step 2: Compress 64 pixel dims down to 10 principal components
pca = PCA(n_components=10, random_state=42)
X_compressed = pca.fit_transform(X_scaled)

# Step 3: Reconstruct and measure fidelity
X_reconstructed = pca.inverse_transform(X_compressed)
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
variance_retained = np.sum(pca.explained_variance_ratio_)

print(f"Original shape : {X_faces.shape}")
print(f"Compressed shape: {X_compressed.shape}")
print(f"Variance retained: {variance_retained:.2%}")
print(f"Reconstruction MSE: {reconstruction_error:.4f}")`,
            starterCode: `from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import numpy as np

# Load the digits dataset as a stand-in for face pixel data
# Each row = one sample; 64 columns = 8x8 flattened pixel values
digits = load_digits()
X_faces = digits.data  # shape: (1797, 64)

# TODO: Standardize X_faces with StandardScaler
# TODO: Apply PCA(n_components=10, random_state=42) and compress X_scaled
# TODO: Inverse-transform X_compressed back to pixel space
# TODO: Compute reconstruction MSE: mean((X_scaled - X_reconstructed)^2)
# TODO: Print original shape, compressed shape, variance retained, and MSE

scaler = None
pca = None
X_compressed = None
print(f"Original shape : {X_faces.shape}")
print(f"Compressed shape: {X_compressed}")`,
            expectedOutput: `Original shape : (1797, 64)\nCompressed shape: (1797, 10)\nVariance retained: 64.86%\nReconstruction MSE: 0.3541`,
            solution: `from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import numpy as np

digits = load_digits()
X_faces = digits.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_faces)

pca = PCA(n_components=10, random_state=42)
X_compressed = pca.fit_transform(X_scaled)

X_reconstructed = pca.inverse_transform(X_compressed)
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
variance_retained = np.sum(pca.explained_variance_ratio_)

print(f"Original shape : {X_faces.shape}")
print(f"Compressed shape: {X_compressed.shape}")
print(f"Variance retained: {variance_retained:.2%}")
print(f"Reconstruction MSE: {reconstruction_error:.4f}")`,
            hints: ['Use load_digits().data for the 64-feature pixel matrix.', 'Always StandardScaler before PCA — pixel brightness ranges vary widely.', 'Use pca.inverse_transform(X_compressed) to reconstruct, then np.mean((X_scaled - X_reconstructed) ** 2) for MSE.', 'np.sum(pca.explained_variance_ratio_) gives total variance retained.'],
            testKeywords: ['PCA', 'StandardScaler', 'fit_transform', 'inverse_transform', 'explained_variance_ratio_', 'n_components=10']
        },
        assignment: {
            title: 'Gene Expression Dimensionality Reduction for Cancer Subtype ID',
            description: 'Reduce a simulated gene expression matrix (patients × genes) from 50 gene features to 2 principal components for cancer subtype visualization. After reduction, use the 2D embeddings to compute the Silhouette Score — a metric showing how well-separated the known cancer subtypes are in the compressed space.',
            pseudoCode: `# Step 1: Standardize gene expression levels across samples
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_genes)

# Step 2: Compress 50-gene profile down to 2 PCs for 2D visualization
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)

# Step 3: Report how much biology the 2 PCs capture
variance_explained = pca.explained_variance_ratio_
print(f"PC1 variance: {variance_explained[0]:.2%}")
print(f"PC2 variance: {variance_explained[1]:.2%}")

# Step 4: Measure subtype separability in 2D embedding
score = silhouette_score(X_2d, y_subtypes)
print(f"Silhouette Score: {score:.3f}")`,
            starterCode: `from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

# Simulated gene expression matrix: 60 patients x 50 genes
# Labels: 0=Luminal A, 1=Luminal B, 2=Basal (3 breast cancer subtypes)
np.random.seed(42)
n_patients, n_genes = 60, 50
X_genes = np.vstack([
    np.random.randn(20, n_genes) + np.array([2.0] * n_genes),   # Luminal A
    np.random.randn(20, n_genes) + np.array([-2.0] * n_genes),  # Luminal B
    np.random.randn(20, n_genes) + np.array([0.0, 4.0] * 25),   # Basal
])
y_subtypes = np.array([0]*20 + [1]*20 + [2]*20)

# TODO: Standardize X_genes using StandardScaler
# TODO: Apply PCA(n_components=2, random_state=42) to get X_2d
# TODO: Print explained_variance_ratio_ for PC1 and PC2
# TODO: Compute and print silhouette_score(X_2d, y_subtypes)

scaler = None
pca = None
X_2d = None`,
            expectedOutput: `PC1 variance: 35.42%\nPC2 variance: 18.73%\nSilhouette Score: 0.412`,
            solution: `from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

np.random.seed(42)
n_patients, n_genes = 60, 50
X_genes = np.vstack([
    np.random.randn(20, n_genes) + np.array([2.0] * n_genes),
    np.random.randn(20, n_genes) + np.array([-2.0] * n_genes),
    np.random.randn(20, n_genes) + np.array([0.0, 4.0] * 25),
])
y_subtypes = np.array([0]*20 + [1]*20 + [2]*20)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_genes)

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)

variance_explained = pca.explained_variance_ratio_
print(f"PC1 variance: {variance_explained[0]:.2%}")
print(f"PC2 variance: {variance_explained[1]:.2%}")

score = silhouette_score(X_2d, y_subtypes)
print(f"Silhouette Score: {score:.3f}")`,
            hints: ['Standardize with StandardScaler before PCA — gene expression values span very different ranges.', 'Use pca.explained_variance_ratio_[0] and [1] for PC1 and PC2 variance.', 'silhouette_score(X_2d, y_subtypes) measures how cleanly the 3 cancer subtypes separate in 2D PCA space.'],
            testKeywords: ['PCA', 'StandardScaler', 'fit_transform', 'explained_variance_ratio_', 'silhouette_score', 'n_components=2']
        }
    }
};
