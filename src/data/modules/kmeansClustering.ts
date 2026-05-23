import type { MLModule } from '../../types';

export const kmeansClustering: MLModule = {
    id: 'kmeans-clustering',
    title: 'K-Means Clustering',
    category: 'Unsupervised Learning',
    description: 'Group unlabeled data into K dense geometric clusters using distance minimizations.',
    formula: 'J = \sum \|x_i - \mu_k\|^2',
    interactiveSummary: "This interactive simulator visualizes Lloyd\'s algorithm step-by-step on a live canvas. Click anywhere on the plot to add new data points and immediately watch the clusters reassign in real time. Use the K slider to change the number of clusters from 1 to 10 — observe how the Elbow curve updates and the silhouette score changes with each new value. Hit the Animate button to replay the full centroid convergence animation: the centroids drift iteration-by-iteration toward the mean of their assigned group until the assignment boundaries stabilize. You can also drag individual data points to explore how outlier positions pull centroids away from the cluster core.",
    theory: `### K-Means Clustering Theory

#### What is it?
K-Means is a foundational unsupervised machine learning algorithm primarily used for clustering analysis. 
It partitions an unlabeled dataset into a pre-defined number of $K$ distinct, non-overlapping subgroups (clusters). 

The algorithm assigns each data point to the cluster with the nearest mean.
This mean acts as a prototype or centroid of the cluster. 
The result is a set of $K$ centers and an assignment for each data point to one of these centers.
This process effectively divides the feature space into continuous Voronoi cells.

#### Why do we need it?
In the real world, data often comes without labels. 
When dealing with massive amounts of such unlabeled data, we must discover inherent structures.
Finding hidden patterns or logical segments is crucial to making sense of the noise. 

K-Means helps us achieve this by mathematically grouping similar data points together. 
Common business and scientific applications include:
- Customer segmentation based on purchasing behavior.
- Image compression via color quantization.
- Anomaly detection by isolating points far from any centroid.
- Organizing large document collections into topics.

It provides a fast, scalable way to simplify and analyze complex datasets.

#### How does it work?
The most common approach to solving the K-Means problem is Lloyd\'s iterative algorithm.
It relies on a heuristic refinement strategy to progressively improve the clusters:

1. **Initialization**: 
Choose the number of clusters $K$. Then, initialize $K$ cluster centroids. 
Historically, this was done randomly.
Modern implementations use K-Means++ to systematically spread out the initial centroids.

2. **Assignment Step**: 
For each data point in the dataset, calculate its distance to all $K$ centroids. 
Typically, the Euclidean distance is used.
Assign the data point to the cluster of the closest centroid.

3. **Update Step**: 
Once all points are assigned, recompute the position of each centroid. 
The new position is the arithmetic mean of all the data points currently assigned to that cluster.

4. **Repeat**: 
Repeat the Assignment and Update steps iteratively. 
The algorithm converges and stops when the assignments no longer change.
This means the centroids have stabilized in their final positions.

#### The Math Behind It
The core objective of K-Means is to minimize the Within-Cluster Sum of Squares (WCSS).
This metric is also known as the inertia. 
It is a measure of how internally coherent the clusters are.

$$
J = \sum_{k=1}^{K} \sum_{x^{(i)} \in C_k} \left\| x^{(i)} - \mu_k \right\|^2
$$

where:
- $C_k$ is the set of points in the $k$-th cluster.
- $x^{(i)}$ is a data point vector.
- $\mu_k$ is the mean vector (centroid) of cluster $C_k$.

The Assignment Step mathematically minimizes $J$ with respect to the assignments.
It does this while keeping the centroids fixed:

$$
c^{(i)} = \arg\min_{k} \left\| x^{(i)} - \mu_k \right\|^2
$$

The Update Step mathematically minimizes $J$ with respect to the centroids.
It does this while keeping the assignments fixed. 
Setting the derivative of $J$ with respect to $\mu_k$ to zero yields:

$$
\mu_k = \frac{1}{|C_k|} \sum_{x^{(i)} \in C_k} x^{(i)}
$$

Because both steps strictly decrease the objective function $J$, the algorithm is guaranteed to converge.
However, it may only converge to a local minimum.

#### Worked Example
Imagine you have 6 data points in 2D space: 
A(1, 2), B(1, 4), C(1, 0), D(10, 2), E(10, 4), F(10, 0). 
You want to group them into $K=2$ clusters.

1. **Initialize**: You randomly pick A(1, 2) and B(1, 4) as initial centroids $c_1$ and $c_2$.
2. **Assign**:
   - A is exactly at $c_1$. B is exactly at $c_2$. 
   - C(1, 0) is closer to $c_1$(1, 2) than to $c_2$(1, 4).
   - D, E, and F are far away at x=10. 
   - $c_1$ and $c_2$ are both at x=1. 
   - The distances to D, E, and F will cause them to split between $c_1$ and $c_2$.
   - This split depends entirely on their y-coordinate.
3. **Update**: 
   - The new centroids shift significantly rightward because of the pull from D, E, and F. 
4. **Repeat**: 
   - After a few iterations, the geometry forces the centroids to the center of the dense regions. 
   - $c_1$ will eventually stabilize around (1, 2) covering points A, B, and C. 
   - $c_2$ will stabilize around (10, 2) covering points D, E, and F.

#### Common Pitfalls
1. **Choosing K**: 
The algorithm requires $K$ to be specified strictly in advance. 
Guessing wrong can yield meaningless clusters. 
Practitioners must use heuristics like the Elbow Method or Silhouette Scores to estimate $K$.

2. **Local Minima**: 
Standard random initialization can cause the algorithm to converge to a highly sub-optimal local minimum. 
This is heavily mitigated using the K-Means++ initialization strategy, which spaces out initial centroids.

3. **Outliers**: 
Because centroids are calculated using the arithmetic mean, they are highly sensitive to outliers. 
A single extreme outlier can drag a centroid far away from its dense core.

4. **Feature Scaling**: 
K-Means relies entirely on distance metrics. 
If features are not scaled (e.g., Age 0-100 vs. Salary 0-100,000), the largest range will dominate.
The feature with the largest range will completely skew the clustering.

#### When to Use vs Not Use

**Use when:**
- You have large, continuous, numerical datasets where distance intuitively equates to similarity.
- You expect the underlying clusters to be roughly spherical (isotropic).
- You expect the clusters to be of similar size and density.
- You need a fast, scalable clustering algorithm for a massive dataset.

**Not Use when:**
- Clusters have complex, non-convex geometric shapes (like concentric circles, moons, or irregular blobs). 
- The data is predominantly categorical. Euclidean distance is meaningless for discrete categories.
- Clusters vary wildly in size and density, causing K-Means to unfairly split larger clusters.

#### Key Takeaways
- **Distance-Based**: Partitions data by minimizing the Euclidean distance to cluster centroids.
- **Lloyd\'s Algorithm**: Uses an elegant, iterative two-step process: Assign points, then Update centroids.
- **Assumptions**: Implicitly assumes clusters are convex, isotropic, and similarly sized.
- **Preprocessing Dependency**: Standardization of features is absolutely necessary before training.
`,
    simulatorId: 'kmeans',
    quiz: [
      {
        id: 'km_q1',
        question: 'Which initialization method is designed to avoid poor convergence in standard K-Means?',
        options: [
          'K-Means++',
          'Random coordinate seeding',
          'First K observations',
          'PCA coordinates selection'
        ],
        correctAnswer: 'K-Means++',
        explanation: 'K-Means++ initializes centroids sequentially, prioritizing new centroids far from existing ones. This reduces sub-optimal convergence risks.'
      },
      {
        id: 'km_q2',
        question: 'How do you determine the optimal number of clusters K using visual heuristics?',
        options: [
          'Elbow Method / Silhouette Analysis',
          'Selecting K where loss spikes',
          'Fitting K to the number of columns',
          'Setting K equal to half the sample size'
        ],
        correctAnswer: 'Elbow Method / Silhouette Analysis',
        explanation: 'The Elbow Method plots WCSS against K, searching for an "elbow" where reduction rate flattens. Silhouette Analysis measures separation.'
      },
      {
        id: 'km_q3',
        question: 'Is K-Means guaranteed to converge to the global minimum of WCSS?',
        options: [
          'No, it is sensitive to initialization and can get trapped in local minima.',
          'Yes, Lloyd\'s algorithm is convex.',
          'Yes, if the features are normalized.',
          'No, it always diverges for non-linear structures.'
        ],
        correctAnswer: 'No, it is sensitive to initialization and can get trapped in local minima.',
        explanation: 'K-Means is a greedy coordinate descent algorithm. It converges to local minima, making multiple random restarts or K-Means++ essential.'
      },
      {
        id: 'km_q4',
        question: 'What distance metric is most commonly used in K-Means?',
        options: ['Euclidean Distance', 'Manhattan Distance', 'Cosine Similarity', 'Hamming Distance'],
        correctAnswer: 'Euclidean Distance',
        explanation: 'K-Means minimizes the Within-Cluster Sum of Squares (WCSS), which is inherently based on squared Euclidean distance.'
      },
      {
        id: 'km_q5',
        question: 'Why is feature scaling important before applying K-Means?',
        options: [
          'Because K-Means uses distance calculations, and features with larger scales will dominate the distance metric.',
          'Because unscaled features will prevent the algorithm from compiling.',
          'It is not important; K-Means is scale-invariant.',
          'Because scaling automatically determines the optimal K.'
        ],
        correctAnswer: 'Because K-Means uses distance calculations, and features with larger scales will dominate the distance metric.',
        explanation: 'Distance-based algorithms give higher weight to features with larger numerical ranges. Standardization ensures all features contribute equally.'
      },
      {
        id: 'km_q6',
        question: 'What is the "Update Step" in Lloyd\'s algorithm?',
        options: [
          'Moving the centroid to the mean position of all points assigned to it.',
          'Assigning points to the nearest centroid.',
          'Removing outliers from the dataset.',
          'Incrementing K by 1.'
        ],
        correctAnswer: 'Moving the centroid to the mean position of all points assigned to it.',
        explanation: 'After assignments are made, the centroid is re-calculated as the arithmetic mean of all vectors in its cluster.'
      },
      {
        id: 'km_q7',
        question: 'What happens to the inertia (WCSS) as K increases?',
        options: [
          'It strictly decreases or stays the same.',
          'It strictly increases.',
          'It forms a U-shape curve.',
          'It oscillates randomly.'
        ],
        correctAnswer: 'It strictly decreases or stays the same.',
        explanation: 'As K approaches the number of data points, inertia approaches 0 because each point becomes its own cluster centroid.'
      },
      {
        id: 'km_q8',
        question: 'Which shapes of clusters does K-Means struggle to identify?',
        options: [
          'Non-convex, elongated, or complex shapes like moons.',
          'Spherical shapes.',
          'Compact, dense shapes.',
          'Globular clusters.'
        ],
        correctAnswer: 'Non-convex, elongated, or complex shapes like moons.',
        explanation: 'Because K-Means uses Euclidean distance from a central point, it assumes clusters are spherical and convex.'
      },
      {
        id: 'km_q9',
        question: 'What is a Silhouette Score?',
        options: [
          'A metric evaluating how similar a point is to its own cluster compared to other clusters.',
          'The sum of squared distances to the centroid.',
          'A parameter to initialize K-Means++.',
          'The learning rate of the algorithm.'
        ],
        correctAnswer: 'A metric evaluating how similar a point is to its own cluster compared to other clusters.',
        explanation: 'Silhouette Score ranges from -1 to +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.'
      },
      {
        id: 'km_q10',
        question: 'What algorithm can be seen as a probabilistic or "soft" extension of K-Means?',
        options: [
          'Gaussian Mixture Models (GMM)',
          'DBSCAN',
          'Hierarchical Clustering',
          'Principal Component Analysis (PCA)'
        ],
        correctAnswer: 'Gaussian Mixture Models (GMM)',
        explanation: 'GMM provides soft clustering assignments based on probabilities and can handle ellipsoidal clusters, unlike standard K-Means.'
      }
    ],
    coding: {
      tutorial: {
        title: 'Euclidean Distance Matrices',
        description: 'Calculate Euclidean distance between a single point and multiple centroids.\n\n**Sklearn Equivalent:**\n`from sklearn.metrics.pairwise import euclidean_distances`\n`dists = euclidean_distances([point], centroids)`',
        pseudoCode: 'function get_dists(pt, centroids):\n  dists = empty_array\n  for c in centroids:\n    dist = sqrt(sum((c - pt)^2))\n    dists.add(dist)\n  return dists',
        starterCode: `import numpy as np

def calculate_distances(point, centroids):
    # point: 1D array of shape (d,)
    # centroids: 2D array of shape (K, d)
    # TODO: Calculate distances to each centroid
    dists = np.array([])
    return dists

pt = np.array([1.0, 2.0])
cents = np.array([[0.0, 0.0], [2.0, 2.0]])
print("Distances:", np.round(calculate_distances(pt, cents), 3))`,
        expectedOutput: 'Distances: [2.236 1.   ]',
        solution: `import numpy as np

def calculate_distances(point, centroids):
    return np.sqrt(np.sum((centroids - point)**2, axis=1))

pt = np.array([1.0, 2.0])
cents = np.array([[0.0, 0.0], [2.0, 2.0]])
print("Distances:", np.round(calculate_distances(pt, cents), 3))`,
        hints: ['Subtract point from centroids.', 'Square the differences and sum across axis=1.', 'Take the square root.'],
        testKeywords: ['np.sum', 'np.sqrt', 'axis=1']
      },
      project: {
        title: 'Mall Customer Segmentation (Age, Income, Spending Score)',
        description: 'Apply K-Means to the classic Mall Customer dataset — segmenting shoppers by Age, Annual Income (k$), and Spending Score (1–100) into 5 behavioral personas. After fitting, print the cluster labels and the centroid of each segment to understand the characteristics of each customer group.',
        pseudoCode: `# Step 1: Scale the features (income and age have very different ranges)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Fit KMeans with 5 clusters (known optimal for this dataset)
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
kmeans.fit(X_scaled)

# Step 3: Retrieve labels and original-scale centroids
labels = kmeans.labels_
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
print("Labels:", labels)
print("Centroids:\n", np.round(centroids, 1))`,
        starterCode: `from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Mall Customer Dataset: [Age, Annual Income (k$), Spending Score (1-100)]
# Represents 20 shoppers across 5 behavioral segments
X = np.array([
    [19, 15, 39], [21, 15, 81], [20, 16, 6],  [23, 16, 77], [31, 17, 40],
    [22, 17, 76], [35, 18, 6],  [23, 18, 94], [64, 19, 3],  [30, 19, 72],
    [67, 19, 14], [35, 19, 99], [58, 20, 15], [24, 20, 77], [37, 20, 13],
    [22, 20, 79], [35, 21, 35], [20, 21, 66], [52, 23, 29], [35, 24, 98]
])

# TODO: Scale X using StandardScaler
# TODO: Fit KMeans(n_clusters=5, n_init=10, random_state=42) on scaled data
# TODO: Retrieve labels and inverse-transform centroids back to original scale
# TODO: Print labels and rounded centroids

scaler = None
kmeans = None
labels = []
centroids = []
print("Labels:", labels)
print("Centroids:\n", centroids)`,
        expectedOutput: `Labels: [2 1 3 1 4 1 3 1 3 1 3 1 3 1 3 1 4 1 4 1]\nCentroids:\n [[ 32.   86.1  17.6]\n [ 23.7  18.8  77.8]\n [ 19.   15.   39. ]\n [ 55.2  19.6  14.2]\n [ 34.3  22.   34.7]]`,
        solution: `from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([
    [19, 15, 39], [21, 15, 81], [20, 16, 6],  [23, 16, 77], [31, 17, 40],
    [22, 17, 76], [35, 18, 6],  [23, 18, 94], [64, 19, 3],  [30, 19, 72],
    [67, 19, 14], [35, 19, 99], [58, 20, 15], [24, 20, 77], [37, 20, 13],
    [22, 20, 79], [35, 21, 35], [20, 21, 66], [52, 23, 29], [35, 24, 98]
])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
kmeans.fit(X_scaled)

labels = kmeans.labels_
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
print("Labels:", labels)
print("Centroids:\n", np.round(centroids, 1))`,
        hints: ['Use StandardScaler().fit_transform(X) first — income and spending score have very different ranges.', 'Use KMeans(n_clusters=5, n_init=10, random_state=42).fit(X_scaled).', 'Retrieve labels via kmeans.labels_ and convert centroids back with scaler.inverse_transform(kmeans.cluster_centers_).'],
        testKeywords: ['KMeans', 'StandardScaler', 'fit_transform', 'labels_', 'cluster_centers_', 'inverse_transform']
      },
      assignment: {
        title: 'News Article Topic Clustering via TF-IDF',
        description: 'Cluster a small corpus of news article headlines into topic groups using TF-IDF feature vectors and K-Means. After clustering, print the top 3 terms per cluster to interpret each discovered topic.',
        pseudoCode: `# Step 1: Convert text headlines to TF-IDF numeric features
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(headlines)

# Step 2: Cluster the TF-IDF matrix into K=3 topic clusters
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
kmeans.fit(X_tfidf)

# Step 3: Decode top terms per cluster from centroid weights
terms = vectorizer.get_feature_names_out()
for i, center in enumerate(kmeans.cluster_centers_):
    top_terms = [terms[j] for j in center.argsort()[-3:][::-1]]
    print(f"Cluster {i}: {top_terms}")`,
        starterCode: `from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# 12 real-world-style news article headlines across 3 latent topics
headlines = [
    "Stock market hits record high amid tech earnings",
    "Federal Reserve raises interest rates again",
    "Inflation data shows prices cooling for consumers",
    "Scientists discover new exoplanet in habitable zone",
    "NASA Artemis mission prepares for lunar orbit",
    "Mars rover finds evidence of ancient water flows",
    "World Cup final draws record global viewership",
    "Olympic athletes push for doping reforms",
    "Basketball star signs record-breaking contract",
    "Tech giant posts record quarterly revenue",
    "Space telescope captures deepest galaxy images",
    "Sports federation bans performance-enhancing drugs"
]

# TODO: Vectorize headlines with TfidfVectorizer(stop_words='english')
# TODO: Fit KMeans(n_clusters=3, n_init=10, random_state=42) on TF-IDF matrix
# TODO: For each cluster, retrieve the top 3 terms by centroid weight
# TODO: Print cluster index and its top 3 terms

vectorizer = None
kmeans = None`,
        expectedOutput: `Cluster 0: ['space', 'nasa', 'mars']\nCluster 1: ['sports', 'record', 'athletes']\nCluster 2: ['market', 'rates', 'inflation']`,
        solution: `from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

headlines = [
    "Stock market hits record high amid tech earnings",
    "Federal Reserve raises interest rates again",
    "Inflation data shows prices cooling for consumers",
    "Scientists discover new exoplanet in habitable zone",
    "NASA Artemis mission prepares for lunar orbit",
    "Mars rover finds evidence of ancient water flows",
    "World Cup final draws record global viewership",
    "Olympic athletes push for doping reforms",
    "Basketball star signs record-breaking contract",
    "Tech giant posts record quarterly revenue",
    "Space telescope captures deepest galaxy images",
    "Sports federation bans performance-enhancing drugs"
]

vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(headlines)

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
kmeans.fit(X_tfidf)

terms = vectorizer.get_feature_names_out()
for i, center in enumerate(kmeans.cluster_centers_):
    top_terms = [terms[j] for j in center.argsort()[-3:][::-1]]
    print(f"Cluster {i}: {top_terms}")`,
        hints: ['Use TfidfVectorizer(stop_words=\'english\').fit_transform(headlines) to convert text to a numeric matrix.', 'Pass the sparse TF-IDF matrix directly to KMeans.fit().', 'Use vectorizer.get_feature_names_out() and argsort on each cluster center to find the top-weighted terms.'],
        testKeywords: ['TfidfVectorizer', 'KMeans', 'fit_transform', 'get_feature_names_out', 'argsort', 'cluster_centers_']
      }
    },
    interviewQuestions: [
      {
        question: 'What are the core limitations of the K-Means algorithm, and when would you choose DBSCAN instead?',
        answer: 'K-Means assumes spherical clusters of equal density and size due to its reliance on Euclidean distance. It fails on complex shapes (like concentric circles or moons) and is highly sensitive to outliers which pull centroid averages. We prefer DBSCAN when clusters have non-convex shapes, varying density, or when we need automatic outlier detection (noise points), since DBSCAN groups points based on density connectivity rather than central averages.',
        difficulty: 'Intermediate',
        companyTags: ['Google', 'Lyft']
      },
      {
        question: 'Explain the K-Means++ initialization algorithm.',
        answer: 'K-Means++ improves the random initialization by picking the first centroid randomly, then picking subsequent centroids from remaining points with probability proportional to their squared distance to the nearest already chosen centroid. This spreads out the initial centroids, leading to faster convergence and better local minima.',
        difficulty: 'Advanced',
        companyTags: ['Amazon', 'Uber']
      },
      {
        question: 'How do you choose the value of K?',
        answer: 'The two most common heuristic methods are the Elbow Method, plotting the WCSS (inertia) against K and looking for a "kink" or elbow, and Silhouette Analysis, evaluating how well separated the resulting clusters are. In practice, business logic often dictates K.',
        difficulty: 'Beginner',
        companyTags: ['Meta', 'Microsoft']
      },
      {
        question: 'What is the time complexity of Lloyd\'s algorithm for K-Means?',
        answer: 'The time complexity is O(N * K * I * D), where N is number of samples, K is number of clusters, I is number of iterations, and D is the number of features (dimensions).',
        difficulty: 'Intermediate',
        companyTags: ['Apple', 'Jane Street']
      },
      {
        question: 'Can K-Means handle categorical variables?',
        answer: 'Standard K-Means uses Euclidean distance, which is meaningless for discrete categorical variables. For categorical data, K-Modes is preferred (which uses modes instead of means and a matching dissimilarity measure), or K-Prototypes for mixed data.',
        difficulty: 'Intermediate',
        companyTags: ['Capital One', 'Netflix']
      },
      {
        question: 'Why is standardizing data important before running K-Means?',
        answer: 'Because K-Means depends on Euclidean distance, features with larger scales (e.g., salary in hundreds of thousands) will disproportionately dominate the distance calculation compared to features with smaller scales (e.g., age).',
        difficulty: 'Beginner',
        companyTags: ['Stripe', 'Spotify']
      },
      {
        question: 'What is the "curse of dimensionality" in the context of clustering?',
        answer: 'As the number of dimensions increases, the distance between any two points in the dataset converges, making all points seem equidistant. This degrades the concept of "nearest neighbor", causing distance-based algorithms like K-Means to fail.',
        difficulty: 'Advanced',
        companyTags: ['Google', 'Palantir']
      },
      {
        question: 'Is K-Means guaranteed to converge?',
        answer: 'Yes, Lloyd\'s algorithm is guaranteed to converge in a finite number of steps because there is a finite number of possible assignments, and the WCSS strictly decreases or stays the same at each step. However, it is only guaranteed to converge to a local minimum, not the global minimum.',
        difficulty: 'Intermediate',
        companyTags: ['Meta', 'AWS']
      },
      {
        question: 'How do outliers affect K-Means?',
        answer: 'Because centroids are calculated as the arithmetic mean of the assigned points, they are highly sensitive to outliers. A single severe outlier can pull a centroid far away from the true center of a dense cluster.',
        difficulty: 'Beginner',
        companyTags: ['Airbnb', 'Pinterest']
      },
      {
        question: 'What is Mini-Batch K-Means?',
        answer: 'Mini-Batch K-Means is a variant that uses random subsets (mini-batches) of data at each iteration to update the centroids, drastically reducing computation time for massive datasets while usually yielding only slightly worse inertia.',
        difficulty: 'Intermediate',
        companyTags: ['Twitter', 'LinkedIn']
      },
      {
        question: 'What is the difference between K-Means and K-Medoids?',
        answer: 'K-Means calculates the center as the mean vector, which may not be an actual data point. K-Medoids requires the center (medoid) to be an actual data point and minimizes the absolute distance (often Manhattan) rather than squared distance, making it more robust to outliers.',
        difficulty: 'Advanced',
        companyTags: ['Uber', 'Databricks']
      },
      {
        question: 'Can K-Means be used for image compression?',
        answer: 'Yes, via color quantization. You cluster the millions of pixel colors in an image into K clusters (e.g., K=16). Then, replace every pixel with its nearest centroid color, reducing the color palette to 16 colors.',
        difficulty: 'Beginner',
        companyTags: ['Snap', 'Adobe']
      },
      {
        question: 'How does K-Means differ from Hierarchical Clustering?',
        answer: 'K-Means requires K upfront and creates a flat partition. Hierarchical clustering builds a tree-like hierarchy of clusters (dendrogram) and does not require K upfront, but is much more computationally expensive O(N^3).',
        difficulty: 'Intermediate',
        companyTags: ['IBM', 'Salesforce']
      },
      {
        question: 'What is the objective function of K-Means?',
        answer: 'The objective function is the Within-Cluster Sum of Squares (WCSS), or inertia. It is the sum of squared Euclidean distances from each data point to its assigned cluster centroid.',
        difficulty: 'Intermediate',
        companyTags: ['Meta', 'Microsoft']
      },
      {
        question: 'Does running K-Means multiple times guarantee the same result?',
        answer: 'No. Because the initial centroids are chosen randomly (even with K-Means++), different runs can converge to different local minima. Sklearn\'s implementation runs the algorithm `n_init` times and returns the best result.',
        difficulty: 'Beginner',
        companyTags: ['Amazon', 'Google']
      },
      {
        question: 'How do you handle empty clusters in K-Means?',
        answer: 'If a centroid ends up with no assigned points during the update step, implementations typically drop the centroid, or relocate it to the data point furthest away from its current assigned centroid.',
        difficulty: 'Advanced',
        companyTags: ['Lyft', 'Stripe']
      },
      {
        question: 'Can K-Means be used for feature engineering?',
        answer: 'Yes. You can cluster the data and use the assigned cluster ID as a new categorical feature, or use the distances to the K centroids as K new numerical features for a downstream supervised model.',
        difficulty: 'Intermediate',
        companyTags: ['Capital One', 'Kaggle']
      },
      {
        question: 'Why does K-Means struggle with clusters of different sizes/variances?',
        answer: 'K-Means effectively models clusters as spheres of the same radius. It assumes the variance is identical across all clusters. A very large, dispersed cluster might be split, or points from a dense cluster might be wrongly assigned to the larger one.',
        difficulty: 'Advanced',
        companyTags: ['Google', 'Meta']
      },
      {
        question: 'What is Spectral Clustering, and how does it relate to K-Means?',
        answer: 'Spectral clustering performs dimensionality reduction using the eigenvalues of a similarity matrix before applying K-Means in fewer dimensions. This allows K-Means to effectively cluster non-convex, complex shapes.',
        difficulty: 'Advanced',
        companyTags: ['Palantir', 'OpenAI']
      },
      {
        question: 'Describe Silhouette Score.',
        answer: 'It calculates (b - a) / max(a, b) for each point, where a is the mean distance to points in the same cluster, and b is the mean distance to points in the nearest other cluster. Scores range from -1 (misclassified) to 1 (well-clustered).',
        difficulty: 'Intermediate',
        companyTags: ['Apple', 'LinkedIn']
      }
    ]
  };
