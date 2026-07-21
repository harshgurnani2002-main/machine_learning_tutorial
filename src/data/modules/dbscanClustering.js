export const dbscanClustering = {
    id: 'dbscan-clustering',
    title: 'DBSCAN Clustering',
    category: 'Unsupervised Learning',
    description: 'Group spatial coordinates by density connectivity, identifying noise points automatically.',
    formula: 'N_\\epsilon(p) = \\{q \\in D \\mid dist(p,q) \\le \\epsilon\\}',
    theory: `### DBSCAN Density Clustering Theory

**What is it?**
Density-Based Spatial Clustering of Applications with Noise (DBSCAN) is an unsupervised, non-parametric, density-based clustering algorithm. Unlike traditional centroid-based algorithms such as K-Means (which assume clusters are convex and spherical) or distribution-based algorithms like Gaussian Mixture Models, DBSCAN groups together points that are closely packed together—defined as points with many nearby neighbors. Most importantly, it explicitly models and identifies points that lie alone in low-density regions as outliers or noise.
DBSCAN was proposed in 1996 by Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu, and remains one of the most highly cited clustering algorithms in scientific literature due to its robust handling of real-world, noisy data.

**Why do we need it?**
K-Means clustering has several major limitations in real-world scenarios:
1. **Requires Predefined K**: You must specify the number of clusters (K) beforehand, which is often unknown in exploratory data analysis.
2. **Assumes Convexity**: It assumes clusters are spherical and isotropic. It fails completely on complex, non-linear geometric shapes, such as two intertwined crescent moons or concentric circles.
3. **Sensitive to Outliers**: K-Means forces every single point into a cluster. Severe outliers aggressively pull centroids away from their natural centers, distorting the entire clustering.

DBSCAN solves all three issues fundamentally:
- It **does not require specifying K**. It discovers the number of clusters automatically based entirely on the underlying data\'s density.
- It can find **arbitrarily shaped clusters** (lines, curves, nested shapes, S-curves) because it simply follows contiguous dense regions, wherever they may lead.
- It has an elegant, built-in mathematical mechanism for **identifying and ignoring outlier noise**, meaning true clusters remain undistorted by extreme anomaly values.

**How does it work?**
DBSCAN defines clusters as continuous regions of high density separated by regions of low density. It relies strictly on two main hyperparameters:
- **Epsilon ($\\epsilon$)**: The maximum spatial distance between two points for them to be considered neighbors. This strictly defines the radius of a local neighborhood around a point.
- **MinPts (min_samples)**: The minimum number of points required within the $\\epsilon$-radius of a point (including the point itself) to form a dense region.

Based on these two parameters, DBSCAN strictly classifies all data points into three mutually exclusive categories:
1. **Core Points**: A point is a core point if there are at least *MinPts* points (including itself) within its $\\epsilon$-neighborhood. These points reside deep in the interior of a dense region and form the foundation of a cluster.
2. **Border Points**: A point is a border point if it is reachable from a core point (it is within the $\\epsilon$-radius of a core point), but it has fewer than *MinPts* in its own neighborhood. Border points essentially form the outer edges or "skin" of a cluster.
3. **Noise (Outliers)**: A point is noise if it is neither a core point nor a border point. It is isolated in a low-density region of the feature space.

**The Math Behind It**
DBSCAN relies on rigorous mathematical definitions of density, reachability, and connectivity:
- **$\\epsilon$-Neighborhood**: For a given point $p$, the neighborhood $N_\\epsilon(p)$ is the set of all points $q$ in the dataset $D$ such that the distance between $p$ and $q$ is less than or equal to $\\epsilon$:
  $$N_\\epsilon(p) = \\{q \\in D \\mid dist(p, q) \\le \\epsilon\\}$$
- **Directly Density-Reachable**: A point $q$ is directly density-reachable from a point $p$ if $p$ is a core point ($|N_\\epsilon(p)| \\ge MinPts$) and $q$ is in $N_\\epsilon(p)$. This relationship is asymmetric (if $q$ is a border point, $p$ is not directly reachable from $q$, because $q$ cannot reach anything).
- **Density-Reachable**: A point $q$ is density-reachable from $p$ if there is a chain of points $p_1, p_2, \\dots, p_n$ with $p_1 = p$ and $p_n = q$, where each $p_{i+1}$ is directly density-reachable from $p_i$. This implies transitivity.
- **Density-Connected**: Two points $p$ and $q$ are density-connected if there is a third point $o$ such that both $p$ and $q$ are density-reachable from $o$. This allows the cluster to snake through space and change directions.
- **Cluster Definition**: A cluster $C$ is mathematically defined as a non-empty subset of $D$ satisfying two strict conditions:
  1. *Maximality*: If $p \\in C$ and $q$ is density-reachable from $p$, then $q \\in C$.
  2. *Connectivity*: For all $p, q \\in C$, $p$ is density-connected to $q$.

**Algorithmic Complexity**
- **Time Complexity**: Without spatial indexing, DBSCAN must calculate the distance between every pair of points, resulting in $O(N^2)$ time complexity. With a spatial index (like an R-tree or KD-tree), querying neighborhoods takes $O(\\log N)$, bringing the overall time complexity to $O(N \\log N)$ for low-dimensional data.
- **Space Complexity**: The algorithm requires $O(N)$ memory to store the labels and spatial index, but creating a full distance matrix (often done in naïve implementations) requires $O(N^2)$ memory, which crashes most machines for $N > 50,000$.

**Worked Example**
Imagine a 2D spatial dataset representing GPS coordinates of people in a park. We set $\\epsilon = 5.0$ meters and $MinPts = 4$.
1. The algorithm picks a random, unvisited person $A$.
2. It queries the spatial index for all people within a 5.0m radius of $A$. It finds 5 people (including $A$).
3. Because $5 \\ge 4$, person $A$ is officially labeled as a **Core Point**. A new cluster, "Group 1", is formed.
4. The algorithm then iterates through all 4 neighbors of $A$. It checks their 5.0m $\\epsilon$-neighborhoods. If neighbor $B$ has 6 people in their radius, $B$ is also a Core Point, and their neighbors are merged into "Group 1".
5. This recursive expansion continues, growing the cluster like a viral infection through the crowd.
6. Eventually, the expansion reaches person $C$ on the edge of the group. $C$ is within 5m of a core point, but their own radius only contains 2 people. $C$ is labeled a **Border Point**. The expansion stops here, because $C$ cannot pull anyone else in.
7. The algorithm picks the next unvisited person, $Z$, sitting alone on a bench. It finds only 1 person (themselves) in their radius. $Z$ is temporarily marked as **Noise**.
8. This repeats until all people are visited. If someone was marked as Noise but later discovered by another growing group, their label updates to Border Point.

**Common Pitfalls & Advanced Techniques**
- **Varying Densities**: Standard DBSCAN's Achilles' heel is datasets where different clusters have significantly different densities. Because $\\epsilon$ is a single, global parameter, a high $\\epsilon$ will accidentally merge distinct, dense clusters together into a massive blob, while a low $\\epsilon$ will treat the sparser clusters completely as pure noise.
  - *Solution*: Use OPTICS (Ordering Points To Identify the Clustering Structure) or HDBSCAN (Hierarchical DBSCAN). These algorithms build a hierarchy of clusters over all possible $\\epsilon$ values and extract the most stable clusters, effortlessly handling varying densities.
- **Curse of Dimensionality**: Like all algorithms relying on Euclidean distance (e.g., KNN), DBSCAN suffers heavily in high-dimensional spaces (e.g., $D > 50$). In high dimensions, the distance between the nearest and farthest neighbors mathematically converges, making the concept of an $\\epsilon$-neighborhood entirely meaningless. 
  - *Solution*: Dimensionality reduction (PCA, t-SNE, UMAP) is mandatory before applying DBSCAN to high-dimensional data like text or images.
- **Choosing Hyperparameters**: Guessing $\\epsilon$ blindly almost never works.
  - *Heuristic for MinPts*: A common rule of thumb is $MinPts \\ge D + 1$ (where $D$ is the number of dimensions). For larger datasets, $MinPts = 2 \times D$ is strongly preferred.
  - *Heuristic for $\\epsilon$*: Use the K-distance graph. Plot the distance to the k-th nearest neighbor (where $k = MinPts$) for all points, sorted descending. Look for the sharp "knee" or "elbow" in the curve; the y-value at this knee is the mathematically optimal starting point for $\\epsilon$.

**When to Use vs Not Use**
- **Use when**: You have low-to-medium dimensional spatial data, geographic/location data, strict anomaly detection requirements, or you visually expect clusters to form non-spherical, arbitrary, complex geometric shapes.
- **Not Use when**: You have text embeddings or image embeddings (very high dimensions), your dataset consists of clusters with vastly differing densities, or you need a strict, fast centroid to assign new incoming data points quickly in a real-time production environment (DBSCAN is transductive, not inductive).

**Key Takeaways**
- DBSCAN groups points by tracing continuous regions of high density connectivity.
- It operates strictly on two parameters: $\\epsilon$ (radius) and MinPts (density threshold).
- It fundamentally categorizes all points into Core Points (bulk of cluster), Border Points (edges of cluster), and Noise (outliers).
- It is profoundly superior to K-Means for discovering complex geometric shapes and automatically filtering out noise, provided the data has relatively uniform density.
#### Python Implementation

\`\`\`python
from sklearn.cluster import DBSCAN
import numpy as np

X = np.random.randn(100, 2)
db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(X)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Clusters found: {n_clusters}")
print(f"Noise points: {list(labels).count(-1)}")
\`\`\`
`,
    interactiveSummary: 'In this interactive simulation, you can click anywhere on the canvas to place GPS-style coordinate points and watch DBSCAN classify them in real time. The Epsilon (ε) slider controls the neighborhood radius — think of it as drawing a circle of that radius around every point and counting how many other points fall inside it. The Min Samples slider sets the density threshold: a point needs at least that many neighbors within ε to become a Core Point (shown in vivid color), otherwise it degrades to a Border Point (lighter ring) or a Noise outlier (grey ×). Try drawing two dense clusters connected by a sparse bridge and watch how increasing ε eventually merges them into one cluster. Then lower Min Samples to see previously noisy isolated points get absorbed as border points. This directly mirrors real anomaly detection workflows where tuning these two parameters determines what counts as "normal traffic" versus a suspicious outlier.',
    simulatorId: 'dbscan',
    quiz: [
        { id: 'dbs_q1', question: 'Which point class represents an outlier that does not belong to any cluster in DBSCAN?', options: ['Noise Point', 'Core Point', 'Border Point', 'Centroid'], correctAnswer: 'Noise Point', explanation: 'Outliers that fail to meet density requirements are labeled as Noise Points.' },
        { id: 'dbs_q2', question: 'What is a Core Point in DBSCAN?', options: ['A point with at least MinPts neighbors within epsilon distance', 'The geometric center of a cluster', 'A point that connects two different clusters', 'A point on the edge of a cluster'], correctAnswer: 'A point with at least MinPts neighbors within epsilon distance', explanation: 'By definition, a core point must have a dense neighborhood, meaning at least MinPts points (including itself) fall within its epsilon radius.' },
        { id: 'dbs_q3', question: 'What is a primary advantage of DBSCAN over K-Means?', options: ['It does not require specifying the number of clusters in advance', 'It always converges faster', 'It calculates the exact global optimum', 'It works perfectly in infinite dimensions'], correctAnswer: 'It does not require specifying the number of clusters in advance', explanation: 'DBSCAN discovers the number of clusters automatically based on the data\'s density, whereas K-Means requires K to be manually set.' },
        { id: 'dbs_q4', question: 'How does DBSCAN handle clusters of arbitrary, non-spherical shapes?', options: ['By chaining together density-connected core points', 'By using multiple centroids per cluster', 'By applying a kernel trick', 'By forcing shapes into a Voronoi diagram'], correctAnswer: 'By chaining together density-connected core points', explanation: 'Because clusters are formed by expanding through connected core points, they can snake and bend into any arbitrary shape.' },
        { id: 'dbs_q5', question: 'What is the main limitation of standard DBSCAN?', options: ['It struggles with clusters of varying densities', 'It cannot detect outliers', 'It requires labels for training data', 'It only works on 2D data'], correctAnswer: 'It struggles with clusters of varying densities', explanation: 'Because epsilon is a global parameter, a value that captures sparse clusters might merge dense clusters together, and vice versa.' },
        { id: 'dbs_q6', question: 'What is a Border Point in DBSCAN?', options: ['A point that has fewer than MinPts neighbors, but lies within the epsilon radius of a core point', 'A point that separates two clusters', 'A point exactly epsilon distance from the origin', 'A noise point that was misclassified'], correctAnswer: 'A point that has fewer than MinPts neighbors, but lies within the epsilon radius of a core point', explanation: 'Border points are not dense enough to be core points, but they are close enough to a core point to be included in its cluster.' },
        { id: 'dbs_q7', question: 'If you increase the epsilon parameter significantly, what is the most likely outcome?', options: ['Fewer clusters, fewer noise points', 'More clusters, more noise points', 'More clusters, fewer noise points', 'Fewer clusters, more noise points'], correctAnswer: 'Fewer clusters, fewer noise points', explanation: 'A larger epsilon expands neighborhoods, merging separate clusters into giant single clusters and absorbing noise points into the clusters.' },
        { id: 'dbs_q8', question: 'If you increase the MinPts parameter significantly, what is the most likely outcome?', options: ['More noise points', 'Fewer noise points', 'Larger clusters', 'All points become core points'], correctAnswer: 'More noise points', explanation: 'A higher MinPts requires a higher density to form a cluster. Points in less dense regions will fail the test and be classified as noise.' },
        { id: 'dbs_q9', question: 'How is the "K-distance graph" used in DBSCAN?', options: ['To help choose an appropriate epsilon value', 'To visualize the final clusters', 'To calculate the Euclidean distance between centroids', 'To find the optimal value for MinPts'], correctAnswer: 'To help choose an appropriate epsilon value', explanation: 'Plotting the distance to the k-th nearest neighbor (where k = MinPts) in sorted order helps identify a "knee" or "elbow", which is a good heuristic for epsilon.' },
        { id: 'dbs_q10', question: 'What happens to a border point that is reachable from the core points of two different clusters?', options: ['It is assigned to the cluster that discovers it first', 'It is marked as noise', 'It merges the two clusters into one', 'It is cloned and assigned to both'], correctAnswer: 'It is assigned to the cluster that discovers it first', explanation: 'DBSCAN is slightly order-dependent for border points. If a border point falls in the radius of core points from two distinct clusters, it belongs to whichever cluster expands to it first.' }
    ],
    coding: {
        tutorial: {
            title: 'Locate Neighbors',
            description: 'Find indices of points within epsilon distance of a target point.',
            pseudoCode: `function find_neighbors(X, target_index, eps):
    target = X[target_index]
    distances = calculate_distance_from_target_to_all_points(X)
    valid_indices = where distances <= eps
    return valid_indices`,
            starterCode: `import numpy as np
def find_neighbors(X, target_idx, eps):
    # TODO: return indices of points where distance <= eps
    return []

X = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0]])
print(find_neighbors(X, 0, 0.5))`,
            expectedOutput: '[0, 1]',
            solution: `import numpy as np
def find_neighbors(X, target_idx, eps):
    target = X[target_idx]
    dists = np.sqrt(np.sum((X - target) ** 2, axis=1))
    return list(np.where(dists <= eps)[0])

X = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0]])
print(find_neighbors(X, 0, 0.5))`,
            hints: ['Calculate distance array to target, filter indices using np.where.'],
            testKeywords: ['np.sum', 'np.where']
        },
        project: {
            title: 'Geospatial Anomaly Detection — GPS Coordinate Clustering',
            description: 'Apply DBSCAN to GPS latitude/longitude coordinates representing delivery vehicle positions. Dense clusters represent normal route hubs; isolated points flagged as noise (-1) are potential anomalies (vehicles far off-route). After clustering, print the labels and identify how many locations are flagged as outliers.',
            pseudoCode: `# Step 1: Scale GPS coordinates (lat/lon have equal units here, but good practice)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(gps_coords)

# Step 2: Cluster with DBSCAN — tight eps captures distinct hubs, min_samples=3
dbscan = DBSCAN(eps=0.5, min_samples=3)
labels = dbscan.fit_predict(X_scaled)

# Step 3: Identify anomalous positions flagged as noise
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)
print(f"Cluster labels: {labels}")
print(f"Clusters found: {n_clusters} | Anomalous locations: {n_noise}")`,
            starterCode: `import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# GPS coordinates [latitude, longitude] of 18 delivery stops
# 3 dense hubs + 2 anomalous outlier positions far off normal routes
gps_coords = np.array([
    # Hub A — city center
    [40.712, -74.006], [40.714, -74.009], [40.710, -74.003],
    [40.713, -74.007], [40.711, -74.005],
    # Hub B — airport area
    [40.641, -73.778], [40.643, -73.780], [40.639, -73.776],
    [40.642, -73.779], [40.640, -73.777],
    # Hub C — Brooklyn
    [40.678, -73.944], [40.680, -73.947], [40.676, -73.941],
    [40.679, -73.945], [40.677, -73.943],
    # Anomalies — vehicles far from any hub
    [40.850, -73.600],
    [40.500, -74.250]
])

# TODO: Scale gps_coords using StandardScaler
# TODO: Initialize DBSCAN(eps=0.5, min_samples=3) and fit_predict on scaled data
# TODO: Count clusters (unique labels excluding -1) and noise points (label == -1)
# TODO: Print labels, number of clusters found, and number of anomalous locations

scaler = None
dbscan = None
labels = []`,
            expectedOutput: `Cluster labels: [ 0  0  0  0  0  1  1  1  1  1  2  2  2  2  2 -1 -1]\nClusters found: 3 | Anomalous locations: 2`,
            solution: `import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

gps_coords = np.array([
    [40.712, -74.006], [40.714, -74.009], [40.710, -74.003],
    [40.713, -74.007], [40.711, -74.005],
    [40.641, -73.778], [40.643, -73.780], [40.639, -73.776],
    [40.642, -73.779], [40.640, -73.777],
    [40.678, -73.944], [40.680, -73.947], [40.676, -73.941],
    [40.679, -73.945], [40.677, -73.943],
    [40.850, -73.600],
    [40.500, -74.250]
])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(gps_coords)

dbscan = DBSCAN(eps=0.5, min_samples=3)
labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)
print(f"Cluster labels: {labels}")
print(f"Clusters found: {n_clusters} | Anomalous locations: {n_noise}")`,
            hints: ['Scale the GPS coordinates with StandardScaler before passing to DBSCAN.', 'Use DBSCAN(eps=0.5, min_samples=3).fit_predict(X_scaled).', 'Noise points have label -1. Count them with np.sum(labels == -1).', 'Unique non-noise clusters: len(set(labels)) - (1 if -1 in labels else 0).'],
            testKeywords: ['DBSCAN', 'StandardScaler', 'fit_predict', '== -1', 'fit_transform']
        },
        assignment: {
            title: 'Network Intrusion Detection — Cluster Normal vs Anomalous Traffic',
            description: 'Use DBSCAN to separate normal network traffic sessions from potential intrusions. Each row is a connection described by [duration_sec, bytes_sent, packet_rate]. Dense clusters represent normal traffic patterns; points flagged as noise (-1) are candidate intrusions. Return the anomaly indices and compute the anomaly rate.',
            pseudoCode: `# Step 1: Scale network traffic features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(traffic_data)

# Step 2: Apply DBSCAN — normal sessions cluster tightly, attacks are sparse
dbscan = DBSCAN(eps=0.6, min_samples=4)
labels = dbscan.fit_predict(X_scaled)

# Step 3: Extract anomaly indices and rate
anomaly_indices = np.where(labels == -1)[0]
anomaly_rate = len(anomaly_indices) / len(labels)
print(f"Anomaly indices: {anomaly_indices}")
print(f"Anomaly rate: {anomaly_rate:.1%}")`,
            starterCode: `import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Network traffic features: [duration_sec, bytes_sent_kb, packets_per_sec]
# Normal sessions cluster tightly; attack traffic is sparse and anomalous
traffic_data = np.array([
    # Normal web browsing (tight cluster)
    [0.5, 12.0, 8.0], [0.6, 14.0, 9.0], [0.4, 11.0, 7.5],
    [0.5, 13.0, 8.5], [0.6, 12.5, 8.0], [0.5, 11.5, 8.2],
    # Normal file transfers (second cluster)
    [5.0, 800.0, 2.0], [5.2, 820.0, 1.9], [4.8, 790.0, 2.1],
    [5.1, 810.0, 2.0], [5.3, 830.0, 1.8],
    # Anomalous traffic — potential intrusions
    [0.01, 5000.0, 900.0],  # DDoS spike
    [120.0, 1.0, 0.01],     # Port scan (long, minimal data)
    [0.1, 2.0, 0.5]         # Stealth probe
])

# TODO: Scale traffic_data with StandardScaler
# TODO: Apply DBSCAN(eps=0.6, min_samples=4) and fit_predict
# TODO: Find anomaly_indices where labels == -1
# TODO: Compute anomaly_rate = len(anomaly_indices) / len(labels)
# TODO: Print anomaly_indices and anomaly_rate formatted as percentage

scaler = None
dbscan = None
labels = []`,
            expectedOutput: `Anomaly indices: [11 12 13]\nAnomaly rate: 21.4%`,
            solution: `import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

traffic_data = np.array([
    [0.5, 12.0, 8.0], [0.6, 14.0, 9.0], [0.4, 11.0, 7.5],
    [0.5, 13.0, 8.5], [0.6, 12.5, 8.0], [0.5, 11.5, 8.2],
    [5.0, 800.0, 2.0], [5.2, 820.0, 1.9], [4.8, 790.0, 2.1],
    [5.1, 810.0, 2.0], [5.3, 830.0, 1.8],
    [0.01, 5000.0, 900.0],
    [120.0, 1.0, 0.01],
    [0.1, 2.0, 0.5]
])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(traffic_data)

dbscan = DBSCAN(eps=0.6, min_samples=4)
labels = dbscan.fit_predict(X_scaled)

anomaly_indices = np.where(labels == -1)[0]
anomaly_rate = len(anomaly_indices) / len(labels)
print(f"Anomaly indices: {anomaly_indices}")
print(f"Anomaly rate: {anomaly_rate:.1%}")`,
            hints: ['StandardScaler is critical here — bytes_sent ranges to thousands while duration is < 1.', 'DBSCAN(eps=0.6, min_samples=4) keeps normal clusters tight while isolating traffic spikes.', 'np.where(labels == -1)[0] returns the row indices of all anomalous connections.'],
            testKeywords: ['DBSCAN', 'StandardScaler', 'fit_transform', 'fit_predict', '== -1', 'np.where']
        }
    },
    interviewQuestions: [
        { question: 'Compare DBSCAN and K-Means. In what scenarios does K-Means fail where DBSCAN excels?', answer: 'K-Means assumes spherical clusters and requires specifying K beforehand. It fails on complex shapes (like moons or nested circles) and is sensitive to outliers. DBSCAN identifies clusters of arbitrary shapes, automatically detects noise, and does not require pre-specifying K.', companyTags: ['Uber', 'Meta'], difficulty: 'Advanced' },
        { question: 'What are the core parameters of DBSCAN and what do they mean?', answer: 'The core parameters are Epsilon (eps) and MinPts (min_samples). Epsilon defines the radius of neighborhood around a point. MinPts is the minimum number of points required within that radius for the central point to be considered a Core Point.', companyTags: ['Google', 'Amazon'], difficulty: 'Easy' },
        { question: 'How does DBSCAN define a Noise point?', answer: 'A noise point is any point in the dataset that is neither a Core Point (does not have MinPts neighbors within eps distance) nor a Border Point (does not fall within the eps radius of a Core Point).', companyTags: ['Apple', 'Microsoft'], difficulty: 'Easy' },
        { question: 'What is the difference between a Core point and a Border point?', answer: 'A Core point has at least MinPts neighbors within its epsilon radius. A Border point has fewer than MinPts neighbors, but lies within the epsilon radius of a Core point. Border points make up the outer edges of a cluster.', companyTags: ['Netflix', 'LinkedIn'], difficulty: 'Medium' },
        { question: 'Why does DBSCAN struggle with datasets containing clusters of varying densities?', answer: 'DBSCAN uses a global Epsilon parameter. If Epsilon is large enough to capture sparse clusters, it will likely merge multiple dense clusters together. If Epsilon is small enough to separate dense clusters, it will label sparse clusters entirely as noise.', companyTags: ['Google', 'Stripe'], difficulty: 'Medium' },
        { question: 'What is HDBSCAN and how does it improve upon standard DBSCAN?', answer: 'Hierarchical DBSCAN (HDBSCAN) extends DBSCAN by converting it into a hierarchical clustering algorithm. It extracts a flat clustering based on cluster stability over varying epsilon values, effectively allowing it to find clusters of varying densities without requiring a fixed epsilon parameter.', companyTags: ['Meta', 'Amazon'], difficulty: 'Hard' },
        { question: 'What is the time complexity of DBSCAN?', answer: 'In the worst case, without spatial indexing, DBSCAN is O(N^2) because it must compute pairwise distances between all points. With spatial indexing structures like kd-trees or R-trees, it can be optimized to O(N log N) in low-dimensional spaces.', companyTags: ['Uber', 'Pinterest'], difficulty: 'Hard' },
        { question: 'How do you determine a good value for the Epsilon parameter in practice?', answer: 'A common technique is to compute the distance to the k-th nearest neighbor (where k = MinPts) for every point in the dataset. Plot these distances sorted in descending order. The point of maximum curvature (the "elbow" or "knee") provides a good heuristic for Epsilon.', companyTags: ['Apple', 'Microsoft'], difficulty: 'Medium' },
        { question: 'Is DBSCAN deterministic?', answer: 'Mostly yes, but not entirely. The core points and noise points are deterministically classified. However, border points that are reachable from core points of multiple different clusters will be assigned to whichever cluster happens to process them first, making it slightly order-dependent.', companyTags: ['Meta', 'Netflix'], difficulty: 'Hard' },
        { question: 'Can DBSCAN be used for predicting clusters on new, unseen data?', answer: 'Not inherently. DBSCAN is a transductive algorithm, meaning it clusters the data it is given. Unlike K-Means, it doesn\'t compute a central centroid to easily assign new points. To predict new points, you typically evaluate if they fall within the epsilon radius of existing core points.', companyTags: ['Google', 'Tesla'], difficulty: 'Medium' },
        { question: 'How does the Curse of Dimensionality affect DBSCAN?', answer: 'In very high-dimensional spaces, the distance between any two points tends to converge, meaning the ratio of nearest neighbor to furthest neighbor approaches 1. This makes the Epsilon threshold meaningless, as all points appear equidistant, causing DBSCAN to fail.', companyTags: ['Amazon', 'Stripe'], difficulty: 'Hard' },
        { question: 'If you have a dataset with millions of rows, would you use DBSCAN or K-Means?', answer: 'K-Means (specifically Mini-Batch K-Means) is generally much faster and scales better linearly (O(N)) compared to standard DBSCAN (O(N^2)). Unless spatial indexing is highly optimized, DBSCAN can be too slow for millions of rows.', companyTags: ['Uber', 'LinkedIn'], difficulty: 'Medium' },
        { question: 'How do you evaluate the quality of DBSCAN clusters?', answer: 'You cannot use inertia (within-cluster sum of squares) because DBSCAN clusters are not spherical. Instead, you use metrics based on pairwise distances, such as the Silhouette Score, or evaluate on downstream tasks if available.', companyTags: ['Apple', 'Pinterest'], difficulty: 'Medium' },
        { question: 'What is the concept of "Density-Reachable" in DBSCAN?', answer: 'Point Q is density-reachable from point P if there is a sequence of core points connecting P to Q, where each point in the sequence is within the epsilon neighborhood of the previous point.', companyTags: ['Microsoft', 'Meta'], difficulty: 'Hard' },
        { question: 'Does DBSCAN require feature scaling?', answer: 'Yes, absolutely. Because it relies heavily on Euclidean distance, unscaled features with larger numerical ranges will dominate the distance calculation, distorting the concept of a neighborhood.', companyTags: ['Netflix', 'Tesla'], difficulty: 'Easy' },
        { question: 'In Scikit-Learn, how are noise points labeled by DBSCAN?', answer: 'In Scikit-Learn\'s implementation of DBSCAN, noise points are assigned the cluster label `-1`.', companyTags: ['Google', 'Amazon'], difficulty: 'Easy' },
        { question: 'What happens if MinPts is set to 1?', answer: 'If MinPts is 1, every single point is technically a core point. DBSCAN then degenerates into a simple single-linkage hierarchical clustering algorithm where clusters are just connected components of a graph with edges defined by epsilon.', companyTags: ['Meta', 'Stripe'], difficulty: 'Hard' },
        { question: 'How is OPTICS related to DBSCAN?', answer: 'OPTICS (Ordering Points To Identify the Clustering Structure) is similar to DBSCAN but addresses the varying density problem. Instead of producing a strict clustering for one epsilon, it produces an augmented ordering of the database representing its density-based clustering structure at a wide range of epsilon values.', companyTags: ['Apple', 'Uber'], difficulty: 'Hard' },
        { question: 'Can DBSCAN handle categorical variables?', answer: 'Standard DBSCAN uses Euclidean distance, which is not suitable for categorical variables. However, you can use DBSCAN with custom distance metrics (like Gower distance or Jaccard similarity) to cluster data containing categorical variables.', companyTags: ['Microsoft', 'LinkedIn'], difficulty: 'Medium' },
        { question: 'What are the memory requirements for DBSCAN?', answer: 'If a distance matrix is precomputed, it requires O(N^2) memory, which is prohibitive for large datasets. Without precomputing, spatial indexing structures require O(N) memory, but keeping track of neighborhood lists during execution can still be memory intensive.', companyTags: ['Google', 'Netflix'], difficulty: 'Medium' }
    ]
};
