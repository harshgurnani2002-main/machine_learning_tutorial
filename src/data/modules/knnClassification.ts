import type { MLModule } from '../../types';

export const knnClassification: MLModule = {
    id: 'knn-classification',
    title: 'K-Nearest Neighbors',
    category: 'Supervised Learning',
    description: 'Classify points based on class majorities of closest neighbors.',
    formula: 'd(p, q) = \\sqrt{\\sum (p_i - q_i)^2}',
    theory: `### K-Nearest Neighbors (KNN)

#### What is it?
K-Nearest Neighbors (KNN) is a simple, non-parametric, and lazy learning algorithm used for both classification and regression tasks. Instead of learning a mathematical model representing the underlying distribution of the data, KNN simply stores the entire training dataset. When it needs to make a prediction for a new unseen data point, it searches through the stored dataset to find the 'K' most similar instances (the nearest neighbors) and assigns the class label based on a majority vote (for classification) or an average (for regression).

#### Why do we need it?
There are many scenarios where the decision boundary between classes is highly irregular and cannot be easily captured by linear models or even simple non-linear models. KNN makes no underlying assumptions about the distribution of data (hence "non-parametric"), which makes it incredibly flexible and powerful for complex, real-world datasets where the relationship between features and labels is unpredictable. Furthermore, it serves as an excellent baseline model because of its simplicity and intuitiveness.

#### How does it work?
1. **Choose K:** Select the number of neighbors, $K$.
2. **Calculate Distance:** For a new query point, compute the distance between this point and all points in the training dataset.
3. **Find Neighbors:** Sort the calculated distances in ascending order and select the top $K$ points (the closest neighbors).
4. **Vote/Average:**
   - **Classification:** Count the class labels of the $K$ neighbors. Assign the most frequent class label to the query point.
   - **Regression:** Calculate the mean or median of the target values of the $K$ neighbors.

#### The Math Behind It
KNN heavily relies on distance metrics to determine similarity. The most common metric is **Euclidean Distance** ($L_2$ norm):
$$d(x, x') = \\sqrt{\\sum_{j=1}^{d} (x_j - x'_j)^2}$$

Other metrics include:
- **Manhattan Distance** ($L_1$ norm): useful for grid-like data or high-dimensional spaces.
$$d(x, x') = \\sum_{j=1}^{d} |x_j - x'_j|$$
- **Minkowski Distance**: a generalization of both Euclidean and Manhattan distances.
$$d(x, x') = \\left( \\sum_{j=1}^{d} |x_j - x'_j|^p \\right)^{1/p}$$

Once the $K$ closest neighbors (denoted as $N_K(x)$) are found, the predicted class $\\hat{y}$ is:
$$\\hat{y} = \\arg\\max_{c} \\sum_{i \\in N_K(x)} \\mathbb{I}(y^{(i)} = c)$$
where $\\mathbb{I}$ is the indicator function that evaluates to 1 if the neighbor's class matches $c$, and 0 otherwise.

#### Worked Example
Suppose we want to classify a fruit as either "Apple" or "Orange" based on Weight (grams) and Sweetness (1-10).
Training Data:
- Point A: (150g, 8) -> Apple
- Point B: (170g, 7) -> Apple
- Point C: (140g, 3) -> Orange
- Point D: (160g, 4) -> Orange

Query: Fruit X with (155g, 6)
Assume we use Euclidean distance and $K=3$. Let's roughly estimate distance (ignoring scale for simplicity):
- Dist to A: $\\sqrt{(155-150)^2 + (6-8)^2} = \\sqrt{25 + 4} = 5.38$
- Dist to B: $\\sqrt{(155-170)^2 + (6-7)^2} = \\sqrt{225 + 1} = 15.03$
- Dist to C: $\\sqrt{(155-140)^2 + (6-3)^2} = \\sqrt{225 + 9} = 15.29$
- Dist to D: $\\sqrt{(155-160)^2 + (6-4)^2} = \\sqrt{25 + 4} = 5.38$

The 3 nearest neighbors are A (Apple), D (Orange), and B (Apple).
Voting: 2 Apples, 1 Orange.
Prediction: Apple.

#### Common Pitfalls
- **Feature Scaling:** Distance metrics are extremely sensitive to the scale of features. If one feature ranges from 0 to 1000 and another from 0 to 1, the former will dominate the distance calculation. Always standardize or normalize features.
- **Curse of Dimensionality:** In high-dimensional spaces, the distance between any two points converges. All points appear equally far away, ruining the concept of "nearest".
- **Choosing an Even K:** For binary classification, choosing an even $K$ can lead to ties. Always pick an odd $K$ to break ties.
- **Computational Cost:** Since it evaluates the distance to every single training point, prediction is $O(N \\times D)$, which is slow for massive datasets.

#### When to Use vs Not Use
**Use When:**
- The dataset is relatively small to medium-sized.
- The data is low-dimensional.
- You need a quick, interpreatable baseline model with no training time.
- The decision boundary is highly non-linear and irregular.

**Not Use When:**
- Dataset is massive (latency during inference is too high).
- High number of features (curse of dimensionality).
- Real-time prediction is required in a constrained environment.

#### Key Takeaways
- KNN is a lazy learner: zero training time, but expensive prediction time.
- Distance metric and $K$ are the core hyperparameters.
- Feature scaling is an absolute prerequisite.
- Susceptible to the curse of dimensionality.`,
    interactiveSummary: "This interactive simulator brings K-Nearest Neighbors to life on a 2D canvas where you place your own training points by clicking — assign each point to Class A (blue) or Class B (orange) to build a custom dataset, then click anywhere to drop a query point and watch the algorithm draw lines to its K nearest neighbors and cast a majority vote. The entire background is shaded in real time to show the decision boundary across the full feature space: blue regions predict Class A, orange regions predict Class B. Use the K slider to see a direct demonstration of the bias-variance tradeoff: at K=1 the boundary is jagged and tightly fitted to individual points (high variance), while at K=15 it smooths out into broad, confident regions (higher bias). This shows exactly why choosing the right K matters — too small and the model memorizes noise; too large and it averages away meaningful patterns.",
    simulatorId: 'knn',
    quiz: [
      {
        id: 'knn_q1',
        question: 'Why is KNN referred to as a "lazy learner"?',
        options: [
          'It bypasses parameter training, storing datasets and delaying all computations until prediction.',
          'It converges extremely slowly during training cycles.',
          'It is written in low-performance script code.',
          'It performs poor classification on linear data.'
        ],
        correctAnswer: 'It bypasses parameter training, storing datasets and delaying all computations until prediction.',
        explanation: 'KNN requires no training phase. It simply stores vectors, deferring distance calculations until a test sample is evaluated.'
      },
      {
        id: 'knn_q2',
        question: 'How does selecting a very small value for neighbor count K affect bias and variance?',
        options: [
          'High variance, low bias (prone to overfitting noisy structures).',
          'High bias, low variance (smooth boundaries).',
          'Low variance, low bias.',
          'No impact on variance.'
        ],
        correctAnswer: 'High variance, low bias (prone to overfitting noisy structures).',
        explanation: 'At K=1, predictions match the single nearest point. Any outlier flips predictions, creating noisy, high-variance decision boundaries.'
      },
      {
        id: 'knn_q3',
        question: 'Why is feature scaling (normalization) critical before using KNN?',
        options: [
          'Features with large numeric ranges will dominate distance calculations.',
          'It transforms non-linear relationships to linear relationships.',
          'Scaling removes duplicate data points.',
          'It is required to compute log likelihoods.'
        ],
        correctAnswer: 'Features with large numeric ranges will dominate distance calculations.',
        explanation: 'Because KNN uses spatial distance metrics, unscaled features dominate distance, rendering smaller range variables irrelevant.'
      },
      {
        id: 'knn_q4',
        question: 'What is the "Curse of Dimensionality" in the context of KNN?',
        options: [
          'In high dimensions, the distance between all pairs of points tends to become equal.',
          'High-dimensional data requires too much memory to store.',
          'High dimensions make it impossible to compute Euclidean distance.',
          'Adding more dimensions decreases the accuracy of K strictly linearly.'
        ],
        correctAnswer: 'In high dimensions, the distance between all pairs of points tends to become equal.',
        explanation: 'As dimensions increase, data becomes sparse, and the relative difference between nearest and farthest neighbors vanishes.'
      },
      {
        id: 'knn_q5',
        question: 'Which distance metric is most appropriate for categorical variables?',
        options: [
          'Hamming distance',
          'Euclidean distance',
          'Cosine similarity',
          'Minkowski distance'
        ],
        correctAnswer: 'Hamming distance',
        explanation: 'Hamming distance measures the number of positions at which corresponding symbols differ, ideal for categorical variables.'
      },
      {
        id: 'knn_q6',
        question: 'What is the time complexity of predicting a single test point in standard KNN?',
        options: [
          'O(N * d)',
          'O(1)',
          'O(log N)',
          'O(N^2)'
        ],
        correctAnswer: 'O(N * d)',
        explanation: 'To predict, KNN must compute the distance to every N training point across d dimensions, resulting in O(N*d) complexity.'
      },
      {
        id: 'knn_q7',
        question: 'Why is an odd value of K typically chosen for binary classification?',
        options: [
          'To prevent tie votes during the majority classification.',
          'Because odd numbers compute faster.',
          'To reduce the variance of the decision boundary.',
          'To ensure the model does not overfit.'
        ],
        correctAnswer: 'To prevent tie votes during the majority classification.',
        explanation: 'With an even K, you could have a 50/50 tie between two classes. An odd K ensures a strict majority.'
      },
      {
        id: 'knn_q8',
        question: 'What is Minkowski distance with p=1 equivalent to?',
        options: [
          'Manhattan distance',
          'Euclidean distance',
          'Chebyshev distance',
          'Mahalanobis distance'
        ],
        correctAnswer: 'Manhattan distance',
        explanation: 'Setting p=1 in the Minkowski generalized formula yields the sum of absolute differences, known as Manhattan distance.'
      },
      {
        id: 'knn_q9',
        question: 'What happens when K is set equal to the total number of training data points (N)?',
        options: [
          'The model simply predicts the majority class of the entire training set for every query.',
          'The model achieves 100% accuracy on the training set.',
          'The model throws a division by zero error.',
          'The decision boundary becomes extremely complex.'
        ],
        correctAnswer: 'The model simply predicts the majority class of the entire training set for every query.',
        explanation: 'If K=N, every prediction polls the entire dataset, so the prediction is always the global majority class, indicating extreme underfitting (high bias).'
      },
      {
        id: 'knn_q10',
        question: 'How can you mitigate the computational cost of KNN at prediction time?',
        options: [
          'Use spatial indexing structures like KD-Trees or Ball-Trees.',
          'Use gradient descent to optimize K.',
          'Drop the class labels from the training set.',
          'Apply L2 regularization.'
        ],
        correctAnswer: 'Use spatial indexing structures like KD-Trees or Ball-Trees.',
        explanation: 'Trees partition the data space, allowing for sub-linear O(log N) nearest neighbor searches instead of brute-force O(N).'
      }
    ],
    interviewQuestions: [
      {
        question: 'Explain the "Curse of Dimensionality" in relation to K-Nearest Neighbors.',
        answer: 'In high-dimensional spaces, the volume of coordinate space increases exponentially, making data points extremely sparse. Consequently, the distance between any two points converges to the same value. This causes KNN to fail because neighbors are no longer locally relevant; all points appear almost equally distant from the query vector.',
        companyTags: ['Amazon', 'Google'],
        difficulty: 'Hard'
      },
      {
        question: 'How do you choose the optimal value of K?',
        answer: 'The optimal K is typically found using cross-validation (e.g., k-fold CV). A very small K leads to low bias but high variance (overfitting to noise), while a very large K leads to high bias and low variance (underfitting). The plot of K vs Error Rate typically forms a U-shape, and we select the K at the minimum error.',
        companyTags: ['Meta', 'Microsoft'],
        difficulty: 'Medium'
      },
      {
        question: 'Can KNN be used for regression?',
        answer: 'Yes. Instead of taking a majority vote of the K nearest neighbors\' classes, KNN for regression takes the mean (or weighted average) of the target continuous values of the K nearest neighbors.',
        companyTags: ['Apple', 'Uber'],
        difficulty: 'Easy'
      },
      {
        question: 'What is Mahalanobis distance and when is it useful in KNN?',
        answer: 'Mahalanobis distance measures the distance between a point and a distribution, taking into account the covariance of the data. It is useful when features are highly correlated or on vastly different scales, as it effectively normalizes the features and decorrelates them.',
        companyTags: ['Bloomberg', 'Stripe'],
        difficulty: 'Hard'
      },
      {
        question: 'What are KD-Trees and Ball-Trees, and why do we use them with KNN?',
        answer: 'KD-Trees and Ball-Trees are space-partitioning data structures. Standard KNN takes O(Nd) time to predict. These trees partition the dataset into regions, allowing nearest neighbor searches in O(d log N) time, significantly speeding up prediction, though their advantage diminishes in very high dimensions.',
        companyTags: ['Netflix', 'Google'],
        difficulty: 'Hard'
      },
      {
        question: 'How do you handle categorical variables in KNN?',
        answer: 'Categorical variables can be one-hot encoded, but Euclidean distance isn\'t ideal for them. A better approach is to use Hamming Distance (counting mismatches) for the categorical parts and combine it with a normalized distance for numerical parts, or use a custom distance metric like Gower\'s distance.',
        companyTags: ['LinkedIn', 'Spotify'],
        difficulty: 'Medium'
      },
      {
        question: 'What is the space complexity of KNN?',
        answer: 'The space complexity is O(N * d), where N is the number of samples and d is the number of features. This is because KNN must store the entire training dataset in memory to compute distances at prediction time.',
        companyTags: ['Amazon', 'Meta'],
        difficulty: 'Easy'
      },
      {
        question: 'What is the difference between Lazy and Eager learning?',
        answer: 'Eager learners (like Neural Networks or SVMs) build a generalized model from the training data immediately during the training phase. Lazy learners (like KNN) simply store the training data and defer all generalization and calculation until a prediction is explicitly requested.',
        companyTags: ['Google', 'OpenAI'],
        difficulty: 'Medium'
      },
      {
        question: 'When would you prefer L1 (Manhattan) over L2 (Euclidean) distance?',
        answer: 'L1 distance is generally preferred in high-dimensional spaces because it is less sensitive to the curse of dimensionality than L2. It is also preferred when the data features represent grid-like paths or when robustness to outliers is strictly required.',
        companyTags: ['Uber', 'TikTok'],
        difficulty: 'Hard'
      },
      {
        question: 'What is Weighted KNN?',
        answer: 'In standard KNN, all K neighbors have an equal vote. In Weighted KNN, closer neighbors are given a higher weight in the vote (often proportional to the inverse of their distance, 1/d). This prevents ties and improves accuracy when data density is uneven.',
        companyTags: ['Microsoft', 'Apple'],
        difficulty: 'Medium'
      },
      {
        question: 'How do unbalanced datasets affect KNN?',
        answer: 'If one class vastly outnumbers another, the majority class will dominate the neighborhoods simply due to its higher frequency. This causes KNN to be biased toward predicting the majority class. Using Weighted KNN or resampling techniques (SMOTE) helps mitigate this.',
        companyTags: ['Netflix', 'IBM'],
        difficulty: 'Medium'
      },
      {
        question: 'Why is feature scaling mandatory in KNN?',
        answer: 'Because KNN calculates physical distance between points. A feature with a range of [0, 10000] will numerically dominate the distance calculation over a feature with a range of [0, 1]. Scaling (e.g., Standardization) ensures all features contribute equally.',
        companyTags: ['Meta', 'Amazon'],
        difficulty: 'Easy'
      },
      {
        question: 'How do you evaluate a KNN classifier?',
        answer: 'It is evaluated like any other classifier using metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC. Cross-validation is highly recommended to evaluate its generalization, especially since KNN has no internal training loss metric.',
        companyTags: ['Google', 'Pinterest'],
        difficulty: 'Easy'
      },
      {
        question: 'How can you reduce the time complexity of KNN?',
        answer: '1) Reduce dimensions using PCA or feature selection. 2) Use tree-based indexing (KD-Trees, Ball-Trees). 3) Use Approximate Nearest Neighbors (ANN) algorithms like Locality Sensitive Hashing (LSH) or HNSW for massive datasets.',
        companyTags: ['Spotify', 'Stripe'],
        difficulty: 'Hard'
      },
      {
        question: 'Is KNN a parametric or non-parametric algorithm?',
        answer: 'KNN is non-parametric. It does not assume any specific mathematical form or underlying distribution for the data (unlike Linear Regression which assumes a linear relationship). It lets the data speak for itself.',
        companyTags: ['Apple', 'LinkedIn'],
        difficulty: 'Easy'
      },
      {
        question: 'Does KNN work well with text data?',
        answer: 'Generally, no. Text data (BoW or TF-IDF) is extremely high-dimensional and sparse, which triggers the curse of dimensionality. Distance metrics lose meaning. Naive Bayes or SVMs are much better suited for text, though Cosine Similarity can make KNN somewhat viable.',
        companyTags: ['Twitter', 'Amazon'],
        difficulty: 'Medium'
      },
      {
        question: 'How does a Voronoi tessellation relate to KNN?',
        answer: 'For a 1-Nearest Neighbor classifier, the decision boundaries form a Voronoi tessellation of the feature space, where every point in a given polygon is closest to the training data point located within that polygon.',
        companyTags: ['DeepMind', 'Google'],
        difficulty: 'Hard'
      },
      {
        question: 'How can KNN be used to impute missing values?',
        answer: 'KNN Imputer identifies the K most similar data points (ignoring the missing feature) and fills in the missing value by averaging (or taking the mode of) the non-missing values of that specific feature from the neighbors.',
        companyTags: ['Microsoft', 'Bloomberg'],
        difficulty: 'Medium'
      },
      {
        question: 'What is a Radius Neighbors Classifier?',
        answer: 'Instead of finding a fixed number of K neighbors, it finds all neighbors within a fixed physical radius R. This is useful when data density is highly variable; sparse regions will use fewer neighbors, while dense regions will use more.',
        companyTags: ['Meta', 'Uber'],
        difficulty: 'Medium'
      },
      {
        question: 'Why do we typically use an odd K for binary classification?',
        answer: 'An odd K prevents 50/50 ties when the neighbors vote. If K=4, you could have 2 votes for Class A and 2 votes for Class B. With K=3 or K=5, a strict majority is mathematically guaranteed.',
        companyTags: ['Google', 'Netflix'],
        difficulty: 'Easy'
      }
    ],
    coding: {
      tutorial: {
        title: 'Minkowski Distance Calculator',
        description: 'Implement Minkowski distance formula for vectors a and b with scale p_norm. Sklearn equivalent: `sklearn.metrics.pairwise.minkowski_distances`.',
        pseudoCode: 'Compute absolute differences between a and b.\nRaise differences to the power of p_norm.\nSum the results.\nTake the (1/p_norm) root of the sum.',
        starterCode: `import numpy as np

def minkowski_distance(a, b, p_norm):
    # TODO: Calculate Minkowski distance
    return 0.0

a = np.array([1.0, 3.0])
b = np.array([2.0, 1.0])
print("Distance:", np.round(minkowski_distance(a, b, 2), 3))`,
        expectedOutput: 'Distance: 2.236',
        solution: `import numpy as np

def minkowski_distance(a, b, p_norm):
    return np.sum(np.abs(a - b) ** p_norm) ** (1 / p_norm)

a = np.array([1.0, 3.0])
b = np.array([2.0, 1.0])
print("Distance:", np.round(minkowski_distance(a, b, 2), 3))`,
        hints: ['Compute absolute differences with np.abs.', 'Exponentiate by p_norm, sum all elements, and take root.'],
        testKeywords: ['np.abs', 'np.sum', '** (1 /']
      },
      project: {
        title: 'Iris Flower Classification with KNN',
        description: 'Classify three species of Iris flowers (Setosa=0, Versicolor=1, Virginica=2) using four real botanical measurements: sepal length, sepal width, petal length, and petal width (all in cm). This is one of the most famous classification benchmarks in machine learning, originally published by Ronald Fisher in 1936. Use `KNeighborsClassifier` from sklearn and evaluate predictions on three held-out flowers.',
        pseudoCode: 'Load iris measurements (sepal_length, sepal_width, petal_length, petal_width)\nknn = KNeighborsClassifier(n_neighbors=3)\nknn.fit(X_train, y_train)\npredictions = knn.predict(X_test)\nspecies = ["Setosa", "Versicolor", "Virginica"]\nprint(predicted species labels)',
        starterCode: `from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Iris Flower Dataset (real measurements in cm)
# Features: [sepal_length, sepal_width, petal_length, petal_width]
# Labels: 0=Setosa, 1=Versicolor, 2=Virginica
X_train = np.array([
    [5.1, 3.5, 1.4, 0.2],  # Setosa
    [4.9, 3.0, 1.4, 0.2],  # Setosa
    [4.7, 3.2, 1.3, 0.2],  # Setosa
    [5.0, 3.6, 1.4, 0.2],  # Setosa
    [7.0, 3.2, 4.7, 1.4],  # Versicolor
    [6.4, 3.2, 4.5, 1.5],  # Versicolor
    [6.9, 3.1, 4.9, 1.5],  # Versicolor
    [5.5, 2.3, 4.0, 1.3],  # Versicolor
    [6.3, 3.3, 6.0, 2.5],  # Virginica
    [5.8, 2.7, 5.1, 1.9],  # Virginica
    [7.1, 3.0, 5.9, 2.1],  # Virginica
    [6.3, 2.9, 5.6, 1.8],  # Virginica
])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

X_test = np.array([
    [4.8, 3.1, 1.5, 0.1],  # should be Setosa
    [6.5, 3.0, 4.6, 1.5],  # should be Versicolor
    [6.7, 3.0, 5.8, 2.2],  # should be Virginica
])

species = ["Setosa", "Versicolor", "Virginica"]

# TODO: Initialize KNeighborsClassifier with n_neighbors=3
# TODO: Fit on X_train, y_train
# TODO: Predict X_test and print the predicted species names
predictions = []
print("Predicted species:", predictions)`,
        expectedOutput: 'Predicted species: [Setosa, Versicolor, Virginica]',
        solution: `from sklearn.neighbors import KNeighborsClassifier
import numpy as np

X_train = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [4.7, 3.2, 1.3, 0.2],
    [5.0, 3.6, 1.4, 0.2],
    [7.0, 3.2, 4.7, 1.4],
    [6.4, 3.2, 4.5, 1.5],
    [6.9, 3.1, 4.9, 1.5],
    [5.5, 2.3, 4.0, 1.3],
    [6.3, 3.3, 6.0, 2.5],
    [5.8, 2.7, 5.1, 1.9],
    [7.1, 3.0, 5.9, 2.1],
    [6.3, 2.9, 5.6, 1.8],
])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

X_test = np.array([
    [4.8, 3.1, 1.5, 0.1],
    [6.5, 3.0, 4.6, 1.5],
    [6.7, 3.0, 5.8, 2.2],
])

species = ["Setosa", "Versicolor", "Virginica"]

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print("Predicted species:", [species[p] for p in predictions])`,
        hints: ['Instantiate KNeighborsClassifier(n_neighbors=3).', 'Call .fit() with the training data, then .predict() on X_test.', 'Use a list comprehension with the species list to convert integer labels to names.'],
        testKeywords: ['KNeighborsClassifier', 'fit', 'predict']
      },
      assignment: {
        title: 'Movie Genre Recommendation with KNN',
        description: 'A streaming platform wants to recommend the genre a user will enjoy based on their ratings for 5 existing genres: action, comedy, drama, horror, and sci-fi (each rated 1–5). Build a KNN Pipeline with `StandardScaler` so that slight differences in rating scale don\'t skew the distance, then evaluate with cross-validation. This mirrors how collaborative filtering systems bootstrap genre preference predictions from rating vectors.',
        pseudoCode: 'Load user rating profiles (action, comedy, drama, horror, scifi ratings 1-5)\npipeline = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=3))])\nscores = cross_val_score(pipeline, X, y, cv=3)\nprint(f"Mean Accuracy: {scores.mean():.2f}")',
        starterCode: `from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Movie Genre Recommendation Dataset
# Features: [action_rating, comedy_rating, drama_rating, horror_rating, scifi_rating] (1-5 scale)
# Target: favorite genre (0=Action, 1=Comedy, 2=Drama)
X = np.array([
    [5, 2, 2, 1, 4],  # Action lover
    [5, 1, 3, 2, 5],  # Action lover
    [4, 2, 1, 1, 5],  # Action lover
    [4, 1, 2, 1, 4],  # Action lover
    [1, 5, 3, 2, 1],  # Comedy lover
    [2, 5, 4, 1, 2],  # Comedy lover
    [1, 4, 2, 1, 1],  # Comedy lover
    [2, 5, 3, 2, 1],  # Comedy lover
    [2, 2, 5, 1, 2],  # Drama lover
    [1, 3, 5, 2, 1],  # Drama lover
    [2, 2, 4, 1, 1],  # Drama lover
    [1, 2, 5, 1, 2],  # Drama lover
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

# TODO: Create a Pipeline with StandardScaler and KNeighborsClassifier(n_neighbors=3)
# TODO: Evaluate using cross_val_score with cv=3
# TODO: Print the mean accuracy formatted to 2 decimal places
mean_accuracy = 0.0
print(f"Mean Accuracy: {mean_accuracy:.2f}")`,
        expectedOutput: 'Mean Accuracy: 1.00',
        solution: `from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

X = np.array([
    [5, 2, 2, 1, 4],
    [5, 1, 3, 2, 5],
    [4, 2, 1, 1, 5],
    [4, 1, 2, 1, 4],
    [1, 5, 3, 2, 1],
    [2, 5, 4, 1, 2],
    [1, 4, 2, 1, 1],
    [2, 5, 3, 2, 1],
    [2, 2, 5, 1, 2],
    [1, 3, 5, 2, 1],
    [2, 2, 4, 1, 1],
    [1, 2, 5, 1, 2],
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=3))
])

scores = cross_val_score(pipeline, X, y, cv=3)
mean_accuracy = scores.mean()

print(f"Mean Accuracy: {mean_accuracy:.2f}")`,
        hints: ['Pipeline takes a list of (name, estimator) tuples — use descriptive names like "scaler" and "knn".', 'cross_val_score evaluates the full pipeline end-to-end on each fold — no manual scaling needed.', 'Call .mean() on the scores array to get average accuracy.'],
        testKeywords: ['Pipeline', 'StandardScaler', 'cross_val_score', 'mean']
      }
    }
  };
