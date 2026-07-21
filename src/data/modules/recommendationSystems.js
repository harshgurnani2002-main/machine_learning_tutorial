export const recommendationSystems = {
    id: 'recommendation-systems',
    title: 'Recommendation Systems',
    category: 'Advanced & MLOps',
    description: 'Matrix factorization, collaborative filtering, SVD-based recommendation, and content-based approaches — building systems that predict what users will love.',
    formula: '\\hat{r}_{ui} = \\mu + b_u + b_i + q_i^T p_u',
    interactiveSummary: 'Explore the interactive recommender simulator to see matrix factorization in action. Click on any unknown rating in the user-item matrix to predict its value using SVD decomposition, visualize latent factors as 2D points, and see how cold-start users get handled with global mean defaults.',
    simulatorId: 'recommender',
    theory: `### Recommendation Systems
  
  #### What Are Recommendation Systems?
  Recommendation systems predict user preferences for items (movies, products, articles). Two dominant paradigms exist:
  
  **Collaborative Filtering (CF)** recommends based on user-item interaction patterns alone — "users like you also liked...". It requires no item metadata, only historical ratings/clicks.
  
  **Content-Based Filtering** recommends based on item features and user profiles — "you liked action movies with female leads, so here are more". It requires rich item metadata.
  
  **Hybrid Approaches** combine both for better coverage and accuracy.
  
  #### User-Based Collaborative Filtering
  Find users \\(v\\) similar to the target user \\(u\\) using **cosine similarity** or **Pearson correlation** on their rating vectors. Predict rating for item \\(i\\) as a weighted average:
  
  $$\\hat{r}_{ui} = \\frac{\\sum_{v \\in N_k(u)} \\text{sim}(u, v) \\cdot r_{vi}}{\\sum_{v \\in N_k(u)} |\\text{sim}(u, v)|}$$
  
  Pros: simple, intuitive. Cons: scales poorly with users, struggles with sparse matrices.
  
  #### Item-Based Collaborative Filtering
  Compute similarity between items instead of users. Recommend items that are similar to ones the user already liked:
  
  $$\\hat{r}_{ui} = \\frac{\\sum_{j \\in N_k(i)} \\text{sim}(i, j) \\cdot r_{uj}}{\\sum_{j \\in N_k(i)} |\\text{sim}(i, j)|}$$
  
  More stable than user-based because item-item relationships change slower than user tastes. Used by Amazon's "customers who bought this also bought".
  
  #### Matrix Factorization (SVD)
  The core of modern CF. Decompose the \\(m \\times n\\) rating matrix \\(R\\) into two lower-rank matrices:
  
  $$R \\approx P \\cdot Q^T$$
  
  Where \\(P\\) is an \\(m \\times k\\) user-factor matrix and \\(Q\\) is an \\(n \\times k\\) item-factor matrix. The predicted rating is:
  
  $$\\hat{r}_{ui} = q_i^T p_u$$
  
  **With bias terms** (used in practice):
  
  $$\\hat{r}_{ui} = \\mu + b_u + b_i + q_i^T p_u$$
  
  where \\(\\mu\\) is the global mean rating, \\(b_u\\) is the user bias (how much a user deviates from the mean), and \\(b_i\\) is the item bias (how much an item deviates from the mean).
  
  Training minimizes regularized squared error:
  
  $$\\min_{P, Q, b} \\sum_{(u,i) \\in \\kappa} (r_{ui} - \\mu - b_u - b_i - q_i^T p_u)^2 + \\lambda(||p_u||^2 + ||q_i||^2 + b_u^2 + b_i^2)$$
  
  Optimized via **SGD** or **ALS** (Alternating Least Squares). SGD update for each observed rating:
  
  $$e_{ui} = r_{ui} - \\hat{r}_{ui}$$
  $$p_u \\leftarrow p_u + \\gamma(e_{ui} \\cdot q_i - \\lambda \\cdot p_u)$$
  $$q_i \\leftarrow q_i + \\gamma(e_{ui} \\cdot p_u - \\lambda \\cdot q_i)$$
  
  #### What Are Latent Factors?
  Each user gets a \\(k\\)-dimensional vector \\(p_u\\) and each item gets a \\(k\\)-dimensional vector \\(q_i\\). These factors automatically learn semantic concepts (genre, tone, quality) from the rating patterns alone. For movies, a factor might represent "action vs drama" or "mainstream vs indie".
  
  #### Content-Based Filtering
  Build a user profile vector \\(\\theta_u\\) from item features and a feature matrix \\(X_i\\) for each item:
  
  $$\\hat{r}_{ui} = \\theta_u^T X_i$$
  
  Feature vectors come from item metadata (genres, tags, descriptions). The user profile can be the average of features of items they rated highly, or learned via regression.
  
  #### Hybrid Approaches
  Combine CF and content-based:
  - **Weighted**: linear combination of both predictions
  - **Feature-augmented**: add content features as side information to matrix factorization
  - **Cascade**: use one method to fill gaps for the other
  
  #### The Cold-Start Problem
  The CF failure case: what to recommend for a new user (no ratings) or a new item (no interactions)?
  
  Solutions:
  - **New users**: ask for initial preferences (onboarding), use demographics/context, recommend popular items, use content-based until enough data
  - **New items**: use content-based features to compare with existing items, recommend to exploratory users
  - **Global mean default**: the simplest fallback — predict the dataset's mean rating
  
  #### Evaluation Metrics
  
  **RMSE (Root Mean Squared Error)**:
  $$\\text{RMSE} = \\sqrt{\\frac{1}{|\\mathcal{T}|} \\sum_{(u,i) \\in \\mathcal{T}} (r_{ui} - \\hat{r}_{ui})^2}$$
  
  Penalizes large errors heavily. Good for explicit rating prediction.
  
  **Precision@k and Recall@k** for top-k recommendation:
  $$\\text{Precision@k} = \\frac{\\#\\text{ relevant items in top-k}}{k}$$
  $$\\text{Recall@k} = \\frac{\\#\\text{ relevant items in top-k}}{\\#\\text{ total relevant items}}$$
  
  Precision@k measures how many of the recommended items the user likes. Recall@k measures how many of the liked items were recommended.
  
  **Hit Rate** (for implicit feedback / top-k evaluation): fraction of users for which the held-out item appears in the top-k recommendations.
  
  #### Industry Architectures
  
  **Netflix**: Three-stage pipeline — (1) candidate generation using matrix factorization and item similarity, (2) ranking with a deep neural network using hundreds of features, (3) bandit-style exploration for new content. Their famous Netflix Prize (2006-2009) popularized SVD-based matrix factorization and ensemble methods.
  
  **YouTube**: Two deep neural networks — (1) candidate generation from user watch history (softmax over millions of videos), (2) ranking network with expected watch time as the objective. Uses implicit feedback (watched/didn't watch) rather than explicit ratings. Features include video embeddings, user demographics, and search history.
  
  Both systems re-train periodically and use A/B testing extensively to validate changes.
  
  #### Key Takeaways
  - Collaborative filtering uses "wisdom of the crowd" — no item metadata needed
  - Matrix factorization learns latent user/item factors from rating patterns
  - Bias terms (global mean, user bias, item bias) capture baseline effects
  - Cold-start requires fallback strategies (popularity, content-based defaults)
  - Evaluation must match the business metric — RMSE for rating accuracy, precision@k for top-k recommendation
  
  #### Python Implementation
  
  \`\`\`python
  from surprise import SVD, Dataset, Reader
  from surprise.model_selection import cross_validate
  
  data = Dataset.load_builtin('ml-100k')
  algo = SVD()
  results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
  print(f"Mean RMSE: {results['test_rmse'].mean():.3f}")
  \`\`\`,
  `,
    quiz: [
        {
            id: 'rec_q1',
            question: 'What is the fundamental difference between collaborative filtering and content-based filtering?',
            options: [
                'Collaborative filtering uses item metadata; content-based uses user ratings',
                'Collaborative filtering uses user-item interaction patterns; content-based uses item features and user profiles',
                'Collaborative filtering only works with implicit feedback; content-based needs explicit ratings',
                'Collaborative filtering requires deep learning; content-based uses only linear models'
            ],
            correctAnswer: 'Collaborative filtering uses user-item interaction patterns; content-based uses item features and user profiles',
            explanation: 'CF finds patterns in who rated what, regardless of what items are about. Content-based needs feature descriptions of items (genre, tags, etc.) to match against user preferences.'
        },
        {
            id: 'rec_q2',
            question: 'In SVD-based matrix factorization for recommendation, what do the matrices P and Q represent?',
            options: [
                'P is user bias matrix; Q is item bias matrix',
                'P is user-factor matrix; Q is item-factor matrix',
                'P is the rating matrix; Q is the latent factor dimension',
                'P is item-item similarity; Q is user-user similarity'
            ],
            correctAnswer: 'P is user-factor matrix; Q is item-factor matrix',
            explanation: 'R ≈ P × Q^T, where P (m×k) maps users to k latent factors and Q (n×k) maps items to k latent factors. Their dot product predicts the rating.'
        },
        {
            id: 'rec_q3',
            question: 'What problem does the bias term b_u (user bias) capture in matrix factorization?',
            options: [
                'The fact that some items are systematically more popular than others',
                'The fact that some users rate systematically higher or lower than the global average',
                'The tendency of users to rate only items they already like',
                'The temporal drift in user preferences over time'
            ],
            correctAnswer: 'The fact that some users rate systematically higher or lower than the global average',
            explanation: 'A generous user who gives 4-5 stars consistently has a positive b_u, while a strict user who gives 2-3 stars has a negative b_u. This separates baseline behavior from true preferences.'
        },
        {
            id: 'rec_q4',
            question: 'What is the cold-start problem in recommendation systems?',
            options: [
                'When the recommendation algorithm runs too slowly to serve real-time traffic',
                'When a new user or new item has no interaction history, making CF-based predictions impossible',
                'When the dataset contains only ratings below 3 stars (cold ratings)',
                'When the system becomes less accurate during winter holidays due to changing user behavior'
            ],
            correctAnswer: 'When a new user or new item has no interaction history, making CF-based predictions impossible',
            explanation: 'CF relies entirely on interaction patterns. A user with 0 ratings or an item with 0 ratings cannot be modeled through factorization alone, requiring fallback strategies like global mean or content-based features.'
        },
        {
            id: 'rec_q5',
            question: 'How does precision@k differ from recall@k in evaluating top-k recommendations?',
            options: [
                'Precision@k measures fraction of recommended items that are relevant; recall@k measures fraction of relevant items that were recommended',
                'Precision@k measures coverage; recall@k measures accuracy',
                'They are the same metric under different names',
                'Precision@k is for implicit feedback; recall@k is for explicit feedback'
            ],
            correctAnswer: 'Precision@k measures fraction of recommended items that are relevant; recall@k measures fraction of relevant items that were recommended',
            explanation: 'If you recommend 10 items and the user likes 3 of them, precision@10 = 0.3. If the user likes 10 total items and you caught 3, recall@10 = 0.3. High precision means few bad recommendations; high recall means you surface most of what they would like.'
        },
        {
            id: 'rec_q6',
            question: 'What does the ALS (Alternating Least Squares) algorithm do in matrix factorization?',
            options: [
                'It alternates between fixing user factors and optimizing item factors, and vice versa',
                'It alternates between collaborative and content-based filtering',
                'It alternates between training on weekdays and weekends',
                'It randomly drops half the ratings each iteration to reduce overfitting'
            ],
            correctAnswer: 'It alternates between fixing user factors and optimizing item factors, and vice versa',
            explanation: 'ALS treats matrix factorization as a quadratic optimization problem. Fixing user factors makes the item factor subproblem convex and solvable by least squares. Alternating between the two converges to a good solution and handles implicit feedback well.'
        },
        {
            id: 'rec_q7',
            question: 'Why does YouTube use expected watch time rather than click-through rate as its ranking objective?',
            options: [
                'Because watch time is easier to compute than click-through rate',
                'Because maximizing clicks would incentivize clickbait titles with low user satisfaction',
                'Because watch time requires fewer features to predict accurately',
                'Because the YouTube algorithm is not allowed to use click data'
            ],
            correctAnswer: 'Because maximizing clicks would incentivize clickbait titles with low user satisfaction',
            explanation: 'A clickbait video might get many clicks but users abandon it after 10 seconds. Optimizing for watch time rewards content that keeps users engaged, which aligns with long-term user satisfaction and platform health.'
        },
        {
            id: 'rec_q8',
            question: 'What is the main advantage of item-based collaborative filtering over user-based collaborative filtering?',
            options: [
                'Item-based CF is always more accurate than user-based CF',
                'Item-item relationships are more stable over time than user-user relationships',
                'Item-based CF requires no matrix storage',
                'Item-based CF works without any rating data'
            ],
            correctAnswer: 'Item-item relationships are more stable over time than user-user relationships',
            explanation: 'Users\' tastes may drift (a user who liked action movies in 2020 may prefer documentaries in 2024), but a sci-fi movie remains similar to other sci-fi movies. Item-item similarities can be precomputed and updated less frequently, making them more practical for large-scale systems.'
        }
    ],
    coding: {
        tutorial: {
            title: 'Cosine Similarity Between Users',
            description: 'Compute cosine similarity between two user rating vectors to measure how similar their tastes are.',
            pseudoCode: `1. Define two user rating vectors (with 0 for unknown)
2. Compute dot product of the vectors
3. Compute magnitude (norm) of each vector
4. Return dot / (mag1 * mag2)`,
            starterCode: `import numpy as np

def cosine_similarity(a, b):
    # TODO: Compute the dot product of a and b
    # TODO: Compute the norm (magnitude) of a and b
    # TODO: Return the cosine similarity (handle zero-division)
    pass

# User rating vectors (0 = no rating)
user1 = np.array([5, 3, 0, 1, 4, 0, 2, 0])
user2 = np.array([4, 0, 0, 1, 5, 2, 0, 3])

sim = cosine_similarity(user1, user2)
print(f"Similarity: {sim:.3f}")`,
            expectedOutput: 'Similarity: 0.857',
            solution: `import numpy as np

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

user1 = np.array([5, 3, 0, 1, 4, 0, 2, 0])
user2 = np.array([4, 0, 0, 1, 5, 2, 0, 3])
sim = cosine_similarity(user1, user2)
print(f"Similarity: {sim:.3f}")`,
            hints: [
                'Use np.dot(a, b) for dot product',
                'Use np.linalg.norm() for magnitude',
                'Check for zero norm to avoid division by zero',
                'Only non-zero entries in both vectors contribute to the dot product'
            ],
            testKeywords: ['cosine_similarity', 'np.dot', 'np.linalg.norm', 'dot /']
        },
        project: {
            title: 'SVD Matrix Factorization with NumPy',
            description: 'Implement a simplified SVD-based matrix factorization using numpy.linalg.svd. Reconstruct the rating matrix, predict missing values, and compute RMSE on held-out test data.',
            pseudoCode: `1. Create a rating matrix with some known values (0 = unknown)
2. Perform SVD: U, S, Vt = np.linalg.svd(R, full_matrices=False)
3. Keep only top-k singular values for dimensionality reduction
4. Reconstruct: R_hat = U_k @ diag(S_k) @ Vt_k
5. Compare predictions with original known ratings
6. Print RMSE`,
            starterCode: `import numpy as np

# Rating matrix (6 users, 8 items), 0 = unknown
R = np.array([
    [5, 3, 0, 1, 4, 0, 2, 0],
    [4, 0, 0, 1, 5, 2, 0, 3],
    [1, 1, 0, 5, 0, 4, 3, 0],
    [0, 0, 2, 4, 0, 5, 0, 1],
    [5, 4, 3, 0, 2, 0, 0, 0],
    [0, 5, 0, 0, 3, 1, 4, 5],
])

k = 3  # number of latent factors

# TODO: Perform SVD on R (use full_matrices=False)
# U, S, Vt = ...

# TODO: Keep only top-k singular values/vectors
# U_k, S_k, Vt_k = ...

# TODO: Reconstruct the matrix
# R_hat = ...

# TODO: Compute RMSE on the known (non-zero) entries
# rmse = ...
# print(f"RMSE: {rmse:.4f}")
# print("Predicted missing at (0,2):", R_hat[0, 2])
# print("Predicted missing at (1,1):", R_hat[1, 1])`,
            expectedOutput: 'RMSE: 1.1082\nPredicted missing at (0,2): 3.4173\nPredicted missing at (1,1): 2.8235',
            solution: `import numpy as np

R = np.array([
    [5, 3, 0, 1, 4, 0, 2, 0],
    [4, 0, 0, 1, 5, 2, 0, 3],
    [1, 1, 0, 5, 0, 4, 3, 0],
    [0, 0, 2, 4, 0, 5, 0, 1],
    [5, 4, 3, 0, 2, 0, 0, 0],
    [0, 5, 0, 0, 3, 1, 4, 5],
])

k = 3
U, S, Vt = np.linalg.svd(R, full_matrices=False)
U_k = U[:, :k]
S_k = np.diag(S[:k])
Vt_k = Vt[:k, :]

R_hat = U_k @ S_k @ Vt_k

mask = R > 0
rmse = np.sqrt(np.mean((R[mask] - R_hat[mask]) ** 2))
print(f"RMSE: {rmse:.4f}")
print("Predicted missing at (0,2):", round(R_hat[0, 2], 4))
print("Predicted missing at (1,1):", round(R_hat[1, 1], 4))`,
            hints: [
                'Use np.linalg.svd(R, full_matrices=False) for the decomposition',
                'Slice U[:, :k], keep S[:k] as diagonal, slice Vt[:k, :]',
                'Reconstruct with U_k @ np.diag(S_k) @ Vt_k',
                'Create a boolean mask with R > 0 to select known entries only'
            ],
            testKeywords: ['np.linalg.svd', 'full_matrices', 'diag', 'RMSE', 'mask']
        },
        assignment: {
            title: 'Build a Content-Based Movie Recommender',
            description: 'Build a simple content-based recommendation system using movie genre features. Given a user profile (their favorite genres and ratings), compute the dot product between user preferences and movie genre vectors to find the best recommendations.',
            pseudoCode: `1. Define movie-genre feature matrix (binary: 1 if movie has that genre)
2. Define user preference vector (rating 0-5 for each genre)
3. Compute predicted score = user_prefs @ movie_features^T
4. Sort movies by predicted score
5. Recommend top-3 movies`,
            starterCode: `import numpy as np

# Movie-genre matrix: 8 movies, 5 genres [Action, Comedy, Drama, SciFi, Romance]
movie_features = np.array([
    [1, 0, 1, 0, 0],  # The Matrix
    [0, 1, 1, 0, 0],  # The Hangover
    [0, 1, 0, 0, 1],  # Romancing the Stone
    [1, 0, 0, 1, 0],  # Interstellar
    [0, 0, 1, 0, 1],  # Titanic
    [1, 0, 0, 0, 1],  # Mr and Mrs Smith
    [0, 1, 0, 1, 0],  # Guardians of the Galaxy
    [0, 0, 1, 1, 0],  # Arrival
])

movie_names = [
    "The Matrix", "The Hangover", "Romancing the Stone",
    "Interstellar", "Titanic", "Mr and Mrs Smith",
    "Guardians of the Galaxy", "Arrival"
]

# User profile: how much they like each genre (0-5)
user_profile = np.array([5, 1, 4, 5, 2])  # loves Action, SciFi, Drama

# TODO: Compute predicted scores for all movies
# scores = ...

# TODO: Get the indices of the top-3 movies
# top_indices = ...

# TODO: Print the top-3 movie recommendations with scores
# for idx in top_indices:
#     print(f"{movie_names[idx]}: {scores[idx]:.2f}")`,
            expectedOutput: 'Interstellar: 10.00\nArrival: 9.00\nThe Matrix: 9.00',
            solution: `import numpy as np

movie_features = np.array([
    [1, 0, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 1, 0],
])

movie_names = [
    "The Matrix", "The Hangover", "Romancing the Stone",
    "Interstellar", "Titanic", "Mr and Mrs Smith",
    "Guardians of the Galaxy", "Arrival"
]

user_profile = np.array([5, 1, 4, 5, 2])

scores = user_profile @ movie_features.T
top_indices = np.argsort(scores)[::-1][:3]

print("Top 3 recommendations:")
for idx in top_indices:
    print(f"{movie_names[idx]}: {scores[idx]:.2f}")`,
            hints: [
                'Use numpy dot product: user_profile @ movie_features.T',
                'Sort descending with np.argsort(scores)[::-1]',
                'Slice [:3] for top-3 indices',
                'The score for a movie is the sum of genre weights that match the user profile'
            ],
            testKeywords: ['movie_features', 'user_profile', '@', 'argsort', 'top-3']
        }
    },
    interviewQuestions: [
        {
            question: 'Explain the bias terms in matrix factorization for recommendation systems. Why are they important?',
            answer: 'The prediction formula is r_hat_ui = μ + b_u + b_i + q_i^T p_u. The global mean μ captures the overall average rating. User bias b_u captures that some users rate systematically higher or lower than others (e.g., a generous user gives 4-5 stars while a strict user gives 2-3 stars). Item bias b_i captures that some items are universally more popular or better-liked (e.g., a blockbuster movie gets higher average ratings regardless of who is watching). Without biases, the latent factors must also explain these baseline effects, wasting capacity. Adding biases typically improves RMSE by 10-20% and produces more interpretable factors.',
            companyTags: ['Netflix', 'Amazon', 'Spotify'],
            difficulty: 'Hard'
        },
        {
            question: 'How would you design a recommendation system for a new video streaming platform that has no user interaction data yet?',
            answer: 'This is the cold-start problem at the platform level. I would use a multi-phase approach: (1) Item metadata embedding: build content-based feature vectors for all content using genre, director, cast, description embeddings. (2) Curated onboarding: ask new users to select favorite genres/movies from a diverse "taste test" — this immediately creates a profile. (3) Popularity baseline: recommend trending/popular content to all users until individual data accumulates. (4) Hybrid transition: gradually shift from content-based to collaborative filtering as interactions grow. (5) A/B testing framework: validate every change against retention and watch time metrics.',
            companyTags: ['Netflix', 'Disney+', 'HBO'],
            difficulty: 'Hard'
        },
        {
            question: 'What is the difference between explicit feedback (ratings) and implicit feedback (clicks, watch time), and how does the modeling approach differ?',
            answer: 'Explicit feedback (star ratings, likes) directly expresses preference magnitude but is sparse — users rate only a tiny fraction of items. Implicit feedback (views, clicks, dwell time) is abundant but noisy — a click might mean "interested" or "accidental". Modeling differs: explicit feedback uses regression-style loss (minimize squared error between predicted and actual rating). Implicit feedback often uses pairwise ranking loss (BPR) or pointwise confidence-weighting (like in ALS for implicit feedback, where confidence = 1 + α * count). The key insight with implicit data is that non-observation may indicate disliking OR unawareness — we need to model confidence, not just presence.',
            companyTags: ['YouTube', 'Spotify', 'Pinterest'],
            difficulty: 'Hard'
        },
        {
            question: 'How does the YouTube recommendation pipeline work from candidate generation to final ranking?',
            answer: 'YouTube uses a two-stage deep learning architecture. Stage 1 — Candidate generation: a deep neural network takes user watch history (video embeddings), search tokens, and demographics as input. It outputs a softmax over millions of videos, producing a few hundred candidates. Training uses implicit feedback (watched/not watched). Stage 2 — Ranking: a deeper network with richer features (video-level, user-level, context) scores each candidate. The objective is expected watch time (weighted logistic regression with watch time as positive weight), not click-through rate — this prevents clickbait. The system is retrained periodically, and exploration is added via forcing in some random candidates. A/B testing is continuous.',
            companyTags: ['YouTube', 'Google', 'TikTok'],
            difficulty: 'Hard'
        },
        {
            question: 'How do you evaluate a recommendation system in a production setting beyond offline metrics like RMSE?',
            answer: 'Offline metrics (RMSE, precision@k, recall@k) are necessary but insufficient. Production evaluation requires: (1) Online A/B testing with metrics like CTR, watch time, retention, and conversion rate. (2) User surveys and explicit feedback to measure satisfaction beyond engagement (engagement can be addictive but unsatisfying). (3) Coverage — are we recommending diverse content or just popular items? (4) Novelty and serendipity — are we showing users things they would not have found themselves? (5) Freshness — how quickly does the system surface new content? (6) Fairness — does the system perform equally well across different user demographics? A common failure mode is optimizing click-through rate at the cost of long-term user satisfaction, so metrics like "long-term retention" in holdout experiments are critical.',
            companyTags: ['Netflix', 'Amazon', 'Meta'],
            difficulty: 'Hard'
        }
    ]
};
