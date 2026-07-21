"""
Add Python sklearn/numpy code snippets to all module files.

For each module file:
1. Reads the file
2. Finds the theory field
3. If it uses JSON string format ("theory": "..."), converts to template literal
4. Adds a relevant Python code snippet after the Key Takeaways section
5. Writes the modified file
"""

import re
import os
import sys

MODULES_DIR = os.path.dirname(os.path.abspath(__file__))

# Map of module filename -> (topic, python snippet)
SNIPPETS = {
    "linearRegression": (
        "LinearRegression",
        "from sklearn.linear_model import LinearRegression\n"
        "from sklearn.model_selection import train_test_split\n\n"
        "X, y = [[1], [2], [3], [4], [5]], [2, 4, 6, 8, 10]\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
        "model = LinearRegression()\n"
        "model.fit(X_train, y_train)\n"
        "predictions = model.predict(X_test)\n"
        "print(f\"R² Score: {model.score(X_test, y_test):.2f}\")"
    ),
    "logisticRegression": (
        "LogisticRegression",
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.model_selection import train_test_split\n\n"
        "X, y = [[1], [2], [3], [4], [5], [6]], [0, 0, 0, 1, 1, 1]\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)\n"
        "model = LogisticRegression()\n"
        "model.fit(X_train, y_train)\n"
        "acc = model.score(X_test, y_test)\n"
        "print(f\"Accuracy: {acc:.2f}\")"
    ),
    "decisionTrees": (
        "DecisionTreeClassifier",
        "from sklearn.tree import DecisionTreeClassifier\n"
        "from sklearn.model_selection import train_test_split\n\n"
        "X, y = [[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]], [0, 0, 0, 1, 1, 1]\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)\n"
        "model = DecisionTreeClassifier(max_depth=3)\n"
        "model.fit(X_train, y_train)\n"
        "print(f\"Accuracy: {model.score(X_test, y_test):.2f}\")"
    ),
    "recurrentNetworks": (
        "LSTM with Keras",
        "from tensorflow.keras.models import Sequential\n"
        "from tensorflow.keras.layers import LSTM, Dense\n\n"
        "model = Sequential([\n"
        "    LSTM(64, input_shape=(10, 1)),\n"
        "    Dense(1, activation='sigmoid')\n"
        "])\n"
        "model.compile(optimizer='adam', loss='binary_crossentropy')\n"
        "print(model.summary())"
    ),
    "transformerAttention": (
        "Scaled Dot-Product Attention (NumPy)",
        "import numpy as np\n\n"
        "def scaled_dot_product_attention(Q, K, V):\n"
        "    scores = Q @ K.T / np.sqrt(K.shape[-1])\n"
        "    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)\n"
        "    return weights @ V\n\n"
        "Q = K = V = np.array([[1, 0], [0, 1], [1, 1]])\n"
        "output = scaled_dot_product_attention(Q, K, V)\n"
        "print(output)"
    ),
    "wordEmbeddings": (
        "Word2Vec (gensim)",
        "from gensim.models import Word2Vec\n\n"
        "sentences = [['cat', 'sat', 'mat'], ['dog', 'ran', 'park']]\n"
        "model = Word2Vec(sentences, vector_size=50, window=3, min_count=1)\n"
        "vector = model.wv['cat']\n"
        "similar = model.wv.most_similar('cat', topn=2)\n"
        "print(f\"Vector: {vector[:5]}...\\nSimilar: {similar}\")"
    ),
    "biasVarianceTradeoff": (
        "Bias-Variance with sklearn",
        "from sklearn.tree import DecisionTreeRegressor\n"
        "from sklearn.model_selection import cross_val_score\n\n"
        "X, y = [[1], [2], [3], [4], [5], [6]], [1.2, 2.1, 3.3, 4.0, 4.9, 6.2]\n"
        "for depth in [1, 3, 10]:\n"
        "    model = DecisionTreeRegressor(max_depth=depth)\n"
        "    scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')\n"
        "    print(f\"Depth {depth}: MSE = {-scores.mean():.3f}\")"
    ),
    "dbscanClustering": (
        "DBSCAN",
        "from sklearn.cluster import DBSCAN\n"
        "import numpy as np\n\n"
        "X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])\n"
        "db = DBSCAN(eps=3, min_samples=2)\n"
        "labels = db.fit_predict(X)\n"
        "print(f\"Cluster labels: {labels}\")"
    ),
    "neuralActivations": (
        "Activation Functions (NumPy)",
        "import numpy as np\n\n"
        "def sigmoid(x): return 1 / (1 + np.exp(-x))\n"
        "def relu(x): return np.maximum(0, x)\n"
        "def tanh(x): return np.tanh(x)\n\n"
        "x = np.array([-2, -1, 0, 1, 2])\n"
        "print(f\"Sigmoid: {sigmoid(x)}\")\n"
        "print(f\"ReLU:    {relu(x)}\")\n"
        "print(f\"Tanh:    {tanh(x)}\")"
    ),
    "optimizationAlgorithms": (
        "Gradient Descent (NumPy)",
        "import numpy as np\n\n"
        "X, y = np.array([[1], [2], [3]]), np.array([2, 4, 6])\n"
        "w, lr = 0.0, 0.01\n"
        "for _ in range(100):\n"
        "    grad = 2 * X.T @ (X @ w - y) / len(y)\n"
        "    w -= lr * grad\n"
        "print(f\"Weight: {w[0]:.3f}\")"
    ),
    "ridgeLassoRegularization": (
        "Ridge vs Lasso",
        "from sklearn.linear_model import Ridge, Lasso\n"
        "import numpy as np\n\n"
        "X, y = np.array([[1], [2], [3], [4]]), np.array([2.1, 4.0, 5.9, 8.2])\n"
        "ridge = Ridge(alpha=1.0).fit(X, y)\n"
        "lasso = Lasso(alpha=1.0).fit(X, y)\n"
        "print(f\"Ridge coeff: {ridge.coef_[0]:.2f}, Lasso coeff: {lasso.coef_[0]:.2f}\")"
    ),
    "tsneVisualization": (
        "t-SNE",
        "from sklearn.manifold import TSNE\n"
        "import numpy as np\n\n"
        "X = np.random.randn(50, 10)\n"
        "tsne = TSNE(n_components=2, perplexity=5, random_state=42)\n"
        "X_embedded = tsne.fit_transform(X)\n"
        "print(f\"Shape reduced from {X.shape} to {X_embedded.shape}\")"
    ),
    "gradientDescent": (
        "Gradient Descent",
        "import numpy as np\n\n"
        "def gradient_descent(X, y, lr=0.01, epochs=100):\n"
        "    m, w, b = len(y), 0.0, 0.0\n"
        "    for _ in range(epochs):\n"
        "        y_pred = w * X + b\n"
        "        dw = (-2/m) * np.sum(X * (y - y_pred))\n"
        "        db = (-2/m) * np.sum(y - y_pred)\n"
        "        w -= lr * dw; b -= lr * db\n"
        "    return w, b\n\n"
        "w, b = gradient_descent(np.array([1,2,3,4]), np.array([2,4,6,8]))\n"
        "print(f\"w={w:.2f}, b={b:.2f}\")"
    ),
    "supportVectorMachines": (
        "SVM Classifier",
        "from sklearn.svm import SVC\n"
        "from sklearn.model_selection import train_test_split\n\n"
        "X, y = [[1,2],[2,1],[2,3],[3,2],[5,5],[6,6]], [0,0,0,0,1,1]\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)\n"
        "model = SVC(kernel='rbf')\n"
        "model.fit(X_train, y_train)\n"
        "print(f\"Accuracy: {model.score(X_test, y_test):.2f}\")"
    ),
    "knnClassification": (
        "KNN Classifier",
        "from sklearn.neighbors import KNeighborsClassifier\n"
        "from sklearn.model_selection import train_test_split\n\n"
        "X, y = [[1],[2],[3],[10],[11],[12]], [0,0,0,1,1,1]\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)\n"
        "model = KNeighborsClassifier(n_neighbors=3)\n"
        "model.fit(X_train, y_train)\n"
        "print(f\"Accuracy: {model.score(X_test, y_test):.2f}\")"
    ),
    "naiveBayes": (
        "Naive Bayes",
        "from sklearn.naive_bayes import GaussianNB\n"
        "from sklearn.model_selection import train_test_split\n\n"
        "X, y = [[1],[2],[3],[10],[11],[12]], [0,0,0,1,1,1]\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)\n"
        "model = GaussianNB()\n"
        "model.fit(X_train, y_train)\n"
        "print(f\"Accuracy: {model.score(X_test, y_test):.2f}\")"
    ),
    "randomForests": (
        "Random Forest",
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.model_selection import train_test_split\n\n"
        "X, y = [[1,2],[2,3],[3,1],[6,5],[7,7],[8,6]], [0,0,0,1,1,1]\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)\n"
        "model = RandomForestClassifier(n_estimators=50, random_state=42)\n"
        "model.fit(X_train, y_train)\n"
        "print(f\"Accuracy: {model.score(X_test, y_test):.2f}\")"
    ),
    "pcaReduction": (
        "PCA",
        "from sklearn.decomposition import PCA\n"
        "import numpy as np\n\n"
        "X = np.random.randn(20, 5)\n"
        "pca = PCA(n_components=2)\n"
        "X_pca = pca.fit_transform(X)\n"
        "print(f\"Explained variance: {pca.explained_variance_ratio_}\")"
    ),
    "kmeansClustering": (
        "K-Means",
        "from sklearn.cluster import KMeans\n"
        "import numpy as np\n\n"
        "X = np.array([[1,2],[1,4],[1,0],[10,2],[10,4],[10,0]])\n"
        "kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)\n"
        "kmeans.fit(X)\n"
        "print(f\"Centers: {kmeans.cluster_centers_}\")"
    ),
    "multiLayerPerceptron": (
        "MLP Classifier",
        "from sklearn.neural_network import MLPClassifier\n"
        "from sklearn.model_selection import train_test_split\n\n"
        "X, y = [[0,0],[0,1],[1,0],[1,1]], [0,1,1,0]\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)\n"
        "model = MLPClassifier(hidden_layer_sizes=(4,), activation='relu', max_iter=1000, random_state=42)\n"
        "model.fit(X_train, y_train)\n"
        "print(f\"Accuracy: {model.score(X_test, y_test):.2f}\")"
    ),
    "singlePerceptron": (
        "Perceptron (NumPy)",
        "import numpy as np\n\n"
        "def perceptron(X, y, lr=0.1, epochs=10):\n"
        "    w = np.zeros(X.shape[1])\n"
        "    for _ in range(epochs):\n"
        "        for xi, yi in zip(X, y):\n"
        "            update = lr * (yi - np.sign(xi @ w))\n"
        "            w += update * xi\n"
        "    return w\n\n"
        "X = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])\n"
        "y = np.array([1,1,1,-1])\n"
        "print(f\"Weights: {perceptron(X, y)}\")"
    ),
    "convolutionalNetworks": (
        "CNN (Keras)",
        "from tensorflow.keras.models import Sequential\n"
        "from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense\n\n"
        "model = Sequential([\n"
        "    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),\n"
        "    MaxPooling2D((2,2)),\n"
        "    Flatten(),\n"
        "    Dense(10, activation='softmax')\n"
        "])\n"
        "model.compile(optimizer='adam', loss='categorical_crossentropy')\n"
        "print(model.summary())"
    ),
    "ensembleTechniquesBaggingBoosting": (
        "Bagging vs Boosting",
        "from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier\n"
        "from sklearn.tree import DecisionTreeClassifier\n\n"
        "X, y = [[1,2],[2,3],[3,1],[6,5],[7,7],[8,6]], [0,0,0,1,1,1]\n"
        "bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=10)\n"
        "boost = GradientBoostingClassifier(n_estimators=10)\n"
        "print(f\"Bagging: {bag.fit(X,y).score(X,y):.2f}, Boosting: {boost.fit(X,y).score(X,y):.2f}\")"
    ),
    "gradientBoosting": (
        "Gradient Boosting",
        "from sklearn.ensemble import GradientBoostingRegressor\n"
        "import numpy as np\n\n"
        "X, y = np.array([[1],[2],[3],[4],[5]]), np.array([1.5, 3.2, 4.8, 6.1, 8.0])\n"
        "model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, random_state=42)\n"
        "model.fit(X, y)\n"
        "print(f\"Score: {model.score(X, y):.2f}\")"
    ),
    "gradientBoostingAdvanced": (
        "XGBoost",
        "from xgboost import XGBClassifier\n"
        "from sklearn.model_selection import train_test_split\n\n"
        "X, y = [[1,2],[2,3],[3,1],[6,5],[7,7],[8,6]], [0,0,0,1,1,1]\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)\n"
        "model = XGBClassifier(n_estimators=20, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')\n"
        "model.fit(X_train, y_train)\n"
        "print(f\"Accuracy: {model.score(X_test, y_test):.2f}\")"
    ),
    "modelEvaluation": (
        "Model Evaluation Metrics",
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n\n"
        "y_true = [1, 0, 1, 1, 0, 1]\n"
        "y_pred = [1, 0, 1, 0, 0, 1]\n"
        "print(f\"Accuracy: {accuracy_score(y_true, y_pred):.2f}\")\n"
        "print(f\"Precision: {precision_score(y_true, y_pred):.2f}\")\n"
        "print(f\"Recall: {recall_score(y_true, y_pred):.2f}\")\n"
        "print(f\"F1: {f1_score(y_true, y_pred):.2f}\")"
    ),
    "featureEngineering": (
        "Polynomial Features",
        "from sklearn.preprocessing import PolynomialFeatures, StandardScaler\n"
        "import numpy as np\n\n"
        "X = np.array([[1], [2], [3], [4]])\n"
        "poly = PolynomialFeatures(degree=2, include_bias=False)\n"
        "X_poly = poly.fit_transform(X)\n"
        "print(f\"Original: {X.T}\")\n"
        "print(f\"Polynomial: {X_poly.T}\")"
    ),
    "hyperparameterTuning": (
        "GridSearchCV",
        "from sklearn.model_selection import GridSearchCV\n"
        "from sklearn.svm import SVC\n\n"
        "X, y = [[1,2],[2,3],[3,1],[6,5],[7,7],[8,6]], [0,0,0,1,1,1]\n"
        "params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}\n"
        "grid = GridSearchCV(SVC(), params, cv=3)\n"
        "grid.fit(X, y)\n"
        "print(f\"Best params: {grid.best_params_}\")"
    ),
    "imbalancedDatasets": (
        "SMOTE (imblearn)",
        "from imblearn.over_sampling import SMOTE\n"
        "from collections import Counter\n\n"
        "X = [[1],[2],[3],[4],[5],[6],[7],[8]]\n"
        "y = [0, 0, 0, 0, 0, 0, 1, 1]\n"
        "print(f\"Before: {Counter(y)}\")\n"
        "X_res, y_res = SMOTE().fit_resample(X, y)\n"
        "print(f\"After: {Counter(y_res)}\")"
    ),
    "naturalLanguageProcessing": (
        "TF-IDF Vectorizer",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n\n"
        "docs = ['cat sat mat', 'dog ran park', 'cat ran home']\n"
        "vectorizer = TfidfVectorizer()\n"
        "X = vectorizer.fit_transform(docs)\n"
        "print(f\"Shape: {X.shape}\")\n"
        "print(f\"Features: {vectorizer.get_feature_names_out()}\")"
    ),
    "timeSeriesAnalysis": (
        "ARIMA (statsmodels)",
        "from statsmodels.tsa.arima.model import ARIMA\n"
        "import numpy as np\n\n"
        "data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\n"
        "model = ARIMA(data, order=(1,1,1))\n"
        "fitted = model.fit()\n"
        "forecast = fitted.forecast(steps=2)\n"
        "print(f\"Forecast: {forecast.values}\")"
    ),
    "qLearningRl": (
        "Q-Learning Table",
        "import numpy as np\n\n"
        "n_states, n_actions = 5, 2\n"
        "Q = np.zeros((n_states, n_actions))\n"
        "lr, gamma, eps = 0.1, 0.9, 0.1\n"
        "for s in range(n_states - 1):\n"
        "    a = np.argmax(Q[s]) if np.random.random() > eps else np.random.randint(n_actions)\n"
        "    sp, r = s + a, 1.0\n"
        "    Q[s, a] += lr * (r + gamma * np.max(Q[sp]) - Q[s, a])\n"
        "print(f\"Q-table:\\n{Q}\")"
    ),
    "anomalyDetection": (
        "Isolation Forest",
        "from sklearn.ensemble import IsolationForest\n"
        "import numpy as np\n\n"
        "X = np.array([[1],[2],[2],[3],[3],[4],[50]])\n"
        "model = IsolationForest(contamination=0.2, random_state=42)\n"
        "preds = model.fit_predict(X)\n"
        "print(f\"Predictions (1=normal, -1=anomaly): {preds}\")"
    ),
    "variationalAutoencoders": (
        "VAE Loss (NumPy)",
        "import numpy as np\n\n"
        "def vae_loss(x, x_recon, mu, logvar):\n"
        "    recon = np.mean((x - x_recon) ** 2)\n"
        "    kl = -0.5 * np.mean(1 + logvar - mu**2 - np.exp(logvar))\n"
        "    return recon + kl\n\n"
        "x = np.array([1, 0, 1, 0])\n"
        "print(f\"Loss: {vae_loss(x, x*0.9, np.zeros(4), np.ones(4)):.3f}\")"
    ),
    "generativeAdversarialNetworks": (
        "GAN Discriminator (Keras)",
        "from tensorflow.keras.models import Sequential\n"
        "from tensorflow.keras.layers import Dense\n\n"
        "discriminator = Sequential([\n"
        "    Dense(128, activation='relu', input_shape=(100,)),\n"
        "    Dense(1, activation='sigmoid')\n"
        "])\n"
        "discriminator.compile(optimizer='adam', loss='binary_crossentropy')\n"
        "print(discriminator.summary())"
    ),
    "recommendationSystems": (
        "Collaborative Filtering (SVD)",
        "from surprise import SVD, Dataset, Reader\n"
        "from surprise.model_selection import cross_validate\n\n"
        "data = Dataset.load_builtin('ml-100k')\n"
        "algo = SVD()\n"
        "results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)\n"
        "print(f\"Mean RMSE: {results['test_rmse'].mean():.3f}\")"
    ),
    "introductionToMl": (
        "Scikit-Learn Basic Pipeline",
        "from sklearn import datasets\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.ensemble import RandomForestClassifier\n\n"
        "iris = datasets.load_iris()\n"
        "X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)\n"
        "model = RandomForestClassifier(n_estimators=10, random_state=42)\n"
        "model.fit(X_train, y_train)\n"
        "print(f\"Test accuracy: {model.score(X_test, y_test):.2f}\")"
    ),
    "decisionTreeAlgorithm": (
        "Decision Tree from Scratch (NumPy)",
        "import numpy as np\n\n"
        "def entropy(y):\n"
        "    p = np.bincount(y) / len(y)\n"
        "    return -np.sum(p * np.log2(p + 1e-9))\n\n"
        "y = np.array([0, 0, 1, 1, 1])\n"
        "print(f\"Entropy: {entropy(y):.3f}\")"
    ),
}


def is_json_string_format(theory_line):
    """Check if theory uses JSON string format: "theory": "..." """
    stripped = theory_line.strip()
    return stripped.startswith('"theory": "') or stripped.startswith('"theory": "#')


def extract_json_string_content(line):
    """Extract content between "theory": "..." """
    match = re.search(r'"theory":\s*"(.*)"\s*,?\s*$', line, re.DOTALL)
    if match:
        return match.group(1)
    return None


def json_to_template_literal(theory_content, snippet_code):
    """Convert JSON string theory content to template literal format with snippet appended."""
    lines = theory_content.split('\\n')
    result_lines = []
    for l in lines:
        result_lines.append(l)
    snippet_indent = "  "
    result_lines.append("")
    result_lines.append("#### Python Implementation")
    result_lines.append("")
    result_lines.append("```python")
    for snippet_line in snippet_code.split('\n'):
        result_lines.append(snippet_line)
    result_lines.append("```")
    return '\n'.join(result_lines)


def extract_backtick_content(file_content):
    """Extract theory content from template literal format."""
    match = re.search(r'(?:^|\n)\s*theory:\s*`(.*?)`\s*,?\s*$', file_content, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(1)
    return None


def process_file(filename):
    filepath = os.path.join(MODULES_DIR, filename)
    with open(filepath, 'r') as f:
        content = f.read()

    basename = filename.replace('.ts', '')

    # Check if we have a snippet for this file
    if basename in SNIPPETS:
        topic, snippet_code = SNIPPETS[basename]
    elif basename.startswith('kaggle'):
        # Kaggle project files - skip
        return (filename, "skipped (Kaggle project)")
    else:
        return (filename, f"skipped (no snippet for {basename})")

    # Determine format
    # Find the theory field
    lines = content.split('\n')

    theory_start_idx = None
    theory_end_idx = None
    is_json = False
    is_backtick = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if '"theory":' in stripped or 'theory:' in stripped:
            # Hybrid format: "theory": `...` (JSON key + backtick value)
            if '"theory": `' in stripped:
                is_backtick = True
                theory_start_idx = i
                for j in range(i, len(lines)):
                    if ('`,' in lines[j] and j > i) or lines[j].strip() == '`,' or lines[j].strip() == '`':
                        theory_end_idx = j
                        break
                break
            # Pure JSON string format: "theory": "..."
            elif '"theory": "' in stripped or '"theory": "#' in stripped:
                is_json = True
                theory_start_idx = i
                if i < len(lines) - 1 and lines[i+1].strip().startswith('"interactiveSummary"'):
                    theory_end_idx = i
                break
            # Template literal format: theory: `...`
            elif 'theory: `' in stripped or 'theory: `#' in stripped:
                is_backtick = True
                theory_start_idx = i
                for j in range(i, len(lines)):
                    if ('`,' in lines[j] and j > i) or lines[j].strip() == '`,' or lines[j].strip() == '`':
                        theory_end_idx = j
                        break
                break
            elif 'theory:\n' in stripped or 'theory:\n' in content:
                is_backtick = True
                theory_start_idx = i
                for j in range(i, len(lines)):
                    if ('`,' in lines[j] and j > i) or lines[j].strip() == '`,' or lines[j].strip() == '`':
                        theory_end_idx = j
                        break
                break

    if theory_start_idx is None or theory_end_idx is None:
        return (filename, "ERROR: could not find theory field")

    # Check if snippet already exists (idempotency)
    if is_json:
        theory_line = lines[theory_start_idx]
        if 'Python Implementation' in theory_line:
            return (filename, "skipped (snippet already exists)")
    elif is_backtick:
        for j in range(theory_start_idx, theory_end_idx + 1):
            if 'Python Implementation' in lines[j]:
                return (filename, "skipped (snippet already exists)")

    if is_json:
        # Extract the theory line content
        theory_line = lines[theory_start_idx]
        match = re.search(r'"theory":\s*"(.*)"\s*,?\s*$', theory_line, re.DOTALL)
        if not match:
            return (filename, "ERROR: could not parse JSON theory string")
        theory_content = match.group(1)

        # Convert to template literal
        theory_parts = theory_content.split('\\n')

        # Find the Key Takeaways section
        snippet_indent = "  "
        new_theory_lines = []
        for l in theory_parts:
            new_theory_lines.append(l)

        new_theory_lines.append("")
        new_theory_lines.append("#### Python Implementation")
        new_theory_lines.append("")
        new_theory_lines.append("```python")
        for snippet_line in snippet_code.split('\n'):
            new_theory_lines.append(snippet_line)
        new_theory_lines.append("```")

        new_theory = '\n'.join(new_theory_lines)

        # Build the template literal line
        theory_literal_lines = new_theory.split('\n')
        indent = "  "
        result_lines_list = [f'{indent}theory: `{theory_literal_lines[0]}']
        for tl in theory_literal_lines[1:]:
            result_lines_list.append(f'{indent}{tl}')
        result_lines_list[-1] += '`,\n'

        # Replace the old theory line with the new multi-line version
        new_content = '\n'.join(
            lines[:theory_start_idx] +
            result_lines_list +
            lines[theory_start_idx+1:]
        )

        with open(filepath, 'w') as f:
            f.write(new_content)

        return (filename, f"converted JSON→template, added {topic} snippet")

    elif is_backtick:
        # Extract the theory content from backtick format
        theory_beginning = lines[theory_start_idx]

        # Detect the opening delimiter format
        if '"theory": `' in theory_beginning:
            theory_key = '"theory"'
            before_backtick, after_backtick = theory_beginning.split('"theory": `', 1)
            content_lines = [after_backtick]
            key_prefix = '"theory": `'
        elif 'theory: `' in theory_beginning:
            theory_key = 'theory'
            before_backtick, after_backtick = theory_beginning.split('theory: `', 1)
            content_lines = [after_backtick]
            key_prefix = 'theory: `'
        else:
            theory_key = 'theory'
            content_lines = []
            key_prefix = 'theory: `'

        for j in range(theory_start_idx + 1, theory_end_idx):
            content_lines.append(lines[j])

        last_line = lines[theory_end_idx]
        if '`,' in last_line:
            before_bq, _ = last_line.rsplit('`', 1)
            content_lines.append(before_bq.replace(',', ''))
        else:
            content_lines.append(last_line.replace('`', '').replace(',', ''))

        theory_content = '\n'.join(content_lines)

        new_theory = theory_content.rstrip() + "\n\n#### Python Implementation\n\n```python\n" + snippet_code + "\n```\n"

        new_theory_lines_list = new_theory.split('\n')

        indent = "  "
        result_lines_list = []
        result_lines_list.append(f'{indent}{theory_key}: `{new_theory_lines_list[0]}')
        for tl in new_theory_lines_list[1:]:
            result_lines_list.append(f'{indent}{tl}')
        result_lines_list[-1] += '`,\n'

        new_content = '\n'.join(
            lines[:theory_start_idx] +
            result_lines_list +
            lines[theory_end_idx+1:]
        )

        with open(filepath, 'w') as f:
            f.write(new_content)

        return (filename, f"added {topic} snippet to template literal")

    else:
        return (filename, "ERROR: unknown format")


def main():
    results = []
    all_files = sorted([f for f in os.listdir(MODULES_DIR) if f.endswith('.ts') and f != 'index.ts' and not f.endswith('.bak')])

    for filename in all_files:
        if filename.endswith('.bak'):
            continue
        result = process_file(filename)
        results.append(result)
        print(f"{result[0]}: {result[1]}")

    print("\n\n=== SUMMARY ===")
    modified = [r for r in results if not r[1].startswith("ERROR")]
    for r in results:
        print(f"  {r[0]}: {r[1]}")


if __name__ == '__main__':
    main()
