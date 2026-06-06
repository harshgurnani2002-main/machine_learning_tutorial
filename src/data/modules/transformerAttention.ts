import type { MLModule } from '../../types';

export const transformerAttention: MLModule = {
  "id": "transformer-attention",
  "title": "Transformer Self-Attention",
  "category": "Deep Learning",
  "description": "Link tokens in context using Query, Key, and Value projection matrix scores.",
  "formula": "Attention(Q,K,V) = Softmax(\frac{QK^T}{\\sqrt{d_k}}) V",
  "theory": "### Transformer Self-Attention\n\n#### What is it?\nSelf-Attention is the foundational mathematical mechanism behind the Transformer architecture (introduced in \"Attention Is All You Need\"). It allows a neural network model to weigh the relative importance of different words in a sequence when processing a specific target word, effectively enabling the model to capture deep contextual meaning and syntactic relationships regardless of the physical distance between words in the sequence.\n\n#### Why do we need it?\nBefore Transformers, Recurrent Neural Networks (RNNs) and LSTMs were the gold standard for Natural Language Processing. However, they process sequences sequentially (word-by-word), which makes them extremely slow to train and prone to losing context over long distances due to vanishing gradients. \nSelf-Attention solves this by processing all words in a sequence simultaneously (in parallel). This drastically speeds up training on modern GPUs and provides direct, short paths for gradients to flow across long distances, completely solving the long-term dependency problem.\n\n#### How does it work?\nIn Self-Attention, each word embedding in a sequence is projected into three distinct vectors using learned weight matrices: Query (Q), Key (K), and Value (V).\n- **Query (Q)**: Represents what the current word is looking for.\n- **Key (K)**: Represents what each word in the sequence has to offer.\n- **Value (V)**: Represents the actual content or meaning of each word.\n\nThe attention score between two words is calculated by taking the dot product of the Query of the target word and the Key of another word. High dot products mean the words are highly relevant to each other. These raw scores are scaled, passed through a Softmax function to turn them into probabilities (weights), and then multiplied by the Value vectors. The result is a weighted sum of the Values, representing the new context-aware embedding for the target word.\n\n#### The Math Behind It\nThe scaled dot-product attention equation is defined as:\n$$Attention(Q, K, V) = Softmax \\left( \frac{Q K^T}{\\sqrt{d_k}} \right) V$$\n\nwhere $Q, K, V$ are projected from the input sequence matrix $X$:\n$$Q = X W_Q, \\quad K = X W_K, \\quad V = X W_V$$\n\n**Scaled Dot Product**: We divide the raw dot product $Q K^T$ by $\\sqrt{d_k}$ (the square root of the key dimension). Why? At large values of dimension $d_k$, dot products grow very large in magnitude. This pushes the Softmax function into flat regions with vanishingly small gradients, effectively stopping learning. Scaling ensures the variance stays close to 1, providing stable gradients.\n\n#### Worked Example\nConsider the sentence: \"The bank of the river\". We want to compute the self-attention output for the word \"bank\".\n1. Generate $Q_{bank}$, and $K$, $V$ for all words using weight matrices.\n2. Compute dot products of $Q_{bank}$ with the keys of all words: $K_{The}, K_{bank}, K_{of}, K_{the}, K_{river}$.\n3. Because \"river\" provides crucial context (it's a riverbank, not a financial bank), the dot product $Q_{bank} \\cdot K_{river}$ will be high.\n4. Scale these scores by $\\sqrt{d_k}$ and apply Softmax to get attention weights that sum to 1 (e.g., 0.1 for \"The\", 0.8 for \"river\").\n5. Multiply these weights by the respective Value vectors and sum them up. The resulting vector for \"bank\" now heavily incorporates the embedding of \"river\", clarifying its contextual meaning.\n\n#### Common Pitfalls\n- **High Memory Complexity**: Standard self-attention requires comparing every word to every other word. This results in an $O(N^2)$ memory and computational complexity relative to sequence length $N$. It struggles with very long documents without specialized sparse attention mechanisms.\n- **Lack of Position Information**: Because the self-attention operation is entirely parallel and permutation invariant, it has no inherent notion of word order. **Positional Encodings** must be added to the input embeddings so the model knows the sequential order.\n\n#### When to Use vs Not Use\n- **Use When**: You are working with Natural Language Processing (NLP), translating text, summarizing documents, or building Large Language Models (LLMs) where deep contextual understanding is critical.\n- **Not Use When**: You are working on resource-constrained edge devices (due to the $O(N^2)$ footprint), dealing with extremely long sequences without chunking, or when the dataset is extremely small (Transformers lack the inductive bias of CNNs/RNNs and require massive amounts of data to train effectively).\n\n#### Key Takeaways\n1. Self-Attention replaces recurrence with parallelizable matrix multiplications.\n2. Queries, Keys, and Values determine how much focus each word places on every other word.\n3. Scaling by $\\sqrt{d_k}$ prevents vanishing gradients during Softmax computation.\n4. Multi-Head Attention is commonly used, which runs multiple self-attention mechanisms in parallel to capture different semantic aspects simultaneously.",
  "interactiveSummary": "This simulator renders a live attention heatmap grid where each cell (i, j) shows how strongly token i attends to token j after the softmax, with brighter colors indicating higher attention weights. You can type a custom sentence and watch the Q, K, and V projection matrices compute in real time before animating the dot-product scores across the sequence. Hovering over any cell in the heatmap reveals the raw pre-softmax logit alongside the scaled, normalised attention weight, making the temperature scaling effect by √d_k visually tangible. The multi-head toggle splits the heatmap into parallel head views, showing how different attention heads simultaneously learn syntactic dependencies, coreference, and semantic similarity between word pairs in the same sentence.",
  "simulatorId": "attention",
  "quiz": [
    {
      "id": "tf_q1",
      "question": "Why is the dot product scaled by 1/sqrt(d_k) in self-attention?",
      "options": [
        "To prevent dot products from growing too large, which flattens the Softmax gradient.",
        "To speed up matrix multiplication.",
        "To reduce the number of features.",
        "To normalize embeddings between 0 and 1."
      ],
      "correctAnswer": "To prevent dot products from growing too large, which flattens the Softmax gradient.",
      "explanation": "Large dimensions yield high dot products, which pushes Softmax into flat regions where derivatives approach zero. Scaling prevents this gradient flattening."
    },
    {
      "id": "tf_q2",
      "question": "In the self-attention mechanism, what is the role of the Query (Q) matrix?",
      "options": [
        "It represents what the current token is looking for in other tokens.",
        "It represents the inherent meaning of the token.",
        "It dictates the positional encoding.",
        "It acts as the final output embedding."
      ],
      "correctAnswer": "It represents what the current token is looking for in other tokens.",
      "explanation": "The Query vector is compared against all Key vectors to determine attention scores."
    },
    {
      "id": "tf_q3",
      "question": "What is the computational complexity of self-attention with respect to the sequence length N?",
      "options": [
        "O(N)",
        "O(N log N)",
        "O(N^2)",
        "O(1)"
      ],
      "correctAnswer": "O(N^2)",
      "explanation": "Self-attention computes a score for every token against every other token, resulting in an N x N attention matrix."
    },
    {
      "id": "tf_q4",
      "question": "Why do Transformers need Positional Encodings?",
      "options": [
        "Self-attention operations are permutation invariant and have no inherent sense of sequence order.",
        "To scale the attention scores.",
        "To convert text into numerical IDs.",
        "To reduce the vocabulary size."
      ],
      "correctAnswer": "Self-attention operations are permutation invariant and have no inherent sense of sequence order.",
      "explanation": "Since all tokens are processed simultaneously in parallel, positional encodings must be injected so the model understands the order of words."
    },
    {
      "id": "tf_q5",
      "question": "What is Multi-Head Attention?",
      "options": [
        "Running several self-attention operations in parallel and concatenating their results.",
        "An attention mechanism that attends to multiple documents at once.",
        "A method to compress the sequence length.",
        "Using multiple GPUs to train the model."
      ],
      "correctAnswer": "Running several self-attention operations in parallel and concatenating their results.",
      "explanation": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions."
    },
    {
      "id": "tf_q6",
      "question": "What does the Softmax function do in the attention formula?",
      "options": [
        "Converts raw dot-product scores into a probability distribution that sums to 1.",
        "Computes the dot product between Q and K.",
        "Extracts the most important feature.",
        "Reduces the dimensionality of the Value matrix."
      ],
      "correctAnswer": "Converts raw dot-product scores into a probability distribution that sums to 1.",
      "explanation": "Softmax creates attention weights that dictate how much focus should be placed on each corresponding Value vector."
    },
    {
      "id": "tf_q7",
      "question": "In an encoder-decoder Transformer architecture (like the original paper), what does the Cross-Attention mechanism do?",
      "options": [
        "The decoder queries the encoder\u2019s keys and values.",
        "The encoder queries the decoder\u2019s keys and values.",
        "It attends to tokens within the same sentence only.",
        "It translates languages directly."
      ],
      "correctAnswer": "The decoder queries the encoder\u2019s keys and values.",
      "explanation": "In cross-attention, the Queries come from the previous decoder layer, while the Keys and Values come from the encoder\u2019s output."
    },
    {
      "id": "tf_q8",
      "question": "What is the purpose of Masked Self-Attention in the Transformer Decoder?",
      "options": [
        "To prevent tokens from attending to future tokens in the sequence.",
        "To hide certain words for the model to guess (like BERT).",
        "To ignore padding tokens.",
        "To regularize the model by dropping out random attention weights."
      ],
      "correctAnswer": "To prevent tokens from attending to future tokens in the sequence.",
      "explanation": "During autoregressive generation, a token should only depend on past tokens. The mask sets future attention scores to negative infinity before the Softmax."
    },
    {
      "id": "tf_q9",
      "question": "If a sequence has 5 tokens, what is the shape of the resulting attention score matrix before multiplying by V?",
      "options": [
        "5 x 5",
        "5 x d_k",
        "1 x 5",
        "d_model x d_model"
      ],
      "correctAnswer": "5 x 5",
      "explanation": "The dot product of an (N x d_k) Query matrix and a (d_k x N) Key matrix results in an N x N matrix representing the attention scores between every token pair."
    },
    {
      "id": "tf_q10",
      "question": "Which component typically follows the self-attention layer in a Transformer block?",
      "options": [
        "A Feed-Forward Neural Network",
        "An LSTM cell",
        "A Convolutional layer",
        "A Max Pooling layer"
      ],
      "correctAnswer": "A Feed-Forward Neural Network",
      "explanation": "Each Transformer block typically consists of a Multi-Head Attention mechanism followed by a position-wise Feed-Forward Neural Network, both surrounded by residual connections and layer normalization."
    }
  ],
  "coding": {
    "tutorial": {
      "title": "Project QKV Matrices",
      "description": "Compute Query projection matrix Q = X * W_Q. While raw NumPy shows the inner mechanics, real-world NLP often uses specialized libraries. In scikit-learn, we can approximate text processing pipelines using TF-IDF or CountVectorizers, but for deep learning, we demonstrate the matrix math directly. Note: While we use NumPy here for educational intuition, production systems use PyTorch/TensorFlow or sklearn equivalents.",
      "pseudoCode": "function project_queries(X, W_q):\n    Q = matrix_multiply(X, W_q)\n    return Q",
      "starterCode": "import numpy as np\n\ndef project_queries(X, W_q):\n    # TODO: return dot product\n    return np.array([])\n\nX = np.array([[1.0, 0.5]])\nW_q = np.array([[0.2, 0.1], [0.5, 0.8]])\nprint(project_queries(X, W_q))",
      "expectedOutput": "[[0.45 0.5 ]]",
      "solution": "import numpy as np\n\ndef project_queries(X, W_q):\n    return np.dot(X, W_q)\n\nX = np.array([[1.0, 0.5]])\nW_q = np.array([[0.2, 0.1], [0.5, 0.8]])\nprint(project_queries(X, W_q))",
      "hints": [
        "Use np.dot to project inputs."
      ],
      "testKeywords": [
        "np.dot"
      ]
    },
    "project": {
      "title": "Sentence Similarity Scoring with Self-Attention",
      "description": "Implement a self-attention layer in PyTorch to produce context-aware token embeddings, then compute a sentence-level similarity score between two sentences by averaging their attended token representations and measuring cosine similarity. This mimics how models like Sentence-BERT produce sentence embeddings.",
      "pseudoCode": "# Sentence Similarity via Self-Attention\n# Step 1: Encode each sentence as a sequence of token embeddings\n# Step 2: Apply scaled dot-product self-attention to each sequence\n#   Q = X @ W_q;  K = X @ W_k;  V = X @ W_v\n#   scores = softmax( (Q @ K.T) / sqrt(d_k) )\n#   attended = scores @ V\n# Step 3: Mean-pool attended tokens -> sentence vector\n# Step 4: Cosine similarity between the two sentence vectors\n\ndef self_attention(X, W_q, W_k, W_v):\n    d_k = W_q.shape[1]\n    Q = matmul(X, W_q)\n    K = matmul(X, W_k)\n    V = matmul(X, W_v)\n    scores = softmax( (Q @ K.T) / sqrt(d_k), dim=-1 )\n    return scores @ V\n\nsim = cosine_similarity(mean(attn_A), mean(attn_B))\nprint(f'Similarity: {sim:.2f}')",
      "starterCode": "import torch\nimport torch.nn.functional as F\n\ntorch.manual_seed(42)\n# Two sentences encoded as token embeddings (3 tokens each, embed_dim=4)\n# Sentence A: \"The cat sat\"  (similar meaning to B)\n# Sentence B: \"A cat rested\"\nsentence_A = torch.tensor([\n    [0.9, 0.1, 0.8, 0.2],  # The\n    [0.8, 0.9, 0.7, 0.8],  # cat\n    [0.6, 0.3, 0.7, 0.5],  # sat\n], dtype=torch.float32)\nsentence_B = torch.tensor([\n    [0.2, 0.8, 0.3, 0.9],  # A\n    [0.8, 0.9, 0.7, 0.8],  # cat  (same embedding as in A)\n    [0.5, 0.4, 0.6, 0.6],  # rested\n], dtype=torch.float32)\n\n# Shared projection matrices\nW_q = torch.randn(4, 4)\nW_k = torch.randn(4, 4)\nW_v = torch.randn(4, 4)\n\n# TODO: Implement self_attention(X, W_q, W_k, W_v)\n# Returns attended output of same shape as X\ndef self_attention(X, W_q, W_k, W_v):\n    d_k = W_q.shape[1]\n    # TODO: Compute Q, K, V using torch.matmul\n    Q, K, V = None, None, None\n    # TODO: Compute scaled dot-product attention scores\n    scores = None\n    # TODO: Multiply scores by V\n    return None\n\n# TODO: Apply self_attention to both sentences\nattn_A = self_attention(sentence_A, W_q, W_k, W_v)\nattn_B = self_attention(sentence_B, W_q, W_k, W_v)\n\n# TODO: Mean-pool to get sentence vectors, compute cosine similarity\nvec_A = None  # shape (4,)\nvec_B = None  # shape (4,)\nsimilarity = F.cosine_similarity(vec_A.unsqueeze(0), vec_B.unsqueeze(0)).item()\nprint(f\"Sentence Similarity: {similarity:.2f}\")",
      "expectedOutput": "Sentence Similarity: 1.00",
      "solution": "import torch\nimport torch.nn.functional as F\n\ntorch.manual_seed(42)\nsentence_A = torch.tensor([\n    [0.9, 0.1, 0.8, 0.2],\n    [0.8, 0.9, 0.7, 0.8],\n    [0.6, 0.3, 0.7, 0.5],\n], dtype=torch.float32)\nsentence_B = torch.tensor([\n    [0.2, 0.8, 0.3, 0.9],\n    [0.8, 0.9, 0.7, 0.8],\n    [0.5, 0.4, 0.6, 0.6],\n], dtype=torch.float32)\n\nW_q = torch.randn(4, 4)\nW_k = torch.randn(4, 4)\nW_v = torch.randn(4, 4)\n\ndef self_attention(X, W_q, W_k, W_v):\n    d_k = float(W_q.shape[1])\n    Q = torch.matmul(X, W_q)\n    K = torch.matmul(X, W_k)\n    V = torch.matmul(X, W_v)\n    scores = F.softmax(torch.matmul(Q, K.transpose(0, 1)) / (d_k ** 0.5), dim=-1)\n    return torch.matmul(scores, V)\n\nattn_A = self_attention(sentence_A, W_q, W_k, W_v)\nattn_B = self_attention(sentence_B, W_q, W_k, W_v)\n\nvec_A = attn_A.mean(dim=0)\nvec_B = attn_B.mean(dim=0)\nsimilarity = F.cosine_similarity(vec_A.unsqueeze(0), vec_B.unsqueeze(0)).item()\nprint(f\"Sentence Similarity: {similarity:.2f}\")",
      "hints": [
        "Q = torch.matmul(X, W_q)",
        "scores = F.softmax(torch.matmul(Q, K.transpose(0,1)) / d_k**0.5, dim=-1)",
        "Mean-pool: attn_A.mean(dim=0)",
        "F.cosine_similarity needs unsqueeze(0) to add batch dim"
      ],
      "testKeywords": [
        "torch.matmul",
        "F.softmax",
        "F.cosine_similarity",
        "self_attention"
      ]
    },
    "assignment": {
      "title": "Named Entity Recognition with PyTorch MultiheadAttention",
      "description": "Use PyTorch's nn.MultiheadAttention to build a token-level NER tagger. Each token embedding is attended to in context and then classified into one of 3 entity categories: Other (0), Person (1), or Location (2). The model applies multi-head self-attention followed by a linear classifier on every token position.",
      "pseudoCode": "# NER with Multi-Head Attention\nclass NERAttentionTagger(nn.Module):\n    __init__:\n        attention = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)\n        classifier = nn.Linear(8, 3)   # 3 NER classes: Other, Person, Location\n    forward(x):\n        attn_out, _ = attention(x, x, x)  # self-attention: Q=K=V=x\n        return classifier(attn_out)        # classify every token position\n\nmodel = NERAttentionTagger()\ncriterion = CrossEntropyLoss()\noptimizer = Adam(model.parameters(), lr=0.01)\nfor epoch in range(200):\n    optimizer.zero_grad()\n    logits = model(X)                     # (batch, seq, 3)\n    loss   = criterion(logits.view(-1,3), y.view(-1))\n    loss.backward(); optimizer.step()\nprint predicted entity tags",
      "starterCode": "import torch\nimport torch.nn as nn\nimport torch.optim as optim\n\n# Token embeddings for sentences (batch=4, seq_len=3 tokens, embed_dim=8)\n# y: NER label for each token -> 0=Other, 1=Person, 2=Location\ntorch.manual_seed(99)\nX = torch.randn(4, 3, 8)\ny = torch.tensor([\n    [1, 0, 2],   # [Alice, visited, Paris]  -> Person, Other, Location\n    [0, 1, 0],   # [The, doctor, left]       -> Other, Person, Other\n    [2, 0, 1],   # [London, welcomed, Emma]  -> Location, Other, Person\n    [1, 2, 0],   # [Bob, London, arrived]    -> Person, Location, Other\n])\n\n# TODO: Define NERAttentionTagger\n# Use nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)\n# and nn.Linear(8, 3) as the per-token classifier\nclass NERAttentionTagger(nn.Module):\n    def __init__(self):\n        super().__init__()\n        # TODO: define attention and classifier\n        pass\n    def forward(self, x):\n        # TODO: apply self-attention (Q=K=V=x), then classify each token\n        pass\n\nmodel = NERAttentionTagger()\n# TODO: CrossEntropyLoss and Adam(lr=0.01). Train 200 epochs.\n# Note: flatten logits and labels for loss: logits.view(-1,3), y.view(-1)\n\nwith torch.no_grad():\n    test = torch.randn(1, 3, 8)\n    out = model(test)\n    tags = out.argmax(dim=-1)[0].tolist()\n    print(f\"Predicted tags: {tags}\")",
      "expectedOutput": "Predicted tags: [1, 0, 2]",
      "solution": "import torch\nimport torch.nn as nn\nimport torch.optim as optim\n\ntorch.manual_seed(99)\nX = torch.randn(4, 3, 8)\ny = torch.tensor([\n    [1, 0, 2],\n    [0, 1, 0],\n    [2, 0, 1],\n    [1, 2, 0],\n])\n\nclass NERAttentionTagger(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.attention = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)\n        self.classifier = nn.Linear(8, 3)\n\n    def forward(self, x):\n        attn_out, _ = self.attention(x, x, x)\n        return self.classifier(attn_out)\n\nmodel = NERAttentionTagger()\ncriterion = nn.CrossEntropyLoss()\noptimizer = optim.Adam(model.parameters(), lr=0.01)\n\nfor epoch in range(200):\n    optimizer.zero_grad()\n    logits = model(X)\n    loss = criterion(logits.view(-1, 3), y.view(-1))\n    loss.backward()\n    optimizer.step()\n\nwith torch.no_grad():\n    preds = model(X).argmax(dim=-1)\n    correct = (preds == y).float().mean().item()\n    print(f\"Token Accuracy: {correct:.2f}\")",
      "hints": [
        "nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)",
        "Self-attention: attn_out, _ = self.attention(x, x, x)",
        "Flatten for loss: logits.view(-1, 3) and y.view(-1)"
      ],
      "testKeywords": [
        "nn.MultiheadAttention",
        "NERAttentionTagger",
        "CrossEntropyLoss",
        "backward"
      ]
    }
  },
  "interviewQuestions": [
    {
      "question": "Explain the difference between Self-Attention and Multi-Head Attention.",
      "answer": "Self-attention computes attention scores once across the entire hidden dimension. Multi-head attention splits queries, keys, and values into multiple subsets, calculates attention for each head in parallel, and concatenates the outputs. This allows the model to attend to information from different representation subspaces simultaneously.",
      "companyTags": [
        "Meta",
        "Google"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "Why does the Transformer require Positional Encodings?",
      "answer": "The self-attention mechanism processes all tokens simultaneously and does not inherently process sequential order like an RNN. Positional encodings are added to the input embeddings to provide information about the relative or absolute position of the tokens in the sequence.",
      "companyTags": [
        "Amazon",
        "Apple"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "What is the purpose of scaling by the square root of d_k in the attention mechanism?",
      "answer": "When the dimension of keys (d_k) is large, the dot products of queries and keys can become very large. This pushes the softmax function into regions where gradients are extremely small (vanishing gradient). Scaling down by sqrt(d_k) ensures the variance of the dot products remains roughly 1, keeping gradients stable.",
      "companyTags": [
        "Google",
        "Microsoft"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "What is Masked Self-Attention?",
      "answer": "Masked self-attention is used in the Transformer decoder. It ensures that the prediction for a specific position can only depend on known outputs at positions before it, preserving the autoregressive property by setting future attention weights to negative infinity.",
      "companyTags": [
        "Netflix",
        "Meta"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "How is the computational complexity of self-attention related to sequence length?",
      "answer": "The complexity is O(N^2) where N is the sequence length. This is because every token must compute an attention score with every other token in the sequence.",
      "companyTags": [
        "Google",
        "Tesla"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "What are Queries, Keys, and Values in the context of attention?",
      "answer": "They are abstractions borrowed from retrieval systems. The Query is the vector representing the current token looking for context. The Key represents what other tokens contain. The Value is the actual content of the token. The dot product of Query and Key determines how much of the Value is incorporated.",
      "companyTags": [
        "Amazon",
        "Apple"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "Describe the Feed-Forward Network component in a Transformer block.",
      "answer": "After the multi-head attention mechanism, the output passes through a position-wise feed-forward network. It consists of two linear transformations with a ReLU (or GELU) activation in between, applied independently and identically to each position.",
      "companyTags": [
        "Microsoft"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "What is the role of Layer Normalization in Transformers?",
      "answer": "Layer normalization stabilizes the training process by normalizing the inputs across the features for each token independently. It is placed before or after the sub-layers (attention and feed-forward) along with residual connections.",
      "companyTags": [
        "Meta"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "How do residual connections help in training deep Transformers?",
      "answer": "Residual connections add the input of a sub-layer to its output (e.g., Output = Sublayer(x) + x). This creates direct paths for gradients to flow backwards, preventing the vanishing gradient problem in very deep networks.",
      "companyTags": [
        "Google",
        "Netflix"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "What is Cross-Attention?",
      "answer": "Cross-attention is found in encoder-decoder Transformers. The Queries come from the previous decoder layer, while the Keys and Values are the output of the final encoder layer. This allows the decoder to attend to the encoded input sequence.",
      "companyTags": [
        "Amazon"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "Why might a Transformer fail on very long documents?",
      "answer": "Due to the O(N^2) memory and time complexity of self-attention, processing very long documents requires massive amounts of memory, often exceeding standard GPU limits. Specialized models like Longformer or sparse attention are needed.",
      "companyTags": [
        "Apple",
        "Meta"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "Can Transformers be used for images? How?",
      "answer": "Yes, via Vision Transformers (ViTs). An image is split into fixed-size patches, linearly embedded, and treated as a sequence of tokens. Positional embeddings are added, and the standard Transformer encoder processes them.",
      "companyTags": [
        "Google",
        "Tesla"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "What is the difference between BERT and GPT architectures?",
      "answer": "BERT uses an encoder-only architecture with bidirectional self-attention and is trained using masked language modeling. GPT uses a decoder-only architecture with masked (unidirectional) self-attention and is trained autoregressively to predict the next word.",
      "companyTags": [
        "Microsoft",
        "OpenAI"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "What is the \"Context Window\" of a large language model?",
      "answer": "The context window is the maximum sequence length (in tokens) that the Transformer can process in a single forward pass, bounded by the positional encodings and the memory required for the O(N^2) attention matrix.",
      "companyTags": [
        "Netflix",
        "Meta"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "How do Transformers solve the vanishing gradient problem of RNNs?",
      "answer": "Instead of recurrent connections that multiply weights over time steps, Transformers use self-attention, providing direct, non-sequential paths between any two words in a sequence, allowing gradients to flow easily.",
      "companyTags": [
        "Google"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "What are the dimensions of the Q, K, and V matrices?",
      "answer": "Typically, the input embedding of dimension d_model is multiplied by weight matrices of shape (d_model, d_k) for Q and K, and (d_model, d_v) for V. In multi-head attention, d_model is usually split equally among the heads.",
      "companyTags": [
        "Amazon"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "What activation function is typically used in the Feed-Forward layer of a Transformer?",
      "answer": "The original Transformer used ReLU. Modern variants like BERT and GPT use GELU (Gaussian Error Linear Unit) or SwiGLU, which provide smoother gradients.",
      "companyTags": [
        "Apple",
        "Microsoft"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "Why are sub-word tokenizers (like BPE or WordPiece) used with Transformers?",
      "answer": "Sub-word tokenizers strike a balance between character-level and word-level representations. They handle out-of-vocabulary words effectively by breaking them into known subwords, keeping the vocabulary size manageable.",
      "companyTags": [
        "Meta",
        "Google"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "What is the purpose of the [CLS] token in models like BERT?",
      "answer": "The [CLS] token is prepended to the input sequence. After passing through the Transformer layers, the final hidden state of the [CLS] token is used as the aggregate sequence representation for classification tasks.",
      "companyTags": [
        "Microsoft",
        "Amazon"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "Explain the concept of \"Attention Score\".",
      "answer": "The attention score is the raw dot product between a Query and a Key vector, representing how relevant the Key token is to the Query token. After scaling and softmax, it becomes an attention weight.",
      "companyTags": [
        "Apple"
      ],
      "difficulty": "Beginner"
    }
  ]
};
