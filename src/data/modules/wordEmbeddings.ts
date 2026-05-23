import type { MLModule } from '../../types';

export const wordEmbeddings: MLModule = {
  "id": "word-embeddings",
  "title": "Word Embeddings & Word2Vec",
  "category": "Deep Learning",
  "description": "Learn dense semantic vector coordinates for words using local contexts.",
  "formula": "cos(\\theta) = \\frac{u \\cdot v}{\\|u\\| \\|v\\|}",
  "theory": "### Word Embeddings & Word2Vec\n\n#### What is it?\nWord embeddings are dense, continuous, low-dimensional vector representations of words. Instead of representing words as sparse, high-dimensional one-hot encoded vectors where all words are equidistant, word embeddings place words with similar meanings close together in a mathematical vector space. Algorithms like Word2Vec (Continuous Bag-of-Words and Skip-Gram) are the foundational methods for generating these vectors by learning from large text corpora.\n\n#### Why do we need it?\nBefore word embeddings, Natural Language Processing (NLP) relied on Bag-of-Words or TF-IDF, treating words as isolated, discrete symbols. The word \"king\" had no mathematical relationship to \"queen\" or \"man\". This resulted in sparse, massive matrices that suffered from the \"curse of dimensionality\" and could not capture semantics or synonyms. Word embeddings solve this by mapping words to dense vectors (e.g., 300 dimensions) where semantic meaning is explicitly captured by the spatial relationships and distances between vectors.\n\n#### How does it work?\nWord2Vec is a predictive model that learns embeddings by trying to predict a word given its context, or context given a word. \nIt has two architectures:\n1. **Continuous Bag of Words (CBOW)**: Predicts the target word based on its surrounding context words.\n2. **Skip-gram**: Predicts the surrounding context words given a single target word.\n\nBy training a shallow neural network on this fake task over billions of words, the weights of the hidden layer become the word embeddings. Because words that appear in similar contexts tend to have similar meanings (the *distributional hypothesis*), their resulting embeddings will be geometrically close.\n\n#### The Math Behind It\nIn the Skip-gram model, given a target word $w_c$, we want to maximize the probability of observing its context words $w_o$. The probability is calculated using the softmax function over the dot product of their vectors:\n$$P(w_o | w_c) = \\frac{\\exp(v_{w_o}^T \\cdot v_{w_c})}{\\sum_{w=1}^V \\exp(v_w^T \\cdot v_{w_c})}$$\n\nWhere:\n- $v_{w_c}$ is the vector representation of the center word.\n- $v_{w_o}$ is the vector representation of the context word.\n- $V$ is the total vocabulary size.\n\n**Negative Sampling**: Calculating the denominator over the entire vocabulary $V$ for every step is computationally impossible for large datasets. Negative sampling solves this by approximating the softmax. Instead of updating all vectors, we only update the true context word and a small number ($k$) of randomly chosen \"negative\" words that do not appear in the context. We maximize the dot product for the true pair and minimize it for the negative pairs using the sigmoid function $\\sigma$:\n$$\\log \\sigma(v_{w_o}^T \\cdot v_{w_c}) + \\sum_{i=1}^k \\mathbb{E}_{w_i \\sim P_n(w)} [ \\log \\sigma(-v_{w_i}^T \\cdot v_{w_c}) ]$$\n\n#### Worked Example\nConsider the sentence: \"The quick brown fox jumps over the lazy dog.\"\nIf we use the Skip-gram model with a window size of 2, and the target word is \"fox\":\nThe context words are: [\"quick\", \"brown\", \"jumps\", \"over\"].\nThe network takes \"fox\" as input and tries to output high probabilities for those context words. As the model trains, the weights for \"fox\" are adjusted so its vector aligns closely with words that share similar contexts, like \"wolf\" or \"hound\".\n\nThis leads to the famous vector arithmetic property:\n$$Vector(\"King\") - Vector(\"Man\") + Vector(\"Woman\") \\approx Vector(\"Queen\")$$\n\n#### Common Pitfalls\n- **Out of Vocabulary (OOV)**: Traditional Word2Vec models assign a distinct vector to each exact word. If a word was not seen during training, it cannot be processed. FastText solves this by using subword n-grams.\n- **Polysemy**: Word2Vec assigns a single, static vector to a word. The word \"bank\" has the same vector whether it means a riverbank or a financial institution. Contextual embeddings like ELMo and BERT were developed to solve this.\n\n#### When to Use vs Not Use\n- **Use When**: You are building traditional NLP models (like sentiment analysis with LSTMs), calculating document similarity, or need lightweight features for text classification without the heavy compute of Transformers.\n- **Not Use When**: You need to capture complex context where words have multiple meanings (polysemy), or when you are building state-of-the-art language models, in which case you should use contextual embeddings directly trained via Transformer architectures (like BERT or GPT).\n\n#### Key Takeaways\n1. Word embeddings are dense vectors where geometric distance represents semantic similarity.\n2. Word2Vec uses the distributional hypothesis: words appearing in similar contexts have similar meanings.\n3. Skip-gram predicts context from a word; CBOW predicts a word from its context.\n4. Negative Sampling makes training on large vocabularies computationally feasible.\n5. They are static representations, meaning they cannot capture multiple meanings of the same word.",
  "interactiveSummary": "Explore the 3D space of learned word embeddings. You can search for words to see their nearest neighbors based on cosine similarity, and test vector arithmetic (like King - Man + Woman).",
  "simulatorId": "word-embeddings",
  "quiz": [
    {
      "id": "w2v_q1",
      "question": "Which Word2Vec architecture predicts context words given a target input word?",
      "options": [
        "Skip-Gram",
        "CBOW",
        "Transformer",
        "PCA"
      ],
      "correctAnswer": "Skip-Gram",
      "explanation": "Skip-gram inputs a single target word and outputs probabilities for surrounding context words."
    },
    {
      "id": "w2v_q2",
      "question": "Which Word2Vec architecture predicts a target word given its surrounding context words?",
      "options": [
        "Skip-Gram",
        "CBOW",
        "LSTM",
        "Autoencoder"
      ],
      "correctAnswer": "CBOW",
      "explanation": "Continuous Bag of Words (CBOW) takes a window of surrounding words and tries to guess the word in the middle."
    },
    {
      "id": "w2v_q3",
      "question": "What is the main limitation of standard Word2Vec embeddings regarding polysemy?",
      "options": [
        "It assigns only one vector per word, ignoring multiple meanings.",
        "The vectors are too high-dimensional.",
        "It cannot compute cosine similarity.",
        "It requires too much memory."
      ],
      "correctAnswer": "It assigns only one vector per word, ignoring multiple meanings.",
      "explanation": "Standard Word2Vec creates a single static embedding. \"Bank\" gets one vector, mixing the meanings of river bank and financial bank."
    },
    {
      "id": "w2v_q4",
      "question": "What technique is commonly used to speed up Word2Vec training by avoiding the calculation of the full softmax denominator?",
      "options": [
        "Negative Sampling",
        "Dropout",
        "Batch Normalization",
        "Max Pooling"
      ],
      "correctAnswer": "Negative Sampling",
      "explanation": "Negative sampling converts the problem into binary classification: distinguishing a true context word from a few randomly sampled noise words."
    },
    {
      "id": "w2v_q5",
      "question": "How is the similarity between two word embeddings usually measured?",
      "options": [
        "Cosine Similarity",
        "Manhattan Distance",
        "Jaccard Index",
        "Euclidean Distance"
      ],
      "correctAnswer": "Cosine Similarity",
      "explanation": "Cosine similarity measures the angle between vectors, which is robust to differences in vector magnitude."
    },
    {
      "id": "w2v_q6",
      "question": "What is the classic example of vector arithmetic learned by Word2Vec?",
      "options": [
        "Vector(\"King\") - Vector(\"Man\") + Vector(\"Woman\") \u2248 Vector(\"Queen\")",
        "Vector(\"Dog\") + Vector(\"Cat\") \u2248 Vector(\"Pet\")",
        "Vector(\"Hot\") * Vector(\"Cold\") \u2248 Vector(\"Warm\")",
        "Vector(\"Paris\") - Vector(\"London\") \u2248 Vector(\"France\")"
      ],
      "correctAnswer": "Vector(\"King\") - Vector(\"Man\") + Vector(\"Woman\") \u2248 Vector(\"Queen\")",
      "explanation": "Word embeddings capture relational semantics, allowing algebraic operations to yield logical semantic results."
    },
    {
      "id": "w2v_q7",
      "question": "Why are one-hot encoded vectors considered inefficient for NLP?",
      "options": [
        "They are highly sparse and capture no semantic meaning between words.",
        "They contain negative values which disrupt neural networks.",
        "They are too dense and computationally heavy.",
        "They require complex matrix multiplications."
      ],
      "correctAnswer": "They are highly sparse and capture no semantic meaning between words.",
      "explanation": "A one-hot vector is mostly zeros, and the dot product between any two distinct one-hot vectors is exactly zero, showing zero similarity."
    },
    {
      "id": "w2v_q8",
      "question": "How does FastText improve upon standard Word2Vec?",
      "options": [
        "It represents words as bags of character n-grams, allowing it to handle out-of-vocabulary words.",
        "It uses deep recurrent neural networks instead of a shallow layer.",
        "It applies self-attention mechanisms.",
        "It only trains on punctuation marks."
      ],
      "correctAnswer": "It represents words as bags of character n-grams, allowing it to handle out-of-vocabulary words.",
      "explanation": "By breaking words into sub-words (n-grams), FastText can construct embeddings for words it has never seen before based on their constituent parts."
    },
    {
      "id": "w2v_q9",
      "question": "What is the \"context window\" in Word2Vec?",
      "options": [
        "The number of words to the left and right of the target word considered during training.",
        "The maximum length of a document.",
        "The size of the vocabulary.",
        "The dimensionality of the embedding vector."
      ],
      "correctAnswer": "The number of words to the left and right of the target word considered during training.",
      "explanation": "The context window defines what is considered \"local context\". A window of 2 means two words before and two words after the target."
    },
    {
      "id": "w2v_q10",
      "question": "What type of neural network is used to train Word2Vec embeddings?",
      "options": [
        "A shallow, single-hidden-layer feedforward network.",
        "A deep Convolutional Neural Network (CNN).",
        "A deep Recurrent Neural Network (RNN).",
        "A Generative Adversarial Network (GAN)."
      ],
      "correctAnswer": "A shallow, single-hidden-layer feedforward network.",
      "explanation": "Word2Vec uses a very simple, shallow network. It does not use deep layers or non-linear activation functions in the hidden layer."
    }
  ],
  "coding": {
    "tutorial": {
      "title": "Cosine Similarity Calculator",
      "description": "Calculate cosine similarity between word vectors u and v. While raw NumPy illustrates the math, real applications use sklearn.metrics.pairwise.cosine_similarity.",
      "pseudoCode": "function cosine_sim(u, v):\n    dot_product = u * v\n    magnitude_u = length(u)\n    magnitude_v = length(v)\n    return dot_product / (magnitude_u * magnitude_v)",
      "starterCode": "import numpy as np\n\ndef cosine_sim(u, v):\n    # TODO: return dot(u,v) / (norm(u)*norm(v))\n    return 0.0\n\nu = np.array([1.0, 0.0])\nv = np.array([1.0, 1.0])\nprint(np.round(cosine_sim(u, v), 3))",
      "expectedOutput": "0.707",
      "solution": "import numpy as np\n\ndef cosine_sim(u, v):\n    norm_u = np.linalg.norm(u)\n    norm_v = np.linalg.norm(v)\n    if norm_u == 0 or norm_v == 0:\n        return 0.0\n    return np.dot(u, v) / (norm_u * norm_v)\n\nu = np.array([1.0, 0.0])\nv = np.array([1.0, 1.0])\nprint(np.round(cosine_sim(u, v), 3))",
      "hints": [
        "Use np.linalg.norm to calculate vector magnitudes."
      ],
      "testKeywords": [
        "np.dot",
        "np.linalg.norm"
      ]
    },
    "project": {
      "title": "Training Word Embeddings with PyTorch",
      "description": "Train a simple Skip-Gram style Word2Vec embedding layer using PyTorch nn.Embedding.",
      "starterCode": "import torch\nimport torch.nn as nn\nimport torch.optim as optim\n\n# Vocabulary: 0: apple, 1: orange, 2: fruit, 3: car, 4: truck\n# Pairs (target, context)\ndata = [(0, 2), (1, 2), (3, 4), (4, 3)]\ntargets = torch.tensor([p[0] for p in data])\ncontexts = torch.tensor([p[1] for p in data])\n\nclass SimpleWord2Vec(nn.Module):\n    def __init__(self, vocab_size, embed_dim):\n        super().__init__()\n        # TODO: Initialize nn.Embedding for target and context\n        self.target_emb = None\n        self.context_emb = None\n\n    def forward(self, target, context):\n        # TODO: get embeddings, compute dot product\n        # Return torch.sum(t_e * c_e, dim=1)\n        pass\n\n# TODO: Train the model using BCEWithLogitsLoss\n# Treat all provided pairs as positive examples (label=1)\n",
      "expectedOutput": "Training completed. Check similarities manually.",
      "solution": "import torch\nimport torch.nn as nn\nimport torch.optim as optim\n\ndata = [(0, 2), (1, 2), (3, 4), (4, 3)]\ntargets = torch.tensor([p[0] for p in data])\ncontexts = torch.tensor([p[1] for p in data])\n\nclass SimpleWord2Vec(nn.Module):\n    def __init__(self, vocab_size, embed_dim):\n        super().__init__()\n        self.target_emb = nn.Embedding(vocab_size, embed_dim)\n        self.context_emb = nn.Embedding(vocab_size, embed_dim)\n\n    def forward(self, target, context):\n        t_e = self.target_emb(target)\n        c_e = self.context_emb(context)\n        return torch.sum(t_e * c_e, dim=1)\n\nmodel = SimpleWord2Vec(5, 2)\noptimizer = optim.Adam(model.parameters(), lr=0.1)\ncriterion = nn.BCEWithLogitsLoss()\nlabels = torch.ones(len(targets))\n\nfor epoch in range(100):\n    optimizer.zero_grad()\n    outputs = model(targets, contexts)\n    loss = criterion(outputs, labels)\n    loss.backward()\n    optimizer.step()\n\nprint(\"Training completed. Check similarities manually.\")\n",
      "hints": [
        "Use `nn.Embedding(vocab_size, embed_dim)`"
      ],
      "testKeywords": [
        "nn.Embedding",
        "BCEWithLogitsLoss"
      ]
    },
    "assignment": {
      "title": "Computing Cosine Similarity in PyTorch",
      "description": "Given a set of pre-trained embeddings, compute the cosine similarity between word vectors to find the most similar words.",
      "starterCode": "import torch\nimport torch.nn.functional as F\n\n# Mock embeddings for [\"king\", \"queen\", \"man\", \"woman\", \"apple\"]\nembeddings = torch.tensor([\n    [1.0, 0.0, 0.5],  # king\n    [0.9, 0.1, 0.6],  # queen\n    [1.0, -0.5, 0.0], # man\n    [0.9, -0.4, 0.1], # woman\n    [-0.5, 0.8, -0.2] # apple\n])\n\n# We want to find the similarity between \"king\" (index 0) and all other words\ntarget_vector = embeddings[0].unsqueeze(0)\n\n# TODO: Compute cosine similarity using F.cosine_similarity\n# hint: pass target_vector and embeddings\nsimilarities = None\n\nprint(\"Similarities:\", torch.round(similarities, decimals=2))\n",
      "expectedOutput": "Similarities: tensor([1.0000, 0.9800, 0.8900, 0.8500, -0.6600])",
      "solution": "import torch\nimport torch.nn.functional as F\n\nembeddings = torch.tensor([\n    [1.0, 0.0, 0.5], \n    [0.9, 0.1, 0.6], \n    [1.0, -0.5, 0.0], \n    [0.9, -0.4, 0.1], \n    [-0.5, 0.8, -0.2] \n])\n\ntarget_vector = embeddings[0].unsqueeze(0)\nsimilarities = F.cosine_similarity(target_vector, embeddings)\n\nprint(\"Similarities:\", torch.round(similarities, decimals=2))\n",
      "hints": [
        "Use `F.cosine_similarity(target_vector, embeddings)`"
      ],
      "testKeywords": [
        "F.cosine_similarity"
      ]
    }
  },
  "interviewQuestions": [
    {
      "question": "Why is standard softmax calculation inefficient in Word2Vec, and how do Hierarchical Softmax or Negative Sampling resolve this?",
      "answer": "Standard Softmax requires calculating denominator exponents across the entire vocabulary size |V| (often 100k+ words), which is computationally expensive. Hierarchical Softmax replaces this with a binary Huffman tree, reducing calculations to O(log |V|). Negative sampling simplifies this to a binary logistic regression task, updating only the target word and a few randomly selected negative words (O(k)).",
      "companyTags": [
        "Google",
        "Meta"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "What is the distributional hypothesis in NLP?",
      "answer": "It is the linguistic theory that words occurring in similar contexts tend to have similar meanings. It summarizes as \"You shall know a word by the company it keeps.\" This is the foundational principle behind Word2Vec.",
      "companyTags": [
        "Amazon",
        "Microsoft"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "Explain the architecture of the Skip-Gram model.",
      "answer": "Skip-Gram uses a target word to predict the surrounding context words within a specified window. It consists of an input layer (one-hot encoded target word), a projection (hidden) layer, and an output layer predicting probabilities for each vocabulary word being in the context.",
      "companyTags": [
        "Apple",
        "Netflix"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "Explain the architecture of the CBOW model.",
      "answer": "Continuous Bag of Words (CBOW) takes a context of surrounding words, averages or sums their embedding vectors in the hidden layer, and attempts to predict the target (center) word.",
      "companyTags": [
        "Google"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "When would you prefer Skip-Gram over CBOW?",
      "answer": "Skip-Gram generally performs better with small datasets and handles rare words well because it treats each context-target pair as a new observation. CBOW is faster to train and has slightly better accuracy for frequent words.",
      "companyTags": [
        "Meta",
        "Amazon"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "What is Subsampling of Frequent Words in Word2Vec?",
      "answer": "Extremely common words like \"the\" or \"is\" provide little semantic value. Subsampling randomly discards training words with a probability inversely proportional to their frequency, speeding up training and improving the representations of less frequent words.",
      "companyTags": [
        "Google"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "How do GloVe embeddings differ from Word2Vec?",
      "answer": "Word2Vec is a predictive model that learns vectors to improve its local context predictions. GloVe (Global Vectors) is a count-based model that performs dimensionality reduction on the global word co-occurrence matrix, capturing global corpus statistics directly.",
      "companyTags": [
        "Microsoft",
        "Apple"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "What are the dimensions of a Word2Vec hidden layer?",
      "answer": "The hidden layer dimension is equal to the desired embedding size (e.g., 100 to 300). There is no non-linear activation function in this hidden layer; it acts as a simple linear projection.",
      "companyTags": [
        "Tesla"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "How does FastText handle Out-Of-Vocabulary (OOV) words?",
      "answer": "FastText breaks words down into character n-grams (e.g., \"apple\" -> \"<ap\", \"app\", \"ppl\", \"ple\", \"le>\"). The word vector is the sum of its n-gram vectors. An OOV word can still be embedded by summing the vectors of its constituent n-grams.",
      "companyTags": [
        "Meta",
        "Netflix"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "What is Cosine Similarity, and why is it preferred over Euclidean distance for word embeddings?",
      "answer": "Cosine similarity measures the cosine of the angle between two vectors. It is preferred over Euclidean distance because embedding magnitude is largely influenced by word frequency; the angle captures semantic direction irrespective of vector length.",
      "companyTags": [
        "Google",
        "Amazon"
      ],
      "difficulty": "Beginner"
    },
    {
      "question": "Give an example of how word embeddings capture analogies.",
      "answer": "If you take the vector for \"King\", subtract \"Man\", and add \"Woman\", the resulting vector will be extremely close in cosine similarity to the vector for \"Queen\". This demonstrates spatial representation of semantic relationships.",
      "companyTags": [
        "Apple",
        "Microsoft"
      ],
      "difficulty": "Beginner"
    },
    {
      "question": "What is the main limitation of static word embeddings like Word2Vec and GloVe?",
      "answer": "They assign exactly one vector to each word, failing to capture polysemy. The word \"bank\" has the same vector whether it means a river edge or a financial institution. Contextual embeddings like BERT solve this.",
      "companyTags": [
        "Meta",
        "Google"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "How do you initialize the embedding matrix in Word2Vec?",
      "answer": "The embedding matrix (weights between the input and hidden layer) is typically initialized with small random values, uniformly or normally distributed, and then optimized via backpropagation.",
      "companyTags": [
        "Amazon"
      ],
      "difficulty": "Beginner"
    },
    {
      "question": "What is an Embedding Layer in frameworks like PyTorch or Keras?",
      "answer": "An Embedding layer is essentially a lookup table that maps integer indices (representing words) to dense vectors. It is a trainable parameter matrix equivalent to the input-to-hidden weights in Word2Vec.",
      "companyTags": [
        "Microsoft",
        "Tesla"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "How can you evaluate the quality of word embeddings?",
      "answer": "Intrinsic evaluation involves testing them on analogy tasks (e.g., \"man is to king as woman is to X\") or comparing cosine similarities with human-annotated word similarity datasets. Extrinsic evaluation involves using them as features in downstream NLP tasks and observing performance.",
      "companyTags": [
        "Netflix",
        "Apple"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "What does \"Word Sense Disambiguation\" mean in the context of embeddings?",
      "answer": "It is the process of identifying which sense of a word is used in a sentence. While static embeddings struggle with this, contextual models like ELMo or BERT perform it naturally by generating context-dependent vectors.",
      "companyTags": [
        "Google",
        "Meta"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "Why is negative sampling necessary for training Word2Vec on large vocabularies?",
      "answer": "Without it, backpropagation would require updating the weights for every single word in the vocabulary (often hundreds of thousands) for every training example, which is computationally intractable.",
      "companyTags": [
        "Amazon"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "Can Word2Vec embeddings be fine-tuned during a downstream task?",
      "answer": "Yes. You can load pre-trained Word2Vec embeddings into an embedding layer and allow the network to update them during training, tailoring the representations to the specific task vocabulary and context.",
      "companyTags": [
        "Microsoft"
      ],
      "difficulty": "Beginner"
    },
    {
      "question": "How are document embeddings typically created using Word2Vec?",
      "answer": "The simplest approach is taking the average or sum of all word vectors in the document. More advanced methods use algorithms like Doc2Vec, which introduces a document-specific vector learned alongside the word vectors.",
      "companyTags": [
        "Google",
        "Apple"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "Are Word2Vec embeddings prone to bias?",
      "answer": "Yes, because they learn from human-generated text corpora, they inherently capture and encode societal biases (e.g., associating \"doctor\" closer to male pronouns and \"nurse\" to female pronouns).",
      "companyTags": [
        "Meta",
        "Netflix"
      ],
      "difficulty": "Intermediate"
    }
  ]
};
