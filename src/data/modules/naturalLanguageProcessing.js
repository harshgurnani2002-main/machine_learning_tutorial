export const naturalLanguageProcessing = {
    id: 'natural-language-processing',
    title: 'Natural Language Processing',
    category: 'Deep Learning',
    description: 'Tokenize text, extract features with TF-IDF, embed words with Word2Vec, and classify sentiment with transformers.',
    formula: '\\text{TF-IDF}(t, d) = \\text{TF}(t, d) \\times \\log\\frac{N}{\\text{DF}(t)}',
    theory: `### Natural Language Processing Theory
  
  #### What is it?
  Natural Language Processing (NLP) is a branch of artificial intelligence that enables machines to understand, interpret, and generate human language. It bridges the gap between human communication and computational understanding by converting unstructured text into structured numerical representations that machine learning models can process.
  
  #### Why do we need it?
  Text is the most abundant form of human-generated data — from social media posts and customer reviews to legal documents and medical records. Manually extracting insights from this data is impossible at scale. NLP automates the analysis, classification, and generation of language, powering search engines, chatbots, translation systems, and sentiment analysis tools that billions of people use daily.
  
  #### How does it work?
  NLP pipelines transform raw text through a series of stages:
  
  1. **Tokenization**: Splitting raw text into individual words, subwords, or characters. For example, "I love ML" becomes tokens ["I", "love", "ML"]. This is the foundational step that converts unstructured text into discrete units.
  
  2. **Bag-of-Words (BoW)**: A simple numerical representation that counts how many times each word in a vocabulary appears in a document. Each document becomes a sparse vector of size $|V|$ (vocabulary size). This representation loses word order but captures term frequency.
  
  3. **TF-IDF (Term Frequency–Inverse Document Frequency)**: Improves upon BoW by down-weighting common words that appear across many documents. A word like "the" has high frequency but low informational value.
  
  4. **Word Embeddings (Word2Vec / GloVe)**: Dense vector representations where semantically similar words occupy nearby positions in vector space. Word2Vec uses a shallow neural network trained on local context windows (Skip-gram predicts context from target; CBOW predicts target from context). GloVe performs matrix factorization on global word co-occurrence counts.
  
  5. **Recurrent Architectures (RNN / LSTM)**: Process sequences token-by-token, maintaining a hidden state that captures contextual information. LSTMs add gating mechanisms (forget, input, output gates) to combat the vanishing gradient problem and capture long-range dependencies.
  
  6. **Transformer Architectures**: Replace recurrence with parallel self-attention. Each token attends to every other token via Query, Key, Value projections, enabling O(1) gradient paths between distant tokens and massive parallelism during training.
  
  7. **BERT and Fine-Tuning**: BERT (Bidirectional Encoder Representations from Transformers) is pre-trained on massive corpora via masked language modeling (predicting masked tokens) and next-sentence prediction. Fine-tuning adds a small task-specific head on top of the pre-trained transformer and updates all weights for the target task (e.g., sentiment classification, NER, QA).
  
  #### The Math Behind It
  
  **TF-IDF Weighting:**
  $$
  \\text{TF-IDF}(t, d) = \\text{TF}(t, d) \\times \\log\\frac{N}{\\text{DF}(t)}
  $$
  where $\\text{TF}(t, d)$ is the frequency of term $t$ in document $d$, $\\text{DF}(t)$ is the number of documents containing $t$, and $N$ is the total number of documents.
  
  **Self-Attention (Transformer):**
  $$
  \\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V
  $$
  Queries $Q$, Keys $K$, and Values $V$ are learned linear projections of input embeddings. The softmax yields attention weights that determine how much each token contributes to the representation of every other token.
  
  **Word2Vec Skip-Gram Objective:**
  $$
  \\max \\sum_{w_c \\in \\text{corpus}} \\sum_{w_o \\in \\text{context}(w_c)} \\log P(w_o | w_c)
  $$
  where $P(w_o | w_c) = \\frac{\\exp(v_{w_o}^T v_{w_c})}{\\sum_{w \\in V} \\exp(v_w^T v_{w_c})}$
  
  **Sentiment Classification with BERT:** A [CLS] token is prepended to the input. After passing through $L$ transformer layers, the final hidden state $h_{[CLS]}$ is fed into a linear layer with softmax:
  $$
  P(y | x) = \\text{softmax}(W h_{[CLS]} + b)
  $$
  
  #### Common Pitfalls
  1. **Out-of-Vocabulary Words**: BoW and TF-IDF cannot handle words unseen during training. Subword tokenization (BPE, WordPiece) solves this.
  2. **Context Ignorance**: BoW completely discards word order, missing negation ("not good" vs "good").
  3. **Fixed Embeddings**: Static word embeddings cannot distinguish polysemy ("bank" as river vs financial institution).
  4. **Sequence Length**: Self-attention O(N²) memory limits transformer context windows.
  5. **Domain Mismatch**: Pre-trained embeddings or models may perform poorly on domain-specific text (medical, legal) without fine-tuning.
  
  #### When to Use
  - **BoW/TF-IDF**: Simple baselines, spam detection, topic modeling with small datasets
  - **Word2Vec/GloVe**: Feature extraction for downstream tasks when compute is limited
  - **RNN/LSTM**: Sequence-to-sequence tasks (translation, generation) where order matters
  - **Transformers (BERT)**: State-of-the-art performance on classification, NER, QA, and any task with sufficient data
  
  #### Key Takeaways
  - **Tokenization** is the critical first step — subword tokenizers handle rare words best
  - **TF-IDF** remains a strong baseline that rivals simple neural approaches on small data
  - **Word embeddings** encode semantics geometrically (king - man + woman ≈ queen)
  - **Transformers** replaced RNNs as the dominant architecture due to parallelization and long-range attention
  - **Fine-tuning** pre-trained models like BERT is the standard approach for most NLP tasks today
  
  #### Python Implementation
  
  \`\`\`python
  from sklearn.feature_extraction.text import TfidfVectorizer
  
  docs = ['cat sat mat', 'dog ran park', 'cat ran home']
  vectorizer = TfidfVectorizer()
  X = vectorizer.fit_transform(docs)
  print(f"Shape: {X.shape}")
  print(f"Features: {vectorizer.get_feature_names_out()}")
  \`\`\`,
  `,
    interactiveSummary: 'Type a sentence and watch the NLP pipeline in action: tokenization splits the text, TF-IDF highlights the most informative keywords, and a simulated classifier predicts sentiment with word-level attention weights visualized as a heatmap. Toggle between a simple Bag-of-Words model and a Deep Transformer model to see how attention patterns change.',
    simulatorId: 'nlp-sentiment',
    quiz: [
        {
            id: 'nlp_q1',
            question: 'What is the primary advantage of TF-IDF over simple Bag-of-Words?',
            options: [
                'It down-weights common words that appear in many documents',
                'It preserves word order in the sequence',
                'It generates dense word embeddings',
                'It handles out-of-vocabulary words automatically'
            ],
            correctAnswer: 'It down-weights common words that appear in many documents',
            explanation: 'TF-IDF multiplies term frequency by inverse document frequency, reducing the weight of common words like "the" or "and" that carry little semantic meaning.'
        },
        {
            id: 'nlp_q2',
            question: 'Which Word2Vec architecture predicts the target word given its surrounding context?',
            options: ['CBOW', 'Skip-Gram', 'GloVe', 'FastText'],
            correctAnswer: 'CBOW',
            explanation: 'Continuous Bag of Words (CBOW) takes the surrounding context words as input and tries to predict the target word in the middle.'
        },
        {
            id: 'nlp_q3',
            question: 'What is the key architectural difference between RNNs and Transformers?',
            options: [
                'Transformers process all tokens in parallel using self-attention, while RNNs process tokens sequentially',
                'RNNs use attention mechanisms while Transformers use recurrence',
                'Transformers cannot handle long sequences but RNNs can',
                'RNNs are parallelizable while Transformers are sequential'
            ],
            correctAnswer: 'Transformers process all tokens in parallel using self-attention, while RNNs process tokens sequentially',
            explanation: 'Self-attention computes pairwise interactions between all tokens simultaneously, enabling massive parallelism. RNNs must process one token at a time, making them slower to train.'
        },
        {
            id: 'nlp_q4',
            question: 'What does the [CLS] token represent in a BERT model used for classification?',
            options: [
                'The aggregate sequence representation fed to the classifier head',
                'A token indicating the start of a new sentence',
                'A separator between two input sentences',
                'A masked token that the model must predict'
            ],
            correctAnswer: 'The aggregate sequence representation fed to the classifier head',
            explanation: 'The final hidden state of the [CLS] token is used as the pooled sequence representation for classification tasks after fine-tuning.'
        },
        {
            id: 'nlp_q5',
            question: 'What problem does subword tokenization (e.g., BPE, WordPiece) solve?',
            options: [
                'It handles out-of-vocabulary words by breaking them into known subword units',
                'It eliminates the need for positional encodings',
                'It reduces the vocabulary size to 26 letters',
                'It automatically labels parts of speech'
            ],
            correctAnswer: 'It handles out-of-vocabulary words by breaking them into known subword units',
            explanation: 'Subword tokenizers decompose rare words into smaller frequent subwords (e.g., "unhappiness" → "un", "happiness"), ensuring any word can be represented.'
        },
        {
            id: 'nlp_q6',
            question: 'How does fine-tuning BERT differ from using BERT as a fixed feature extractor?',
            options: [
                'Fine-tuning updates all pre-trained weights for the target task; feature extraction freezes them',
                'Fine-tuning only updates the embedding layer',
                'Feature extraction requires more data than fine-tuning',
                'They are identical approaches with different names'
            ],
            correctAnswer: 'Fine-tuning updates all pre-trained weights for the target task; feature extraction freezes them',
            explanation: 'Fine-tuning backpropagates through the entire pre-trained model, adapting representations to the task. Feature extraction uses BERT as a fixed encoder and only trains the classification head.'
        },
        {
            id: 'nlp_q7',
            question: 'What is a major limitation of static word embeddings like Word2Vec?',
            options: [
                'They assign a single vector per word, failing to capture multiple meanings',
                'They require too much memory for deployment',
                'They cannot be trained on GPU hardware',
                'They only work with English text'
            ],
            correctAnswer: 'They assign a single vector per word, failing to capture multiple meanings',
            explanation: 'Static embeddings give one representation per word regardless of context. "Bank" has the same vector whether referring to a river bank or a financial institution.'
        },
        {
            id: 'nlp_q8',
            question: 'What is the computational complexity of self-attention with respect to sequence length N?',
            options: ['O(N²)', 'O(N)', 'O(N log N)', 'O(1)'],
            correctAnswer: 'O(N²)',
            explanation: 'Self-attention computes pairwise scores between all N tokens, resulting in an N×N attention matrix. This quadratic complexity limits context window sizes.'
        }
    ],
    coding: {
        tutorial: {
            title: 'Build a Bag-of-Words Classifier',
            description: 'Implement a simple bag-of-words text representation using a count vectorizer, then train a logistic regression classifier for sentiment analysis on toy data.\n\n**Sklearn Equivalent:**\n`from sklearn.feature_extraction.text import CountVectorizer`\n`from sklearn.linear_model import LogisticRegression`',
            pseudoCode: 'function bow_classifier(texts, labels):\n  vocab = build_vocabulary(texts)\n  X = count_vectorize(texts, vocab)\n  model = LogisticRegression()\n  model.fit(X, labels)\n  return model\n\nfunction predict(text, model, vocab):\n  vec = count_vectorize([text], vocab)\n  return model.predict(vec)',
            starterCode: `import numpy as np
from sklearn.linear_model import LogisticRegression

texts = [
    "this movie was amazing and exciting",
    "i loved every minute of this film",
    "what a wonderful experience",
    "terribly boring and dull movie",
    "i hated this film it was awful",
    "worst movie ever made"
]
labels = [1, 1, 1, 0, 0, 0]  # 1 = positive, 0 = negative

# TODO: Build vocabulary from texts (unique words)
# TODO: Create bag-of-words vectors (count per word)
# TODO: Train LogisticRegression on X, labels
# TODO: Predict "this film was amazing" and print result

vocab = []
X = np.array([])
model = None
prediction = None
print("Sentiment:", "positive" if prediction == 1 else "negative")`,
            expectedOutput: 'Sentiment: positive',
            solution: `import numpy as np
from sklearn.linear_model import LogisticRegression

texts = [
    "this movie was amazing and exciting",
    "i loved every minute of this film",
    "what a wonderful experience",
    "terribly boring and dull movie",
    "i hated this film it was awful",
    "worst movie ever made"
]
labels = [1, 1, 1, 0, 0, 0]

vocab = sorted(set(w for t in texts for w in t.split()))
def vectorize(text):
    vec = np.zeros(len(vocab))
    for w in text.split():
        if w in vocab:
            vec[vocab.index(w)] += 1
    return vec

X = np.array([vectorize(t) for t in texts])
model = LogisticRegression()
model.fit(X, labels)
prediction = model.predict([vectorize("this film was amazing")])[0]
print("Sentiment:", "positive" if prediction == 1 else "negative")`,
            hints: ['Build a sorted vocabulary from all unique words.', 'Create a function that counts word occurrences per text.', 'Use LogisticRegression().fit(X, labels) and .predict().'],
            testKeywords: ['LogisticRegression', 'vocab', 'fit', 'predict', 'vectorize']
        },
        project: {
            title: 'Sentiment Analysis with sklearn Pipeline',
            description: 'Build a complete sentiment analysis pipeline using TF-IDF vectorization and a linear SVM classifier on a larger set of movie review snippets. Evaluate accuracy using train/test split.',
            pseudoCode: `# Step 1: Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3)

# Step 2: Build pipeline with TF-IDF + LinearSVC
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LinearSVC())
])

# Step 3: Train and evaluate
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")`,
            starterCode: `from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np

texts = [
    "absolutely loved this film it was brilliant",
    "great acting and stunning visuals",
    "a masterpiece of modern cinema",
    "wonderful story with amazing performances",
    "boring predictable and waste of time",
    "terrible acting ruined the entire movie",
    "worst film i have seen all year",
    "disappointing dull and poorly directed",
    "fantastic movie highly recommend watching",
    "not worth the ticket price at all",
    "incredible soundtrack and beautiful cinematography",
    "poor script and awful dialogue throughout",
]
labels = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0]

# TODO: Split into train/test with test_size=0.3
# TODO: Create Pipeline with TfidfVectorizer and LinearSVC
# TODO: Train and print accuracy

X_train, X_test, y_train, y_test = None, None, None, None
pipeline = None
accuracy = 0.0
print(f"Accuracy: {accuracy:.2f}")`,
            expectedOutput: 'Accuracy: 1.00',
            solution: `from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np

texts = [
    "absolutely loved this film it was brilliant",
    "great acting and stunning visuals",
    "a masterpiece of modern cinema",
    "wonderful story with amazing performances",
    "boring predictable and waste of time",
    "terrible acting ruined the entire movie",
    "worst film i have seen all year",
    "disappointing dull and poorly directed",
    "fantastic movie highly recommend watching",
    "not worth the ticket price at all",
    "incredible soundtrack and beautiful cinematography",
    "poor script and awful dialogue throughout",
]
labels = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0]

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LinearSVC())
])
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")`,
            hints: ['Use train_test_split(texts, labels, test_size=0.3, random_state=42).', 'Build Pipeline([("tfidf", TfidfVectorizer()), ("clf", LinearSVC())]).', 'Call pipeline.fit() then pipeline.score() for evaluation.'],
            testKeywords: ['Pipeline', 'TfidfVectorizer', 'LinearSVC', 'train_test_split', 'fit', 'score']
        },
        assignment: {
            title: 'Tiny Transformer Attention Block',
            description: 'Implement a minimal scaled dot-product self-attention mechanism in NumPy. Given a sequence of token embeddings, compute the Q, K, V projections, calculate attention scores, and produce context-aware output vectors.',
            pseudoCode: `# Tiny Self-Attention Block
# Input: X (N x d_model) — sequence of N tokens
# Weights: W_q, W_k, W_v (d_model x d_k)
def self_attention(X, W_q, W_k, W_v):
    d_k = W_q.shape[1]
    Q = X @ W_q       # (N x d_k)
    K = X @ W_k       # (N x d_k)
    V = X @ W_v       # (N x d_v)
    scores = (Q @ K.T) / sqrt(d_k)  # (N x N)
    weights = softmax(scores, axis=-1)  # (N x N)
    return weights @ V  # (N x d_v)`,
            starterCode: `import numpy as np

# Token embeddings: sequence of 4 tokens, each embedding dim 8
X = np.array([
    [0.9, 0.1, 0.5, 0.2, 0.7, 0.3, 0.8, 0.4],
    [0.8, 0.9, 0.2, 0.7, 0.1, 0.6, 0.3, 0.5],
    [0.3, 0.4, 0.9, 0.8, 0.5, 0.2, 0.1, 0.7],
    [0.2, 0.7, 0.4, 0.9, 0.3, 0.8, 0.5, 0.1],
])

# Random projection matrices
np.random.seed(42)
d_model = 8
d_k = 4
W_q = np.random.randn(d_model, d_k)
W_k = np.random.randn(d_model, d_k)
W_v = np.random.randn(d_model, d_k)

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def self_attention(X, W_q, W_k, W_v):
    # TODO: Compute Q, K, V projections
    # TODO: Compute scaled dot-product scores
    # TODO: Apply softmax to get attention weights
    # TODO: Multiply weights by V and return
    return None

output = self_attention(X, W_q, W_k, W_v)
print("Output shape:", output.shape)
print("First token output:", np.round(output[0], 3))`,
            expectedOutput: 'Output shape: (4, 4)\nFirst token output: [0.725 0.472 0.583 0.379]',
            solution: `import numpy as np

X = np.array([
    [0.9, 0.1, 0.5, 0.2, 0.7, 0.3, 0.8, 0.4],
    [0.8, 0.9, 0.2, 0.7, 0.1, 0.6, 0.3, 0.5],
    [0.3, 0.4, 0.9, 0.8, 0.5, 0.2, 0.1, 0.7],
    [0.2, 0.7, 0.4, 0.9, 0.3, 0.8, 0.5, 0.1],
])

np.random.seed(42)
d_model = 8
d_k = 4
W_q = np.random.randn(d_model, d_k)
W_k = np.random.randn(d_model, d_k)
W_v = np.random.randn(d_model, d_k)

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def self_attention(X, W_q, W_k, W_v):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    scores = (Q @ K.T) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    return weights @ V

output = self_attention(X, W_q, W_k, W_v)
print("Output shape:", output.shape)
print("First token output:", np.round(output[0], 3))`,
            hints: ['Project X using matrix multiplication: X @ W_q', 'Scale scores by dividing by sqrt(d_k)', 'Use the provided softmax function on the last axis', 'Return weights @ V for the final attended output'],
            testKeywords: ['self_attention', 'softmax', '@', 'np.sqrt', 'Q @ K.T']
        }
    },
    interviewQuestions: [
        {
            question: 'Explain the Transformer self-attention formula in detail. Why is scaling by sqrt(d_k) necessary?',
            answer: 'The formula is Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V. Q, K, V are learned linear projections of input embeddings. The dot product QK^T measures similarity between each query-key pair. Scaling by sqrt(d_k) prevents the dot products from growing too large when d_k is high, which would push softmax into flat gradient regions. After scaling, softmax converts scores to probabilities, and the weighted sum of V produces the context-aware output.',
            difficulty: 'Advanced',
            companyTags: ['Google', 'OpenAI', 'Meta']
        },
        {
            question: 'How does BERT differ from GPT in architecture and training objectives?',
            answer: 'BERT uses an encoder-only architecture with bidirectional self-attention, trained via Masked Language Model (predicting masked tokens) and Next Sentence Prediction. GPT uses a decoder-only architecture with masked (causal) self-attention, trained autoregressively to predict the next token. BERT excels at understanding tasks (classification, NER, QA), while GPT excels at generation tasks.',
            difficulty: 'Advanced',
            companyTags: ['Google', 'OpenAI', 'Microsoft']
        },
        {
            question: 'Compare Bag-of-Words, TF-IDF, and Word Embeddings. When would you choose each?',
            answer: 'BoW is simplest — sparse count vectors, loses all semantics and order. TF-IDF improves BoW by down-weighting common words; useful for baselines and small datasets. Word embeddings (Word2Vec/GloVe) produce dense vectors capturing semantic similarity; good for transfer learning and downstream tasks. For deep learning, always prefer embeddings (pre-trained or learned). For linear models on small data (<10k examples), TF-IDF often matches or beats simple neural approaches.',
            difficulty: 'Intermediate',
            companyTags: ['Amazon', 'Apple', 'Netflix']
        },
        {
            question: 'What is the OOV (Out-of-Vocabulary) problem and how do modern tokenizers solve it?',
            answer: 'OOV occurs when a model encounters a word not seen during training. BoW and TF-IDF simply ignore them. Word2Vec can\'t represent them. Modern subword tokenizers (BPE, WordPiece, SentencePiece) break words into frequent subword units — e.g., "unhappiness" → "un" + "happiness" or even "un" + "happy" + "ness". Any word can be represented as a sequence of known subwords, effectively eliminating OOV issues. This is standard in all modern Transformer models.',
            difficulty: 'Intermediate',
            companyTags: ['Meta', 'Google', 'Apple']
        },
        {
            question: 'Describe the process of fine-tuning BERT for a sentiment classification task.',
            answer: '1. Load pre-trained BERT model and tokenizer. 2. Add a classification head (a linear layer + softmax on top of the [CLS] token output). 3. Tokenize input sentences with padding/truncation to a fixed max length. 4. Train the entire model (or just the head) on labeled sentiment data using Cross-Entropy loss. 5. Fine-tune for 2-4 epochs with a small learning rate (2e-5 to 5e-5). 6. The [CLS] token\'s final hidden state serves as the aggregate representation for classification. Key advantage: BERT transfers rich linguistic knowledge learned from billions of words to the specific task.',
            difficulty: 'Advanced',
            companyTags: ['Microsoft', 'Amazon', 'Meta']
        }
    ]
};
