export const naiveBayes = {
    id: 'naive-bayes',
    title: 'Naive Bayes Classifier',
    category: 'Supervised Learning',
    description: 'Predict classes using conditional probability likelihoods and Bayes Theorem.',
    formula: 'P(y|x) \\propto P(y) \\prod P(x_i|y)',
    theory: `### Naive Bayes & Probabilistic Classification

#### What is it?
Naive Bayes is a family of probabilistic machine learning models based on applying Bayes' theorem. It is heavily utilized for classification tasks, particularly text classification (like spam detection or sentiment analysis). It operates by calculating the probability of each class given a set of features and selecting the class with the highest probability. The "naive" aspect comes from the assumption that all features are conditionally independent of each other given the class label.

#### Why do we need it?
In many real-world scenarios, we deal with datasets containing thousands of features (e.g., word counts in a document, where the vocabulary size is 10,000+). Complex algorithms might overfit, require vast amounts of data to converge, or take excessive computation time. Naive Bayes scales linearly with the number of predictors and training rows. It requires a surprisingly small amount of training data to estimate the necessary parameters (means and variances, or frequencies), making it highly efficient, robust, and a strong baseline for classification tasks.

#### How does it work?
The model computes the posterior probability of a class, given the input features. To do this efficiently, it makes a strict assumption: the presence or value of a particular feature in a class is completely independent of the presence of any other feature. Even though this assumption is almost always false in reality (e.g., the word "machine" often appears with "learning"), Naive Bayes performs exceptionally well in practice because the relative ranking of classes often remains accurate even if the exact probabilities are skewed.

#### The Math Behind It
Bayes' Theorem provides a way of calculating posterior probability $P(y|x)$:
$$P(y|x) = \frac{P(x|y) P(y)}{P(x)}$$

- $P(y|x)$ is the **posterior probability** of class ($y$, target) given predictor ($x$, attributes).
- $P(y)$ is the **prior probability** of the class.
- $P(x|y)$ is the **likelihood** which is the probability of predictor given class.
- $P(x)$ is the **prior probability** of the predictor.

Using the naive conditional independence assumption, we can decompose the likelihood:
$$P(x_1, x_2, \\dots, x_n | y) = \\prod_{i=1}^{n} P(x_i | y)$$

This simplifies the posterior calculation. Since $P(x)$ is constant for all classes, we can drop the denominator:
$$P(y | x) \\propto P(y) \\prod_{i=1}^{n} P(x_i | y)$$

To prevent zero probabilities for features unseen during training, we use **Laplace Smoothing**:
$$P(x_i | y) = \frac{count(x_i, y) + \\alpha}{count(y) + \\alpha \\cdot |V|}$$
where $\\alpha$ is the smoothing parameter (usually 1.0) and $|V|$ is the number of unique features.

#### Worked Example
Consider a dataset classifying emails as "Spam" or "Not Spam" based on words.
- $P(Spam) = 0.4$, $P(Not Spam) = 0.6$
- Email text: "win money"
- Likelihoods for Spam: $P('win'|Spam) = 0.2$, $P('money'|Spam) = 0.3$
- Likelihoods for Not Spam: $P('win'|Not Spam) = 0.05$, $P('money'|Not Spam) = 0.01$

Calculate proportional posteriors:
Spam: $0.4 \times 0.2 \times 0.3 = 0.024$
Not Spam: $0.6 \times 0.05 \times 0.01 = 0.0003$
Since $0.024 > 0.0003$, the email is classified as Spam.

#### Common Pitfalls
- **Numeric Underflow**: Multiplying many small probabilities results in floating-point underflow. Work around this by summing log probabilities instead: $\\log P(y|x) \\propto \\log P(y) + \\sum \\log P(x_i|y)$.
- **Correlated Features**: The independence assumption is violated if features are highly correlated, degrading probability estimation (though the ranking of classes may remain correct).
- **Zero Frequency**: Handled via Laplace Smoothing, otherwise unseen words zero out the entire product.
- **Continuous Data Assumptions**: Gaussian Naive Bayes assumes features follow a normal distribution. If they don\'t, transformations (like log or Box-Cox) may be needed.

#### When to Use vs Not Use
**Use when**:
- Dataset is massive, and you need a fast, scalable algorithm.
- Text classification (NLP tasks like spam filtering or topic categorization).
- Features are largely independent.
- You need a simple baseline model to benchmark more complex algorithms.

**Not Use when**:
- Features have complex conditional dependencies.
- You need perfectly calibrated probability scores, not just categorical predictions.
- Dataset is tiny and you have highly complex, non-linear decision boundaries.

#### Key Takeaways
- Based on Bayes' Theorem with a naive independence assumption.
- Highly scalable, fast to train, and computationally efficient at inference.
- Requires logarithmic transformations to prevent mathematical underflow.
- Handles missing or unseen data gracefully using Laplace smoothing.
#### Python Implementation

\`\`\`python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
gnb = GaussianNB()
gnb.fit(X, y)
print(f"Accuracy: {gnb.score(X, y):.2f}")
\`\`\`
`,
    interactiveSummary: "This interactive simulator shows Naive Bayes in action as a live probability calculator. Each time you adjust a feature slider or toggle a word, the bar chart on the right updates in real time to display the posterior probability P(Class | Features) for each class — the taller the bar, the more confident the model is that the input belongs to that class. You can see how the prior probability (the baseline class frequency) shifts the bars before any features are added, and how each additional feature multiplies the likelihood and pushes probability mass toward the correct class. The visualization also highlights what happens with rare or unseen features: without Laplace smoothing, a single zero probability would collapse the entire bar to nothing, demonstrating why smoothing is critical. This makes it easy to build intuition for how Naive Bayes combines independent evidence into a final, probabilistic verdict.",
    simulatorId: 'naive-bayes',
    quiz: [
        {
            id: 'nb_q1',
            question: 'What is the "Naive" assumption in the Naive Bayes classifier?',
            options: [
                'All features are conditionally independent of each other given the class label.',
                'The features always follow a Gaussian normal distribution.',
                'Prior probabilities are equal for all classes.',
                'The data is linearly separable.'
            ],
            correctAnswer: 'All features are conditionally independent of each other given the class label.',
            explanation: 'It assumes that the presence of one feature does not influence the probability of another feature, given the class label.'
        },
        {
            id: 'nb_q2',
            question: 'Why is Laplace smoothing applied during Naive Bayes training?',
            options: [
                'To prevent zero probabilities for unseen features from canceling the entire product.',
                'To scale features between 0 and 1.',
                'To reduce variance of support vectors.',
                'To make the decision boundary continuous.'
            ],
            correctAnswer: 'To prevent zero probabilities for unseen features from canceling the entire product.',
            explanation: 'If a feature never appeared with class y, P(x_i|y) = 0, which zeroes out the cumulative product. Laplace smoothing adds dummy counts.'
        },
        {
            id: 'nb_q3',
            question: 'Which variant of Naive Bayes is designed for continuous variables?',
            options: [
                'Gaussian Naive Bayes',
                'Multinomial Naive Bayes',
                'Bernoulli Naive Bayes',
                'Bivariate Naive Bayes'
            ],
            correctAnswer: 'Gaussian Naive Bayes',
            explanation: 'Gaussian Naive Bayes assumes continuous features are normally distributed within each class, calculating likelihoods via Gaussian probability density function.'
        },
        {
            id: 'nb_q4',
            question: 'How do we solve the numerical underflow problem in Naive Bayes?',
            options: [
                'By summing the logarithms of probabilities instead of multiplying them.',
                'By dividing all probabilities by a large constant.',
                'By applying a sigmoid function to the probabilities.',
                'By removing features with low probabilities.'
            ],
            correctAnswer: 'By summing the logarithms of probabilities instead of multiplying them.',
            explanation: 'Multiplying many probabilities (values < 1) causes floating-point underflow. Using log(P) allows adding values instead of multiplying them.'
        },
        {
            id: 'nb_q5',
            question: 'What does Multinomial Naive Bayes typically model?',
            options: [
                'Discrete counts, like term frequencies in text classification.',
                'Binary presence/absence of features.',
                'Continuous numeric variables.',
                'Ordinal variables with strict rankings.'
            ],
            correctAnswer: 'Discrete counts, like term frequencies in text classification.',
            explanation: 'Multinomial Naive Bayes is widely used in NLP tasks where features represent the counts or frequencies of words in a document.'
        },
        {
            id: 'nb_q6',
            question: 'What happens to the Naive Bayes classifier if features are highly correlated?',
            options: [
                'The estimated probabilities become overconfident, but the classification boundary might still be accurate.',
                'The model cannot be trained and will throw an error.',
                'The model automatically performs feature selection to remove correlated features.',
                'The accuracy will always drop to 0%.'
            ],
            correctAnswer: 'The estimated probabilities become overconfident, but the classification boundary might still be accurate.',
            explanation: 'Because of the naive independence assumption, correlated evidence is double-counted, leading to skewed probability estimates, though the final predicted class often remains correct.'
        },
        {
            id: 'nb_q7',
            question: 'Which equation represents Bayes Theorem?',
            options: [
                'P(A|B) = P(B|A) * P(A) / P(B)',
                'P(A|B) = P(B|A) + P(A) - P(B)',
                'P(A|B) = P(A) * P(B) / P(B|A)',
                'P(A|B) = P(A AND B) * P(A)'
            ],
            correctAnswer: 'P(A|B) = P(B|A) * P(A) / P(B)',
            explanation: 'Bayes Theorem states that the posterior probability P(A|B) is the likelihood P(B|A) multiplied by the prior P(A), divided by the evidence P(B).'
        },
        {
            id: 'nb_q8',
            question: 'What is the role of prior probability P(y) in Naive Bayes?',
            options: [
                'It reflects the baseline frequency of class y in the training set.',
                'It normalizes the posterior probabilities to sum to 1.',
                'It measures the similarity between features.',
                'It prevents numerical underflow.'
            ],
            correctAnswer: 'It reflects the baseline frequency of class y in the training set.',
            explanation: 'The prior P(y) is calculated as the proportion of samples belonging to class y in the entire training dataset.'
        },
        {
            id: 'nb_q9',
            question: 'Bernoulli Naive Bayes is best suited for what kind of data?',
            options: [
                'Binary/boolean features (e.g., word presence vs absence).',
                'Continuous measurements.',
                'Frequency counts of words.',
                'Time-series continuous data.'
            ],
            correctAnswer: 'Binary/boolean features (e.g., word presence vs absence).',
            explanation: 'Bernoulli Naive Bayes assumes features are independent boolean variables.'
        },
        {
            id: 'nb_q10',
            question: 'What is the computational complexity of training a Naive Bayes classifier?',
            options: [
                'O(N * d)',
                'O(N^2 * d)',
                'O(N * d^2)',
                'O(2^N)'
            ],
            correctAnswer: 'O(N * d)',
            explanation: 'Training requires a single pass over the dataset to compute counts, taking linear time with respect to the number of samples N and features d.'
        }
    ],
    interviewQuestions: [
        {
            question: 'Explain what numeric underflow is in Naive Bayes and how to solve it.',
            answer: 'Numeric underflow occurs when multiplying many fractional probabilities (value range [0, 1]), causing the product to decay to zero and exceed standard floating-point capacity. The solution is to use the logarithmic property log(A * B) = log(A) + log(B), transforming the product of probabilities into a sum of log probabilities.',
            companyTags: ['Google', 'Bloomberg'],
            difficulty: 'Medium'
        },
        {
            question: 'What is the impact of violating the independence assumption in Naive Bayes?',
            answer: 'Violating the conditional independence assumption means correlated features will have their evidence double-counted. This makes the model overconfident (predicted probabilities approach 0 or 1). However, the relative ranking of classes often remains correct, which is why Naive Bayes can still achieve high accuracy even when the assumption fails.',
            companyTags: ['Meta', 'Amazon'],
            difficulty: 'Medium'
        },
        {
            question: 'Why do we need Laplace smoothing in Naive Bayes?',
            answer: 'Without smoothing, if a feature value (e.g., a specific word) is entirely unseen in the training set for a particular class, its conditional probability is 0. Because Naive Bayes multiplies all feature probabilities together, a single 0 will wipe out all other evidence. Laplace smoothing adds a small pseudo-count to all feature frequencies to prevent this.',
            companyTags: ['LinkedIn', 'Microsoft'],
            difficulty: 'Easy'
        },
        {
            question: 'Compare Gaussian vs Multinomial vs Bernoulli Naive Bayes.',
            answer: 'Gaussian NB assumes continuous features are normally distributed. Multinomial NB is used for discrete counts (e.g., term frequencies in text). Bernoulli NB is used for discrete binary/boolean features (e.g., presence or absence of a word in a document).',
            companyTags: ['Apple', 'Uber'],
            difficulty: 'Medium'
        },
        {
            question: 'Is Naive Bayes a discriminative or generative model?',
            answer: 'Naive Bayes is a generative model. It learns the joint probability P(x, y) = P(x|y)P(y) and then uses Bayes Theorem to estimate the posterior P(y|x). Discriminative models like Logistic Regression directly learn the decision boundary P(y|x) without modeling the distribution of the features.',
            companyTags: ['DeepMind', 'OpenAI'],
            difficulty: 'Hard'
        },
        {
            question: 'How does Naive Bayes handle missing data during testing?',
            answer: 'Naive Bayes can handle missing data gracefully. During inference, if a feature is missing, the algorithm simply omits that feature\'s probability from the product calculation, implicitly integrating it out, relying on the remaining available features to make a prediction.',
            companyTags: ['Netflix', 'Spotify'],
            difficulty: 'Medium'
        },
        {
            question: 'Can Naive Bayes be used for regression?',
            answer: 'Standard Naive Bayes is designed for classification. To use probabilistic models for regression, one typically uses Bayesian Regression models (e.g., Bayesian Linear Regression), which apply Bayes theorem to estimate a continuous distribution rather than discrete class categories.',
            companyTags: ['Stripe', 'Square'],
            difficulty: 'Medium'
        },
        {
            question: 'What is the time complexity of Naive Bayes for training and prediction?',
            answer: 'Training time complexity is O(Nd) where N is the number of samples and d is the number of features, as it only requires computing frequencies/means. Prediction time complexity is O(d * C) where C is the number of classes, making it extremely fast for real-time inference.',
            companyTags: ['TikTok', 'Amazon'],
            difficulty: 'Easy'
        },
        {
            question: 'How do Outliers affect Naive Bayes?',
            answer: 'Outliers can significantly affect Gaussian Naive Bayes because extreme values distort the mean and variance estimations of the normal distributions. However, Multinomial and Bernoulli variants are more robust to outliers since they rely on counts or binary flags.',
            companyTags: ['Google', 'Meta'],
            difficulty: 'Medium'
        },
        {
            question: 'What does the "naive" mean in Naive Bayes?',
            answer: '"Naive" refers to the assumption that all input features are conditionally independent of each other given the target class. This is rarely true in the real world (e.g., the words "machine" and "learning" are correlated), but it drastically simplifies the math.',
            companyTags: ['Microsoft', 'IBM'],
            difficulty: 'Easy'
        },
        {
            question: 'How do you handle continuous variables in a Multinomial Naive Bayes model?',
            answer: 'Multinomial NB expects counts. You can either bin/discretize the continuous variables into categorical buckets (e.g., age groups) or switch to Gaussian Naive Bayes if the continuous features roughly follow a normal distribution.',
            companyTags: ['Pinterest', 'Snap'],
            difficulty: 'Medium'
        },
        {
            question: 'Explain the difference between P(y|x) and P(x|y) in Bayes Theorem.',
            answer: 'P(y|x) is the posterior probability: the probability of the class given the observed features. P(x|y) is the likelihood: the probability of observing these specific features assuming the class is y. Bayes Theorem connects the two via the prior P(y).',
            companyTags: ['Anthropic', 'Google'],
            difficulty: 'Medium'
        },
        {
            question: 'What is the denominator P(x) in Bayes Theorem and why is it often ignored?',
            answer: 'P(x) is the evidence, representing the overall probability of seeing the feature set across all classes. Since P(x) is identical for all classes being compared during prediction, it acts merely as a scaling factor and can be dropped to maximize the numerator instead (MAP estimation).',
            companyTags: ['Meta', 'Uber'],
            difficulty: 'Medium'
        },
        {
            question: 'Why does Naive Bayes perform well on text classification?',
            answer: 'Text classification usually involves high-dimensional, sparse data (vocabularies of 10,000+ words). Naive Bayes is immune to the curse of dimensionality compared to distance-based algorithms, and the simple independence assumption works surprisingly well to isolate strongly predictive words.',
            companyTags: ['Twitter', 'LinkedIn'],
            difficulty: 'Easy'
        },
        {
            question: 'Does Naive Bayes require feature scaling?',
            answer: 'Multinomial and Bernoulli Naive Bayes do not require scaling as they deal with counts and binary values. Gaussian Naive Bayes evaluates features independently using individual standard deviations, so it is generally invariant to monotonic feature scaling.',
            companyTags: ['Amazon', 'Bloomberg'],
            difficulty: 'Medium'
        },
        {
            question: 'What happens if you set alpha=0 in Laplace smoothing?',
            answer: 'Setting alpha=0 disables smoothing. If an unseen feature occurs in the test set, its conditional probability P(x|y) will be evaluated to exactly zero, nullifying the entire product equation and resulting in a posterior probability of 0 for that class.',
            companyTags: ['Google', 'Meta'],
            difficulty: 'Medium'
        },
        {
            question: 'How is Naive Bayes related to Logistic Regression?',
            answer: 'Both are linear classifiers (the log-odds decision boundary of Naive Bayes is linear). However, Logistic Regression is discriminative (learns weights iteratively to maximize conditional likelihood), while Naive Bayes is generative (computes counts analytically in a single pass).',
            companyTags: ['DeepMind', 'Google'],
            difficulty: 'Hard'
        },
        {
            question: 'Can Naive Bayes learn XOR?',
            answer: 'No. Naive Bayes is fundamentally a linear classifier and cannot capture complex feature interactions (like XOR) because of its strict conditional independence assumption.',
            companyTags: ['Apple', 'Microsoft'],
            difficulty: 'Hard'
        },
        {
            question: 'What is Maximum A Posteriori (MAP) estimation?',
            answer: 'MAP estimation is the process of choosing the class that maximizes the posterior probability P(y|x). Naive Bayes relies on MAP by choosing the class with the highest proportional score (Prior * Likelihood).',
            companyTags: ['OpenAI', 'Anthropic'],
            difficulty: 'Hard'
        },
        {
            question: 'How do you tune the alpha parameter in Naive Bayes?',
            answer: 'The smoothing parameter alpha is typically tuned using cross-validation (e.g., GridSearch). While alpha=1 (Laplace smoothing) is a common default, alpha < 1 (Lidstone smoothing) might yield better results depending on the dataset\'s sparsity.',
            companyTags: ['Uber', 'Stripe'],
            difficulty: 'Medium'
        }
    ],
    coding: {
        tutorial: {
            title: 'Calculate Prior Probabilities',
            description: 'Calculate prior probabilities P(y) for each class in a label array. Sklearn equivalent: `MultinomialNB().fit(X, y)` automatically calculates priors as `model.class_log_prior_`.',
            pseudoCode: 'Count occurrences of each label class.\nDivide counts by the total number of labels to get proportions.',
            starterCode: `import numpy as np

def calculate_priors(labels):
    # TODO: Return prior probability array [P(y=0), P(y=1)]
    return np.array([0.0, 0.0])

lbls = np.array([0, 1, 1, 1, 0])
print("Priors:", calculate_priors(lbls))`,
            expectedOutput: 'Priors: [0.4 0.6]',
            solution: `import numpy as np

def calculate_priors(labels):
    counts = np.bincount(labels)
    return counts / len(labels)

lbls = np.array([0, 1, 1, 1, 0])
print("Priors:", calculate_priors(lbls))`,
            hints: ['Use np.bincount to count labels.', 'Divide counts by total length of labels array.'],
            testKeywords: ['bincount', 'len']
        },
        project: {
            title: 'Spam Email Detection with Naive Bayes',
            description: 'Build a real-world spam email detector using a `CountVectorizer` to convert raw email text into word-frequency features, then train a `MultinomialNB` classifier. The training corpus contains 10 emails drawn from real-world spam patterns — promotional language ("free", "win", "cash", "prize") vs. professional communication ("meeting", "report", "schedule", "agenda"). Test on two new emails to verify the classifier generalizes correctly.',
            pseudoCode: 'corpus = [10 real-style email strings]\nlabels = [1=spam or 0=legitimate]\nvectorizer = CountVectorizer()\nX_train = vectorizer.fit_transform(corpus)\nnb = MultinomialNB(alpha=1.0)\nnb.fit(X_train, labels)\nX_test = vectorizer.transform(["win free cash prize", "send the quarterly report"])\npredictions = nb.predict(X_test)\nprint("Predictions:", predictions)',
            starterCode: `from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Realistic spam detection corpus
corpus = [
    "win free cash now limited offer",          # spam
    "congratulations you have won a prize",     # spam
    "click here to claim your free gift card",  # spam
    "exclusive deal free money today",           # spam
    "you are selected winner act now",           # spam
    "meeting scheduled for thursday morning",   # legitimate
    "please review the attached quarterly report",  # legitimate
    "team lunch agenda for next friday",        # legitimate
    "project deadline is end of month reminder", # legitimate
    "updated schedule for the client presentation",  # legitimate
]
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1=spam, 0=legitimate

test_emails = ["win free cash prize", "send the quarterly report"]

# TODO: Initialize CountVectorizer and fit_transform the corpus
# TODO: Initialize MultinomialNB and fit it
# TODO: Transform test_emails and predict their labels
predictions = []
print("Predictions:", predictions)`,
            expectedOutput: 'Predictions: [1 0]',
            solution: `from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

corpus = [
    "win free cash now limited offer",
    "congratulations you have won a prize",
    "click here to claim your free gift card",
    "exclusive deal free money today",
    "you are selected winner act now",
    "meeting scheduled for thursday morning",
    "please review the attached quarterly report",
    "team lunch agenda for next friday",
    "project deadline is end of month reminder",
    "updated schedule for the client presentation",
]
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

test_emails = ["win free cash prize", "send the quarterly report"]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(corpus)

nb = MultinomialNB()
nb.fit(X_train, labels)

X_test = vectorizer.transform(test_emails)
predictions = nb.predict(X_test)

print("Predictions:", predictions)`,
            hints: ['Call fit_transform() on the training corpus, then transform() (not fit_transform) on test emails.', 'MultinomialNB expects count-based features from CountVectorizer.', 'Call predict() on the fitted classifier with the transformed test data.'],
            testKeywords: ['CountVectorizer', 'MultinomialNB', 'fit_transform', 'predict']
        },
        assignment: {
            title: 'News Category Classification Pipeline',
            description: 'A news aggregator wants to automatically tag articles into topics: Technology, Sports, or Politics. Each article is a short headline. Build an sklearn Pipeline combining `TfidfVectorizer` and `MultinomialNB` to classify 12 training headlines, then evaluate generalization using cross-validation. TF-IDF (term frequency–inverse document frequency) weights important, distinctive words more heavily than common words across all articles.',
            pseudoCode: 'Import Pipeline, TfidfVectorizer, MultinomialNB, cross_val_score\nheadlines = [12 news headline strings]\ncategories = [0=Technology, 1=Sports, 2=Politics]\npipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())])\nscores = cross_val_score(pipeline, headlines, categories, cv=3)\nprint(f"Mean Accuracy: {scores.mean():.2f}")',
            starterCode: `from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

# News headline dataset: 0=Technology, 1=Sports, 2=Politics
headlines = [
    "new smartphone chip breaks performance records",       # Technology
    "open source AI framework launches developer preview", # Technology
    "electric vehicle battery range reaches new milestone",# Technology
    "cloud computing costs surge for small businesses",    # Technology
    "championship team wins national soccer title",        # Sports
    "marathon runner breaks world record in berlin",       # Sports
    "basketball trade shakes up the entire league",        # Sports
    "olympic swimmer wins gold at world aquatics meet",    # Sports
    "senate passes new infrastructure spending bill",      # Politics
    "election results overturn longtime incumbent",        # Politics
    "government announces tax reform policy update",       # Politics
    "diplomatic summit addresses global climate deal",     # Politics
]
categories = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

# TODO: Create a Pipeline with TfidfVectorizer and MultinomialNB
# TODO: Evaluate using cross_val_score with cv=3
# TODO: Print the mean accuracy formatted to 2 decimal places
scores = [0.0]
print(f"Mean Accuracy: {scores[0]:.2f}")`,
            expectedOutput: 'Mean Accuracy: 1.00',
            solution: `from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

headlines = [
    "new smartphone chip breaks performance records",
    "open source AI framework launches developer preview",
    "electric vehicle battery range reaches new milestone",
    "cloud computing costs surge for small businesses",
    "championship team wins national soccer title",
    "marathon runner breaks world record in berlin",
    "basketball trade shakes up the entire league",
    "olympic swimmer wins gold at world aquatics meet",
    "senate passes new infrastructure spending bill",
    "election results overturn longtime incumbent",
    "government announces tax reform policy update",
    "diplomatic summit addresses global climate deal",
]
categories = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

scores = cross_val_score(pipeline, headlines, categories, cv=3)
print(f"Mean Accuracy: {scores.mean():.2f}")`,
            hints: ['A Pipeline takes a list of (name, estimator) tuples.', 'TfidfVectorizer replaces CountVectorizer — no separate fit/transform calls needed inside the pipeline.', 'Use cross_val_score and call .mean() on the result.'],
            testKeywords: ['Pipeline', 'TfidfVectorizer', 'cross_val_score', 'mean']
        }
    }
};
