import type { MLModule } from '../../types';

export const recurrentNetworks: MLModule = {
  "id": "recurrent-networks",
  "title": "Recurrent Networks & LSTMs",
  "category": "Deep Learning",
  "description": "Process sequential observations using recurrent states and gating corridors.",
  "formula": "h_t = \tanh(W x_t + U h_{t-1} + b)",
    theory: `### Recurrent Networks & LSTMs
  
  #### What is it?
  Recurrent Neural Networks (RNNs) are a class of artificial neural networks designed to recognize patterns in sequences of data, such as text, genomes, handwriting, spoken words, or numerical time series data. Unlike traditional feedforward networks, RNNs have a "memory" that captures information about what has been calculated so far. Long Short-Term Memory (LSTM) networks are a special kind of RNN, capable of learning long-term dependencies.
  
  #### Why do we need it?
  Standard feedforward neural networks assume all inputs (and outputs) are independent of each other. However, for many tasks like language translation, speech recognition, and time series prediction, context from previous inputs is essential. If you want to predict the next word in a sentence, you need to know which words came before it. RNNs and LSTMs maintain a hidden state that carries this context, enabling them to process sequential data effectively.
  
  #### How does it work?
  An RNN processes sequences by passing a hidden state vector $h_t$ over time. At each time step $t$, the RNN cell takes the current input $x_t$ and the previous hidden state $h_{t-1}$ to produce the next hidden state. 
  Standard RNNs, however, suffer from the vanishing gradient problem, where they completely forget information from further back in the sequence. LSTMs solve this by introducing a cell state $C_t$ controlled by three gates (Forget, Input, Output). These gates act as filters, regulating the flow of information, allowing the network to add or remove information to the cell state as needed.
  
  #### The Math Behind It
  Standard RNN Cell Equation:
  The hidden state at time $t$ is a non-linear function of the previous hidden state and the current input:
  $h_t = 	anh(W_{hh} h_{t-1} + W_{xh} x_t + b)$
  
  LSTM gating solution:
  Long Short-Term Memory networks solve vanishing gradients by routing information through a constant-flow cell state $C_t$ controlled by three gates:
  - **Forget Gate**: Decides what information to discard from the cell state.
    $f_t = \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f)$
  - **Input Gate**: Decides which values to update in the cell state.
    $i_t = \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i)$
    $	ilde{C}_t = 	anh(W_c \\cdot [h_{t-1}, x_t] + b_c)$
  - **Cell State Update**: Combining the old cell state and the new candidate values.
    $C_t = f_t \\ast C_{t-1} + i_t \\ast 	ilde{C}_t$
  - **Output Gate**: Decides what to output based on the updated cell state.
    $o_t = \\sigma(W_o \\cdot [h_{t-1}, x_t] + b_o)$
    $h_t = o_t \\ast 	anh(C_t)$
  
  #### Worked Example
  Consider predicting the next word in the sentence "I love machine learning". 
  1. **Time step 1 ($t=1$)**: Input is "I" ($x_1$). Previous hidden state $h_0$ is zero. The RNN computes $h_1 = 	anh(W_{hh} h_0 + W_{xh} x_1 + b)$.
  2. **Time step 2 ($t=2$)**: Input is "love" ($x_2$). The RNN computes $h_2 = 	anh(W_{hh} h_1 + W_{xh} x_2 + b)$. 
  The state $h_2$ now contains information about "I" and "love". 
  In an LSTM, at $t=2$, the forget gate might decide to keep the fact that the subject is "I", while the input gate adds the concept of "love" to the cell state. The output gate then reveals this combined context to the hidden state $h_2$.
  
  #### Common Pitfalls
  - **Vanishing/Exploding Gradients**: Standard RNNs struggle with sequences longer than ~10 steps due to vanishing gradients (where weight updates become infinitesimally small). Always prefer LSTMs or GRUs for long sequences.
  - **Overfitting**: Recurrent networks are highly prone to overfitting because they have a large number of parameters. Regularization techniques like Dropout (specifically variational dropout for sequences) and early stopping are crucial.
  - **Slow Training**: Due to their sequential processing, RNNs cannot be parallelized over time steps, making training significantly slower than CNNs or Transformers.
  
  #### When to Use vs Not Use
  - **Use When**: You have sequential data (e.g., text, time series, audio, stock prices) where temporal order fundamentally matters and the sequence length is variable.
  - **Not Use When**: You have independent, non-sequential data (like images or tabular data). For very long text sequences where global context is needed simultaneously, Transformers perform better, handle longer context windows, and train faster due to parallelization.
  
  #### Key Takeaways
  1. RNNs are designed for sequence processing, handling variable-length inputs by maintaining a hidden state.
  2. The vanishing gradient problem hinders standard RNNs from learning long-term dependencies because gradients shrink during Backpropagation Through Time.
  3. LSTMs mitigate this using a dedicated cell state and gating mechanisms (Forget, Input, Output gates) to carefully regulate memory.
  4. While powerful, RNNs and LSTMs suffer from slow training speeds because they must process data sequentially.
#### Python Implementation

\`\`\`python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(64, input_shape=(10, 32), return_sequences=False),
    Dense(1, activation="sigmoid")
])
model.summary()
\`\`\`
`,
  "interactiveSummary": "This simulator visualizes how sequential data flows through an RNN one time step at a time, with animated arrows showing the hidden state vector h_t being passed forward from each step to the next. You can scrub through individual time steps to watch how the hidden state evolves as it absorbs new input tokens, and toggle the LSTM gate panel to see the forget, input, and output gate activations as colored bar charts. The cell state conveyor belt animation illustrates how long-term information persists without vanishing across many steps, in contrast to the standard RNN which gradually loses early context. Adjust the forget gate slider to zero out memory and observe how the model's predictions degrade on long sequences, reinforcing why gated architectures are critical for real-world sequential tasks like stock forecasting and sentiment analysis.",
  "simulatorId": "rnn",
  "quiz": [
    {
      "id": "rnn_q1",
      "question": "Which gate controls how much of the past cell state memory is discarded in an LSTM cell?",
      "options": [
        "Forget gate",
        "Input gate",
        "Output gate",
        "Update gate"
      ],
      "correctAnswer": "Forget gate",
      "explanation": "The forget gate outputs a value between 0 and 1, scaling the past cell state to discard or retain historical inputs."
    },
    {
      "id": "rnn_q2",
      "question": "What activation function is typically used for the gates in an LSTM?",
      "options": [
        "Sigmoid",
        "ReLU",
        "Tanh",
        "Linear"
      ],
      "correctAnswer": "Sigmoid",
      "explanation": "Sigmoid outputs values between 0 and 1, acting as a valve to let information through or block it."
    },
    {
      "id": "rnn_q3",
      "question": "What is the primary purpose of the cell state in an LSTM?",
      "options": [
        "To carry long-term dependencies across time steps",
        "To apply non-linear transformations",
        "To reduce dimensionality",
        "To regularize the network"
      ],
      "correctAnswer": "To carry long-term dependencies across time steps",
      "explanation": "The cell state acts as a conveyor belt, allowing information to flow unchanged down the sequence, mitigating vanishing gradients."
    },
    {
      "id": "rnn_q4",
      "question": "In a standard RNN, why do gradients vanish?",
      "options": [
        "Repeated multiplication of weight matrices during backpropagation through time",
        "Use of the ReLU activation function",
        "Insufficient training data",
        "High learning rates"
      ],
      "correctAnswer": "Repeated multiplication of weight matrices during backpropagation through time",
      "explanation": "Backpropagation Through Time (BPTT) involves multiplying the same weight matrix repeatedly, causing gradients to shrink exponentially if weights are < 1."
    },
    {
      "id": "rnn_q5",
      "question": "What activation function is used to compute the candidate cell state (C_tilde) in an LSTM?",
      "options": [
        "Tanh",
        "Sigmoid",
        "ReLU",
        "Softmax"
      ],
      "correctAnswer": "Tanh",
      "explanation": "Tanh squashes values between -1 and 1, regulating the network and keeping values zero-centered."
    },
    {
      "id": "rnn_q6",
      "question": "Which RNN variant uses only two gates (Reset and Update)?",
      "options": [
        "GRU",
        "LSTM",
        "BiRNN",
        "Vanilla RNN"
      ],
      "correctAnswer": "GRU",
      "explanation": "Gated Recurrent Units (GRUs) simplify the LSTM architecture by merging the forget and input gates into a single update gate."
    },
    {
      "id": "rnn_q7",
      "question": "What problem does clipping gradients solve in RNNs?",
      "options": [
        "Exploding gradients",
        "Vanishing gradients",
        "Overfitting",
        "Underfitting"
      ],
      "correctAnswer": "Exploding gradients",
      "explanation": "Gradient clipping caps the maximum value of gradients during backpropagation, preventing them from growing too large and causing numerical instability."
    },
    {
      "id": "rnn_q8",
      "question": "How is a sequence of variable length typically handled in mini-batch training for RNNs?",
      "options": [
        "Padding shorter sequences with zeros",
        "Truncating all sequences to length 1",
        "Sorting sequences alphabetically",
        "Repeating the sequences continuously"
      ],
      "correctAnswer": "Padding shorter sequences with zeros",
      "explanation": "Padding ensures all sequences in a batch have the same length, allowing for efficient matrix operations. Masking is then used to ignore the padded values."
    },
    {
      "id": "rnn_q9",
      "question": "What does a Bidirectional RNN do?",
      "options": [
        "Processes the sequence in both forward and backward directions",
        "Trains twice as fast",
        "Predicts two outputs per time step",
        "Uses two different loss functions"
      ],
      "correctAnswer": "Processes the sequence in both forward and backward directions",
      "explanation": "BiRNNs look at both past and future context by running two RNNs (forward and backward) and concatenating their hidden states."
    },
    {
      "id": "rnn_q10",
      "question": "Which gate is responsible for filtering the cell state to produce the final hidden state output in an LSTM?",
      "options": [
        "Output gate",
        "Forget gate",
        "Input gate",
        "Update gate"
      ],
      "correctAnswer": "Output gate",
      "explanation": "The output gate determines which parts of the cell state should be output as the hidden state for the current time step."
    }
  ],
  "coding": {
    "tutorial": {
      "title": "RNN Hidden Update Step",
      "description": "Update the hidden state vector using inputs, weights, and tanh activation. Note: In real-world applications, you would typically use `sklearn.neural_network.MLPRegressor` or frameworks like PyTorch/Keras for sequence modeling. Since scikit-learn does not have a native RNN/LSTM implementation, we build this intuition using raw NumPy.",
      "pseudoCode": "function rnn_step(x, h_prev, Wx, Wh, b):\n    linear_combination = (Wx * x) + (Wh * h_prev) + b\n    h_next = tanh(linear_combination)\n    return h_next",
      "starterCode": "import numpy as np\n\ndef rnn_step(x, h_prev, Wx, Wh, b):\n    # TODO: return tanh(Wx*x + Wh*h_prev + b)\n    return h_prev\n\nx = 1.0\nh_prev = 0.5\nprint(np.round(rnn_step(x, h_prev, 0.5, 0.8, -0.1), 3))",
      "expectedOutput": "0.664",
      "solution": "import numpy as np\n\ndef rnn_step(x, h_prev, Wx, Wh, b):\n    return np.tanh(Wx * x + Wh * h_prev + b)\n\nx = 1.0\nh_prev = 0.5\nprint(np.round(rnn_step(x, h_prev, 0.5, 0.8, -0.1), 3))",
      "hints": [
        "Use np.tanh for the hyperbolic tangent activation."
      ],
      "testKeywords": [
        "np.tanh"
      ]
    },
    "project": {
      "title": "Stock Price Trend Prediction with PyTorch LSTM",
      "description": "Build a PyTorch LSTM to predict the next-day stock price direction (up/down) from 5 days of historical closing prices. Each sample is a window of 5 sequential price values. The model processes this time series and outputs a single logit predicting whether the price will rise (1) or fall (0) on day 6.",
      "pseudoCode": "# Stock Price Trend LSTM\nclass StockLSTM(nn.Module):\n    __init__:\n        lstm   = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)\n        linear = nn.Linear(16, 1)   # 1 output: probability of price rising\n    forward(x):\n        out, _ = lstm(x)            # out shape: (batch, seq_len=5, hidden=16)\n        last   = out[:, -1, :]      # take the last time step's hidden state\n        return linear(last)         # raw logit\n\nmodel = StockLSTM()\ncriterion = BCEWithLogitsLoss()\noptimizer = Adam(model.parameters(), lr=0.05)\nfor epoch in range(200):\n    optimizer.zero_grad()\n    preds = model(X)\n    loss  = criterion(preds, y)\n    loss.backward()\n    optimizer.step()\nprint prediction",
      "starterCode": "import torch\nimport torch.nn as nn\n\n# Historical 5-day closing price windows (normalised) -> next-day direction\n# X shape: (batch=8, seq_len=5, features=1)\ntorch.manual_seed(0)\nX = torch.tensor([\n    [[0.10],[0.12],[0.15],[0.14],[0.17]],  # rising trend\n    [[0.80],[0.78],[0.75],[0.73],[0.70]],  # falling trend\n    [[0.30],[0.32],[0.35],[0.37],[0.40]],  # rising\n    [[0.60],[0.58],[0.55],[0.52],[0.50]],  # falling\n    [[0.20],[0.22],[0.25],[0.28],[0.31]],  # rising\n    [[0.90],[0.87],[0.83],[0.80],[0.77]],  # falling\n    [[0.40],[0.43],[0.46],[0.49],[0.52]],  # rising\n    [[0.70],[0.67],[0.63],[0.60],[0.57]],  # falling\n])\ny = torch.tensor([[1.],[0.],[1.],[0.],[1.],[0.],[1.],[0.]])  # 1=up, 0=down\n\n# TODO: Define StockLSTM with nn.LSTM(input_size=1, hidden_size=16, batch_first=True)\n# and nn.Linear(16, 1)\nclass StockLSTM(nn.Module):\n    def __init__(self):\n        super().__init__()\n        # TODO: define lstm and linear layers\n        pass\n    def forward(self, x):\n        # TODO: pass through lstm, take last timestep out[:,-1,:], apply linear\n        return x\n\nmodel = StockLSTM()\nprint(model.__class__.__name__)",
      "expectedOutput": "StockLSTM",
      "solution": "import torch\nimport torch.nn as nn\n\ntorch.manual_seed(0)\nX = torch.tensor([\n    [[0.10],[0.12],[0.15],[0.14],[0.17]],\n    [[0.80],[0.78],[0.75],[0.73],[0.70]],\n    [[0.30],[0.32],[0.35],[0.37],[0.40]],\n    [[0.60],[0.58],[0.55],[0.52],[0.50]],\n    [[0.20],[0.22],[0.25],[0.28],[0.31]],\n    [[0.90],[0.87],[0.83],[0.80],[0.77]],\n    [[0.40],[0.43],[0.46],[0.49],[0.52]],\n    [[0.70],[0.67],[0.63],[0.60],[0.57]],\n])\ny = torch.tensor([[1.],[0.],[1.],[0.],[1.],[0.],[1.],[0.]])\n\nclass StockLSTM(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)\n        self.linear = nn.Linear(16, 1)\n\n    def forward(self, x):\n        out, _ = self.lstm(x)\n        out = out[:, -1, :]\n        return self.linear(out)\n\nmodel = StockLSTM()\nprint(model.__class__.__name__)",
      "hints": [
        "Use nn.LSTM(input_size=1, hidden_size=16, batch_first=True)",
        "Extract last time step: out[:, -1, :]",
        "nn.Linear(16, 1) maps hidden state to 1 output logit"
      ],
      "testKeywords": [
        "nn.LSTM",
        "batch_first=True",
        "out[:, -1, :]",
        "StockLSTM"
      ]
    },
    "assignment": {
      "title": "Movie Review Sentiment Analysis with PyTorch RNN",
      "description": "Build an RNN classifier to perform sentiment analysis on movie reviews. Each review is represented as a pre-computed sequence of 4 sentiment embedding vectors (dim=3). The model reads the sequence of word embeddings left-to-right and outputs a single binary label: Positive (1) or Negative (0).",
      "pseudoCode": "# Movie Review Sentiment RNN\nclass SentimentRNN(nn.Module):\n    __init__:\n        rnn     = nn.RNN(input_size=3, hidden_size=8, batch_first=True)\n        linear  = nn.Linear(8, 1)\n        sigmoid = nn.Sigmoid()\n    forward(x):\n        out, _ = rnn(x)          # out: (batch, seq=4, hidden=8)\n        last   = out[:, -1, :]   # last token's hidden state\n        return sigmoid(linear(last))\n\nmodel = SentimentRNN()\ncriterion = BCELoss()\noptimizer = Adam(model.parameters(), lr=0.05)\nfor epoch in range(100):\n    optimizer.zero_grad()\n    preds = model(X)\n    loss  = criterion(preds, y)\n    loss.backward()\n    optimizer.step()\nprint class label",
      "starterCode": "import torch\nimport torch.nn as nn\n\n# Movie review embeddings: (batch=8, seq_len=4 words, embed_dim=3)\n# Positive reviews have mostly high-value embeddings; negative ones are low\ntorch.manual_seed(5)\nX = torch.tensor([\n    [[0.9,0.8,0.7],[0.8,0.9,0.8],[0.7,0.8,0.9],[0.9,0.7,0.8]],  # positive\n    [[0.1,0.2,0.1],[0.2,0.1,0.2],[0.1,0.2,0.1],[0.2,0.1,0.2]],  # negative\n    [[0.8,0.7,0.9],[0.9,0.8,0.7],[0.8,0.9,0.8],[0.7,0.8,0.9]],  # positive\n    [[0.2,0.1,0.3],[0.1,0.3,0.2],[0.3,0.1,0.2],[0.2,0.2,0.1]],  # negative\n    [[0.7,0.9,0.8],[0.8,0.7,0.9],[0.9,0.8,0.7],[0.8,0.9,0.8]],  # positive\n    [[0.3,0.2,0.1],[0.1,0.2,0.3],[0.2,0.3,0.1],[0.3,0.1,0.2]],  # negative\n    [[0.9,0.9,0.8],[0.8,0.8,0.9],[0.9,0.8,0.8],[0.8,0.9,0.9]],  # positive\n    [[0.1,0.1,0.2],[0.2,0.1,0.1],[0.1,0.2,0.1],[0.2,0.1,0.2]],  # negative\n])\ny = torch.tensor([[1.],[0.],[1.],[0.],[1.],[0.],[1.],[0.]])\n\n# TODO: Define SentimentRNN with nn.RNN(input_size=3, hidden_size=8, batch_first=True)\n# Use nn.Linear(8, 1) and nn.Sigmoid()\nclass SentimentRNN(nn.Module):\n    def __init__(self):\n        super().__init__()\n        # TODO: define rnn, linear, sigmoid\n        pass\n    def forward(self, x):\n        # TODO: rnn -> take last timestep -> linear -> sigmoid\n        pass\n\nmodel = SentimentRNN()\n# TODO: BCELoss and Adam(lr=0.05). Train 100 epochs.\n\nwith torch.no_grad():\n    test = torch.tensor([[[0.85,0.90,0.80],[0.90,0.85,0.90],[0.80,0.90,0.85],[0.90,0.80,0.90]]])\n    pred = model(test).item()\n    print(f\"Sentiment: {'Positive' if pred > 0.5 else 'Negative'}\")\n",
      "expectedOutput": "Sentiment: Positive",
      "solution": "import torch\nimport torch.nn as nn\n\ntorch.manual_seed(5)\nX = torch.tensor([\n    [[0.9,0.8,0.7],[0.8,0.9,0.8],[0.7,0.8,0.9],[0.9,0.7,0.8]],\n    [[0.1,0.2,0.1],[0.2,0.1,0.2],[0.1,0.2,0.1],[0.2,0.1,0.2]],\n    [[0.8,0.7,0.9],[0.9,0.8,0.7],[0.8,0.9,0.8],[0.7,0.8,0.9]],\n    [[0.2,0.1,0.3],[0.1,0.3,0.2],[0.3,0.1,0.2],[0.2,0.2,0.1]],\n    [[0.7,0.9,0.8],[0.8,0.7,0.9],[0.9,0.8,0.7],[0.8,0.9,0.8]],\n    [[0.3,0.2,0.1],[0.1,0.2,0.3],[0.2,0.3,0.1],[0.3,0.1,0.2]],\n    [[0.9,0.9,0.8],[0.8,0.8,0.9],[0.9,0.8,0.8],[0.8,0.9,0.9]],\n    [[0.1,0.1,0.2],[0.2,0.1,0.1],[0.1,0.2,0.1],[0.2,0.1,0.2]],\n])\ny = torch.tensor([[1.],[0.],[1.],[0.],[1.],[0.],[1.],[0.]])\n\nclass SentimentRNN(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.rnn = nn.RNN(input_size=3, hidden_size=8, batch_first=True)\n        self.linear = nn.Linear(8, 1)\n        self.sigmoid = nn.Sigmoid()\n\n    def forward(self, x):\n        out, _ = self.rnn(x)\n        out = out[:, -1, :]\n        return self.sigmoid(self.linear(out))\n\nmodel = SentimentRNN()\ncriterion = nn.BCELoss()\noptimizer = torch.optim.Adam(model.parameters(), lr=0.05)\n\nfor epoch in range(100):\n    optimizer.zero_grad()\n    preds = model(X)\n    loss = criterion(preds, y)\n    loss.backward()\n    optimizer.step()\n\nwith torch.no_grad():\n    test = torch.tensor([[[0.85,0.90,0.80],[0.90,0.85,0.90],[0.80,0.90,0.85],[0.90,0.80,0.90]]])\n    pred = model(test).item()\n    print(f\"Sentiment: {'Positive' if pred > 0.5 else 'Negative'}\")\n",
      "hints": [
        "Use nn.RNN(input_size=3, hidden_size=8, batch_first=True)",
        "Take last timestep: out[:, -1, :]",
        "Use nn.BCELoss() since Sigmoid is applied in forward"
      ],
      "testKeywords": [
        "nn.RNN",
        "nn.BCELoss",
        "backward",
        "SentimentRNN"
      ]
    }
  },
  "interviewQuestions": [
    {
      "question": "How do LSTMs prevent vanishing gradients over long sequences compared to standard RNNs?",
      "answer": "LSTMs introduce a cell state pathway with constant error carousel updates. The forget gate determines how much information is retained. Because the cell state update is linear ($C_t = f_t \\ast C_{prev} + i_t \\ast \tilde{C}$), derivatives can flow backward without being repeatedly multiplied by weights, mitigating vanishing gradients.",
      "companyTags": [
        "Google",
        "Meta"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "What is the purpose of the Forget Gate in an LSTM?",
      "answer": "The Forget gate decides what information from the previous cell state should be discarded or kept, by outputting a number between 0 and 1 for each number in the cell state.",
      "companyTags": [
        "Amazon"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "Explain the difference between GRU and LSTM.",
      "answer": "GRUs are simpler than LSTMs. They merge the cell state and hidden state, and combine the forget and input gates into a single update gate. This makes GRUs faster to train while often achieving similar performance.",
      "companyTags": [
        "Apple",
        "Microsoft"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "What is Backpropagation Through Time (BPTT)?",
      "answer": "BPTT is the application of backpropagation to recurrent neural networks. It involves unrolling the RNN through time to form a standard feedforward network, calculating the error at each time step, and accumulating the gradients for the shared weights across all time steps.",
      "companyTags": [
        "Netflix"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "Why do we use Tanh activations in RNNs instead of ReLU?",
      "answer": "Tanh keeps hidden states bounded between -1 and 1, preventing the states from exploding during the repeated matrix multiplications over time. While ReLU can be used with careful initialization, it often leads to exploding gradients in standard RNNs.",
      "companyTags": [
        "Meta"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "What is teacher forcing in the context of training sequence-to-sequence RNNs?",
      "answer": "Teacher forcing is a training strategy where the ground truth previous output is fed as the next input to the RNN, rather than using the RNN\u2019s own generated prediction. It speeds up convergence but can cause exposure bias during inference.",
      "companyTags": [
        "Google"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "How do you handle variable-length sequences in batch training for RNNs?",
      "answer": "We pad the shorter sequences with a special token (like 0) to match the length of the longest sequence in the batch. We also use a masking mechanism so the loss function and gradients ignore the padded steps.",
      "companyTags": [
        "Amazon",
        "Tesla"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "What is an Exploding Gradient, and how do we solve it in RNNs?",
      "answer": "Exploding gradients occur when large weight updates result in unstable networks, often causing NaN losses. The standard solution is gradient clipping, where gradients exceeding a threshold are scaled down.",
      "companyTags": [
        "Apple"
      ],
      "difficulty": "Beginner"
    },
    {
      "question": "What is a Bidirectional RNN?",
      "answer": "A Bidirectional RNN consists of two independent RNNs: one processes the sequence forward, and the other backward. Their hidden states are concatenated at each time step to capture both past and future context.",
      "companyTags": [
        "Microsoft"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "Can RNNs be parallelized during training?",
      "answer": "Standard RNNs process sequences sequentially, making it difficult to parallelize computation across time steps. This is a major limitation compared to CNNs or Transformers. However, parallelization across different sequences in a batch is possible.",
      "companyTags": [
        "Meta",
        "Netflix"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "Describe the cell state in an LSTM.",
      "answer": "The cell state is the \"memory\" of the LSTM. It runs straight down the entire chain with only minor linear interactions, allowing information to flow unchanged, which is key to learning long-term dependencies.",
      "companyTags": [
        "Google"
      ],
      "difficulty": "Beginner"
    },
    {
      "question": "What is the role of the Input Gate in an LSTM?",
      "answer": "The Input gate controls which new information will be stored in the cell state. It works together with the candidate cell state (typically a tanh layer) to update the overall cell state.",
      "companyTags": [
        "Amazon"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "What is the role of the Output Gate in an LSTM?",
      "answer": "The Output gate determines what information from the cell state should be exposed as the hidden state output for the current time step.",
      "companyTags": [
        "Microsoft"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "Explain the concept of sequence-to-sequence models.",
      "answer": "Seq2Seq models map an input sequence to an output sequence (e.g., machine translation). They typically consist of an Encoder RNN that compresses the input into a context vector, and a Decoder RNN that generates the output sequence from that context.",
      "companyTags": [
        "Google",
        "Apple"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "Why might you choose a 1D CNN over an RNN for sequence modeling?",
      "answer": "1D CNNs can capture local temporal patterns, are highly parallelizable, and often train much faster than RNNs. They are useful when long-term temporal dependencies are less critical or when the receptive field is large enough to cover the required context.",
      "companyTags": [
        "Meta"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "What is an attention mechanism in the context of RNNs?",
      "answer": "Attention mechanisms allow the Decoder RNN to focus on different parts of the Encoder\u2019s hidden states at each step of output generation, rather than relying on a single static context vector. This vastly improves performance on long sequences.",
      "companyTags": [
        "Google",
        "Netflix"
      ],
      "difficulty": "Advanced"
    },
    {
      "question": "What happens if we initialize all weights of an RNN to zero?",
      "answer": "If weights are initialized to zero, all neurons will compute the same gradients during backpropagation and undergo the exact same weight updates, failing to break symmetry and preventing the network from learning complex patterns.",
      "companyTags": [
        "Amazon"
      ],
      "difficulty": "Beginner"
    },
    {
      "question": "How is the loss computed for an RNN?",
      "answer": "The loss is typically computed at each time step (e.g., cross-entropy for classification) and then summed or averaged over the entire sequence. Backpropagation Through Time is then used to update weights.",
      "companyTags": [
        "Microsoft"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "What are the main drawbacks of RNNs compared to Transformers?",
      "answer": "RNNs suffer from slow training due to their sequential nature (lack of parallelization across time) and have difficulty retaining context over very long sequences, despite LSTM improvements. Transformers solve both issues with self-attention.",
      "companyTags": [
        "Meta",
        "Tesla"
      ],
      "difficulty": "Intermediate"
    },
    {
      "question": "What is Truncated Backpropagation Through Time?",
      "answer": "Truncated BPTT is a technique for training RNNs on very long sequences. The sequence is divided into smaller chunks, and BPTT is only performed within each chunk, reducing computational cost and mitigating vanishing/exploding gradients while still retaining the hidden state between chunks.",
      "companyTags": [
        "Google",
        "Apple"
      ],
      "difficulty": "Advanced"
    }
  ]
};
