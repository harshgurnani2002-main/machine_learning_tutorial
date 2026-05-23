import type { MLModule } from '../../types';

export const qLearningRl: MLModule = {
    id: 'q-learning-rl',
    title: 'Q-Learning',
    category: 'Advanced & MLOps',
    description: 'Train agents to navigate environments using state-action value iterations.',
    formula: 'Q(s,a) \\leftarrow Q(s,a) + \\alpha [r + \\gamma \\max Q(s\',a\') - Q(s,a)]',
    theory: `### Q-Learning & Temporal Difference Updates

#### What is it?
Q-learning is a prominent model-free reinforcement learning algorithm. Its primary goal is to learn the value of an action in a particular state, enabling an agent to make optimal decisions over time. 
"Model-free" means the agent does not need to know or learn the underlying transition probabilities and reward mechanisms of the environment; it learns strictly by interacting, observing results, and adapting.
The algorithm maintains a mapping, called a Q-table, that records the perceived quality (the 'Q' value) of taking a specific action in a specific state. Over repeated episodes, this table converges to the optimal action-value function.

#### Why do we need it?
In many real-world scenarios—like robotics, self-driving cars, or complex game environments—an agent must learn how to behave interactively rather than relying on a hardcoded set of rules or a perfect mathematical model of the world.
Q-learning provides a robust, mathematical way for an agent to learn an optimal policy entirely by trial and error. It aims to maximize the cumulative future rewards, ensuring the agent doesn\'t just act greedily for immediate payoffs but plans for long-term success. It forms the conceptual and algorithmic foundation for modern Deep Reinforcement Learning, including Deep Q-Networks (DQN).

#### How does it work?
Q-learning relies on an agent interacting with a Markov Decision Process (MDP).
1. The agent observes its current state $s$.
2. It chooses an action $a$ based on its current policy (often an $\\epsilon$-greedy strategy, which balances exploring new actions with exploiting known good actions).
3. The agent executes action $a$, receives an immediate reward $r$, and transitions to a new state $s'$.
4. The agent updates its Q-table for the pair $(s, a)$ using the Temporal Difference (TD) error, incorporating the reward and the maximum expected future reward from the new state $s'$.

Over time, through thousands of interactions, the Q-values converge to the optimal action-value function $Q^*(s, a)$. The optimal policy is then simply to always choose the action with the highest Q-value in any given state.

#### The Math Behind It
The core of Q-learning is based on the Bellman Optimality Equation, which defines the optimal action-value recursively:
$$ Q^*(s, a) = \\mathbb{E}[R(s, a) + \\gamma \\max_{a'} Q^*(s', a')] $$

In practice, the Q-table is updated iteratively using the Temporal Difference (TD) target:
$$ Q(s, a) \\leftarrow Q(s, a) + \\alpha \\left[ r + \\gamma \\max_{a'} Q(s', a') - Q(s, a) \\right] $$

Breaking down the update rule:
- **$\\alpha$ (Learning Rate):** Determines to what extent newly acquired information overrides old information (0 = learn nothing, 1 = consider only the most recent information).
- **$\\gamma$ (Discount Factor):** Determines the importance of future rewards. A factor of 0 makes the agent myopic (only considering current rewards), while a factor approaching 1 makes it strive for long-term high returns.
- **$r$ (Reward):** The immediate reward received after taking action $a$ in state $s$.
- **$\\max_{a'} Q(s', a')$:** The maximum estimated Q-value for the next state $s'$. The agent assumes it will act optimally from $s'$ onwards.
- **The term in brackets $[ ... ]$** is the Temporal Difference (TD) Error.

#### Worked Example
Consider an agent navigating a grid.
1. The agent is in state $s_1$ and takes action $a_1$ (move Right).
2. It receives a reward $r = 10$ and arrives at state $s_2$. 
3. Assume hyperparameters are $\\alpha = 0.1$, $\\gamma = 0.9$. 
4. The agent checks its table: Current $Q(s_1, a_1) = 5$.
5. The agent checks the table for the new state $s_2$. The maximum Q-value for any action in $s_2$ is $\\max Q(s_2, a') = 8$.
6. Calculate TD Target = $10 + 0.9 \\times 8 = 17.2$.
7. Calculate TD Error = $17.2 - 5 = 12.2$.
8. Update Q-value: $Q(s_1, a_1) = 5 + 0.1 \\times 12.2 = 6.22$.

#### Common Pitfalls
- **Exploration vs. Exploitation Dilemma:** If the agent exploits known rewards too early, it will get stuck in suboptimal policies. An $\\epsilon$-greedy strategy is required to force exploration, decaying the randomness over time.
- **Curse of Dimensionality:** Standard tabular Q-learning allocates memory for every unique state-action pair. If the state space is continuous or extremely large (like chess board configurations), the table becomes too vast to store or train effectively.
- **Overestimation Bias:** Because the update equation uses the $\\max$ operator over noisy Q-value estimates, Q-learning often systematically overestimates the value of states. This is addressed by extensions like Double Q-Learning.

#### When to Use vs Not Use
- **Use Q-Learning when:** 
  - The environment can be modeled as a discrete Markov Decision Process (MDP).
  - The state space and action space are small enough to fit in a table.
  - You do not have an explicit model of the environment's transition dynamics.
- **Do Not Use Q-Learning when:**
  - The state space is highly dimensional or continuous (use Deep Q-Networks instead).
  - The action space is continuous, like steering angles or motor torques (use algorithms like DDPG, SAC, or PPO).

#### Key Takeaways
1. Q-learning is a model-free, off-policy reinforcement learning algorithm.
2. It learns an action-value function mapping states and actions to expected future rewards.
3. It utilizes the Bellman equation and Temporal Difference updates.
4. It requires a delicate balance of exploration and exploitation to discover optimal policies.
5. While limited by discrete state spaces, it forms the theoretical foundation for advanced Deep RL techniques.`,
    interactiveSummary: 'This interactive simulation demonstrates a grid-world environment where an agent learns to navigate to a goal while avoiding obstacles. You can adjust the learning rate, discount factor, and epsilon decay to observe how the agent balances exploration and exploitation. The Q-table is visualized as a heatmap overlaying the grid, showing how value slowly propagates backward from the goal.',
    simulatorId: 'q-learning',
    quiz: [
      { id: 'ql_q1', question: 'What is the purpose of the discount factor gamma in the Q-learning update equation?', options: ['To balance the value of immediate rewards against future rewards.', 'To set the step size of parameter updates.', 'To scale down feature matrices.', 'To regularize weights.'], correctAnswer: 'To balance the value of immediate rewards against future rewards.', explanation: 'Gamma (value between 0 and 1) discounts future rewards, prioritizing immediate rewards (low gamma) or long-term goals (high gamma).' },
      { id: 'ql_q2', question: 'Which term describes Q-learning\'s ability to learn an optimal policy independently of the agent\'s actions?', options: ['Off-policy', 'On-policy', 'Model-based', 'Supervised'], correctAnswer: 'Off-policy', explanation: 'Q-learning is off-policy because it learns the optimal policy regardless of the policy being followed during exploration.' },
      { id: 'ql_q3', question: 'What does the Q in Q-learning stand for?', options: ['Quality', 'Quantity', 'Quotient', 'Query'], correctAnswer: 'Quality', explanation: 'Q stands for Quality, representing how useful a given action is in gaining some future reward.' },
      { id: 'ql_q4', question: 'How does an epsilon-greedy strategy work?', options: ['It chooses a random action with probability epsilon, otherwise the best known action.', 'It always chooses the best action.', 'It only explores.', 'It chooses actions based on a softmax distribution.'], correctAnswer: 'It chooses a random action with probability epsilon, otherwise the best known action.', explanation: 'Epsilon-greedy ensures exploration by taking random actions a fraction (epsilon) of the time.' },
      { id: 'ql_q5', question: 'What happens if the learning rate (alpha) is set to 0?', options: ['The agent learns nothing.', 'The agent learns instantly.', 'The agent only exploits.', 'The agent forgets everything.'], correctAnswer: 'The agent learns nothing.', explanation: 'An alpha of 0 means the Q-values are never updated with new information.' },
      { id: 'ql_q6', question: 'What is a major limitation of tabular Q-learning?', options: ['It cannot handle continuous state spaces natively.', 'It requires a perfect environment model.', 'It only works for deterministic environments.', 'It cannot use a discount factor.'], correctAnswer: 'It cannot handle continuous state spaces natively.', explanation: 'Tabular Q-learning requires discrete states to populate the Q-table. Continuous states require discretization or function approximation.' },
      { id: 'ql_q7', question: 'Which algorithm combines Q-learning with deep neural networks?', options: ['Deep Q-Networks (DQN)', 'Support Vector Machines', 'K-Means', 'Principal Component Analysis'], correctAnswer: 'Deep Q-Networks (DQN)', explanation: 'DQN uses neural networks to approximate the Q-value function, solving the curse of dimensionality in tabular Q-learning.' },
      { id: 'ql_q8', question: 'What is the "TD Target" in the Q-learning update?', options: ['reward + gamma * max(Q(next_state, all_actions))', 'reward * gamma', 'alpha * reward', 'max(Q) - current(Q)'], correctAnswer: 'reward + gamma * max(Q(next_state, all_actions))', explanation: 'The Temporal Difference (TD) target is the immediate reward plus the discounted maximum expected future reward.' },
      { id: 'ql_q9', question: 'Why might Q-learning overestimate Q-values?', options: ['Because it takes the maximum over estimated values, which includes positive noise.', 'Because the learning rate is too high.', 'Because epsilon is too low.', 'Because the rewards are negative.'], correctAnswer: 'Because it takes the maximum over estimated values, which includes positive noise.', explanation: 'The max operator in the TD target can systematically overestimate values if the estimates are noisy.' },
      { id: 'ql_q10', question: 'If gamma is set to 0, what does the agent focus on?', options: ['Only immediate rewards.', 'Only long-term rewards.', 'Exploring new states.', 'Minimizing the learning rate.'], correctAnswer: 'Only immediate rewards.', explanation: 'A discount factor of 0 zeros out all future expectations, making the agent completely myopic.' }
    ],
    coding: {
      tutorial: {
        title: 'Temporal Difference Target',
        description: 'Calculate the Temporal Difference target: reward + gamma * maxQ(s\', a\'). Note: While Q-learning is a Reinforcement Learning algorithm, in supervised contexts like scikit-learn, predicting the Q-value could be framed as a regression task (e.g. using `sklearn.neural_network.MLPRegressor` to approximate the Q-function).',
        pseudoCode: `function td_target(reward, gamma, next_state_q_values):
    max_next_q = find_maximum(next_state_q_values)
    return reward + gamma * max_next_q`,
        starterCode: `import numpy as np

def td_target(reward, gamma, next_state_q):
    # TODO: return reward + gamma * max(next_state_q)
    return 0.0

next_q = np.array([1.5, 3.0, 0.5])
print(td_target(1.0, 0.9, next_q))`,
        expectedOutput: '3.7',
        solution: `import numpy as np

def td_target(reward, gamma, next_state_q):
    return reward + gamma * np.max(next_state_q)

next_q = np.array([1.5, 3.0, 0.5])
print(td_target(1.0, 0.9, next_q))`,
        hints: ['Find max value of next_state_q with np.max and scale by gamma.'],
        testKeywords: ['np.max']
      },
      project: {
        title: 'Tabular Q-Learning Step',
        description: 'Implement a single Q-table update for a discrete grid environment using the Temporal Difference equation.',
        pseudoCode: `function update_q_table(Q, state, action, reward, next_state, alpha, gamma):
    max_next = max(Q[next_state])
    td_target = reward + gamma * max_next
    Q[state, action] = Q[state, action] + alpha * (td_target - Q[state, action])
    return Q`,
        starterCode: `import numpy as np

# Q is a matrix of 5 states and 2 actions
Q = np.zeros((5, 2))
alpha = 0.1
gamma = 0.9
reward = 10.0
state = 0
action = 1
next_state = 2

# Assume we discovered the next state has some existing values
Q[next_state] = [2.0, 5.0]

# TODO: Calculate td_target
# TODO: Update Q[state, action]

td_target = 0.0
# Q[state, action] = ...

print(round(Q[state, action], 2))`,
        expectedOutput: '1.45',
        solution: `import numpy as np

Q = np.zeros((5, 2))
alpha = 0.1
gamma = 0.9
reward = 10.0
state = 0
action = 1
next_state = 2

Q[next_state] = [2.0, 5.0]

td_target = reward + gamma * np.max(Q[next_state])
Q[state, action] = Q[state, action] + alpha * (td_target - Q[state, action])

print(round(Q[state, action], 2))`,
        hints: ['Use np.max(Q[next_state]) to find the highest future reward', 'Follow the formula: Q_old + alpha * (Target - Q_old)'],
        testKeywords: ['np.max', 'alpha', 'gamma']
      },
      assignment: {
        title: 'Deep Q-Network (DQN) Loss',
        description: 'In DQN, neural networks replace the tabular Q-values. Compute the Mean Squared Error loss between predicted Q-values and the TD targets using PyTorch.',
        pseudoCode: `function dqn_loss(current_q, target_q):
    return mean_squared_error(current_q, target_q)`,
        starterCode: `import torch
import torch.nn.functional as F

# Simulated outputs from a Q-Network and a Target-Network
current_q = torch.tensor([[2.0]])
target_q = torch.tensor([[3.0]])

# TODO: Calculate Mean Squared Error loss between current_q and target_q
loss = 0.0

print(round(float(loss), 1))`,
        expectedOutput: '1.0',
        solution: `import torch
import torch.nn.functional as F

current_q = torch.tensor([[2.0]])
target_q = torch.tensor([[3.0]])

loss = F.mse_loss(current_q, target_q)

print(round(float(loss.item()), 1))`,
        hints: ['Use F.mse_loss() from torch.nn.functional'],
        testKeywords: ['F.mse_loss']
      }
    },
    interviewQuestions: [
      { question: 'What is the exploration-exploitation dilemma in Reinforcement Learning?', answer: 'It is the trade-off between choosing known actions that yield high reward (exploitation) versus trying new actions to discover potentially higher rewards (exploration).', companyTags: ['Google', 'DeepMind'], difficulty: 'Medium' },
      { question: 'How does epsilon-greedy decay address the exploration-exploitation dilemma?', answer: 'Epsilon-greedy explores with probability epsilon. Decaying epsilon over time allows the agent to explore heavily early on, discovering the environment, and shift to exploitation as its Q-value estimates become accurate.', companyTags: ['Tesla', 'Meta'], difficulty: 'Medium' },
      { question: 'What is the Bellman Equation?', answer: 'The Bellman Equation recursively defines the value of a state or action in terms of the immediate reward and the discounted value of the next state, serving as the foundation for RL algorithms.', companyTags: ['Amazon', 'Google'], difficulty: 'Hard' },
      { question: 'Why is Q-Learning considered an "off-policy" algorithm?', answer: 'Because it learns the value of the optimal policy (by taking the max over next actions) independently of the agent\'s behavior policy (which might be epsilon-greedy).', companyTags: ['DeepMind', 'OpenAI'], difficulty: 'Hard' },
      { question: 'What happens if the learning rate (alpha) is 1 in a deterministic environment?', answer: 'The Q-value will immediately update to the new target without retaining any of its previous estimate, which is optimal for fully deterministic environments but unstable in stochastic ones.', companyTags: ['Microsoft', 'Apple'], difficulty: 'Medium' },
      { question: 'What is the curse of dimensionality in tabular Q-learning?', answer: 'As the number of state variables increases, the size of the state space (and the Q-table) grows exponentially, making tabular representations computationally infeasible and memory-intensive.', companyTags: ['Netflix', 'Uber'], difficulty: 'Medium' },
      { question: 'How does Deep Q-Learning (DQN) solve the state space explosion?', answer: 'DQN replaces the Q-table with a neural network that approximates the Q-value function, generalizing across similar states instead of storing every state-action pair.', companyTags: ['Tesla', 'OpenAI'], difficulty: 'Hard' },
      { question: 'What is experience replay in DQN?', answer: 'Experience replay stores the agent\'s past experiences in a buffer and samples them randomly during training. This breaks the temporal correlation of sequential data and stabilizes neural network training.', companyTags: ['Google', 'Meta'], difficulty: 'Hard' },
      { question: 'Why do we use a target network in DQN?', answer: 'A target network is a separate, slowly-updating copy of the Q-network used to compute the TD target. It prevents the moving target problem, making training much more stable.', companyTags: ['OpenAI', 'DeepMind'], difficulty: 'Hard' },
      { question: 'How is Double Q-Learning different from standard Q-Learning?', answer: 'Standard Q-learning overestimates Q-values due to the max operator. Double Q-learning uses two separate estimators: one to select the best action and another to evaluate its value, mitigating the overestimation bias.', companyTags: ['Google', 'Uber'], difficulty: 'Advanced' },
      { question: 'What is the role of the discount factor (gamma)?', answer: 'Gamma determines the importance of future rewards. A gamma of 0 makes the agent myopic (only caring about immediate rewards), while a gamma near 1 makes it strive for long-term high returns.', companyTags: ['Amazon', 'Apple'], difficulty: 'Easy' },
      { question: 'Can Q-learning be used for continuous action spaces?', answer: 'No, Q-learning relies on taking the maximum over all possible actions, which is computationally intractable for continuous action spaces. Algorithms like DDPG or SAC are used instead.', companyTags: ['Tesla', 'OpenAI'], difficulty: 'Medium' },
      { question: 'What is a Markov Decision Process (MDP)?', answer: 'An MDP is a mathematical framework for modeling decision making, defined by a set of states, actions, transition probabilities, rewards, and a discount factor. It assumes the Markov property.', companyTags: ['Meta', 'Microsoft'], difficulty: 'Medium' },
      { question: 'What does the Markov property state?', answer: 'The Markov property states that the future dynamics of the environment depend only on the current state and action, completely independent of the history of past states and actions.', companyTags: ['Google', 'DeepMind'], difficulty: 'Medium' },
      { question: 'What is the difference between model-free and model-based RL?', answer: 'Model-free RL (like Q-learning) learns policies directly from trial and error without predicting the environment dynamics. Model-based RL attempts to learn a transition model of the environment to plan ahead.', companyTags: ['Uber', 'Amazon'], difficulty: 'Hard' },
      { question: 'What is an episode in Reinforcement Learning?', answer: 'An episode is a complete sequence of interactions from an initial state to a terminal state (e.g., winning/losing a game or reaching a goal).', companyTags: ['Netflix', 'Apple'], difficulty: 'Easy' },
      { question: 'How do sparse rewards affect Q-learning?', answer: 'Sparse rewards make it difficult for the agent to learn because it rarely receives positive feedback, causing random exploration to take an impractically long time to stumble upon the goal.', companyTags: ['Google', 'Tesla'], difficulty: 'Medium' },
      { question: 'What is Reward Shaping?', answer: 'Reward shaping involves modifying the reward function to provide intermediate hints or heuristics to the agent, helping guide it towards the ultimate sparse goal.', companyTags: ['Meta', 'OpenAI'], difficulty: 'Medium' },
      { question: 'What is a Policy in RL?', answer: 'A policy is the strategy the agent employs to determine the next action based on the current state. It maps states to actions.', companyTags: ['Microsoft', 'Amazon'], difficulty: 'Easy' },
      { question: 'Explain the difference between Value Iteration and Q-Learning.', answer: 'Value iteration requires a known transition model of the environment to iteratively compute state values. Q-learning is model-free and learns action values directly from experience samples.', companyTags: ['DeepMind', 'Google'], difficulty: 'Hard' }
    ]
};
