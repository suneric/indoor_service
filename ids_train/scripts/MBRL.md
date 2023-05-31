# Model-Based Reinforcement Learning

Typical deep RL algorithms require tremendous training samples, resulting in much high sample complexity, thus are hard to be directly applied in real-world tasks, where trial-and-error can be highly costly. Model-based reinforcement learning (MBRL) is on of the most important method to improve sample efficiency, and it is believed to have the great potential to make RL algorithms significantly more sample efficiency [1]. In MBRL, the environment model refers to the abstraction of the environment dynamics with the learning agents interacts. **Learning the model corresponds to recovering the state transition dynamics M and the reward function R**, in a environment formulated as a MDP <S,A,M,R,$$\gamma$$>.   
With an environment model, the agent can have the imagination ability. It can interact with the model to sample the interaction data, which is **simulation data**. Compared to the model-free reinforcement learning (MFRL) methods, where the agent can only use the data sampled from the interaction with the real environment, called **experienced data**.

## Model Learning

In MDP <S,A,M,R,$$\gamma$$>, the state transition dynamics M and the reward function R are to be learned.

### Model learning in tabular settings
At the early stage of RL research, the state and actions spaces are finite and small, the model learning is considered with the tabular MDPs.

### Model learning via prediction loss
For large-scale MDPs and MDPs with continuous state space and action space, Approximation functions are therefore employed in the general setting.

### Model learning with reduced error
To solve the major issue (horizon-qaured compounding error) due to the use of prediction loss to learn an unconstrained model.

### Model learning for complex environments dynamics
The mainstream realization of the environment dynamics model is an ensemble of Gaussian processes where the mean vector and covariance matrix for the distribution of the next state are built based on neural networks fed in the current state-action pair.
- Partial Observability, partial observable MDP, observation model p(o_t|s_t) and a latent transition model p(s_{t+1}|s_t,a_t) are learned via maximizing a posterior and the posterior distribution p(s_t|o_1,...o_t) can be inferred.
- **Representation learning**, for high-dimension state space such as images, representation learning that learns informative latent state or action representation will much benefit the environment model building so as to improve the effectiveness and sample efficiency of model-based RL.


## Reference
[1] A survey on model-based reinforcement learning
