# The History of Reinforcement Learning

If we had asked an AI researcher in the early 2010s, "What is reinforcement learning?" he would likely draw a feedback loop diagram showing an agent interacting with an environment and explain that it is primarily used for robot control and playing games. However, if we rewind the clock a century or fast-forward to today's large model era, we will discover that reinforcement learning (Reinforcement Learning, RL) has undergone a dramatic evolution — it has grown from psychologist's animal experiments into the core engine driving today's most advanced AI systems.

Before we begin our code practice, let us take a few minutes to briefly review this history spanning over a century. Understanding these milestones will help you better grasp why modern RL algorithms are designed the way they are today.

## 1. Enlightenment and Foundation: From Psychology to Mathematical Framework (1890s - 1950s)

The idea of reinforcement learning did not originate in computer science, but rather in **psychology and neuroscience**.
In 1898, psychologist Edward Thorndike conducted the famous "Cat in a Puzzle Box" experiment and proposed the **Law of Effect**: if a behavior leads to a good outcome, that behavior is reinforced; conversely, it is weakened. This is the very origin of "trial-and-error learning."

![Thorndike's Puzzle Box](../../../preface/brief-history/images/puzzle_box.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 1: The Puzzle Box designed by Thorndike. Source: <a href="https://commons.wikimedia.org/wiki/File:Original_%22Puzzle_Box%22_Apparatus_Design.png" target="_blank" rel="noopener noreferrer">Wikimedia Commons</a></em>
</div>

Over half a century later, with the rise of cybernetics, this biological instinct began to be rigorously formalized mathematically. In 1957, Richard Bellman introduced the **Markov Decision Process (MDP)** and the **Bellman Equation** [^1]. He abstracted real-world sequential decision problems into a precise mathematical object using a five-tuple $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$, where $\mathcal{S}$ is the set of states, $\mathcal{A}$ is the set of actions, $P(s'|s,a)$ is the transition probability, $R(s,a)$ is the reward function, and $\gamma$ is the discount factor. Within this framework, the agent's goal is to find a policy $\pi(a|s)$ that maximizes the expected long-term discounted cumulative reward:

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

To measure "how good a policy is," Bellman introduced the concept of the **value function** — $V^\pi(s)$ represents the expected cumulative reward obtainable from state $s$ when always following policy $\pi$. Among all possible policies, the optimal one corresponds to the **optimal value function** $V^*(s)$. Bellman proved that this function satisfies a beautiful recursive relationship — the **Bellman Optimality Equation**:

$$V^*(s) = \max_a \left[ R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) \, V^*(s') \right]$$

This equation carries profound meaning: the optimal value of the current state is equal to the "immediate reward" plus the "discounted expected optimal value of all future possible states." It transforms a seemingly infinite sequence of decision problems into a solvable equation — this is the origin of the **dynamic programming** idea. This marked the formal establishment of a solid theoretical foundation for reinforcement learning.

## 2. Theoretical Formation and Temporal Difference and Model-Free Learning (1980s - 1990s)

Although Bellman's dynamic programming is mathematically flawless, it has two critical limitations in practical applications. **First, it requires a complete model of the environment** — that is, the transition probabilities $P(s'|s,a)$ and the reward function $R(s,a)$ must be known in advance. However, in reality, a robot does not know how wide the corridor is after pushing a door, and an AI does not know where the opponent will move next in a game. **Second, it suffers from a severe "curse of dimensionality"** — the Bellman equation requires solving for each state individually, and the size of the state space grows exponentially with the complexity of the problem. For example, in Go, the number of board states is approximately $3^{361} \approx 10^{170}$, which is far beyond the capacity of the entire universe's atoms to store.

To enable agents to learn in **unknown environments** and **without relying on complete state tables**, pioneers began to seek new approaches.

- **In 1988**, Richard Sutton, hailed as the "father of reinforcement learning" (RL), systematically proposed **Temporal Difference (TD) learning** [^2]. It cleverly combined Monte Carlo sampling with the bootstrap property of dynamic programming, allowing agents to learn on the fly without a complete environment model. The core update rule of TD is extremely simple:

$$V(s_t) \leftarrow V(s_t) + \alpha \left[ \underbrace{r_{t+1} + \gamma V(s_{t+1}) - V(s_t)}_{\text{TD error } \delta_t} \right]$$

Here, $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ is called the **TD error**. Intuitively, it measures the discrepancy between the "new estimate" and the "old estimate" — if the situation is better than expected after taking the next step ($\delta_t > 0$), the value of the current state is increased; otherwise, it is decreased. This "learning as you go" mechanism is one of the core ideas of modern RL.

- In **1989**, Chris Watkins introduced **Q-learning** in his doctoral dissertation [^3]. This model-free, off-policy algorithm remains one of the standard starting points for reinforcement learning. Its update rule is

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$

Q-learning directly estimates the **action-value function** $Q(s,a)$: the expected return of taking action $a$ in state $s$ and then continuing from the next state. Once this table has converged, the agent can act greedily by selecting $\arg\max_a Q(s,a)$ in each state.

- In **1992**, IBM researcher Gerald Tesauro developed **TD-Gammon** [^4]. By combining TD learning with a shallow neural network, the program reached a level comparable to the world's strongest backgammon players. It became an early demonstration of reinforcement learning with neural function approximation.

![TD-Gammon / Backgammon](../../../preface/brief-history/images/backgammon.jpg)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 2: Backgammon, the classic game that TD-Gammon conquered. Source: <a href="https://commons.wikimedia.org/wiki/File:Backgammon_lg.jpg" target="_blank" rel="noopener noreferrer">Wikimedia Commons</a></em>
</div>

In 1998, Sutton and Barto published the influential classic textbook _Reinforcement Learning: An Introduction_ [^5], marking the formal establishment of the theoretical framework for modern reinforcement learning.

## 3. Deep Revolution and When RL Meets Deep Learning (2013 - 2019)

After the turn of the 21st century, although the theory of reinforcement learning continued to mature, traditional table-based methods and linear function approximations were fundamentally incapable of handling the high-dimensional and complex inputs (such as images) present in the real world. It was not until the breakthrough of deep learning that reinforcement learning truly entered its "golden age."

- **2013**, DeepMind introduced the **Deep Q-Network (DQN)** [^6], which for the first time seamlessly integrated deep neural networks with reinforcement learning, enabling AI to learn and surpass human performance in multiple Atari arcade games by merely observing screen pixels. This marked the official beginning of the era of deep reinforcement learning. The core idea of DQN is to use a neural network $ Q(s,a;\theta) $ with parameters $\theta$ to approximate the Q-value function. Its loss function is defined as:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^{-}) - Q(s, a; \theta) \right)^2 \right]$$

Here, $\theta^{-}$ denotes the parameters of the **target network** (which is periodically copied from $\theta$ rather than being updated at every step), and $\mathcal{D}$ represents the **experience replay buffer** (Experience Replay Buffer). These two seemingly simple engineering techniques—target networks and experience replay—completely resolved the training instability issues that arose when combining deep networks with Q-learning, and they were key to the success of DQN.

![DQN Atari Performance](../../../preface/brief-history/images/dqn_atari.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 3: DQN's performance across dozens of Atari games, surpassing human expert performance in most cases. Source: <a href="https://research.google/blog/from-pixels-to-actions-human-level-control-through-deep-reinforcement-learning/" target="_blank" rel="noopener noreferrer">Google Research Blog</a></em>
</div>

- **2016**, a year destined to be etched into history. DeepMind's **AlphaGo** [^7] combined deep reinforcement learning with Monte Carlo tree search to defeat the Go world champion Lee Sedol with a score of 4:1. This event not only shocked the world but also brought RL into the public eye for the first time in an extremely impactful manner.

![AlphaGo](../../../preface/brief-history/images/alphago.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 4: Screenshot of AlphaGo's game against European Go champion Fan Hui. Source: <a href="https://commons.wikimedia.org/wiki/File:AlphaGo_Fan_Huiren_aurka.png" target="_blank" rel="noopener noreferrer">Wikimedia Commons</a></em>
</div>

- **In 2017**, OpenAI proposed the **PPO (Proximal Policy Optimization, Proximal Policy Optimization)** algorithm [^8]. Compared to the high variance and fragility of earlier policy gradient methods, PPO found an excellent balance between training stability and sampling efficiency. Its core idea is to restrict the magnitude of each policy update through **clipping**, preventing "overstepping" that could lead to training collapse:

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_t, \; \text{clip}\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_t \right) \right]$$

Here, $\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}$ represents the **ratio of new to old policy probabilities**, $\hat{A}_t$ is the **estimate of the advantage function**, and $\epsilon$ is typically set to 0.1~0.2. The clipping mechanism ensures that the policy does not deviate too far from the old policy after each update — this is akin to adding a "safety barrier" to the learning rate. Due to its ease of tuning and excellent robustness, PPO quickly became the de facto standard algorithm in industry. Subsequently, OpenAI used a large-scale distributed system based on PPO, **OpenAI Five**, to defeat the world champion team in DOTA 2.

## 4. The Age of Large Models and New Paradigms in Alignment and Reasoning (2020s to Present)

Just as people were beginning to think that the application scope of reinforcement learning (RL) was mainly limited to games and robot control, the rise of large language models (LLMs) has given RL a new mission — **alignment** and **reasoning**.

- **In 2022**, OpenAI released ChatGPT. Behind this achievement was the **RLHF (Reinforcement Learning from Human Feedback)** [^9]. By training a reward model to simulate human preferences, and then using the PPO algorithm to optimize the language model, RL successfully transformed LLMs from "statistical machines that can respond to prompts" into "intelligent assistants that know when to speak and when to hold back." The training of RLHF consists of two steps: first, training a reward model $ r*\phi(x, y) $ using human preference data, and then using this reward model as the signal to optimize the language model's policy $ \pi*\theta $ with PPO:

$$\max_\theta \; \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) - \beta \, \text{KL}\left(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)\right) \right]$$

The KL divergence penalty term $ \beta \, \text{KL}(\pi*\theta \| \pi*{\text{ref}}) $ ensures that the model does not deviate too far from its original behavior in pursuit of high rewards — this is a key constraint in RLHF to prevent **reward hacking**.

![Example of Early ChatGPT Interface](../../../preface/brief-history/images/chatgpt.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 5: Example of the early interface of ChatGPT. The release of ChatGPT in 2022 brought RLHF from research papers on large model post-training to real-world products, marking the beginning of reinforcement learning's entry into the alignment and reasoning of large models. Source: OpenAI <a href="https://openai.com/index/chatgpt/" target="_blank" rel="noopener noreferrer">Introducing ChatGPT</a></em>
</div>

- **In 2023**, Stanford University and others proposed **DPO (Direct Preference Optimization)** [^10]. Researchers found that the cumbersome training of reward models could be bypassed, and a simple classification loss function could be used to fine-tune a language model directly on human preference data. The loss function of DPO is directly derived from the objective of RLHF:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

where $y_w$ (winner) and $y_l$ (loser) are the human-labeled "good answer" and "bad answer," respectively, and $\sigma$ is the sigmoid function. This formula elegantly eliminates the implicit reward model in RLHF — the model simply needs to learn that "the probability of a good answer increases relatively, and the probability of a bad answer decreases relatively." DPO significantly lowers the engineering barriers of RLHF and quickly swept through the open-source community.

- From **2024 to 2025**, reasoning models such as OpenAI o1 and DeepSeek-R1 [^11] brought reinforcement learning back to the center of model training. DeepSeek-R1-Zero showed that, on tasks with objective checks such as mathematical correctness or code execution, a strong base model can develop long reasoning traces through RL without an initial SFT stage. Its **GRPO (Group Relative Policy Optimization)** algorithm removes the critic network used by PPO and estimates advantages from the relative rewards of several responses to the same prompt. For a prompt $q$, it samples $\{o_1, o_2, \ldots, o_G\}$ and normalizes their rewards:

$$\tilde{r}_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}$$

Then directly optimize the policy using the clipped objective:

$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_q \left[ \frac{1}{G} \sum_{i=1}^{G} \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} \tilde{r}_i, \; \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) \tilde{r}_i \right) \right]$$

This lightweight architecture does not require an additional Critic network, and instead uses the **relative ranking between the same set of responses** to drive learning, making it feasible to perform pure RL reasoning at scale on large clusters.

## 5. Industrial Explosion with the GRPO Family, Reasoning Models, and Agents (2025 - 2026)

If 2024 was the period of conceptual popularization of RLHF and GRPO, then the years 2025 to 2026 mark the stage where RL truly enters an industrial explosion. Three things happen simultaneously: **rapid evolution of the GRPO algorithm family**, **reasoning models becoming an independent product category**, and **Agentic RL entering production**.

### 5.1 GRPO Improvement Family and Four Independent Evolutionary Paths

After the R1 paper, open-source teams and industrial laboratories proposed several GRPO variants, each addressing a different training failure mode:

- **DAPO** (ByteDance and Tsinghua University, March 2025, [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)) introduced asymmetric clipping, dynamic sampling, token-level loss, and overlong-sample filtering. These changes target excessive reasoning length and inefficient sampling in R1-Zero-style training.
- **Dr.GRPO** (Liu et al., 2025, [arXiv:2503.20783](https://arxiv.org/abs/2503.20783)) showed that standard-deviation and length normalization can bias the update. Removing them produced more stable training in the reported experiments.
- **GSPO** (Zheng et al., Qwen3 team, July 2025, [arXiv:2507.18071](https://arxiv.org/abs/2507.18071)) moved the importance-sampling ratio from individual tokens to the whole sequence, improving the stability of RL training for MoE models.
- **CISPO** (MiniMax, June 2025, [arXiv:2506.13585](https://arxiv.org/abs/2506.13585)) clips importance-sampling weights rather than discarding token updates, preserving gradient contributions from more tokens.
- **VAPO** (ByteDance Seed, April 2025, [arXiv:2504.05118](https://arxiv.org/abs/2504.05118)) reintroduced a value model and showed that a critic can still be useful for long chain-of-thought reasoning.

By early 2026, the question of "which GRPO variant to use" has transformed from an open question into a selection decision table.

### 5.2 Reasoning Models and Formal RL

OpenAI's o1, o3, and o4 series established **test-time compute scaling** as another way to improve model performance. _Competitive Programming with Large Reasoning Models_ ([arXiv:2502.06807](https://arxiv.org/abs/2502.06807)) showed that complex test-time strategies can emerge from end-to-end reinforcement learning rather than being specified by hand.

At the same time, DeepMind's **AlphaProof** and **AlphaGeometry 2** reached silver-medal performance at the 2024 International Mathematical Olympiad. Their use of formal languages and search connected reinforcement learning with machine-checkable proofs. DeepSeek-Prover-V2 ([arXiv:2504.21801](https://arxiv.org/abs/2504.21801)) continued this direction with Lean 4. A proof assistant serves as a strict verifier: a proof either passes the checker or it does not.

### 5.3 Agentic RL Enters Production

Another development was the expansion of RL from single-turn question answering to long-horizon tasks:

- **Meta's SWE-RL** ([arXiv:2502.18449](https://arxiv.org/abs/2502.18449)) trained a model on software-evolution data and evaluated it on repository-level issue resolution.
- **Claude Computer Use** and **OpenAI Operator** moved model actions into browsers and desktop interfaces.
- ByteDance's **UI-TARS-2** ([arXiv:2509.02544](https://arxiv.org/abs/2509.02544)) and Zhipu's **AutoGLM** explored multi-turn GUI interaction and asynchronous rollout systems.

### 5.4 The Rise of Chinese Laboratories

Chinese laboratories have taken on a unique position in this wave of RL industrialization. **DeepSeek has the highest transparency**—it publicly disclosed that the V3 pre-training used 2.664M H800 GPU hours, and R1-Zero used 128K GPU hours.

[Stanford CRFM Transparency Report](https://crfm.stanford.edu/fmti/)

**Qwen3 adopts GSPO as the new standard replacing PPO**. **Kimi K2 introduces the MuonClip optimizer** to address the stability of RL training.

[arXiv:2507.20534](https://arxiv.org/abs/2507.20534)

**ByteDance is the largest contributor to the GRPO family of improvements** (DAPO + VAPO + UI-TARS + DanceGRPO + Seedance in a full chain). **Zhipu's GLM-4.5/4.6/5 series** first introduced "difficulty curriculum RL" as a mainstream training paradigm.

[arXiv:2508.06471](https://arxiv.org/abs/2508.06471)

**Step3-VL by Starry Sky** proposes PaCoRe parallel coordinated reasoning, opening up another path for test-time scaling.

In November 2025, **Anthropic publishes "Natural Emergent Misalignment from Reward Hacking"** (

[arXiv:2511.18397](https://arxiv.org/abs/2511.18397)

）bringing reward hacking research into a new stage — the naturally emerging misalignment behaviors during RL training have become a cutting-edge security topic. In the same month and year, **Microsoft's Reinforcement Pre-Training (RPT)** (

[arXiv:2506.08007](https://arxiv.org/abs/2506.08007)

This approach challenges the boundary between pre-training and fine-tuning, directly introducing reinforcement learning into the pre-training phase. **DeepMind's AlphaEvolve** (2025.05) combines LLMs, evolutionary algorithms, and automatic evaluators, achieving a 23% acceleration in matrix multiplication. This represents a new paradigm for search algorithms in the LLM era.

RL has walked from the 1890s maze to the 2026 industrial cluster, spanning over a century. Yet its core has never changed — **letting agents trial and error in the environment, guided solely by accumulated rewards, to explore and find the optimal strategy on their own**.

## Summary

From Thorndike's maze, to Bellman's equation; from the DQN in the Atari game console, to today's rapidly iterating DPO and GRPO in the cloud clusters. The history of reinforcement learning is an epic of **"learning from the environment, evolving from feedback, and moving from single machines to super models"**.

Today, reinforcement learning is no longer a theoretical toy in the ivory tower. It is a necessary path toward general artificial intelligence (AGI). In the following chapters, we will follow the trajectory of this history, starting from the first line of code, and implement these great algorithms ourselves.

## References

[^1]: Bellman, R. (1957). A Markovian Decision Process. _Journal of Mathematics and Mechanics_, 6(5), 679-684.

[DOI](https://doi.org/10.1512/iumj.1957.6.56038)

[^2]: Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. _Machine Learning_, 3(1), 9-44.

[PDF](http://incompleteideas.net/papers/sutton-88.pdf)

[^3]: Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. _PhD Thesis, King's College, Cambridge_.

[PDF](https://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf)

[^4]: Tesauro, G. (1995). Temporal difference learning and TD-Gammon. _Communications of the ACM_, 38(3), 58-68.

[DOI](https://doi.org/10.1145/203330.203343)

[^5]: Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press.

[Online Reading](http://incompleteideas.net/book/the-book.html)

[^6]: Mnih, V., et al. (2013). Playing Atari with Deep Reinforcement Learning. _arXiv preprint_.

[arXiv:1312.5602](https://arxiv.org/abs/1312.5602)

[^7]: Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. _Nature_, 529(7587), 484-489.

[DOI](https://doi.org/10.1038/nature16961)

[^8]: Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. _arXiv preprint_.

[arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

[^9]: Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. _arXiv preprint_.

[arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

[^10]: Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. _arXiv preprint_.

[arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

[^11]: DeepSeek-AI, et al. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. _arXiv preprint_.

[arXiv:2501.12948](https://arxiv.org/abs/2501.12948)

[^12]: Yu, Q., et al. (2025). DAPO: An Open-Source LLM Reinforcement Learning System at Scale. _arXiv preprint_.

[arXiv:2503.14476](https://arxiv.org/abs/2503.14476)

[^13]: Liu, Y., et al. (2025). Understanding r1-zero-like training. _arXiv preprint_.

[arXiv:2503.20783](https://arxiv.org/abs/2503.20783)

[^14]: Zheng, C., et al. (2025). GSPO: Group Sequence Policy Optimization. _arXiv preprint_.

[arXiv:2507.18071](https://arxiv.org/abs/2507.18071)

[^15]: MiniMax, et al. (2025). MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention. _arXiv preprint_.

[arXiv:2506.13585](https://arxiv.org/abs/2506.13585)

[^16]: ByteDance Seed, et al. (2025). VAPO: Value-based Augmented PPO. _arXiv preprint_.

[arXiv:2504.05118](https://arxiv.org/abs/2504.05118)

[^17]: OpenAI (2025). Competitive Programming with Large Reasoning Models.

[arXiv:2502.06807](https://arxiv.org/abs/2502.06807)

[^18]: Meta (2025). SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution.

[arXiv:2502.18449](https://arxiv.org/abs/2502.18449)

[^19]: Anthropic (2025). Emergent Misalignment: Researching the impact of reward hacking.

[arXiv:2511.18397](https://arxiv.org/abs/2511.18397)

[^20]: Microsoft Research (2025). Reinforcement Pre-Training.

[arXiv:2506.08007](https://arxiv.org/abs/2506.08007)
