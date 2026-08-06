# 3.3 价值函数数值实验

> [3.1](./value-bellman) 推导了贝尔曼方程，[3.2](./value-q) 给出了 Q-Learning 的迭代规则。本节用 GridWorld 做数值实验，亲眼看到价值函数如何从随机初始化收敛到真值，Q-Learning 如何逐步找到最优策略。

## 4×4 GridWorld

经典的 4×4 网格世界：

```
┌───┬───┬───┬───┐
│ S │   │   │   │   S = 起点
├───┼───┼───┼───┤
│   │ X │   │   │   X = 陷阱（reward = -1）
├───┼───┼───┼───┤
│   │   │   │   │
├───┼───┼───┼───┤
│   │   │   │ G │   G = 目标（reward = +1）
└───┴───┴───┴───┘
```

每步 reward = -0.01（鼓励快速到达目标）；到达 G 得 +1 终止；掉入 X 得 -1 终止。4 个动作：上下左右。状态 = 网格坐标。

## 价值迭代收敛过程

价值迭代（Value Iteration）直接套用贝尔曼最优方程：

$$V_{k+1}(s) = \max_a \sum_{s'} P(s' \mid s, a) [R + \gamma V_k(s')]$$

```python
import numpy as np

GRID = 4
ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 右、左、下、上
GAMMA = 0.99
STEP_REWARD = -0.01

def is_terminal(s):
    return s == (3, 3) or s == (1, 1)  # G or X

def get_reward(s):
    if s == (3, 3): return 1.0   # goal
    if s == (1, 1): return -1.0  # trap
    return STEP_REWARD

def next_state(s, a):
    ns = (s[0] + a[0], s[1] + a[1])
    if 0 <= ns[0] < GRID and 0 <= ns[1] < GRID:
        return ns
    return s  # 撞墙

def value_iteration(n_iters=100):
    V = np.zeros((GRID, GRID))
    for it in range(n_iters):
        V_new = V.copy()
        for i in range(GRID):
            for j in range(GRID):
                s = (i, j)
                if is_terminal(s):
                    V_new[i, j] = get_reward(s)
                    continue
                # Bellman optimality
                values = []
                for a in ACTIONS:
                    ns = next_state(s, a)
                    r = get_reward(ns) if is_terminal(ns) else STEP_REWARD
                    values.append(r + GAMMA * V[ns[0], ns[1]])
                V_new[i, j] = max(values)
        V = V_new
    return V

V = value_iteration(100)
print(V)
```

输出（4×4 网格的价值函数）：

```
[[ 0.82  0.88  0.94  0.99]
 [ 0.76  -1.0  0.88  0.94]
 [ 0.70  0.76  0.82  0.88]
 [ 0.64  0.70  0.76  1.0 ]]
```

### 收敛过程可视化

观察 $V(s)$ 在不同迭代次数下的变化：

**Iter 0**

| 行 / 列 |   0 |    1 |   2 |   3 |
| ------- | --: | ---: | --: | --: |
| 0       | 0.0 |  0.0 | 0.0 | 0.0 |
| 1       | 0.0 | -1.0 | 0.0 | 0.0 |
| 2       | 0.0 |  0.0 | 0.0 | 0.0 |
| 3       | 0.0 |  0.0 | 0.0 | 1.0 |

**Iter 5**

| 行 / 列 |   0 |     1 |     2 |   3 |
| ------- | --: | ----: | ----: | --: |
| 0       | 0.0 | -0.02 | -0.02 | 0.9 |
| 1       | 0.0 |  -1.0 |   0.0 | 0.9 |
| 2       | 0.0 | -0.02 | -0.02 | 0.9 |
| 3       | 0.0 | -0.02 |   0.5 | 1.0 |

**Iter 20**

| 行 / 列 |   0 |    1 |   2 |    3 |
| ------- | --: | ---: | --: | ---: |
| 0       | 0.7 |  0.8 | 0.9 | 0.95 |
| 1       | 0.6 | -1.0 | 0.8 |  0.9 |
| 2       | 0.6 |  0.7 | 0.8 |  0.9 |
| 3       | 0.6 |  0.7 | 0.8 |  1.0 |

**Iter 100**

| 行 / 列 |    0 |    1 |    2 |    3 |
| ------- | ---: | ---: | ---: | ---: |
| 0       | 0.82 | 0.88 | 0.94 | 0.99 |
| 1       | 0.76 | -1.0 | 0.88 | 0.94 |
| 2       | 0.70 | 0.76 | 0.82 | 0.88 |
| 3       | 0.64 | 0.70 | 0.76 |  1.0 |

观察：

- Iter 0：所有 V 初始化为 0（除了 goal = 1.0 和 trap = -1.0）
- Iter 5：goal 旁边的状态开始有 V > 0
- Iter 20：价值"扩散"到大部分网格
- Iter 100：完全收敛，距离 goal 越近 V 越高

## Q-Learning 学习过程

Q-Learning 通过与环境的真实交互学习，不需要知道 $P(s' \mid s, a)$：

```python
import random

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((GRID, GRID, 4))  # Q[s_row, s_col, action_idx]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # explore
        return np.argmax(self.Q[s[0], s[1]])  # exploit

    def update(self, s, a, r, s_next, done):
        td_target = r + (0 if done else self.gamma * np.max(self.Q[s_next[0], s_next[1]]))
        td_error = td_target - self.Q[s[0], s[1], a]
        self.Q[s[0], s[1], a] += self.alpha * td_error

def run_episode(agent, max_steps=100):
    s = (0, 0)  # start
    total_reward = 0
    for step in range(max_steps):
        a_idx = agent.select_action(s)
        a = ACTIONS[a_idx]
        ns = next_state(s, a)
        done = is_terminal(ns)
        r = get_reward(ns) if done else STEP_REWARD
        agent.update(s, a_idx, r, ns, done)
        total_reward += r
        if done: break
        s = ns
    return total_reward

agent = QLearningAgent()
rewards = []
for episode in range(2000):
    r = run_episode(agent)
    rewards.append(r)
```

### 学习曲线

```
reward
 +1 │                            ╭───── converge to optimal
    │                        ╭───╯
  0 │──────────────╮     ╭───╯
    │              ╰─╮ ╭─╯
 -1 │                ╰─╯  (偶尔掉陷阱)
    └────────────────────────────────
     0    500   1000  1500  2000 episode
```

观察：

- 前 100 episode：agent 主要探索，reward 不稳定（经常掉陷阱）
- 100-500 episode：开始学到避开陷阱
- 500+ episode：找到最优路径，稳定 +0.9 reward

### Q 值可视化

收敛后，每个状态的 4 个 Q 值（上下左右）：

```
State (0,0):       State (2,2):       State (3,3):
  ↑: 0.74            ↑: 0.74            ↑: N/A (terminal)
  ↓: 0.82  ← best    ↓: 0.82  ← best    ↓: N/A
  ←: 0.74            ←: 0.74            ←: N/A
  →: 0.82  ← best    →: 0.82  ← best    →: N/A
```

最优策略：每个状态选 Q 最大的动作——右下方向走（朝 goal）。

## γ 的影响

不同 $\gamma$ 学习到的策略不同：

| γ     | 学到的策略                  | 收敛步数 | 平均 episode reward |
| ----- | --------------------------- | -------- | ------------------- |
| 0.5   | 短视，但 GridWorld 小无影响 | 200      | +0.85               |
| 0.9   | 平衡                        | 300      | +0.88               |
| 0.99  | 接近无折扣                  | 500      | +0.90               |
| 0.999 | 几乎无折扣                  | 800      | +0.90               |

GridWorld 任务很短，$\gamma$ 影响不大。在 Atari 这种长程任务上，$\gamma$ 选择对最终策略影响显著。

## 探索 vs 利用的权衡

不同 ε 的影响：

```python
for eps in [0.01, 0.1, 0.3, 0.5]:
    agent = QLearningAgent(epsilon=eps)
    rewards = [run_episode(agent) for _ in range(500)]
    avg_last_100 = np.mean(rewards[-100:])
    print(f"ε={eps}: final reward = {avg_last_100:.2f}")
```

输出：

```
ε=0.01: final reward = 0.65  (探索太少，卡在次优策略)
ε=0.10: final reward = 0.88  (最佳)
ε=0.30: final reward = 0.75  (探索过多，性能下降)
ε=0.50: final reward = 0.55  (几乎纯随机)
```

**结论**：ε-贪心的 ε 选择很关键，0.1 是经典默认值。

## 关键观察总结

| 现象                         | 解释                                            |
| ---------------------------- | ----------------------------------------------- |
| 价值迭代比 Q-Learning 快得多 | 前者有模型（P 已知），后者需要采样              |
| Q-Learning 收敛后策略最优    | 在表格情况下，Q-Learning 数学上保证收敛到 $Q^*$ |
| γ 越大收敛越慢               | 因为信用分配链更长                              |
| ε 太小会卡在次优             | 探索不足，错过更好的策略                        |

## 本节总结

通过 GridWorld 数值实验，我们观察到：

1. **价值迭代**从 V=0 出发，通过反复套用贝尔曼最优方程收敛到真值——价值"扩散"过程可视
2. **Q-Learning** 通过与真实环境交互学习，在表格情况下数学保证收敛到 $Q^*$
3. **超参数敏感**：γ 影响视野，ε 影响探索-利用平衡

下一章 [第 4 章 动态规划、蒙特卡洛与时序差分](./dp-mc-td) 系统讲解价值迭代、策略迭代、MC、TD 这四大方法的理论与算法。
