# 18.2 判别式 PRM 路线

判别式 PRM 是最早被系统研究的 PRM 路线。它的思路最直接——把"判断推理步骤对错"建模为一个**分类任务**：输入 prompt + step，输出"这步对/错"的概率。

这一节我们以 OpenAI 的 [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)（Lightman et al. 2023）为主线，讲解判别式 PRM 的方法、数据集、应用、局限。

## 18.2.1 OpenAI 的过程监督研究

2023 年 5 月，OpenAI 发表了 [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)。这篇论文的背景是：

- GPT-4 在数学推理上取得了突破，但仍然经常在多步推理中出错
- ORM（Outcome Reward Model）只能给最终答案打分，无法定位错误步骤
- 一个直觉性的改进：**让人类逐步标注推理步骤，训练一个 PRM**

OpenAI 比较了三种监督方式：

| 监督方式              | 标注内容        | 标注成本 |
| --------------------- | --------------- | -------- |
| **ORM**（Outcome RM） | 最终答案对错    | 低       |
| **PRM**（Process RM） | 每步推理对错    | 高       |
| **PRM800K**（全步骤） | 80 万步推理标注 | 极高     |

实验结果（在 MATH 数据集上）：

- ORM：53.9%
- PRM（少量步骤标注）：56.6%
- **PRM800K**（全步骤标注）：**78.2%**

这是一个**显著的提升**——同样的 base model，仅仅通过更细致的监督方式，准确率从 53.9% 提升到 78.2%。这证明了 PRM 的价值。

## 18.2.2 PRM800K 数据集

[PRM800K](https://github.com/openai/prm800k) 是 OpenAI 公开的步骤级标注数据集。规模：

- **800K** 个步骤级标注
- 来自 **75K** 个数学问题的解题过程
- 每个步骤标注为三种状态之一：
  - **good**（正确）：推理合理，可以继续
  - **bad**（错误）：推理有误，应该停止或回退
  - **neutral**（中性）：可能是中间过渡，不评价对错

标注流程：

1. 让 base model 生成解题过程
2. 把解题过程切分为"步骤"（按换行符、句号、明显的推理标记切分）
3. 人类标注员逐步判断每一步是 good / bad / neutral
4. 收集 800K 标注

### PRM800K 的成本

OpenAI 没有公开 PRM800K 的标注成本，但可以从规模估算：

- 800K 步骤 × 平均 30 秒/步 = 24,000 小时（约 12 人年）
- 按美国 ML 标注员时薪 $30-50 计算，成本约 $72 万 - $120 万

这个成本对工业实验室（OpenAI、Anthropic、Google）可以承受，但对学术研究和小公司几乎不可行。这就是判别式 PRM 的核心瓶颈——**标注成本太高**。

## 18.2.3 判别式 PRM 的训练

有了 PRM800K，训练一个判别式 PRM 就是标准的分类任务。

### 模型架构

OpenAI 用的架构（论文里）：

- Base：GPT-4 的预训练 backbone（或更小的 fine-tune 版本）
- 输入：`<prompt> <response_up_to_step_i> <step_i>`
- 输出：三分类（good / bad / neutral）

后续工作（如 [Math-Shepherd](https://arxiv.org/abs/2312.08935)）用更小的 base（LLaMA-7B、Qwen-7B），加上分类头。

### 训练目标

交叉熵损失：

$$\mathcal{L}_{\text{PRM}} = -\sum_{i} \sum_{c \in \{good, bad, neutral\}} y_{i,c} \log p_\theta(c | q, o_{\leq i}, s_i)$$

其中 $y_{i,c}$ 是第 $i$ 步的真实标签的 one-hot 编码，$s_i$ 是第 $i$ 步的内容。

### 训练数据增强

PRM800K 只有 800K 标注，对训练一个强大的 PRM 不够。常见增强方式：

- **自动标注**：用 GPT-4 自动标注未标注的步骤（带噪声）
- **合成数据**：从已知正确的解题过程中生成"看起来像错"的步骤作为负样本
- **数据混合**：把 PRM800K 和 Math-Shepherd 等其他数据集合并

## 18.2.4 PRM 作为 Re-ranking 模型

PRM 的主要应用不是直接用于 RL 训练，而是用于 **Re-ranking（重排序）**。

Re-ranking 的工作流程：

1. **生成**：让 base model 对一个数学题生成 N 个候选解（N 通常 4-64）
2. **打分**：用 PRM 给每个候选解的每一步打分，得到整条解题过程的总分
3. **选择**：选总分最高的候选解作为最终答案

OpenAI 的实验显示，**Re-ranking with PRM 比单次生成 + ORM 的效果好很多**：

| 方法                   | MATH 准确率 |
| ---------------------- | ----------- |
| 单次生成（greedy）     | ~40%        |
| 单次生成 + ORM         | ~50%        |
| **N=64 + PRM Re-rank** | **78.2%**   |

这说明 PRM 的价值不只是"训练时给密集 reward"，更在于"推理时选择最佳候选"。

### Re-ranking 的 token 级 vs 步骤级

PRM Re-ranking 有两种打分方式：

**Token 级**：每个 token 一个分数，整条回答的总分 = 所有 token 分数的某种聚合（mean、sum、min）。

**步骤级**：每个推理步骤一个分数，整条回答的总分 = 所有步骤分数的某种聚合。

OpenAI 用的是步骤级——这样更符合人类判断"这步对不对"的直觉。

聚合方式的选择：

- **Mean**：所有步骤的平均分。鲁棒，但可能稀释关键错误步骤的影响
- **Min**：所有步骤的最低分。保守，倾向于"任何一个步骤错了就否决整条"
- **Product**：所有步骤分数的乘积。比 min 更严格

OpenAI 实验显示 **min** 在数学任务上效果最好——一个步骤错了，整条推理就不可信。但在 creative writing 等任务上 mean 更合适。

## 18.2.5 判别式 PRM 在 RL 训练中的使用

Re-ranking 是 inference-time 应用。PRM 也可以用于 RL 训练——把 PRM 的步骤级分数作为 RL 的密集 reward。

具体做法：

```python
# 用 PRM 作为 RL 训练的 reward 函数
def prm_reward(prompt, response):
    # 切分 response 为推理步骤
    steps = split_into_steps(response)

    # 对每一步打分
    step_scores = [prm(prompt, steps[:i+1]) for i in range(len(steps))]

    # 聚合为整条 response 的 reward
    # 选择 min：任何一个步骤错了就否决
    return min(step_scores)
```

这个 reward 函数可以替代 RLHF 中的 ORM，用于 PPO / GRPO 训练。

[Math-Shepherd](https://arxiv.org/abs/2312.08935) 是这种做法的代表作。它在 GSM8K 和 MATH 上用 PRM reward 训练 LLaMA，比 ORM reward 训练的 baseline 高 5-10 个百分点。

## 18.2.6 判别式 PRM 的局限

判别式 PRM 虽然有效，但有几个根本局限：

### 标注成本爆炸

PRM800K 的标注成本是百万美元级。对于一个新的领域（如代码生成、生物医学推理），从头标注一个 PRM800K 规模的数据集几乎不可行。

这是判别式 PRM 推广的最大障碍——**每个新领域都要重新标注**。

### 泛化能力弱

判别式 PRM 在训练领域表现好，但跨领域泛化差。比如在数学数据上训练的 PRM，用在代码生成任务上效果大幅下降——它学到的"什么是好的推理步骤"是领域相关的。

而 [生成式 PRM](./generative-prm)（下一节）因为用自然语言评价，泛化能力更强。

### 噪声标注

即使有 PRM800K，标注也不是 100% 准确：

- 不同标注员对同一步骤可能判断不同
- 复杂推理步骤的对错本身就有主观性
- 中性（neutral）类别的边界模糊

[Lightman et al.](https://arxiv.org/abs/2305.20050) 论文里也承认标注噪声问题，并用集成（ensemble）方式缓解。

### 固定的步骤切分

判别式 PRM 需要把回答切分为"步骤"——但怎么切分本身是个问题：

- 按换行符切：太机械，一个"步骤"可能跨多行
- 按句号切：太碎，一个完整推理可能包含多个句号
- 用 LLM 切：引入新的 LLM 调用成本

不同的切分方式会显著影响 PRM 的判断。这是判别式 PRM 的工程复杂度来源。

## 18.2.7 判别式 PRM 的工业实践

尽管有这些局限，判别式 PRM 仍然是工业实践的主流：

- **OpenAI o1/o3**：内部据说用 PRM 引导推理（未公开）
- **DeepSeek**：Math-Shepherd 路线，PRM 用于 RL 训练
- **Anthropic**：Claude 的内部 PRM（推测）
- **阿里 Qwen-Math**：用 PRM Re-ranking 提升 MATH benchmark

工业上的几个趋势：

1. **PRM + ORM 混合**：用 PRM 给过程奖励，用 ORM 给最终奖励，加权求和
2. **领域特化 PRM**：为不同任务训练不同 PRM（数学 PRM、代码 PRM、生物医学 PRM）
3. **PRM 自动标注**：用强 LLM（GPT-4、Claude）自动标注步骤，减少人工成本

## 小结

判别式 PRM 是 PRM 的经典路线，OpenAI 的 [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) 和 PRM800K 数据集奠定了基础。它的核心思想——**训练一个分类器判断每步推理的对错**——简单直接，在数学任务上证明了价值。

但判别式 PRM 有三个根本瓶颈：标注成本高、跨领域泛化弱、步骤切分敏感。这三个瓶颈催生了下一节的**生成式 PRM** 路线——用更少的标签达到更好的效果。
