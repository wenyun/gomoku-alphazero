# 算法讲解

这篇讲的是**算法本身怎么运作**：三个部件如何配合、数据怎么流动、哪几处最容易写错。

关于**为什么选 AlphaZero 而不是 PPO**，见 [README 第 1 节](README.md#1-为什么用-alphazero-而不是-ppo)。
关于**每个文件负责什么**以及**训练过程中踩的坑**，见 README 的第 3 节和第 4 节。

---

## 1. 一句话：这是一个策略迭代

经典强化学习里的策略迭代（policy iteration）有两步：评估当前策略，然后据此改进它。
AlphaZero 把这两步换成了：

| 策略迭代的步骤 | AlphaZero 的做法 | 代码 |
|---|---|---|
| **策略改进**（policy improvement） | MCTS 前瞻 512 步，输出一个比网络更强的策略 | `gomoku_zero/mcts.py` |
| **把改进吸收回参数**（projection） | 训练网络去拟合搜索的输出 | `gomoku_zero/learner.py` |

整个算法就是这两步无限循环。

**关键洞察是：MCTS 的输出严格强于它的输入。**
网络只是"看一眼就下"的直觉；MCTS 拿着这个直觉真的往前算了 512 步，而且在终局处拿到的是**精确答案**
而不是估计值。所以搜索的结论一定比直觉更可靠——每转一圈就是一次确定的爬坡。

这一点是 AlphaZero 和 PPO 最本质的差别。PPO 的改进步是"沿策略梯度挪一小步"，改进量小、方差大；
AlphaZero 的改进步是"做一次几百次模拟的搜索"，改进量大且几乎无偏。

---

## 2. 部件一：网络 = 一个直觉函数

`network.py` 里的网络就是一个函数：

```
f(s) -> (p, v)
```

* `p`：225 维概率分布，"这个局面我该走哪里"（策略头）
* `v ∈ [-1, +1]`：一个标量，"该我走棋的这一方赢面如何"（价值头）

它**不做任何前瞻**，纯粹是模式识别——相当于人类棋手的棋感。

结构是标准的 AlphaZero：卷积 stem → 10 个残差块（128 通道）→ 两个头，共 299 万参数。
两个头**共用同一个主干**，这很重要：MCTS 每展开一个叶子节点，`p` 和 `v` 两个都要用，
共享主干意味着一次前向就够了。

两个头各自的设计取舍（策略头用 1×1 卷积保持平移等变；价值头必须先压到 1 个通道，
否则 tanh 会饱和把价值头彻底训死）写在 README 的第 3.3 节和第 4 节，那是这个项目里
花时间最多的一个坑。

---

## 3. 部件二：MCTS = 把直觉放大成棋力

一次搜索重复 512 次"模拟"（simulation），每次模拟做四件事：

```
        ┌─────────────────────────────────────────────┐
        │                                             │
   1. select        从根节点出发，每一步都取 PUCT 分数最大的边，
        │           一直走到某条边还没有子节点为止
        ▼
   2. expand        这条边的终点成为新叶子；把叶子局面送进网络，
        │           拿到它的先验 p 和价值 v
        ▼
   3. backup        把 v 沿路径回传到每条边上，每上一层翻一次符号
        │
        ▼
   4. repeat  ──────┘  （共 512 次）

   最后：根节点每条边的访问次数 N，就是这次搜索的输出
```

### PUCT：选择哪条边

搜索的全部智能集中在这一个公式里（`mcts.py` 第 240–247 行）：

```python
def _select(self, tree: GameTree, node: int) -> int:
    """PUCT: pick the child that best trades off value against uncertainty."""
    n = tree.N[node]
    q = tree.W[node] / np.maximum(n, 1)      # unvisited edges score Q = 0
    u = (self.cfg.c_puct * math.sqrt(tree.visits[node] + 1)) * tree.P[node] / (1 + n)
    scores = q + u
    scores[~tree.legal[node]] = -np.inf
    return int(scores.argmax())
```

写成数学形式：

```
              ┌──── 利用 ────┐   ┌──────────── 探索 ────────────┐
score(s,a) =    W(s,a)/N(s,a)  +  c_puct · P(s,a) · √(ΣN(s,·)+1)
                                                    ─────────────
                                                      1 + N(s,a)
```

两项各管一件事：

* **`q = W/N`（利用）**：这一手到目前为止试出来的平均结果。试得越多越可信。
* **`u`（探索）**：
  * 分子里有网络先验 `P` → **网络看好的先试**，这是网络指导搜索的唯一通道；
  * 分母是 `1 + N` → **试过的被压下去**，避免所有预算都砸在第一个还不错的着法上；
  * 系数里的 `√(总访问次数)` → 搜索越深入，探索项整体越大，保证不会过早锁死在一条变化上。

`c_puct = 2.0` 就是这两者的权重（在 `config.py` 里）。

### 为什么用访问次数 N 当输出，而不是 Q？

这是我觉得整个算法里最漂亮的一个设计。搜索完成后，我们要输出一个"改进后的策略"，
候选是每条边的平均价值 `Q` 或访问次数 `N`。AlphaZero 用 `N`，原因有两个：

1. **`N` 已经隐含了 `Q` 的信息。** PUCT 会不断把访问预算挪到 `Q` 高的分支上，
   所以 512 次模拟结束后，`N` 的分布本身就是"考虑过前瞻之后的着法优劣排序"。
2. **`N` 自带置信度，`Q` 没有。** 一条只被访问 3 次的边，它的 `Q` 可能纯粹是噪声；
   但它的 `N` 很小，在归一化之后权重自然就低。如果直接用 `Q` 当训练目标，
   一次幸运的模拟就能把某个坏着法捧成"最优解"。

搜索完就用 `N` 落子（`mcts.py` 第 265–278 行）：

```python
def sample_move(visits, temperature, rng):
    """Turn root visit counts into an actual move."""
    if temperature <= 1e-6:
        return int(visits.argmax())
    weights = visits.astype(np.float64) ** (1.0 / temperature)
    ...
```

温度的用法（`selfplay.py`）：

| 场景 | 温度 | 效果 |
|---|---|---|
| 自我对弈，前 12 手 | 1.0 | 正比于 `N` 采样 → 开局多样性 |
| 自我对弈，12 手之后 | 0.25 | 正比于 `N⁴` → 接近最优，但保留一点变化 |
| 真正和人下棋 | 0 | 直接取访问最多的着法 |

自我对弈时温度不设成 0 是刻意的：完全贪心会让同一个网络反复下出同一盘棋。

---

## 4. 部件三：训练 = 把搜索的成果蒸馏回网络

损失函数只有两项（`learner.py` 第 109–113 行）：

```python
logits, value_pred = self.net(planes)
# Cross-entropy against a soft target distribution.
policy_loss = -(policy_target * F.log_softmax(logits.float(), dim=1)).sum(1).mean()
value_loss = F.mse_loss(value_pred.float(), value_target)
loss = policy_loss + self.cfg.train.value_loss_weight * value_loss
```

两项在做完全不同的事：

* **策略项**：让网络的直觉 `p` 去追搜索的结论 `π`（访问分布）。
  搜索比网络强，所以这一项是在**往上拉网络**。这是一种自蒸馏：老师和学生是同一个网络，
  只不过老师额外获得了 512 次前瞻的算力。
* **价值项**：让 `v` 去拟合**真实的对局结果**。这是整个系统里唯一的"地面真值"来源。

### 飞轮

这两项互相喂养，就是 AlphaZero 能自我提升的全部原因：

```
  v 变准 ──► MCTS 叶子评估变准 ──► 搜索出的 π 更好 ──► p 变强
    ▲                                                    │
    │                                                    ▼
    └──── 对局质量更高，胜负标签更有信息 ◄──── 自我对弈棋力更强
```

注意这个环里没有任何外部数据、没有人类棋谱、也没有手写的评估函数。
唯一的外部输入就是**规则**（谁连成五就赢）。

### 唯一的额外技巧：对称性增强

五子棋在正方形的 8 个对称变换（4 个旋转 × 是否镜像）下不变，所以每个局面有 8 种等价写法，
相当于免费的 8 倍数据。实现上把一个 batch 切成 8 份，每份施加一种变换，全程向量化。

**策略目标必须施加和输入平面完全相同的变换**——所以先把 225 维的策略 reshape 成 15×15
的"图像"再一起旋转。这里我踩过一个坑（固定 batch 下增强会完全失效），写在 README 第 4 节坑 5。

---

## 5. 一局棋的完整数据流

自我对弈时，每走一步就记录一条训练样本（`selfplay.py` 第 67–75 行）：

```python
# Record the position *before* the move, with the search policy.
self.trajectories[i].append((
    (board.stones * board.to_play).astype(np.int8),   # 输入：局面
    board.last_move,
    board.prev_move,
    1 if board.to_play == 1 else 0,
    (counts / total).astype(np.float16),              # 策略目标：访问分布
    board.to_play,
))
```

价值目标要等一局下完才能填，因为它就是最终胜负（`selfplay.py` 第 104–108 行）：

```python
# The result, seen from whoever was about to move in that position.
if winner == 0:
    value = np.zeros(t, dtype=np.float16)
else:
    value = np.where(movers == winner, 1.0, -1.0).astype(np.float16)
```

所以一条训练样本就是三元组 **`(局面 s, 搜索访问分布 π, 最终胜负 z)`**。全流程：

```
   局面 s
     │
     ├──────── MCTS 512 次模拟 ────────► 访问分布 π ──── 按温度采样 ────► 落子
     │                                      │
     │（存为网络输入）                      │（存为策略目标）
     ▼                                      ▼
  ┌──────────────────────────────────────────────────┐
  │  回放池 ReplayBuffer（环形，300 万个局面）        │
  │  一局结束后回填胜负 z 作为价值目标               │
  └──────────────────────┬───────────────────────────┘
                         │  均匀采样 batch = 1024
                         ▼
                  梯度更新（AdamW）
                         │
                         │  每 500 步发布 latest.pt
                         ▼
             42 个自我对弈进程重新加载权重 ──► 回到最上面
```

回放池用**滑动窗口**（只保留最近 300 万个局面，约 30 分钟的自我对弈）而不是全部历史。
原因是网络在跟自己学：随着它变强，很老的数据只是"一个更弱的自己当年怎么下"。
滑动窗口让网络始终拟合当前水平附近的数据。

`train.py` 里还有一个限流机制（`_may_train`），把"每个局面平均被喂给网络几次"压在 6 以内。
不管它的话，learner 会在少量新数据上反复空转，这是这类训练最经典的崩溃方式。

---

## 6. 三个最容易写错的细节

### (1) 符号翻转

两人零和博弈里，你的 +1 就是我的 −1。价值回传时每上一层都要翻号
（`mcts.py` 第 249–262 行）：

```python
@staticmethod
def _backup(tree: GameTree, path, leaf_value: float) -> None:
    """Propagate `leaf_value` up the path, flipping sign at every ply.

    `leaf_value` is from the point of view of the player to move *at the
    leaf*.  The parent of the leaf is the opponent, so it sees -leaf_value,
    the grandparent sees +leaf_value, and so on.
    """
    value = -leaf_value
    for node, action in reversed(path):
        tree.N[node, action] += 1
        tree.W[node, action] += value
        tree.visits[node] += 1
        value = -value
```

配套的约定是：**所有东西都用"当前该走棋一方"的视角**。

* 棋盘存成 `stones * to_play`（我方 +1、对方 −1）
* 价值目标存成"该走棋这方最后赢没赢"
* 网络输出的 `v` 也是这个视角

统一视角之后，网络不用分别学黑白两套镜像概念，参数利用率翻倍。
代价是这个符号约定必须贯穿全部代码，任何一处搞反都会让价值信号变成噪声——
而且很难发现，因为损失曲线看起来仍然在下降。

### (2) `√(visits + 1)` 里的 `+1`

如果写成 `√visits`，根节点第一次选择时探索项**整体是 0**，`Q` 也全是 0，
于是 `argmax` 退化成"永远选第 0 号动作"，完全无视网络先验。加了 `+1` 之后，
第一次选择就严格按 `P` 排序。

### (3) 根节点必须加 Dirichlet 噪声

MCTS 本身是**完全确定性**的。没有噪声，同一个网络自我对弈会反复下出同一盘棋，
训练数据的多样性归零，飞轮直接停转（`mcts.py` 第 148–159 行）：

```python
def _add_dirichlet(self, priors, mask):
    """Mix Dirichlet noise into the root priors so self-play keeps exploring.

    Without this, self-play would play (almost) the same game forever and the
    network would never see anything new.
    """
    legal = np.flatnonzero(mask)
    noise = self.rng.dirichlet([self.cfg.dirichlet_alpha] * len(legal))
    eps = self.cfg.dirichlet_epsilon
    out = priors.copy()
    out[legal] = (1.0 - eps) * out[legal] + eps * noise
    return out
```

注意**只在根节点加**。树内部不加——那会污染搜索本身的质量，
而我们要的是"探索不同的开局走向"，不是"把搜索搞乱"。
参数是 `alpha=0.15`、`epsilon=0.25`（即 25% 的权重给噪声）。

---

## 7. 冷启动：为什么随机初始化的网络也能起步

这是整个算法最反直觉的地方。训练刚开始时 `p` 和 `v` 都是随机的：
先验没有意义，叶子评估也没有意义。搜索似乎无从下手，那第一点信号从哪来？

**答案是：终局是精确的。** 搜索走到棋局结束时不需要网络，直接拿到真值 ±1
（`mcts.py` 第 177–184 行）：

```python
if leaf.is_over:
    proven = leaf.terminal_value()
    mask = None
else:
    # Cheap exact check for "somebody makes five right now", which
    # either settles the node outright or forces a single reply.
    mask, proven = interior_analysis(leaf)
```

所以最初的**全部**学习信号来自"谁真的连成了五"。bootstrap 是这样发生的：

1. 随机对弈，偶然有人连成五 → 搜索在终局节点拿到精确的 ±1
2. 这批数据教会网络最基本的事："连成五就赢"、"对方有四子很危险"
3. `v` 有了一点信号 → 搜索的叶子评估比随机好一点 → 搜索出的棋比随机好一点
4. 产生的对局质量更高 → 训练信号更好 → 回到第 2 步

实测轨迹：价值损失 0.526 → 0.096，全程 +958 Elo（详见 [RESULTS.md](RESULTS.md)）。

---

## 8. 和标准 AlphaZero 的三处不同

| 加了什么 | 为什么 | 效果 |
|---|---|---|
| **候选着点剪枝**（`board.py`）：只考虑距已有棋子 2 格以内的空点 | 五子棋里远离所有棋子的一手永远没用 | 分支因子 225 → 约 40，同样的模拟次数搜索深得多 |
| **精确成五检测**（`tactics.py`，126 行） | 冷启动期网络对威胁是瞎的：70 个候选各分到 5 次访问，根本发现不了"不堵就输" | 消灭整整一类失误，代价只有几微秒 |
| **跨对局批量化 + CUDA graph**（`mcts.py` / `network.py`） | 一次模拟只需一次网络推理，batch=1 极度浪费 GPU | 96 局齐步走共享一次推理；一次前向的约 70 次 kernel launch 压成 1 次 |

前两个改的是**算法**。`tactics.py` 是全项目**唯一**写入人类棋理的地方，而且只判断"一步成五"——
活三、双三、VCF 一概交给网络自己学。它在搜索里的作用是：

* 我能一步成五 → 该节点价值**精确等于 +1**，不需要网络评估
* 对方有唯一一个成五点 → 我**只能**下那一格，候选缩到 1
* 对方有两个不同的成五点 → 一颗子只能堵一个，价值**精确等于 −1**

第三个纯粹是**工程**，不影响搜索结果。`tests/test_core.py` 里有一个测试专门验证
"96 局一起搜出来的树 == 每局单独搜出来的树"，逐个访问次数完全相同。

---

## 9. 继续读代码的顺序

如果要顺着这篇文档去读实现，建议：

1. **`gomoku_zero/mcts.py` 的 `search()`**（第 90–108 行）—— 主循环，10 行看完全貌
2. **`_simulate_round()`**（第 161–207 行）—— 一轮模拟怎么跨 96 局共享一次网络推理
3. **`_select()` 和 `_backup()`**（第 240–262 行）—— PUCT 和符号翻转
4. **`gomoku_zero/learner.py` 的 `train_step()`** —— 学的是什么
5. **`gomoku_zero/board.py`** —— 规则和候选剪枝

这五处看懂了，整个算法就通了。
