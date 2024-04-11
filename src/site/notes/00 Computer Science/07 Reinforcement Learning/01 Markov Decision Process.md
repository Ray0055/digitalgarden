---
{"dg-publish":true,"permalink":"/00-computer-science/07-reinforcement-learning/01-markov-decision-process/"}
---


[eth slide](https://stat.ethz.ch/education/semesters/ss2016/seminar/files/slides/seminar_week6_DynamicProgramming.pdf)

# From bandits to Markov Decision Processes
![Pasted image 20230729191740.png|500](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/Pasted%20image%2020230729191740.png)
- Markov Decision Processes and Markov Reward Processes are different!
![[Pasted image 20230501095332.png#C|Markov Reward Process|300]] 
![[Pasted image 20230501095740.png#C|Markov Decision Process|300]]


- Markov Reward Process vs.  Markov Decision Process
	- MRP没有Agent, 也就是没有action, 从一个状态跳到下一个状态完全由系统的概率决定; MDP有Agent, Agent会做出相应的action, action就是策略, 也是一个概率函数.
	- 不同于马尔可夫奖励过程，在马尔可夫决策过程中，通常存在一个智能体来执行动作。例如，一艘小船在大海中随着水流自由飘荡的过程就是一个马尔可夫奖励过程，它如果凭借运气漂到了一个目的地，就能获得比较大的奖励；如果有个水手在控制着这条船往哪个方向前进，就可以主动选择前往目的地获得比较大的奖励。马尔可夫决策过程是一个与时间相关的不断进行的过程，在智能体和环境 MDP 之间存在一个不断交互的过程。
![Pasted image 20230430101718.png|400](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/Pasted%20image%2020230430101718.png)

# Agent - Environment Interaction Loop
![Pasted image 20230729192719.png|400](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/Pasted%20image%2020230729192719.png)
This image describes the structure of Agent-Environment Loop which can be seen as the basic of reinforcement learning. 
- Agent will 
	- receives **State** over time t $S_{t}\in S$
	- selects an **Action** $A_{t}\in{\mathcal{A}}(S_{t})$
	- receives **Reward** $R_{t+1}\in \mathbb{R}$
- Repeat the decision process and we can get a sequence: $S_{0},A_{0},R_{1},S_{1},A_{1},R_{2},S_{2},\dots$

# Goals, Rewards and Returns
## Rewards
对于Reward有不同的定义：
![Pasted image 20230422183944.png|500](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/Pasted%20image%2020230422183944.png)
> 他们是包含关系，比如$r(s,a)$如果只考虑现在的状态为s，那么从上一步能到s的transition有很多，$s_1',s_2',s_3'...$ 都可以到现在的状态s，而$r(s,a,s')$ 只考虑一种。

![[Pasted image 20230525164004.png#C|Example for Reward|300]]
## Returns
- Our goal is not maximising immediate reward but cumulative rewards, which can be called as **return**.
- $R_t$ 表示在时刻t所获得的奖励**reward**, $G_t$ 表示从 t 时刻状态 $S_t$ 开始一直到终止，所有的奖励之和称之为 **return**
- Typically we use **discounted return**:$$G_{t}=R_{t+1}+\gamma R_{t+2}+\gamma^{2}R_{t+3}+\ldots=\sum_{i=0}^{\infty}\gamma^{i}R_{t+i+1}$$
	- Why we use discounted return?
		- it's used to handle the trade-off between immediate rewards and future rewards. 
		- **Preference for Immediate Rewards**: immediate rewards are considered more valuable than future rewards, so we use discounted rewards to weaken the impact future on the present.
		- **Infinite Horizon Problems**: some MDPs have infinite steps while return will be also infinite without discounted rate.
	- We can express $G_t$ in terms of future returns: $$\begin{array}{c}G_{t}&&=&&R_{t+1}+\gamma R_{t+2}+\gamma^{2}R_{t+3}+\gamma^{3}R_{t+4}\cdot\cdot\cdot\\ &&=&&R_{t+1}+\gamma(R_{t+2}+\gamma R_{t+3}+\gamma^{2}R_{t+4}\cdot\cdot\cdot)\\&&=&&R_{t+1}+\gamma G_{t+1}\end{array}$$
	- Although it is a sum of an infinite number of terms, it's still finite if the reward is nonzero and constant while $\gamma < 1$ $$G_{t}=\sum_{k=0}^{\infty}\gamma^{k}={\frac{1}{1-\gamma}}$$

## Episodic and non-episodic returns
任务通常被分为两种类型：Episode任务和Continuing任务。这两种任务类型区别在于环境和智能体的互动是否具有明确的开始和结束。

1. **Episode任务**：Episode任务有明确的开始和结束点，每个开始和结束的周期被称为一个Episode。在每个Episode结束后，智能体通常会被重置到一种初始状态或初始状态集，然后开始新的Episode。这种任务类型的回报（returns）是Episode性的（episodic），因为它们是在一个特定的Episode内计算的。

2. **Continuing任务**：对于Continuing任务，智能体和环境的互动是连续的，没有明确的开始和结束点。这种类型的任务通常**需要引入折扣因子来确保未来回报的总和是有限的**。这种任务类型的回报是非Episode性的（non-episodic）。

例如，棋类游戏通常被视为Episode任务，因为每局游戏有明确的开始（棋盘初始状态）和结束（一方获胜或和棋）。另一方面，股票投资可能被视为Continuing任务，因为投资决策和结果是连续不断的，没有明确的结束点。

# Policy and Value functions

## Policy
A policy g is a strategy that the agent uses to decide which action to take at each state which means a mapping from states to actions.

Almost all reinforcement learning use **value-funtion/ state-value function** to estimate *how good* the policy is. And *how good* is defined in terms of expected returns.

1. **Deterministic Policy**: A deterministic policy provides a specific action for each state. This can be represented as: $$a = \pi(s)$$This formula means that, given state $s$, the policy $\pi$ will always choose action $a$.
2. **Stochastic Policy**: A stochastic policy provides a probability distribution over possible actions for each state. This can be represented as:$$\pi(a\mid s)=\operatorname*{Pr}\left\{A_{t}=a\mid S_{t}=s\right\}$$
## Value function
分为两种 state value function and action value function
 
- value: 一个状态的期望回报（即从这个状态出发的未来累积奖励的期望）被称为这个状态的**价值**。 
- value function: 所有状态的价值就构成了价值函数。输入为某个状态，输出这个状态对应的价值。
- value和return之间的关系：我觉得return强调的是采取某一个确定的策略所获得的奖励之和，不考虑action的概率，只考虑这条策略涉及到的；value强调的是所有从这个状态出发所获得的所有的奖励之和，不考虑采取了何种策略，”计算这一步棋的代价期望“。
 
 >为什么要定义这么两个概念，不就是在状态s多执行了一个动作a么？这样做有什么意义呢？
> 这是两个概念，分别是状态价值V(s)和动作价值Q(s,a)，前者是对环境中某一个状态的价值大小做的评估，后者是对在某状态下的动作的价值大小的评估。概念类似，但主要区别应该是体现在用途以及算法上吧，比如对于离散型的动作空间，可以单纯基于动作Q值去寻优（DQN算法），如果是动作空间巨大或者动作是连续型的，那么可以判断状态价值并结合策略梯度来迭代优化（AC算法）
> [节选自知乎](https://zhuanlan.zhihu.com/p/34021617)

### State-value function: 
==Definition==: the expected reward of a state under a given policy. It will be used to evaluate how good is the state. 
$$v_{\pi}(s)=\mathbb{E}_{\pi}\left[G_{t}\mid S_{t}=s\right]=\mathbb{E}[R_{t+1}+\gamma R_{t+2}+\gamma^{2}R_{t+3}+\cdot\cdot\cdot\mid S_{t}=s]$$

> 为什么这里的state value需要求一个期望？因为策略 $\pi$ 本身是一个概率函数，比如我当前的策略是让机器人右转，如果当前是确定性策略，那么机器人右转的概率就是1，如果选择了随机性概率，那么右转的概率为0.8，剩下有0.2的概率左转；因此需要考虑到期望。

继续推导state-value function公式：$$\begin{aligned}
v_{\pi}(s) & =\mathbb{E}_{\pi}\left[G_{t} \mid S_{t}=s\right] \\
& =\mathbb{E}_{\pi}\left[R_{t+1}+\gamma G_{t+1} \mid S_{t}=s\right] \\
& =\sum_{a} \pi(a \mid s) \sum_{s^{\prime}} \sum_{r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma \mathbb{E}_{\pi}\left[G_{t+1} \mid S_{t+1}=s^{\prime}\right]\right] \\
& =\sum_{a} \pi(a \mid s) \sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma v_{\pi}\left(s^{\prime}\right)\right] \text { for all } s \in \mathcal{S}
\end{aligned}$$
- $\sum_a{\pi(a|s)}$ 表示采取策略pi时可能出现的所有的action
- $\sum_{s^{\prime}} \sum_{r} p\left(s^{\prime}, r \mid s, a\right)$ 表示在确定一种action时，出现的所有的下一状态$s'$ 和相应的reward $r$ 时对应的概率

> 等号2到等号3是如何实现的?
> 参考[知乎回答](https://www.zhihu.com/question/60018282)
> ![Pasted image 20230526133946.png|600](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/Pasted%20image%2020230526133946.png)



### Action value function
$$q_{\pi}(s,a)=\mathbb{E}_{\pi}[G_{t}|S_{t}=s,A_{t}=a]$$

继续推导公式$$\begin{aligned}
q_{\pi}(s, a) & =\mathbb{E}_{\pi}\left[G_{t} \mid S_{t}=s, A_{t}=a\right] \\
& =\mathbb{E}_{\pi}\left[\sum_{i=0}^{\infty} \gamma^{i} R_{t+i+1} \mid S_{t}=s, A_{t}=a\right] \\
& =\mathbb{E}_{\pi}\left[R_{t+1}+\gamma G_{t+1} \mid S_{t}=s, A_{t}=a\right] \\
& =\sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma \mathbb{E}_{\pi}\left[G_{t+1} \mid S_{t+1}=s^{\prime}, A_{t+1}=a^{\prime}\right]\right] \\
& =\sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma \sum_{a^{\prime}} \pi\left(a^{\prime} \mid s^{\prime}\right) q_{\pi}\left(s^{\prime}, a^{\prime}\right)\right]
\end{aligned}$$

## Relationship between $v_{\pi}$ and $q_{\pi}$ 
![e9f3dbbbacba337161eda5712585364.jpg](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/e9f3dbbbacba337161eda5712585364.jpg)
最后可以得到两者的关系: $$\begin{array}{c}{{\displaystyle v_{\pi}(s)=E_{\pi}\left[G_{t}\mid S_{t}=s\right]}}\\ {{\displaystyle=\sum_{a}\pi\big(a|s\big)q_{\pi}(s,a)}}\end{array}$$
## Backup diagram
![c02bfb6155efc83dcf44f38e101cc8a.jpg](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/c02bfb6155efc83dcf44f38e101cc8a.jpg)

# Bellman equation
[价值与贝尔曼方程](https://stepneverstop.github.io/%E4%BB%B7%E5%80%BC%E4%B8%8E%E8%B4%9D%E5%B0%94%E6%9B%BC%E6%96%B9%E7%A8%8B.html)
## Transition Matrix
![Pasted image 20230422185600.png](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/Pasted%20image%2020230422185600.png)
- $P(s_1|s_2)$ 表示从$s_2$ 到 $s_1$ 的概率
- 矩阵中所有的行的概率之和为1


## Bellman equation in matrix form
The Bellman equation can be expressed concisely using matrices:
$$V = R+\gamma P V$$
然后可以直接得到线性解:$$V = (1-\gamma P)^{-1}R$$![Pasted image 20230730110550.png|400](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/Pasted%20image%2020230730110550.png)
Computational complexity is $O(n^3)$ for n states
- 如何得到matrix form?
- ![Pasted image 20230526135803.png|400](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/Pasted%20image%2020230526135803.png)


## Bellman equation for optimal value
The core formula is: $$v_{*}(s)=\,\mathrm{max}\,q_{*}(s,a)$$
### Optimal equation for $v_*$

![Pasted image 20230430230905.png|400](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/Pasted%20image%2020230430230905.png)

注意这里是以action为对比项, 而不是以state项! 

### optimal action-value function 
$$\begin{align}{{q_{*}(s,a)=\mathbb{E}_{\pi_{*}}\left[R_{t+1}+\gamma\operatorname*{max}_{a^{\prime}}q_{*}(S_{t+1},a^{\prime})\mid S_{t}=s,A_{t}=a\right]}}\\ 
{{=\sum_{s^{\prime},r}p(s^{\prime},r\mid s,a)\left[r+\gamma\operatorname*{max}q_{*}(s^{\prime},a^{\prime})\right]}}\end{align}$$


![[Pasted image 20230730111109.png#C|Backup diagram for optimal value function]]
![[Pasted image 20230730111205.png#C|Backup diagram for optimal action-value function]]
![File_002.png](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/File_002.png)

> 注意这里的的两个max并不是一样的位置, 对于state value function而言, 取的是q的最大值, 因此应在最外层也就是第一层; 而对于state action value 而言, 同样取的是q的最大值, 而此时的q在第二层, 所以这个max要在括号里面. 
# Summary
![Pasted image 20230730105916.png](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/Pasted%20image%2020230730105916.png)

# Exercise 2
1. ![Pasted image 20230808115704.png](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/Pasted%20image%2020230808115704.png)![Pasted image 20230808115713.png](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/Pasted%20image%2020230808115713.png)

2. ![Pasted image 20230808120412.png](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/Pasted%20image%2020230808120412.png)![Pasted image 20230808120444.png](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/Pasted%20image%2020230808120444.png)用到了这个公式: $$G_{t}=\sum_{k=0}^{\infty}\gamma^{k}={\frac{1}{1-\gamma}}$$
3. ![Pasted image 20230808121857.png](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/Pasted%20image%2020230808121857.png)
$$\mathbb{E}[R_{t+1}]S_{t}= s]=\sum_{a}\pi(a|S_{t})\sum_{s^{\prime},r}p(s^{\prime},r|s,a)r$$

4. ![Pasted image 20230808124233.png](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/Pasted%20image%2020230808124233.png)![Pasted image 20230808124245.png](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/Pasted%20image%2020230808124245.png)
5. ![Pasted image 20230808125353.png](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/Pasted%20image%2020230808125353.png)
6. ![Pasted image 20230808130240.png](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/Pasted%20image%2020230808130240.png)Now it changes to a episode task not a continue task, we could calculate the new value function as followed:![Pasted image 20230808152933.png](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/Pasted%20image%2020230808152933.png)This means every new state value function will not add a same constant term rather term with respect to episode length T. 
7. ![Pasted image 20230808153623.png](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/Pasted%20image%2020230808153623.png)![File_001.png](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/File_001.png)
8. ![Pasted image 20230807165436.png](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/Pasted%20image%2020230807165436.png)
9. $\mu$ is probabilistic policy that is greedy, $\pi$ is a deterministic policy, prove $v_{\mu}(s) \geq v_{\pi}(s)$ for all state s![b2ef2f43b7c925b19ed5879d32eb797.png](/img/user/00%20Computer%20Science/07%20Reinforcement%20Learning/attachments/b2ef2f43b7c925b19ed5879d32eb797.png)