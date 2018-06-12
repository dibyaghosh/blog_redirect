---
layout: post
title: "Control as Inference"
description: "Reinforcement learning is traditionally inspired by maximizing reward. This article presents the interpretation of reinforcement as doing inference in a probabilistic graphical model, as introduced in the control-as-inference literature."
tagline: "Stuffing RL into a graphical model"
author: "Dibya Ghosh"
categories: rl
tags: [rl, variational]
image: controlasinference.png
---

A [paper of mine recently](https://arxiv.org/abs/1805.11686) recently proposed an algorithm to do weakly-supervised inverse RL (this is a gross generalization: do read the paper!). The algorithm is derived through an interesting framework called "control as inference", which analogizes (max-ent) reinforcement learning as inference in a graphical model. This framework has been gaining traction recently, and it's been used to justify many recent contributions in IRL ([Finn et al](https://arxiv.org/abs/1603.00448), [Fu et al](https://arxiv.org/abs/1710.11248)), and some interesting RL algorithms like Soft Q-Learning([Haarnoja et al](https://arxiv.org/abs/1702.08165)). 

I personally think the framework is very cute, and it's an interesting paradigm which can explain some weird quirks that show up in RL. This document is a writeup which explains exactly what "control as inference" is. Once you've finished reading this, you may also enjoy [this lecture](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_11_control_and_inference.pdf) in Sergey Levine's CS294-112 class, or his [primer on control as inference](https://arxiv.org/abs/1805.00909) as a more detailed reference. 

### The MDP

In this article, we'll focus on a finite-horizon MDP with horizon $T$ : this is simply for convenience, and all the derivations and proofs can be extended to the infinite horizon case simply. Recall that an MDP is $(\mathcal{S}, \mathcal{A}, \mathcal{T}, \rho, R)$ , where $\mathcal{S,A}$ are the state and action spaces, $T(\cdot \vert s,a)$ is the transition kernel, $\rho$ the initial state distribution, and $R$ the reward.  

## The Graphical Model

Trajectories in an MDP as detailed above can be modelled by the following graphical model.

<img style='width:100%' src="{{ "/assets/posts/controlasinference/state_action.png" | absolute_url }}">

For each timestep $t$, we have a state variable $S_t$, an action variable $A_t$. 

For the state variables, we define the distribution of $S_0$  to be $\rho(s)$ . For subsequent $S_{t}$, we define the distribution using the transition probabilities ($P(S_{t+1} = s' \vert S_{t}=s, A_t = a) = T(s' \vert a,s)$). 



For the action variables, we assume that actions $A_t$ are sampled uniformly in the graphical model  ($P(A_t = a) = C$). It may seem odd that the actions are sampled uniformly, but don't worry! These are only _prior_ probabilities, and we'll get interesting action distributions once we start conditioning (Hang tight!)  

The probability of a trajectory $\tau = (s_0, a_0, s_1, a_1 , \dots s_T,a_T)$ in this model factorizes as 

$$\begin{align*}P(\tau) &= P(S_0 = s_0) \prod_{t=0}^{T-1} P(A_t = a_t)P(S_{t+1} = s_{t+1} | S_t = s_t, A_t = a_t)\\ &= C^T \left(\rho(s_0) \prod_{t=0}^{T-1} T(s_{t+1} \vert s_t, a_t)\right)\\  &\propto \left(\rho(s_0)\prod_{t=0}^{T-1} T(s_{t+1} | s_t,a_t)\right) \end{align*}$$

The probability of a trajectory in our graphical model is directly proportional to it's probability under the system dynamics. In the special case that dynamics are deterministic, then $P(\tau) \propto \mathbb{1} \\{\text{Feasible}\\}$ (that is, all trajectories are equally likely.

### Adding Rewards

So far, we have a general structure for describing the likelihood of trajectories in an MDP, but it's highly uninteresting since at the moment, all trajectories are equally likely. To highlight interesting trajectories, we'll introduce the concept of *optimality*. 

We'll say that an agent is **optimal** at timestep $t$ with some probability which depends on the current state and action : $P(\text{Optimal at } t) = f(s_t,a_t)$. We'll embed optimality into our graphical model with a binary random variable at every timestep $e_t$, where $P(e_t = 1 \vert S_t=s_t, A_t=a_t) = f(s_t,a_t)$. 

While we're at it, let's define a function $r(s,a)$ to be $r(s_t,a_t) = \log f(s_t,a_t)$ . The notation is very suggestive, and indeed we'll see very soon that this function $r(s,a)$ plays the role of a reward function. 

The final graphical model, presented below, ends up looking much like one for a Hidden Markov Model.

<img style='width:100%' src="{{"/assets/posts/controlasinference/state_action_reward.png" | absolute_url }}">

For a trajectory $\tau$, let's find the probability that it's optimal at all timesteps

$$\begin{align*}P(\text{All } e_t=1 | \tau) &= \prod_{t=0}^T P(e_t = 1 \vert S_t=s_t, A_t=a_t) \\ &= \prod_{t=0}^T f(s_t,a_t) \\ &= \prod_{t=0}^T \exp{r(s_t,a_t)} \\ &=  \exp (\sum_{t=0}^T r(s_t,a_t))\end{align*}$$

The probability that a trajectory is optimal is exponential in the total "reward" received in the trajectory. Now, the optimal agent is one that maximizes it's chances of being optimal at all time steps.  Let's investigate the probability of a trajectory $\tau$ under the constraint that the agent is always optimal.

$$\begin{align*}P(\tau \vert \text{All } e_t =1) &= \frac{P(\text{All } e_t =1 \vert \tau)P(\tau)}{P(\text{All } e_t=1)} \\ &\propto P(\text{All } e_t =1 \vert \tau)P(\tau) \\ &\propto \exp(\sum_{t=0}^T r(s_t,a_t))P(\tau)\end{align*}$$ 

Under deterministic dynamics, the probability of a trajectory is given by $P(\tau \vert \text{All } e_t =1) \propto \exp(\sum_{t=0}^T r(s_t,a_t))$, which is a special form of an energy-based model.

## The Old School Approach: Inference for Planning

What do good actions look like under this model of optimality? Let's try to compute $P(A_t=a \vert S_t=s, \text{All } e_t =1)$.  

Recall that since the environment is Markovian, we can write this as $P(A_t=a \vert S_t=s, e_{t:T} =1)$. 

$$\begin{align*} P(A_t =a \vert S_t=s, e_{t:T}=1) &= \frac{P(A_t =a, e_{t:T}=1 \vert S_t=s)}{P(e_{t:T}=1\vert S_t=s)}\\ &= \frac{P(e_{t:T}=1 \vert A_t =a,  S_t=s) P(A_t=a \vert S_t = s)}{P(e_{t:T}=1\vert S_t=s)}\\ &= \frac{P(e_{t:T}=1 \vert A_t =a,  S_t=s) C}{P(e_{t:T}=1\vert S_t=s)}\\\end{align*}$$



Thus, in order to find optimal actions, we'll first need to find expressions of probabilities of future optimality $P(e_{t:T}=1 \vert A_t =a,  S_t=s)$ and $P(e_{t:T}=1 \vert S_t=s)$. We'll derive these probabilities in a recursive style


$$
\begin{align*} 
P(e_{t:T} = 1&\vert A_t =a, S_t=s)\\ &= \int_{\mathcal{S}} P(e_{t:T}=1, S_{t+1}=s' \vert S_t=s, A_t=a) ds'\\
&= \int_{\mathcal{S}} P(e_t = 1 | S_t=s, A_t=a)P(e_{t+1:T}=1, S_{t+1}=s' \vert S_t=s, A_t=a) ds'\\
&= P(e_t = 1 | S_t=s, A_t=a) \int_{\mathcal{S}} P(e_{t+1:T}=1 \vert S_{t+1}=s') P(S_{t+1} = s' \vert S_t=s, A_t=a) ds'\\
&= e^{r(s,a)} \mathbb{E}_{s' \sim T(\cdot \vert s,a)}[P(e_{t+1:T}=1 \vert S_{t+1}=s')]\\
P(e_{t:T} = 1&\vert S_t=s)\\ &= \int_{\mathcal{A}} P(e_{t:T} = 1, A_t=a \vert S_t=s) da\\
&= \int_{\mathcal{A}} P(e_{t:T} = 1 \vert A_t=a , S_t=s) P(A_t=a) da \\ 
&= \mathbb{E}_{a}[P(e_{t:T} = 1 \vert A_t=a, S_t =s)]\\
\end{align*}
$$




Well, that looks ugly: maybe the expressions will look cleaner in log-probability space. Let's define $Q_t(s,a) = P(e_{t:T} = 1\vert A_t =a, S_t=s)$ and $V_t(s) = P(e_{t:T} = 1 \vert S_t=s)$: very suggestively named for a good reason. Rewriting the above expressions with $Q(\cdot, \cdot)$ and $V(\cdot)$:

$$Q_t(s,a)  = r(s,a) + \log  \mathbb{E}_{s' \sim T(\cdot \vert s,a)}[e^{V_{t+1}(s')}]$$

$$V_t(s) = \log \mathbb{E}_a [e^{Q_t(s,a)}]$$

Recalling that the $\log \mathbb{E}[\exp(f(X))] $ function acts as a "soft" maximum operation (we'll denote it as $\text{soft} \max$ - but don't get it confused with the actual softmax operator), these get rewritten to

$$Q_t(s,a) = r(s,a) + \text{soft} \max_{s'} V_{t+1}(s')$$

$$V_t(s) = \text{soft} \max_{a} Q_{t}(s,a)$$

These recursive equations look very much like the Bellman backup equations! 

These are the **soft Bellman backup equations**. They differ from the traditional Bellman backup in two ways:

1. The value function is a "soft" maximum over actions, not a hard maximum.
2. The q-value function is a "soft" maximum over next states, not an expectation: this makes the Q-value "optimistic wrt the system dynamics" or "risk-seeking". It'll favor actions which have a low probability of going to a really good state over actions which have high probability of going to a somewhat good state. When dynamics are deterministic, then the Q-update is equivalent to the normal backup: $Q_t(s,a) = r(s,a) + V_{t+1}(s')$.

Armed with this notation, let's  revisit what the optimal action distribution is:
$$
\begin{align*}
P(A_t =a \vert S_t=s, e_{t:T}=1) &= \frac{P(e_{t:T}=1 \vert A_t =a,  S_t=s) C}{P(e_{t:T}=1\vert S_t=s)}\\
&= \frac{e^{Q_t(s,a)}C}{e^{V_t(s)}}\\
&= C * \exp(Q_t(s,a) - V_t(s))\\
&\propto \exp(A_t(s,a))
\end{align*}
$$


If we define the *advantage* $A_t(s,a) = Q_t(s,a) - V_t(s)$, then we find that the optimal probability of picking an action is simply proportional to the advantage! 

[Haarnoja et al](https://arxiv.org/abs/1702.08165) use philosophies similar to this to derive an algorithm called Soft Q-Learning. In their paper, they show that the soft bellman backup update is a contraction, and so Q-learning with the soft backup equations have the same convergence guarantees that Q-learning has in the discrete case. Empirically, they show that this algorithm can learn complicated continuous control tasks with high sample efficiency. In follow-up works, they deploy the algorithms [on robots](https://arxiv.org/abs/1803.06773) and also present  [actor-critic methods](https://arxiv.org/abs/1801.01290)  in this framework.

## The New School Approach: Variational Inference

Let's try to look at inference in this graphical model in a different way. Instead of doing exact inference in the original model to get a policy distribution, we can attempt to learn a *variational* approximation to our intended distribution $q_{\theta}(\tau) \approx P(\tau \vert \text{All } e_t=1)$.

The motivation is the following: we want to learn a policy $\pi(a \vert s)$ such that sampling actions from $\pi$ causes the trajectory distribution to look as close to $P(\tau \vert \text{All }e_t = 1)$ as possible. To capture this in a variational method, we'll define a variational distribution $q_{\theta}(\tau)$ as follows:

$$q_\theta(\tau) = P(S_0 = s_0) \prod_{t=0}^T q_{\theta}(a_t \vert s_t) P(S_{t+1} = s_{t+1} \vert S_{t} = s_t, A_t = a_t) = \left(\prod_{t=0}^T q(a_t | s_t)\right) P(\tau)$$

This variational distribution can change the distribution of actions, but fixes the system dynamics in place. Intuitively, we should think of variational inference in this setting as finding the function $q_{\theta}(a \vert s)$ which minimizes the KL divergence with the true distribution. 

$$\min_{\theta} D_{KL}(q_{\theta}(\tau) \| P(\tau \vert \text{All } e_t = 1))$$  

So what does this look like? Let's do the math! Remember from the first section that $P(\tau \vert \text{All } e_t=1) = P(\tau) \exp(\sum_{t=0}^T r(s_t,a_t))$


$$
\begin{align*}
D_{KL}(q_{\theta}(\tau) \| P(\tau \vert \text{All } e_t = 1)) &= -\mathbb{E}_{\tau \sim q}[\log \frac{P(\tau \vert \text{All } e_t = 1)}{q_{\theta}(\tau)}]\\
 &= -\mathbb{E}_{\tau \sim q}[\log \frac{P(\tau) \exp(\sum_{t=0}^T r(s_t,a_t))}{ P(\tau)\left(\prod_{t=0}^T q(a_t | s_t)\right)}]\\
 &= -\mathbb{E}_{\tau \sim q}[\log \frac{\exp(\sum_{t=0}^T r(s_t,a_t))}{\prod_{t=0}^T q(a_t | s_t)}]\\
&= -\mathbb{E}_{\tau \sim q}[\log \frac{\exp(\sum_{t=0}^T r(s_t,a_t))}{\exp (\sum_{t=0}^T \log q(a_t | s_t)}]\\
 &=  -\mathbb{E}_{\tau \sim q}[ \sum_{t=0}^T  r(s_t,a_t) - \log q(a_t | s_t))]\\
\arg\min_{\theta} D_{KL}(q_{\theta}(\tau) \| P(\tau \vert \text{All } e_t = 1)) &= \arg \max_{\theta} \mathbb{E}_{\tau \sim q}[ \sum_{t=0}^T  r(s_t,a_t) - \log q(a_t | s_t))]
\end{align*}
$$
Recalling that $-\log q(a_t | s_t)$ is a point estimate of the entropy of $q_{\theta}$: $\mathcal{H}(q(\cdot \vert s))$, the best policy $q_{\theta}(a|s)$ is thus the one that maximizes expected reward with an entropy bonus, which is the objective for **maximum entropy reinforcement learning.** Thus, performing variational inference with this distribution to minimize the KL divergence with the optimal trajectory distribution is equivalent to doing reinforcement learning in the max-ent setting!



