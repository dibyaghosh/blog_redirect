---
qlayout: post
title: "Control as Inference"
description: "Reinforcement learning is traditionally inspired by maximizing reward. This article presents the interpretation of reinforcement as doing inference in a probabilistic graphical model, as introduced in the control-as-inference literature."
tagline: "What can you not put in a graphical model?"
author: "Dibya Ghosh"
categories: rl, px
tags: [rl, variationaluickml.png
---



I was recently part of a paper, "Variational Inverse Control with Events" (Fu et al 2018), which proposes a new algorithm for inverse RL much more weakly supervised than standard IRL algorithms. The algorithm presents itself in a framework called "control as inference", which analogizes (max-ent) reinforcement learning as inference in a graphical model. This framework has been a cornerstone of many modern RL and IRL algorithms, including Guided Cost Learning (Finn et al), Soft Q Learning (Haarnoja et al), and Soft Actor-Critic (Haarnoja et al), so I thought it would be a good idea to have a writeup which explains exactly what "control as inference" is.

Before I get started, it'd be amiss of me not to mention Sergey Levine's CS294-112 lecture for a video (link here) , or his primer on control as inference as a more detailed reference (link here). 

### The MDP

In this article, we'll focus on a finite-horizon MDP with horizon $T$ : this is simply for convenience, and all the derivations and proofs can be extended to the infinite horizon case simply. Recall that an MDP is $(\mathcal{S}, \mathcal{A}, \mathcal{T}, \rho, R)$ , where $\mathcal{S,A}$ are the state and action spaces, $T(\cdot | s,a)$ is the transition kernel, $\rho$ the initial state distribution, and $R$ the reward.  

### The Graphical Model

Trajectories in an MDP as detailed above can be modelled by the following graphical model. 

At each timestep $t​$, we have a state variable $S_t​$, an action variable $A_t​$. We match the distribution of $S_0​$  to the initial state distribution of the MDP ($P(S_0 = s_0) = \rho(s_0)​$ ) , and use the transition kernel to describe the other state variables ($P(S_{t+1} = s' | S_{t}=s, A_t = a) = T(s'|a,s)​$). . We place a uniform prior distribution on actions $A_t​$ ($P(A_t = a) = C​$),  so that the probability of a trajectory in our graphical model is directly proportional to it's probability under the system dynamics.

The probability of a trajectory $\tau = (s_0, a_0, s_1, a_1 , \dots s_T,a_T)$ factorizes as 

$$\begin{align*}P(\tau) &= P(S_0 = s_0) \prod_{t=0}^{T-1} P(A_t = a_t)P(S_{t+1} = s_{t+1} | S_t = s_t, A_t = a_t)\\ &= C^T \left(P(S_0=s_0) \prod_{t=0}^{T-1} P(S_{t+1} = s_{t+1} | S_t = s_t, A_t = a_t\right)\\  &\propto \left(P(S_0=s_0) \prod_{t=0}^{T-1} P(S_{t+1} = s_{t+1} | S_t = s_t, A_t = a_t\right) \end{align*}$$

If dynamics are deterministic, then $P(\tau) \propto \mathbb{1} \{\text{Feasible}\}$.

This graphical model thus provides a general structure for action and exploration in an MDP, as though there were no reward.  To bring in rewar 