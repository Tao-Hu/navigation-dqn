# Environment

This project aims to train an agent to navigate in a large, square world, and collect bananas.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* 0: move forward
* 1: move backward
* 2: turn left
* 3: turn right

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

The environment used in this project is adopted from Banana Collector environment of [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents). To run the environment, user don't need to install the Unity, but download the environment configuration files from [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation#getting-started).

# Dependence

This project is build on Python 3.6. Following dependencies are required:

* unityagents==0.4.0
* torch==1.4.0
* tensorflow==1.7.1
* numpy>=1.11.0

# Instruction

* `Navigation.ipynb`: the driving file to set up the environment and train the agent with different algorithms, include:
    - DQN with fixed Q targets
    - Double DQN
    - DQN with prioritized experience replay
* `dqn_agent.py`: define Agent class
* `model.py`: define neural netowrk architecture for estimating Q functions