## Deep Reinforcement Learning Nano Degree
### Project 2 - Continuous Control

This repository contains the solution for the Project 2 of the Deep Reinforcement Learning Nano Degree from Udacity. The goal of this project is to train an reinforcement learning agent to for a continous control problem.

The Reacher environment consists in a dual-jointed robotic arm that needs to follow a target location. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible. Its observation space has 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. The action space corresponds to 4 continuous variables varying between [-1, 1], where each corresponds to torque applicable to two joints.

The task is episodic, and the used environment provides 20 arms simultaneously. So in order to solve the environment, the agent must get an average score averaged over the 20 agents of +30 over 100 consecutive episodes.

### Getting Started
1. Download the environment from one of the links below. You need only select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Place the file in the `drlnd-p2` GitHub repository, in the `reacher/Reacher_Linux` folder, and unzip (or decompress) the file.

2. Install the dependencies
```
pip install -r requirements.txt
cd ./python
pip install .
```

### Reproduce
Open and execute the `reacher/Report.ipynb` notebook.
