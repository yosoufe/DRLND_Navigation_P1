# Project 1: Navigation

This is my submission to the first project of [Deep Reinforcement Learning 
Nanodegree by Udacity](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

## Project:
### Goal: 
The goal is to train an RL agent to receive the highest reward in the given environment.

### Environment:

![Banana Environment](images_videos/environment.gif "Banana Environment")

The environment that is used here is custom version of 
[Banana environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector) of 
[Unity ML-Agents Toolkit](https://unity3d.com/machine-learning). You can 
find the links to download the environment from 
[readme file in Udacity repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation#getting-started).

* The environment has only one agent.
* Rewards:
    * +1 if it collects a yellow banana.
    * -1 if it collects a blue banana.
* Observation: The observation is a vector of 37 elements containing 
the agent's velocity, along with ray-based perception of objects around 
agent's forward direction.
* Actions: Action can be only an integer between 0 and 3:
    * `0` - move forward.
    * `1` - move backward.
    * `2` - turn left.
    * `3` - turn right.
* The problem is considered to be solved when the agent gets 
an average score of +13 over 100 consecutive episodes.

### Getting Started:

#### Python Dependencies:
* numpy: `pip install numpy`
* pytorch: [Installation Manual](https://pytorch.org/get-started/locally/)
* Jupyter notebook
* tqdm: `pip install tqdm`
* ipywidgets: `pip install ipywidgets` [Manual](https://ipywidgets.readthedocs.io/en/latest/user_install.html)
* mlagents_envs: `pip install mlagents-envs` or from source:
```bash
git clone https://github.com/Unity-Technologies/ml-agents.git
cd ml-agents/ml-agents-envs
pip install .
```

#### Clone the Repo:
I have some of the files on lfs. If you are not familiar with 
[git lfs](https://git-lfs.github.com/), please download the project
as zip file.

If you have `git lfs` installed, you can simply clone the repo like:
```bash
git clone https://github.com/yosoufe/DRLND_Navigation_P1.git
```

#### Prepare the Environment:
You need to download the environment from the links
[here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation#getting-started)
according to your OS and pass the path of the environment in the following jupyter notebooks.

#### How to train:
`Train the agent.ipynb` is the notebook that does the training and save the trained model.
You need to replace the pass of the simulator in one of the cells like bellow to point to where you 
placed your simulator:
```python
env = UnityEnvironment(file_name="../Banana_Linux/Banana.x86_64", no_graphics=True)
``` 

#### How to run the trained model:
`Load and Evaluate The Agent.ipynb` is loading the agent and run it in the environment with graphics.
You need to replace the pass of the simulator in one of the cells like bellow to point to where you 
placed your simulator:
```python
env = UnityEnvironment(file_name="../Banana_Linux/Banana.x86_64", no_graphics=False)
``` 

### Report:
You can find the report [here](report.md).