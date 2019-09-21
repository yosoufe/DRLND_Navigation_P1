# Report

## Introduction:

The environment and the goal is introduced in main readme file [here](readme.md). 

## Learning Algorithm:

Deep Q-Learning is used to train an RL agent to score higher in the environment.

### Deep Q-Learning:

Deep Q-learning method is introduced in 
[this paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) 
where a same algorithm with the same neural network architecture and same hyperparameters
are trained in different Atari Game environment and it could outperform human in some of the environment.

In this algorithm there is a neural network from states to actions. The Neural Network learns the value 
of each action at each state. For example in this project the input of the neural network has dimension of 37 
because the environment's state has 37 dimension and the output of the network is an array of 4 values because
there are 4 possible actions for the agent. Basically the neural network is approximating the following
function:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;Q^\pi(s,a)=\mathbb{E}_{s'}\[r+\gamma\max_{a'}Q(s',a'|s,a)\]"/>

The vanilla form of the algorithm is as the following steps in loop until convergence 
(Source: [CS285 at UC Berkeley](http://rail.eecs.berkeley.edu/deeprlcourse-fa18/static/slides/lec-7.pdf)):
<ol>
  <li>take some action <img align="middle" src="https://latex.codecogs.com/svg.latex?\Large&space;a_i"/> 
  and observe <img src="https://latex.codecogs.com/svg.latex?\Large&space;(mathbf{s}_i,\mathbf{a}_i,\mathbf{s}'_i,r_i)"/> </li>
  <li><img align="middle" src="https://latex.codecogs.com/svg.latex?\Large&space;y_i=r(\mathbf{s}_i,\mathbf{a}_i)+\gamma\max_{\mathbf{a}'}Q_{\phi}(\mathbf{s}'_i,\mathbf{a}'_i)"/></li>
  <li><img align="middle" src="https://latex.codecogs.com/svg.latex?\Large&space;\phi\leftarrow\phi-\alpha\frac{dQ_{\phi}}{d\phi}(\mathbf{s}_i,\mathbf{a}_i)(Q_{\phi}(\mathbf{s}_i,\mathbf{a}_i)-\mathbf{y}_i)"/></li>
</ol>

#### Some Tricks:
There are some tricks required to make this algorithm work

##### Experience Replay:
Learning on single sample is usually ending up having a very high variance. Therefore it is essential to 
have a buffer to save the experiments. Then on each step random samples are drawn from the buffer and 
the network is trained on them. This is like we are replaying the old experience.

#####  Fixed Q-Targets
In the steps above, it is mathematically wrong to ignore the dependency of target value 
<img align="middle" src="https://latex.codecogs.com/svg.latex?\Large&space;y_i"/> with respect to 
the neural network parameters in gradient descent optimization. In order to resolve this issue,
a second neural network as target network can be considered to calculate the target value with 
the same architecture of the first network and its parameters are kept fixed. Then the parameters
of the target network is updated from the first network every C steps.

In this project we did not do this exactly, but we update the parameters of target network with linear
combination of old values and the parameters of agent's network with large weight on 
the old values. We called it `soft_update`.

```python
def soft_update(self, local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
```

### Model:
Model basically consists of 3 linear layers with `relu` as activation function for 
the first and second layer. 37 is the number of states and 4 is the action size.
```python
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

### Hyperparameters and Tuning:
Basically I did not change that much the hyperparameters from the previous exercises. Because on first few
tries I could get the algorithm working and the parameters looked ok. The only difference was that
the simulation was taking longer time but the network was learning relatively faster.

The first time that I get a model that feels success was two train the agent for 2000 episodes but 
only allowance for 100 steps. In this way the simulation was done faster. Because usually environment
allowed 300 steps. In training mode, the simulations was taking almost one second for each hundred 
steps. This model is saved in a file called `checkpoint_not_criteria.pth` because no criteria was used
to stop the training.

The second try I let the environment decide when each episode should end. I know that the maximum steps
for each episode was 300 steps. So each episode was taking 3X more time. But I used a criteria to stop 
the training. The training was stopped when the agent gets an average score of +18 over 100 consecutive episodes.
I know that the goal of this project is +13 score but I saw that if I target +18 I get a much better
agent. The following graph is showing the rewards vs episodes. As you can see the agent got to the target score 
in a bit more than 300 episodes. This model is saved as `checkpoint.pth`:

![Reward Graph](images_videos/Learning%20Curve.png "Learning Curve or Rewards vs. Episodes") 

## Result:

Here is a video of my trained agent collecting the yellow bananas: https://youtu.be/B9ldSkz2VBs 

And the gif version of the video:

[![Banana Environment](images_videos/RLP1.gif "My Trained Agent, Click for Youtube Video")](https://youtu.be/B9ldSkz2VBs)

### Future Ideas:
* Using image as the observation rather the ray-casted measurements. This would be more difficult 
because the state space become suddenly a lot larger as the input of the network would be images.
Convolution Neural Networks would be a good choice in that case.
* Using Dueling Networks and Double DQN.
* Prioritized experience replay would increase the efficiency.
* Using RNNs as the model.