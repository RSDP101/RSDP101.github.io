### The paper: RT-1 Robotics transformer for real-world control at scale

Here is the URL: <https://arxiv.org/pdf/2212.06817>

### 1-line summary

They built RT-1, a model that takes as input sequence of images (history of robot's camera view) + textual description ("pick up object A") and outputs robot actions.

### Main Problem

The main problem they are trying to solve is how to build generalizable robotics policies and whether this is possible to do by leveraging world-understanding from transformer models.

### Data

The human controls the robot itself through teleoperation. The training data was in total 130K demonstrations.

The demonstrations were very simple things like:

- Move object Near Object (e.g. move pepsi can near blueberry bar)
- Pick object from Receptacle and place it on counter (pick chip bag from paper bowl and place it on counter)
- ...

Each episode is annotated with textual description. The robot is being teleoperated, so the system records both the camera and the control commands at the same time.

An episode can be thought of as an initial text instruction $i$ and a sequence of states and actions $\{(x_t, a_t)\}$ for $t = 0, 1, \cdots, T$.

At each time step, the robot uses the history of images $x_{t-5}, \cdots, x_{t}$ and instruction $i$ to produce a distribution over the list of valid actions:

$$\pi(\,\cdot \mid i,\, \{x_{t-5}, \cdots, x_t\})$$

Policy $\pi$ is parametrized by a transformer.

### The model

#### Inputs

- Short sequence of 6 images + natural language instructions.

#### Outputs

- An action for the robot at each time step. Each action is 11-dimensional (x, y, z, roll, pitch, etc.)

#### The pipeline

<div style="overflow-x: auto; max-width: 100%; border: 1px solid #333; border-radius: 6px;">
  <img src="posts/assets/RT-1/rt1_training_pipeline_serpentine.png" alt="RT-1 training pipeline" style="height: 420px; max-width: none; display: block;">
</div>

Or also:

<div style="overflow-x: auto; max-width: 100%; border: 1px solid #333; border-radius: 6px; margin-top: 1rem;">
  <img src="posts/assets/RT-1/rt1_training_pipeline_flow_graph.png" alt="RT-1 training pipeline (flow graph)" style="height: 420px; max-width: none; display: block;">
</div>

### Training (imitation learning)

Given a dataset $\mathcal{D}$, a policy $\pi$ is trained.

Imitation learning is basically composed of minimizing the negative log-likelihood of actions $a_t$:

$$\theta^{*} = \arg\min \sum_{n=1}^{N} \sum_{t=0}^{T} -\log \pi_{\theta}\bigl(a_t^{(n)} \mid i^{(n)}, x_{t-5,\,\cdots,\,t}^{(n)}\bigr)$$

This is related to the loss below, which measures the "expected negative log-probability of the model choosing the right action":

$$\mathbb{E}_{(i, x, a) \sim \mathcal{D}}\;\bigl[-\log \pi_{\theta}(a \mid x, i)\bigr]$$

### Evaluation

- RT-1 solves **97%** of the seen tasks
- RT-1 solves **76%** of unseen tasks
- Robustness: **83%** of the distractor robustness tasks and **59%** of the background robustness tasks
