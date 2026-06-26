## 0. Introduction
In this post, I plan to provide a deep dive into one of the coolest and most important papers in the early 2020s for ML and Robotics. The RT-x series of models basically showed that it is possible to leverage vision capabilities from pretrained models and Transformer's attention mechanism to learn mappings between natural-language instructions + frames to real robot actions. 

What is nice about this paper is the level of generalizabilty that their model learns. It can generalizes task completion for unseen objects, backgrounds and instructions in the training data and is very efficient (only a few Million parameters which allows it to run in around at ~3Hz i.e. 3 actions per second).

To get a better sense of how the model works, let's review each component of the paper: the research-question, data, model, training and the evaluation results. 

## 1. The problem it's trying to solve

Basically, the problem the paper addresses is: "Is it possible to design robotic policies from vision-language backbones? And how generalizable are they?" 

To answer these, it's important to understand what kind of behavior is expected from robotic policies look like in practice, i.e. what does the data look like.

---

## 2. The data

The training set consists of **~130,000 human demonstrations**, gathered over **17 months** with a fleet of **13 robots** (Everyday Robots mobile manipulators: a 7-DoF arm, a two-finger gripper, and a mobile base). A person teleoperates the robot, and the system records the camera stream and the control commands *at the same time* - so every demonstration is a synchronized "this is what I saw, this is what I did." Each episode is then annotated with a natural-language description of what was accomplished.

The demonstrations cover **744 distinct instructions**, grouped into skills like:

- **Pick** an object (130 instructions) - e.g. *"pick iced tea can"*
- **Move** object near object (337) - e.g. *"move pepsi can near rxbar blueberry"*
- **Place into receptacle** (84), **pick from receptacle and place on counter** (162)
- plus opening/closing drawers, knocking things over, placing items upright, and so on.

Notice the lopsided counts — "move near" alone is almost half the instructions. That breadth matters more than it looks, and we'll see later that **diversity of tasks turns out to matter more than sheer volume of data.**

### 2.1 Math representaiton

It helps to write down what one episode actually *is*. An episode is an instruction $i$ together with a sequence of observation–action pairs:

$$\{(x_t, a_t)\}_{t=0}^{T}$$

where $x_t$ is the camera image at time $t$, $a_t$ is the action the robot took, and $T$ is when the episode ends. Every episode in the dataset is a **success** (the robot finished the instruction), so there are no "what not to do" examples — only good behavior to imitate.

The robot doesn't get to see the whole future; at each step it decides based on the instruction and a short **history of the last six frames**. So the thing we're learning is a probability distribution over actions:

$$\pi\big(\cdot \mid i,\; \{x_{t-5}, \dots, x_t\}\big)$$

---

## 3. The model:

Given the data above, we can process them through a pipeline to output concrete robot decisions. This is full architecture used to produce these predictions. 
<div style="overflow-x: auto; max-width: 100%;">
  <img src="posts/assets/RT-1/architecture.png" alt="RT-1 architecture: instruction and 6 images flow through USE, FiLM EfficientNet-B3, TokenLearner, Transformer to an 11-D action" style="height: 460px; max-width: none; display: block;">
</div>

### 3.1 Inputs
We start with the language instruction as well as the history of the 6 latest frames taken by the robot's cameras. The laguage instruction comes in text (something like "pick apple from top drawer") and the robot frames consists of 6 images, each $300 \times 300 \times 3$ RGB.

### 3.2 Universal Sentence Encoder (USE)

The USEncoder is pretrained model that is useful for encoding short sentences into an embedding space. 
 
We would like to have a single token representation instead of multi-token since the instructions generally have capped size, and this will help on the next step when we are mixing the sentence's context to the vision backbone.

### 3.3 FiLM EfficientNet-B3 - language-aware vision-backbone

The six camera-frames from 3.1 inputs are sent to  an **EfficientNet-B3** (an ImageNet-pretrained convolutional network, 26 MBConv blocks, ~16M parameters). 

On its own, EfficientNet turns each $300 \times 300 \times 3$ image into a $9 \times 9 \times 512$ feature map — a small grid of rich visual descriptors. Flattened, that's **81 tokens of 512 dimensions** per image.

But instead of just using the vanilla EfficientNet, the paper introduces **FiLM** layers interwoven through the EfficientNet. FiLM ("Feature-wise Linear Modulation") takes the USE sentence embedding and uses it to **rescale and shift the image features as they're being computed**:

$$\text{FiLM}(x) = (1 + \gamma)\,x + \beta,$$

where $\gamma$ and $\beta$ are produced from the instruction embedding. 
*Why it's here:* the FiLM EfficientNet-B3 is suppose to provide an **early fusion** of language with vision. The text instruction reaches in and biases the feature extractor itself — so a feature map computed under *"pick the apple"* can already emphasize the apple.

### 3.4 **TokenLearner — the compression step.** 
For latency-bounded systems like robots, Eighty-one tokens per image, times six images, is a little too much for a Transformer to process very quickly. **TokenLearner** is a small attention module (~34K parameters) that learns to **soft-select the informative tokens**, squeezing 81 tokens down to just **8** per image.

<div style="overflow-x: auto; max-width: 100%;">
  <img src="posts/assets/RT-1/tokenlearner.png" alt="TokenLearner mechanism: learns 8 spatial attention maps over the 9x9 grid, weights and pools the feature map to produce 8 tokens" style="height: 420px; max-width: none; display: block;">
</div>

*How it works:* TokenLearner learns **8 spatial attention maps** over the $9 \times 9$ grid (a couple of convolutions followed by a spatial softmax). It then **multiplies the feature map by each mask and averages over space**, collapsing each masked grid into one 512-dim vector. Eight masks → eight tokens.

*Why it's here:* purely for **speed**. Fewer tokens means the Transformer has less to attend over, and that's crucial for allowing robots perform in real-time

**Concatenate + positional encoding.** Eight tokens per image across six frames gives $6 \times 8 = 48$ tokens, each $512$-dimensional — a $48 \times 512$ sequence. A positional encoding is added so the Transformer knows *which frame and which slot* each token came from (without it, the sequence is just an unordered bag).

### 3.5 **Transformer.** 
A **decoder-only Transformer**, 8 self-attention layers, ~19M parameters. TokenLearner already mixed information *spatially within* each frame; the Transformer's job is to mix information **across tokens and across time** - to reason about how the scene is changing over the six-frame history and what to do next. The sequence shape stays $48 \times 512$ through every layer.

Each self-attention layer can only relate tokens *pairwise in one hop*. It makes sense to stack them because it lets the model build up **multi-step, relational reasoning**: an early layer might line up the gripper with an object across frames, a later layer might use that to decide a direction of motion. 

### 3.6 **Action output.** 
Finally the Transformer emits the action. Each action is **11-dimensional**, and — here's the move — **each dimension is discretized into 256 bins**. So the output is effectively an $11 \times 256$ block of logits: eleven little 256-way classification problems. The eleven dimensions are:

- **7 for the arm:** $x, y, z$, roll, pitch, yaw, and gripper opening
- **3 for the base:** $x, y$, yaw
- **1 mode:** switch between *controlling the arm*, *controlling the base*, or *terminating* the episode

The whole model is **35M parameters**, runs **closed-loop at 3 Hz**, with **under 100 ms** of inference per step. 

---

## 4. Training

RT-1 is trained by **imitation learning** on timesteps sampled from several episodes collected on the training data. 

<div style="overflow-x: auto; max-width: 100%;">
  <img src="posts/assets/RT-1/training.png" alt="RT-1 training pipeline: 130K demos, USE frozen, FiLM-EfficientNet/TokenLearner/Transformer trained, cross-entropy loss" style="height: 420px; max-width: none; display: block;">
</div>

Concretely, we want the policy to assign **high probability to the action took by the teloperated robot**. Over the whole dataset $\mathcal{D}$ of $N$ episodes, we minimize the **negative log-likelihood** of the demonstrated actions:

$$\theta^{*} = \arg\min_{\theta} \sum_{n=1}^{N} \sum_{t=0}^{T} -\log \pi_{\theta}\big(a_t^{(n)} \mid i^{(n)},\, x_{t-5:t}^{(n)}\big)$$

To unpack what each symbol means:

- $\theta$ are the network weights; $\theta^*$ is the best setting we're solving for.
- The outer sum $\sum_{n=1}^N$ runs over all $N$ demonstrations; the inner sum $\sum_{t=0}^T$ runs over every timestep within an episode.
- $\pi_\theta(a_t^{(n)} \mid i^{(n)}, x_{t-5:t}^{(n)})$ is the probability the model assigns to **the human's action** $a_t^{(n)}$, given that episode's instruction and the six frames leading up to step $t$.
- $-\log(\cdot)$ is the standard "surprise" penalty: if the model already thought the right action was likely, the penalty is tiny; if it was caught off guard, the penalty is large.

Equivalently, over the data, we're minimizing the **expected negative log-probability of choosing the right action**:

$$\mathbb{E}_{(i,x,a)\sim \mathcal{D}}\big[-\log \pi_{\theta}(a \mid x, i)\big]$$

The negative-log-probability of choosing the right action is high when we're very uncertain over the correct box, and low when the model is sharply confident on the correct action. 


**Why this makes sense, and the intuition for the loop** 

The action factorizes into 11 independent 256-way bins, so this single loss corresponds to **eleven cross-entropy classification losses summed up**.

The training loop is then basically: 
- sample a batch of (instruction, 6-frame history, action) tuples
- Run through the model and predict the 11 action distributions 
- score them against the demonstrated bins and sum over the 11 bins.
- backprop the loss through the example. 

---

## 5. Evaluations

The comparisons are against three strong baselines — **Gato**, **BC-Z**, and a scaled-up **BC-Z XL** — all retrained on RT-1's own data so the comparison is fair. Performance is measured as **task success rate** across four buckets, over **3,000+ real-world trials**:

<div style="overflow-x: auto; max-width: 100%;">
  <img src="posts/assets/RT-1/experiments.png" alt="Bar chart: RT-1 vs Gato, BC-Z, BC-Z XL on seen tasks, unseen tasks, distractors, backgrounds" style="height: 460px; max-width: none; display: block;">
</div>

- **Seen tasks: 97%** (next best 72%). Over 200 training instructions, still with randomized object placement and lighting.
- **Unseen tasks: 76%** (next best 52%) — 21 brand-new instructions that recombine known skills and objects in ways never demonstrated. **+24 points** over the runner-up. Language conditioning is doing real work here: the model composes *"move X to Y"* for an X and Y it never saw moved together.
- **Distractor robustness: 83%** — cluttered scenes with up to nine distractor objects, some occluding the target.
- **Background robustness: 59%** — new kitchens, patterned tablecloths, unfamiliar counters.

Across the board RT-1 wins by the largest margins for unseen tasks and novel scenes. 

The authors also stress-tested generalization in graded **levels** — L1 (new layout + lighting), L2 (+ unseen distractors), L3 (drastically new settings, like an object near a sink it's never worked near). RT-1 degrades gracefully (88 → 75 → 50%).

## 6. Key Take-aways

- **Pretrained image encoders transfer to robotics.** An ImageNet-pretrained backbone (EfficientNet-B3) already produces visual features rich enough to drive control — you don't need to learn to see from scratch; you reuse a model that already knows how, and let language modulate it.
- **Architecture + attention + diverse data → generalization to unseen tasks.** The Transformer's attention mechanism, trained on a broad dataset spanning many tasks, objects, and scenes, learns to *recombine* known skills and objects — so it succeeds on instructions, objects, and environments it never saw demonstrated.
- **Training can be as simple as single-step behavioral cloning.** Behavioral cloning treats the demonstrations as a fixed, offline dataset and does pure supervised regression/classification onto the demonstrated actions — no RL, no reward model, no environment interaction. A strikingly simple recipe that nonetheless works very effectively at scale.

In summary, RT-1 is an existence proof that trainin one big model on broad dataset using backboned vision models and attention mechanisms allows you to learn real-world robotic actions.