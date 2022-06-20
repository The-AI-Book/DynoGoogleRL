# DynoGoogleRL

## Observations, Actions and Rewards

- The observation is a RGB numpy array with shape of (150, 600, 3).
- The available actions are 0: do nothing, 1: jump, and 2: duck.
- A positive reward 0.01 is given when the dinosaur is alive; a negative penalty -1.0 is given when the dinosaur hits an obstable, which might be a cactus or a bird.

- For the DeepMind DQN recipe, where we give 4-stacked resized grayscaled frames (80, 160, 4) to the agent, we provide a wrapping method make_dino(). It also comes with a timer wrapper, which reports the interval between env.step().