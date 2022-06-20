from gc import callbacks
import constants
from tensorflow.keras.models import Sequential
import random
import numpy as np
import gym
from typing import List
 
def choose_next_action(env, model: Sequential, state: np.ndarray, rand_action_prob: float) -> int:
    """
    Input: TF Model, State and float.
    Output: 0 (do nothing) or 1 (jump)
    """
    new_action = 0
    if random.random() <= rand_action_prob:
        # Choose an action randomly.
        return np.random.randint(0, env.action_space.n)
    # Otherwise, use the model to make prediction of the Q-target.
    action_values = model.predict(state, verbose = 0)
    action_index = np.argmax(action_values[0])
    return action_index

def train(model: Sequential, mini_batch: List[tuple]):
    """
    Train the network on a single minibatch
    :para model: Sequential model
    :param mini_batch: the minibatch
    """
    for state, action, reward, next_state, done in mini_batch:
        target = reward
        if not done:
            target = (reward + 
                      constants.GAMMA_FACTOR * np.amax(model.predict(next_state, verbose = 0)[0]))
        target_f = model.predict(state, verbose = 0)
        target_f[0][action] = target
        model.fit(state, target_f, epochs = 1, verbose = 0)
    return model