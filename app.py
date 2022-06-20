import gym
import gym_chrome_dino
import constants
from model import get_model, save_model
from agent import choose_next_action, train
from preprocessing import process_image
import numpy as np
import random
from collections import deque

#print(model.summary())


def play():
    experience_replay = deque(maxlen = constants.MAX_LENGTH_MEMORY_REPLACE)
    env = gym.make("ChromeDinoNoBrowser-v0")
    model = get_model(env)

    total_reward = 0
    episode = 1
    time_step = 0
    epsilon_prob = constants.EPSILON_VALUE
    episode_lengths = list()

    done = True
    while True:
        if done:
            time_step = 0
            last_state = process_image(env.reset())

            if episode % constants.EPSIODES_TO_SAVE_MODEL == 0:
                save_model(model)
        
        # Select next action.
        last_action = choose_next_action(env, model, last_state, epsilon_prob)

        # Get info from the environment.
        current_state, reward, done, info = env.step(last_action)
        current_state = process_image(current_state)
        total_reward += reward
        time_step += 1

        # If terminal
        if done:
            episode += 1
            episode_lengths.append(time_step)
            print("Episose: %s; Steps before fail %s; Epsilon: %.2f reward %s" %
                 (episode, time_step, epsilon_prob, total_reward))
            total_reward = 0
        
        # Store the transition in previous observations.
        observation = (last_state, last_action, reward, current_state, done)
        experience_replay.append(observation)

        # If the size of the experience replay is enough.
        if len(experience_replay) >= constants.MIN_EXPERIENCE_REPLAY_SIZE:
            # Get mini batch from the experience replay.
            mini_batch = random.sample(experience_replay, constants.MINI_BATCH_SAMPLE_SIZE)

            # Train the network.
            model = train(model, mini_batch)

            # Increase time step by 1.
            time_step += 1

        # Last state now is current state.
        last_state = current_state

        # Gradully reduce the probability of a random action 
        # Starting from 1 and going to 0 
        if epsilon_prob > 0 and len(experience_replay) > constants.MIN_EXPERIENCE_REPLAY_SIZE:
            epsilon_prob *= constants.EPSILON_DECAY

if __name__ == "__main__":
    play()