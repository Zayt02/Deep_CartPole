import numpy as np
import random as rd
import gym
import time
import tensorflow as tf
import tensorflow.keras as kr
from collections import deque

EPISODES = 100


class DeepAgent:
    def __init__(self, state_info_size, action_size, discount_rate=0.95, epsilon=1.0, episodes=100, memory_size=1000, batch_size=32):
        self.state_size = state_info_size
        self.action_size = action_size
        self.epsilon = epsilon
        # self.epsilon_min = 0.01
        # self.epsilon_decay = 0.995
        self.epsilon_speed = episodes
        self.discount_rate = discount_rate
        self.memories = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.model = self.build_model()
        self.target_model = self.build_model()

    def build_model(self):
        model = kr.models.Sequential()
        model.add(kr.layers.Dense(48, input_shape=(self.state_size,), activation='relu'))
        # self.model.add(kr.layers.Dropout(0.3))
        model.add(kr.layers.Dense(32, activation='relu'))
        model.add(kr.layers.Dense(self.action_size, activation='linear'))

        model.compile(loss='mse',
                           optimizer=kr.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))
        return model

    def choose_action(self, state):
        prob = np.random.uniform(0, 1)
        if prob >= self.epsilon:
            return np.argmax(self.model.predict(state)[0])
        else:
            return np.random.choice((0, 1))

    def update_epsilon(self, loop_counter):
        if self.epsilon > 0.01:
            self.epsilon *= 1/(1+loop_counter/self.epsilon_speed)

    def remember(self, state, action, reward, next_state, done):
        self.memories.append([state, action, reward, next_state, done])

    def train(self):
        memory = rd.sample(self.memories, 32)
        for state, action, reward, next_state, done in memory:
            actual = reward
            if not done:
                # idea of double DQNs
                actual = (reward + self.discount_rate
                          * self.target_model.predict(next_state)[0][np.argmax(self.model.predict(next_state)[0])])
            Q_values = self.model.predict(state)
            Q_values[0][action] = actual
            self.model.fit(state, Q_values, epochs=1, verbose=0)

    def update_weight_to_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_to(self, file_name):
        self.model.save_weights(file_name)


if __name__ == '__main__':
    # with tf.device('/cpu:0'):
    # make environment
    env = gym.make("CartPole-v1")
    state_info_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DeepAgent(state_info_size, action_size)

    for episode in range(EPISODES):
        alpha = 0.5
        state = env.reset()
        state = np.reshape(state, (1, state_info_size))
        score = 0
        s = 0
        done = False

        for i in range(500):
            env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            s += reward
            # termination = True if (next_state[0] >= 2.4) else False # this is for moving to the right as fast as possible
            # reward = next_state[0]/10
            # if termination and done :
            #     reward = 10
            # elif not termination and done :
            #     reward = -10
            score += reward
            next_state = np.reshape(next_state, [1, state_info_size])
            # print(agent.model.predict(np.concatenate((state, next_state))))
            # time.sleep(20)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, s = {}, e: {:.2})"
                      .format(episode, EPISODES, score, s, agent.epsilon))
                break
            if len(agent.memories) > 32 :
                agent.train()
        agent.update_weight_to_target()
        agent.update_epsilon(episode+1)

    agent.save_to('deepCartpole.h5')
    env.close()
