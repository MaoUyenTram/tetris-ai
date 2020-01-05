from keras.models import Sequential, save_model, load_model
from keras.layers import Dense
from collections import deque
import numpy as np
import random

class agent:

    def __init__(self, state_size, mem_size=10000,
                 replay_start_size=None):

        self.state_size = state_size
        self.memory = deque(maxlen=mem_size)
        self.explore = 1.0
        self.explore_min = .01
        self.explore_decay = 0.975
        if not replay_start_size:
            replay_start_size = mem_size / 2
        self.replay_start_size = replay_start_size

        self.model = Sequential()
        self.model.add(Dense(32, input_dim=state_size, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(loss='mse', optimizer='Adam')


    def remember(self, current_state, next_state, reward, done):
        self.memory.append((current_state, next_state, reward, done))

    def best_state(self, states):
        max_value = None
        best_state = None

        if np.random.rand() <= self.explore:
            return random.choice(list(states))
        else:
            for state in states:
                rstate = np.reshape(state, [1, self.state_size])
                value = self.model.predict(rstate)[0]
                if not max_value or value > max_value:
                    max_value = value
                    best_state = state

        return best_state
            
    def train(self):

        if len(self.memory) >= 64 and len(self.memory) >= self.replay_start_size:
            batch = random.sample(self.memory, 64) # random batch out of memory
            next_states = np.array([x[1] for x in batch])
            next_qs = [x[0] for x in self.model.predict(next_states)]

            x = []
            y = []

            # Build xy structure to fit the model in batch (better performance)
            for i, (state, _, reward, done) in enumerate(batch):
                if done:
                    # Partial Q formula
                    new_q = reward 
                else:
                    new_q = reward + 0.95 * next_qs[i]

                x.append(state)
                y.append(new_q)

            # Fit the model to the given values
            self.model.fit(np.array(x), np.array(y),batch_size = 64, epochs=3, verbose=0,shuffle=True)
            
            if(self.explore > self.explore_min):
                self.explore *= self.explore_decay
            
            
