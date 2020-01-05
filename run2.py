from dqn_agent2 import agent
from tetris import Tetris
from datetime import datetime
from statistics import mean, median
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
        
scores = []
xval = []

# Run dqn with Tetris
def dqn():
    env = Tetris()
    episodes = 2000
    mem_size = 20000
    replay_start_size = 2000
    train_every = 1
    render_delay = None
    render = False

    dqnagent = agent(env.get_state_size(), mem_size=mem_size, replay_start_size=replay_start_size)

    

    for episode in tqdm(range(episodes)):
        current_state = env.reset()
        done = False

        #if episode % render_every == 0:
        #    render = True
            
        #else:
        #    render = False

        # Game
        while not done:
            next_states = env.get_next_states()
            best_state = dqnagent.best_state(next_states.values())
            
            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break
                    
                
            reward,done = env.play(best_action[0], best_action[1],
                                    render=render, render_delay=render_delay)
            
            if (len(scores) > 0):
                if (env.get_game_score() > max(scores)):
                    render = True
                
            dqnagent.remember(current_state, next_states[best_action], reward, done)
            current_state = next_states[best_action]

        scores.append(env.get_game_score())
        xval.append(episode)
        render = False
        dqnagent.train()


if __name__ == "__main__":
    dqn()
    plt.plot(xval,scores)
    plt.xlabel('episodes')
    plt.ylabel('score')
    plt.show()
