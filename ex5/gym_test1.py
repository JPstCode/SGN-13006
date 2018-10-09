import gym
import random
import numpy as np
import time


env = gym.make("Taxi-v2")
#next_state = -1000*np.ones((501,6))
#next_reward = -1000*np.ones((501,6))

# Training
# THIS YOU NEED TO IMPLEMENT


#Actions: 0-S, 1-N, 2-E, 3-W, 4-Pick, 5-Drop

#Possible states:
#Taxi can be 5x5 places,
#Passenger can be 4+1 places (inside the taxi = +1)
#4 Possible destinations
#5x5x5x4 = 500

#Q-Table fo every state and action
q_table = np.zeros([500,6])

#Parameters

alpha = 0.5 #learning rate
gamma = 0.9 #discount factor
epsilon = 0.1 #exploration vs explotation


for i in range(0,1000):

    state = env.reset()
    done = False

    while not done:

        if random.uniform(0,1) < epsilon:
            action = random.randint(0,5) #take random action

        else:
            action = np.argmax(q_table[state]) #choose best action for value

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state,action]
        next_max_value = np.max(q_table[next_state])

        new_value = (1-alpha)*old_value + alpha*(reward +
                                        gamma*next_max_value)

        q_table[state,action] = new_value

        state = next_state


print("training done")


# Testing
all_actions = []
all_rewards = []

past_observation = -1

for j in range(0,10):

    observation = env.reset()
    done = False
    test_tot_reward = 0
    test_tot_actions = 0


    while not done:
        test_tot_actions = test_tot_actions+1
        action = np.argmax(q_table[observation])

        if (observation == past_observation):
        # This is done only if gets stuck
            action = random.sample(range(0,6),1)
            action = action[0]

        past_observation = observation
        observation, reward, done, _ = env.step(action)
        test_tot_reward = test_tot_reward+reward
        #env.render()
        #time.sleep(1)

    all_actions.append(test_tot_actions)
    all_rewards.append(test_tot_reward)

all_actions = np.asarray(all_actions)
all_rewards = np.asarray(all_rewards)


print("Total reward: ")
print(np.mean(all_rewards))
print("Total actions: ")
print(np.mean(all_actions))