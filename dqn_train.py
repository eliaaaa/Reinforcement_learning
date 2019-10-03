import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import json
from torch.autograd import Variable
import random
from collections import deque
import matplotlib.pyplot as plt
# %%
import roboschool



class NNet(nn.Module):
    def __init__(self, ):
        super(NNet, self).__init__()
        self.fc1 = nn.Linear(states_num, 20)
        self.fc1.weight.data.normal_(0, 0.05)
        self.fc2 = nn.Linear(20, 30)
        self.fc2.weight.data.normal_(0, 0.05)
        self.fc3 = nn.Linear(30, 50)
        self.fc3.weight.data.normal_(0, 0.05)
        self.out = nn.Linear(50, actions_num)
        self.out.weight.data.normal_(0, 0.05)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


class DQN(object):
    def __init__(self):
        self.policy_net = NNet()
        self.target_net = NNet()

        self.action_counter = 0
        self.memory_counter = 0

        # store transitions for 2 states, reward, action, and done_boolean
        self.memory = np.zeros((memory_size, states_num * 2 + 3))

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.loss_func = nn.MSELoss()

        self.epsilon = 1  # start exploration rate
        self.epsilon_min = 0.1 # min exploration rate
        self.epsilon_decay = 0.95

    def take_action(self, x, test_on=False):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if test_on: # always take action sugguested by neural network
            actions_index = torch.max(self.policy_net.forward(x), 1)[1].data.numpy()
            action = ACTIONS[actions_index]
        else: # always take action sugguested by neural network
            if np.random.uniform() > self.epsilon:
                actions_index = torch.max(self.policy_net.forward(x), 1)[1].data.numpy()
                action = ACTIONS[actions_index]
            else: # random sample
                actions_sample = env.action_space.sample()
                actions_index = int(np.round((actions_sample + 1)/0.1))
                action = np.round(10 * actions_sample)/10
            # decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        return action, actions_index

    def reset_epsilon(self):
        self.epsilon = 0.1  # exploration rate

    def memorize(self, state, action, reward, s_new, d):
        self.memory_counter += 1
        index = random.randint(0, memory_size - 1)
        transition = np.hstack((state, [action, reward], s_new, d))
        # random pop up memory
        self.memory[index, :] = transition


    def train(self):


        sample_index = np.random.choice(memory_size, Batch_size)
        sample_batch = self.memory[sample_index, :]
        # get state, action, reward, new state from the memory[batch]
        batch_state = torch.FloatTensor(sample_batch[:, :states_num])
        batch_action = torch.LongTensor(sample_batch[:, states_num:states_num+1].astype(int))
        batch_reward = torch.FloatTensor(sample_batch[:, states_num+1:states_num+2])
        batch_state_new = torch.FloatTensor(sample_batch[:, -states_num-1:-1])
        batch_done = torch.FloatTensor(sample_batch[:, -1])

        # calculate loss from batch
        loss = Variable(torch.FloatTensor([0.]), requires_grad=True)
        for i in range(Batch_size):
            reward = batch_reward[i, 0]
            if batch_done[i]:
                next_q = reward
            else:
                next_q = Gamma * self.target_net(batch_state_new[i,:]).detach().max(0)[0].view(1, 1) + reward
            current_q = self.policy_net(batch_state[i,:]).gather(0, batch_action[i,:])
            loss = loss + (next_q - current_q) ** 2

        loss /= Batch_size

        self.optimizer.zero_grad()
        loss.backward()


        torch.save(self.policy_net, 'policy_net21.pt')
        torch.save(self.target_net, 'target_net21.pt')
        self.optimizer.step()
'''
def test():
    for i_episode in range(100):
        state = env.reset()
        episode_reward = 0 # sum of reward for this episode
        for time in range(1000):
            env.render()
            action, action_index = dqn.take_action(state, test_on=True)
            s_new, reward, done, info = env.step(action)

            episode_reward += reward
            if done:
                print('Episode: ', i_episode, '| Episode reward: ', round(episode_reward, 2), 'time: ', time)
                break
            state = s_new
'''

if __name__ == "__main__":
    env_name = 'RoboschoolInvertedDoublePendulum-v1'

    env = gym.make(env_name)

    actions_num = 21
    ACTIONS = np.linspace(-1, 1, actions_num)
    states_num = env.observation_space.shape[0]

    Batch_size = 32
    Gamma = 0.9999  # reward discount
    Update_target = 100  # update target network
    memory_size = 5000


    dqn = DQN()

    # train
    ep_score = []
    index = []
    for i_episode in range(1200):
        state = env.reset()
        episode_reward = 0
        for time in range(1000):
            env.render()
            action, action_index = dqn.take_action(state)
            s_new, reward, done, info = env.step(action)
            dqn.memorize(state, action_index, reward, s_new, done)

            episode_reward += reward
            if dqn.memory_counter > memory_size:
                dqn.train()

                # update target every 100 iterations
                if dqn.action_counter % Update_target == 0:
                    dqn.target_net.load_state_dict(dqn.policy_net.state_dict())
                dqn.action_counter += 1

            if done:
                print('Episode: ', i_episode, '| Episode reward: ', round(episode_reward, 2), 'time: ', time)
                break
            state = s_new

        ep_score.append(episode_reward)
        index.append(i_episode)
        with open("ep_score21.json", "w") as jfile:
            json.dump([index, ep_score], jfile)

    '''
    with open("ep_score.json", "r") as jfile:
        plot_file = json.load(jfile)
    plt.plot(plot_file[0], plot_file[1], 'r')
    plt.savefig("test.jpg")
    '''
    #time = np.linspace(0, len(ep_score))
    #plt.plot(ep_score, time, 'r')

    # test
    #test()
