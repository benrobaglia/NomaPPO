import random
import time
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim 
from collections import namedtuple
import torch

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'state_next', 'done')
)


class ReplayBuffer:
    def __init__(self, capacity):
        self._capacity = capacity
        self._memory = {}
        self._position = 0

    def add(self, *args):
        self._memory[self._position] = Transition(*args)
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size, device=None):
        batch = random.sample(list(self._memory.values()), batch_size)
        return [torch.stack(tensors).to(device) for tensors in zip(*batch)]

    def __len__(self):
        return len(self._memory)


def init_weights(m, gain):
    if (type(m) == nn.Linear) | (type(m) == nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MLP(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, hiddens=[100, 100], **kwargs):
        super().__init__()
        layers = []
        for hidden in hiddens:
            layers.append(nn.Linear(n_inputs, hidden))
            layers.append(nn.ReLU())
            n_inputs = hidden 
        
        if n_outputs is not None:
            layers.append(nn.Linear(hidden, n_outputs))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(lambda x: init_weights(x, 3))

    def forward(self, obs):
        return self.layers(obs)


class DQN:
    def __init__(
        self,
        env,
        replay_start_size=50000,
        replay_buffer_size=1000000,
        gamma=0.99,
        update_target_frequency=10000,
        minibatch_size=32,
        learning_rate=1e-3,
        update_frequency=1,
        initial_exploration_rate=1,
        final_exploration_rate=0.1,
        adam_epsilon=1e-8,
        loss='huber',
        seed=None,
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.replay_start_size = replay_start_size
        self.replay_buffer_size = replay_buffer_size
        self.gamma = gamma
        self.update_target_frequency = update_target_frequency
        self.minibatch_size = minibatch_size 
        self.learning_rate = learning_rate
        self.update_frequency = update_frequency
        self.initial_exploration_rate = initial_exploration_rate
        self.final_exploration_rate = final_exploration_rate
        self.adam_epsilon = adam_epsilon
        if callable(loss):
            self.loss = loss
        else:
            self.loss = {'huber':torch.functional.F.smooth_l1_loss, 'mse': torch.functional.F.mse_loss}[loss]
        
        self.env = env
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.seed = random.randint(0, 1e6) if seed is None else seed

        state_size = self.env.deadlines.max() + 1

        self.network = MLP(state_size, 2).to(self.device)
        self.target_network = MLP(state_size, 2).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, eps=self.adam_epsilon)


    def train_step(self, transitions):
        states, actions, rewards, states_next, dones = transitions

        with torch.no_grad():
            q_value_target = self.target_network(states_next).max(1, True)[0]
 
        td_target = rewards + (1 - dones) * self.gamma * q_value_target
        q_value = self.network(states).gather(1, actions)

        loss = self.loss(q_value, td_target, reduction='mean')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def act(self, state, is_training_ready=True):
        if is_training_ready and random.uniform(0,1) >= self.epsilon:
            action = self.predict(state)
        else:
            action = np.random.randint(0, 2)
            # action = self.env.timestep % self.env.n
            
        return action 
    
    def update_epsilon(self, timestep, horizon_eps=1000):
        eps = self.initial_exploration_rate - (self.initial_exploration_rate - self.final_exploration_rate) * (timestep / horizon_eps)
        self.epsilon = max(eps, self.final_exploration_rate)

    @torch.no_grad()
    def predict(self, state):
        action = self.network(state).argmax().item()
        return action

class iDQN:
    def __init__(
        self,
        env,
        replay_start_size=50000,
        replay_buffer_size=1000000,
        gamma=0.99,
        update_target_frequency=10000,
        minibatch_size=32,
        learning_rate=1e-3,
        update_frequency=1,
        initial_exploration_rate=1,
        final_exploration_rate=0.1,
        adam_epsilon=1e-8,
        loss='huber',
        seed=None,
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.replay_start_size = replay_start_size
        self.replay_buffer_size = replay_buffer_size
        self.gamma = gamma
        self.update_target_frequency = update_target_frequency
        self.minibatch_size = minibatch_size 
        self.learning_rate = learning_rate
        self.update_frequency = update_frequency
        self.initial_exploration_rate = initial_exploration_rate
        self.final_exploration_rate = final_exploration_rate
        self.adam_epsilon = adam_epsilon
        if callable(loss):
            self.loss = loss
        else:
            self.loss = {'huber':torch.functional.F.smooth_l1_loss, 'mse': torch.functional.F.mse_loss}[loss]
        
        self.env = env
        self.agents = [DQN(env,
                        replay_start_size=replay_start_size,
                        replay_buffer_size=replay_buffer_size,
                        gamma=gamma,
                        update_target_frequency=update_target_frequency,
                        minibatch_size=minibatch_size,
                        learning_rate=learning_rate,
                        update_frequency=update_frequency,
                        initial_exploration_rate=initial_exploration_rate,
                        final_exploration_rate=final_exploration_rate,
                        adam_epsilon=adam_epsilon,
                        loss=loss,
                        seed=seed) for _ in range(env.k)]
        self.seed = random.randint(0, 1e6) if seed is None else seed


    def train_idqn(self, n_episodes):
        
        received_list = []
        discarded_list = []

        for ep in range(n_episodes):
            score = 0
            is_training_ready = ep >= self.replay_start_size
            done = False
            _ = self.env.reset()
            state = self.env.current_state.copy()
            lst_feedback = np.zeros((self.env.k, 1)) + self.env.last_feedback
            state = np.concatenate([state, lst_feedback], axis=1)
            state = torch.tensor(state, dtype=torch.float).to(self.device)

            while not done:
                actions = []
                for i,a in enumerate(self.agents):
                    state_n = state[i]
                    
                    actions.append(a.act((state_n.to(a.device)),
                                        is_training_ready=is_training_ready))
                    a.update_epsilon(ep)
                    
                _, rewards, done = self.env.step(np.array(actions))
                score += rewards
                
                state_next = self.env.current_state.copy()
                lst_feedback = np.zeros((self.env.k, 1)) + self.env.last_feedback
                state_next = np.concatenate([state_next, lst_feedback], axis=1)

                state_next = torch.tensor(state_next, dtype=torch.float).to(self.device)
                
                for i, a in enumerate(self.agents):
                    action_n = torch.as_tensor([actions[i]], dtype=torch.long)
                    reward_n = torch.as_tensor([rewards], dtype=torch.float)
                    done = torch.as_tensor([done], dtype=torch.float)
                    state_n = state[i]
                    state_next_n = state_next[i]
                    
                    a.replay_buffer.add(state_n,
                                        action_n,
                                        reward_n,
                                        state_next_n,
                                        done)
                    state = state_next

            if ep % 100 == 0:
                score_tst = self.test_idqn(10)
                print(f"Episode: {ep}, Train score: {score}, Test score: {score_tst}")
                    
            received_list.append(self.env.received_packets.sum())
            discarded_list.append(self.env.discarded_packets)
                
            if is_training_ready:
                if ep % self.update_frequency == 0:
                    for i,a in enumerate(self.agents):
                        tr = a.replay_buffer.sample(a.minibatch_size, a.device)
                        loss = a.train_step(tr)
                        # loss_list.append(loss)
                        if ep % a.update_target_frequency == 0:
                            a.target_network.load_state_dict(a.network.state_dict())

        # Save models
    #     state_dicts = []
    #     for a in agents:
    #         state_dicts.append(a.network.state_dict())
                    
        return 1 - np.sum(discarded_list) / np.sum(received_list)


    def test_idqn(self, n_episodes, verbose=False):
        
        received_list = []
        discarded_list = []

        for ep in range(n_episodes):
            
            _ = self.env.reset()
            state = self.env.current_state.copy()
            lst_feedback = np.zeros((self.env.k, 1)) + self.env.last_feedback
            state = np.concatenate([state, lst_feedback], axis=1)
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            done = False

            while not done:
                actions = []
                for i,a in enumerate(self.agents):
                    state_n = state[i]
                    actions.append(a.predict(state_n.to(self.device)))
                    
                _, reward, done = self.env.step(np.array(actions))
                
                state_next = self.env.current_state.copy()
                lst_feedback = np.zeros((self.env.k, 1)) + self.env.last_feedback
                state_next = np.concatenate([state_next, lst_feedback], axis=1)

                if verbose:
                    print(f"Timestep: {self.env.timestep}")
                    print(f"State: {state}")
                    print(f"Next state: {state_next}")
                    print(f"Action: {actions}")
                    print(f"Reward: {reward}")
                    print(f"Number discarded: {self.env.discarded_packets}")


                state_next = torch.tensor(state_next, dtype=torch.float).to(self.device)       
                state = state_next

            received_list.append(self.env.received_packets.sum())
            discarded_list.append(self.env.discarded_packets)
                                
        return 1 - np.sum(discarded_list) / np.sum(received_list)

