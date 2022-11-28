import random
from collections import namedtuple
import torch
import torch.nn as nn
import numpy as np 
import torch.optim as optim 
from itertools import combinations

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'state_next', 'done')
)

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


class DQN:
    def __init__(
        self,
        env,
        network=MLP,
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

        self.observation_size = 2 * env.k + 1
        self.action_space_combinatorial = []
        for i in range(self.env.max_simultaneous_devices, self.env.k + 1):
            self.action_space_combinatorial += list(combinations(range(self.env.k), i))

        self.network = network(self.observation_size, len(self.action_space_combinatorial)).to(self.device)
        self.target_network = network(self.observation_size, len(self.action_space_combinatorial)).to(self.device)
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

    def build_obs(self, obs):
        #last_time_transmitted = self.env.last_time_transmitted
        last_time_since_polled = self.env.last_time_since_polled
        last_feedback = self.env.last_feedback
        prep_obs = np.concatenate([obs, 1 / last_time_since_polled, [last_feedback]])
        return torch.tensor(prep_obs, dtype=torch.float).to(self.device)

    def act(self, state, is_training_ready=True):
        if is_training_ready and random.uniform(0,1) >= self.epsilon:
            action = self.predict(state)
        else:
            action = np.random.randint(0, len(self.action_space_combinatorial))
            # action = self.env.timestep % self.env.n
            
        return action 
    
    def update_epsilon(self, timestep, horizon_eps=1000):
        eps = self.initial_exploration_rate - (self.initial_exploration_rate - self.final_exploration_rate) * (timestep / horizon_eps)
        self.epsilon = max(eps, self.final_exploration_rate)

    @torch.no_grad()
    def predict(self, state):
        action = self.network(state).argmax().item()
        return action

    def test(self, n_episodes):
        received_list = []
        discarded_list = []

        for ep in range(n_episodes):
            done = False
            obs = self.env.reset()
            while not done:
                pobs = self.env.preprocess_state(obs)
                pobs = self.build_obs(pobs)
                action_id = self.predict(pobs.to(self.device))
                action = list(self.action_space_combinatorial[action_id])
                action_binary = np.zeros(self.env.k)
                action_binary[action] = 1.
                obs, _, done = self.env.step(action_binary)

            received_list.append(self.env.received_packets.sum())
            discarded_list.append(self.env.discarded_packets)
                                
        return 1 - np.sum(discarded_list) / np.sum(received_list)

    def train(self, n_episodes):
        received_list = []
        discarded_list = []
        for ep in range(n_episodes):
            score = 0
            is_training_ready = ep >= self.replay_start_size
            self.update_epsilon(ep)
            done = False
            obs = self.env.reset()
            pobs = self.env.preprocess_state(obs)
            pobs = self.build_obs(pobs)
            while not done:
                action_id = self.act(pobs.to(self.device).float(), is_training_ready=is_training_ready)
                action = list(self.action_space_combinatorial[action_id])
                action_binary = np.zeros(self.env.k)
                action_binary[action] = 1.
                next_obs, reward, done = self.env.step(action_binary)
                score += reward
                next_obs = self.env.preprocess_state(next_obs)
                next_obs = self.build_obs(next_obs)
                action = torch.as_tensor([action_id], dtype=torch.long)
                reward = torch.as_tensor([reward], dtype=torch.float)
                done = torch.as_tensor([done], dtype=torch.float)
                self.replay_buffer.add(pobs, action, reward, next_obs, done)
                pobs = next_obs
            
            if ep % 100 == 0:
                ts = self.test(10)
                print(f"Episode: {ep}, Train score: {score}, Test score: {ts}, epsilon:{self.epsilon}")

            received_list.append(self.env.received_packets.sum())
            discarded_list.append(self.env.discarded_packets)

            if is_training_ready:

                if ep % self.update_frequency == 0:
                    transitions = self.replay_buffer.sample(self.minibatch_size, self.device)

                    loss = self.train_step(transitions)
                if ep % self.update_target_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())
                                
        return 1 - np.sum(discarded_list) / np.sum(received_list)


