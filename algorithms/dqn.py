import random
from collections import namedtuple, deque
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


class RNN(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_size=100, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.GRU(n_inputs, hidden_size, 1)
        layers = []
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, n_outputs))
        
        self.layers = nn.Sequential(*layers)
        self.layers.apply(lambda x: init_weights(x, 3))
    
    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))

    def forward(self, obs):
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(0) # Add batch dimension
            
        batch_size = obs.size(0)
        self.hidden = self.init_hidden(batch_size=batch_size)
        lstm_out, self.hidden = self.lstm(obs.permute(1, 0, 2))
        out = lstm_out[-1]
        return self.layers(out)


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


class ReplayBufferRecurrent:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
        self.capacity = capacity

    def add(self, transition):
        self.buffer.append(transition)

    def sample_chunk(self, batch_size, chunk_size):
        start_idx = np.random.randint(0, len(self.buffer) - chunk_size, batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

        for idx in start_idx:
            for chunk_step in range(idx, idx + chunk_size):
                s, a, r, s_prime, done = self.buffer[chunk_step]
                s_lst.append(s)
                a_lst.append(a)
                r_lst.append(r)
                s_prime_lst.append(s_prime)
                done_lst.append(done)

        obs_size = s_lst[0].shape[0]
        return torch.stack(s_lst).view(batch_size, chunk_size, obs_size).to(self.device), \
               torch.tensor(np.stack(a_lst), dtype=torch.long).view(batch_size, chunk_size).to(self.device), \
               torch.tensor(np.stack(r_lst), dtype=torch.float).view(batch_size, chunk_size).to(self.device), \
               torch.stack(s_prime_lst).view(batch_size, chunk_size, obs_size).to(self.device), \
               torch.tensor(done_lst, dtype=torch.float).view(batch_size, chunk_size, 1).to(self.device)

    def reset(self):
        self.buffer = deque(maxlen=self.capacity)

    def __len__(self):
        return len(self.buffer)


class DQN:
    def __init__(
        self,
        env,
        network=MLP,
        replay_start_size=50000,
        replay_buffer_size=1000000,
        gamma=0.99,
        history_len=20,
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
        self.history_len = history_len
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
        # self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.replay_buffer = ReplayBufferRecurrent(self.replay_buffer_size, self.device)

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
 
        td_target = rewards[:, -1] + (1 - dones[:, -1, 0]) * self.gamma * q_value_target.squeeze()
        q_value = self.network(states).gather(1, actions[:, -1].unsqueeze(1))

        loss = self.loss(q_value.squeeze(), td_target, reduction='mean')

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
        average_reward = []
        jains_list = []

        for ep in range(n_episodes):
            score = 0
            history = []
            done = False
            obs = self.env.reset()
            while not done:
                pobs = self.env.preprocess_state(obs)
                pobs = self.build_obs(pobs)
                history.append(pobs.to(self.device))
                # Update history
                if len(history) > self.history_len:
                    del history[0]
                history_tensor = torch.stack(history) # dim: (L, state_size)
                action_id = self.predict(history_tensor)
                action = list(self.action_space_combinatorial[action_id])
                action_binary = np.zeros(self.env.k)
                action_binary[action] = 1.
                obs, rwd, done = self.env.step(action_binary)
                score += rwd

            received_list.append(self.env.received_packets.sum())
            discarded_list.append(self.env.discarded_packets)
            average_reward.append(score)
            jains_list.append(self.env.compute_jains())
                                
        return 1 - np.sum(discarded_list) / np.sum(received_list), np.mean(jains_list),np.mean(average_reward)

    def train(self, n_episodes):
        received_list = []
        discarded_list = []
        average_reward = []
        test_scores = []
        for ep in range(n_episodes):
            score = 0
            history = []
            is_training_ready = ep >= self.replay_start_size
            self.update_epsilon(ep)
            done = False
            obs = self.env.reset()
            pobs = self.env.preprocess_state(obs)
            pobs = self.build_obs(pobs)
            history.append(pobs.to(self.device))
            while not done:
                history_tensor = torch.stack(history) # dim: (L, state_size)
                action_id = self.act(history_tensor.float(), is_training_ready=is_training_ready)
                action = list(self.action_space_combinatorial[action_id])
                action_binary = np.zeros(self.env.k)
                action_binary[action] = 1.
                next_obs, reward, done = self.env.step(action_binary)
                score += reward
                next_obs = self.env.preprocess_state(next_obs)
                
                # Update history
                next_obs = self.build_obs(next_obs)
                if len(history) > self.history_len:
                    del history[0]

                history.append(next_obs.to(self.device))
                action = torch.as_tensor([action_id], dtype=torch.long)
                reward = torch.as_tensor([reward], dtype=torch.float)
                done = torch.as_tensor([done], dtype=torch.float)
                self.replay_buffer.add((pobs, action, reward, next_obs, done))
                pobs = next_obs
            
            if ep % 20 == 0:
                ts, tj, tr = self.test(10)
                test_scores.append(ts)
                average_reward.append(tr)
                print(f"Episode: {ep}, Train score: {score}, Test score: {ts}, epsilon:{self.epsilon}")

            received_list.append(self.env.received_packets.sum())
            discarded_list.append(self.env.discarded_packets)

            if is_training_ready:

                if ep % self.update_frequency == 0:
                    # transitions = self.replay_buffer.sample(self.minibatch_size, self.device)
                    transitions = self.replay_buffer.sample_chunk(self.minibatch_size, self.history_len)

                    loss = self.train_step(transitions)
                if ep % self.update_target_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())
                                
        return test_scores, average_reward


