import random
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from collections import namedtuple


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


class ReplayBufferPER:
    # https://arxiv.org/pdf/1511.05952.pdf
    def __init__(self, capacity, prob_alpha=0.6):
        self._capacity = capacity
        self._memory = {}
        self._position = 0
        self._priorities = np.zeros((capacity,))
        self.prob_alpha = prob_alpha

    def add(self, *args):
        self._memory[self._position] = Transition(*args)
        max_prio = self._priorities.max() if len(self._memory) > 1 else 1.0
        self._priorities[self._position] = max_prio
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size, beta=0.4, device=None):
        if len(self._memory) == self._capacity:
            prios = self._priorities
        else:
            prios = self._priorities[:self._position]

        probs  = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self._memory), batch_size, p=probs)
        samples = [self._memory[idx] for idx in indices]
        
        output = [torch.stack(tensors).to(device) for tensors in zip(*samples)]
        
        # Compute IS weights
        N = len(self._memory)
        weights  = (N * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights)
        
        output.append(indices)
        output.append(weights)
        
        return output

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self._priorities[idx] = prio

    def __len__(self):
        return len(self._memory)

class BDQ:
    def __init__(
        self,
        env,
        state_size,
        action_size,
        n_devices,
        network,
        replay_start_size=50000,
        replay_buffer_size=1000000,
        gamma=0.99,
        update_target_frequency=10000,
        minibatch_size=32,
        learning_rate=1e-3,
        update_frequency=1,
        initial_exploration_rate=1,
        final_exploration_rate=0.1,
        end_exploration=10,
        adam_epsilon=1e-8,
        loss='mse',
        seed=None,
        scheduler=False
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_size = state_size
        self.action_size = action_size
        self.n_devices = n_devices

        self.replay_start_size = replay_start_size
        self.replay_buffer_size = replay_buffer_size
        self.gamma = gamma
        self.update_target_frequency = update_target_frequency
        self.minibatch_size = minibatch_size 
        self.learning_rate = learning_rate
        self.update_frequency = update_frequency
        self.initial_exploration_rate = initial_exploration_rate
        self.final_exploration_rate = final_exploration_rate
        self.end_exploration = end_exploration
        self.adam_epsilon = adam_epsilon
        self.use_scheduler = scheduler

        if callable(loss):
            self.loss = loss
        else:
            self.loss = {'huber':torch.functional.F.smooth_l1_loss, 'mse': torch.functional.F.mse_loss}[loss]
        
        self.env = env
        self.replay_buffer = ReplayBufferPER(self.replay_buffer_size, 0.6)
        # self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        
        self.seed = random.randint(0, 1e6) if seed is None else seed

        # self.network = network(state_size, action_size, n_devices).to(self.device)
        # self.target_network = network(state_size, action_size, n_devices).to(self.device)

        self.network = BranchingQNetwork(state_size, action_size, n_devices).to(self.device)
        self.target_network = BranchingQNetwork(state_size, action_size, n_devices).to(self.device)

        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)

        lr_lambda = lambda epoch: 0.999 ** epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)


    def train_step(self, transitions):
        states, actions, rewards, states_next, dones, indices, weights = transitions
        # states, actions, rewards, states_next, dones = transitions
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        states_next = states_next.to(self.device)
        dones = dones.to(self.device)
        weights = torch.tensor(weights, dtype=torch.float).to(self.device)

        with torch.no_grad():
            # Compute the action via the network (the one that maximizes)
            actions_target = self.network(states_next).max(2, True)[1]

            # Evaluate the value with the target network.
            q_value_target = self.target_network(states_next.float()).gather(2, actions_target.to(self.device)) # Shape: batch x n_devices x 1

            td_target = rewards.repeat(1, self.n_devices).unsqueeze(2) + (1 - dones.repeat(1, self.n_devices).unsqueeze(2)) * self.gamma * q_value_target
        
        q_value = self.network(states.float())

        q_value = q_value.gather(2, actions.unsqueeze(2))

        loss = torch.mean((td_target - q_value)**2, 1) * weights
        # loss = torch.mean((td_target - q_value)**2, 1)
        prios = (td_target - q_value).abs().sum(1) + 1e-5
        prios = torch.nan_to_num(prios, 1e-5)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.replay_buffer.update_priorities(indices, prios.cpu().detach().numpy())
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 80)
        # for p in self.network.parameters(): 
        #     p.grad.data.clamp_(-1.,1.)

        self.optimizer.step()
        if self.use_scheduler:
            self.scheduler.step()

        return loss.item()

    def act(self, state, is_training_ready=True):
        if is_training_ready and random.uniform(0,1) >= self.epsilon:
            action_binary = self.predict(state).numpy()
        else:
            # potential_devices = np.argwhere(state[:, 0] < 0).reshape(-1).cpu().tolist()
            # if len(potential_devices) == 0:
            #     action = random.sample(self.env.action_space, self.n_devices)
            # else:
            #     # num_polled = np.random.randint(0, len(potential_devices))
            #     # num_polled = min(num_polled, self.env.max_simultaneous_devices)
            #     # action = random.sample(potential_devices, num_polled)
            action = np.random.choice(range(self.env.k), self.env.max_simultaneous_devices, replace=False)
            action_binary = np.zeros(self.env.k)
            action_binary[action] = 1
            
        return action_binary
    
    def update_epsilon(self, timestep, horizon_eps=1000):
        eps = self.initial_exploration_rate - (self.initial_exploration_rate - self.final_exploration_rate) * (timestep / horizon_eps)
        self.epsilon = max(eps, self.final_exploration_rate)

    @torch.no_grad()
    def predict(self, state):
        action_binary = self.network(state.reshape(-1)).argmax(2).squeeze()
        # else:
        #     q_values = self.network(state.reshape(-1)).squeeze()
        #     mask = state.squeeze()[:,0] >= 0
        #     q_values[mask, 1] = - np.inf
        #     top_k_action = q_values[:, 1].topk(k=3)[1].cpu().numpy()

        #     action_binary = torch.zeros(self.n_devices).to(self.device)
        #     action_binary[top_k_action] = 1

        return action_binary.cpu()

    def build_obs(self, obs):
        last_time_transmitted = self.env.last_time_transmitted
        last_time_since_polled = self.env.last_time_since_polled
        last_feedback = self.env.last_feedback
        prep_obs = np.concatenate([obs, 1 / last_time_transmitted, 1 / last_time_since_polled, [last_feedback]])
        return torch.tensor(prep_obs, dtype=torch.float).to(self.device)


    def test(self, n_episodes=100, verbose=False):
        received_list = []
        discarded_list = []
        sense_list = []
        channel_errors = []
        average_reward = []
        n_collisions = []
        jains_list = []
        for e in range(n_episodes):
            score = 0
            obs = self.env.reset()
            done = False
            while not done:
                pobs = self.env.preprocess_state(obs)
                pobs = self.build_obs(pobs)
                ac = self.predict(pobs)
                if verbose:
                    print(f"Timestep: {self.env.timestep}")
                    print(f"Current state: {self.env.current_state}")
                    print(f"Action: {ac}")
                    print(f"Number discarded: {self.env.discarded_packets.sum()}")
                obs, rwd, done = self.env.step(ac.numpy())
                score += rwd
                if verbose:
                    print(f"reward: {rwd}\n\n")
 
            n_collisions.append(self.env.n_collisions)
            received_list.append(self.env.received_packets.sum())
            discarded_list.append(self.env.discarded_packets)
            sense_list.append(self.env.nb_sense)
            channel_errors.append(self.env.channel_losses)
            average_reward.append(score)
            jains_list.append(self.env.compute_jains())
        return 1 - np.sum(discarded_list) / np.sum(received_list), np.mean(jains_list), np.mean(n_collisions)

    def train(self, n_episodes, test_frequency=500, test_length=20, early_stopping=True):
        received_list = []
        discarded_list = []
        average_reward = []
        test_scores = []

        for ep in range(n_episodes):
            obs = self.env.reset()
            pobs = self.env.preprocess_state(obs)
            pobs = self.build_obs(pobs)
            done = False
            current_ep_reward = 0

            while not done:

                is_training_ready = ep > self.replay_start_size
                self.update_epsilon(ep, 1000)
                action = self.act(pobs.to(self.device), is_training_ready)
                next_obs, reward, done = self.env.step(action)
                next_obs = self.env.preprocess_state(next_obs)
                next_obs = self.build_obs(next_obs)

                action = torch.as_tensor(action, dtype=torch.long)
                reward = torch.as_tensor([reward], dtype=torch.float)
                done = torch.as_tensor([done], dtype=torch.float)
                self.replay_buffer.add(pobs.reshape(-1), action, reward, next_obs.reshape(-1), done)
                pobs = next_obs

            if ep % test_frequency == 0:
                ts, tj, tr = self.test(test_length)
                test_scores.append(ts)
                average_reward.append(tr)
                print(f"Episode: {ep}, Test score: {ts}, Test jains: {tj}, epsilon:{self.epsilon}, lr: {self.optimizer.param_groups[0]['lr']}")
                if (ts == 1) & (early_stopping):
                    return test_scores, average_reward
            received_list.append(self.env.received_packets.sum())
            discarded_list.append(self.env.discarded_packets)

            if is_training_ready:

                if ep % self.update_frequency == 0:
                    beta = beta_by_frame(ep, 0.4, ep//self.end_exploration)
                    transitions = self.replay_buffer.sample(self.minibatch_size, beta, self.device)
                    # transitions = self.replay_buffer.sample(self.minibatch_size, self.device)
                    loss = self.train_step(transitions)

                if ep % self.update_target_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())
                                
        return test_scores, average_reward


def init_weights(m, gain):
    if (type(m) == nn.Linear) | (type(m) == nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class bdq_network(nn.Module):
    def __init__(self, num_inputs, num_outputs, n_devices):
        super(bdq_network, self).__init__()
        
        self.n_devices = n_devices
        self.num_outputs = num_outputs
        self.feature = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()

        )
        
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs * n_devices)
        )
        
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_devices)
        )
        self.feature.apply(lambda x: init_weights(x, 2))
        self.advantage.apply(lambda x: init_weights(x, 2))
        self.value.apply(lambda x: init_weights(x, 2))
        
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = self.feature(x)
        advantage = self.advantage(x)
        value     = self.value(x).unsqueeze(-1)
        advantage = advantage.view(x.shape[0], self.n_devices, self.num_outputs)
        # return value + advantage  - advantage.max(-1, keepdim=True)[0]
        return value + advantage  - torch.mean(advantage, -1, keepdim=True)



def beta_by_frame (frame_idx, beta_start=0.4, beta_frames=10000):
    return min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

class DuelingNetwork(nn.Module): 

    def __init__(self, obs, ac): 

        super().__init__()

        self.model = nn.Sequential(nn.Linear(obs, 128), 
                                   nn.ReLU(), 
                                   nn.Linear(128,128), 
                                   nn.ReLU())

        self.value_head = nn.Linear(128, 1)
        self.adv_head = nn.Linear(128, ac)

    def forward(self, x): 

        out = self.model(x)

        value = self.value_head(out)
        adv = self.adv_head(out)

        q_val = value + adv - adv.mean(1).reshape(-1,1)
        return q_val


class BranchingQNetwork(nn.Module):

    def __init__(self, obs, n, ac_dim): 

        super().__init__()

        self.ac_dim = ac_dim
        self.n = n 

        self.model = nn.Sequential(nn.Linear(obs, 128), 
                                   nn.ReLU(),
                                   nn.Linear(128,128), 
                                   nn.ReLU())

        self.value_head = nn.Linear(128, 1)
        self.adv_heads = nn.ModuleList([nn.Linear(128, n) for i in range(ac_dim)])

    def forward(self, x): 
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        out = self.model(x)
        value = self.value_head(out)
        advs = torch.stack([l(out) for l in self.adv_heads], dim = 1)

        # print(advs.shape)
        # print(advs.mean(2).shape)
        test =  advs.mean(2, keepdim = True)
        # input(test.shape)
        q_val = value.unsqueeze(2) + advs - advs.mean(2, keepdim = True )
        # input(q_val.shape)

        return q_val
