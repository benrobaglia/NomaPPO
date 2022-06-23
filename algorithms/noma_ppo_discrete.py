from mimetypes import init
import time
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
import numpy as np

def init_weights(m, gain):
    if (type(m) == nn.Linear):
        nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def compute_gae(rewards, dones, values, gamma, lbda=0.95):
    gae = 0
    values_np = values.detach().numpy()
    adv = [rewards[-1] - values_np[-1]]
    for step in reversed(range(len(rewards)-1)):
        delta = rewards[step] + gamma * values_np[step + 1] * (1-dones[step]) - values_np[step]
        gae = delta + gamma * lbda * (1-dones[step]) * gae
        adv.insert(0, gae + values_np[step])
    adv = np.array(adv)
    if (adv.std(0) > 0).all():
        adv = (adv - adv.mean(0)) / adv.std(0)
    return torch.tensor(adv, dtype=torch.float)

def discount_rewards(rewards, gamma, dones, normalize=True):
    returns = []
    R = 0
    for i in reversed(range(len(rewards))):
        R = rewards[i] + R * gamma * (1 - dones[i])
        returns.insert(0, R)

    returns = torch.tensor(np.array(returns), dtype=torch.float)
    
    if normalize:
        if (returns.std(0) > 0).all():
            returns = (returns - returns.mean(0)) / returns.std(0)
    return returns


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.values = []
        self.sensing_states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.values[:]
        del self.sensing_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]



class Actor(nn.Module):
    def __init__(self, state_dim, n_devices, hidden_size=64, recurrent=True):
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.n_devices = n_devices
        self.hidden_size = hidden_size
        self.recurrent = recurrent
        if self.recurrent:
            self.lstm_actor = nn.LSTM(state_dim, hidden_size // 2, 1, bidirectional=False)
            self.linear_sensing_actor = nn.Linear(n_devices, hidden_size // 2)
        else:
            self.linear_sensing_actor = nn.Linear(n_devices, hidden_size)
        # actor
        self.actor_head = nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Linear(hidden_size, hidden_size * 2),
                        nn.ReLU(),
                        nn.Linear(hidden_size * 2, hidden_size * 2),
                        nn.ReLU(),
                        nn.Linear(hidden_size * 2, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, n_devices),
                        nn.Sigmoid()
                    )
        
        self.actor_head.apply(lambda x: init_weights(x, 3))
        init_weights(self.linear_sensing_actor, 3)

    
    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size // 2),
                torch.zeros(1, batch_size, self.hidden_size // 2))

    def forward(self, history, sensing_obs):
        if len(history.shape) == 2:
            history = history.unsqueeze(0) # Add batch dimension
            
        if len(sensing_obs.shape) == 1:
            sensing_obs = sensing_obs.unsqueeze(0)
        
        batch_size = history.size(0)
        hidden = self.init_hidden(batch_size)
        if self.recurrent:
            lstm_out, hidden = self.lstm_actor(history.permute(1, 0, 2), hidden)
            out1 = lstm_out[-1]
            out2 = self.linear_sensing_actor(sensing_obs)
            out = torch.cat([out1, out2], dim=1)
        else:
            out = self.linear_sensing_actor(sensing_obs)
        probs = self.actor_head(out)
        dist = Bernoulli(probs)
        
        return dist
    


class Critic(nn.Module):
    def __init__(self, state_dim, n_devices, hidden_size=64, recurrent=True):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.n_devices = n_devices
        self.hidden_size = hidden_size
        self.recurrent = recurrent

        if self.recurrent:
            self.lstm_critic = nn.LSTM(state_dim, hidden_size // 2, 1, bidirectional=False)
            self.linear_sensing_critic = nn.Linear(n_devices, hidden_size // 2)
        else:
            self.linear_sensing_critic = nn.Linear(n_devices, hidden_size)

        self.critic_head = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size * 2),
                        nn.ReLU(),
                        nn.Linear(hidden_size * 2, hidden_size * 2),
                        nn.ReLU(),
                        nn.Linear(hidden_size * 2, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, 1)
                    )

        self.critic_head.apply(lambda x: init_weights(x, 3))
        init_weights(self.linear_sensing_critic, 3)
    
    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size // 2),
                torch.zeros(1, batch_size, self.hidden_size // 2))

    def forward(self, histories, sensing_obs):
        batch_size = histories.size(0)

        if self.recurrent:
            hidden = self.init_hidden(batch_size)
            lstm_out, hidden = self.lstm_critic(histories.permute(1, 0, 2), hidden)
            out1 = lstm_out[-1]
            out2 = self.linear_sensing_critic(sensing_obs)
            out = torch.cat([out1, out2], dim=1)
        else:
            out = self.linear_sensing_critic(sensing_obs)
        return self.critic_head(out)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_devices, hidden_size, recurrent=True):
        super(ActorCritic, self).__init__()

        self.state_dim = state_dim
        self.n_devices = n_devices
        self.hidden_size = hidden_size
        self.recurrent = recurrent

        if self.recurrent:
            self.lstm_common = nn.LSTM(state_dim, hidden_size // 2, 1, bidirectional=False)
            self.linear_sensing_common = nn.Linear(n_devices, hidden_size // 2)
        else:
            self.linear_sensing_common = nn.Linear(n_devices, hidden_size)


        self.critic_head = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size * 2),
                        nn.ReLU(),
                        nn.Linear(hidden_size * 2, hidden_size * 2),
                        nn.ReLU(),
                        nn.Linear(hidden_size * 2, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, 1)
                    )
        
        self.actor_head = nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Linear(hidden_size, hidden_size * 2),
                        nn.ReLU(),
                        nn.Linear(hidden_size * 2, hidden_size * 2),
                        nn.ReLU(),
                        nn.Linear(hidden_size * 2, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, n_devices),
                        nn.Sigmoid()
                    )
        
        self.actor_head.apply(lambda x: init_weights(x, 3))
        init_weights(self.linear_sensing_common, 3)
    
    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size // 2),
                torch.zeros(1, batch_size, self.hidden_size // 2))
        
    def forward(self, x, mask):
        pass



class NomaPPO:
    def __init__(self,
                 env, 
                 lr_actor=3e-4,
                 lr_critic=1e-3,
                 hidden_size=64,
                 gamma=0.99,
                 K_epochs=4,
                 eps_clip=0.1,
                 value_clip=None,
                 recurrent=True):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.env = env

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.value_clip = value_clip
        
        self.buffer = RolloutBuffer()
        self.recurrent = recurrent
        self.policy = Actor(env.observation_space, env.n, hidden_size=hidden_size, recurrent=self.recurrent).to(self.device)
        self.critic = Critic(env.observation_space, env.n, hidden_size=hidden_size, recurrent=self.recurrent).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.policy_old = Actor(env.observation_space, env.n, hidden_size=hidden_size, recurrent=self.recurrent).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()


    def act(self, history, sensing_obs, train=True):

        with torch.no_grad():
            dist = self.policy_old(history, sensing_obs)
            if train:
                action = dist.sample()
            else:
                action = dist.probs.squeeze() > 0.5
                action = action * 1.
            action_logprob = dist.log_prob(action).mean(-1)
        
        return action.detach(), action_logprob.detach()


    def evaluate(self, histories, sensing_obs, action):
        # Histories: (batch, seq len, state size)
        dist = self.policy(histories, sensing_obs)

        action_logprobs = dist.log_prob(action).mean(-1)
        dist_entropy = dist.entropy().mean(-1)

        state_values = self.critic(histories, sensing_obs)
        return action_logprobs, state_values, dist_entropy


    def update(self):

        # Monte Carlo estimate of returns
        # rewards = []
        # discounted_reward = 0
        # for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
        #     if is_terminal:
        #         discounted_reward = 0
        #     discounted_reward = reward + (self.gamma * discounted_reward)
        #     rewards.insert(0, discounted_reward)
            
        # # Normalizing the rewards
        # rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        returns = discount_rewards(self.buffer.rewards, self.gamma, self.buffer.is_terminals, True)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        # old_values = torch.tensor(self.buffer.values, dtype=torch.float32).detach().to(self.device)
        old_sensing_states = torch.squeeze(torch.stack(self.buffer.sensing_states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_sensing_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            # if self.value_clip is not None:
            #     state_values = old_values + torch.clamp(state_values - old_values, -self.value_clip, self.value_clip)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            # advantages = rewards - state_values.detach()   
            advantages = compute_gae(self.buffer.rewards, self.buffer.is_terminals, state_values, self.gamma, 0.7)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            policy_loss = - torch.min(surr1, surr2) - 0.1 * dist_entropy
            # value_loss = self.MseLoss(state_values, rewards)
            value_loss = self.MseLoss(state_values, returns)

            # take gradient step
            self.policy_optimizer.zero_grad()
            policy_loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 80)
            self.policy_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            value_loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 80)
            self.critic_optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        return policy_loss.mean().item(), value_loss.mean().item()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
    def test(self, n_episodes=100):
        score_list = []
        for e in range(n_episodes):
            state, sensing_obs = self.env.reset()
            done = False
            while not done:
                state = torch.tensor(state, dtype=torch.float).to(self.device)
                sensing_obs = torch.tensor(sensing_obs, dtype=torch.float).to(self.device)
        
                # Normalizing the sensing_obs
                sensing_obs /= 5

                ac,_ = self.act(state, sensing_obs, train=False)
                ac = ac.detach().cpu().numpy().flatten()
                state, sensing_obs, rwd, done = self.env.step(ac)
        #         print(f"state: {env.current_state}\n action: {ac}\n reward = {rwd} \n")

            if self.env.received_packets.sum() > 0:
                score_list.append(1 - self.env.discarded_packets/self.env.received_packets.sum())
        return np.mean(score_list)
    
    def learn(self,
                n_episodes,
                update_frequency=10,
                test_length=100,
                early_stopping=True
                ):

        # printing and logging variables
        training_scores = []
        rewards_list = []
        rollup_rewards = []
        policy_loss_list = []
        value_loss_list = []

        # training loop
        for episode in range(n_episodes):
            
            state, sensing_obs = self.env.reset()
            done = False
            current_ep_reward = 0

            while not done:
                state = torch.tensor(state, dtype=torch.float).to(self.device)
                sensing_obs = torch.tensor(sensing_obs, dtype=torch.float).to(self.device)

                # Normalizing the sensing_obs
                sensing_obs /= 5

                # select action with policy
                action, action_logprob = self.act(state, sensing_obs, train=True)
                value = self.critic(state, sensing_obs).item()
                self.buffer.states.append(state)
                self.buffer.values.append(value)
                self.buffer.sensing_states.append(sensing_obs)
                self.buffer.actions.append(action.detach().cpu().squeeze())
                self.buffer.logprobs.append(action_logprob)

                state, sensing_obs, reward, done = self.env.step(action.detach().cpu().squeeze().numpy())
                rollup_rewards.append(np.mean(reward))
                # saving reward and is_terminals
                self.buffer.rewards.append(reward)
                self.buffer.is_terminals.append(done)
                
                current_ep_reward += reward


            
            # update PPO agent
            if episode % update_frequency == 0:
                pl, vl = self.update()
                policy_loss_list.append(pl)
                value_loss_list.append(vl)
                ts = self.test(test_length)
                if early_stopping:
                    if ts == 1:
                        return training_scores, rewards_list, policy_loss_list, value_loss_list
                training_scores.append(ts)
                print(f"Episode : {episode} \t Average Reward : {np.mean(rollup_rewards)} \t Test score : {ts}")
                rollup_rewards = []


        return training_scores, rewards_list, policy_loss_list, value_loss_list




