import torch
import torch.nn as nn
from torch.distributions import Bernoulli
import numpy as np

# torch.autograd.set_detect_anomaly(True)


def init_weights(m, gain):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)

def compute_gae(rewards, dones, values, gamma, lbda=0.95):
    gae = 0
    values_np = values.cpu().detach().numpy()
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
        self.priors = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.values[:]
        del self.priors[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]



class Actor(nn.Module):
    def __init__(self, state_dim, n_devices, hidden_size=64):
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.n_devices = n_devices
        self.hidden_size = hidden_size
        # actor
        self.encoder = nn.Linear(state_dim, hidden_size)
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, n_devices)
        
        init_weights(self.encoder, 2)
        init_weights(self.hidden1, 2)
        init_weights(self.output_layer, 2)

    def forward(self, x): 
        x = torch.relu(self.encoder(x))
        x = torch.relu(self.hidden1(x))
        probs = torch.sigmoid(self.output_layer(x))
        return probs
    


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=64):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.hidden_size = hidden_size

        self.encoder = nn.Linear(state_dim, hidden_size)
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

        init_weights(self.encoder, 2)
        init_weights(self.hidden1, 2)
        init_weights(self.output_layer, 2)
    
    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = torch.relu(self.hidden1(x))
        value = self.output_layer(x)

        return value



class NomaPPO:
    def __init__(self,
                 env, 
                 lr_actor=3e-4,
                 lr_critic=1e-3,
                 hidden_size=64,
                 gamma=0.99,
                 K_epochs=4,
                 eps_clip=0.1,
                 prior_weight=0.6,
                 beta=0.1,
                 scheduler=False
):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.env = env

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.prior_weight = prior_weight
        self.beta = beta
        self.scheduler = scheduler
        
        self.buffer = RolloutBuffer()
        self.observation_size = 2 * env.k + 1
        self.policy = Actor(self.observation_size, env.k, hidden_size=hidden_size).to(self.device)
        self.critic = Critic(self.observation_size, hidden_size=hidden_size).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        lr_lambda = lambda epoch: 0.99 ** epoch
        self.policy_scheduler = torch.optim.lr_scheduler.LambdaLR(self.policy_optimizer, lr_lambda)

        self.policy_old = Actor(self.observation_size, env.k, hidden_size=hidden_size).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def compute_prior(self, x):
        # Compute the prior given the internal state of the agent.
        agg_state = self.env.preprocess_state(x)
        prior = torch.zeros(self.env.k)
        # has_a_packet: idx of the devices that have a packet to transmit.
        has_a_packet = (agg_state + 1).nonzero()[0]
        sorted_idx = agg_state[has_a_packet].argsort()[:self.env.max_simultaneous_devices]

        if len(has_a_packet) >= self.env.max_simultaneous_devices:
            # We set the prior according to the edf scheduler
            prior[has_a_packet[sorted_idx]] = 1.
        elif (len(has_a_packet) < self.env.max_simultaneous_devices) & (len(has_a_packet) > 0):
            # We set the prior of the default prior to prior_weight
            prior += self.prior_weight
            prior[has_a_packet[sorted_idx]] = 1.
        else:
            # We explore: no prior
            prior += 1.
        return prior.to(self.device)

    def build_obs(self, obs):
        #last_time_transmitted = self.env.last_time_transmitted
        last_time_since_polled = self.env.last_time_since_polled
        last_feedback = self.env.last_feedback
        prep_obs = np.concatenate([obs, 1 / last_time_since_polled, [last_feedback]])
        return torch.tensor(prep_obs, dtype=torch.float).to(self.device)

    def act(self, pobs, prior, train=True):

        probs = self.policy_old(pobs.unsqueeze(0))
        probs *= prior
        dist = Bernoulli(probs=probs)
        if train:
            action = dist.sample()
            action = action.squeeze()
        else:
            action = dist.probs.squeeze() > 0.5
            action = action * 1.
        action_logprob = dist.log_prob(action).mean(-1)
        
        return action.detach().cpu(), action_logprob.detach().cpu()


    def evaluate(self, states, priors, action):
        probs = self.policy(states)
        dist = Bernoulli(probs=probs*priors.detach())
        action_logprobs = dist.log_prob(action).mean(-1)
        dist_entropy = dist.entropy().mean(-1)

        state_values = self.critic(states)
        return action_logprobs, state_values, dist_entropy


    def update(self):

        returns = discount_rewards(self.buffer.rewards, self.gamma, self.buffer.is_terminals, True)
        returns = returns.to(self.device)

        # convert list to tensor
        old_states = torch.stack(self.buffer.states).detach().to(self.device)
        old_actions = torch.stack(self.buffer.actions).detach().to(self.device)
        old_logprobs = torch.stack(self.buffer.logprobs).detach().to(self.device)
        old_priors = torch.stack(self.buffer.priors).detach().to(self.device)
        
        pl_list = []
        vl_list = []
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_priors, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = compute_gae(self.buffer.rewards, self.buffer.is_terminals, state_values, self.gamma, 0.97)
            advantages = advantages.squeeze().to(self.device)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            policy_loss = - torch.min(surr1, surr2) - self.beta * dist_entropy
            value_loss = self.MseLoss(state_values, returns.to(self.device))

            self.policy_optimizer.zero_grad()
            policy_loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 80)
            self.policy_optimizer.step()
            if self.scheduler:
                self.policy_scheduler.step()
            
            self.critic_optimizer.zero_grad()
            value_loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 80)
            self.critic_optimizer.step()

            pl_list.append(policy_loss.mean().item())
            vl_list.append(value_loss.mean().item())

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        return pl_list, vl_list

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
    def test(self, n_episodes=100, verbose=False):
        received_list = []
        discarded_list = []
        for e in range(n_episodes):
            obs = self.env.reset()
            done = False
            while not done:
                pobs = self.env.preprocess_state(obs)
                pobs = self.build_obs(pobs)
                prior = self.compute_prior(obs)
                ac,_ = self.act(pobs, prior, train=False)
                if verbose:
                    print(f"Timestep: {self.env.timestep}")
                    print(f"Current state: {self.env.current_state}")
                    print(f"obs: {pobs}, Prior: {prior}")
                    print(f"Action: {ac}")
                    print(f"Number discarded: {self.env.discarded_packets}")
                obs, rwd, done = self.env.step(ac.numpy())
                if verbose:
                    print(f"reward: {rwd}\n\n")

            received_list.append(self.env.received_packets.sum())
            discarded_list.append(self.env.discarded_packets)
        return 1 - np.sum(discarded_list) / np.sum(received_list)
    
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
        rolling_scores = []

        # training loop
        for episode in range(n_episodes):
            
            obs = self.env.reset()
            done = False
            current_ep_reward = 0

            while not done:
                pobs = self.env.preprocess_state(obs)
                pobs = self.build_obs(pobs)
                prior = self.compute_prior(obs)
                action, action_logprob = self.act(pobs, prior, train=True)
                value = self.critic(pobs).item()
                
                self.buffer.states.append(pobs)
                self.buffer.values.append(value)
                self.buffer.priors.append(prior)
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(action_logprob)

                obs, reward, done = self.env.step(action.numpy())
                rollup_rewards.append(np.mean(reward))
                # saving reward and is_terminals
                self.buffer.rewards.append(reward)
                self.buffer.is_terminals.append(done)
                
                current_ep_reward += reward
            rolling_scores.append(1 - self.env.discarded_packets / self.env.received_packets.sum())


            
            # update PPO agent
            if episode % update_frequency == 0:
                pl, vl = self.update()
                policy_loss_list.extend(pl)
                value_loss_list.extend(vl)
                ts = self.test(test_length)
                if early_stopping:
                    if ts == 1:
                        return training_scores, rewards_list, policy_loss_list, value_loss_list
                training_scores.append(ts)
                print(f"Episode : {episode}, Reward : {np.mean(rollup_rewards)}, Train scores: {np.mean(rolling_scores)}, Test score : {ts}, policy lr: {self.policy_optimizer.param_groups[0]['lr']}")
                rollup_rewards = []
                rolling_scores = []


        return training_scores, rewards_list, policy_loss_list, value_loss_list




