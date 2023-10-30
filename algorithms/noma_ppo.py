import torch
import torch.nn as nn
from torch.distributions import Bernoulli
import numpy as np


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
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, n_devices)
        
        init_weights(self.encoder, 2)
        init_weights(self.hidden1, 2)
        init_weights(self.hidden2, 2)
        init_weights(self.output_layer, 2)

    def forward(self, x): 
        x = torch.relu(self.encoder(x))
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        probs = torch.sigmoid(self.output_layer(x))
        return probs
    


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=64):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.hidden_size = hidden_size

        self.encoder = nn.Linear(state_dim, hidden_size)
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

        init_weights(self.encoder, 2)
        init_weights(self.hidden1, 2)
        init_weights(self.hidden2, 2)
        init_weights(self.output_layer, 2)
    
    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
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
                 prior_weight=None,
                 channel_prior_threshold=1,
                 normalize_channel=True,
                 beta=0.1,
                 scheduler=False,
                 channel_model=None
):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.name = "NOMA-PPO"
        self.env = env
        if channel_model is None:
            self.channel_model = self.env.channel_model
        else:
            self.channel_model = channel_model
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.prior_weight = prior_weight
        self.beta = beta
        self.channel_prior_threshold = channel_prior_threshold
        self.scheduler = scheduler
        self.normalize_channel = normalize_channel
        Tf = (5*(1 / 30 * 1e-3 + 2.34e-6))
        self.coherence_time = self.env.get_coherence_time(self.env.v) / Tf

        self.buffer = RolloutBuffer()
        if self.channel_model == 'collision':
            self.observation_size = 3 * env.k + 1
        elif self.channel_model == 'interference':
            self.observation_size = 4 * env.k + 1
        self.policy = Actor(self.observation_size, env.k, hidden_size=hidden_size).to(self.device)
        self.critic = Critic(self.observation_size, hidden_size=hidden_size).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        lr_lambda = lambda epoch: 0.99 ** epoch
        self.policy_scheduler = torch.optim.lr_scheduler.LambdaLR(self.policy_optimizer, lr_lambda)

        self.policy_old = Actor(self.observation_size, env.k, hidden_size=hidden_size).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        

    def compute_prior_collision(self, x):
        # Compute the prior given the internal state of the agent and the last time since polled
        if self.prior_weight is None:
            return torch.ones(self.env.k).to(self.device)
        elif self.prior_weight == 'toy_model':
            print(torch.tensor(self.env.compute_prior_channel()))
            return torch.tensor(self.env.compute_prior_channel(), dtype=torch.float).to(self.device)
        elif self.prior_weight == 'channel':
            channel_prior = (self.env.Ha / self.env.N) 
            prior = (channel_prior > self.channel_prior_threshold)*1
            return torch.tensor(prior, dtype=torch.float)

        else:
            last_time_since_polled = self.env.last_time_since_polled
            agg_state = self.env.preprocess_state(x)
            prior = torch.zeros(self.env.k)
            # has_a_packet: idx of the devices that have a packet to transmit. We filter agg_state with the channel quality
            has_a_packet = (agg_state + 1).nonzero()[0]
            sorted_idx = agg_state[has_a_packet].argsort()[:self.env.max_simultaneous_devices]

            if len(has_a_packet) >= self.env.max_simultaneous_devices:
                # We set the prior according to the edf scheduler
                prior[has_a_packet[sorted_idx]] = 1.
            elif (len(has_a_packet) < self.env.max_simultaneous_devices) & (len(has_a_packet) > 0):
                # We set the prior of the default prior to prior_weight and the oldest devices to 1.
                # oldest = (-last_time_since_polled).argsort()[:self.env.max_simultaneous_devices - len(has_a_packet)]
                prior += self.prior_weight
                prior[has_a_packet[sorted_idx]] = 1.
            else:
                # We explore: with priority on the devices that have not been polled more that delta frames
                time_to_poll = last_time_since_polled > self.env.deadlines
                if sum(time_to_poll) > 0:
                    prior += self.prior_weight
                    prior[time_to_poll.nonzero()[0]] = 1
                else:
                    prior += 1
            
            # channel_prior = (self.env.Ha > self.env.Ha.mean()) * 1.
            return prior.to(self.device)
        
    # def compute_prior_channel(self, x):
    #     # Compute the prior given the internal state of the agent and the last time since polled
    #     if self.prior_weight is None:
    #         return torch.ones(self.env.k).to(self.device)
    #     elif self.prior_weight == 'toy_model':
    #         print(torch.tensor(self.env.compute_prior_channel()))
    #         return torch.tensor(self.env.compute_prior_channel(), dtype=torch.float).to(self.device)
    #     else:
    #         # Filter the good/bad channels
    #         channel_prior = (self.env.Ha / self.env.N) 
    #         channel_quality = (channel_prior > self.channel_prior_threshold)*1

    #         last_time_since_polled = self.env.last_time_since_polled
    #         agg_state = self.env.preprocess_state(x)
    #         prior = torch.zeros(self.env.k)
    #         # has_a_packet: idx of the devices that have a packet to transmit. We filter agg_state with the channel quality
    #         agg_state = np.where(channel_quality == 1, agg_state, -1)
    #         has_a_packet = (agg_state + 1).nonzero()[0]
    #         sorted_idx = agg_state[has_a_packet].argsort()[:self.env.max_simultaneous_devices]

    #         if len(has_a_packet) >= self.env.max_simultaneous_devices:
    #             # We set the prior according to the edf scheduler
    #             prior[has_a_packet[sorted_idx]] = 1.
    #         elif (len(has_a_packet) < self.env.max_simultaneous_devices) & (len(has_a_packet) > 0):
    #             # We set the prior of the default prior to prior_weight and the oldest devices to 1.
    #             # oldest = (-last_time_since_polled).argsort()[:self.env.max_simultaneous_devices - len(has_a_packet)]
    #             prior += self.prior_weight
    #             prior[has_a_packet[sorted_idx]] = 1.
    #             # prior[oldest] = 1.
    #         else:
    #             # We explore: with priority on the devices that have not been polled more that delta frames
    #             time_to_poll = last_time_since_polled > self.env.deadlines
    #             if sum(time_to_poll) > 0:
    #                 prior += self.prior_weight
    #                 prior[time_to_poll.nonzero()[0]] = 1
    #             else:
    #                 prior += 1
            
    #         # channel_prior = (self.env.Ha > self.env.Ha.mean()) * 1.
    #         return prior.to(self.device)


    def compute_prior_channel(self, x):
        if self.prior_weight is None:
            return torch.ones(self.env.k).to(self.device)
        else:
            # Put 1 when there is a good channel and decorrelation < ct and when decorrelation > ct
            channel_prior = (self.env.Ha / self.env.N) 
            decorrelation = self.env.last_time_since_active

            bad_channels = ((channel_prior < self.channel_prior_threshold) & (decorrelation < self.coherence_time))*1
            bad_channels = bad_channels.nonzero()[0]

            prior_channel = torch.ones(self.env.k)
            prior_channel[bad_channels] = 0            

            agg_state = self.env.preprocess_state(x)
            # has_a_packet: idx of the devices that have a packet to transmit. We filter agg_state with the channel quality
            has_a_packet = (agg_state + 1)
            has_a_packet[bad_channels] = 0
            has_a_packet = has_a_packet.nonzero()[0]
            sorted_idx = agg_state[has_a_packet].argsort()[:self.env.max_simultaneous_devices]
            prior_edf = torch.zeros(self.env.k)
            if len(has_a_packet) >= self.env.max_simultaneous_devices:
                # We set the prior according to the edf scheduler
                prior_edf[has_a_packet[sorted_idx]] = 1.
            elif (len(has_a_packet) < self.env.max_simultaneous_devices) & (len(has_a_packet) > 0):
                # We set the prior of the default prior to prior_weight and the oldest devices to 1.
                # oldest = (-last_time_since_polled).argsort()[:self.env.max_simultaneous_devices - len(has_a_packet)]
                prior_edf += self.prior_weight
                prior_edf[has_a_packet[sorted_idx]] = 1.

            else:
                prior_edf = 1.
            prior = prior_edf * prior_channel
            return prior.to(self.device)

        

    def build_obs(self, obs):
        last_time_transmitted = self.env.last_time_transmitted.copy()
        last_time_since_polled = self.env.last_time_since_polled.copy()
        last_feedback = self.env.last_feedback
        obs_norm = 1 / (obs+0.01)
        if self.channel_model == 'interference':
            last_channel = self.env.Ha.copy()
        else:
            last_channel = np.array([])
        if self.normalize_channel:
            last_channel_norm = (last_channel / self.env.N) > self.channel_prior_threshold
        else:
            last_channel_norm = last_channel / self.env.N
        prep_obs = np.concatenate([obs_norm,
                                   last_channel_norm,
                                   1 / last_time_since_polled,
                                   1 / last_time_transmitted,
                                   [last_feedback]])
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
        
        # if self.env.timestep == 0:
        #     action = torch.ones(self.env.k)

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
        print("Model saved!")

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
    def test(self, n_episodes=100, verbose=False):
        self.env.verbose = verbose
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
                if self.channel_model == 'collision':
                    prior = self.compute_prior_collision(obs)
                else:
                    prior = self.compute_prior_channel(obs)
                ac,_ = self.act(pobs, prior, train=False)
                obs, rwd, done = self.env.step(ac.numpy())
                score += rwd
                if verbose:
                    print(f"reward: {rwd}\n\n")

            n_collisions.append(self.env.n_collisions)
            received_list.append(self.env.received_packets.sum())
            discarded_list.append(self.env.discarded_packets)
            # sense_list.append(self.env.nb_sense)
            channel_errors.append(self.env.channel_losses)
            average_reward.append(score)
            jains_list.append(self.env.compute_jains())
        return 1 - np.sum(discarded_list) / np.sum(received_list), np.mean(jains_list), np.mean(average_reward), np.mean(n_collisions), np.mean(channel_errors)
    
    def learn(self,
                n_episodes,
                update_frequency=10,
                test_length=100,
                test_frequency=20,
                model_path=None,
                early_stopping=True
                ):

        # printing and logging variables
        training_scores = [0]
        training_jains = []
        rewards_list = []
        rollup_rewards = []
        policy_loss_list = []
        value_loss_list = []
        rolling_scores = []
        test_rewards = []
        channel_errors_list = []
        collisions_list = []

        # training loop
        for episode in range(n_episodes):
            
            obs = self.env.reset()
            done = False
            current_ep_reward = 0

            while not done:
                pobs = self.env.preprocess_state(obs)
                pobs = self.build_obs(pobs)
                if self.channel_model == 'collision':
                    prior = self.compute_prior_collision(obs)
                else:
                    prior = self.compute_prior_channel(obs)
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
            if episode % test_frequency == 0:
                ts, tj, tr, tc, ce = self.test(test_length)
                test_rewards.append(tr)
                if (ts > max(training_scores))&(model_path is not None):
                    self.save(model_path)
                training_scores.append(ts)
                training_jains.append(tj)
                collisions_list.append(tc)
                channel_errors_list.append(ce)

                if early_stopping:
                    if ts == 1:
                        return training_scores, test_rewards, policy_loss_list, value_loss_list
                print(f"Episode : {episode}, Reward : {np.mean(rollup_rewards)}, Test score : {ts}, Test jains: {tj}, Collisions: {tc}, Channel errors: {ce}, policy lr: {self.policy_optimizer.param_groups[0]['lr']}")
                
                rollup_rewards = []
                rolling_scores = []


        return training_scores, training_jains, test_rewards, policy_loss_list, value_loss_list, collisions_list, channel_errors_list




