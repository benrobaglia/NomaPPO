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
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.values[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class RNN(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_size=100, combinatorial=False, use_activation=True, **kwargs):

        super().__init__()
        self.hidden_size = hidden_size
        self.combinatorial = combinatorial
        self.use_activation = use_activation
        self.lstm = nn.GRU(n_inputs, hidden_size, 1)
        layers = []
        # layers.append(nn.Linear(hidden_size, hidden_size))
        # layers.append(nn.ReLU())
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
        out = self.layers(out)
        if self.use_activation:
            if not self.combinatorial:
                out = torch.softmax(out, dim=1)
            else:
                out = torch.sigmoid(out)
        return out



class NomaPPO:
    def __init__(self,
                 env, 
                 lr_actor=3e-4,
                 lr_critic=1e-3,
                 hidden_size=64,
                 gamma=0.99,
                 K_epochs=4,
                 eps_clip=0.1,
                 history_len=5,
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
        self.history_len = history_len
        self.beta = beta
        self.channel_prior_threshold = channel_prior_threshold
        self.scheduler = scheduler
        self.normalize_channel = normalize_channel
        Tf = (5*(1 / 30 * 1e-3 + 2.34e-6))
        self.coherence_time = self.env.get_coherence_time(self.env.v) / Tf

        self.buffer = RolloutBuffer()
        if self.channel_model == 'collision':
            self.observation_size = env.k + 1
        elif self.channel_model == 'interference':
            self.observation_size = 2 * env.k + 1
        self.policy = RNN(n_inputs=self.observation_size,
                          n_outputs=env.k,
                          hidden_size=hidden_size,
                          combinatorial=True,
                          use_activation=True).to(self.device)
        self.critic = RNN(n_inputs=self.observation_size,
                          n_outputs=1,
                          hidden_size=hidden_size,
                          combinatorial=False,
                          use_activation=False).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        lr_lambda = lambda epoch: 0.99 ** epoch
        self.policy_scheduler = torch.optim.lr_scheduler.LambdaLR(self.policy_optimizer, lr_lambda)

        self.policy_old = RNN(n_inputs=self.observation_size,
                                n_outputs=env.k,
                                hidden_size=hidden_size,
                                combinatorial=True,
                                use_activation=True).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        

    def act(self, pobs, train=True):
        probs = self.policy_old(pobs.unsqueeze(0))
        dist = Bernoulli(probs=probs)
        if train:
            action = dist.sample()
            action = action.squeeze()
        else:
            action = dist.probs.squeeze() > 0.5
            action = action * 1.
        
        action_logprob = dist.log_prob(action).mean(-1)
        # entropy = dist.entropy().mean(-1)
        return action.detach().cpu(), action_logprob.detach().cpu()


    def evaluate(self, states, action):
        probs = self.policy(states)
        dist = Bernoulli(probs=probs)
        action_logprobs = dist.log_prob(action).mean(-1)
        dist_entropy = dist.entropy().mean(-1)

        state_values = self.critic(states)
        return action_logprobs, state_values, dist_entropy


    def update(self):

        returns = discount_rewards(self.buffer.rewards, self.gamma, self.buffer.is_terminals, True)
        returns = returns.to(self.device)

        # convert list to tensor
        old_states = torch.stack(self.buffer.states).to(self.device)
        old_actions = torch.stack(self.buffer.actions).detach().to(self.device)
        old_logprobs = torch.stack(self.buffer.logprobs).detach().to(self.device)
        
        pl_list = []
        vl_list = []
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)

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
        
    def build_obs(self, obs):
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
                                   [last_feedback]])
        return torch.tensor(prep_obs, dtype=torch.float).to(self.device)


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
            # Pad the history with zero tensors so that every input has the same size. 
            history = [torch.zeros(self.observation_size).to(self.device) for _ in range(self.history_len)]

            score = 0
            obs = self.env.reset()
            done = False
            while not done:
                pobs = self.env.preprocess_state(obs)
                pobs = self.build_obs(pobs)
                history.append(pobs)
                history_tensor = torch.stack(history[-self.history_len:])
                ac,_ = self.act(history_tensor, train=False)
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
            history = [torch.zeros(self.observation_size).to(self.device) for _ in range(self.history_len)]
            obs = self.env.reset()
            done = False
            current_ep_reward = 0

            while not done:
                pobs = self.env.preprocess_state(obs)
                pobs = self.build_obs(pobs)
                history.append(pobs)
                history_tensor = torch.stack(history[-self.history_len:])

                action, action_logprob = self.act(history_tensor, train=True)
                value = self.critic(history_tensor).item()
                
                self.buffer.states.append(history_tensor)
                self.buffer.values.append(value)
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




