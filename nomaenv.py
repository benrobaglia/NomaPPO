import numpy as np
from itertools import combinations
from scipy.stats import norm
from scipy.special import jv


class NomaEnv:
    """
    NOMA-PPO environment
    Below, we give details to clarify the use of the parameters.
    - traffic_model: can be 'aperiodic', 'periodic', 'heterogeneous:
        - 'aperiodic': 'lbdas' and 'aperiodic_type' need to be specified. period, arrival_probs, offsets are not used and can be set to None. 'aperiodic_type' can generate packets based on inter-arrival times or poisson distribution. Prefer 'lambda'.
        - 'periodic': 'period', 'arrival_probs', 'offsets' need to be specified. 'aperiodic': 'lbdas' and 'aperiodic_type' are not used and can be set to None.
        - 'heterogeneous': 'lbdas', 'aperiodic_type', 'period', 'arrival_probs', 'offsets' need to be specified. You need to specify the indices of periodic devices in'periodic_devices'. The complementary is aperiodic by default.
    - channel_model: can be 'collision' or 'interference'
    - full_obs: whether we want full observability of the channel state.
    - reward_type: 0 is the SARL reward, 1 is the MARL reward, 2 is the SARL reward penalized with the number of dropped packet in the frame 3 is the SARL reward penalized with the number of channel errors.
    The channel parameters are detailed in the code.
    """
    def __init__(self,
                 k,
                 deadlines,
                 lbdas,
                 period=5,
                 arrival_probs=None,
                 offsets=None,
                 episode_length=100,
                 max_simultaneous_devices=3,
                 traffic_model='aperiodic',
                 channel_model='collision',
                 aperiodic_type='lambda',
                 full_obs=False,
                 periodic_devices=[],
                 reward_type=0,
                 nt=4,
                 a=None,
                 v=3,
                 radius=None,
                 manual_distances=None,
                 manual_shadowing=None,
                 path_loss=True,
                 shadowing=True,
                 fast_fading=True,
                 verbose=False
    ):  
        self.k = k
        self.max_simultaneous_devices = max_simultaneous_devices
        self.lbdas = lbdas
        self.period = period
        self.deadlines = deadlines
        self.arrival_probs = arrival_probs
        self.full_obs = full_obs
        self.offsets = offsets
        self.episode_length = episode_length
        self.verbose = verbose
        self.traffic_model = traffic_model
        self.channel_model = channel_model
        self.aperiodic_type = aperiodic_type
        self.reward_type = reward_type
        self.periodic_devices = periodic_devices
        self.aperiodic_devices = [i for i in range(self.k) if i not in periodic_devices]

        self.action_space = list(range(k))
        # self.action_space_combinatorial = []
        # for i in range(1, k + 1):
        #     self.action_space_combinatorial += list(combinations(range(k), i))
        
        self.radius = radius
        self.manual_distances = manual_distances
        self.manual_shadowing = manual_shadowing
        self.path_loss = path_loss
        self.shadowing = shadowing
        self.fast_fading = fast_fading
        # Channel parameters
        self.W = 38.16e6 # 40MHz minus guard bands
        self.Ti = 1 / 30 * 1e-3 
        Tp = 2.34e-6
        T = self.Ti+Tp # slot duration
        self.Tf = 5*T # frame duration
        self.n = self.W*self.Ti
        L_info = 32 # information bits
        L_crc = 2 # CRC 38.212
        L_mac = 2 # 1 MAC subheader 38.321
        L_rlc = 0 # RLC header for Transparent Mode (TM) 38.322 
        L_pdcp = 2 # PDCP header format for 12 bits PDCP SN 38.323
        L_sdap = 0 # SDAP header 37.324
        L_ip = 40 # IP header v6
        self.L = (L_info+L_crc+L_mac+L_rlc+L_pdcp+L_sdap+L_ip)*8
        self.R = self.L/self.n # coding rate
        self.nt = nt # number of antennas
        self.Td = 100e-9 # Delay spread
        self.n_pilots = self.W * 2 * self.Td
        self.hb = 3
        self.hd = 1.5
        NF = 5
        N_db = -174+10*np.log10(self.W)+NF
        self.N = 10**(N_db/10)
        self.pt = 10**(23/10)
        self.fc = 4
        self.c = 3e8 # speed of light in m/s
        self.v = v * 1000/3600 # device speed in m/s
        self.sigma = 8.03
        self.G_BS = 5# + 10 * np.log10(self.nt)
        if a is None:
            self.a = self.compute_a(self.v)
        else:
            self.a = a
        self.init_devices()
        
    def get_coherence_time(self, v):
        return self.c/(8*self.fc*1e9*v)

    def icdf(self, t, i):
        return - np.log(1-t) / self.lbdas[i]

    # Functions to compute the error probability epsilon with Finite Block Length regime
    def compute_a(self, v):
        return jv(0,2*np.pi*self.fc*1e9*v/self.c*self.Tf)    
    
    def compute_data_rate(self, n_pilots):
        n = (self.W - n_pilots * 30e3) * self.Ti
        return self.L / n

    def capa(self, gamma):
        return np.log2(1+gamma)
    
    def disp(self, gamma):
        return gamma/2*(gamma+2)/(gamma+1)**2*np.log2(np.e)**2
    
    def compute_epsilon(self, gamma, n_pilots):
        epsilon = 1-norm.cdf(np.sqrt(self.n/self.disp(gamma))*(self.capa(gamma)-self.compute_data_rate(n_pilots=n_pilots)))
        return epsilon

    # Functions to initialize the devices and compute path loss and shadowing

    def add_path_loss(self, d):
        # 1<=d3D<=150m d en m
        d3D = np.sqrt(d**2+(self.hb-self.hd)**2)
        PLL = 32.4 + 17.3 * np.log10(d3D) + 20 * np.log10(self.fc)
        PLN = 38.3 * np.log10(d3D) + 17.3 + 24.9 * np.log10(self.fc)
        return np.maximum(PLL, PLN)

    def init_shadowing(self):
        if self.manual_shadowing is None:
            return np.random.normal(0, self.sigma, (self.k))
        else:
            return self.manual_shadowing
        
    def init_devices(self):
        if self.radius is None:
            if self.manual_distances is None:
                self.distances = np.random.uniform(1, 150, (self.k))
            else:
                self.distances = self.manual_distances
        else:
            # self.distances = np.ones(self.k) * self.distances 
            self.distances = np.random.uniform(1, self.radius, (self.k))

        if self.path_loss:
            pl = - self.add_path_loss(self.distances)
        else:
            pl = np.zeros(self.k)
        
        if self.shadowing:
            shd = self.init_shadowing()
        else:
            shd = np.zeros(self.k)
        
        self.G_UE = pl + self.G_BS + shd
        
        if self.verbose:
            print(f"Distances: {self.distances}\nGain UE: {self.G_UE}\nn:{self.n}, a:{self.a}")
  

    # fast fading functions

    def add_fading(self, size):
        return np.random.exponential(1, size)

    def sample_multivariate(self, a):
        #size: (k, nt, 2)
        return np.random.multivariate_normal(np.array([0, 0]), (1-a**2)*0.5 * np.diag([1, 1]), size=(self.k, self.nt))
    
    def update_h(self, h):
        z = self.sample_multivariate(self.a)
        return self.a*h + z
    
    # Initialize the environment

    def reset(self):

        self.current_state = np.zeros((self.k, np.max(self.deadlines)))        
        # Set arrival times
        self.arrival_times = [[] for _ in range(self.k)]
        
        if self.traffic_model == 'aperiodic':
            for i in range(self.k):
                t1 = np.random.rand()
                self.arrival_times[i].append(np.ceil(self.icdf(t1, i)))
            if self.aperiodic_type == 'lambda':
                for i in range(self.k):
                    self.current_state[i, self.deadlines[i]-1] = np.random.poisson(self.lbdas[i])
        
        elif self.traffic_model == 'periodic':
            active_offsets = np.where(self.offsets == 0)[0]
            for ao in active_offsets:
                self.current_state[int(ao), self.deadlines[ao]-1] = np.random.binomial(1, self.arrival_probs[ao])
        
        elif self.traffic_model == 'heterogeneous':
            for i in self.aperiodic_devices:
                self.current_state[i, self.deadlines[i]-1] = np.random.poisson(self.lbdas[i])
            
            for i in self.periodic_devices:
                if self.offsets[i] == 0:
                    self.current_state[int(i), self.deadlines[i]-1] = np.random.binomial(1, self.arrival_probs[i])

        self.agent_obs = np.zeros((self.k, np.max(self.deadlines)))

        # Initialize channel
        self.init_devices()
        
        # Initialize h
        self.h_history = []
        h = self.sample_multivariate(0)
        self.h_history.append(h)
        
        cmat = np.matrix(h[:,:, 0] + 1j*(h[:, :, 1]))
        h_coeffs = cmat.dot(cmat.getH())
        h2 = np.absolute(np.diag(h_coeffs))
        if self.verbose:
            print(f"h2: {h2}")

        self.sinrs = []
        self.nb_sense = 0
        self.timestep = 0
        self.discarded_packets = np.zeros(self.k)
        self.received_packets = np.copy(self.current_state).sum(1)
        self.successful_transmissions = 0
        self.channel_losses = 0
        self.last_time_transmitted = np.ones(self.k)
        self.last_time_since_polled = np.ones(self.k)
        self.last_time_since_active = np.ones(self.k)*self.episode_length
        self.last_feedback = 0
        self.n_collisions = 0
        self.Ha = np.zeros(self.k)
        # self.Ha = np.zeros(self.k) + self.episode_length        
        return np.copy(self.agent_obs).astype(np.float32)
    
    def preprocess_state(self, state):
        # We preprocess the buffers in this function and return the vector of time-to-deadline for each user.
        output = []
        for row in state:
            packets = np.nonzero(row)[0]
            if len(packets) > 0:
                output.append(packets.min())
            else:
                output.append(-1)
        return np.array(output)

    def compute_channel_components(self):
        # Function to evolve the channel and compute the new channel components.
        g = 10**(self.G_UE / 10)
        new_h = self.update_h(self.h_history[-1])
        self.h_history.append(new_h)
        cmat = np.matrix(new_h[:,:, 0] + 1j*(new_h[:, :, 1]))
        h_coeffs = cmat.dot(cmat.getH())
        return self.pt * g, h_coeffs
    
    def decode_signal(self, attempts, pg, h_coeffs):
        # Function to decode based on the attempts (binary vector) and the channel components.
        h_coeffs = np.array(h_coeffs)
        attempts_idx = attempts.nonzero()[0]
        h2 = np.absolute(np.diag(h_coeffs))
        eta = pg * h2.reshape(-1)
        eta_attempts = eta[attempts_idx]
        decoding_order = (-eta_attempts).argsort() # We put "-" because we want to sort by descending order
        # To have the real indices of the devices to decode: attempts_idx[decoding_order]
        if self.verbose:
            print(f"attempts: {attempts_idx}, eta: {eta}, decoding order: {attempts_idx[decoding_order]}")
            print(f"fast fading ||h||^2: {h2}")
            # print(f"h_coeffs: {h_coeffs}")
        decoded_idx = []
        sinrs_attempts = []
        eps_list = []
        for i, device in enumerate(decoding_order):
            interference = np.delete(decoding_order, decoded_idx + [i])
            sinr = eta_attempts[device] / (self.N + (pg[attempts_idx[interference]] * np.absolute(h_coeffs[attempts_idx[device],
                                                                                                         attempts_idx[interference]])**2 / h2[attempts_idx[device]]).sum())
            # sinr = eta_attempts[device] / (self.N + (eta_attempts[interference]).sum())

            sinrs_attempts.append(sinr)
            eps = self.compute_epsilon(sinr, self.n_pilots)
            eps_list.append(eps)
            rv = np.random.binomial(1, 1-eps)
            if rv == 1:
                decoded_idx.append(i)
            if self.verbose:
                print(f"Device: {attempts_idx[device]}, sinr {sinr}, interference: {attempts_idx[interference]}")
                print(f"epsilon: {eps}, decoded: {rv}")
        decoded_idx = attempts_idx[decoding_order[decoded_idx]]
        return decoded_idx, sinrs_attempts, eps_list

    def compute_jains(self):
        urllc_scores = []
        for k in range(self.k):
            if self.received_packets[k] > 0:
                urllc_scores.append(1 - self.discarded_packets[k] / self.received_packets[k])
            else:
                urllc_scores.append(1)
        urllc_scores = np.array(urllc_scores)
        jains = urllc_scores.sum() ** 2 / self.k / (urllc_scores ** 2).sum()
        return jains

    def compute_urllc(self):
        urllc_scores = 1 - self.discarded_packets.sum() / self.received_packets.sum()
        return urllc_scores

    def step(self, action):
        # Action is a np.array of bool (1 if device activated)
        devices_polled = action.nonzero()[0]
        self.timestep += 1
        reward_marl = np.zeros(self.k)
        reward_sarl = 0
        discarded_step = 0

        next_state = np.copy(self.current_state)
        next_obs = np.copy(self.agent_obs)
        
        # Increment the scheduling features (tau)
        self.last_time_since_polled += 1
        self.last_time_transmitted += 1
        self.last_time_since_active += 1
        self.last_time_since_polled[devices_polled] = 1.

        # Check what device has a packet
        has_a_packet = (self.current_state.sum(1) > 0) * 1.
        
        attempts = action * has_a_packet
        active_idx = attempts.nonzero()[0]
        self.last_time_since_active[active_idx] = 1
        
        # Executing the action and decoding the packets
        m = attempts.sum() # Number of transmission attempts

        pg, h_coeffs = self.compute_channel_components()
        h2 = np.absolute(np.diag(h_coeffs))
        if not self.full_obs:
            self.Ha[active_idx] = pg[active_idx]*h2[active_idx].copy()
        else:
            self.Ha = pg*h2.copy()
            self.last_time_since_active[:] = 1

        if m <= self.max_simultaneous_devices:
            if self.channel_model == 'collision':
                reward_sarl = m
                self.successful_transmissions += m
                decoded_idx = active_idx
            elif self.channel_model == 'interference':
                decoded_idx, sinrs, _ = self.decode_signal(attempts, pg, h_coeffs)
                self.sinrs.append(sinrs)
                reward_sarl = len(decoded_idx)
                self.successful_transmissions += len(decoded_idx)
                self.channel_losses += (len(active_idx) - len(decoded_idx)>0)*1
            self.last_time_transmitted[decoded_idx] = 1.
            reward_marl[:] = len(decoded_idx)            
            
            # Remove the decoded packets in the buffers 
            devices_polled_obs = []
            for d in decoded_idx:
                col = next_state[d].nonzero()[0]
                if len(col) > 0:
                    devices_polled_obs.append(col.min())
                else:
                    devices_polled_obs.append(0)
                    print("error, should not be possible")

            next_state[decoded_idx, devices_polled_obs] -= 1
            
            # Same for the agent obs 
            devices_polled_obs = []
            for d in decoded_idx:
                col = next_obs[d].nonzero()[0]
                if len(col) > 0:
                    devices_polled_obs.append(col.min())
                else:
                    devices_polled_obs.append(0)

            next_obs[decoded_idx, devices_polled_obs] = 0.


        elif m > self.max_simultaneous_devices:
            reward_sarl = 0
            reward_marl[:] = -1
            self.n_collisions += 1
            self.channel_losses += 1

        else:
            raise ValueError("m takes an impossible value")           

        # Penalize the reward
        # to_penalize = (self.last_time_since_polled > self.deadlines.max())*1
        # reward_sarl -= 1/self.k * to_penalize.sum()
        
        # Incrementation
        exp_arr = np.zeros(self.k)
        row, col = np.nonzero(next_state)
        new_col = col - 1
        expired = new_col < 0
        exp_packets = row[expired.nonzero()[0]]
        exp_arr[exp_packets] = 1
        self.discarded_packets += exp_arr
        discarded_step = expired.sum()
        new_col = new_col[~expired] # Remove the expired packets
        new_row = row[~expired]
        next_state[row, col] = 0
        next_state[new_row, new_col] = 1

        # reward_marl -= 0.1 * expired.sum()
        
        row, col = np.nonzero(next_obs)
        new_col = col - 1
        expired = new_col < 0
        new_col = new_col[~expired] # Remove the expired packets
        new_row = row[~expired]
        next_obs[row, col] = 0
        next_obs[new_row, new_col] = 1
        
        # Receive new packets
        if self.traffic_model == 'aperiodic':
            if self.aperiodic_type == "inter-arrival":
                for i in range(self.k):
                    if self.arrival_times[i][-1] == self.timestep:
                        # A packet is generated and we sample another interarrival time
                        next_state[i, self.deadlines[i]-1] = 1.
                        self.received_packets[i] += 1.
                        t1 = np.random.rand()
                        self.arrival_times[i].append(self.arrival_times[i][-1] + np.ceil(self.icdf(t1, i)))
            elif self.aperiodic_type == "lambda":
                for i in range(self.k):
                    next_state[i, self.deadlines[i]-1] = np.random.poisson(self.lbdas[i])
                    self.received_packets[i] += next_state[i, self.deadlines[i]-1]
        elif self.traffic_model == 'periodic':
            active_offsets = np.where(self.timestep % self.period == self.offsets)[0]
            for ao in active_offsets:
                next_state[ao, self.deadlines[ao]-1] = np.random.binomial(1, self.arrival_probs[ao])
                self.received_packets[ao] += next_state[ao, self.deadlines[ao]-1]
        
        elif self.traffic_model == 'heterogeneous':
            for i in self.aperiodic_devices:
                next_state[i, self.deadlines[i]-1] = np.random.poisson(self.lbdas[i])
                self.received_packets[i] += next_state[i, self.deadlines[i]-1]
            
            for i in self.periodic_devices:
                if self.timestep % self.period[i] == self.offsets[i]:
                    next_state[int(i), self.deadlines[i]-1] = np.random.binomial(1, self.arrival_probs[i])
                    self.received_packets[i] += next_state[i, self.deadlines[i]-1]

        
        # Load buffers of the polled devices through the transmitted packets.
        if m <= self.max_simultaneous_devices:
            state = next_state.copy()
            state[:, -1] = 0 # The agent cannot see the new arrivals
            next_obs[devices_polled] = state[devices_polled].copy()
        else:
            # There was a collision
            # Get whether a device has a packet or not through the pilots
            sensing_state = next_state.copy()
            # Don't sense the devices that have not been polled
            not_sense = np.setdiff1d(np.arange(self.k), devices_polled)
            sensing_state[not_sense] = 0
            sensing_state[:, -1] = 0 # The agent cannot see the new arrivals. In sensing state, we have the buffers of the polled devices without the new packets

            next_state_buffers = sensing_state.sum(1).nonzero()[0]
            next_obs_buffers = next_obs.sum(1).nonzero()[0] # Ids of the devices that have a packet in the obs.
            obs_to_add = np.setdiff1d(next_state_buffers, next_obs_buffers) # We do not update the buffers of a device that we know has a packet.
            # next_obs[obs_to_add, 3] = 1.
            next_obs[devices_polled] = sensing_state[devices_polled].copy()

        if self.reward_type == 0:
            reward = reward_sarl
            self.last_feedback = reward
        elif self.reward_type == 1:
            reward = reward_marl
            self.last_feedback = reward.sum()
        elif self.reward_type == 2:
            reward = reward_sarl - discarded_step / self.k
            self.last_feedback = reward
        elif self.reward_type == 3:
            reward = reward_sarl - self.channel_losses / self.k

        if self.verbose:
            print(f"Timestep {self.timestep}")
            print(f"State {self.current_state}")
            print(f"Observation {self.agent_obs}")
            print(f"Action {action}")
            print(f"Devices polled {devices_polled}")
            print(f'Next state {next_state}')
            print(f'Agent next obs {next_obs}')
            print(f"Reward {reward}")
            print(f"Received packets {self.received_packets}")
            print(f"Number of discarded packets {self.discarded_packets.sum()}")
            print("")
        
        if (self.timestep >= self.episode_length) :    
            done = True
        else:
            done = False
            
        self.current_state = np.copy(next_state)
        self.agent_obs = np.copy(next_obs)
        
        return np.copy(self.agent_obs).astype(np.float32), reward, done
        

class ToyChannelEnv:
    """
    Toy model to study the behaviour of the algorithm under different channel conditions.
    Channel is modeled according to a Gilbert-Elliot model with:
    - channel_decoding probability
    - channel_switch probability
    This environment is not used in the paper and was only used for research purposes.
    """
    def __init__(self,
                 k,
                 deadlines,
                 lbdas,
                 period=5,
                 arrival_probs=None,
                 offsets=None,
                 episode_length=100,
                 max_simultaneous_devices=3,
                 traffic_model='aperiodic',
                 reward_type=0,
                 channel_switch=0.2,
                 channel_decoding=0.8,
                 channel_observability='partial',
                 verbose=False
    ):  
        self.k = k
        self.max_simultaneous_devices = max_simultaneous_devices
        self.lbdas = lbdas
        self.period = period
        self.deadlines = deadlines
        self.arrival_probs = arrival_probs
        self.offsets = offsets
        self.episode_length = episode_length
        self.verbose = verbose
        self.traffic_model = traffic_model
        self.reward_type = reward_type

        self.channel_switch = channel_switch
        self.channel_decoding = channel_decoding
        self.channel_observability = channel_observability

        self.action_space = list(range(k))
                
    
    def reset(self):

        self.current_state = np.zeros((self.k, np.max(self.deadlines)))        
        # Set arrival times
        self.arrival_times = [[] for _ in range(self.k)]
        
        if self.traffic_model == 'aperiodic':
            for i in range(self.k):
                t1 = np.random.rand()
                self.arrival_times[i].append(np.ceil(self.icdf(t1, i)))
                self.current_state[i, self.deadlines[i]-1] = np.random.poisson(self.lbdas[i])
        
        elif self.traffic_model == 'periodic':
            active_offsets = np.where(self.offsets == 0)[0]
            for ao in active_offsets:
                self.current_state[int(ao), self.deadlines[ao]-1] = np.random.binomial(1, self.arrival_probs[ao])
        
        self.agent_obs = np.zeros((self.k, np.max(self.deadlines)))

        # Set channel
        self.channel_state = np.random.choice([self.channel_decoding, 1-self.channel_decoding], self.k)

        self.timestep = 0
        self.discarded_packets = np.zeros(self.k)
        self.received_packets = np.copy(self.current_state).sum(1)
        self.successful_transmissions = 0
        self.channel_losses = 0
        self.last_time_transmitted = np.ones(self.k)*10
        self.last_time_since_polled = np.ones(self.k)*10
        self.last_time_since_active = np.ones(self.k)*10
        self.last_feedback = 0
        self.n_collisions = 0
        self.Ha = np.zeros(self.k)
        
        return np.copy(self.agent_obs).astype(np.float32)
    
    def preprocess_state(self, state):
        output = []
        for row in state:
            packets = np.nonzero(row)[0]
            if len(packets) > 0:
                output.append(packets.min())
            else:
                output.append(-1)
        return np.array(output)

    
    def decode_signal(self, attempts):
        attempts_idx = attempts.nonzero()[0]
        decoded_result = np.random.binomial(1, self.channel_state[attempts_idx])
        decoded_result_idx = decoded_result.nonzero()[0]
        decoded_idx = attempts_idx[decoded_result_idx]

        # Change the channel
        change_idx = np.random.binomial(1, self.channel_switch, self.k).nonzero()[0]
        self.channel_state[change_idx] = 1 - self.channel_state[change_idx]
        return decoded_idx

    def compute_jains(self):
        urllc_scores = []
        for k in range(self.k):
            if self.received_packets[k] > 0:
                urllc_scores.append(1 - self.discarded_packets[k] / self.received_packets[k])
            else:
                urllc_scores.append(1)
        urllc_scores = np.array(urllc_scores)
        jains = urllc_scores.sum() ** 2 / self.k / (urllc_scores ** 2).sum()
        return jains

    def compute_urllc(self):
        urllc_scores = 1 - self.discarded_packets.sum() / self.received_packets.sum()
        return urllc_scores

    def compute_prior_channel(self):
        P = np.matrix([[1-self.channel_switch, self.channel_switch], [self.channel_switch, 1-self.channel_switch]])
        Pns = [np.linalg.matrix_power(P, int(n)) for n in self.last_time_since_polled]
        priors = [Pns[k][int(self.Ha[k]), 1] for k in range(self.k)]
        return np.array(priors)


    def step(self, action):
        # Action is a np.array of bool (1 if device activated)
        devices_polled = action.nonzero()[0]
        self.timestep += 1
        reward_marl = np.zeros(self.k)
        reward_sarl = 0
        discarded_step = 0

        next_state = np.copy(self.current_state)
        next_obs = np.copy(self.agent_obs)
        
        # Increment the scheduling features
        self.last_time_since_polled += 1
        self.last_time_since_polled[devices_polled] = 1.
        self.last_time_transmitted += 1
        
        # Check what device has a packet
        has_a_packet = (self.current_state.sum(1) > 0) * 1.
        
        attempts = action * has_a_packet
        attempts_idx = attempts.nonzero()[0]
        decoded_idx = 0
        # Executing the action
        m = attempts.sum() # Number of transmission attempts

        if self.channel_observability == 'partial':
            self.Ha[devices_polled] = self.channel_state[devices_polled].copy()
        elif self.channel_observability == 'full':
            self.Ha = self.channel_state.copy()

        # print(f"Test print: {self.Ha}, {self.channel_state}")
        if m <= self.max_simultaneous_devices:
            decoded_idx = self.decode_signal(attempts)
            reward_sarl = len(decoded_idx)
            self.successful_transmissions += len(decoded_idx)
            self.channel_losses += (len(attempts_idx) - len(decoded_idx) > 0) * 1
            self.last_time_transmitted[decoded_idx] = 1.
            reward_marl[:] = len(decoded_idx)            
            
            # Remove the decoded packets in the buffers 
            devices_polled_obs = []
            for d in decoded_idx:
                col = next_state[d].nonzero()[0]
                if len(col) > 0:
                    devices_polled_obs.append(col.min())
                else:
                    devices_polled_obs.append(0)
                    print("error, should not be possible")

            next_state[decoded_idx, devices_polled_obs] -= 1
            
            # Same for the agent obs 
            devices_polled_obs = []
            for d in decoded_idx:
                col = next_obs[d].nonzero()[0]
                if len(col) > 0:
                    devices_polled_obs.append(col.min())
                else:
                    devices_polled_obs.append(0)

            next_obs[decoded_idx, devices_polled_obs] = 0.


        elif m > self.max_simultaneous_devices:
            reward_sarl = 0
            reward_marl[:] = -1
            self.n_collisions += 1
            self.channel_losses += 1

        else:
            raise ValueError("m takes an impossible value")           

        # Penalize the reward
        to_penalize = (self.last_time_since_polled > self.deadlines.max())*1
        reward_sarl -= 1/self.k * to_penalize.sum()
        
        # Incrementation
        exp_arr = np.zeros(self.k)
        row, col = np.nonzero(next_state)
        new_col = col - 1
        expired = new_col < 0
        exp_packets = row[expired.nonzero()[0]]
        exp_arr[exp_packets] = 1
        self.discarded_packets += exp_arr
        discarded_step = expired.sum()
        new_col = new_col[~expired] # Remove the expired packets
        new_row = row[~expired]
        next_state[row, col] = 0
        next_state[new_row, new_col] = 1

        # reward_marl -= 0.1 * expired.sum()
        
        row, col = np.nonzero(next_obs)
        new_col = col - 1
        expired = new_col < 0
        new_col = new_col[~expired] # Remove the expired packets
        new_row = row[~expired]
        next_obs[row, col] = 0
        next_obs[new_row, new_col] = 1
        
        # Receive new packets
        if self.traffic_model == 'aperiodic':
            for i in range(self.k):
                next_state[i, self.deadlines[i]-1] = np.random.poisson(self.lbdas[i])
                self.received_packets[i] += next_state[i, self.deadlines[i]-1]
        elif self.traffic_model == 'periodic':
            active_offsets = np.where(self.timestep % self.period == self.offsets)[0]
            for ao in active_offsets:
                next_state[ao, self.deadlines[ao]-1] = np.random.binomial(1, self.arrival_probs[ao])
                self.received_packets[ao] += next_state[ao, self.deadlines[ao]-1]
        
        # Load buffers of the polled devices through the transmitted packets.
        if m <= self.max_simultaneous_devices:
            state = next_state.copy()
            state[:, -1] = 0 # The agent cannot see the new arrivals
            next_obs[devices_polled] = state[devices_polled]
        else:
            # There was a collision
            # Get whether a device has a packet or not through the pilots
            sensing_state = next_state.copy()
            # Don't sense the devices that have not been polled
            not_sense = np.setdiff1d(np.arange(self.k), devices_polled)
            sensing_state[not_sense] = 0
            sensing_state[:, -1] = 0 # The agent cannot see the new arrivals. In sensing state, we have the buffers of the polled devices without the new packets

            next_state_buffers = sensing_state.sum(1).nonzero()[0]
            next_obs_buffers = next_obs.sum(1).nonzero()[0] # Ids of the devices that have a packet in the obs.
            obs_to_add = np.setdiff1d(next_state_buffers, next_obs_buffers) # We do not update the buffers of a device that we know has a packet.
            # next_obs[obs_to_add, 3] = 1.
            next_obs[devices_polled] = sensing_state[devices_polled]

        if self.reward_type == 0:
            reward = reward_sarl
            self.last_feedback = reward
        elif self.reward_type == 1:
            reward = reward_marl
            self.last_feedback = reward.sum()
        elif self.reward_type == 2:
            reward = reward_sarl - discarded_step / self.k
            self.last_feedback = reward
        elif self.reward_type == 3:
            reward = reward_sarl - self.channel_losses / self.k

        if self.verbose:
            print(f"Timestep {self.timestep}")
            print(f"State {self.current_state}")
            print(f"Observation {self.agent_obs}")
            print(f"Action {action}")
            print(f"Devices polled {devices_polled}")
            print(f"Devices decoded {decoded_idx}")
            print(f"Channel errors: {self.channel_losses}")
            print(f"Channel state: {self.channel_state}")
            print(f'Next state {next_state}')
            print(f'Agent next obs {next_obs}')
            print(f"Reward {reward}")
            print(f"Received packets {self.received_packets}")
            print(f"Number of discarded packets {self.discarded_packets.sum()}")
            print("")
        
        if (self.timestep >= self.episode_length) :    
            done = True
        else:
            done = False
            
        self.current_state = np.copy(next_state)
        self.agent_obs = np.copy(next_obs)
        
        
        return np.copy(self.agent_obs).astype(np.float32), reward, done