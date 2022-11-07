import numpy as np
from itertools import combinations
from scipy.stats import norm
from scipy.special import jv

class NomaEnv:

    def __init__(self,
                 k,
                 deadlines,
                 lbda=1/9.3,
                 period=5,
                 arrival_probs=None,
                 offsets=None,
                 episode_length=100,
                 max_simultaneous_devices=3,
                 traffic_model='aperiodic',
                 channel_model='collision',
                 distances=None,
                 path_loss=True,
                 shadowing=True,
                 fast_fading=True,
                 verbose=False
    ):  
        self.k = k
        self.max_simultaneous_devices = max_simultaneous_devices
        self.lbda = lbda
        self.period = period
        self.deadlines = deadlines
        self.arrival_probs = arrival_probs
        self.offsets = offsets
        self.episode_length = episode_length
        self.verbose = verbose
        self.traffic_model = traffic_model
        self.channel_model = channel_model

        self.action_space = list(range(k))
        # self.action_space_combinatorial = []
        # for i in range(1, k + 1):
        #     self.action_space_combinatorial += list(combinations(range(k), i))
        
        self.distances = distances
        self.path_loss = path_loss
        self.shadowing = shadowing
        self.fast_fading = fast_fading
        # Channel parameters
        self.W = 38.16e6 # 40MHz minus guard bands
        Ti = 1 / 30 * 1e-3 
        Tp = 2.34e-6
        T = Ti+Tp # slot duration
        self.Tf = 6*T # frame duration
        self.n = self.W*Ti
        L_info = 32 # information bits
        L_crc = 2 # CRC 38.212
        L_mac = 2 # 1 MAC subheader 38.321
        L_rlc = 0 # RLC header for Transparent Mode (TM) 38.322 
        L_pdcp = 2 # PDCP header format for 12 bits PDCP SN 38.323
        L_sdap = 0 # SDAP header 37.324
        L_ip = 40 # IP header v6
        self.L = (L_info+L_crc+L_mac+L_rlc+L_pdcp+L_sdap+L_ip)*8
        self.R = self.L/self.n # coding rate
        self.nt = 4 # number of antennas
        self.hb = 3
        self.hd = 1.5
        NF = 5
        N_db = -174+10*np.log10(self.W)+NF
        self.N = 10**(N_db/10)
        self.pt = 10**(23/10)
        self.fc = 4
        c = 3e8 # speed of light in m/s
        v = 3000/3600 # device speed in m/s
        self.sigma = 8.03
        self.G_BS = 5 + 10 * np.log10(self.nt)
        self.a = jv(0,2*np.pi*self.fc*1e9*v/c*self.Tf)
        self.init_devices()
        
    def icdf(self, t):
        return - np.log(1-t) / self.lbda
    
    def capa(self, gamma):
        return np.log2(1+gamma)
    
    def disp(self, gamma):
        return gamma/2*(gamma+2)/(gamma+1)**2*np.log2(np.e)**2
    
    def compute_epsilon(self, gamma):
        epsilon = 1-norm.cdf(np.sqrt(self.n/self.disp(gamma))*(self.capa(gamma)-self.R))
        return epsilon
    
    def add_path_loss(self, d):
        # 1<=d3D<=150m d en m
        d3D = np.sqrt(d**2+(self.hb-self.hd)**2)
        PLL = 32.4 + 17.3 * np.log10(d3D) + 20 * np.log10(self.fc)
        PLN = 38.3 * np.log10(d3D) + 17.3 + 24.9 * np.log10(self.fc)
        return np.maximum(PLL, PLN)

    def init_shadowing(self):
        return np.random.normal(0, self.sigma, (self.k))
    
    def add_fading(self, size):
        return np.random.exponential(1, size)

    def init_devices(self):
        if self.distances is None:
            self.distances = np.random.uniform(1, 150, (self.k))
        else:
            self.distances = np.ones(self.k) * self.distances 
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
  

    #### fast fading functions ####
    def sample_multivariate(self, a):
        #size: (k, nt, 2)
        return np.random.multivariate_normal(np.array([0, 0]), (1-a**2)*0.5 * np.diag([1, 1]), size=(self.k, self.nt))
    
    def update_h(self, h):
        z = self.sample_multivariate(self.a)
        return self.a*h + z
    
    def reset(self):

        self.current_state = np.zeros((self.k, np.max(self.deadlines)))        
        # Set arrival times
        self.arrival_times = [[] for _ in range(self.k)]
        
        if self.traffic_model == 'aperiodic':
            for i in range(self.k):
                t1 = np.random.rand()
                self.arrival_times[i].append(np.ceil(self.icdf(t1)))
        
        elif self.traffic_model == 'periodic':
            active_offsets = np.where(self.offsets == 0)[0]
            for ao in active_offsets:
                self.current_state[int(ao), self.deadlines[ao]-1] = np.random.binomial(1, self.arrival_probs[ao])

        self.agent_obs = np.zeros((self.k, np.max(self.deadlines)))
        
        # Initialize h
        self.h_history = []
        h = self.sample_multivariate(0)
        self.h_history.append(h)
        
        self.sinrs = []
        self.timestep = 0
        self.discarded_packets = 0
        self.received_packets = np.copy(self.current_state).sum(1)
        self.successful_transmissions = 0
        self.last_time_transmitted = np.ones(self.k)
        self.last_time_since_polled = np.ones(self.k)
        self.last_feedback = 0
        
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

    def compute_channel_components(self):
        g = 10**(self.G_UE / 10)
        new_h = self.update_h(self.h_history[-1])
        self.h_history.append(new_h)
        cmat = np.matrix(new_h[:,:, 0] + 1j*(new_h[:, :, 1]))
        h_coeffs = cmat.dot(cmat.getH())
        return self.pt * g, h_coeffs
    
    def decode_signal(self, attempts, pg, h_coeffs):
        attempts_idx = attempts.nonzero()[0]
        h2 = np.linalg.norm(h_coeffs.dot(h_coeffs.getH()).diagonal(), axis=0)
        eta = pg*h2
        eta_attempts = eta[attempts_idx]
        decoding_order = (-eta_attempts).argsort() # We put "-" because we want to sort by descending
        # To have the real indices of the devices to decode: attempts_idx[decoding_order]
        if self.verbose:
            print(f"attempts: {attempts_idx}, eta: {eta}, decoding order: {attempts_idx[decoding_order]}")
            print(f"fast fading ||h||^2: {h2}")
            print(f"h_coeffs: {h_coeffs}")
        decoded_idx = []
        sinrs_attempts = []
        for i, device in enumerate(decoding_order):
            interference = np.delete(decoding_order, decoded_idx + [i])
            sinr = eta_attempts[device] / (self.N + (pg[attempts_idx[interference]] * np.linalg.norm(h_coeffs[attempts_idx[device],
                                                                                                         attempts_idx[interference]],axis=0) / h2[attempts_idx[device]]).sum())
            # sinr = eta_attempts[device] / (self.N + (eta_attempts[interference]).sum())

            sinrs_attempts.append(sinr)
            eps = self.compute_epsilon(sinr)
            rv = np.random.binomial(1, 1-eps)
            if rv == 1:
                decoded_idx.append(i)
            if self.verbose:
                print(f"Device: {attempts_idx[device]}, sinr {sinr}, interference: {attempts_idx[interference]}")
                print(f"epsilon: {eps}, decoded: {rv}")
        decoded_idx = attempts_idx[decoding_order[decoded_idx]]
        return decoded_idx, sinrs_attempts

    
    def step(self, action):
        # Action is a np.array of bool (1 if device activated)
        devices_polled = action.nonzero()[0]
        self.timestep += 1

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
        
        # Executing the action
        m = attempts.sum() # Number of transmission attempts

        if m <= self.max_simultaneous_devices:
            if self.channel_model == 'collision':
                reward = m
                self.successful_transmissions += m
                decoded_idx = attempts_idx
            elif self.channel_model == 'interference':
                pg, h_coeffs = self.compute_channel_components()
                decoded_idx, sinrs = self.decode_signal(attempts, pg, h_coeffs)
                self.sinrs.append(sinrs)
                reward = len(decoded_idx)
                self.successful_transmissions += len(decoded_idx)
            
            # Remove the decoded packets in the buffers 
            devices_polled_obs = []
            for d in decoded_idx:
                col = next_state[d].nonzero()[0]
                if len(col) > 0:
                    devices_polled_obs.append(col.min())
                else:
                    devices_polled_obs.append(0)

            next_state[decoded_idx, devices_polled_obs] = 0.
            
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
            reward = -1

        else:
            raise ValueError("m takes an impossible value")           

        
        # Incrementation
        row, col = np.nonzero(next_state)
        new_col = col - 1
        expired = new_col < 0
        self.discarded_packets += expired.sum()
        new_col = new_col[~expired] # Remove the expired packets
        new_row = row[~expired]
        next_state[row, col] = 0
        next_state[new_row, new_col] = 1
        
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
                if self.arrival_times[i][-1] == self.timestep:
                    # A packet is generated and we sample another interarrival time
                    next_state[i, self.deadlines[i]-1] = 1.
                    self.received_packets[i] += 1.
                    t1 = np.random.rand()
                    self.arrival_times[i].append(self.arrival_times[i][-1] + np.ceil(self.icdf(t1)))
        elif self.traffic_model == 'periodic':
            active_offsets = np.where(self.timestep % self.period == self.offsets)[0]
            for ao in active_offsets:
                next_state[ao, self.deadlines[ao]-1] = np.random.binomial(1, self.arrival_probs[ao])
                self.received_packets[ao] += next_state[ao, self.deadlines[ao]-1]
        
        # Load buffers of the polled devices
        if (next_state[devices_polled].sum(1) > 0).sum() <= self.max_simultaneous_devices:
            next_obs[devices_polled] = next_state[devices_polled]
        
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
            print(f"Number of discarded packets {self.discarded_packets}")
            print("")
        
        if (self.timestep >= self.episode_length) :    
            done = True
        else:
            done = False
            
        self.current_state = np.copy(next_state)
        self.agent_obs = np.copy(next_obs)
        self.last_feedback = reward
        
        return np.copy(self.agent_obs).astype(np.float32), reward, done
        
