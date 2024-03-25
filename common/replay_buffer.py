import numpy as np
import threading


class ReplayBuffer:
    
    def __init__(self, args):
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents_per_party = self.args.n_agents_per_party
        self.obs_shape = self.args.obs_shape
        self.size = self.args.buffer_size
        self.current_idx = 0
        self.current_size = 0
        
        self.buffers = {'o':             np.empty([self.size,  2 * self.n_agents_per_party, self.obs_shape]),
                        'u':             np.empty([self.size,  2 * self.n_agents_per_party,  1]),
                        'r':             np.empty([self.size,  1]),
                        'o_next':        np.empty([self.size,  2 * self.n_agents_per_party, self.obs_shape]),
                        'terminated':    np.empty([self.size,  1])
                        }

        # thread lock
        self.lock = threading.Lock()

    
    
    
    # store the episode
    def store_episode(self, episode_batch):
        batch_size = episode_batch['o'].shape[0]  # episode_number
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['o'][idxs] = episode_batch['o']
            self.buffers['u'][idxs] = episode_batch['u']
            self.buffers['r'][idxs] = episode_batch['r']
            self.buffers['o_next'][idxs] = episode_batch['o_next']
            self.buffers['terminated'][idxs] = episode_batch['terminated']

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
