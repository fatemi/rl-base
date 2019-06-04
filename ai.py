import numpy as np
from utils import ExperienceReplay
from model import Network, LargeNetwork, NatureNetwork
import torch
import torch.nn as nn
import torch.optim as optim


class AI(object):
    def __init__(self, state_shape, nb_actions, action_dim, reward_dim, history_len=1, gamma=.99,
                 learning_rate=0.00025, epsilon=0.05, final_epsilon=0.05, test_epsilon=0.0, annealing_steps=1000,
                 minibatch_size=32, replay_max_size=100, update_freq=50, learning_frequency=1, ddqn=False,
                 network_size='nature', normalize=1., rng=None, device=None):
        self.rng = rng
        self.history_len = history_len
        self.state_shape = state_shape
        self.nb_actions = nb_actions
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.start_epsilon = epsilon
        self.test_epsilon = test_epsilon
        self.final_epsilon = final_epsilon
        self.decay_steps = annealing_steps
        self.minibatch_size = minibatch_size
        self.network_size = network_size
        self.update_freq = update_freq
        self.update_counter = 0
        self.normalize = normalize
        self.learning_frequency = learning_frequency
        self.replay_max_size = replay_max_size
        self.transitions = ExperienceReplay(max_size=self.replay_max_size, history_len=history_len,
                                            state_shape=state_shape, action_dim=action_dim, reward_dim=reward_dim)
        self.ddqn = ddqn
        self.device = device
        self.network = self._build_network()
        self.target_network = self._build_network()
        self.weight_transfer(from_model=self.network, to_model=self.target_network)
        self.network.to(self.device)
        self.target_network.to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, amsgrad=True)

    def _build_network(self):
        if self.network_size == 'small':
            return Network()
        elif self.network_size == 'large':
            return LargeNetwork(state_shape=self.state_shape, nb_channels=4, nb_actions=self.nb_actions)
        elif self.network_size == 'nature':
            return NatureNetwork(state_shape=self.state_shape, nb_channels=4, nb_actions=self.nb_actions)
        else:
            raise ValueError('Invalid network_size.')

    def train_on_batch(self, s, a, r, s2, t):
        s = torch.FloatTensor(s).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        t = torch.FloatTensor(np.float32(t)).to(self.device)

        r.clamp_(min=-1, max=1)

        q = self.network(s / self.normalize)
        q2 = self.target_network(s2 / self.normalize).detach()
        q_pred = q.gather(1, a.unsqueeze(1)).squeeze(1) 
        if self.ddqn:
            q2_net = self.network(s2 / self.normalize).detach()
            q2_max = q2.gather(1, torch.max(q2_net, 1)[1].unsqueeze(1)).squeeze(1)
        else:
            q2_max = torch.max(q2, 1)[0]
        bellman_target = r + self.gamma * q2_max.detach() * (1 - t)
        
        errs = (bellman_target - q_pred).unsqueeze(1)
        quad = torch.min(torch.abs(errs), 1)[0]
        lin = torch.abs(errs) - quad
        loss = torch.sum(0.5 * quad.pow(2) + lin)
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_q(self, s):
        s = torch.FloatTensor(s).to(self.device).unsqueeze(0)
        return self.network(s / self.normalize).detach().cpu().numpy()

    def get_max_action(self, s):
        s = torch.FloatTensor(s).to(self.device).unsqueeze(0)
        q = self.network(s / self.normalize).detach()
        return q.max(1)[1].cpu().numpy()

    def get_action(self, states, evaluate):
        # get action WITH e-greedy exploration
        eps = self.epsilon if not evaluate else self.test_epsilon
        if self.rng.binomial(1, eps):
            return self.rng.randint(self.nb_actions)
        else:
            return self.get_max_action(states)[0]

    def learn(self):
        """ Learning from one minibatch """
        assert self.minibatch_size <= self.transitions.size, 'not enough data in the pool'
        s, a, r, s2, term = self.transitions.sample(self.minibatch_size)
        self.train_on_batch(s, a, r, s2, term)
        if self.update_counter == self.update_freq:
            self.weight_transfer(from_model=self.network, to_model=self.target_network)
            self.update_counter = 0
        else:
            self.update_counter += 1

    def anneal_eps(self, step):
        if self.epsilon > self.final_epsilon:
            decay = (self.start_epsilon - self.final_epsilon) * step / self.decay_steps
            self.epsilon = self.start_epsilon - decay
        if step >= self.decay_steps:
            self.epsilon = self.final_epsilon

    def dump_network(self, weights_file_path):
        torch.save(self.network.state_dict(), weights_file_path)

    def load_weights(self, weights_file_path, target=False):
        self.network.load_state_dict(torch.load(weights_file_path))
        if target:
            self.weight_transfer(from_model=self.network, to_model=self.target_network)

    @staticmethod
    def weight_transfer(from_model, to_model):
        to_model.load_state_dict(from_model.state_dict())

    def __getstate__(self):
        _dict = {k: v for k, v in self.__dict__.items()}
        del _dict['device']  # is not picklable
        del _dict['transitions']  # huge object (if you need the replay buffer, save it with np.save)
        return _dict
