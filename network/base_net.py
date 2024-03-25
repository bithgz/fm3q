import torch
import torch.nn as nn
import torch.nn.functional as f



class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args
        self.seed = torch.cuda.manual_seed(self.args.seed)
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs):
        x = f.relu(self.fc1(obs))
        h = f.relu(self.fc2(x))
        q = self.fc3(h)
        return q


