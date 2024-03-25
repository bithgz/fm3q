import argparse

"""
Here are the param for the training

"""


def get_common_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--env_num', type=int, default=16, help='16 number of the workers')

    parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')
    
    parser.add_argument('--alg', type=str, default='fm3q', help='the algorithm to train the agent')

    # parser.add_argument('--n_steps', type=int, default=4e6, help='total time steps')
    parser.add_argument('--n_eps', type=int, default=5001, help='5000 total eps')
    
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--alpha', type=float, default=0.05, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')

    parser.add_argument('--evaluate_cycle', type=int, default=100, help='100 how often to evaluate the model, every 100 ep')
    parser.add_argument('--evaluate_epoch', type=int, default=10, help='10 number of the epoch to evaluate the agent, test 100 ep')


    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--log_dir', type=str, default='./log', help='log directory of the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')

    
    parser.add_argument('--cuda', type=bool, default=False, help='whether to use the GPU')
    parser.add_argument('--test', type=bool, default=False, help='whether to use the GPU')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='1', help='log directory of the policy')

    args = parser.parse_args()
    return args




def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 128


    args.hyper_hidden_dim = 32              # betwwen hyper layer and inputs
    args.qmix_hidden_dim = 32               # betwwen two hyper layers 

    args.two_hyper_layers = False

    args.lr = 5e-4

    # epsilon greedy
    args.epsilon = 0.2
    args.min_epsilon = 0.1
    anneal_steps = int(5e2)

    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'episode'

    # the number of the train steps in one epoch
    args.train_steps = 30

    # experience replay
    args.batch_size = 2000
    args.buffer_size = int(4e6)

    # how often to save the model
    args.save_cycle = 50

    # how often to update the target_net
    args.target_update_cycle = 5

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args



