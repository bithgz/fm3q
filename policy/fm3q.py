import torch
import os
from network.base_net import RNN
import time
import numpy as np
from collections import OrderedDict

class FM3Q:

    def __init__(self, args, is_cuda):

        self.n_actions = args.n_actions
        self.n_agents_per_party = args.n_agents_per_party 
        self.obs_size = args.obs_shape
        self.actions_ind = np.arange(self.n_actions)
        

        # 神经网络
        self.eval_rnn = RNN(self.obs_size, args)  # 每个agent选动作的网络
        self.target_rnn = RNN(self.obs_size, args)
    
        self.args = args
        self.is_cuda = is_cuda

        if self.is_cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
   
        self.model_dir = self.args.save_path
        

        self.eval_rnn.load_state_dict(self.target_rnn.state_dict())

        self.eval_parameters = list(self.eval_rnn.parameters()) 

        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)
        else:
            self.optimizer = torch.optim.Adam(self.eval_parameters, lr=args.lr)
        
        print('Init alg FM3Q')



    def get_para(self):
        device = torch.device('cpu')

        state_dict = self.eval_rnn.state_dict()

        new_state_dict = OrderedDict()

        for k, v in state_dict.items():

            name = k # remove `module.`

            new_state_dict[name] = v.to(device)
        # model.load_state_dict(new_state_dict)
        return new_state_dict




    def load_para(self, para):
        self.eval_rnn.load_state_dict(para)



    def train_q(self, batch, end_train_step, target_update):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        '''
        在learn的时候，抽取到的数据是四维的，四个维度分别为 1——第几个episode 2——episode中第几个transition
        3——第几个agent的数据 4——具体obs维度。因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，然后一次给神经网络
        传入每个episode的同一个位置的transition
        '''
        # transition_num = batch['o'].shape[0]

        # self.init_hidden(episode_num)


        for key in batch.keys():  # 把batch里的数据转化成tensor

            if self.is_cuda:
                if key == 'u':
                    batch[key] = torch.tensor(batch[key], dtype=torch.long).cuda()
                else:
                    batch[key] = torch.tensor(batch[key], dtype=torch.float32).cuda()
            else:
                if key == 'u':
                    batch[key] = torch.tensor(batch[key], dtype=torch.long)
                else:
                    batch[key] = torch.tensor(batch[key], dtype=torch.float32)


        u, r, terminated = batch['u'], batch['r'], batch['terminated']
                                                             

        # 得到每个agent对应的Q值，维度为(transition_num个数, n_agents， n_actions)
        q_evals, q_targets = self.get_q_values(batch)
        
        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        q_evals = torch.gather(q_evals, dim=2, index=u).squeeze(2)
        q_evals_p = torch.split(q_evals, self.n_agents_per_party, 1)[0]
        q_evals_n = torch.neg(torch.split(q_evals, self.n_agents_per_party, 1)[1])

        # 得到target_q
        q_targets = q_targets.max(dim=2)[0]
        q_targets_p = torch.split(q_targets, self.n_agents_per_party, 1)[0]
        q_targets_n = torch.neg(torch.split(q_targets, self.n_agents_per_party, 1)[1])


        q_total_eval = q_evals_p + q_evals_n

        q_total_target = q_targets_p + q_targets_n

        targets = r + self.args.gamma * q_total_target * (1 - terminated)

        td_error = q_total_eval - targets.detach()

    
        loss = (td_error ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if end_train_step and target_update:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
           



    def _get_inputs(self, batch):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next = batch['o'], batch['o_next']
        transition_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)

        inputs = torch.cat([x.reshape(transition_num * 2 * self.args.n_agents_per_party, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(transition_num * 2 * self.args.n_agents_per_party, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next


    def get_q_values(self, batch):

        transition_num = batch['o'].shape[0]

        inputs, inputs_next = self._get_inputs(batch)  # 给obs加last_action、agent_id

        q_eval = self.eval_rnn(inputs)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
        q_target = self.target_rnn(inputs_next)

        # 把q_eval维度重新变回(8, 5,n_actions)
        q_eval = q_eval.view(transition_num, 2 * self.n_agents_per_party, -1)
        q_target = q_target.view(transition_num, 2 * self.n_agents_per_party, -1)

        return q_eval, q_target





    def save_model(self, eps):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + str(eps) + '_q_net_params.pkl')

    def load_model(self, eps):
        if os.path.exists(self.model_dir + '/' + str(eps) + '_q_params.pkl'):
            path_actor = self.model_dir + '/' + str(eps) + '_q_params.pkl'
            map_location = 'cuda:'+ self.args.CUDA_VISIBLE_DEVICES
            self.eval_rnn.load_state_dict(torch.load(path_actor, map_location=map_location))
            print('Successfully load the model: {}'.format(path_actor))
        else:
            raise Exception("No model!")
        
        
    def get_action(self, state, epsilon):
        if self.is_cuda:
            state = torch.from_numpy(state).float().cuda()
        else:
            state = torch.from_numpy(state).float()
        with torch.no_grad():
            q_value = self.eval_rnn(state).cpu().numpy()

        actions = []
        for i in range(q_value.shape[0]):
            if np.random.uniform() < epsilon:
                actions.append(np.random.choice(self.actions_ind))  
            else:
                actions.append(np.argmax(q_value[i,:]))

        return actions
