import numpy as np
import os
import ray 
import time 
from common.rollout import RolloutWorker
from agent.agent import Agents
from common.replay_buffer import ReplayBuffer
from tensorboardX import SummaryWriter



ray.init()



@ray.remote(num_cpus=2)
class RayRolloutWorker():
    def __init__(self, id, seed, args):
        self.port = 10000 + seed + 2 * id
        self.args = args
        self.rayrolloutWorker = RolloutWorker(self.port, self.args, no_graphics = True, time_scale = 100)

    def rollout(self, model):
        return self.rayrolloutWorker.generate_episode(model)
    
    def rollout_test(self, model):
        return self.rayrolloutWorker.generate_episode_test(model)
    
    def close(self):
        pass






class Runner:
    def __init__(self, args):

        self.args = args

        self.workers = None

        # 用来保存plt和pkl
        stamp = int(time.time())
        self.args.save_path = self.args.model_dir + '/' + str(args.seed) + '/' + (time.strftime("%m_%d_%H_%M_%S", time.localtime(stamp)))
        
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)


        self.log_path = self.args.log_dir + '/' + str(args.seed) + '/' + (time.strftime("%Y_%m_%d_%H_%M_%s", time.localtime(stamp)))
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.writer = SummaryWriter(self.log_path)

        self.is_cuda = True

        self.agents = Agents(args, self.is_cuda)
        
        self.buffer = ReplayBuffer(args)
        




    def run(self):

        self.workers = [RayRolloutWorker.remote(i, self.args.seed, self.args) for i in range(self.args.env_num)]

        ep_count= 0

        while ep_count < self.args.n_eps:

            print(" ep start : ", ep_count)
           
            if ep_count % self.args.evaluate_cycle == 0:

                rnn_para = self.agents.policy.get_para()

                model_id = ray.put(rnn_para)
            
                win_times = 0

                for _ in range(self.args.evaluate_epoch):

                    buffers_ids = [worker.rollout_test.remote(model_id) for worker in self.workers]

                    for batch in range(self.args.env_num):
                    
                        [buffers_id], buffers_ids = ray.wait(buffers_ids)
                        
                        if_win = ray.get(buffers_id)
                        
                        win_times += if_win

                win_rates = win_times / self.args.env_num / self.args.evaluate_epoch

                print("testing ...... ep : ", ep_count * self.args.env_num,  "  win_rate : ", win_rates)
            
                self.writer.add_scalar('test_win_rate', win_rates, global_step = ep_count * self.args.env_num)

                self.agents.policy.save_model(ep_count * self.args.env_num)
            
        


        
            rnn_para = self.agents.policy.get_para()

            model_id = ray.put(rnn_para)
        
            buffers_ids = [
                worker.rollout.remote(model_id) for worker in self.workers
            ]

            episodes = []

            for _ in range(self.args.env_num):
                [buffers_id], buffers_ids = ray.wait(buffers_ids)
                episode = ray.get(buffers_id)
                episodes.append(episode)
                
           
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

            # train qmix here
            self.buffer.store_episode(episode_batch)

            if ep_count > 50:
                need_train_step = int (self.buffer.current_size // self.args.batch_size / 5)  
                if need_train_step < 50:
                    need_train_step = 50 

                if (ep_count-1) % self.args.target_update_cycle == 0:
                    target_update = True
                else:
                    target_update = False

                for train_step in range(need_train_step):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_step, target_update)
                
            ep_count += 1


           





