class Agents:
    def __init__(self, args, is_cuda):

        from policy.fm3q import FM3Q
        self.policy = FM3Q(args, is_cuda)
        self.args = args

    
    def get_action(self, obs, epsilon):
        return self.policy.get_action(obs, epsilon)



    def train(self, batch, train_step, target_update): 
        self.policy.train_q(batch, train_step, target_update)