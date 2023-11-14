import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from IHiterEnv.parameter import *

np.random.seed(1)
torch.manual_seed(1)
class DuelingDQN(nn.Module):
    def __init__(self, 
                ActionDim=TP.ActionDim, 
                StateDim=TP.StateDim, 
                LearningRate=0.01, 
                RewardDecay=0.9, 
                eGreedy=0.1, 
                ReplaceTargetIter=1000, 
                MemorySize=10000, 
                BatchSize=64, 
                eGreedyDecay=0.9, 
                train_dir=''):
        super(DuelingDQN, self).__init__()
        self.ActionDim = ActionDim
        self.StateDim = StateDim
        self.Gamma = RewardDecay
        self.LearningRate = LearningRate
        self.EpsilonMin = eGreedy
        self.ReplaceTargetIter = ReplaceTargetIter
        self.MemorySize = MemorySize
        self.BatchSize = BatchSize
        self.eGreedyDecay = eGreedyDecay
        self.Epsilon = 0.98
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device=torch.device("cuda")
        # 文件保存部分
        self.train_dir = train_dir
        # PyTorch 不需要分离 CheckpointsFile 和 Summary 文件夹

        # total learning step
        self.LearnStepCounter = 0

        # initialize zero memory [state, action, reward, next_state]
        self.memory = np.zeros((self.MemorySize, StateDim * 2 + ActionDim + 1))

        # self.eval_net, self.target_net = self._build_net(), self._build_net()
        self.eval_net, self.target_net = self._build_net().to(self.device), self._build_net().to(self.device)

        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=self.LearningRate)
        self.loss_func = nn.MSELoss()

    def _build_net(self):
        class Net(nn.Module):
            def __init__(self, state_dim, action_dim):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(state_dim, 50)
                self.fc2 = nn.Linear(50, 50)
                # Dueling DQN 部分
                self.value = nn.Linear(50, 1)
                self.advantage = nn.Linear(50, action_dim)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                value = self.value(x)
                advantage = self.advantage(x)
                return value + advantage - advantage.mean()

        return Net(self.StateDim, self.ActionDim)

    def StoreTransition(self, state, action, reward, next_state):
        '''
            存储记忆库
            
            @ state: [4x8+6,]
            @ action: [17x4,]
            @ reward: [2,]
        '''
        if not hasattr(self, 'MemoryCounter'):
            self.MemoryCounter = 0
        transition = np.hstack((state, action, reward, next_state))
        # replace the old memory with new memory
        index = self.MemoryCounter % self.MemorySize

        self.memory[index, :] = np.hstack((state, action, reward, next_state))
        self.MemoryCounter += 1


    def Train_decision(self, state):
        '''
            agent在训练的时候做出action，会有一部分几率进行随机探索

            @ state：[38,]

            return：直接将网络的原始action数据进行输出[2x34,]
        '''
        if np.random.uniform() < self.Epsilon:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # 将 NumPy 数组转换为 PyTorch 张量，并添加一个维度
            # forward feed the observation and get q value for every actions
            action_raw = self.eval_net(state).detach().cpu().numpy().reshape(-1)  # 使用 PyTorch 模型进行前向传播，并转换回 NumPy 数组
        else:
            action_raw = np.random.randn(self.ActionDim)
        return action_raw

    def Eval_decision(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_raw = self.eval_net(state).detach().cpu().numpy()  # 从 PyTorch 模型获取动作值
        return action_raw


    def Learn(self):
        # if self.LearnStepCounter % self.ReplaceTargetIter == 0:
        #     self.target_net.load_state_dict(self.eval_net.state_dict())
        #     # print('\ntarget_params_replaced\n')

        # if self.MemoryCounter > self.MemorySize:
        #     sample_index = np.random.choice(self.MemorySize, size=self.BatchSize)
        # else:
        #     sample_index = np.random.choice(self.MemoryCounter, size=self.BatchSize)
        # batch_memory = self.memory[sample_index, :]

        # b_s = torch.FloatTensor(batch_memory[:, :self.StateDim])
        # b_a = torch.LongTensor(batch_memory[:, self.StateDim:self.StateDim + self.ActionDim])
        # b_r = torch.FloatTensor(batch_memory[:, self.StateDim + self.ActionDim:self.StateDim + self.ActionDim + 1])
        # b_s_ = torch.FloatTensor(batch_memory[:, -self.StateDim:])

        # # Q Learning 的核心部分
        # print("b_a shape:", b_a.shape)
        # print("b_a:", b_a)
        # print("q_eval shape:", self.eval_net(b_s).shape)
        
        # q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        # q_next = self.target_net(b_s_).detach()     # 分离出来，不参与梯度更新
        # q_target = b_r + self.Gamma * q_next.max(1)[0].view(self.BatchSize, 1)  # shape (batch, 1)

        # loss = self.loss_func(q_eval, q_target)

        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # # 更新 epsilon
        # self.Epsilon = max(self.Epsilon - self.eGreedyDecay, self.EpsilonMin)

        # self.LearnStepCounter += 1
        if self.LearnStepCounter % self.ReplaceTargetIter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        # 从所有记忆中抽取批次记忆
        if self.MemoryCounter > self.MemorySize:
            sample_index = np.random.choice(self.MemorySize, size=self.BatchSize)
        else:
            sample_index = np.random.choice(self.MemoryCounter, size=self.BatchSize)
        batch_memory = self.memory[sample_index, :]

        # 分离出状态、奖励和下一个状态
        b_s = torch.FloatTensor(batch_memory[:, :self.StateDim]).to(self.device)
        b_r = torch.FloatTensor(batch_memory[:, self.StateDim:self.StateDim+1]).to(self.device)
        b_s_ = torch.FloatTensor(batch_memory[:, -self.StateDim:]).to(self.device)

        # 第一方的训练
        self.optimizer.zero_grad()
        q_eval = self.eval_net(b_s)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.Gamma * q_next.max(1)[0].view(self.BatchSize, 1)

        loss = self.loss_func(q_eval, q_target)
        loss.backward()
        self.optimizer.step()

        # 更新 epsilon
        self.Epsilon = max(self.Epsilon - self.eGreedyDecay, self.EpsilonMin)

        if self.LearnStepCounter % 10 == 0:
            if self.LearnStepCounter % 10000 == 0:
                print('Train Times : ', self.LearnStepCounter, ' cost : ', loss.item())

        self.LearnStepCounter += 1
        
    def Save(self, step):
        torch.save(self.eval_net.state_dict(), f'{self.train_dir}/model_{step}.pth')


if __name__ == '__main__':
    DQN = DuelingDQN()