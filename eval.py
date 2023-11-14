from IHiterEnv.env import ICRA_Env
from DuelingDQN import DuelingDQN  # 假设 DuelingDQN 已经被改写为 PyTorch 版本
import time
from IHiterEnv.policy import *
from IHiterEnv.agent import *
import torch

MaxEpisode = 2000
MaxEpisodeSteps = 500

train_file = './train_data/'

RLBrain = DuelingDQN(train_dir=train_file)
env = ICRA_Env()
Random = RandomPolicy()
team_action = TeamAction()

def run_n_episode(n):
    checkpoint_file = f'{RLBrain.train_dir}/final_model.pth'
    print('Loading model from:', checkpoint_file)
    RLBrain.eval_net.load_state_dict(torch.load(checkpoint_file))
    RLBrain.eval_net.eval()  # 设置为评估模式

    blue_win_times, red_win_times = 0, 0
    for episode in range(1, n + 1):
        Episode_steps, total_reward = 0, 0
        obs = env.reset()
        while True:
            env.render()
            Blue_Action = RLBrain.Eval_decision(obs)  # 注意这里使用 PyTorch 版本的评估决策函数
            next_state, StepReward, isGameOver, _ = env.step(Blue_Action)
            obs = next_state
            Episode_steps += 1
            total_reward += StepReward
            if isGameOver:
                break
        print(' ')
        print('Episode : ', episode, 'winner : ', env.Winner)
        print('Episode steps : ', Episode_steps, "average reward", 
            total_reward/Episode_steps)
        if env.Winner == 'Blue':
            blue_win_times += 1
        elif env.Winner == 'Red':
            red_win_times += 1
    print('\nBlue win rate : ', blue_win_times/n, 'Red win rate : ', 
        red_win_times/n, '%')

if __name__ == "__main__":
    run_n_episode(10)
