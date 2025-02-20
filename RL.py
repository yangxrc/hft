import torch
import torch.nn as nn
import random
import numpy as np


seed = 1
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)  # torch.cuda.manual_seed_all()用于设置CUDA的随机数生成器的种子（如果可用）
todevice = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = 'cuda:1'


class Model(nn.Module):
    def __init__(self, n_feature, n_action, n_l1, n_l2, n_l3):
        super(Model, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(n_feature, n_l1),
            nn.LeakyReLU(),
            nn.Linear(n_l1, n_l2),
            nn.LeakyReLU(),
            nn.Linear(n_l2, n_l3),
            nn.LeakyReLU(),
            nn.Linear(n_l3, n_action),
        )

        self.critic = nn.Sequential(
            nn.Linear(n_feature, n_l1),
            nn.LeakyReLU(),
            nn.Linear(n_l1, n_l2),
            nn.LeakyReLU(),
            nn.Linear(n_l2, n_l3),
            nn.LeakyReLU(),
            nn.Linear(n_l3, 1),
        )

    def forward(self, x):
        action = self.actor(x)
        pi_prob = nn.functional.softmax(action, dim=-1)
        v = self.critic(x)
        return pi_prob[0], v[0]


class DeepQNetwork:
    def __init__(
            self,
            n_actions,  # 动作的个数
            n_features,  # 状态的特征数，状态涉及到几个特征，n_features就是几
            learning_rate,
    ):

        self.n_actions = n_actions  # 动作的个数
        self.n_features = n_features  # 状态的特征数，状态涉及到几个特征，n_features就是几
        self.lr_a = learning_rate
        self.lr_c = 2 * self.lr_a

        self.cycle_size = 100  # 学习率周期上升下降的周期长度
        self.max_lr = learning_rate  # 学习率的上界
        self.base_lr = 0.000001  # 学习率的下界

        self.policy = Model(self.n_features, self.n_actions, n_l1=256, n_l2=256, n_l3=128).to(todevice)
        self.load_net()

        self.mse_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.lr_a},
            {'params': self.policy.critic.parameters(), 'lr': self.lr_c}
        ])

    # 将环境状态作为网络的输入，然后根据网络的输出选择动作
    def choose_action(self, observation):
        with torch.no_grad():  # 不参与梯度下降
            prob_weights, _ = self.policy(torch.tensor(observation[np.newaxis, :], dtype=torch.float32, device=device))
            a = torch.argmax(prob_weights)
        return a.item()

    # 加载网络的参数
    def load_net(self):
        self.policy.load_state_dict(torch.load('policy.pth', map_location=device))  # 加载模型参数
        self.policy.eval().to(todevice)  # 使用加载的参数更新模型的参数

