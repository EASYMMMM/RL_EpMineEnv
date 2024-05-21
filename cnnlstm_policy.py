import torch
import torch.nn as nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
from stable_baselines3.common.policies import ActorCriticCnnPolicy

# 定义一个 ConvLSTM 单元
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding, bias=bias)

    def forward(self, x, states):
        h_cur, c_cur = states
        combined = torch.cat([x, h_cur], dim=1)  # Concatenate input and previous hidden state
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

# 构建一个 ConvLSTM 网络
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, bias=True):
        super(ConvLSTM, self).__init__()
        self.layers = nn.ModuleList([ConvLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size, bias) for i in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        b, c, h, w = x.size()
        if hidden is None:
            hidden = [(torch.zeros(b, self.layers[i].hidden_dim, h, w, device=x.device),
                       torch.zeros(b, self.layers[i].hidden_dim, h, w, device=x.device)) for i in range(self.num_layers)]
        new_hidden = []
        for i, layer in enumerate(self.layers):
            h, c = hidden[i]
            h, c = layer(x, (h, c))
            new_hidden.append((h, c))
            x = h  # 更新输入为当前层的输出
        return x, new_hidden

class ConvLSTMFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(ConvLSTMFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.conv_lstm = ConvLSTM(input_dim=3, hidden_dim=64, kernel_size=3, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64 * observation_space.shape[1] * observation_space.shape[2], features_dim)
        self.hidden = None  # 存储 LSTM 的隐藏状态

    def reset_hidden(self, batch_size=1, height=128, width=128, device=torch.device("cpu")):
        # 在每次情景开始时调用此函数来重置 LSTM 状态
        self.hidden = [(torch.zeros(batch_size, self.conv_lstm.layers[0].hidden_dim, height, width, device=device),
                        torch.zeros(batch_size, self.conv_lstm.layers[0].hidden_dim, height, width, device=device))
                       for _ in range(self.conv_lstm.num_layers)]

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        b, c, h, w = observations.size()
        if self.hidden is None:
            # 如果 LSTM 状态未初始化，则重置它
            self.reset_hidden(batch_size=b, height=h, width=w, device=observations.device)

        x, self.hidden = self.conv_lstm(observations, self.hidden)
        x = x.reshape(x.size(0), -1)  # Flatten the output for each sample in the batch
        return self.fc(x)

# 定义自定义 CNN Policy
class CnnLSTMPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super(CnnLSTMPolicy, self).__init__(*args, **kwargs,
                                              features_extractor_class=ConvLSTMFeaturesExtractor,
                                              features_extractor_kwargs={'features_dim': 512})
    def reset_lstm_hidden(self, batch_size, height, width, device):
        # 调用特征提取器的 reset_hidden 方法
        self.features_extractor.reset_hidden(batch_size, height, width, device)


# 定义自定义环境包装器
class LSTMResetWrapper(gym.Wrapper):
    def __init__(self, env, feature_extractor):
        super(LSTMResetWrapper, self).__init__(env)
        self.feature_extractor = feature_extractor

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # 获取观测的形状
        height, width = obs.shape[-2], obs.shape[-1]
        # 重置 LSTM 隐藏状态
        self.feature_extractor.reset_hidden(batch_size=1, height=height, width=width, device=obs.device)
        return obs

    def step(self, action):
        return self.env.step(action)
    
# 示例用法
observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 128, 128), dtype=np.uint8)
extractor = ConvLSTMFeaturesExtractor(observation_space)
observations = torch.randn(8, 3, 128, 128)  # 例如一个 batch 中有 8 个样本，每个样本有 3 个通道，大小为 128x128

# 重置隐藏状态
extractor.reset_hidden(batch_size=8, height=128, width=128, device=torch.device("cpu"))

# 前向传播
features = extractor(observations)
features = extractor(observations)
print(features.shape)  # 输出形状
