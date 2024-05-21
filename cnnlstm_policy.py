import torch
import torch.nn as nn
import gym
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

# 定义一个 ConvLSTM 单元
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=kernel_size // 2, bias=bias)

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

    def forward(self, x, hidden=None):
        b, seq_len, c, h, w = x.size()
        if hidden is None:
            hidden = [(torch.zeros(b, self.layers[0].hidden_dim, h, w, device=x.device), torch.zeros(b, self.layers[0].hidden_dim, h, w, device=x.device)) for _ in self.layers]
        new_hidden = []
        for i, layer in enumerate(self.layers):
            h, c = hidden[i]
            output_inner = []
            for t in range(seq_len):
                h, c = layer(x[:, t], (h, c))
                output_inner.append(h)
            x = torch.stack(output_inner, dim=1)
            new_hidden.append((h, c))
        return x, new_hidden

# 特征提取器
class ConvLSTMFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(ConvLSTMFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.conv_lstm = ConvLSTM(input_dim=3, hidden_dim=64, kernel_size=3, num_layers=1)
        self.fc = nn.Linear(64 * observation_space.shape[1] * observation_space.shape[2], features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.permute(0, 4, 1, 2, 3)  # Reorder dimensions to match ConvLSTM input
        x, _ = self.conv_lstm(observations)
        x = x[:, -1].reshape(x.size(0), -1)  # Flatten the last output for each sample in the batch
        return self.fc(x)

# 使用该特征提取器创建 PPO 策略
class CnnLSTMPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, features_extractor_class=ConvLSTMFeaturesExtractor, **kwargs)
