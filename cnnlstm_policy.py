import torch
import torch.nn as nn
import gym
from typing import Any, Dict, List, Optional, Type, Union


from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

# LSTM元结构
class ConvLSTMCell(nn.Module):
    def __init__(self,
                 input_dim, # 输入通道数
                 hidden_dim, #
                 kernel_size,
                 bias):
        # 父类初始化
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        # LSTM中，当前输入与上一时间步的隐藏状态(h_cur)会拼接在一起，因此通道数为input_dim + hidden_dim
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              # i f o g
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=kernel_size[0] // 2,
                              bias=bias)

    def forward(self, x, h_cur, c_cur):
        # 拼接当前输入和隐藏量
        combined = torch.cat([x, h_cur], dim=1)
        # 对其进行卷积
        combined_conv = self.conv(combined)
        # 对卷积层的输出进行拆分
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g) # 新信息
        # 下一个时间步的cell状态
        c_next = f * c_cur + i * g
        # 下一个时间步的隐藏状态
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 kernel_size,
                 num_layers,
                 batch_first=False,
                 bias=True,
                 return_all_layers=False):
        super(ConvLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers

        # 创建ConvLSTM单元列表
        cell_list = []
        for i in range(0, num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=hidden_dim,
                                          kernel_size=kernel_size,
                                          bias=bias))
        self.cell_list = nn.ModuleList(cell_list)

    # 前向传播
    def forward(self, x, hidden_state=None):
        # 调整输入张亮的维度
        if not self.batch_first:
            x = x.permute(1, 0, 2, 3, 4)
        # 初始化隐藏状态
        b, _, _, h, w = x.size() # 批量大小b，图像的高度h，宽度w
        hidden_state = self.init_hidden(batch_size=b, image_size=(h, w))
        # 初始化输出列表
        layer_output_list = []
        last_state_list = []

        # 处理每一层的输入
        cur_layer_input = x
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(x.size(1)):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], h, c)
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

# 特征提取器
class ConvLSTMFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: gym.spaces.Box,
                 features_dim: int = 512):
        super(ConvLSTMFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.conv_lstm = ConvLSTM(input_dim=3,
                                  hidden_dim=64,
                                  kernel_size=(3, 3),
                                  num_layers=1,
                                  batch_first=True)
        self.fc = nn.Linear(64, features_dim)

    # obs:(batch_size, seq_len, C, H, W)
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, C, H, W = observations.size()
        x, _ = self.conv_lstm(observations)
        # 从x中取出最后一个时间步的输出动作作为特征本表示
        x = x[-1][:, -1, :, :, :].view(batch_size, -1)
        x = self.fc(x)
        return x



# 定义策略
class CustomLSTMPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = ConvLSTMFeaturesExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(CustomLSTMPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


