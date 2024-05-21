import gym
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.envs import DummyVecEnv

# 自定义特征提取器，使用CNN提取图像特征并通过LSTM处理
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # 计算最后一个卷积层输出的尺寸
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
        # 使用LSTM处理卷积特征
        self.lstm = nn.LSTM(n_flatten, features_dim)
        self.features_dim = features_dim

    def forward(self, observations):
        cnn_out = self.cnn(observations)
        cnn_out = cnn_out.view(-1, 1, cnn_out.size(1))  # 为LSTM调整形状
        lstm_out, _ = self.lstm(cnn_out)
        return lstm_out[:, -1, :]

# 自定义策略，使用自定义的特征提取器
class CustomLSTMPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomLSTMPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=256),
        )

    def forward(self, obs, lstm_states, episode_starts, deterministic=False):
        features = self.extract_features(obs)
        value = self.value_net(features)
        distribution = self._get_action_dist_from_latent(features)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, value, log_prob, lstm_states

    def _get_latent(self, obs, lstm_states, episode_starts):
        features = self.extract_features(obs)
        return features, lstm_states

    def _predict(self, obs, deterministic=False):
        features = self.extract_features(obs)
        distribution = self._get_action_dist_from_latent(features)
        return distribution.get_actions(deterministic=deterministic)