"""FinRL PPO model using ``stable-baselines3``."""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym


class _ArrayEnv(gym.Env):
    """Simple environment reading from arrays for demonstration."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        super().__init__()
        self.X = X
        self.y = y
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(X.shape[1],))
        self.step_idx = 0

    def reset(self, *, seed=None, options=None):  # type: ignore[override]
        self.step_idx = 0
        return self.X[self.step_idx], {}

    def step(self, action):  # type: ignore[override]
        reward = -abs(float(action) - self.y[self.step_idx])
        self.step_idx += 1
        done = self.step_idx >= len(self.X)
        obs = self.X[self.step_idx] if not done else self.X[-1]
        return obs, reward, done, False, {}


def train(X: np.ndarray, y: np.ndarray, params: dict | None = None) -> dict:
    if params is None:
        params = {}

    lr = float(params.get("lr", 1e-3))
    n_steps = int(params.get("n_steps", 128))
    env = DummyVecEnv([lambda: _ArrayEnv(X, y)])
    model = PPO("MlpPolicy", env, learning_rate=lr, n_steps=n_steps, verbose=0)
    model.learn(total_timesteps=n_steps)
    return {"model": model, "env": env}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    env = model["env"]
    policy: PPO = model["model"]
    obs = env.reset()
    preds = []
    for i in range(len(X)):
        action, _ = policy.predict(obs, deterministic=True)
        preds.append(action[0])
        obs, _, done, _, _ = env.step(action)
        if done:
            break
    return np.array(preds)


