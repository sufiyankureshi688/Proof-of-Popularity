import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from pop_rl_env import MultiAgentPopEnv

# create env
raw_env = MultiAgentPopEnv(other_agents=20, max_steps=100, seed=0)
env = DummyVecEnv([lambda: raw_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=1e9)

# train
model = DQN("MlpPolicy", env, verbose=1, learning_starts=1000)
model.learn(total_timesteps=30000)

model.save("pop_rl_dqn_competition_v1")
env.save("vec_normalize_stats.pkl")

# evaluate
num_eval_episodes = 6
episode_stats = {
    'weights': [], 'engagements': [], 'social_capitals': [], 'followers': [], 'friends': [], 'tokens': []
}

for ep in range(num_eval_episodes):
    eval_env = MultiAgentPopEnv(other_agents=20, max_steps=100, seed=ep+100)
    obs, _ = eval_env.reset()
    done = False
    w_trace = []; e_trace = []; s_trace = []; f_trace = []; fr_trace = []; t_trace = []
    while not done:
        vec_obs = np.expand_dims(obs, 0)
        action, _ = model.predict(vec_obs, deterministic=True)
        action = int(action[0])
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated

        wt = obs[0]
        engagement = obs[1] + obs[2] + obs[3] + obs[4]
        social_cap = engagement + obs[5] + obs[6]

        w_trace.append(wt)
        e_trace.append(engagement)
        s_trace.append(social_cap)
        f_trace.append(obs[5])
        fr_trace.append(obs[6])
        t_trace.append(eval_env.tokens_received)

    episode_stats['weights'].append(w_trace)
    episode_stats['engagements'].append(e_trace)
    episode_stats['social_capitals'].append(s_trace)
    episode_stats['followers'].append(f_trace)
    episode_stats['friends'].append(fr_trace)
    episode_stats['tokens'].append(t_trace)

# plot last eval
plt.figure(figsize=(16,8))
plt.subplot(2,3,1); plt.plot(episode_stats['weights'][-1]); plt.title("Weight")
plt.subplot(2,3,2); plt.plot(episode_stats['engagements'][-1]); plt.title("Engagement")
plt.subplot(2,3,3); plt.plot(episode_stats['social_capitals'][-1]); plt.title("Social Capital")
plt.subplot(2,3,4); plt.plot(episode_stats['followers'][-1]); plt.title("Followers")
plt.subplot(2,3,5); plt.plot(episode_stats['friends'][-1]); plt.title("Friends")
plt.subplot(2,3,6); plt.plot(episode_stats['tokens'][-1]); plt.title("Cumulative Tokens")

plt.tight_layout(); plt.show()

# quick evaluation on vec-normalized env
vec_for_eval = DummyVecEnv([lambda: MultiAgentPopEnv(other_agents=20, max_steps=100, seed=999)])
vec_for_eval = VecNormalize(vec_for_eval, norm_obs=True, norm_reward=True, clip_obs=1e9)
mean_reward, std_reward = evaluate_policy(model, vec_for_eval, n_eval_episodes=5)
print(f"Eval mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
