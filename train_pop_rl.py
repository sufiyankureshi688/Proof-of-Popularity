import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from pop_rl_env import MultiAgentPopEnv

# create env
raw_env = MultiAgentPopEnv(other_agents=12, max_steps=100, seed=0)
# Wrap in DummyVecEnv and normalize observations/rewards for stability
env = DummyVecEnv([lambda: raw_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=1e6)

# create and train DQN
model = DQN("MlpPolicy", env, verbose=1, learning_starts=1000)
model.learn(total_timesteps=25000)

# save model and VecNormalize statistics
model.save("pop_rl_dqn_weight_v1")
env.save("vec_normalize_stats.pkl")

# evaluate and collect traces
num_eval_episodes = 6
episode_stats = {
    'weights': [],
    'engagements': [],
    'social_capitals': [],
    'followers': [],
    'friends': [],
    'tokens': []
}

# For evaluation use an un-normalized env copy
eval_env = MultiAgentPopEnv(other_agents=12, max_steps=100, seed=1)

for ep in range(num_eval_episodes):
    obs, _ = eval_env.reset()
    done = False
    weights = []
    engagements = []
    social_caps = []
    followers_list = []
    friends_list = []
    tokens_list = []
    while not done:
        # SB3 models trained on VecEnv expect vectorized obs; wrap accordingly
        vec_obs = np.expand_dims(obs, 0)
        action, _states = model.predict(vec_obs, deterministic=True)
        action = int(action[0])
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated

        wt = obs[0]
        likes = obs[1]
        shares = obs[2]
        comments = obs[3]
        saves = obs[4]
        followers = obs[5]
        friends = obs[6]

        engagement = likes + shares + comments + saves
        social_cap = engagement + followers + friends

        weights.append(wt)
        engagements.append(engagement)
        social_caps.append(social_cap)
        followers_list.append(followers)
        friends_list.append(friends)
        tokens_list.append(eval_env.tokens_received)

    episode_stats['weights'].append(weights)
    episode_stats['engagements'].append(engagements)
    episode_stats['social_capitals'].append(social_caps)
    episode_stats['followers'].append(followers_list)
    episode_stats['friends'].append(friends_list)
    episode_stats['tokens'].append(tokens_list)

# Plot the last evaluation episode as an example
plt.figure(figsize=(18,6))
plt.subplot(2,3,1)
plt.plot(episode_stats['weights'][-1])
plt.title("Agent Weight Over Eval Episode")
plt.xlabel("Step")
plt.ylabel("Weight")

plt.subplot(2,3,2)
plt.plot(episode_stats['engagements'][-1])
plt.title("Engagement Over Eval Episode")
plt.xlabel("Step")
plt.ylabel("Engagement")

plt.subplot(2,3,3)
plt.plot(episode_stats['social_capitals'][-1])
plt.title("Social Capital Over Eval Episode")
plt.xlabel("Step")
plt.ylabel("Social Capital")

plt.subplot(2,3,4)
plt.plot(episode_stats['followers'][-1])
plt.title("Followers Over Eval Episode")
plt.xlabel("Step")

plt.subplot(2,3,5)
plt.plot(episode_stats['friends'][-1])
plt.title("Friends Over Eval Episode")
plt.xlabel("Step")

plt.subplot(2,3,6)
plt.plot(episode_stats['tokens'][-1])
plt.title("Cumulative Tokens Received Over Eval Episode")
plt.xlabel("Step")

plt.tight_layout()
plt.show()

# Optionally evaluate average reward (wrap eval_env similarly to training VecEnv if needed)
print('Evaluating policy on VecNormalize-wrapped env (approx):')
vec_for_eval = DummyVecEnv([lambda: MultiAgentPopEnv(other_agents=12, max_steps=100, seed=2)])
vec_for_eval = VecNormalize(vec_for_eval, norm_obs=True, norm_reward=True, clip_obs=1e6)
mean_reward, std_reward = evaluate_policy(model, vec_for_eval, n_eval_episodes=5)
print(f"Eval mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
