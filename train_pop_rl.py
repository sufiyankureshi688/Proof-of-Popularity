# train_pop_rl.py
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from pop_rl_env import MultiAgentPopEnv

# create env
env = MultiAgentPopEnv(other_agents=10, max_steps=100, seed=0)

# create and train DQN
model = DQN("MlpPolicy", env, verbose=1, learning_starts=1000)
model.learn(total_timesteps=20000)

# save model
model.save("pop_rl_dqn_v2")

# evaluate and collect traces
num_eval_episodes = 10
episode_stats = {
    'reputations': [],
    'engagements': [],
    'social_capitals': [],
    'followers': []
}

for ep in range(num_eval_episodes):
    obs, _ = env.reset()
    done = False
    reputations = []
    engagements = []
    social_caps = []
    followers_list = []
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        rep = obs[0]
        likes = obs[1]
        shares = obs[2]
        comments = obs[3]
        saves = obs[4]
        followers = obs[5]

        engagement = likes + shares + comments + saves
        social_cap = engagement + followers

        reputations.append(rep)
        engagements.append(engagement)
        social_caps.append(social_cap)
        followers_list.append(followers)

    episode_stats['reputations'].append(reputations)
    episode_stats['engagements'].append(engagements)
    episode_stats['social_capitals'].append(social_caps)
    episode_stats['followers'].append(followers_list)

# Plot the last evaluation episode as an example
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.plot(episode_stats['reputations'][-1])
plt.title("Agent Reputation Over Eval Episode")
plt.xlabel("Step")
plt.ylabel("Reputation")

plt.subplot(1,3,2)
plt.plot(episode_stats['engagements'][-1])
plt.title("Engagement Over Eval Episode")
plt.xlabel("Step")
plt.ylabel("Engagement")

plt.subplot(1,3,3)
plt.plot(episode_stats['social_capitals'][-1])
plt.title("Social Capital Over Eval Episode")
plt.xlabel("Step")
plt.ylabel("Social Capital")

plt.tight_layout()
plt.show()

# Optionally evaluate average reward
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
print(f"Eval mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
