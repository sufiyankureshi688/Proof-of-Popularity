import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from pop_rl_env import MultiAgentPopEnv

# Load the trained model
print("Loading trained model...")
raw_env = MultiAgentPopEnv(other_agents=20, max_steps=100, seed=0)
env = DummyVecEnv([lambda: raw_env])
env = VecNormalize.load("vec_normalize_stats.pkl", env)
env.training = False
env.norm_reward = False

model = DQN.load("pop_rl_dqn_competition_v1", env=env)

# Choose a random seed for this evaluation run
eval_seed = np.random.randint(0, 10000)
print(f"Evaluating on seed: {eval_seed}")

# Create evaluation environment (NOT wrapped in VecNormalize)
eval_env = MultiAgentPopEnv(other_agents=20, max_steps=100, seed=eval_seed)
obs, _ = eval_env.reset()

# Track metrics
w_trace = []
e_trace = []
sc_trace = []
fol_trace = []
fr_trace = []
tok_trace = []

# Run one episode
done = False
while not done:
    # Model expects vectorized input
    vec_obs = np.expand_dims(obs, 0)
    action, _ = model.predict(vec_obs, deterministic=True)
    action = int(action[0])
    
    obs, reward, terminated, truncated, info = eval_env.step(action)
    done = terminated or truncated
    
    # Extract metrics
    weight = obs[0]
    likes = obs[1]
    shares = obs[2]
    comments = obs[3]
    saves = obs[4]
    followers = obs[5]
    friends = obs[6]
    
    engagement = likes + shares + comments + saves
    social_capital = engagement + followers + friends
    
    w_trace.append(weight)
    e_trace.append(engagement)
    sc_trace.append(social_capital)
    fol_trace.append(followers)
    fr_trace.append(friends)
    tok_trace.append(eval_env.tokens_received)  # FIXED: Access directly from environment

# Plot results
fig = plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.plot(w_trace)
plt.title('Weight')
plt.xlabel('Step')
plt.ylabel('Weight')

plt.subplot(2, 3, 2)
plt.plot(e_trace)
plt.title('Engagement')
plt.xlabel('Step')
plt.ylabel('Engagement')

plt.subplot(2, 3, 3)
plt.plot(sc_trace)
plt.title('Social Capital')
plt.xlabel('Step')
plt.ylabel('Social Capital')

plt.subplot(2, 3, 4)
plt.plot(fol_trace)
plt.title('Followers')
plt.xlabel('Step')
plt.ylabel('Followers')

plt.subplot(2, 3, 5)
plt.plot(fr_trace)
plt.title('Friends')
plt.xlabel('Step')
plt.ylabel('Friends')

plt.subplot(2, 3, 6)
plt.plot(tok_trace)
plt.title('Cumulative Tokens')
plt.xlabel('Step')
plt.ylabel('Tokens')

plt.tight_layout()
plt.savefig(f'evaluation_seed_{eval_seed}.png')
print(f"Plot saved as evaluation_seed_{eval_seed}.png")
plt.show()
