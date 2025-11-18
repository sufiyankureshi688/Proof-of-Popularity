# train.py
# DQN training for the SocialMediaEnv using PyTorch.
# Requirements: torch

import random
import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
from environment import SocialMediaEnv, Action, INTERACT_SUBACTIONS

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# discrete actions mapping:
# 0 = idle
# 1 = post
# 2..6 = interact like/comment/share/save/follow (in that order)
ACTION_SIZE = 7

def select_target_post_for_interact(env, agent_id):
    """
    Heuristic target selection used by the agent when it chooses an 'interact' action.
    Choose a recent post whose author is most likely to follow back / engage.
    Strategy: prefer recent posts of authors with high weight but who do NOT already follow agent.
    """
    if not env.posts:
        return None
    recent = env.posts[-50:]
    scores = []
    for p in recent:
        author = env.agents[p.author]
        # prefer authors with high weight but who don't yet follow agent
        not_following = 2.0 if (agent_id not in author.following) else 0.7
        score = (author.weight ** 0.5) * not_following + p.total_engagement_score() * 0.1
        scores.append(max(0.001, score))
    return random.choices(recent, weights=scores, k=1)[0]

def state_from_env(env, n_recent=10):
    vec = env.get_obs_vector(None, n_recent=n_recent)
    return np.array(vec, dtype=np.float32)

def train_dqn(episodes=200, steps_per_episode=100, batch_size=64, gamma=0.99,
              lr=1e-3, buffer_capacity=200000, min_buffer=500, target_update=10):
    env = SocialMediaEnv()
    obs0 = env.reset()
    n_recent = 10
    state_dim = 3 + n_recent
    device = torch.device("cpu")
    policy_net = DQN(state_dim, ACTION_SIZE).to(device)
    target_net = DQN(state_dim, ACTION_SIZE).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay = ReplayBuffer(capacity=buffer_capacity)

    epsilon_start = 1.0
    epsilon_final = 0.05
    epsilon_decay = 0.995

    epsilon = epsilon_start
    steps_done = 0

    for ep in range(1, episodes + 1):
        env.reset()
        state = state_from_env(env, n_recent=n_recent)
        total_reward_ep = 0.0
        for t in range(steps_per_episode):
            steps_done += 1
            # epsilon-greedy over discrete actions
            if random.random() < epsilon:
                a = random.randrange(ACTION_SIZE)
            else:
                with torch.no_grad():
                    s_t = torch.from_numpy(state).unsqueeze(0).to(device)
                    qvals = policy_net(s_t).cpu().numpy()[0]
                    a = int(np.argmax(qvals))

            # translate discrete action a into env Action
            if a == 0:
                env_action = Action("idle", None, None)
            elif a == 1:
                env_action = Action("post", None, None)
            else:
                sub_idx = a - 2
                sub = INTERACT_SUBACTIONS[sub_idx]
                # choose a target post (heuristic)
                target = select_target_post_for_interact(env, env.controlled_id)
                if target is None:
                    env_action = Action("idle", None, None)
                else:
                    env_action = Action("interact", target, sub)

            _, reward, _, info = env.step({env.controlled_id: env_action})
            total_reward_ep += reward
            next_state = state_from_env(env, n_recent=n_recent)
            done = False  # environment is continuing; we use episodic wrapper externally
            replay.push(state, a, reward, next_state, done)
            state = next_state

            # training step
            if len(replay) >= min_buffer:
                batch = replay.sample(batch_size)
                states = torch.tensor(batch.state, dtype=torch.float32).to(device)
                actions = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(device)
                rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(device)
                next_states = torch.tensor(batch.next_state, dtype=torch.float32).to(device)
                dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(device)

                q_values = policy_net(states).gather(1, actions)
                with torch.no_grad():
                    q_next = target_net(next_states).max(1)[0].unsqueeze(1)
                    q_target = rewards + gamma * q_next * (1.0 - dones)

                loss = nn.functional.mse_loss(q_values, q_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # decay epsilon
            epsilon = max(epsilon_final, epsilon * epsilon_decay)

        # update target network
        if ep % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {ep}/{episodes} - total_reward={total_reward_ep:.2f} - epsilon={epsilon:.3f}")

        # periodic save
        if ep % 20 == 0:
            torch.save(policy_net.state_dict(), "dqn_agent.pth")
            print("Saved dqn_agent.pth")

    # final save
    torch.save(policy_net.state_dict(), "dqn_agent.pth")
    print("Training complete. Model saved to dqn_agent.pth")
    return policy_net

if __name__ == "__main__":
    # Small debugging run (reduce episodes/steps for quick test)
    model = train_dqn(episodes=120, steps_per_episode=80)
