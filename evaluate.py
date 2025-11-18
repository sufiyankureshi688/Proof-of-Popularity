# evaluate.py
# Load dqn_agent.pth (if present) and run a number of steps while printing events and controlled agent reward.
import os
import torch
import numpy as np
from environment import SocialMediaEnv, Action, INTERACT_SUBACTIONS
from train import DQN, ACTION_SIZE, state_from_env, select_target_post_for_interact

def load_policy(path="dqn_agent.pth", state_dim=13):
    device = torch.device("cpu")
    net = DQN(state_dim, ACTION_SIZE).to(device)
    if os.path.exists(path):
        net.load_state_dict(torch.load(path, map_location=device))
        net.eval()
        print(f"Loaded policy from {path}")
        return net
    else:
        print(f"No model file at {path} — running random policy.")
        return None

def act_from_policy(net, state):
    if net is None:
        # fallback: random policy
        a = np.random.randint(0, ACTION_SIZE)
        return int(a)
    s = torch.from_numpy(state).unsqueeze(0)
    with torch.no_grad():
        q = net(s).cpu().numpy()[0]
    return int(np.argmax(q))

def run_eval(steps=50, model_path="dqn_agent.pth"):
    env = SocialMediaEnv()
    env.reset()
    n_recent = 10
    state_dim = 3 + n_recent
    net = load_policy(model_path, state_dim=state_dim)
    total_reward = 0.0
    for t in range(steps):
        state = np.array(env.get_obs_vector(None, n_recent=n_recent), dtype=np.float32)
        a = act_from_policy(net, state)

        if a == 0:
            env_action = Action("idle", None, None)
        elif a == 1:
            env_action = Action("post", None, None)
        else:
            sub_idx = a - 2
            sub = INTERACT_SUBACTIONS[sub_idx]
            target = select_target_post_for_interact(env, env.controlled_id)
            if target is None:
                env_action = Action("idle", None, None)
            else:
                env_action = Action("interact", target, sub)

        obs, reward, done, info = env.step({env.controlled_id: env_action})
        total_reward += reward
        print(env.render_last_step())
        print(f"Controlled agent received {env.agents[env.controlled_id].tokens_received:.2f} tokens this step; shaped reward={reward:.2f}; new_followers={info.get('new_followers',0)} new_friends={info.get('new_friends',0)}\n")

    print("=== Final Summary ===")
    for a in env.agents:
        print(f"Agent {a.id}: followers={len(a.followers)}, friends={len(a.friends)}, weight={a.weight:.2f}, cum_eng={a.cumulative_engagement_received:.1f}, tokens_total={a.tokens_received:.2f}")
    print(f"Total shaped reward accumulated during eval run: {total_reward:.2f}")

if __name__ == "__main__":
    run_eval(steps=40, model_path="dqn_agent.pth")
