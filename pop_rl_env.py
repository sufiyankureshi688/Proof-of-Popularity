import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MultiAgentPopEnv(gym.Env):
    """
    Multi-agent social env:
      Actions: 0=post, 1=interact, 2=idle
      Obs: [weight, likes, shares, comments, saves, followers, friends, visibility]
    Key rules:
      - Each agent (controlled + others) can generate posts; posts have 'quality'.
      - Interactions received (by any agent) over a short window determine pull on token generator.
      - Tokens minted each step are distributed among all agents proportional to (recent_received^alpha).
      - Weight still accumulates from relations + cumulative interactions but uses sqrt/log scaling.
      - Visibility decays; popular posts propagate (increase visibility).
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, other_agents=20, max_steps=100, seed=None):
        super().__init__()
        if seed is not None:
            np.random.seed(seed)

        self.max_steps = max_steps
        self.other_agents = int(other_agents)

        # observation: 8 values for controlled agent
        self.observation_space = spaces.Box(low=0.0, high=1e12, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        # other agents' state
        self.other_weights = np.random.uniform(50, 600, size=self.other_agents)
        # recent received windows for all agents (controlled + others)
        self.window_len = 5
        self.self_recent = [0.0] * self.window_len
        self.other_recent = np.zeros((self.other_agents, self.window_len), dtype=float)

        # base mint params
        self.base_mint_per_step = 20.0
        self.mint_noise_scale = 5.0

        # propagation thresholds
        self.propagation_threshold = 5.0  # received interaction value above which post propagates
        self.max_weight_clip = 1e6

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # controlled agent state
        self.engagement = np.zeros(4, dtype=float)   # cumulative received (likes, shares, comments, saves)
        self.followers = 0.0
        self.friends = 0.0
        self.visibility = 0.0
        self.weight = 0.0
        self.reciprocity_bonus = 0.0
        self.tokens_received = 0.0

        # keep recent posts list to allow propagation modelling
        self.recent_posts = []

        # reset recent windows
        self.self_recent = [0.0] * self.window_len
        self.other_recent = np.zeros((self.other_agents, self.window_len), dtype=float)

        self._update_weight()
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        # 1) Other agents may post; simulate their posts and the interactions they receive,
        #    and some fraction of those interactions may go to the controlled agent (propagation/exposure).
        exposure, others_received = self._simulate_other_agents_activity()

        # 2) Controlled agent action effects (no direct token reward; actions affect state)
        # if post: choose quality (stochastic but improved by weight) -> attracts interactions
        if action == 0:  # post
            post_quality = self._sample_post_quality(self.weight)
            # interactions controlled agent receives from own post (before others' interactions)
            own_received = self._simulate_post_receive(post_quality)
            # apply to engagement and recent window
            self.engagement += own_received
            received_value = self._received_value(own_received)
            self.self_recent.pop(0); self.self_recent.append(received_value)
            # propagation: if this post is popular, it increases visibility and may trigger extra incoming interactions
            if received_value >= self.propagation_threshold:
                boost = min(5.0, np.sqrt(received_value)) * 0.1
                self.visibility = min(100.0, self.visibility + boost)
            # small follower chance from own post
            self.followers += np.random.binomial(self.other_agents, min(0.01 + self.weight/5000.0, 0.2))
        elif action == 1:  # interact
            # interacting increases visibility slightly and increases chance of reciprocity,
            # and may cause friend formation; no direct token reward
            interact_effect = self._perform_interactions()
            self.engagement += interact_effect  # these represent actions that may be reciprocated
            received_value = 0.0  # performing interactions does not immediately create received interactions
            self.self_recent.pop(0); self.self_recent.append(received_value)
        else:  # idle
            # idle reduces visibility slightly
            self.visibility = max(0.0, self.visibility * 0.95 - 0.05)
            self.self_recent.pop(0); self.self_recent.append(0.0)

        # 3) Apply incoming interactions from other agents' posts (exposure -> some interactions land on us)
        # others_received is shape (4,), representing interactions from others directed to us this step
        self.engagement += others_received
        other_received_value = self._received_value(others_received)
        # add others' contributions into self_recent window (we already appended own post value)
        # add to last element (current step)
        self.self_recent[-1] += other_received_value

        # 4) Reciprocity / follower conversions triggered by exposure
        recov = self._simulate_reciprocity(exposure)
        if recov > 0:
            self.followers += recov

        # 5) Update Weight (but apply sqrt/log style scaling to damp growth)
        self._update_weight()

        # 6) Tokens minting: compute recent_received for all agents and distribute tokens proportionally
        # compute controlled agent's window sum
        self_window_sum = float(sum(self.self_recent))
        other_window_sums = self.other_recent.sum(axis=1)  # other_agents vector

        # clamp numerators
        alpha = 1.1
        MAX_NUM = 1e6
        safe_self = float(np.clip(self_window_sum, 0.0, MAX_NUM))
        safe_others = np.clip(other_window_sums, 0.0, MAX_NUM)

        pull_self = (safe_self ** alpha) if safe_self > 0 else 0.0
        pull_others = np.sum(safe_others ** alpha)
        minted = max(0.0, np.random.normal(self.base_mint_per_step, self.mint_noise_scale))

        denom = pull_self + pull_others + 1e-9
        if denom > 0:
            self_share = (pull_self / denom) * minted
        else:
            self_share = 0.0

        # update tokens and provide reward = tokens received this step (no other direct reward)
        self.tokens_received += self_share
        reward = float(self_share)

        # 7) decay visibility moderately (propagation is temporary)
        self.visibility = max(0.0, self.visibility * 0.92)

        # 8) update other agents' recent windows: shift and append their per-step received values
        # (we have computed per-step received for each other agent inside simulate_other_agents_activity)
        # Note: that function already shifted and updated self.other_recent.
        # 9) clamp other_weights to avoid overflow
        self.other_weights = np.clip(self.other_weights, 1.0, self.max_weight_clip)

        terminated = self.current_step >= self.max_steps
        truncated = False
        obs = self._get_obs()
        info = {
            'step': self.current_step,
            'minted': minted,
            'self_share': self_share,
            'self_window_sum': self_window_sum
        }
        return obs, reward, terminated, truncated, info

    # ---------------------
    # Helper / mechanics
    # ---------------------
    def _sample_post_quality(self, weight):
        """Return a post quality (0.01..5.0). Higher-weight accounts tend to produce better-quality posts."""
        base = np.clip(np.log1p(weight) / 6.0, 0.1, 3.0)  # moderate scaling
        quality = np.random.normal(loc=base, scale=0.5)
        return float(np.clip(quality, 0.01, 5.0))

    def _simulate_post_receive(self, quality):
        """
        Given a post quality, simulate the raw counts of interactions it receives from the population.
        Returns vector [likes, shares, comments, saves] (float counts).
        Uses sqrt/log damping so quality has diminishing returns.
        """
        # base expectation per interaction type
        base = np.array([2.0, 0.6, 0.4, 0.3])
        # effective multiplier from quality with sqrt damping
        mult = 1.0 + np.sqrt(quality)
        lam = base * mult
        received = np.random.poisson(lam)
        # cap per-step per-type to avoid huge spikes
        received = np.clip(received, 0, 500)
        return received.astype(float)

    def _perform_interactions(self):
        """
        Simulate the agent performing interactions across other agents' posts.
        Returns small positive numbers representing actions performed (which may be reciprocated later).
        """
        candidate_posts = max(1, int(self.other_agents * np.random.poisson(1.0)))
        p_base = np.array([0.12, 0.03, 0.02, 0.03])
        p_eff = np.clip(p_base * (1.0 + np.log1p(self.weight)/10.0), 0.0, 0.95)
        raw = np.array([np.random.binomial(candidate_posts, p) for p in p_eff], dtype=float)
        # scale down: these are actions performed, not received interactions
        return raw * 0.3

    
    def _simulate_other_agents_activity(self):
        """
        Simulate each other agent posting probabilistically, compute how many interactions
        they receive on their posts, and compute how many interactions are directed to the controlled agent.
        Also update other agents' recent windows (for token pulling).
        Returns:
        exposure_to_self (float) and received_to_self (4-vector)
        """
        exposure = 0.0
        received_to_self = np.zeros(4, dtype=float)

        post_probs = np.clip(self.other_weights / 1000.0, 0.01, 0.7)
        posts = np.random.rand(self.other_agents) < post_probs

        # For each other agent that posts: simulate their post's received interactions (their popularity)
        for i, did_post in enumerate(posts):
            if not did_post:
                # shift their recent window with zero if they didn't receive this step
                # use np.roll to shift left then set last entry to 0.0
                self.other_recent[i] = np.roll(self.other_recent[i], -1)
                self.other_recent[i, -1] = 0.0
                continue

            # compute their post quality (based on their weight)
            qual = np.clip(np.log1p(self.other_weights[i]) / 6.0 + np.random.normal(0, 0.5), 0.01, 5.0)
            # they receive interactions from the population (Poisson with sqrt damping)
            base = np.array([1.5, 0.4, 0.2, 0.15])
            mult = 1.0 + np.sqrt(qual)
            lam = base * mult * (1.0 + self.visibility/50.0)  # our visibility slightly affects exposure
            rec = np.random.poisson(lam).astype(float)
            rec = np.clip(rec, 0.0, 1000.0)

            # a fraction of their received interactions propagate to the controlled agent based on exposure & our visibility
            # propagation probability increases with their recent popularity and our visibility
            propagate_prob = np.clip(0.02 + np.log1p(rec.sum())/20.0 + self.visibility/200.0, 0.0, 0.6)
            if np.random.rand() < propagate_prob:
                # amount that falls to us is a small fraction proportional to popularity
                frac = np.clip(np.log1p(rec.sum()) / (50.0 + np.log1p(self.other_agents)), 0.0, 0.2)
                contrib = rec * frac * np.random.uniform(0.1, 0.5)
                received_to_self += contrib
                exposure += rec.sum() * 0.01
            # update other agent's recent window by their total received (weighted vector)
            val = float(np.dot(np.array([1.0,2.0,2.5,1.0]), rec))
            # shift the row left and set last entry to val
            self.other_recent[i] = np.roll(self.other_recent[i], -1)
            self.other_recent[i, -1] = val

            # small chance their followers increase (simulate network dynamics)
            if rec.sum() > 2:
                self.other_weights[i] = min(self.max_weight_clip, self.other_weights[i] * (1.0 + 0.001 * np.log1p(rec.sum())))

        # clamp and apply received_to_self to controlled agent later
        received_to_self = np.clip(received_to_self, 0.0, 1e6)
        return float(exposure), received_to_self


    def _simulate_reciprocity(self, exposure):
        """
        Small probability of followers converting due to exposure/reciprocity bonus.
        """
        base_p = 0.01 + exposure/(1.0 + self.other_agents*0.5) + self.reciprocity_bonus*0.01
        p = np.clip(base_p * np.clip(1.0 + self.weight/1000.0, 1.0, 3.0), 0.0, 0.5)
        new_followers = np.random.binomial(self.other_agents, p)
        self.reciprocity_bonus = max(0.0, self.reciprocity_bonus * 0.9 - new_followers * 0.01)
        return float(new_followers)

    def _received_value(self, received_vec):
        """Compute scalar value of a received interactions vector."""
        w = np.array([1.0, 2.0, 2.5, 1.0])
        return float(np.dot(w, received_vec))

    def _update_weight(self):
        """Weight uses sqrt/log scaling to avoid runaway growth."""
        engagement_sum = float(np.sum(self.engagement))
        engagement_score = np.sqrt(engagement_sum + 1.0) * 5.0   # damped
        follower_score = np.log1p(self.followers) * 10.0
        friend_score = np.log1p(self.friends) * 12.0
        visibility_score = np.sqrt(self.visibility + 1.0) * 2.0

        self.weight = float(engagement_score + follower_score + friend_score + visibility_score + self.reciprocity_bonus)
        # clip
        self.weight = float(np.clip(self.weight, 0.0, self.max_weight_clip))

    def _get_obs(self):
        return np.array([
            self.weight,
            self.engagement[0],
            self.engagement[1],
            self.engagement[2],
            self.engagement[3],
            self.followers,
            self.friends,
            self.visibility
        ], dtype=np.float32)

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        print(f"Weight: {self.weight:.2f}")
        print(f"Engagement: Likes={self.engagement[0]:.0f}, Shares={self.engagement[1]:.0f}, "
              f"Comments={self.engagement[2]:.0f}, Saves={self.engagement[3]:.0f}")
        print(f"Followers: {self.followers:.0f}, Friends: {self.friends:.0f}, Visibility: {self.visibility:.2f}")
        print(f"Tokens received (cum): {self.tokens_received:.4f}")
        print(f"Other agents mean weight: {self.other_weights.mean():.1f}")
