# pop_rl_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MultiAgentPopEnv(gym.Env):
    """
    Multi-agent-like social environment with 3 actions:
      0 = post
      1 = interact (with other agents' posts)
      2 = idle

    Observation (7-dim):
      [reputation, likes, shares, comments, saves, followers, visibility]
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, other_agents=8, max_steps=100, seed=None):
        super().__init__()
        if seed is not None:
            np.random.seed(seed)

        self.max_steps = max_steps
        self.current_step = 0

        # Observation: 7 values as floats (reputation, likes, shares, comments, saves, followers, visibility)
        self.observation_space = spaces.Box(low=0.0, high=1e6, shape=(7,), dtype=np.float32)

        # Actions: 0=post, 1=interact, 2=idle
        self.action_space = spaces.Discrete(3)

        # Multi-agent: maintain other agents' reputations and posting probabilities
        self.other_agents = int(other_agents)
        # Random initial reputations for other agents (50-600)
        self.other_reps = np.random.uniform(50, 600, size=self.other_agents)
        # Posting probability per-step for other agents (higher rep => more likely to post)
        self.other_base_post_p = np.clip(self.other_reps / 1000.0, 0.05, 0.5)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Agent state
        self.engagement = np.zeros(4, dtype=np.float32)  # likes, shares, comments, saves
        self.followers = 0.0
        self.visibility = 0.0  # ephemeral boost -> increases chance that others see/reciprocate
        self.reputation = 0.0

        # track bonuses / long-term effects
        self.reciprocity_bonus = 0.0

        # small history of recent posts (for reciprocity logic)
        self.recent_posts = []  # list of dicts describing recent posts; kept short
        self._update_reputation()

        return self._get_obs(), {}

    def step(self, action):
        """
        Execute one timestep.
        Returns: obs, reward, terminated, truncated, info
        """
        self.current_step += 1
        reward = 0.0

        # First: other agents may post this step; collect exposure from those posts
        exposure_from_others = self._simulate_other_agents_activity()

        if action == 0:  # post
            # Post effectiveness scales with reputation
            rep_scale = np.clip(self.reputation / 200.0 + 0.1, 0.1, 10.0)

            # Expected engagement per type (base rates)
            base = np.array([3.0, 1.5, 0.8, 0.5])  # likes, shares, comments, saves (per-post base)
            # sample engagement as Poisson around base*rep_scale (floored to ints)
            lam = base * rep_scale
            delta = np.random.poisson(lam)

            # followers gained from posting (some fraction)
            followers_gain = np.random.binomial(n=max(1, self.other_agents * 3),
                                                p=min(0.05 + self.reputation / 2000.0, 0.5))
            followers_gain = float(followers_gain)

            # update internal state
            self.engagement += delta
            self.followers += followers_gain

            # record recent post for possible reciprocity
            self.recent_posts.append({'eng': delta.copy(), 'followers_gain': followers_gain, 'step': self.current_step})
            if len(self.recent_posts) > 10:
                self.recent_posts.pop(0)

            # reward: weighted engagement + follower bonus + small visibility boost
            weights = np.array([1.0, 2.0, 2.5, 1.0])  # shares/comments worth more
            reward += float(np.dot(weights, delta)) + followers_gain * 3.0

            # posting increases ephemeral visibility which may cause others to reciprocate on their post step
            self.visibility = min(self.visibility + np.sum(delta) * 0.05 + followers_gain * 0.2, 100.0)

        elif action == 1:  # interact with others' posts
            # -----------------------------
            # KEY CHANGE: interactions are *valued more* if the agent has higher reputation.
            # We apply rep_influence as a multiplicative factor to:
            #   - the effective number of interactions produced
            #   - the immediate reward per interaction
            # -----------------------------
            # rep_influence >= 1.0; small reps -> ~1.0, large reps -> >1.0
            rep_influence = np.clip(1.0 + self.reputation / 500.0, 1.0, 10.0)

            # Interaction probabilities base (per candidate post)
            p_base = np.array([0.15, 0.05, 0.03, 0.04, 0.02])  # like, comment, share, save, follow base probabilities

            # Candidate posts seen this step (depends on other_agents and their activity)
            candidate_posts = max(1, int(self.other_agents * np.random.poisson(1.2)))

            # Effective probability scales mildly with rep (so high-rep users are more likely to find interactions)
            p_eff = np.clip(p_base * np.clip(self.reputation / 300.0 + 1.0, 1.0, 5.0), 0.0, 0.95)

            # Number of raw interactions before reputation multiplier
            raw_gained = np.array([np.random.binomial(candidate_posts, pi) for pi in p_eff], dtype=float)

            # Now scale the *value* (and effective weight) of those interactions by rep_influence
            # We treat the first four as engagement you cause and the fifth as follower gain.
            gained = raw_gained.copy()
            # increase the effective impact of each interaction type by rep_influence
            effective_engagement = gained[:4] * rep_influence
            effective_followers = gained[4] * max(1.0, rep_influence * 0.5)  # follows scale but less aggressively

            # apply to state
            self.engagement += effective_engagement
            self.followers += effective_followers

            # reward from interacting: interactions themselves yield immediate reward,
            # scaled by rep_influence so high-rep agents get more value performing interactions.
            weights = np.array([0.6, 1.5, 1.7, 0.8])
            reward += float(np.dot(weights, effective_engagement)) + effective_followers * 2.0 * rep_influence

            # interacting raises visibility (others see you more, increasing chance of reciprocation)
            self.visibility = min(self.visibility + np.sum(effective_engagement) * 0.04 + effective_followers * 0.3, 100.0)

            # stronger reciprocity bonus for higher reputation
            self.reciprocity_bonus += np.sum(effective_engagement) * 0.02 + effective_followers * 0.15

        else:  # idle
            reward -= 0.5  # small negative reward for doing nothing
            # visibility decays slowly
            self.visibility = max(self.visibility * 0.95 - 0.1, 0.0)

        # After action, some other agents may reciprocate influenced by both exposure and your reputation
        reciprocal_followers = self._simulate_reciprocity(exposure_from_others)
        if reciprocal_followers > 0:
            self.followers += reciprocal_followers
            reward += reciprocal_followers * 2.5

        # Update reputation from current state
        self._update_reputation()

        terminated = self.current_step >= self.max_steps
        truncated = False
        obs = self._get_obs()
        info = {
            'step': self.current_step
        }
        return obs, float(reward), terminated, truncated, info

    def _update_reputation(self):
        """Calculate reputation based on engagement, followers, visibility, reciprocity bonus"""
        engagement_weight = 0.35
        followers_weight = 2.0
        visibility_weight = 0.5

        engagement_score = np.sum(self.engagement) * engagement_weight
        follower_score = self.followers * followers_weight
        visibility_score = self.visibility * visibility_weight

        self.reputation = float(engagement_score + follower_score + visibility_score + self.reciprocity_bonus)

    def _get_obs(self):
        return np.array([
            self.reputation,
            self.engagement[0],  # likes
            self.engagement[1],  # shares
            self.engagement[2],  # comments
            self.engagement[3],  # saves
            self.followers,
            self.visibility
        ], dtype=np.float32)

    def _simulate_other_agents_activity(self):
        """
        Simulate posts from other agents this timestep.
        Returns an exposure metric (how much exposure they generated to the controlled agent).
        """
        exposure = 0.0
        # other agents post probabilistically based on their base post prob (and their reputation)
        post_probs = np.clip(self.other_base_post_p * (1.0 + self.other_reps / 1000.0), 0.01, 0.95)
        posts = np.random.rand(self.other_agents) < post_probs

        for i, did_post in enumerate(posts):
            if not did_post:
                continue
            popularity = np.clip(self.other_reps[i] / 200.0 + np.random.rand() * 0.5, 0.1, 10.0)
            exposure += popularity * (0.05 + self.visibility / 200.0)
            if np.random.rand() < min(0.02 + self.visibility / 200.0 + self.reputation / 5000.0, 0.6):
                exposure += 0.2

        return exposure

    def _simulate_reciprocity(self, exposure):
        """
        Convert exposure and reciprocity_bonus into actual follower gains from other agents.
        Reciprocity now also considers the agent's reputation directly.
        """
        # base probability that a particular other agent converts to follower
        base_p = 0.01 + exposure / (1.0 + self.other_agents * 0.5) + self.reciprocity_bonus * 0.01
        # reputation influence: higher reputation makes each other agent more likely to convert
        rep_factor = np.clip(1.0 + self.reputation / 1000.0, 1.0, 5.0)
        p = np.clip(base_p * rep_factor, 0.0, 0.6)

        new_followers = np.random.binomial(self.other_agents, p)
        # decay reciprocity bonus slightly after use
        self.reciprocity_bonus = max(0.0, self.reciprocity_bonus * 0.9 - new_followers * 0.01)
        return float(new_followers)

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        print(f"Reputation: {self.reputation:.2f}")
        print(f"Engagement: Likes={self.engagement[0]:.0f}, Shares={self.engagement[1]:.0f}, "
              f"Comments={self.engagement[2]:.0f}, Saves={self.engagement[3]:.0f}")
        print(f"Followers: {self.followers:.0f}, Visibility: {self.visibility:.2f}")
        print(f"Other agents: {self.other_agents} (sample reps mean {self.other_reps.mean():.1f})\n")


if __name__ == "__main__":
    env = MultiAgentPopEnv(other_agents=6, max_steps=50, seed=42)
    obs, _ = env.reset()
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated:
            break
