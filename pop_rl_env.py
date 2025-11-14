import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MultiAgentPopEnv(gym.Env):
    """
    Multi-agent-like social environment with 3 actions:
      0 = post
      1 = interact (with other agents' posts)
      2 = idle

    Observation (8-dim):
      [weight, likes, shares, comments, saves, followers, friends, visibility]

    Token mint: tokens are the only reward. Tokens are distributed each step based on
    the *interactions you receive* that step (not directly on Weight). Weight still
    accumulates from relations + interactions and affects social dynamics, but
    pulling on the minting is driven by received interactions.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, other_agents=8, max_steps=100, seed=None):
        super().__init__()
        if seed is not None:
            np.random.seed(seed)

        self.max_steps = max_steps
        self.current_step = 0

        # Observation: 8 values as floats (weight, likes, shares, comments, saves, followers, friends, visibility)
        self.observation_space = spaces.Box(low=0.0, high=1e9, shape=(8,), dtype=np.float32)

        # Actions: 0=post, 1=interact, 2=idle
        self.action_space = spaces.Discrete(3)

        # Multi-agent: maintain other agents' weights and posting probabilities
        self.other_agents = int(other_agents)
        # Random initial weights for other agents (50-600)
        self.other_weights = np.random.uniform(50, 600, size=self.other_agents)
        # Posting probability per-step for other agents (higher weight => more likely to post)
        self.other_base_post_p = np.clip(self.other_weights / 1000.0, 0.05, 0.5)

        # Token generator parameters
        self.base_mint_per_step = 10.0  # base tokens minted each step
        self.mint_noise_scale = 3.0

        # keep a short window of received interactions for pulling
        self.recent_received_window = []  # list of floats (sum of interactions received per step)
        self.window_len = 5

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Agent state
        self.engagement = np.zeros(4, dtype=np.float32)  # likes, shares, comments, saves (cumulative)
        self.followers = 0.0  # one-way
        self.friends = 0.0    # count of mutual friendships
        self.visibility = 0.0  # ephemeral boost -> increases chance that others see/reciprocate
        self.weight = 0.0

        # track reciprocity / relation quality
        self.reciprocity_bonus = 0.0

        # token accounting
        self.tokens_received = 0.0

        # small history of recent posts (for reciprocity logic)
        self.recent_posts = []  # list of dicts describing recent posts; kept short
        self.recent_received_window = [0.0] * self.window_len

        self._update_weight()

        return self._get_obs(), {}

    def step(self, action):
        """
        Execute one timestep.
        Returns: obs, reward, terminated, truncated, info
        Note: reward is ONLY the tokens received that step. Posting/interacting do NOT give direct reward.
        """
        self.current_step += 1

        # track interactions received this step (will be filled by other agents' activity simulation)
        interactions_received_this_step = np.zeros(4, dtype=float)

        # First: other agents may post this step; collect exposure from those posts
        # and simulate some interactions they generate toward our agent
        exposure_from_others, interactions_received_this_step = self._simulate_other_agents_activity()

        # No direct reward from actions; tokens minted at end are the reward
        # still allow small step penalty for idle to discourage doing nothing
        immediate_penalty = 0.0

        if action == 0:  # post
            # Post effectiveness scales with Weight (increases how much engagement you attract)
            weight_scale = np.clip(self.weight / 200.0 + 0.1, 0.1, 20.0)

            # Expected engagement per type (base rates)
            base = np.array([3.0, 1.5, 0.8, 0.5])  # likes, shares, comments, saves
            lam = base * weight_scale
            delta = np.random.poisson(lam)

            # followers gained from posting (some fraction, influenced by weight)
            followers_gain = np.random.binomial(n=max(1, self.other_agents * 3),
                                                p=min(0.03 + self.weight / 3000.0, 0.6))
            followers_gain = float(followers_gain)

            # possible friendship formation: small chance some other agents become friends mutually
            new_friends = self._maybe_form_friends_on_post(delta.sum())

            # update internal state (these affect Weight but do NOT directly give RL reward)
            self.engagement += delta
            self.followers += followers_gain
            self.friends += new_friends

            # record recent post for possible reciprocity
            self.recent_posts.append({'eng': delta.copy(), 'followers_gain': followers_gain, 'step': self.current_step})
            if len(self.recent_posts) > 10:
                self.recent_posts.pop(0)

            # increase visibility
            self.visibility = min(self.visibility + np.sum(delta) * 0.05 + followers_gain * 0.2 + new_friends * 0.5, 100.0)

        elif action == 1:  # interact with others' posts
            # Interacting causes your account to be seen and can eventually change relations but
            # does NOT directly provide RL reward. It modifies state which can lead to more tokens later.
            weight_influence = np.clip(1.0 + self.weight / 500.0, 1.0, 20.0)

            # Interaction probabilities base (per candidate post)
            p_base = np.array([0.15, 0.05, 0.03, 0.04, 0.02])  # like, comment, share, save, follow base probabilities

            candidate_posts = max(1, int(self.other_agents * np.random.poisson(1.2)))
            p_eff = np.clip(p_base * np.clip(self.weight / 300.0 + 1.0, 1.0, 8.0), 0.0, 0.99)
            raw_gained = np.array([np.random.binomial(candidate_posts, pi) for pi in p_eff], dtype=float)

            # interacting increases your engagement counters (representing actions you performed which may be reciprocated)
            # these are scaled by weight influence to model higher-quality interactions
            effective_engagement = raw_gained[:4] * weight_influence
            effective_followers = raw_gained[4] * max(1.0, weight_influence * 0.5)

            self.engagement += effective_engagement
            self.followers += effective_followers

            # interacting raises visibility and reciprocity bonus
            self.visibility = min(self.visibility + np.sum(effective_engagement) * 0.04 + effective_followers * 0.3, 100.0)
            self.reciprocity_bonus += np.sum(effective_engagement) * 0.02 + effective_followers * 0.15

            # chance to form mutual friendships when interacting (depends on partner weight)
            new_friends = self._maybe_form_friends_on_interact(candidate_posts, weight_influence)
            if new_friends > 0:
                self.friends += new_friends

        else:  # idle
            immediate_penalty = -0.2
            self.visibility = max(self.visibility * 0.95 - 0.1, 0.0)

        # After action, reciprocity/followers may convert (as before)
        reciprocal_followers = self._simulate_reciprocity(exposure_from_others)
        if reciprocal_followers > 0:
            self.followers += reciprocal_followers

        # Update weight from current state (Weight still matters for social dynamics)
        self._update_weight()

        # --- Token minting based on interactions YOU RECEIVED this step ---
        # compute received interactions total (weighted sum of interaction types)
        # interactions_received_this_step comes from other agents' activity above
        received_weights = np.array([1.0, 2.0, 2.5, 1.0])
        received_interaction_value = float(np.dot(received_weights, interactions_received_this_step))

        # keep a short moving window of received interaction value
        self.recent_received_window.pop(0)
        self.recent_received_window.append(received_interaction_value)
        window_sum = float(sum(self.recent_received_window))

        # Mint tokens this step (safe numeric handling)
        minted = max(0.0, np.random.normal(self.base_mint_per_step, self.mint_noise_scale))

        # Pull numerator based on recent interactions received (not Weight)
        # add a small epsilon baseline so zero activity still gets tiny share
        eps = 1e-6
        alpha = 1.2
        MAX_NUMERATOR = 1e6
        safe_numerator = float(np.clip(window_sum, 0.0, MAX_NUMERATOR))
        pull_numerator = (safe_numerator ** alpha) if safe_numerator > 0 else 0.0

        others_numerators = float(np.sum(np.clip(self.other_weights, 1.0, MAX_NUMERATOR) ** alpha))
        denom = pull_numerator + others_numerators + 1e-9
        if denom > 0:
            self_share = (pull_numerator / denom) * minted
        else:
            self_share = 0.0

        self.tokens_received += self_share

        # FINAL reward: tokens received this step minus any immediate penalty
        reward = float(self_share + immediate_penalty)

        # clamp other_weights to prevent runaway growth
        self.other_weights = np.clip(self.other_weights, 1.0, 1e6)

        terminated = self.current_step >= self.max_steps
        truncated = False
        obs = self._get_obs()
        info = {
            'step': self.current_step,
            'minted': minted,
            'self_share': self_share,
            'received_interaction_value': received_interaction_value
        }
        return obs, float(reward), terminated, truncated, info

    def _update_weight(self):
        """Calculate Weight based on engagement, followers, friends, visibility and reciprocity bonus"""
        # weights for different components
        engagement_weight = 0.35
        follower_weight = 2.0
        friend_quality_weight = 3.0
        visibility_weight = 0.5

        engagement_score = np.sum(self.engagement) * engagement_weight
        follower_score = self.followers * follower_weight
        friend_score = self.friends * friend_quality_weight
        visibility_score = self.visibility * visibility_weight

        # reciprocity bonus adds directly
        self.weight = float(engagement_score + follower_score + friend_score + visibility_score + self.reciprocity_bonus)

    def _get_obs(self):
        return np.array([
            self.weight,
            self.engagement[0],  # likes
            self.engagement[1],  # shares
            self.engagement[2],  # comments
            self.engagement[3],  # saves
            self.followers,
            self.friends,
            self.visibility
        ], dtype=np.float32)

    def _simulate_other_agents_activity(self):
        """
        Simulate posts from other agents this timestep.
        Returns an exposure metric and a vector of interactions they produced toward our agent.
        That vector is: [likes, shares, comments, saves] received this step.
        """
        exposure = 0.0
        received = np.zeros(4, dtype=float)

        post_probs = np.clip(self.other_base_post_p * (1.0 + self.other_weights / 1000.0), 0.01, 0.95)
        posts = np.random.rand(self.other_agents) < post_probs

        for i, did_post in enumerate(posts):
            if not did_post:
                continue
            popularity = np.clip(self.other_weights[i] / 200.0 + np.random.rand() * 0.5, 0.1, 10.0)

            # exposure increases with popularity and our visibility
            exposure += popularity * (0.05 + self.visibility / 200.0)

            # chance this other agent interacts with us directly (they like/comment/share/save our content)
            # probability increases if our visibility is higher and if their weight is higher
            base_p = 0.02 + min(self.visibility / 200.0, 0.3)
            partner_influence = np.clip(self.other_weights[i] / 500.0, 0.01, 2.0)
            p_interact = np.clip(base_p * partner_influence, 0.0, 0.9)

            if np.random.rand() < p_interact:
                # they may perform several interaction types; model counts with small Poisson
                # expected interactions scale with their weight and our visibility
                lam = np.array([0.5, 0.15, 0.1, 0.08]) * (1.0 + self.other_weights[i] / 400.0)
                contrib = np.random.poisson(lam)
                # add to received interactions
                received += contrib
                # sometimes they also follow
                if np.random.rand() < 0.05 * (1.0 + self.other_weights[i] / 600.0):
                    self.followers += 1.0

            # small chance they directly follow without interacting
            if np.random.rand() < 0.01 + self.visibility / 1000.0:
                self.followers += 0.2

        # clamp received to reasonable values
        received = np.clip(received, 0.0, 1e6)

        # apply these received interactions to our engagement counters (they increase Weight)
        self.engagement += received

        return exposure, received

    def _simulate_reciprocity(self, exposure):
        """
        Convert exposure and reciprocity_bonus into actual follower gains from other agents.
        Reciprocity also considers the agent's weight directly.
        """
        base_p = 0.01 + exposure / (1.0 + self.other_agents * 0.5) + self.reciprocity_bonus * 0.01
        rep_factor = np.clip(1.0 + self.weight / 1000.0, 1.0, 5.0)
        p = np.clip(base_p * rep_factor, 0.0, 0.6)

        new_followers = np.random.binomial(self.other_agents, p)
        # decay reciprocity bonus slightly after use
        self.reciprocity_bonus = max(0.0, self.reciprocity_bonus * 0.9 - new_followers * 0.01)
        # When followers convert, they might increase some other agents' weights slightly (simulate network)
        # Here we model it by slightly increasing a few random other_weights based on your visibility
        if new_followers > 0:
            idx = np.random.choice(self.other_agents, size=min(new_followers, self.other_agents), replace=False)
            for i in idx:
                self.other_weights[i] = max(1.0, self.other_weights[i] * (1.0 + min(self.visibility / 500.0, 0.2)))
        # Clamp other_weights to prevent runaway growth
        self.other_weights = np.clip(self.other_weights, 1.0, 1e6)
        return float(new_followers)

    def _maybe_form_friends_on_post(self, engagement_sum):
        """Chance to form mutual friendships when you post: depends on engagement and visibility"""
        p_friend_per_agent = np.clip(0.001 + engagement_sum * 0.0005 + self.visibility / 2000.0, 0.001, 0.05)
        new = np.random.binomial(self.other_agents, p_friend_per_agent)
        if new > 0:
            partners = np.random.choice(self.other_agents, size=min(new, self.other_agents), replace=False)
            for i in partners:
                partner_w = self.other_weights[i]
                # forming friendship affects both sides by partner weight and own weight
                self.reciprocity_bonus += partner_w * 0.01
                # partner also gets a little boost from befriending you
                self.other_weights[i] = max(1.0, self.other_weights[i] + self.weight * 0.005)
        # Clamp other_weights to keep numbers bounded
        self.other_weights = np.clip(self.other_weights, 1.0, 1e6)
        return float(new)

    def _maybe_form_friends_on_interact(self, candidate_posts, weight_influence):
        """Chance to form mutual friendships while interacting: depends on candidate posts seen and weight.
        Befriending low-weight partners can apply small negative drag on your weight (modeled via reciprocity bonus reduction).
        """
        p_base = np.clip(0.005 + candidate_posts * 0.0008 + self.visibility / 3000.0, 0.005, 0.1)
        new = np.random.binomial(self.other_agents, p_base)
        if new > 0:
            partners = np.random.choice(self.other_agents, size=min(new, self.other_agents), replace=False)
            for i in partners:
                partner_w = self.other_weights[i]
                # if partner is low-weight relative to you, your reciprocity_bonus can be slightly reduced
                if partner_w < 0.3 * self.weight:
                    # small drag for low-quality relation
                    self.reciprocity_bonus = max(0.0, self.reciprocity_bonus - 0.02)
                    # partner gets small gain
                    self.other_weights[i] += self.weight * 0.002
                else:
                    # quality friendship: both benefit
                    self.reciprocity_bonus += partner_w * 0.02
                    self.other_weights[i] += self.weight * 0.005
        # Clamp other_weights to prevent runaway growth
        self.other_weights = np.clip(self.other_weights, 1.0, 1e6)
        return float(new)

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        print(f"Weight: {self.weight:.2f}")
        print(f"Engagement: Likes={self.engagement[0]:.0f}, Shares={self.engagement[1]:.0f}, "
              f"Comments={self.engagement[2]:.0f}, Saves={self.engagement[3]:.0f}")
        print(f"Followers: {self.followers:.0f}, Friends: {self.friends:.0f}, Visibility: {self.visibility:.2f}")
        print(f"Tokens received (cum): {self.tokens_received:.2f}")
        print(f"Other agents: {self.other_agents} (sample weights mean {self.other_weights.mean():.1f})")
