# environment.py
# (Full environment, with a helper to return a fixed-size numeric observation vector for an agent)
import random
import math
from collections import deque, defaultdict, namedtuple

random.seed(0)

Action = namedtuple("Action", ["type", "target_id", "subaction"])  # target_id or None, subaction for interact

INTERACT_SUBACTIONS = ["like", "comment", "share", "save", "follow"]
SUBACTION_VALUES = {"like": 1.0, "comment": 2.0, "share": 3.0, "save": 1.0, "follow": 2.5}

class Post:
    def __init__(self, author_id, step_id, text=""):
        self.author = author_id
        self.step = step_id
        self.text = text
        # engagement counters
        self.engagement = defaultdict(int)  # subaction -> count
        self.id = f"p{author_id}s{step_id}"

    def engage(self, subaction):
        self.engagement[subaction] += 1

    def total_engagement_score(self):
        return sum(SUBACTION_VALUES.get(k, 0) * v for k, v in self.engagement.items())

class AgentState:
    def __init__(self, agent_id):
        self.id = agent_id
        self.followers = set()      # who follows me
        self.following = set()      # who I follow
        self.friends = set()        # mutual followers
        self.cumulative_engagement_received = 0.0
        self.weight = 1.0           # initial minimal weight
        self.posts = deque(maxlen=50)
        # for statistics
        self.tokens_received = 0.0

class SocialMediaEnv:
    def __init__(self, n_autonomous=10, token_per_step=50, seed=0):
        random.seed(seed)
        self.n_autonomous = n_autonomous
        self.n_agents = n_autonomous + 1
        self.controlled_id = self.n_agents - 1  # last agent is controlled RL agent
        self.token_per_step = token_per_step
        self.step_count = 0

        # agents
        self.agents = [AgentState(i) for i in range(self.n_agents)]

        # global posts list for lookup (most recent first)
        self.posts = []

        # per-step event log
        self.last_step_events = []

        # influence scaling hyperparameter (tunable)
        # This controls how strong actor influence is for token allocation.
        # Higher -> more influence of high-weight actors on token shares.
        # It's normalized by current max weight to avoid runaway numeric explosion.
        self.actor_influence_scale = 5.0

    def reset(self):
        self.step_count = 0
        self.posts = []
        for a in self.agents:
            a.followers.clear(); a.following.clear(); a.friends.clear()
            a.cumulative_engagement_received = 0.0
            a.weight = 1.0
            a.posts.clear()
            a.tokens_received = 0.0
        self.last_step_events = []
        return self._observation()

    def _observation(self):
        """
        Minimal dictionary observation (keeps backward compatibility).
        """
        obs = {
            "step": self.step_count,
            "my_id": self.controlled_id,
            "my_followers": len(self.agents[self.controlled_id].followers),
            "my_friends": len(self.agents[self.controlled_id].friends),
            "my_weight": self.agents[self.controlled_id].weight,
            "recent_posts": [(p.id, p.author, p.step, p.total_engagement_score()) for p in self.posts[-10:]]
        }
        return obs

    def get_obs_vector(self, agent_id=None, n_recent=10):
        """
        New helper: returns a numeric vector (floats) for use by the RL agent.
        Vector layout (length = 3 + n_recent):
        [my_weight, my_num_followers, my_num_friends, recent_post_scores...]
        recent_post_scores are the total_engagement_score of the most recent n_recent posts,
        normalized by (1 + max_author_weight) to keep values in manageable range.
        If there are fewer posts than n_recent, zero-fill at the end.
        """
        if agent_id is None:
            agent_id = self.controlled_id
        my = self.agents[agent_id]
        recent = list(self.posts[-n_recent:]) if self.posts else []
        # normalization factor
        max_w = max(1.0, max((a.weight for a in self.agents), default=1.0))
        vec = [my.weight / (1.0 + max_w), len(my.followers) / 10.0, len(my.friends) / 10.0]
        # recent posts: include score and whether I'm the author (as separate scalar added to score)
        for p in recent[-n_recent:]:
            score = p.total_engagement_score() / (1.0 + max_w)
            is_my_post = 1.0 if (p.author == agent_id) else 0.0
            vec.append(score + 0.1 * is_my_post)
        # pad
        while len(vec) < 3 + n_recent:
            vec.append(0.0)
        return [float(x) for x in vec]

    def _register_post(self, author_id, text=""):
        p = Post(author_id, self.step_count, text)
        self.posts.append(p)
        self.agents[author_id].posts.append(p)
        self.last_step_events.append(f"Agent {author_id} posted {p.id}")
        return p

    def _form_friendship_if_needed(self, a_id, b_id):
        a = self.agents[a_id]
        b = self.agents[b_id]
        # friendship occurs if mutual follow
        if (b_id in a.following) and (a_id in b.following):
            if b_id not in a.friends:
                a.friends.add(b_id)
                b.friends.add(a_id)
                self.last_step_events.append(f"Friendship formed between {a_id} and {b_id}")

    def _update_weights(self):
        # base weight = cumulative_engagement_received + follower_count + friendship_count
        decay_factor = 0.995  # small decay per step (0.995 -> ~0.5 after ~138 steps)
        for a in self.agents:
            a.cumulative_engagement_received *= decay_factor
        base_weights = []
        for a in self.agents:
            base = a.cumulative_engagement_received + len(a.followers) + len(a.friends)
            base_weights.append(base if base > 0 else 1.0)
        for i, a in enumerate(self.agents):
            a.weight = base_weights[i]
        # friendship smoothing nudge
        for a in self.agents:
            if not a.friends:
                continue
            friend_weights = [self.agents[f].weight for f in a.friends]
            avg_friend = sum(friend_weights) / len(friend_weights)
            nudge = 0.1
            a.weight = a.weight + nudge * (avg_friend - a.weight)

    def _simulate_autonomous_action(self, agent_id):
        """
        Heuristic for autonomous agents: post with prob 0.08, interact with prob 0.4, else idle.
        When interacting choose recent posts with weight bias (flattened).
        """
        prob_post = 0.08
        prob_interact = 0.4
        r = random.random()
        if r < prob_post:
            return Action("post", None, None)
        elif r < prob_post + prob_interact:
            if not self.posts:
                return Action("idle", None, None)
            recent = self.posts[-50:]
            weights = []
            for p in recent:
                # flatten weight bias to avoid extreme rich-get-richer
                weights.append(max(0.01, self.agents[p.author].weight ** 0.5))
            choice = random.choices(recent, weights=weights, k=1)[0]
            sub = random.choices(INTERACT_SUBACTIONS, k=1)[0]
            return Action("interact", choice, sub)
        else:
            return Action("idle", None, None)

    def _scaled_engagement_for_tokens(self, actor_id, base_value):
        """
        Compute a bounded, normalized scaled engagement value for token allocation
        based on the actor's weight. This *does not* change cumulative_engagement_received.
        - actor influence is normalized by current max weight.
        - we use sqrt to flatten extremes, then scale by actor_influence_scale.
        Returns scaled_value >= base_value.
        """
        # avoid division by zero
        max_w = max(1.0, max((a.weight for a in self.agents), default=1.0))
        actor_w = max(0.0, self.agents[actor_id].weight)
        # normalized influence in [0,1]
        norm = (actor_w / max_w)
        # flatten extremes
        influence = math.sqrt(norm)
        # final scaled value (bounded): base * (1 + influence * scale)
        scaled = base_value * (1.0 + influence * self.actor_influence_scale)
        return scaled

    def step(self, actions):
        """
        actions: dict {agent_id: Action} for controlled agent only.
        Returns observation, reward, done, info
        reward: tokens_received + bonuses for new followers/friends this step
        """
        self.step_count += 1
        self.last_step_events = []
        # prepare actions for all agents
        planned = {}
        for i in range(self.n_agents):
            if i == self.controlled_id:
                planned[i] = actions.get(i, Action("idle", None, None))
            else:
                planned[i] = self._simulate_autonomous_action(i)

        # First pass: execute posts
        for i, act in planned.items():
            if act.type == "post":
                self._register_post(i, text=f"Post by {i} at step {self.step_count}")

        # record follower/friend changes to compute shaped reward
        followers_before = [set(a.followers) for a in self.agents]
        friends_before = [set(a.friends) for a in self.agents]

        # Engagement accumulators for token distribution
        engagement_scores_by_agent = [0.0 for _ in range(self.n_agents)]

        # Second pass: handle interacts and idles
        for i, act in planned.items():
            if act.type == "idle":
                continue
            elif act.type == "interact":
                target_post = act.target_id
                if not isinstance(target_post, Post):
                    matches = [p for p in self.posts if p.id == target_post]
                    target_post = matches[-1] if matches else None
                if target_post is None:
                    continue
                sub = act.subaction
                if sub not in INTERACT_SUBACTIONS:
                    sub = random.choice(INTERACT_SUBACTIONS)

                # record interaction on the post (keeps original post-level counters)
                target_post.engage(sub)

                # base engagement value (does NOT depend on who acted)
                base_engagement = SUBACTION_VALUES.get(sub, 1.0)

                # actor-scaled engagement used ONLY for token allocation / short-term score
                scaled_for_tokens = self._scaled_engagement_for_tokens(i, base_engagement)

                author_id = target_post.author

                # engagement pool uses the scaled value (so influencer interactions still affect tokens)
                engagement_scores_by_agent[author_id] += scaled_for_tokens

                # BUT weight (cumulative engagement) increases only by the base value (independent of actor)
                self.agents[author_id].cumulative_engagement_received += base_engagement

                self.last_step_events.append(
                    f"Agent {i} {sub}ed post {target_post.id} (author {author_id}) [base={base_engagement:.2f}, scaled_for_tokens={scaled_for_tokens:.2f}]"
                )

                if sub == "follow":
                    if author_id != i and (author_id not in self.agents[i].following):
                        self.agents[i].following.add(author_id)
                        self.agents[author_id].followers.add(i)
                        self.last_step_events.append(f"Agent {i} followed Agent {author_id}")
                        # make follow more valuable for tokens (but weight increases by base follow only)
                        follow_bonus = SUBACTION_VALUES.get("follow", 2.5) * 1.5
                        follow_bonus_scaled = self._scaled_engagement_for_tokens(i, SUBACTION_VALUES.get("follow", 2.5) * 1.5)
                        engagement_scores_by_agent[author_id] += follow_bonus_scaled
                        # weight increases only by the base follow value
                        self.agents[author_id].cumulative_engagement_received += SUBACTION_VALUES.get("follow", 2.5)
                        self.last_step_events.append(f"Agent {author_id} gained follow bonus {follow_bonus_scaled:.2f}")
                        self._form_friendship_if_needed(i, author_id)

                # author might respond
                author = self.agents[author_id]
                follow_back_prob = min(0.5, 0.05 + 0.002 * author.weight)
                if random.random() < follow_back_prob and i != author_id:
                    if random.random() < 0.4:
                        if i not in author.following:
                            author.following.add(i)
                            self.agents[i].followers.add(author_id)
                            self.last_step_events.append(f"Agent {author_id} followed back Agent {i}")
                            self._form_friendship_if_needed(author_id, i)
                            # tokens: use scaled value based on the follower (author) weight
                            scaled_follow = self._scaled_engagement_for_tokens(author_id, SUBACTION_VALUES["follow"])
                            engagement_scores_by_agent[i] += scaled_follow
                            # but cumulative engagement (weight) for i increases only by base follow
                            self.agents[i].cumulative_engagement_received += SUBACTION_VALUES["follow"]
                    else:
                        if self.agents[i].posts:
                            target = self.agents[i].posts[-1]
                            chosen_sub = random.choice(INTERACT_SUBACTIONS[:-1])
                            target.engage(chosen_sub)
                            base_val = SUBACTION_VALUES[chosen_sub]
                            scaled_val = self._scaled_engagement_for_tokens(author_id, base_val)
                            engagement_scores_by_agent[i] += scaled_val
                            # weight increases only by base_val
                            self.agents[i].cumulative_engagement_received += base_val
                            self.last_step_events.append(f"Agent {author_id} {chosen_sub}ed Agent {i}'s post {target.id}")

        # Update weights
        self._update_weights()

        # Token distribution
        total_engagement = sum(engagement_scores_by_agent)
        if total_engagement <= 0:
            for a in self.agents:
                a.tokens_received = 0.0
            self.last_step_events.append("No engagement this step; no tokens distributed.")
        else:
            for idx, score in enumerate(engagement_scores_by_agent):
                frac = score / total_engagement
                tokens = frac * self.token_per_step
                self.agents[idx].tokens_received = tokens
                self.last_step_events.append(f"Agent {idx} received {tokens:.2f} tokens (engagement {score:.2f})")

        # shaped reward: tokens_received + follower/friend bonuses (only for controlled agent)
        new_followers = len(self.agents[self.controlled_id].followers - followers_before[self.controlled_id])
        new_friends = len(self.agents[self.controlled_id].friends - friends_before[self.controlled_id])
        follower_bonus = 5.0 * new_followers
        friend_bonus = 15.0 * new_friends

        reward = self.agents[self.controlled_id].tokens_received + follower_bonus + friend_bonus

        obs = self._observation()
        info = {"events": list(self.last_step_events),
                "new_followers": new_followers,
                "new_friends": new_friends}
        done = False
        return obs, reward, done, info

    def render_last_step(self):
        lines = [f"--- Step {self.step_count} ---"]
        lines.extend(self.last_step_events)
        lines.append("Agent summaries:")
        for a in self.agents:
            lines.append(f"Agent {a.id}: weight={a.weight:.2f}, followers={len(a.followers)}, friends={len(a.friends)}, cum_eng={a.cumulative_engagement_received:.1f}, tokens={a.tokens_received:.2f}")
        return "\n".join(lines)

if __name__ == "__main__":
    env = SocialMediaEnv()
    env.reset()
    for _ in range(5):
        action = Action("post", None, None) if random.random() < 0.3 else Action("idle", None, None)
        obs, r, d, info = env.step({env.controlled_id: action})
        print(env.render_last_step())
        print()
