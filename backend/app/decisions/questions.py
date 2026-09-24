"""Laya question schemas for MiroFast decisions.

Every schema is defined at runtime (no fine-tuning): criteria are natural
language descriptions the checkpoint scores in a single forward pass.

Round schemas must be identical across all agent states in a round so
`Router.predict_batch` can share forward passes (one pass per round).
"""

from typing import Dict, Any, List

# How many feed posts a decision state can offer as targets. The target
# criteria are built per round to exactly match the largest feed in that
# round, so the model never sees options that do not exist.
DEFAULT_MAX_FEED_POSTS = 5

_TWITTER_ACTION_CRITERIA = {
    "do_nothing": (
        "the user scrolls past the feed without reacting — statistically "
        "the most common outcome for any given user and feed"
    ),
    "like_post": (
        "the mildest and most common reaction: a silent endorsement of one "
        "visible post, without writing anything"
    ),
    "repost": (
        "the user shares one visible post to their own followers because it "
        "strongly deserves wider attention"
    ),
    "quote_post": (
        "the user shares one visible post while adding their own commentary "
        "on top of it"
    ),
    "create_post": (
        "the user publishes an original new post that is not a reply to any "
        "visible post"
    ),
}

_REDDIT_ACTION_CRITERIA = {
    "do_nothing": (
        "the user lurks past the feed without reacting — statistically the "
        "most common outcome for any given user and feed"
    ),
    "like_post": (
        "the mildest and most common reaction: a silent upvote of one "
        "visible post, without writing anything"
    ),
    "dislike_post": (
        "the user downvotes one visible post they disagree with or find "
        "low quality"
    ),
    "create_comment": (
        "the user writes a reply comment under one visible post"
    ),
    "create_post": (
        "the user publishes an original new post to the community, not a "
        "reply to any visible post"
    ),
}

_PLATFORM_ACTIONS = {
    "twitter": _TWITTER_ACTION_CRITERIA,
    "reddit": _REDDIT_ACTION_CRITERIA,
}

# Actions that require LLM text generation (the only remaining LLM calls in
# the hot loop). Everything else executes as a direct platform action.
GENERATIVE_ACTIONS = {
    "create_post",
    "create_comment",
    "quote_post",
}


def action_target_questions(platform: str, max_feed_posts: int = DEFAULT_MAX_FEED_POSTS) -> Dict[str, Any]:
    """Round decision schema: action choice + target post + stance, one pass.

    Args:
        platform: "twitter" or "reddit".
        max_feed_posts: largest feed size this round; the target criteria are
            generated to match so every offered label is real.
    """
    platform = platform.lower()
    if platform not in _PLATFORM_ACTIONS:
        raise ValueError(f"Unknown platform: {platform}")

    target_criteria: Dict[str, str] = {}
    for i in range(1, max_feed_posts + 1):
        target_criteria[f"post_{i}"] = (
            f"the feed post labeled post_{i}"
        )
    target_criteria["none"] = (
        "no specific visible post — only valid for staying silent or "
        "publishing an original new post"
    )

    return {
        "action": {
            "type": "choice",
            "instructions": (
                "Based on this user's profile and the social media feed they "
                "are looking at right now, which single action will they "
                "take? Choose the one that best reflects their personality, "
                "opinions and current inclination."
            ),
            "criteria": dict(_PLATFORM_ACTIONS[platform]),
        },
        "target_post": {
            "type": "choice",
            "instructions": (
                "Which feed post does the chosen action target? Liking, "
                "reposting, quoting and commenting must name the specific "
                "post; choose none only for staying silent or an original "
                "new post."
            ),
            "criteria": target_criteria,
        },
        "stance": {
            "type": "score",
            "instructions": (
                "This user's current stance toward the central topic or "
                "event discussed in the feed."
            ),
            "criteria": [
                "strongly opposed",
                "opposed",
                "neutral or undecided",
                "supportive",
                "strongly supportive",
            ],
        },
    }


def moderation_questions() -> Dict[str, Any]:
    """Content safety gate applied to LLM-generated text before posting."""
    return {
        "toxic": {
            "type": "noul",
            "instructions": (
                "Is this social media post toxic: hateful, harassing, "
                "threatening, or severely uncivil?"
            ),
        },
        "on_persona": {
            "type": "noul",
            "instructions": (
                "Is this post consistent with the author's personality and "
                "opinions, not a generic statement anyone could write?"
            ),
        },
    }


def stance_question(topic: str) -> Dict[str, Any]:
    """Stance scoring for arbitrary texts (report analytics, config setup)."""
    return {
        "stance": {
            "type": "score",
            "instructions": (
                f"The author's stance toward: {topic}."
            ),
            "criteria": [
                "strongly opposed",
                "opposed",
                "neutral or undecided",
                "supportive",
                "strongly supportive",
            ],
        },
    }


def sentiment_question() -> Dict[str, Any]:
    """Sentiment scoring for arbitrary texts (report analytics)."""
    return {
        "sentiment": {
            "type": "score",
            "instructions": "The emotional sentiment of this text.",
            "criteria": [
                "very negative",
                "negative",
                "neutral",
                "positive",
                "very positive",
            ],
        },
    }


def interviewee_selection_question(question: str, profiles: List[str], max_agents: int) -> Dict[str, Any]:
    """Dynamic-choice question: pick which simulated agents to interview.

    `profiles` are one-line descriptions; they become the choice criteria so
    the checkpoint scores each candidate in one forward pass.
    """
    criteria = {
        f"agent_{i}": desc for i, desc in enumerate(profiles)
    }
    return {
        "selection": {
            "type": "choice",
            "instructions": (
                f"To answer the question: {question} — which of these "
                f"simulated users are the most valuable to interview? "
                f"At most {max_agents} should be chosen overall; pick the "
                "single best candidate from this list."
            ),
            "criteria": criteria,
        }
    }


def report_tool_question(question: str, tools: Dict[str, str]) -> Dict[str, Any]:
    """Dynamic-choice question: which report tool answers this need next."""
    return {
        "tool": {
            "type": "choice",
            "instructions": (
                f"To progress the report task: {question} — which tool "
                "should be called next?"
            ),
            "criteria": dict(tools),
        },
        "need_more_evidence": {
            "type": "noul",
            "instructions": (
                "Is more evidence from the simulation data needed before "
                "this section can be written well?"
            ),
        },
    }
