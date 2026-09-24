"""Compact Laya state builder for per-agent simulation decisions.

Renders an agent's persona and visible feed into a short labeled document
that fits the multilingual checkpoint's 1024-token context. Feed posts are
labeled post_1..post_N so the round's target question can reference them
and the caller can map a chosen label back to a platform post_id.
"""

from typing import Any, Dict, List, Optional, Tuple

MAX_PERSONA_CHARS = 600
MAX_POST_CHARS = 240
MAX_BIO_CHARS = 160


def _clip(text: Optional[str], limit: int) -> str:
    if not text:
        return ""
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def build_agent_state(
    name: str,
    persona: str,
    bio: str,
    posts: List[Dict[str, Any]],
    simulated_hour: int,
) -> Tuple[str, List[int]]:
    """Build a Laya decision state from an agent's profile and feed.

    Args:
        name: display name of the simulated user.
        persona: full persona text (will be clipped).
        bio: profile bio (will be clipped).
        posts: platform post dicts as returned by SocialAction.refresh()
            (post_id, content, num_likes, user_id, created_at, ...).
        simulated_hour: current simulated hour 0-23.

    Returns:
        (state_text, post_ids) where post_ids[i] is the platform post_id of
        the post rendered as post_{i+1} in the state.
    """
    lines = [
        "SIMULATED SOCIAL MEDIA USER",
        f"Name: {_clip(name, 80)}",
        f"Profile: {_clip(persona, MAX_PERSONA_CHARS)}",
        f"Bio: {_clip(bio, MAX_BIO_CHARS)}",
        f"Current local time: {simulated_hour:02d}:00",
        "",
        "VISIBLE FEED (posts this user sees right now)",
    ]
    post_ids: List[int] = []
    if not posts:
        lines.append("(the feed is empty)")
    for i, post in enumerate(posts, start=1):
        post_id = post.get("post_id")
        content = _clip(post.get("content", ""), MAX_POST_CHARS)
        likes = post.get("num_likes", 0)
        author = post.get("user_name") or f"user {post.get('user_id', '?')}"
        lines.append(f"post_{i}: @{author} (likes: {likes}) {content}")
        if post_id is not None:
            post_ids.append(int(post_id))
    return "\n".join(lines), post_ids


def extract_persona(agent: Any) -> Tuple[str, str, str]:
    """Extract (name, persona, bio) from an OASIS SocialAgent."""
    user_info = getattr(agent, "user_info", None)
    name = getattr(user_info, "name", "") or ""
    bio = getattr(user_info, "description", "") or ""
    persona = ""
    try:
        profile = getattr(user_info, "profile", None) or {}
        other = (profile or {}).get("other_info", {})
        persona = other.get("user_profile", "") or other.get("persona", "")
    except Exception:
        pass
    if not persona:
        persona = bio
    return str(name), str(persona), str(bio)
