"""
Approval text-intercept — resolve pending approvals via plain text.

When the agent blocks on a dangerous-command approval gate, this hook
intercepts plain-text responses (yes, no, session, always, deny, etc.)
and resolves the approval without requiring /approve or /deny slash
commands.

This is critical for Matrix clients like weechat-matrix where slash
commands are intercepted by the client and reactions are unsupported.

Returns ``handled`` when a pending approval is resolved, otherwise
returns ``None`` (allow) so normal dispatch continues.
"""

import logging

logger = logging.getLogger("hermes.hooks.approval_text_intercept")


def handle(event_type: str, context: dict):
    """Check for pending approvals and resolve via plain-text keywords.

    Returns:
        None — no pending approval or text didn't match → allow dispatch
        {"decision": "handled", "message": str} — approval resolved
    """
    text = (context.get("text") or "").strip().lower()
    if not text:
        return None

    session_key = context.get("session_key")
    if not session_key:
        return None

    # Import gateway internals — this is a builtin hook with full access.
    try:
        from tools.approval import has_blocking_approval, resolve_gateway_approval
        from gateway.config import DEFAULT_APPROVAL_KEYWORDS, _merge_approval_keywords
    except ImportError:
        logger.debug("approval modules not available, skipping")
        return None

    # Bail early if no pending approval exists for this session.
    if not has_blocking_approval(session_key):
        return None

    # Load configured keywords (defaults + user overrides).
    keywords = _get_keywords()

    # Match the response against known approval keywords.
    if text not in keywords:
        return None

    choice = keywords[text]

    if choice == "deny":
        resolve_gateway_approval(session_key, "deny")
        logger.info("Approval denied via text: %r (session=%s)", text, session_key)
        return {
            "decision": "handled",
            "message": "❎ Approval denied.",
        }

    # "once", "session", "always" → resolve approval at that level.
    resolve_gateway_approval(session_key, choice)
    logger.info(
        "Approval granted via text: %r → %s (session=%s)",
        text, choice, session_key,
    )
    labels = {"once": "this command", "session": "this session", "always": "permanently"}
    return {
        "decision": "handled",
        "message": f"✅ Approved {labels.get(choice, choice)}.",
    }


def _get_keywords() -> dict:
    """Resolve the effective keyword map (defaults + user overrides).

    Caches the result on the function object to avoid re-reading config
    on every message.
    """
    if not hasattr(_get_keywords, "_cache"):
        try:
            from gateway.config import load_gateway_config
            cfg = load_gateway_config()
            user_kw = cfg.approval_keywords if hasattr(cfg, "approval_keywords") else {}
        except Exception:
            user_kw = {}
        from gateway.config import DEFAULT_APPROVAL_KEYWORDS, _merge_approval_keywords
        _get_keywords._cache = _merge_approval_keywords(DEFAULT_APPROVAL_KEYWORDS, user_kw)
    return _get_keywords._cache
