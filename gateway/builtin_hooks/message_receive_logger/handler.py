"""
message:receive logger — proof-of-concept builtin hook.

Logs every incoming message before dispatch. Disabled by default; enable via:

    gateway:
      hooks:
        message_receive_logger:
          enabled: true

Returns ``None`` (allow) for every event — never intercepts.
"""

import logging

logger = logging.getLogger("hermes.hooks.message_receive_logger")


def handle(event_type: str, context: dict) -> None:
    """Log the incoming message and return None (allow dispatch)."""
    text = context.get("text", "")
    platform = context.get("source", {}).get("platform", "unknown")
    session_key = context.get("session_key", "?")
    is_command = context.get("is_command", False)

    logger.debug(
        "message:receive [%s] session=%s cmd=%s text=%.80s",
        platform, session_key, is_command, text,
    )
    # Return None → decision is "allow" → normal dispatch continues.
    return None
