"""Message-receive logger — proof-of-concept builtin hook.

Logs every incoming message at DEBUG level via the message:receive
event. Demonstrates the hook protocol; can be disabled by deleting
the HOOK.yaml or handler.py.
"""

import logging

logger = logging.getLogger("hermes.hooks.message_receive_logger")


def handle(event_type: str, context: dict):
    """Log every incoming message."""
    text = (context.get("text") or "")[:200]
    source = context.get("source", {})
    platform = source.get("platform", "?")
    chat_id = source.get("chat_id", "?")
    session_key = context.get("session_key", "?")
    is_command = context.get("is_command", False)
    logger.debug(
        "[%s] message:receive chat=%s session=%s cmd=%s text=%r",
        platform, chat_id, session_key, is_command, text,
    )
