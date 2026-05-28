import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from plugins.platforms.discord.bot_msg_protocol import (
    build_discord_bot_msg_v1,
    is_bot_msg_required_error,
    parse_discord_bot_msg_v1,
)


class _RetryAdapter(BasePlatformAdapter):
    @property
    def name(self):
        return "retry-stub"

    def __init__(self):
        self.calls = []

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.calls.append({"chat_id": chat_id, "content": content, "reply_to": reply_to, "metadata": metadata})
        return SendResult(
            success=False,
            error="Outbound raw mention of allowed bot 777 requires send_bot_message(...) to create a BOT_MSG v1 envelope",
        )

    async def get_chat_info(self, chat_id):
        return {}

    def _is_terminal_send_error(self, error):
        return is_bot_msg_required_error(error)


def test_protocol_body_is_opaque_to_header_looking_text():
    body = "line 1\nreply_expected: false\nkind: audit\n---\nstill body"
    envelope = build_discord_bot_msg_v1(
        recipient_bot_id="777",
        body=body,
        reply_expected=True,
        kind="handoff",
        correlation_id="corr-1",
    )

    parsed = parse_discord_bot_msg_v1(envelope, "777")

    assert parsed is not None
    assert parsed["reply_expected"] is True
    assert parsed["kind"] == "handoff"
    assert parsed["body"] == body


@pytest.mark.asyncio
async def test_send_with_retry_treats_bot_msg_required_error_as_terminal():
    adapter = _RetryAdapter()

    result = await adapter._send_with_retry("555", "plain ping <@777>")

    assert result.success is False
    assert is_bot_msg_required_error(result.error)
    assert len(adapter.calls) == 1
    assert not adapter.calls[0]["content"].startswith("(Response formatting failed")


def test_send_message_rejects_raw_allowed_discord_bot_mention(monkeypatch):
    import tools.send_message_tool as smt
    from gateway.config import Platform

    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "777")

    class FakeConfig:
        platforms = {Platform.DISCORD: PlatformConfig(enabled=True, token="***")}

        def get_home_channel(self, platform):
            return None

    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: FakeConfig())

    result = json.loads(smt._handle_send({"target": "discord:555", "message": "bad <@777>"}))

    assert "error" in result
    assert "send_bot_message" in result["error"]


def test_send_bot_message_requires_recipient_allowlist(monkeypatch):
    import tools.send_message_tool as smt
    from gateway.config import Platform

    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "888")

    class FakeConfig:
        platforms = {Platform.DISCORD: PlatformConfig(enabled=True, token="***")}

        def get_home_channel(self, platform):
            return None

    async def fake_send(*args, **kwargs):  # pragma: no cover - must not be reached
        return {"success": True, "message_id": "sent"}

    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: FakeConfig())
    monkeypatch.setattr(smt, "_send_bot_message_to_discord", fake_send)

    result = json.loads(
        smt._handle_send_bot_message(
            {
                "target": "discord:555",
                "recipient_bot_id": "777",
                "kind": "status",
                "reply_expected": False,
                "body": "body",
                "correlation_id": "corr-1",
            }
        )
    )

    assert "error" in result
    assert "allowlisted" in result["error"]


def test_send_bot_message_idempotency_returns_existing_delivery(monkeypatch, tmp_path):
    import tools.send_message_tool as smt
    from gateway.config import Platform

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "777")
    calls = []

    class FakeConfig:
        platforms = {Platform.DISCORD: PlatformConfig(enabled=True, token="***")}

        def get_home_channel(self, platform):
            return None

    async def fake_send(*args, **kwargs):
        calls.append(kwargs)
        return {"success": True, "message_id": "m-1"}

    payload = {
        "target": "discord:555",
        "recipient_bot_id": "777",
        "kind": "status",
        "reply_expected": False,
        "body": "body",
        "correlation_id": "corr-1",
    }
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: FakeConfig())
    monkeypatch.setattr(smt, "_send_bot_message_to_discord", fake_send)

    first = json.loads(smt._handle_send_bot_message(dict(payload)))
    second = json.loads(smt._handle_send_bot_message(dict(payload)))

    assert first["success"] is True
    assert first["message_id"] == "m-1"
    assert second["success"] is True
    assert second["skipped"] is True
    assert second["message_id"] == "m-1"
    assert len(calls) == 1


def test_send_bot_message_idempotency_survives_omitted_correlation_id(monkeypatch, tmp_path):
    import tools.send_message_tool as smt
    from gateway.config import Platform

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "777")
    calls = []

    class FakeConfig:
        platforms = {Platform.DISCORD: PlatformConfig(enabled=True, token="***")}

        def get_home_channel(self, platform):
            return None

    async def fake_send(*args, **kwargs):
        calls.append(kwargs)
        return {"success": True, "message_id": f"m-{len(calls)}"}

    payload = {
        "target": "discord:555",
        "recipient_bot_id": "777",
        "kind": "status",
        "reply_expected": False,
        "body": "body",
    }
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: FakeConfig())
    monkeypatch.setattr(smt, "_send_bot_message_to_discord", fake_send)

    first = json.loads(smt._handle_send_bot_message(dict(payload)))
    second = json.loads(smt._handle_send_bot_message(dict(payload)))

    assert first["success"] is True
    assert first["message_id"] == "m-1"
    assert first["correlation_id"].startswith("botmsg-")
    assert second["success"] is True
    assert second["skipped"] is True
    assert second["message_id"] == "m-1"
    assert second["correlation_id"] == first["correlation_id"]
    assert len(calls) == 1


def test_send_bot_message_rejects_non_numeric_reply_to(monkeypatch, tmp_path):
    import tools.send_message_tool as smt
    from gateway.config import Platform

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "777")

    class FakeConfig:
        platforms = {Platform.DISCORD: PlatformConfig(enabled=True, token="***")}

        def get_home_channel(self, platform):
            return None

    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: FakeConfig())
    result = json.loads(
        smt._handle_send_bot_message(
            {
                "target": "discord:555",
                "recipient_bot_id": "777",
                "kind": "status",
                "reply_expected": False,
                "body": "body",
                "reply_to": "not-a-snowflake",
            }
        )
    )

    assert "error" in result
    assert "reply_to" in result["error"]


def test_send_bot_message_idempotency_key_includes_protocol_context(monkeypatch, tmp_path):
    import tools.send_message_tool as smt
    from gateway.config import Platform

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "777")
    calls = []

    class FakeConfig:
        platforms = {Platform.DISCORD: PlatformConfig(enabled=True, token="***")}

        def get_home_channel(self, platform):
            return None

    async def fake_send(*args, **kwargs):
        calls.append(kwargs)
        return {"success": True, "message_id": f"m-{len(calls)}"}

    payload = {
        "target": "discord:555",
        "recipient_bot_id": "777",
        "kind": "status",
        "reply_expected": False,
        "body": "body",
        "correlation_id": "corr-1",
    }
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: FakeConfig())
    monkeypatch.setattr(smt, "_send_bot_message_to_discord", fake_send)

    first = json.loads(smt._handle_send_bot_message(dict(payload)))
    changed_reply = dict(payload, reply_expected=True, correlation_id="corr-2")
    second = json.loads(smt._handle_send_bot_message(changed_reply))
    third = json.loads(smt._handle_send_bot_message(dict(payload, reply_to="123")))

    assert first["message_id"] == "m-1"
    assert second["message_id"] == "m-2"
    assert third["message_id"] == "m-3"
    assert len(calls) == 3


def test_discord_adapter_marks_bot_msg_required_as_terminal_send_error():
    from gateway.platforms.discord import DiscordAdapter

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))

    assert adapter._is_terminal_send_error(
        "Outbound raw mention of allowed bot 777 requires send_bot_message(...) to create a BOT_MSG v1 envelope"
    ) is True
    assert adapter._is_terminal_send_error("ordinary failure") is False


def test_inbound_bot_msg_rejects_body_over_configured_cap(monkeypatch):
    from gateway.platforms.discord import DiscordAdapter

    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "12345")
    monkeypatch.setenv("DISCORD_BOT_MSG_MAX_BODY_CHARS", "4")
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=99999))
    bot = SimpleNamespace(bot=True, id=12345)
    content = build_discord_bot_msg_v1(
        recipient_bot_id="99999",
        body="12345",
        reply_expected=True,
        kind="status",
        correlation_id="corr-1",
    )
    msg = SimpleNamespace(author=bot, content=content, channel=SimpleNamespace(id=222), id=1)

    assert adapter._should_accept_bot_message(msg, "mentions") is False
    assert adapter._should_react_malformed_bot_message(msg) is False


def test_malformed_bot_msg_reaction_only_for_invalid_envelope(monkeypatch):
    from gateway.platforms.discord import DiscordAdapter

    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "12345")
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=99999))
    bot = SimpleNamespace(bot=True, id=12345)

    malformed = SimpleNamespace(author=bot, content="<@99999> free-form raw mention")
    valid_but_rejected = SimpleNamespace(
        author=bot,
        content=build_discord_bot_msg_v1(
            recipient_bot_id="99999",
            body="too long for a cap, but syntactically valid",
            reply_expected=True,
            kind="status",
            correlation_id="corr-oversize",
        ),
    )

    assert adapter._should_react_malformed_bot_message(malformed) is True
    assert adapter._should_react_malformed_bot_message(valid_but_rejected) is False
