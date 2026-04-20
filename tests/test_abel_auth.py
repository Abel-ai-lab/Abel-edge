"""Tests for Abel auth and credential helpers."""

import os

import causal_edge.plugins.abel.credentials as credentials_module
from causal_edge.plugins.abel.auth import login_with_oauth
from causal_edge.plugins.abel.credentials import (
    MissingAbelApiKeyError,
    persist_env_value,
    require_api_key,
    resolve_api_key,
    resolve_api_key_record,
    resolve_auth_base_url,
    resolve_cap_base_url,
)


def test_resolve_api_key_prefers_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ABEL_API_KEY", "Bearer abel_env")
    (tmp_path / ".env").write_text("ABEL_API_KEY=abel_file\n", encoding="utf-8")

    api_key = resolve_api_key(env_path=tmp_path / ".env")

    assert api_key == "abel_env"


def test_resolve_api_key_record_reports_env_source(monkeypatch, tmp_path):
    monkeypatch.setenv("ABEL_API_KEY", "Bearer abel_env")

    record = resolve_api_key_record(env_path=tmp_path / ".env")

    assert record == {
        "api_key": "abel_env",
        "source": "env_var",
        "path": None,
    }


def test_resolve_api_key_reads_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv("ABEL_API_KEY", raising=False)
    monkeypatch.delenv("CAP_API_KEY", raising=False)
    (tmp_path / ".env").write_text("CAP_API_KEY=Bearer cap_file\n", encoding="utf-8")

    api_key = resolve_api_key(env_path=tmp_path / ".env")

    assert api_key == "cap_file"


def test_resolve_api_key_reads_explicit_auth_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv("ABEL_API_KEY", raising=False)
    monkeypatch.delenv("CAP_API_KEY", raising=False)
    auth_env = tmp_path / "shared" / ".env.skill"
    auth_env.parent.mkdir(parents=True)
    auth_env.write_text("ABEL_API_KEY=abel_shared\n", encoding="utf-8")
    monkeypatch.setenv("ABEL_AUTH_ENV_FILE", str(auth_env))

    api_key = resolve_api_key(env_path=tmp_path / ".env")

    assert api_key == "abel_shared"


def test_resolve_api_key_record_reports_shared_auth_file(monkeypatch, tmp_path):
    monkeypatch.delenv("ABEL_API_KEY", raising=False)
    monkeypatch.delenv("CAP_API_KEY", raising=False)
    auth_env = tmp_path / "shared" / ".env.skill"
    auth_env.parent.mkdir(parents=True)
    auth_env.write_text("ABEL_API_KEY=abel_shared\n", encoding="utf-8")
    monkeypatch.setenv("ABEL_AUTH_ENV_FILE", str(auth_env))

    record = resolve_api_key_record(env_path=tmp_path / ".env")

    assert record == {
        "api_key": "abel_shared",
        "source": "shared_auth_file",
        "path": str(auth_env),
    }


def test_resolve_api_key_auto_discovers_causal_abel_skill_env(monkeypatch, tmp_path):
    monkeypatch.delenv("ABEL_API_KEY", raising=False)
    monkeypatch.delenv("CAP_API_KEY", raising=False)
    monkeypatch.delenv("ABEL_AUTH_ENV_FILE", raising=False)
    project_dir = tmp_path / "project" / "workspace"
    project_dir.mkdir(parents=True)
    skill_env = tmp_path / "project" / ".agents" / "skills" / "causal-abel" / ".env.skill"
    skill_env.parent.mkdir(parents=True)
    skill_env.write_text("ABEL_API_KEY=abel_skill\n", encoding="utf-8")

    monkeypatch.chdir(project_dir)
    api_key = resolve_api_key(env_path=project_dir / ".env")

    assert api_key == "abel_skill"


def test_resolve_api_key_auto_discovers_opencode_global_skill_env(monkeypatch, tmp_path):
    monkeypatch.delenv("ABEL_API_KEY", raising=False)
    monkeypatch.delenv("CAP_API_KEY", raising=False)
    monkeypatch.delenv("ABEL_AUTH_ENV_FILE", raising=False)
    home_dir = tmp_path / "home"
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True)
    skill_env = home_dir / ".config" / "opencode" / "skills" / "causal-abel" / ".env.skill"
    skill_env.parent.mkdir(parents=True)
    skill_env.write_text("ABEL_API_KEY=abel_opencode\n", encoding="utf-8")
    monkeypatch.setattr(credentials_module.Path, "home", lambda: home_dir)

    api_key = resolve_api_key(env_path=project_dir / ".env")

    assert api_key == "abel_opencode"


def test_resolve_api_key_auto_discovers_codex_global_skill_env(monkeypatch, tmp_path):
    monkeypatch.delenv("ABEL_API_KEY", raising=False)
    monkeypatch.delenv("CAP_API_KEY", raising=False)
    monkeypatch.delenv("ABEL_AUTH_ENV_FILE", raising=False)
    home_dir = tmp_path / "home"
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True)
    skill_env = home_dir / ".codex" / "skills" / "causal-abel" / ".env.skill"
    skill_env.parent.mkdir(parents=True)
    skill_env.write_text("ABEL_API_KEY=abel_codex\n", encoding="utf-8")
    monkeypatch.setattr(credentials_module.Path, "home", lambda: home_dir)

    api_key = resolve_api_key(env_path=project_dir / ".env")

    assert api_key == "abel_codex"


def test_resolve_api_key_prefers_project_dotenv_over_skill_env(monkeypatch, tmp_path):
    monkeypatch.delenv("ABEL_API_KEY", raising=False)
    monkeypatch.delenv("CAP_API_KEY", raising=False)
    monkeypatch.delenv("ABEL_AUTH_ENV_FILE", raising=False)
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / ".env").write_text("ABEL_API_KEY=abel_project\n", encoding="utf-8")
    skill_env = project_dir / ".agents" / "skills" / "causal-abel" / ".env.skill"
    skill_env.parent.mkdir(parents=True)
    skill_env.write_text("ABEL_API_KEY=abel_skill\n", encoding="utf-8")

    monkeypatch.chdir(project_dir)
    api_key = resolve_api_key(env_path=project_dir / ".env")

    assert api_key == "abel_project"


def test_resolve_api_key_record_reports_project_env_source(monkeypatch, tmp_path):
    monkeypatch.delenv("ABEL_API_KEY", raising=False)
    monkeypatch.delenv("CAP_API_KEY", raising=False)
    monkeypatch.delenv("ABEL_AUTH_ENV_FILE", raising=False)
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    env_path = project_dir / ".env"
    env_path.write_text("ABEL_API_KEY=abel_project\n", encoding="utf-8")

    record = resolve_api_key_record(env_path=env_path)

    assert record == {
        "api_key": "abel_project",
        "source": "project_env",
        "path": str(env_path.resolve()),
    }


def test_resolve_api_key_prefers_explicit_auth_env_file_over_global_skill(monkeypatch, tmp_path):
    monkeypatch.delenv("ABEL_API_KEY", raising=False)
    monkeypatch.delenv("CAP_API_KEY", raising=False)
    home_dir = tmp_path / "home"
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True)
    shared_env = tmp_path / "shared" / ".env.skill"
    shared_env.parent.mkdir(parents=True)
    shared_env.write_text("ABEL_API_KEY=abel_explicit\n", encoding="utf-8")
    skill_env = home_dir / ".config" / "opencode" / "skills" / "causal-abel" / ".env.skill"
    skill_env.parent.mkdir(parents=True)
    skill_env.write_text("ABEL_API_KEY=abel_global\n", encoding="utf-8")
    monkeypatch.setenv("ABEL_AUTH_ENV_FILE", str(shared_env))
    monkeypatch.setattr(credentials_module.Path, "home", lambda: home_dir)

    api_key = resolve_api_key(env_path=project_dir / ".env")

    assert api_key == "abel_explicit"


def test_require_api_key_raises_when_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("ABEL_API_KEY", raising=False)
    monkeypatch.delenv("CAP_API_KEY", raising=False)
    monkeypatch.delenv("ABEL_AUTH_ENV_FILE", raising=False)
    monkeypatch.setattr(credentials_module.Path, "home", lambda: tmp_path / "home")

    try:
        require_api_key(env_path=tmp_path / ".env")
    except MissingAbelApiKeyError as exc:
        assert "ABEL_API_KEY" in str(exc)
        assert "ABEL_AUTH_ENV_FILE" in str(exc)
    else:
        raise AssertionError("Expected MissingAbelApiKeyError")


def test_require_api_key_mentions_auth_status_when_skill_installed(monkeypatch, tmp_path):
    monkeypatch.delenv("ABEL_API_KEY", raising=False)
    monkeypatch.delenv("CAP_API_KEY", raising=False)
    monkeypatch.delenv("ABEL_AUTH_ENV_FILE", raising=False)
    monkeypatch.setattr(credentials_module.Path, "home", lambda: tmp_path / "home")
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    skill_dir = project_dir / ".agents" / "skills" / "causal-abel"
    skill_dir.mkdir(parents=True)

    monkeypatch.chdir(project_dir)
    try:
        require_api_key(env_path=project_dir / ".env")
    except MissingAbelApiKeyError as exc:
        assert "auth-status --compact" in str(exc)
        assert "causal-abel" in str(exc)
    else:
        raise AssertionError("Expected MissingAbelApiKeyError")


def test_resolve_cap_base_url_uses_public_default(monkeypatch, tmp_path):
    monkeypatch.delenv("ABEL_CAP_BASE_URL", raising=False)

    base_url = resolve_cap_base_url(env_path=tmp_path / ".env")

    assert base_url == "https://cap.abel.ai/api"


def test_resolve_cap_base_url_reads_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ABEL_CAP_BASE_URL", "https://cap.custom.abel.ai/api/")

    base_url = resolve_cap_base_url(env_path=tmp_path / ".env")

    assert base_url == "https://cap.custom.abel.ai/api"


def test_resolve_cap_base_url_reads_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv("ABEL_CAP_BASE_URL", raising=False)
    (tmp_path / ".env").write_text(
        "ABEL_CAP_BASE_URL=https://cap.file.abel.ai/api/\n", encoding="utf-8"
    )

    base_url = resolve_cap_base_url(env_path=tmp_path / ".env")

    assert base_url == "https://cap.file.abel.ai/api"


def test_resolve_auth_base_url_uses_public_default(monkeypatch, tmp_path):
    monkeypatch.delenv("ABEL_AUTH_BASE_URL", raising=False)

    base_url = resolve_auth_base_url(env_path=tmp_path / ".env")

    assert base_url == "https://api.abel.ai/echo"


def test_resolve_auth_base_url_reads_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv("ABEL_AUTH_BASE_URL", raising=False)
    (tmp_path / ".env").write_text(
        "ABEL_AUTH_BASE_URL=https://api.custom.abel.ai/echo/\n", encoding="utf-8"
    )

    base_url = resolve_auth_base_url(env_path=tmp_path / ".env")

    assert base_url == "https://api.custom.abel.ai/echo"


def test_resolve_api_key_does_not_mutate_process_env(monkeypatch, tmp_path):
    monkeypatch.delenv("ABEL_API_KEY", raising=False)
    monkeypatch.delenv("CAP_API_KEY", raising=False)
    monkeypatch.delenv("ABEL_AUTH_ENV_FILE", raising=False)
    (tmp_path / ".env").write_text("ABEL_API_KEY=abel_file\n", encoding="utf-8")

    api_key = resolve_api_key(env_path=tmp_path / ".env")

    assert api_key == "abel_file"
    assert os.getenv("ABEL_API_KEY") is None


def test_persist_env_value_adds_and_updates_key(tmp_path):
    env_path = tmp_path / ".env"

    persist_env_value(env_path=env_path, key="ABEL_API_KEY", value="first")
    persist_env_value(env_path=env_path, key="ABEL_API_KEY", value="second")

    assert env_path.read_text(encoding="utf-8") == "ABEL_API_KEY=second\n"


def test_persist_env_value_creates_parent_directory(tmp_path):
    env_path = tmp_path / "config" / ".env"

    persist_env_value(env_path=env_path, key="ABEL_API_KEY", value="created")

    assert env_path.read_text(encoding="utf-8") == "ABEL_API_KEY=created\n"


def test_login_with_oauth_persists_api_key(tmp_path, monkeypatch):
    class StubSession:
        def __init__(self):
            self.calls = []
            self.poll_count = 0

        def get(self, url, timeout=20):
            self.calls.append(url)
            if url.endswith("/authorize/agent"):
                return StubResponse(
                    {
                        "data": {
                            "authUrl": "https://example.com/auth",
                            "pollToken": "poll-123",
                        }
                    }
                )
            self.poll_count += 1
            if self.poll_count == 1:
                return StubResponse({"data": {"status": "pending"}})
            return StubResponse({"data": {"status": "authorized", "apiKey": "abel_key"}})

    class StubResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    opened_urls = []
    notices = []
    handoffs = []
    pending = []
    monkeypatch.setattr(credentials_module.Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr("webbrowser.open", lambda url: opened_urls.append(url) or True)
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    result = login_with_oauth(
        env_path=str(tmp_path / ".env"),
        session=StubSession(),
        notify=notices.append,
        on_handoff=handoffs.append,
        on_pending=pending.append,
        timeout_seconds=5,
    )

    assert result["status"] == "authorized"
    assert result["api_key"] == "abel_key"
    assert result["opened_browser"] is True
    assert opened_urls == ["https://example.com/auth"]
    assert "https://example.com/auth" in notices[0]
    assert handoffs == [
        {
            "status": "awaiting_authorization",
            "auth_url": "https://example.com/auth",
            "env_path": str(tmp_path / ".env"),
            "opened_browser": True,
            "result_url": None,
            "poll_token": "poll-123",
            "poll_interval_seconds": 2.0,
            "timeout_seconds": 5,
        }
    ]
    assert pending == [
        {
            "status": "waiting_for_authorization",
            "polls": 1,
            "poll_interval_seconds": 2.0,
            "timeout_seconds": 5,
        }
    ]
    assert "ABEL_API_KEY=abel_key" in (tmp_path / ".env").read_text(encoding="utf-8")


def test_login_with_oauth_skips_browser_when_disabled(tmp_path, monkeypatch):
    class StubSession:
        def get(self, url, timeout=20):
            if url.endswith("/authorize/agent"):
                return StubResponse(
                    {
                        "data": {
                            "authUrl": "https://example.com/auth",
                            "pollToken": "poll-123",
                        }
                    }
                )
            return StubResponse({"data": {"status": "authorized", "apiKey": "abel_key"}})

    class StubResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    opened_urls = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened_urls.append(url) or True)

    result = login_with_oauth(
        env_path=str(tmp_path / ".env"),
        session=StubSession(),
        open_browser=False,
        timeout_seconds=5,
    )

    assert result["opened_browser"] is False
    assert opened_urls == []


def test_login_with_oauth_returns_existing_key(monkeypatch, tmp_path):
    monkeypatch.setenv("ABEL_API_KEY", "abel_existing")

    result = login_with_oauth(env_path=str(tmp_path / ".env"))

    assert result["status"] == "already_configured"
    assert result["api_key"] == "abel_existing"
    assert result["source"] == "env_var"
    assert result["source_path"] is None
