from pathlib import Path
from typing import cast

import pytest

from ..LLLM.container_env import (
    DEFAULT_CONTAINER_ENV_IMAGE,
    ContainerEnv,
)


class FakeExecResult:
    def __init__(self, exit_code: int, output: object) -> None:
        self.exit_code = exit_code
        self.output = output


class FakeContainer:
    def __init__(self) -> None:
        self.stopped = False
        self.exec_calls: list[dict[str, object]] = []
        self.exec_result = FakeExecResult(0, (b"out", b"err"))

    def exec_run(self, command: list[str], **kwargs: object) -> FakeExecResult:
        self.exec_calls.append({"command": command, **kwargs})
        return self.exec_result

    def stop(self, *, timeout: int) -> None:
        del timeout
        self.stopped = True


class FakeContainers:
    def __init__(self, container: FakeContainer) -> None:
        self.container = container
        self.calls: list[dict[str, object]] = []

    def run(self, image: str, **kwargs: object) -> FakeContainer:
        self.calls.append({"image": image, **kwargs})
        return self.container


class FakeDockerClient:
    def __init__(self) -> None:
        self.container = FakeContainer()
        self.containers = FakeContainers(self.container)


def test_container_env_start_creates_container_with_mounts(tmp_path: Path) -> None:
    client = FakeDockerClient()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    env = ContainerEnv(client=client)
    env.start([first, second], network=False)

    call = client.containers.calls[0]
    assert call["image"] == DEFAULT_CONTAINER_ENV_IMAGE
    assert call["command"] == ["sleep", "infinity"]
    assert call["detach"] is True
    assert call["remove"] is True
    assert call["working_dir"] == "/tmp"
    assert call["network_mode"] == "none"
    assert cast(dict[str, dict[str, str]], call["volumes"]) == {
        str(first.resolve()): {
            "bind": "/tmp/0",
            "mode": "rw",
        },
        str(second.resolve()): {
            "bind": "/tmp/1",
            "mode": "rw",
        },
    }


def test_container_env_start_allows_empty_mount_points() -> None:
    client = FakeDockerClient()

    env = ContainerEnv(client=client)
    env.start([])

    assert client.containers.calls[0]["volumes"] == {}


def test_container_env_start_with_network_uses_default_network() -> None:
    client = FakeDockerClient()

    env = ContainerEnv(client=client)
    env.start([], network=True)

    assert "network_mode" not in client.containers.calls[0]


def test_container_env_start_is_idempotent() -> None:
    client = FakeDockerClient()
    env = ContainerEnv(client=client)

    env.start([])
    env.start([])

    assert len(client.containers.calls) == 1


def test_container_env_exec_returns_completed_process() -> None:
    client = FakeDockerClient()
    client.container.exec_result = FakeExecResult(7, (b"hello\n", b"bad\n"))
    env = ContainerEnv(client=client)
    env.start([])

    result = env.exec(["python", "-c", "print(1)"], timeout=3)

    assert result.args == ["python", "-c", "print(1)"]
    assert result.returncode == 7
    assert result.stdout == "hello\n"
    assert result.stderr == "bad\n"
    assert client.container.exec_calls == [
        {
            "command": ["timeout", "3", "python", "-c", "print(1)"],
            "stdout": True,
            "stderr": True,
            "demux": True,
        }
    ]


def test_container_env_exec_translates_timeout_exit_code() -> None:
    client = FakeDockerClient()
    client.container.exec_result = FakeExecResult(124, (b"", b""))
    env = ContainerEnv(client=client)
    env.start([])

    with pytest.raises(TimeoutError, match="timed out"):
        env.exec(["python", "-c", "while True: pass"], timeout=3)


def test_container_env_exec_before_start_raises() -> None:
    env = ContainerEnv(client=FakeDockerClient())

    with pytest.raises(RuntimeError, match="not been started"):
        env.exec(["python"])


def test_container_env_exec_rejects_empty_command() -> None:
    env = ContainerEnv(client=FakeDockerClient())
    env.start([])

    with pytest.raises(ValueError, match="command"):
        env.exec([])


def test_container_env_exec_rejects_stdin_for_now() -> None:
    env = ContainerEnv(client=FakeDockerClient())
    env.start([])

    with pytest.raises(NotImplementedError, match="stdin"):
        env.exec(["python"], input="")


def test_container_env_close_stops_container() -> None:
    client = FakeDockerClient()
    env = ContainerEnv(client=client)
    env.start([])

    env.close()

    assert client.container.stopped


def test_container_env_can_override_image() -> None:
    client = FakeDockerClient()

    env = ContainerEnv(image="custom", client=client, auto_remove=False)
    env.start([])

    assert client.containers.calls[0]["image"] == "custom"
    assert client.containers.calls[0]["remove"] is False
