"""
A container layer at the tool level.
Each tool receive the ContainerEnv class (started).
The tool can then use it to execute itself, depending on its policy.
Tools can be forced to execute inside ContainerEnv (shell, python),
or can just ignore ContainerEnv (wiki).
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, cast

from loguru import logger

DEFAULT_CONTAINER_ENV_IMAGE = "python:3.13-slim"
DEFAULT_CONTAINER_WORKDIR = "/tmp"
DEFAULT_CONTAINER_MOUNT_ROOT = "/tmp"


@dataclass(frozen=True)
class ContainerMount:
    host_path: str | Path
    container_path: str

    def as_volume_spec(self) -> dict[str, str]:
        return {"bind": self.container_path, "mode": "rw"}


class ContainerEnv:
    """Long-lived container used by tools during an agent run."""

    def __init__(
        self,
        *,
        image: str = DEFAULT_CONTAINER_ENV_IMAGE,
        client: Any | None = None,
        auto_remove: bool = True,
    ) -> None:
        if not image:
            raise ValueError("image must not be empty")
        self.image = image
        self._client = client
        self.auto_remove = auto_remove
        self._container: Any | None = None

    def __enter__(self) -> ContainerEnv:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def start(
        self,
        mount_points: Sequence[str | Path],
        network: bool = False,
    ) -> None:
        """Start the container if needed.

        Args:
            mount_points: Host paths mounted read-write at ``/tmp/{index}``.
                Empty sequences are allowed.
            network: Enable Docker's default network when true. Otherwise the
                container starts with networking disabled.
        """
        if self._container is not None:
            return
        client = self._get_client()
        kwargs: dict[str, object] = {
            "command": ["sleep", "infinity"],
            "detach": True,
            "remove": self.auto_remove,
            "working_dir": DEFAULT_CONTAINER_WORKDIR,
            "volumes": self._volumes(mount_points),
        }
        if not network:
            kwargs["network_mode"] = "none"
        logger.info(
            "Starting tool container image={} mount_count={} network={}",
            self.image,
            len(mount_points),
            network,
        )
        self._container = client.containers.run(self.image, **kwargs)

    def exec(
        self,
        command: Sequence[str],
        *,
        input: str | None = None,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a command inside the started container."""
        if self._container is None:
            raise RuntimeError("container env has not been started")
        if not command:
            raise ValueError("command must not be empty")
        if input is not None:
            raise NotImplementedError("container env stdin is not implemented")

        exec_command = self._timeout_command(command, timeout)
        result = self._container.exec_run(
            exec_command,
            stdout=True,
            stderr=True,
            demux=True,
        )
        if hasattr(result, "exit_code") and hasattr(result, "output"):
            exit_code = int(result.exit_code)
            output = cast(object, result.output)
        else:
            result_tuple = cast(tuple[Any, object], result)
            exit_code = int(result_tuple[0])
            output = result_tuple[1]
        stdout_bytes, stderr_bytes = self._split_output(output)
        if timeout is not None and exit_code == 124:
            raise TimeoutError(f"container command timed out after {timeout}s")
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=exit_code,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
        )

    def close(self) -> None:
        if self._container is None:
            return
        container = self._container
        self._container = None
        logger.info("Stopping tool container")
        container.stop(timeout=5)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import docker

        self._client = docker.from_env()
        return self._client

    def _volumes(
        self,
        mount_points: Sequence[str | Path],
    ) -> dict[str, dict[str, str]]:
        volumes: dict[str, dict[str, str]] = {}
        for index, mount_point in enumerate(mount_points):
            host_path = str(Path(mount_point).expanduser().resolve())
            mount = ContainerMount(
                host_path=host_path,
                container_path=f"{DEFAULT_CONTAINER_MOUNT_ROOT}/{index}",
            )
            volumes[host_path] = mount.as_volume_spec()
        return volumes

    def _timeout_command(
        self,
        command: Sequence[str],
        timeout: int | None,
    ) -> list[str]:
        if timeout is None:
            return list(command)
        if timeout < 1:
            raise ValueError("timeout must be positive")
        return ["timeout", str(timeout), *command]

    def _split_output(self, output: object) -> tuple[bytes, bytes]:
        if isinstance(output, tuple):
            output_tuple = cast(tuple[object, ...], output)
            if len(output_tuple) == 2:
                return self._to_bytes(output_tuple[0]), self._to_bytes(output_tuple[1])
        if isinstance(output, bytes):
            return output, b""
        if isinstance(output, str):
            return output.encode("utf-8"), b""
        output_bytes = cast(bytes, output)
        return output_bytes, b""

    def _to_bytes(self, value: object) -> bytes:
        if value is None:
            return b""
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode("utf-8")
        return cast(bytes, value)
