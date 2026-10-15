# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""A fixed-session model channel for candidates with no network or host mounts."""

from __future__ import annotations

import asyncio
import base64
import json
from urllib.parse import urlparse

from rl.agentic.core.program_runner import request_gateway_json
from rl.agentic.envs.docker_workspace import DockerWorkspace


# This stdlib-only endpoint runs inside the candidate. Only the controller knows
# the upstream address and credentials; untrusted frames contain body bytes only.
_ENDPOINT = r'''
import base64
import json
import select
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

request_limit, response_limit, timeout = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
paths = ("/responses", "/v1/responses")

class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(timeout)

    def do_POST(self):
        if self.path not in paths:
            self.send_error(403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > request_limit or self.headers.get("Transfer-Encoding"):
                raise ValueError("Invalid request size")
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("Incomplete request")
            print(json.dumps({"body": base64.b64encode(body).decode()}), flush=True)
            line = sys.stdin.buffer.readline(response_limit * 2 + 4096)
            reply = json.loads(line)
            payload = base64.b64decode(reply["body"], validate=True)
            if len(payload) > response_limit:
                raise ValueError("Response exceeds limit")
            self.send_response(reply["status"])
            self.send_header("Content-Type", reply["content_type"])
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
        except Exception:
            print(json.dumps({"delivered": False}), flush=True)
            self.close_connection = True
            return
        print(json.dumps({"delivered": True}), flush=True)

    def log_message(self, *args):
        pass

server = HTTPServer(("127.0.0.1", 0), Handler)
server.timeout = 0.1
print(json.dumps({"port": server.server_port}), flush=True)
try:
    while True:
        if select.select([sys.stdin.buffer], [], [], 0)[0]:
            break
        server.handle_request()
finally:
    server.server_close()
'''


class RelayError(RuntimeError):
    """A transport failure that must override an otherwise trainable model failure."""


def _budget_termination(reply: dict) -> dict | None:
    """Recognize only the fixed Gateway's structured completion-budget response."""
    if reply["status"] != 409:
        return None
    try:
        body = json.loads(base64.b64decode(reply["body"], validate=True))
    except (ValueError, UnicodeError):
        return None
    if not isinstance(body, dict) or not isinstance(body.get("error"), dict):
        return None
    termination = body.get("termination")
    if (body["error"].get("type") != "completion_budget_exhausted" or not isinstance(termination, dict)
            or set(termination) != {"reason", "limit", "completed", "policy_version"}):
        return None
    if (termination["reason"] != "max_completions"
            or any(not isinstance(termination[key], int) or isinstance(termination[key], bool)
                   for key in ("limit", "completed"))
            or termination["limit"] <= 0 or termination["completed"] != termination["limit"]
            or not isinstance(termination["policy_version"], int) or isinstance(termination["policy_version"], bool)
            or termination["policy_version"] < 0):
        return None
    return dict(termination)


class CandidateRelay:
    """Forward one fixed model route to a controller-bound Gateway session."""

    def __init__(self, workspace: DockerWorkspace, gateway_url: str, session_id: str,
                 request_timeout: float = 600, max_request_bytes: int = 8 * 1024 * 1024,
                 max_response_bytes: int = 32 * 1024 * 1024, *,
                 admin_url: str | None = None, admin_token: str | None = None) -> None:
        """Bind the target outside the candidate, excluding candidate headers and routing."""
        parsed = urlparse(gateway_url)
        if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.query or parsed.fragment:
            raise ValueError("Candidate relay requires a fixed local HTTP gateway URL")
        self.workspace = workspace
        self.target = parsed
        self.session_id = session_id
        self.timeout = request_timeout
        self.request_limit, self.response_limit = max_request_bytes, max_response_bytes
        self.admin_url, self.admin_token = admin_url or gateway_url, admin_token
        self._process = None
        self._pump_task = None
        self._stderr_task = None
        self._failure = None
        self._closing = False
        self._closed = False
        self._idle = asyncio.Event()
        self._idle.set()
        self.budget_exhausted = asyncio.Event()
        self.budget_termination: dict | None = None

    async def start(self) -> str:
        """Start an owned Docker exec channel and return the candidate loopback URL."""
        if self._process is not None:
            raise RuntimeError("Candidate relay already started")
        creation = asyncio.create_task(asyncio.create_subprocess_exec(
            *self.workspace.docker_command("exec", "-i", self.workspace.name),
            "/usr/local/python3.12.13/bin/python", "-I", "-u", "-c", _ENDPOINT,
            str(self.request_limit), str(self.response_limit), str(self.timeout),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            limit=max(self.request_limit * 2 + 4096, 65536),
        ))
        try:
            self._process = await asyncio.shield(creation)
        except asyncio.CancelledError:
            self._process = await creation
            await self.close()
            raise
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        try:
            ready = json.loads(await asyncio.wait_for(self._process.stdout.readline(), 15))
            port = ready["port"]
            if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
                raise ValueError("Invalid relay port")
            self._pump_task = asyncio.create_task(self._pump())
            return f"http://127.0.0.1:{port}"
        except BaseException:
            await self.close()
            raise

    async def _drain_stderr(self) -> None:
        """Drain bounded Docker diagnostics without exposing candidate content in errors."""
        count = 0
        while True:
            chunk = await self._process.stderr.read(4096)
            if not chunk:
                return
            count += len(chunk)
            if count > 65536:
                self._failure = RelayError("Candidate relay diagnostics exceeded their byte limit")
                return

    async def _exchange(self, body: bytes) -> dict:
        """Exchange one bounded HTTP response under an overall caller deadline."""
        reader, writer = await asyncio.open_connection(self.target.hostname, self.target.port or 80)
        try:
            path = self.target.path.rstrip("/") + "/responses"
            header = (f"POST {path} HTTP/1.0\r\nHost: {self.target.netloc}\r\n"
                      f"Authorization: Bearer {self.session_id}\r\nContent-Type: application/json\r\n"
                      f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n")
            writer.write(header.encode("ascii") + body)
            await writer.drain()
            status_line = await reader.readline()
            status = int(status_line.split()[1])
            header_bytes = len(status_line)
            content_type, content_length = "application/json", None
            while True:
                line = await reader.readline()
                header_bytes += len(line)
                if header_bytes > 16384 or not line:
                    raise RelayError("Invalid gateway response headers")
                if line == b"\r\n":
                    break
                key, value = line.decode("ascii").split(":", 1)
                if key.lower() == "content-type":
                    content_type = value.strip()
                elif key.lower() == "content-length":
                    content_length = int(value)
                elif key.lower() == "transfer-encoding":
                    raise RelayError("Unsupported gateway response framing")
            if content_length is not None and not 0 <= content_length <= self.response_limit:
                raise RelayError("Gateway response exceeds relay limit")
            payload = bytearray()
            while True:
                chunk = await reader.read(min(65536, self.response_limit + 1 - len(payload)))
                if not chunk:
                    break
                payload.extend(chunk)
                if len(payload) > self.response_limit:
                    raise RelayError("Gateway response exceeds relay limit")
            if content_length is not None and len(payload) != content_length:
                raise RelayError("Incomplete gateway response")
            return {"status": status, "content_type": content_type,
                    "body": base64.b64encode(payload).decode("ascii")}
        finally:
            writer.close()
            await asyncio.wait_for(writer.wait_closed(), 2)

    async def _pump(self) -> None:
        """Accept body-only frames, then require the local HTTP delivery acknowledgement."""
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    if self._closing:
                        return
                    raise RelayError("Candidate model channel exited unexpectedly")
                self._idle.clear()
                frame = json.loads(line)
                if set(frame) != {"body"}:
                    raise RelayError("Invalid candidate request frame")
                body = base64.b64decode(frame["body"], validate=True)
                if not 0 < len(body) <= self.request_limit:
                    raise RelayError("Candidate request exceeds relay limit")
                reply = await asyncio.wait_for(self._exchange(body), self.timeout)
                self._process.stdin.write(json.dumps(reply).encode("utf-8") + b"\n")
                await asyncio.wait_for(self._process.stdin.drain(), self.timeout)
                ack = json.loads(await asyncio.wait_for(self._process.stdout.readline(), self.timeout))
                if ack != {"delivered": True}:
                    raise RelayError("Candidate disconnected before model response delivery")
                self._idle.set()
                termination = _budget_termination(reply)
                if termination is not None and self._failure is None:
                    # The endpoint acknowledged delivery, not CLI consumption; close still checks transport errors.
                    self.budget_termination = termination
                    self.budget_exhausted.set()
                    return
        # All relay-task failures must reach the controller, including third-party transport errors.
        except Exception as error:  # pylint: disable=broad-exception-caught
            self._failure = RelayError(f"Candidate model transport failed ({type(error).__name__}): {str(error)[:300]}")
            self._idle.set()

    async def close(self) -> None:
        """Drain delivery, stop the relay and invalidate incomplete Gateway evidence."""
        process = self._process
        if process is None or self._closed:
            return
        try:
            await asyncio.wait_for(self._idle.wait(), self.timeout)
        except asyncio.TimeoutError:
            self._failure = RelayError("Timed out draining candidate model transport")
        self._closing = True
        try:
            if process.returncode is None:
                process.stdin.write(b'{"close":true}\n')
                await asyncio.wait_for(process.stdin.drain(), 2)
                await asyncio.wait_for(process.wait(), 3)
        except (OSError, asyncio.TimeoutError):
            self._failure = self._failure or RelayError("Candidate model channel did not close cleanly")
            if process.returncode is None:
                process.kill()
                await asyncio.wait_for(process.wait(), 3)
        finally:
            tasks = [task for task in (self._pump_task, self._stderr_task) if task is not None]
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._closed = True
        if self._failure is not None:
            await asyncio.to_thread(
                request_gateway_json, "Codex", "POST",
                f"{self.admin_url}/internal/sessions/{self.session_id}/failure",
                {"reason": "Candidate model transport failed"}, min(self.timeout, 10), self.admin_token,
            )
            raise self._failure
