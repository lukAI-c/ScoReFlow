#!/usr/bin/env python3
"""Run a command in an Inspire notebook Jupyter terminal and print stdout.

This is a temporary operator tool for cases where the notebook SSH bootstrap is
broken but the JupyterLab terminal WebSocket is reachable.
"""

from __future__ import annotations

import argparse
import base64
import io
import shlex
import tarfile
import uuid
from pathlib import Path

from playwright.sync_api import sync_playwright

from inspire.platform.web.session import get_web_session
from inspire.platform.web.browser_api.playwright_notebooks import (
    _launch_browser,
    _new_context,
    open_notebook_lab,
)
from inspire.platform.web.browser_api.rtunnel import (
    _build_terminal_websocket_url,
    _create_terminal_via_api,
    _delete_terminal_via_api,
)


def build_wrapped_command(command: str, marker: str) -> str:
    quoted = shlex.quote(command)
    marker_prefix = "__CODEX_DONE_"
    marker_suffix = marker.removeprefix(marker_prefix)
    return (
        f"bash -lc {quoted}; "
        f"__codex_rc=$?; "
        f"__codex_m1={shlex.quote(marker_prefix)}; "
        f"__codex_m2={shlex.quote(marker_suffix)}; "
        f"echo \"$__codex_m1$__codex_m2:$__codex_rc\""
    )


def build_upload_command(paths: list[str], remote_dir: str) -> str:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as tar:
        for path_text in paths:
            path = Path(path_text)
            if not path.exists():
                raise FileNotFoundError(path_text)
            tar.add(path, arcname=path.as_posix())
    encoded = base64.b64encode(payload.getvalue()).decode("ascii")
    remote_q = shlex.quote(remote_dir)
    return "\n".join(
        [
            "set -e",
            f"mkdir -p {remote_q}",
            "UPLOAD_B64=/tmp/codex_scoreflow_clean_upload.tgz.b64",
            "cat > \"$UPLOAD_B64\" <<'CODEX_SCOREFLOW_UPLOAD_EOF'",
            encoded,
            "CODEX_SCOREFLOW_UPLOAD_EOF",
            f"base64 -d \"$UPLOAD_B64\" | tar -xz -C {remote_q}",
            f"find {remote_q} -maxdepth 3 -type f | sort | sed -n '1,120p'",
        ]
    )


def run_notebook_command(notebook_id: str, command: str, timeout_ms: int) -> tuple[int, str]:
    marker = f"__CODEX_DONE_{uuid.uuid4().hex}__"
    wrapped_command = build_wrapped_command(command, marker)
    stdin_payload = wrapped_command.rstrip("\r\n") + "\r"

    session = get_web_session()
    with sync_playwright() as playwright:
        browser = _launch_browser(playwright, headless=True)
        context = _new_context(browser, storage_state=session.storage_state)
        page = context.new_page()
        term_name = None
        try:
            lab_frame = open_notebook_lab(page, notebook_id=notebook_id)
            term_name = _create_terminal_via_api(context, lab_frame.url)
            if not term_name:
                raise RuntimeError("failed to create Jupyter terminal")
            ws_url = _build_terminal_websocket_url(lab_frame.url, term_name)
            result = lab_frame.evaluate(
                """
                async ({ wsUrl, stdinData, timeoutMs, marker }) => {
                  return await new Promise((resolve) => {
                    let settled = false;
                    let sent = false;
                    let socket = null;
                    let output = "";
                    const finish = (ok, rc) => {
                      if (settled) return;
                      settled = true;
                      try { if (socket) socket.close(); } catch (_) {}
                      resolve({ ok, rc, output });
                    };
                    const timer = setTimeout(() => finish(false, 124), timeoutMs);
                    const sendInput = () => {
                      if (sent || settled) return;
                      sent = true;
                      const chunk = 2048;
                      let idx = 0;
                      const sendNext = () => {
                        if (settled) return;
                        try {
                          socket.send(JSON.stringify(["stdin", stdinData.slice(idx, idx + chunk)]));
                        } catch (_) {
                          clearTimeout(timer);
                          finish(false, 125);
                          return;
                        }
                        idx += chunk;
                        if (idx < stdinData.length) setTimeout(sendNext, 50);
                      };
                      sendNext();
                    };
                    try {
                      socket = new WebSocket(wsUrl);
                    } catch (_) {
                      clearTimeout(timer);
                      finish(false, 126);
                      return;
                    }
                    let promptBuf = "";
                    const promptRe = /[$#]\\s*$/;
                    socket.addEventListener("open", () => setTimeout(sendInput, 3000));
                    socket.addEventListener("message", (ev) => {
                      try {
                        const msg = JSON.parse(ev.data);
                        if (Array.isArray(msg) && msg[0] === "stdout") {
                          const text = String(msg[1]);
                          output += text;
                          if (!sent) {
                            promptBuf += text;
                            if (promptRe.test(promptBuf)) sendInput();
                          }
                          const pos = output.lastIndexOf(marker + ":");
                          if (pos >= 0) {
                            const tail = output.slice(pos + marker.length + 1);
                            const match = tail.match(/(\\d+)/);
                            clearTimeout(timer);
                            finish(true, match ? Number(match[1]) : 0);
                          }
                        }
                      } catch (_) {}
                    });
                    socket.addEventListener("error", () => {
                      clearTimeout(timer);
                      finish(false, 127);
                    });
                    socket.addEventListener("close", () => {
                      if (!settled) {
                        clearTimeout(timer);
                        finish(false, 128);
                      }
                    });
                  });
                }
                """,
                {
                    "wsUrl": ws_url,
                    "stdinData": stdin_payload,
                    "timeoutMs": timeout_ms,
                    "marker": marker,
                },
            )
            output = str(result.get("output", ""))
            marker_pos = output.rfind(marker + ":")
            if marker_pos >= 0:
                output = output[:marker_pos]
            return int(result.get("rc", 1)), output
        finally:
            if term_name:
                _delete_terminal_via_api(context, lab_url=lab_frame.url, term_name=term_name)
            context.close()
            browser.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook_id")
    parser.add_argument("command", nargs="?")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--upload-dir")
    parser.add_argument("--paths", nargs="*", default=[])
    args = parser.parse_args()

    command = args.command
    if args.upload_dir:
        if not args.paths:
            raise SystemExit("--upload-dir requires --paths")
        command = build_upload_command(args.paths, args.upload_dir)
    if not command:
        raise SystemExit("command is required unless --upload-dir is used")

    rc, output = run_notebook_command(
        args.notebook_id,
        command,
        timeout_ms=max(args.timeout, 1) * 1000,
    )
    print(output, end="" if output.endswith("\n") else "\n")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
