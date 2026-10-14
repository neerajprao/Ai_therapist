"""Build the cloud Docker image and measure its size and memory under real requests.

  python -m eval.cloud_memory          # needs Docker and GROQ_API_KEY (in .env)

Steps:
  1. docker build (cloud image: no PyTorch).
  2. Run the container without a memory limit, send real voice clips through
     /process_audio_stream (Groq STT + LLM), and record peak memory from the
     container's cgroup (memory.peak) plus `docker stats` samples.
  3. Run it again with --memory=512m and the same requests, and check that every
     request succeeds and the container is not OOM-killed.
Writes eval/results/cloud_memory.json and .log.
"""

from __future__ import annotations

import json
import os
import platform
import random
import re
import secrets
import subprocess
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
IMAGE = "soulsync-cloud:eval"
PORT = 8765
N_CLIPS = 6


def sh(*args: str, check: bool = True) -> str:
    return subprocess.run(args, check=check, capture_output=True, text=True).stdout.strip()


def groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key and (ROOT / ".env").exists():
        m = re.search(r"^GROQ_API_KEY=(\S+)", (ROOT / ".env").read_text(), re.M)
        key = m.group(1) if m else ""
    if not key:
        raise SystemExit("GROQ_API_KEY not set")
    return key


def to_mib(text: str) -> float:
    num, unit = re.match(r"([\d.]+)\s*([KMG]i?B)", text).groups()
    return float(num) * {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024, "kB": 1 / 1024, "MB": 1, "GB": 1024}[unit]


def clips() -> list[Path]:
    from training.data import DATA, read_split

    rows = [r for r in read_split("split_v2.csv") if r["split"] == "test" and float(r["duration"]) >= 2.0]
    random.Random(1).shuffle(rows)
    return [DATA / r["path"] for r in rows[:N_CLIPS]]


def run_case(name: str, memory_limit: str | None) -> dict:
    container = f"soulsync-eval-{name}"
    sh("docker", "rm", "-f", container, check=False)
    args = ["docker", "run", "-d", "--name", container, "-p", f"{PORT}:8000",
            "-e", "MODE=cloud", "-e", f"SECRET_KEY={secrets.token_hex(16)}", "-e", "COOKIE_SECURE=false",
            "-e", "GROQ_API_KEY"]
    if memory_limit:
        args += ["--memory", memory_limit, "--memory-swap", memory_limit]
    started = time.time()
    subprocess.run(args + [IMAGE], check=True, capture_output=True, env={**os.environ, "GROQ_API_KEY": groq_key()})

    client = httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=60)
    health = None
    for _ in range(120):
        try:
            health = client.get("/health").json()
            break
        except httpx.HTTPError:
            time.sleep(0.5)
    if health is None:
        logs = sh("docker", "logs", container, check=False)
        sh("docker", "rm", "-f", container, check=False)
        raise SystemExit(f"container did not become healthy:\n{logs[-2000:]}")
    ready_seconds = time.time() - started
    idle_mib = to_mib(sh("docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container).split("/")[0])

    samples: list[float] = []
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            out = sh("docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container, check=False)
            if out:
                samples.append(to_mib(out.split("/")[0]))

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()

    page = client.get("/")
    session = re.search(r'const SESSION_ID = "([0-9a-f]{32})"', page.text).group(1)
    statuses, replies = [], []
    for path in clips():
        try:
            with open(path, "rb") as f:
                r = client.post("/process_audio_stream", files={"audio": (path.name, f, "audio/wav")}, data={"session_id": session})
            statuses.append(r.status_code)
            replies.append(r.text[:160])
        except httpx.HTTPError as exc:
            statuses.append(f"error: {type(exc).__name__}")
            replies.append("")
        time.sleep(3.5)  # stay under Groq's per-minute limits
    try:
        forget = client.delete("/memory").status_code
    except httpx.HTTPError as exc:
        forget = f"error: {type(exc).__name__}"
    stop.set()
    sampler.join()

    peak_cgroup = sh("docker", "exec", container, "cat", "/sys/fs/cgroup/memory.peak", check=False)
    events = sh("docker", "exec", container, "cat", "/sys/fs/cgroup/memory.events", check=False)
    state = json.loads(sh("docker", "inspect", container))[0]["State"]
    restarts = json.loads(sh("docker", "inspect", container))[0].get("RestartCount", 0)
    logs = sh("docker", "logs", container, check=False)
    sh("docker", "rm", "-f", container, check=False)
    return {
        "memory_limit": memory_limit or "none",
        "health": health,
        "seconds_to_healthy": round(ready_seconds, 1),
        "idle_after_start_mib": round(idle_mib, 1),
        "requests": len(statuses),
        "request_status_codes": statuses,
        "forget_me_status": forget,
        "all_requests_ok": all(s == 200 for s in statuses) and forget == 200,
        "peak_mib_cgroup_memory_peak": round(int(peak_cgroup) / 2**20, 1) if peak_cgroup.isdigit() else None,
        "peak_mib_docker_stats_samples": round(max(samples), 1) if samples else None,
        "docker_stats_samples": len(samples),
        "cgroup_memory_events": dict(line.split() for line in events.splitlines()) if events else None,
        "oom_killed": state.get("OOMKilled"),
        "exit_code": state.get("ExitCode"),
        "restart_count": restarts,
        "still_running_at_end": state.get("Running"),
        "example_replies": replies[:2],
        "log_tail": logs[-600:],
    }


def main() -> None:
    t0 = time.time()
    build = subprocess.run(["docker", "build", "-t", IMAGE, str(ROOT)], capture_output=True, text=True)
    if build.returncode != 0:
        raise SystemExit(build.stderr[-3000:])
    build_seconds = time.time() - t0
    size_bytes = int(sh("docker", "image", "inspect", IMAGE, "--format", "{{.Size}}"))
    arch = sh("docker", "image", "inspect", IMAGE, "--format", "{{.Os}}/{{.Architecture}}")
    has_torch = sh("docker", "run", "--rm", IMAGE, "python", "-c",
                   "import importlib.util as u; print(bool(u.find_spec('torch')))")

    unlimited = run_case("unlimited", None)
    limited = run_case("limit512", "512m")
    peak = unlimited["peak_mib_cgroup_memory_peak"] or unlimited["peak_mib_docker_stats_samples"]
    result = {
        "image": IMAGE,
        "platform": arch,
        "built_on": f"{platform.machine()} {platform.platform()}, Docker {sh('docker', 'version', '--format', '{{.Server.Version}}')}",
        "image_size_mb": round(size_bytes / 1e6, 1),
        "torch_installed_in_image": has_torch == "True",
        "build_seconds": round(build_seconds, 1),
        "unlimited_run": unlimited,
        "run_with_512m_limit": limited,
        "peak_memory_mib": peak,
        "fits_512mb": bool(limited["all_requests_ok"] and not limited["oom_killed"]),
        "notes": "Peak memory is cgroup memory.peak (includes page cache) for one container serving one user "
        "sequentially, with the knowledge base and memory enabled. Requests use real test clips through Groq. "
        "Built for the host architecture; a linux/amd64 host image will differ slightly in size.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "cloud_memory.json").write_text(json.dumps(result, indent=2))
    log = (
        f"image {result['image_size_mb']} MB ({arch}), torch in image: {result['torch_installed_in_image']}\n"
        f"unlimited: peak {unlimited['peak_mib_cgroup_memory_peak']} MiB (cgroup), {unlimited['peak_mib_docker_stats_samples']} MiB (stats), "
        f"idle {unlimited['idle_after_start_mib']} MiB, healthy after {unlimited['seconds_to_healthy']} s, ok={unlimited['all_requests_ok']}\n"
        f"512m limit: ok={limited['all_requests_ok']}, oom_killed={limited['oom_killed']}, peak {limited['peak_mib_cgroup_memory_peak']} MiB\n"
        f"fits 512 MB: {result['fits_512mb']}\n"
    )
    (RESULTS / "cloud_memory.log").write_text(log)
    print(log)


if __name__ == "__main__":
    main()
