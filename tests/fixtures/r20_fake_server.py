#!/usr/bin/env python3
"""Minimal pinned-protocol test double; no GGUF inference or performance simulation."""
import fcntl
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import signal
import sys


def arg(key):
    return sys.argv[sys.argv.index(key) + 1]


mode = os.environ.get("R20_TEST_MODE")
if mode == "oom":
    print("cudaMalloc(3694.73 MiB): out of memory; free = 3475 MiB", flush=True)
    sys.exit(1)
lock = open(os.environ["R20_TEST_LOCK"], "a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
print(f"offloaded {arg('-ngl')}/42 layers to GPU", flush=True)


class Handler(BaseHTTPRequestHandler):
    def reply(self, result):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def do_GET(self):
        if self.path == "/props":
            self.reply({"total_slots": 1, "model_path": arg("-m"), "model_alias": arg("--alias"),
                        "default_generation_settings": {"n_ctx": int(arg("-c"))}})
        elif self.path == "/health":
            self.reply({"status": "ok"})
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            return self.reply({"tokens": [1] + [3] * len(body["content"].split())})
        assert self.path == "/completion" and body["temperature"] == 0 and body["cache_prompt"] is False
        assert body["ignore_eos"] and body["return_tokens"]
        n, output = len(body["prompt"]), body["n_predict"]
        tokens = [4] * output
        if mode == "mismatch" and os.getenv("LLAMA_EXPERIMENTAL_MOE_PLAN_REUSE") == "1":
            tokens[-1] = 9
        timing = {"cache_n": 0, "prompt_n": n, "predicted_n": output, "prompt_ms": 500,
                  "predicted_ms": 2000, "prompt_per_second": 2*n, "predicted_per_second": output/2}
        result = {"stop": True, "truncated": False, "tokens": tokens, "tokens_predicted": output,
                  "tokens_evaluated": n, "timings": timing, "content": ""}
        if not body["stream"]:
            return self.reply(result)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        result["tokens"] = []
        def event(row):
            self.wfile.write(b"data: " + json.dumps(row).encode() + b"\n\n")
            self.wfile.flush()
        event({"stop": False, "tokens": []})
        # Pinned server omits partial events while a UTF-8 sequence is incomplete.
        for token in tokens[1:]:
            event({"stop": False, "tokens": [token]})
        event(result)


HTTPServer(("127.0.0.1", int(arg("--port"))), Handler).serve_forever()
