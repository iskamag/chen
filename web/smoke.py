#!/usr/bin/env python3
"""Bounded real-engine API checks; owns and retires its server process."""
import json
import socket
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path


class ApiSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cls.url = f"http://127.0.0.1:{port}"
        cls.process = subprocess.Popen(
            [str(Path(sys.argv[1]).resolve())],
            env={**os.environ, "CHEN_WEB_PORT": str(port)},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                cls.call("/api/state")
                return
            except OSError:
                time.sleep(0.03)
        cls.process.terminate()
        cls.process.wait(timeout=3)
        raise RuntimeError("Server did not start")

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        try:
            cls.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            cls.process.kill()
            cls.process.wait(timeout=3)

    @classmethod
    def call(cls, path, body=None):
        request = urllib.request.Request(
            cls.url + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={} if body is None else {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            return json.load(response)

    def submit(self, **options):
        return self.call("/api/new", {"side": "black", "level": 20,
            "time_ms": 1000, **options})["id"]

    def done(self, job):
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            result = self.call("/api/jobs/" + job)
            if result["phase"] == "error":
                self.fail(result["error"])
            if result["phase"] in ("done", "cancelled"):
                return result
            time.sleep(0.02)
        self.fail("Job did not finish")

    def test_01_real_engine_and_legal_moves(self):
        white = self.done(self.submit(side="white"))["state"]
        self.assertEqual(len(white["legal"]), 20)
        result = self.call("/api/move", {"side": "white", "level": 20,
            "time_ms": 50, "moves": [], "fen": white["fen"], "move": "e2e4"})
        state = self.done(result["id"])["state"]
        self.assertEqual(state["moves"][0], "e2e4")
        self.assertEqual(len(state["moves"]), 2)
        self.assertEqual(state["side"], "white")
        self.assertGreater(state["telemetry"]["nodes"], 0)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.call("/api/move", {"side": "white", "level": 20,
                "moves": [], "fen": white["fen"], "move": "e2e5"})
        self.assertEqual(error.exception.code, 400)
        error.exception.close()

    def test_02_queue_wait_does_not_consume_search_time(self):
        first = self.submit()
        second = self.submit()
        a, b = self.done(first), self.done(second)
        self.assertEqual(a["phase"], "done")
        self.assertEqual(b["phase"], "done")
        for result in (a, b):
            self.assertGreaterEqual(result["state"]["telemetry"]["ms"], 900)
            self.assertLess(result["state"]["telemetry"]["ms"], 1400)
        self.assertGreaterEqual(b["queue_ms"], 700)

    def test_03_cancel_active_and_queued_jobs(self):
        first = self.submit(time_ms=5000)
        second = self.submit(time_ms=5000)
        for job in (first, second):
            self.call("/api/cancel", {"id": job})
        self.assertEqual(self.done(first)["phase"], "cancelled")
        self.assertEqual(self.done(second)["phase"], "cancelled")

    def test_04_limits_and_independent_games(self):
        a = self.done(self.submit(nodes=128, depth=2))["state"]
        b = self.done(self.submit(side="white"))["state"]
        self.assertEqual(len(a["moves"]), 1)
        self.assertEqual(b["moves"], [])
        self.assertLessEqual(a["telemetry"]["depth"], 2)
        self.assertLessEqual(a["telemetry"]["nodes"], 128)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.submit(time_ms=5001)
        self.assertEqual(error.exception.code, 400)
        error.exception.close()

    def test_05_queue_is_bounded(self):
        jobs = [self.submit(time_ms=5000)]
        deadline = time.monotonic() + 2
        while self.call("/api/jobs/" + jobs[0])["phase"] != "running":
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
        try:
            for _ in range(8):
                jobs.append(self.submit(time_ms=5000))
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.submit(time_ms=5000)
            self.assertEqual(error.exception.code, 429)
            error.exception.close()
        finally:
            for job in jobs:
                self.call("/api/cancel", {"id": job})
            for job in jobs:
                self.assertEqual(self.done(job)["phase"], "cancelled")

    def test_06_catnip_dispatch_controls_and_legal_reply(self):
        cat = self.done(self.submit(backend="catnip", delta_cp=240,
            theta=0.75, nodes=128, depth=3))["state"]
        self.assertEqual(cat["backend"], "catnip")
        self.assertEqual(cat["delta_cp"], 240)
        self.assertAlmostEqual(cat["theta"], 0.75)
        self.assertEqual(len(cat["moves"]), 1)
        self.assertEqual(cat["side"], "black")
        self.assertLessEqual(cat["telemetry"]["nodes"], 128)
        self.assertLessEqual(cat["telemetry"]["catnip"]["deepest_ply"], 3)
        self.assertIn("fail_meow", cat["telemetry"]["catnip"])
        self.assertIn("fail_zoomies", cat["telemetry"]["catnip"])
        distracted = self.done(self.submit(backend="catnip", theta=1, nodes=128))["state"]
        self.assertGreater(distracted["telemetry"]["catnip"]["ignored_boring"], 0)
        serious = self.done(self.submit(nodes=128))["state"]
        self.assertEqual(serious["backend"], "serious")
        self.assertIsNone(serious["telemetry"]["catnip"])
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.submit(backend="catnip", theta=1.1)
        self.assertEqual(error.exception.code, 400)
        error.exception.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 web/smoke.py web/target/release/chen-web")
    unittest.main(argv=[sys.argv[0]], verbosity=2)
