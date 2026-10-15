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


class Server(unittest.TestCase):
    workers = 1
    @classmethod
    def setUpClass(cls):
        import os
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cls.url = f"http://127.0.0.1:{port}"
        cls.process = subprocess.Popen(
            [str(Path(sys.argv[1]).resolve())],
            env={**os.environ, "CHEN_WEB_PORT": str(port), "CHEN_WEB_WORKERS": str(cls.workers)},
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



class ApiSmoke(Server):
    def test_01_real_engine_and_legal_moves(self):
        white = self.done(self.submit(side="white"))["state"]
        self.assertEqual(len(white["legal"]), 20)
        self.assertEqual(white["backend"], "hybrid")
        result = self.call("/api/move", {"side": "white", "level": 20,
            "time_ms": 50, "moves": [], "fen": white["fen"], "move": "e2e4"})
        state = self.done(result["id"])["state"]
        self.assertEqual(state["moves"][0], "e2e4")
        self.assertEqual(len(state["moves"]), 2)
        self.assertEqual(state["san_moves"][0], "e4")
        self.assertEqual(len(state["san_moves"]), 2)
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

    def test_06_drowsy_dispatch_controls_and_legal_reply(self):
        cat = self.done(self.submit(backend="drowsy", delta_cp=240,
            theta=0.75, nodes=128, depth=3))["state"]
        self.assertEqual(cat["backend"], "drowsy")
        self.assertEqual(cat["delta_cp"], 240)
        self.assertAlmostEqual(cat["theta"], 0.75)
        self.assertEqual(len(cat["moves"]), 1)
        self.assertEqual(cat["side"], "black")
        self.assertLessEqual(cat["telemetry"]["nodes"], 128)
        self.assertLessEqual(cat["telemetry"]["drowsy"]["deepest_ply"], 3)
        self.assertIn("fail_meow", cat["telemetry"]["drowsy"])
        self.assertIn("fail_zoomies", cat["telemetry"]["drowsy"])
        distracted = self.done(self.submit(backend="drowsy", theta=1, nodes=128))["state"]
        self.assertGreater(distracted["telemetry"]["drowsy"]["ignored_boring"], 0)
        serious = self.done(self.submit(backend="serious", nodes=128))["state"]
        self.assertEqual(serious["backend"], "serious")
        self.assertIsNone(serious["telemetry"]["drowsy"])
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.submit(backend="drowsy", theta=1.1)
        self.assertEqual(error.exception.code, 400)
        error.exception.close()


    def test_07_replay_notation_without_search_or_board_change(self):
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1"]
        state = self.call("/api/state", {"moves": moves, "side": "white", "level": 20})
        self.assertEqual(state["moves"], moves)
        self.assertEqual(state["san_moves"], ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "O-O"])
        self.assertIsNone(state["telemetry"])

    def test_08_opening_book_replies_instantly(self):
        # Chen is black, so she opens from the book without searching.
        state = self.done(self.submit(side="black", book=True))["state"]
        self.assertEqual(len(state["moves"]), 1)
        self.assertTrue(state["telemetry"]["book"])
        self.assertIsInstance(state["telemetry"]["opening"], str)
        self.assertLess(state["telemetry"]["ms"], 900)
        self.assertTrue(state["book"])
        # The toggle is off by default and never blocks a normal search.
        plain = self.done(self.submit(side="black", time_ms=50))["state"]
        self.assertFalse(plain["telemetry"]["book"])
        self.assertGreater(plain["telemetry"]["nodes"], 0)


    def test_09_hybrid_default_finds_mate_and_preserves_history(self):
        options = {"side": "white", "level": 20, "time_ms": 100,
                   "moves": ["f2f3", "e7e5"], "backend": "hybrid"}
        before = self.call("/api/state", options)
        ticket = self.call("/api/move", {**options, "fen": before["fen"], "move": "g2g4"})
        state = self.done(ticket["id"])["state"]
        self.assertEqual(state["backend"], "hybrid")
        self.assertEqual(state["moves"], ["f2f3", "e7e5", "g2g4", "d8h4"])
        self.assertEqual(state["san_moves"][-1], "Qh4#")
        self.assertEqual(state["outcome"], "Checkmate — Chen wins")


class PoolSmoke(Server):
    workers = 6

    def running(self, jobs):
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            if all(self.call("/api/jobs/" + job)["phase"] == "running" for job in jobs):
                return
            time.sleep(0.01)
        self.fail("Six searches did not overlap; receiver may serialize search")

    def test_parallel_searches_queue_timing_and_cancellation(self):
        first = [self.submit(time_ms=2000) for _ in range(6)]
        self.running(first)
        queued = self.submit(time_ms=1000)
        self.assertEqual(self.call("/api/jobs/" + queued)["phase"], "queued")
        for job in first:
            result = self.done(job)
            self.assertEqual(result["phase"], "done")
            self.assertEqual(len(result["state"]["moves"]), 1)
        result = self.done(queued)
        self.assertGreaterEqual(result["queue_ms"], 1200)
        self.assertGreaterEqual(result["state"]["telemetry"]["ms"], 900)
        active = [self.submit(time_ms=5000) for _ in range(6)]
        self.running(active)
        waiting = [self.submit(time_ms=5000) for _ in range(8)]
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.submit(time_ms=5000)
        self.assertEqual(error.exception.code, 429)
        error.exception.close()
        for job in waiting + active:
            self.call("/api/cancel", {"id": job})
        for job in waiting + active:
            self.assertEqual(self.done(job)["phase"], "cancelled")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 web/smoke.py web/target/release/chen-web")
    unittest.main(argv=[sys.argv[0]], verbosity=2)
