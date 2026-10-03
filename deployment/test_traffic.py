#!/usr/bin/env python3
"""Check request classification and rotation without SSH or live traffic."""
import datetime
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import traffic


class TrafficSummary(unittest.TestCase):
    def test_moves_not_visits_and_rotated_logs(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        def line(path, status=200, method='GET', agent='Mozilla/5.0 Firefox/150',
                 ip='192.0.2.1', when=now):
            stamp = when.strftime('%d/%b/%Y:%H:%M:%S %z')
            return f'{ip} - - [{stamp}] "{method} {path} HTTP/1.1" {status} 42 "-" "{agent}"\n'
        prefix = '/games/play-with-chen/'
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'access.log').write_text(''.join([
                line(prefix),
                line(prefix + 'api/new', 202, 'POST'),
                line(prefix + 'api/move', 202, 'POST'),
                line(prefix + 'api/move', 400, 'POST'),
                line(prefix + 'api/jobs/opaque-id'),
                line(prefix, agent='facebookexternalhit/1.1'),
                line(prefix + 'api/move', 202, 'POST', agent='HeadlessChrome/131'),
                line(prefix, method='HEAD'),
                line('/games/play-with-chen-other/'),
                line(prefix, when=now - datetime.timedelta(days=2)),
                'malformed unrelated log line\n',
            ]))
            with gzip.open(Path(directory, 'access.log.1.gz'), 'wt') as log:
                log.write(line(prefix + 'api/move', 202, 'POST', ip='192.0.2.2'))
            result = subprocess.run([sys.executable, '-', '24'],
                input=traffic.REMOTE.replace('/var/log/nginx', directory),
                capture_output=True, text=True, check=True)
            report = json.loads(result.stdout)
        browser = report['groups']['browser_like']
        self.assertEqual(browser['page_loads'], 1)
        self.assertEqual(browser['game_start_requests'], 1)
        self.assertEqual(browser['accepted_player_moves'], 2)
        self.assertEqual(browser['rough_playing_clients'], 2)
        self.assertEqual(browser['result_polls'], 1)
        self.assertEqual(browser['http_errors'], 1)
        self.assertEqual(report['groups']['crawler']['page_loads'], 1)
        self.assertEqual(report['groups']['checks']['accepted_player_moves'], 1)
        self.assertNotIn('192.0.2.', json.dumps(report))


if __name__ == '__main__':
    unittest.main()
