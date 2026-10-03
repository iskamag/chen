#!/usr/bin/env python3
"""Summarize Chen's nginx traffic without copying visitor identifiers locally."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

REMOTE = r'''
import collections, datetime, gzip, json, pathlib, re, socket, subprocess, sys
now = datetime.datetime.now(datetime.timezone.utc)
since = now - datetime.timedelta(hours=float(sys.argv[1]))
pattern = re.compile(r'^(\S+) \S+ \S+ \[([^]]+)\] "(\S+) (\S+) [^"]+" (\d+) \S+ "[^"]*" "(.*)"$')
fields = ('page_loads', 'game_start_requests', 'accepted_player_moves', 'result_polls', 'http_errors')
groups = {kind: dict.fromkeys(fields, 0) for kind in ('browser_like', 'own_proxy', 'crawler', 'checks')}
def local_addresses():
    own = set()
    interfaces = json.loads(subprocess.check_output(['ip', '-j', 'address', 'show'], text=True))
    for interface in interfaces:
        for address in interface.get('addr_info', []):
            if address.get('scope') == 'global': own.add(address['local'])
    own.update(item[4][0] for item in socket.getaddrinfo('iskamag.com', 443))
    return own
own = local_addresses()
players = collections.defaultdict(set)
first = last = None
prefix = '/games/play-with-chen/'
files = [pathlib.Path('/var/log/nginx/access.log')]
rotated = pathlib.Path('/var/log/nginx/access.log.1')
files.append(rotated if rotated.exists() else rotated.with_suffix('.1.gz'))
read_files = []
for path in files:
    if not path.exists(): continue
    read_files.append(str(path))
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', errors='replace') as log:
        for line in log:
            if '/games/play-with-chen' not in line: continue
            match = pattern.match(line)
            if not match: continue
            ip, stamp, method, route, status, agent = match.groups()
            route = route.split('?')[0]
            if route != prefix[:-1] and not route.startswith(prefix): continue
            stamp = datetime.datetime.strptime(stamp, '%d/%b/%Y:%H:%M:%S %z')
            if not since <= stamp <= now: continue
            first = stamp if first is None else min(first, stamp)
            last = stamp if last is None else max(last, stamp)
            agent_lower = agent.lower()
            kind = ('checks' if any(word in agent_lower for word in
                ('headlesschrome', 'playwright', 'curl/', 'python-urllib', 'chen-monitor'))
                else 'own_proxy' if ip in own
                else 'crawler' if any(word in agent_lower for word in
                ('bot', 'spider', 'crawl', 'facebookexternalhit', 'linkring', 'flipboardproxy'))
                else 'browser_like')
            group = groups[kind]
            status = int(status)
            utc = stamp.astimezone(datetime.timezone.utc).isoformat()
            if method == 'GET' and route == prefix and status == 200:
                group['page_loads'] += 1
                group['last_page_load_utc'] = max(group.get('last_page_load_utc', utc), utc)
            if method == 'POST' and route == prefix + 'api/new' and status == 202:
                group['game_start_requests'] += 1
            if method == 'POST' and route == prefix + 'api/move' and status == 202:
                group['accepted_player_moves'] += 1
                group['last_player_move_utc'] = max(group.get('last_player_move_utc', utc), utc)
                players[kind].add((ip, agent))
            if method == 'GET' and route.startswith(prefix + 'api/jobs/') and status == 200:
                group['result_polls'] += 1
            if status >= 400: group['http_errors'] += 1
for kind, group in groups.items():
    group['rough_playing_clients'] = len(players[kind])
    group.setdefault('last_page_load_utc', None)
    group.setdefault('last_player_move_utc', None)
iso = lambda stamp: None if stamp is None else stamp.astimezone(datetime.timezone.utc).isoformat()
print(json.dumps({'status': 'ok', 'updated_at_utc': now.isoformat(), 'since_utc': since.isoformat(),
    'first_observed_utc': iso(first), 'last_observed_utc': iso(last), 'log_files': read_files,
    'groups': groups}))
'''


def fetch(hours):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
        '-o', 'ConnectTimeout=5', 'root@iskamag.com', 'python3', '-', str(hours)],
        input=REMOTE, text=True, capture_output=True, timeout=15)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or 'Remote log read failed')
    return json.loads(result.stdout)


def write_snapshot(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as temporary:
        json.dump(report, temporary, indent=2)
        temporary.write('\n')
        pending = temporary.name
    os.replace(pending, path)


def display(report):
    group = report['groups']['browser_like']
    print(f"Chen activity since {report['since_utc']} (snapshot {report['updated_at_utc']})")
    print(f"  Browser-like outside your proxy: {group['page_loads']} page loads, "
          f"{group['game_start_requests']} game-start requests, "
          f"{group['accepted_player_moves']} accepted player moves")
    print(f"  Rough playing clients: {group['rough_playing_clients']}; "
          f"last accepted move: {group['last_player_move_utc'] or 'none'}")
    for kind in ('own_proxy', 'crawler', 'checks'):
        item = report['groups'][kind]
        print(f"  {kind}: {item['page_loads']} page loads, "
              f"{item['accepted_player_moves']} accepted moves (separate from browser-like counts)")
    print('  User-agent categories and client counts are estimates, not verified people or games.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours', type=float, default=24)
    parser.add_argument('--watch', type=float, nargs='?', const=60)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    if not 0 < args.hours <= 48 or (args.watch is not None and args.watch < 10):
        parser.error('hours must be 0..48; watch interval must be at least 10 seconds')
    if args.quiet and args.output is None:
        parser.error('--quiet requires --output')
    while True:
        try:
            report = fetch(args.hours)
            if args.output: write_snapshot(args.output, report)
            if not args.quiet:
                if args.json: print(json.dumps(report, indent=2), flush=True)
                else: display(report)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            if args.output and args.output.exists():
                try:
                    report = json.loads(args.output.read_text())
                    report.update(status='stale', last_error=str(error),
                        failed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                    write_snapshot(args.output, report)
                except (OSError, ValueError): pass
            print(f'Chen traffic monitor: {error}', file=sys.stderr, flush=True)
            if args.watch is None: return 1
        if args.watch is None: return 0
        time.sleep(args.watch)


if __name__ == '__main__':
    try: sys.exit(main())
    except KeyboardInterrupt: pass
