# Public Chen checkpoint

**https://iskamag.com/games/play-with-chen/** proxies through nginx on
`iskamag.com` to this machine's native CPU server. Both serious Chen and the
experimental δ–θ Drowsy mood use the same bounded request queue. The selected
search time excludes queue wait. Six independent single-threaded searches can
run at once; eight additional requests may wait. HTTP handling is separate.

The installed OpenRC **user services** are `chen-web` and `chen-tunnel`. They
supervise the server and reconnecting reverse SSH tunnel, and are registered in
the user's default runlevel. They need this machine and its user OpenRC session
running. The public host does not run the engine; loss of the local machine or
tunnel makes the site unavailable.

```sh
rc-service --user chen-web status
rc-service --user chen-tunnel status
```

The server binds only `127.0.0.1:8788`. The SSH tunnel binds only the public host's
`127.0.0.1:18787`; [chen.nginx.conf](chen.nginx.conf) redirects the bare path and
strips the prefix before forwarding. The existing nginx HTTPS server includes
`/etc/nginx/snippets/chen.conf`. Its original config is backed up at
`/etc/nginx/sites-available/default.chen-20261003.bak`. The homepage is unchanged.

To update the installed engine after committing and checking a new build:

```sh
cargo build --release --manifest-path web/Cargo.toml
install -m755 web/target/release/chen-web "$HOME/.local/share/chen-host/chen-web.next"
mv "$HOME/.local/share/chen-host/chen-web.next" "$HOME/.local/share/chen-host/chen-web"
setsid rc-service --user chen-web restart
```

The detached service actions also survive an automated runner cleaning up its
process group. A restart cancels in-flight jobs; browsers retain their last completed board.
The binary's build label identifies its engine checkout. Logs are in
`~/.local/state/chen-host/`. The checked-in OpenRC scripts are installed as
`~/.config/rc/init.d/chen-web` and `chen-tunnel`. To reinstall them:

```sh
mkdir -p "$HOME/.config/rc/init.d" "$HOME/.local/share/chen-host" "$HOME/.local/state/chen-host"
install -m755 deployment/chen-web.openrc "$HOME/.config/rc/init.d/chen-web"
install -m755 deployment/chen-tunnel.openrc "$HOME/.config/rc/init.d/chen-tunnel"
rc-update --user add chen-web default
rc-update --user add chen-tunnel default
setsid rc-service --user chen-web start
setsid rc-service --user chen-tunnel start
```

To take the checkpoint offline, stop `chen-tunnel` and `chen-web`. Remove the
Chen include from the existing nginx server and validate/reload nginx to remove
the public route; restore the saved full config only after checking that no other
changes have occurred since the backup. These services do not run GPU or training
jobs.

## Activity monitoring

`chen-traffic` is a supervised OpenRC user service that refreshes
`~/.local/state/chen-host/traffic.json` every 60 seconds by reading nginx's
existing access logs over SSH. It does not change the game server or add browser
tracking. The report contains aggregate counts, not visitor addresses or agents.

```sh
python3 deployment/traffic.py             # fetch and summarize now
python3 deployment/traffic.py --watch 15  # watch in the terminal; Ctrl+C to stop
rc-service --user chen-traffic status
cat "$HOME/.local/state/chen-host/traffic.json"
```

Page loads, accepted game-start requests and accepted player moves are separate.
Only successful `POST /api/move` responses count as accepted player moves: the
server has validated the move/history and admitted the job. This indicates play
requests, not guaranteed completed turns or games. Starting as White need not
produce a game-start request, so these counters cannot estimate unique games.
Crawler and QA user agents are counted separately. Browser-like agents can still
be bots, and an IP/agent pair is only a rough playing-client estimate; it cannot
identify people or distinguish the owner from friends.

The default window is the last 24 hours, using the current and latest rotated
nginx log. The report lists its input files and observed time range. It is not a
lifetime counter. Failed refreshes retain the last good counters and mark the
snapshot `stale` with an error and failure time. The service needs network and
existing root SSH access to `iskamag.com`; it runs independently of the engine
and tunnel. Errors go to `~/.local/state/chen-host/traffic-monitor-error.log`.

The installed script is `~/.local/share/chen-host/traffic.py`. To install/start or
stop this monitor:

```sh
install -m644 deployment/traffic.py "$HOME/.local/share/chen-host/traffic.py"
install -m755 deployment/chen-traffic.openrc "$HOME/.config/rc/init.d/chen-traffic"
rc-update --user add chen-traffic default
setsid rc-service --user chen-traffic start
# Stop monitoring:
rc-service --user chen-traffic stop
```

`python3 deployment/test_traffic.py` checks visit/play separation, crawler/QA
classification and compressed-log rotation using synthetic local logs, without
SSH or live game traffic.
