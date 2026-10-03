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
