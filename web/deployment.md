# Proxy the browser site back to this machine

Chosen URL: **https://iskamag.com/games/play-with-chen/**. The bare path should
redirect to its trailing-slash form so browser assets and API requests resolve
inside the prefix. The installed nginx/SSH/OpenRC setup is documented in
[the deployment runbook](../deployment/README.md). The examples below also allow
independent local hosting. The server performs real native Chen searches on the
machine running it. A reverse proxy forwards both the board assets and API.

1. Build and run on the engine machine:

   ```sh
   cargo build --release --manifest-path web/Cargo.toml
   web/target/release/chen-web
   ```

2. If the public host cannot directly reach this machine, open a tunnel **from
   the engine machine** (replace the SSH destination):

   ```sh
   ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
     -R 127.0.0.1:18787:127.0.0.1:8787 user@public-host
   ```

3. On the public host, merge the following into a Caddy configuration using the
   existing `iskamag.com` site. The tunnel binds to loopback, so searches remain behind
   the proxy rather than exposing the engine port publicly.

   ```caddyfile
   # Merge these routes into the existing iskamag.com site.
   redir /games/play-with-chen /games/play-with-chen/ 308
   handle_path /games/play-with-chen/* {
       reverse_proxy 127.0.0.1:18787
   }
   ```

   When proxy and engine run on the same host, use `127.0.0.1:8787` instead.
   Configure HTTPS on the proxy. `handle_path` strips the public prefix before
   forwarding; assets and browser API calls are relative to the trailing-slash URL.
   Local hosting at `/` continues to work.

   Equivalent locations for an existing Nginx HTTPS server:

   ```nginx
   location = /games/play-with-chen {
       return 308 /games/play-with-chen/;
   }
   location ^~ /games/play-with-chen/ {
       proxy_pass http://127.0.0.1:18787/;
       proxy_http_version 1.1;
       proxy_set_header Host $host;
   }
   ```

The browser posts to `/api/new` or `/api/move` and receives a job ID immediately.
It polls `/api/jobs/<id>` until `queued → running → done`; it can cancel with
`POST /api/cancel`. CORS is not enabled. Jobs carry independent move histories,
so visitors never share a game. API writes require JSON; request sizes, concurrent
connections, queue length, retained results, histories, time/depth/nodes are bounded.

Compute currently runs on one search worker with up to eight waiting jobs, a
50–5000 ms time budget, depth 1–64 and optional 1–20M node cap. Queue-full requests
return 429. This is a small experimental service rather than a production hosting
platform; expose it only at the intended checkpoint audience/load. Search timing
excludes queue wait, replay, HTTP transport and TT allocation. Polling does not
block behind the running engine. Restarting the server forgets pending results,
while tabs retain their last completed board and can submit a new request.

The installed public route forwards to local port 8788 through remote port 18787.
The existing homepage stays unchanged.
