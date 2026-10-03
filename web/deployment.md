# Proxy the browser site back to this machine

Deployment preparation only: no service, public DNS, tunnel, or remote host is
changed by this checkout. The server performs real native Chen searches on the
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
   actual chosen hostname. The tunnel binds to loopback, so searches remain behind
   the proxy rather than exposing the engine port publicly.

   ```caddyfile
   chen.example.com {
       reverse_proxy 127.0.0.1:18787
   }
   ```

   When proxy and engine run on the same host, use `127.0.0.1:8787` instead.
   Configure HTTPS on the proxy. Use a dedicated hostname with the app at `/`;
   this prototype's asset/API paths are absolute.

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

Public publishing still needs a real hostname, SSH/proxy destination and the
user's deployment instruction. No existing homepage or server configuration is
modified by these examples.
