mod hybrid;
mod notation;

use chen::backend::SearchBackend;
use chen::drowsy::{DrowsyConfig, DrowsySearch};
use chen::strength::{BudgetStrength, StrengthPolicy};
use chen::{Color, OpeningBook, Position, SearchConfig, SearchLimits, Searcher, Tapered};
use hybrid::HybridSearch;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{self, SyncSender},
    },
    time::{Duration, Instant},
};

/// Half-moves of book play offered before the search takes over.
const BOOK_PLIES: usize = 12;

/// The bundled book is parsed once and shared by every game.
fn bundled_book() -> Option<Arc<OpeningBook>> {
    static BOOK: OnceLock<Option<Arc<OpeningBook>>> = OnceLock::new();
    BOOK.get_or_init(|| OpeningBook::bundled().map(Arc::new))
        .clone()
}

fn random_seed() -> u64 {
    let mut bytes = [0u8; 8];
    match std::fs::File::open("/dev/urandom").and_then(|mut file| file.read_exact(&mut bytes)) {
        Ok(()) => u64::from_le_bytes(bytes),
        Err(_) => 0,
    }
}

enum WebSearch {
    Hybrid(Box<HybridSearch>),
    Serious(Box<Searcher>),
    Drowsy(Box<DrowsySearch>),
}
impl WebSearch {
    fn stop_handle(&self) -> Arc<AtomicBool> {
        match self {
            Self::Hybrid(searcher) => searcher.stop_handle(),
            Self::Serious(searcher) => searcher.stop_handle(),
            Self::Drowsy(searcher) => searcher.stop_handle(),
        }
    }
    fn set_book(&mut self, book: Option<Arc<OpeningBook>>, seed: u64) {
        match self {
            Self::Hybrid(searcher) => searcher.set_book(book, seed, BOOK_PLIES),
            Self::Serious(searcher) => searcher.set_book(book, seed, BOOK_PLIES),
            Self::Drowsy(searcher) => searcher.set_book(book, seed, BOOK_PLIES),
        }
    }
    fn search(
        &mut self,
        position: &mut Position,
        limits: SearchLimits,
        history: &[u64],
    ) -> Result<chen::SearchResult, String> {
        match self {
            Self::Hybrid(searcher) => Ok(searcher.search(position, limits, history)),
            Self::Serious(searcher) => Ok(searcher.search(position, limits, history, |_| {})),
            Self::Drowsy(searcher) => searcher.search(position, limits, history, &mut |_| {}),
        }
    }
    fn drowsy_stats(&self) -> Value {
        match self {
            Self::Serious(_) | Self::Hybrid(_) => Value::Null,
            Self::Drowsy(searcher) => {
                let stats = searcher.stats();
                json!({"ignored_boring":stats.ignored_boring, "fail_meow":stats.fail_meow,
                    "fail_zoomies":stats.fail_zoomies, "noticed_drift":stats.noticed_drift,
                    "deepest_ply":stats.deepest_ply, "peak_walkers":stats.peak_walkers})
            }
        }
    }
}
struct Game {
    position: Position,
    history: Vec<u64>,
    moves: Vec<String>,
    san_moves: Vec<String>,
    human: Color,
    level: u8,
    milliseconds: u64,
    depth_limit: u8,
    node_limit: Option<u64>,
    backend: &'static str,
    delta_cp: i32,
    theta: f32,
    book: bool,
    book_seed: u64,
    searcher: WebSearch,
    telemetry: Value,
}
impl Game {
    fn new() -> Self {
        Self {
            position: Position::startpos(),
            history: Vec::new(),
            moves: Vec::new(),
            san_moves: Vec::new(),
            human: Color::White,
            level: 20,
            milliseconds: 1000,
            depth_limit: 64,
            node_limit: None,
            backend: "hybrid",
            delta_cp: 80,
            theta: 0.55,
            book: false,
            book_seed: 0,
            searcher: WebSearch::Hybrid(Box::new(HybridSearch::new())),
            telemetry: Value::Null,
        }
    }
    fn outcome(&self) -> Option<String> {
        if self.position.legal_moves().is_empty() {
            return Some(if self.position.is_in_check(self.position.side_to_move()) {
                if self.position.side_to_move().opposite() == self.human {
                    "Checkmate — You win".into()
                } else {
                    "Checkmate — Chen wins".into()
                }
            } else {
                "Draw — stalemate".into()
            });
        }
        if self.moves.len() >= 512 {
            return Some("Game stopped — checkpoint 512-ply limit".into());
        }
        if self.position.insufficient_material() {
            return Some("Draw — insufficient material".into());
        }
        if self.position.halfmove_clock() >= 100 {
            return Some("Draw — fifty-move rule".into());
        }
        if self
            .history
            .iter()
            .filter(|&&h| h == self.position.hash())
            .count()
            >= 2
        {
            return Some("Draw — threefold repetition".into());
        }
        None
    }
    fn push(&mut self, mv: chen::Move) {
        self.history.push(self.position.hash());
        self.moves.push(mv.to_string());
        self.san_moves.push(notation::san(&self.position, mv));
        self.position.make_move(mv);
    }
    fn reply(&mut self) -> Result<(), String> {
        if self.position.side_to_move() == self.human || self.outcome().is_some() {
            return Ok(());
        }
        let limits = BudgetStrength::new(self.level)
            .unwrap()
            .apply(SearchLimits {
                time: Some(Duration::from_millis(self.milliseconds)),
                depth: self.depth_limit,
                nodes: self.node_limit,
            });
        let result = self
            .searcher
            .search(&mut self.position, limits, &self.history)?;
        let opening = result.book_move.as_ref().map(|entry| entry.name.clone());
        self.telemetry = json!({ "depth": result.depth, "nodes": result.nodes,
            "nps": result.nps(), "ms": result.elapsed.as_millis(),
            "book": result.book_move.is_some(), "opening": opening,
            "pv": result.pv.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "drowsy": self.searcher.drowsy_stats() });
        if let Some(mv) = result.best_move {
            self.push(mv);
        }
        Ok(())
    }
    fn state(&self) -> Value {
        let pieces: Vec<_> = (0..64)
            .map(|sq| {
                self.position.piece_at(sq).map(|(color, piece)| {
                    let c = ["p", "n", "b", "r", "q", "k"][piece.index()];
                    if color == Color::White {
                        c.to_uppercase()
                    } else {
                        c.into()
                    }
                })
            })
            .collect();
        json!({ "pieces": pieces, "fen": self.position.to_fen(),
            "legal": self.position.legal_moves().iter().map(ToString::to_string).collect::<Vec<_>>(),
            "side": if self.position.side_to_move() == Color::White { "white" } else { "black" },
            "human": if self.human == Color::White { "white" } else { "black" },
            "check": self.position.is_in_check(self.position.side_to_move()),
            "outcome": self.outcome(), "moves": self.moves, "san_moves": self.san_moves, "telemetry": self.telemetry,
            "revision": env!("CHEN_REVISION"), "level": self.level, "milliseconds": self.milliseconds, "depth_limit": self.depth_limit, "node_limit": self.node_limit,
            "backend": self.backend, "delta_cp": self.delta_cp, "theta": self.theta, "book": self.book })
    }
    fn from_request(path: &str, data: Value) -> Result<Self, String> {
        let mut game = Self::new();
        game.human = match data["side"].as_str() {
            Some("white") => Color::White,
            Some("black") => Color::Black,
            _ => return Err("Choose white or black".into()),
        };
        game.level = data["level"]
            .as_u64()
            .filter(|n| (1..=20).contains(n))
            .ok_or("Level must be 1–20")? as u8;
        game.milliseconds = data["time_ms"].as_u64().unwrap_or(1000);
        if !(50..=5000).contains(&game.milliseconds) {
            return Err("Move time must be 50–5000 ms".into());
        }
        game.depth_limit = match data.get("depth") {
            None | Some(Value::Null) => 64,
            Some(value) => value
                .as_u64()
                .filter(|n| (1..=64).contains(n))
                .ok_or("Depth must be 1–64")? as u8,
        };
        game.node_limit = match data.get("nodes") {
            None | Some(Value::Null) => None,
            Some(value) => Some(
                value
                    .as_u64()
                    .filter(|n| (1..=20_000_000).contains(n))
                    .ok_or("Nodes must be 1–20000000")?,
            ),
        };
        game.backend = match data["backend"].as_str() {
            None | Some("hybrid") => "hybrid",
            Some("serious") => "serious",
            Some("drowsy") => "drowsy",
            _ => return Err("Backend must be hybrid, serious or drowsy".into()),
        };
        game.delta_cp = match data.get("delta_cp") {
            None => 80,
            Some(value) => value
                .as_i64()
                .filter(|n| (0..=1000).contains(n))
                .ok_or("Drowsy drift must be 0–1000 cp")? as i32,
        };
        game.theta = match data.get("theta") {
            None => 0.55,
            Some(value) => value
                .as_f64()
                .filter(|n| n.is_finite() && (0.0..=1.0).contains(n))
                .ok_or("Drowsy attention must be 0–1")? as f32,
        };
        game.book = data.get("book").and_then(Value::as_bool).unwrap_or(false);
        game.book_seed = random_seed();
        if game.backend == "serious" {
            game.searcher = WebSearch::Serious(Box::new(Searcher::new(
                Box::new(Tapered),
                SearchConfig {
                    pvs: true,
                    ..Default::default()
                },
            )));
        } else if game.backend == "drowsy" {
            game.searcher = WebSearch::Drowsy(Box::new(DrowsySearch::new(DrowsyConfig {
                delta_cp: game.delta_cp,
                theta: game.theta,
                ..Default::default()
            })?));
        }
        if game.book {
            game.searcher.set_book(bundled_book(), game.book_seed);
        }
        if path == "/api/new" {
            return Ok(game);
        }
        if path != "/api/move" && path != "/api/state" {
            return Err("Unknown endpoint".into());
        }
        let moves = data["moves"].as_array().ok_or("Missing move history")?;
        if moves.len() > 512 || (path == "/api/move" && moves.len() == 512) {
            return Err("Game reached this checkpoint's 512-ply limit".into());
        }
        for text in moves {
            let mv = game
                .position
                .parse_uci_move(text.as_str().ok_or("Invalid move history")?)?;
            if game.outcome().is_some() {
                return Err("Move history continues after game end".into());
            }
            game.push(mv);
        }
        if path == "/api/state" {
            return Ok(game);
        }
        if game.position.side_to_move() != game.human {
            return Err("It is Chen's turn".into());
        }
        if game.outcome().is_some() {
            return Err("The game has ended".into());
        }
        if data["fen"].as_str() != Some(game.position.to_fen().as_str()) {
            return Err("Board and history disagree".into());
        }
        let mv = game
            .position
            .parse_uci_move(data["move"].as_str().ok_or("Missing move")?)?;
        game.push(mv);
        Ok(game)
    }
}
struct Job {
    id: String,
    game: Game,
    created: Instant,
}
struct Entry {
    created: Instant,
    status: Value,
    stop: Arc<AtomicBool>,
    finished: bool,
}
type Jobs = Arc<Mutex<HashMap<String, Entry>>>;
fn job_id() -> std::io::Result<String> {
    let mut random = [0u8; 16];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut random)?;
    Ok(random.iter().map(|b| format!("{b:02x}")).collect())
}
fn worker(receiver: Arc<Mutex<mpsc::Receiver<Job>>>, jobs: Jobs, queued: Arc<AtomicUsize>) {
    loop {
        // Share admission only: release the receiver lock before doing any search.
        let received = receiver.lock().unwrap().recv();
        let Ok(mut job) = received else { break };

        queued.fetch_sub(1, Ordering::SeqCst);
        let queue_ms = job.created.elapsed().as_millis();
        let stop = job.game.searcher.stop_handle();
        if !stop.load(Ordering::Relaxed) {
            jobs.lock().unwrap().get_mut(&job.id).unwrap().status =
                json!({"phase":"running", "queue_ms":queue_ms});
            // Search's own timer starts inside Searcher::search, after dequeue and history replay.
            if let Err(error) = job.game.reply() {
                if let Some(entry) = jobs.lock().unwrap().get_mut(&job.id) {
                    entry.status = json!({"phase":"error", "error":error});
                    entry.finished = true;
                }
                continue;
            }
        }
        let result = if stop.load(Ordering::Relaxed) {
            json!({"phase":"cancelled"})
        } else {
            json!({"phase":"done", "queue_ms":queue_ms, "state":job.game.state()})
        };
        if let Some(entry) = jobs.lock().unwrap().get_mut(&job.id) {
            entry.status = result;
            entry.finished = true;
        }
    }
}

fn response(stream: &mut TcpStream, status: &str, mime: &str, body: &[u8]) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)
}
fn handle(
    stream: &mut TcpStream,
    sender: &SyncSender<Job>,
    jobs: &Jobs,
    queued: &Arc<AtomicUsize>,
) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(3)))?;
    stream.set_write_timeout(Some(Duration::from_secs(3)))?;
    let mut bytes = Vec::new();
    let mut buffer = [0; 2048];
    let end = loop {
        let n = stream.read(&mut buffer)?;
        if n == 0 {
            return Ok(());
        }
        bytes.extend_from_slice(&buffer[..n]);
        if bytes.len() > 16_384 {
            return response(
                stream,
                "413 Content Too Large",
                "text/plain",
                b"Request too large",
            );
        }
        if let Some(i) = bytes.windows(4).position(|b| b == b"\r\n\r\n") {
            break i + 4;
        }
    };
    let header = String::from_utf8_lossy(&bytes[..end]).to_string();
    let mut first = header.lines().next().unwrap_or("").split_whitespace();
    let method = first.next().unwrap_or("");
    let path = first.next().unwrap_or("");
    if method == "GET" {
        let (mime, body): (&str, Vec<u8>) = match path {
            "/" => (
                "text/html; charset=utf-8",
                include_bytes!("../static/index.html").to_vec(),
            ),
            "/app.js" => (
                "text/javascript; charset=utf-8",
                include_bytes!("../static/app.js").to_vec(),
            ),
            "/style.css" => (
                "text/css; charset=utf-8",
                include_bytes!("../static/style.css").to_vec(),
            ),
            "/chen.svg" => (
                "image/svg+xml",
                include_bytes!("../static/chen.svg").to_vec(),
            ),
            "/api/state" => (
                "application/json",
                Game::new().state().to_string().into_bytes(),
            ),
            _ if path.starts_with("/api/jobs/") => {
                let id = &path[10..];
                let map = jobs.lock().unwrap();
                let Some(entry) = map.get(id) else {
                    return response(
                        stream,
                        "404 Not Found",
                        "application/json",
                        br#"{"error":"Job expired or unknown"}"#,
                    );
                };
                let mut status = entry.status.clone();
                if status["phase"] == "queued" {
                    status["waiting_ms"] = json!(entry.created.elapsed().as_millis());
                }
                ("application/json", status.to_string().into_bytes())
            }
            _ => return response(stream, "404 Not Found", "text/plain", b"Not found"),
        };
        return response(stream, "200 OK", mime, &body);
    }
    if method != "POST" {
        return response(
            stream,
            "405 Method Not Allowed",
            "text/plain",
            b"Use GET or POST",
        );
    }
    let mut length = None;
    let mut json_type = false;
    for line in header.lines().skip(1) {
        if let Some((key, value)) = line.split_once(':') {
            if key.eq_ignore_ascii_case("content-length") {
                length = value.trim().parse::<usize>().ok();
            }
            if key.eq_ignore_ascii_case("content-type") {
                json_type = value.trim() == "application/json";
            }
        }
    }
    let Some(length) = length.filter(|n| *n <= 8192) else {
        return response(
            stream,
            "400 Bad Request",
            "text/plain",
            b"Invalid request length",
        );
    };
    // JSON-only writes, and no CORS permission, prevent cross-origin browser forms.
    if !json_type {
        return response(
            stream,
            "415 Unsupported Media Type",
            "text/plain",
            b"Use application/json",
        );
    }
    while bytes.len() < end + length {
        let n = stream.read(&mut buffer)?;
        if n == 0 {
            return response(stream, "400 Bad Request", "text/plain", b"Incomplete body");
        }
        bytes.extend_from_slice(&buffer[..n]);
    }
    let data: Value = match serde_json::from_slice(&bytes[end..end + length]) {
        Ok(data) => data,
        Err(_) => {
            return response(
                stream,
                "400 Bad Request",
                "application/json",
                br#"{"error":"Invalid JSON"}"#,
            );
        }
    };
    if path == "/api/cancel" {
        let map = jobs.lock().unwrap();
        if let Some(entry) = data["id"].as_str().and_then(|id| map.get(id)) {
            entry.stop.store(true, Ordering::Relaxed);
        }
        return response(
            stream,
            "200 OK",
            "application/json",
            br#"{"cancelled":true}"#,
        );
    }
    if path != "/api/new" && path != "/api/move" && path != "/api/state" {
        return response(
            stream,
            "404 Not Found",
            "application/json",
            br#"{"error":"Unknown endpoint"}"#,
        );
    }
    let game = match Game::from_request(path, data) {
        Ok(game) => game,
        Err(error) => {
            return response(
                stream,
                "400 Bad Request",
                "application/json",
                json!({"error":error}).to_string().as_bytes(),
            );
        }
    };
    if path == "/api/state" {
        return response(
            stream,
            "200 OK",
            "application/json",
            game.state().to_string().as_bytes(),
        );
    }
    let id = job_id()?;
    let created = Instant::now();
    {
        let mut map = jobs.lock().unwrap();
        map.retain(|_, entry| {
            !entry.finished || entry.created.elapsed() < Duration::from_secs(120)
        });
        if map.len() >= 64 {
            let oldest = map
                .iter()
                .filter(|(_, entry)| entry.finished)
                .min_by_key(|(_, entry)| entry.created)
                .map(|(id, _)| id.clone());
            if let Some(oldest) = oldest {
                map.remove(&oldest);
            }
        }
        if map.len() >= 64 {
            return response(
                stream,
                "429 Too Many Requests",
                "application/json",
                br#"{"error":"Chen is busy; try again shortly"}"#,
            );
        }
        map.insert(
            id.clone(),
            Entry {
                created,
                status: json!({"phase":"queued"}),
                stop: game.searcher.stop_handle(),
                finished: false,
            },
        );
    }
    queued.fetch_add(1, Ordering::SeqCst);
    if sender
        .try_send(Job {
            id: id.clone(),
            game,
            created,
        })
        .is_err()
    {
        queued.fetch_sub(1, Ordering::SeqCst);
        jobs.lock().unwrap().remove(&id);
        return response(
            stream,
            "429 Too Many Requests",
            "application/json",
            br#"{"error":"Chen's queue is full; try again shortly"}"#,
        );
    }
    response(
        stream,
        "202 Accepted",
        "application/json",
        json!({"id":id,"phase":"queued"}).to_string().as_bytes(),
    )
}
fn main() -> std::io::Result<()> {
    let port = std::env::var("CHEN_WEB_PORT")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(8787);
    let workers = match std::env::var("CHEN_WEB_WORKERS") {
        Ok(value) => value
            .parse::<usize>()
            .ok()
            .filter(|n| (1..=6).contains(n))
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "CHEN_WEB_WORKERS must be 1..6",
                )
            })?,
        Err(std::env::VarError::NotPresent) => 6,
        Err(error) => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, error)),
    };
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    println!(
        "Chen {} is playing at http://127.0.0.1:{port} ({workers} independent search workers) — Ctrl+C to stop",
        env!("CHEN_REVISION")
    );
    let jobs: Jobs = Arc::new(Mutex::new(HashMap::new()));
    let queued = Arc::new(AtomicUsize::new(0));
    let readers = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = mpsc::sync_channel(8);
    let receiver = Arc::new(Mutex::new(receiver));
    for _ in 0..workers {
        let receiver = receiver.clone();
        let worker_jobs = jobs.clone();
        let worker_queued = queued.clone();
        std::thread::spawn(move || worker(receiver, worker_jobs, worker_queued));
    }
    for connection in listener.incoming() {
        let mut stream = connection?;
        if readers.fetch_add(1, Ordering::SeqCst) >= 32 {
            readers.fetch_sub(1, Ordering::SeqCst);
            let _ = response(
                &mut stream,
                "503 Service Unavailable",
                "application/json",
                br#"{"error":"Too many connections"}"#,
            );
            continue;
        }
        let sender = sender.clone();
        let jobs = jobs.clone();
        let queued = queued.clone();
        let readers = readers.clone();
        std::thread::spawn(move || {
            if let Err(error) = handle(&mut stream, &sender, &jobs, &queued) {
                eprintln!("Request ended: {error}");
            }
            readers.fetch_sub(1, Ordering::SeqCst);
        });
    }
    Ok(())
}
