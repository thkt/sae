use crate::native::backend_available;
pub use crate::native::gpu_available;
use crate::{embedding::Embed, native::Gemma};
use anyhow::{Context, Result, ensure};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    env,
    fmt::{self, Write as _},
    fs,
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process::{self, Command, Stdio},
    sync::{Arc, OnceLock},
    thread,
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
// Field order contributes to the persisted manifest fingerprint.
pub struct ModelSpec {
    pub model_id: String,
    pub revision: String,
    pub model_url: String,
    pub model_sha256: String,
    pub tokenizer_revision: String,
    pub tokenizer_url: String,
    pub tokenizer_sha256: String,
    pub dimensions: usize,
    pub max_tokens: usize,
    pub overlap_tokens: usize,
    pub query_prefix: String,
    pub document_prefix: String,
    pub pooling: String,
    pub implementation: String,
    pub backend_revision: String,
    pub bindings_revision: String,
}
impl Default for ModelSpec {
    fn default() -> Self {
        Self {
            model_id: "embeddinggemma-2-bf16".into(),
            revision: "bfcd298762cc34d0357ece5ebdd31791a3a374d8".into(),
            model_url: "https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/resolve/bfcd298762cc34d0357ece5ebdd31791a3a374d8/embeddinggemma-2-BF16.gguf".into(),
            model_sha256: "68bae29d62fb8c7d23e98d21fd4662753ddd636e6b62b8a70dfe059a9844f216".into(),
            tokenizer_revision: "914f7f89142e33e77833254d9c9b90c3cef7303b".into(),
            tokenizer_url: "https://huggingface.co/google/embeddinggemma-2/resolve/914f7f89142e33e77833254d9c9b90c3cef7303b/tokenizer.json".into(),
            tokenizer_sha256: "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4".into(),
            dimensions: 768,
            max_tokens: 8192,
            overlap_tokens: 2048,
            query_prefix: "task: search result | query: ".into(),
            document_prefix: "title: none | text: ".into(),
            pooling: "mean-l2-noncausal".into(),
            implementation: "sae-gemma-llama-v1".into(),
            backend_revision: "1b70350c92563449576bcbbf552994b1d1ed847d".into(),
            bindings_revision: "849e04446cb7b6a1fe3cee650e635a081ec072be".into(),
        }
    }
}
impl ModelSpec {
    pub fn validate(&self) -> Result<()> {
        for hash in [&self.model_sha256, &self.tokenizer_sha256] {
            ensure!(
                hash.len() == 64
                    && hash
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                "artifact SHA256 must be 64 lowercase hex characters"
            );
        }
        for revision in [&self.revision, &self.tokenizer_revision] {
            ensure!(
                revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit()),
                "model revisions must be pinned full commit IDs"
            );
        }
        ensure!(!self.model_id.is_empty(), "model_id must not be empty");
        ensure!(
            self.dimensions == 768 && self.max_tokens == 8192 && self.overlap_tokens == 2048,
            "unsupported model dimensions or token splitting contract"
        );
        ensure!(
            self.pooling == "mean-l2-noncausal" && self.implementation == "sae-gemma-llama-v1",
            "unsupported encoder implementation or pooling"
        );
        ensure!(
            self.backend_revision == "1b70350c92563449576bcbbf552994b1d1ed847d"
                && self.bindings_revision == "849e04446cb7b6a1fe3cee650e635a081ec072be",
            "manifest requires a different encoder backend; use its matching binary"
        );
        for url in [&self.model_url, &self.tokenizer_url] {
            ensure!(
                url.starts_with("https://"),
                "artifact downloads require HTTPS"
            );
        }
        Ok(())
    }
    pub fn identity(&self) -> String {
        digest_hex(&Sha256::digest(
            serde_json::to_vec(self).expect("model spec is serializable"),
        ))
    }
}

pub(crate) fn digest_hex(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(hex, "{byte:02x}").expect("writing to String is infallible");
    }
    hex
}

static SELECTED: OnceLock<ModelSpec> = OnceLock::new();
pub fn selected() -> &'static ModelSpec {
    SELECTED.get_or_init(ModelSpec::default)
}
pub fn hash_file(path: &Path) -> Result<String> {
    let mut h = Sha256::new();
    let mut f = fs::File::open(path)?;
    let mut buf = vec![0; 1024 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(digest_hex(&h.finalize()))
}
pub fn cache_dir() -> PathBuf {
    if let Some(p) = env::var_os("SAE_MODEL_DIR") {
        return p.into();
    }
    let base = env::var_os("HF_HOME")
        .map(PathBuf::from)
        .map(|p| p.join("sae"))
        .or_else(|| {
            env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .filter(|p| p.is_absolute())
                .map(|p| p.join("sae"))
        })
        .unwrap_or_else(|| {
            env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_default()
                .join(".cache/sae")
        });
    base.join("models").join(selected().identity())
}
pub fn cached_artifacts() -> Result<Option<PathBuf>> {
    let root = cache_dir();
    if !root.join("model.gguf").is_file() || !root.join("tokenizer.json").is_file() {
        return Ok(None);
    }
    ensure!(
        hash_file(&root.join("model.gguf"))? == selected().model_sha256,
        "model checksum mismatch"
    );
    ensure!(
        hash_file(&root.join("tokenizer.json"))? == selected().tokenizer_sha256,
        "tokenizer checksum mismatch"
    );
    Ok(Some(root))
}
pub fn load_verified() -> Result<Arc<dyn Embed>> {
    let root = cached_artifacts()?.context("model not installed")?;
    Ok(Arc::new(Gemma::load(
        &root.join("model.gguf"),
        &root.join("tokenizer.json"),
        selected().clone(),
    )?))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradedReason {
    NotInstalled,
    ProbeFailed,
    BackendUnavailable,
    #[cfg(test)]
    Disabled,
}
impl fmt::Display for DegradedReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::NotInstalled => "model not installed",
                Self::ProbeFailed => "model probe failed",
                Self::BackendUnavailable => "backend unavailable",
                #[cfg(test)]
                Self::Disabled => "model disabled",
            }
        )
    }
}
pub struct EmbedderDegraded(pub DegradedReason);
impl EmbedderDegraded {
    pub fn user_note(&self, download: &str) -> Option<String> {
        match self.0 {
            #[cfg(test)]
            DegradedReason::Disabled => None,
            DegradedReason::NotInstalled => Some(format!(
                "embedding model not installed; run `{download}` to enable semantic search"
            )),
            DegradedReason::ProbeFailed | DegradedReason::BackendUnavailable => {
                Some("embedding model unavailable; results from text search only".into())
            }
        }
    }
}
pub fn record_degraded(reason: DegradedReason, stage: &str) {
    tracing::warn!(%reason,stage,"semantic search unavailable");
}
pub fn degrade_with_warn<E: fmt::Display>(
    stage: &'static str,
    reason: DegradedReason,
) -> impl FnOnce(E) -> DegradedReason {
    move |e| {
        tracing::warn!(error=%e,stage,"model operation failed");
        reason
    }
}

pub fn handle_probe_if_needed() {
    if env::var_os("__SAE_GEMMA_PROBE").is_none() {
        return;
    }
    let result = (|| -> Result<()> {
        let spec: ModelSpec = serde_json::from_str(&env::var("__SAE_GEMMA_SPEC")?)?;
        spec.validate()?;
        SELECTED
            .set(spec)
            .map_err(|_| anyhow::anyhow!("model already selected"))?;
        if !backend_available()? {
            process::exit(2);
        }
        let model = load_verified().context("loading probe model")?;
        let v = model
            .embed_query("model probe")
            .context("encoding probe query")?;
        ensure!(
            v.len() == 768 && v.iter().all(|x| x.is_finite()),
            "invalid probe embedding"
        );
        Ok(())
    })();
    match result {
        Ok(()) => process::exit(0),
        Err(error) => {
            eprintln!("model probe failed: {error:#}");
            process::exit(1);
        }
    }
}
fn probe() -> Result<bool> {
    let mut command = Command::new(env::current_exe()?);
    command
        .env("__SAE_GEMMA_PROBE", "1")
        .env("__SAE_GEMMA_SPEC", serde_json::to_string(selected())?);
    run_probe(&mut command, Duration::from_secs(30))
}

fn run_probe(command: &mut Command, timeout: Duration) -> Result<bool> {
    // A file avoids pipe backpressure and inherited-pipe EOF waits on timeout.
    let mut stderr = tempfile::tempfile()?;
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(stderr.try_clone()?)
        .spawn()?;
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait()? {
            if status.code() == Some(2) {
                return Ok(false);
            }
            if !status.success() {
                anyhow::bail!(
                    "isolated model probe failed ({status}): {}",
                    probe_diagnostics(&mut stderr)?
                );
            }
            return Ok(true);
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!(
                "isolated model probe timed out: {}",
                probe_diagnostics(&mut stderr)?
            )
        }
        thread::sleep(Duration::from_millis(10));
    }
}
fn probe_diagnostics(stderr: &mut fs::File) -> Result<String> {
    let len = stderr.metadata()?.len();
    stderr.seek(SeekFrom::Start(len.saturating_sub(8192)))?;
    let mut bytes = Vec::new();
    stderr.take(8192).read_to_end(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).trim().to_owned())
}

pub fn try_load_embedder_default_logging() -> Result<Arc<dyn Embed>, DegradedReason> {
    match cached_artifacts() {
        Ok(None) => return Err(DegradedReason::NotInstalled),
        Err(error) => {
            tracing::warn!(%error,"model artifact validation failed");
            return Err(DegradedReason::ProbeFailed);
        }
        Ok(Some(_)) => {}
    }
    match probe() {
        Ok(false) => Err(DegradedReason::BackendUnavailable),
        Ok(true) => {
            load_verified().map_err(degrade_with_warn("model load", DegradedReason::ProbeFailed))
        }
        Err(error) => Err(degrade_with_warn(
            "model probe",
            DegradedReason::ProbeFailed,
        )(error)),
    }
}
#[derive(Debug, thiserror::Error)]
pub enum ModelDownloadError {
    #[error("model download failed: {0}")]
    DownloadFailed(String),
    #[error("model probe failed: {0:?}")]
    ProbeFailed(String),
    #[error("backend unavailable")]
    BackendUnavailable,
}

// Never replace a cache entry with incomplete or unverified bytes.
fn install_artifact(root: &Path, name: &str, expected: &str, reader: &mut impl Read) -> Result<()> {
    let mut temporary = tempfile::Builder::new()
        .prefix(&format!(".{name}."))
        .suffix(".download")
        .tempfile_in(root)?;
    io::copy(reader, temporary.as_file_mut())?;
    temporary.flush()?;
    temporary.as_file().sync_all()?;
    ensure!(
        hash_file(temporary.path())? == expected,
        "download checksum mismatch"
    );
    temporary
        .persist(root.join(name))
        .map_err(|error| error.error)?;
    Ok(())
}
/// Downloads verified artifacts without loading the inference backend.
pub fn download_model_artifacts() -> Result<(), ModelDownloadError> {
    selected()
        .validate()
        .map_err(|error| ModelDownloadError::DownloadFailed(error.to_string()))?;
    download_artifacts_at(&cache_dir(), selected())
        .map_err(|error| ModelDownloadError::DownloadFailed(error.to_string()))
}

fn download_artifacts_at(root: &Path, spec: &ModelSpec) -> Result<()> {
    fs::create_dir_all(root)?;
    let client = Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;
    for (name, url, expected) in [
        ("model.gguf", &spec.model_url, &spec.model_sha256),
        (
            "tokenizer.json",
            &spec.tokenizer_url,
            &spec.tokenizer_sha256,
        ),
    ] {
        let destination = root.join(name);
        if destination.exists() && hash_file(&destination)? == *expected {
            continue;
        }
        install_artifact(
            root,
            name,
            expected,
            &mut client.get(url).send()?.error_for_status()?,
        )?;
    }
    Ok(())
}

fn verify_download(run: impl FnOnce() -> Result<bool>) -> Result<(), ModelDownloadError> {
    match run() {
        Ok(true) => Ok(()),
        Ok(false) => Err(ModelDownloadError::BackendUnavailable),
        Err(error) => Err(ModelDownloadError::ProbeFailed(error.to_string())),
    }
}

pub fn download_and_verify_model() -> Result<(), ModelDownloadError> {
    download_model_artifacts()?;
    verify_download(probe)
}

#[cfg(test)]
mod tests {
    use super::{hash_file, install_artifact};
    use std::{
        env, fs,
        io::{self, Cursor, Read},
        path::PathBuf,
        process::{self, Command, Stdio},
        thread::{self, JoinHandle},
        time::{Duration, Instant},
    };
    use tempfile::tempdir;

    fn child_test(name: &str) -> Command {
        let mut command = Command::new(env::current_exe().unwrap());
        command.args(["--exact", name, "--nocapture"]);
        command
    }

    #[test]
    fn isolated_probe_child() {
        let Ok(mode) = env::var("__SAE_TEST_PROBE_MODE") else {
            return;
        };
        if mode == "failure" || mode == "large-stderr" {
            use std::io::Write as _;
            let mut stderr = io::stderr().lock();
            if mode == "large-stderr" {
                // Exceed typical OS pipe capacity before writing the diagnostic tail.
                stderr.write_all(&vec![b'x'; 256 * 1024]).unwrap();
            }
            writeln!(stderr, "\nSAE_TEST_PROBE_CAUSE: fixture backend failed").unwrap();
            stderr.flush().unwrap();
            process::exit(17);
        }
        if mode == "sleep" {
            let path = env::var_os("__SAE_TEST_PROBE_PID").unwrap();
            fs::write(path, process::id().to_string()).unwrap();
            thread::sleep(Duration::from_secs(60));
        }
        process::exit(mode.parse().unwrap_or(99));
    }

    #[test]
    fn isolated_probe_distinguishes_success_unavailable_and_failure() {
        for (code, expected) in [("0", Some(true)), ("2", Some(false)), ("1", None)] {
            let mut command = child_test("model::tests::isolated_probe_child");
            command.env("__SAE_TEST_PROBE_MODE", code);
            let result = super::run_probe(&mut command, Duration::from_secs(5));
            match expected {
                Some(value) => assert_eq!(result.unwrap(), value, "exit {code}"),
                None => assert!(
                    result
                        .unwrap_err()
                        .to_string()
                        .contains("isolated model probe failed"),
                    "exit {code} must be classified as a failed probe"
                ),
            }
        }
        let mut missing = Command::new("/sae-test-no-such-program");
        assert!(super::run_probe(&mut missing, Duration::from_secs(1)).is_err());
    }

    #[test]
    fn isolated_probe_failure_reports_exit_status_and_stderr_cause() {
        let mut command = child_test("model::tests::isolated_probe_child");
        command.env("__SAE_TEST_PROBE_MODE", "failure");
        let error = super::run_probe(&mut command, Duration::from_secs(5)).unwrap_err();
        let message = format!("{error:#}");
        assert!(
            message.contains("17"),
            "exit status must be visible: {message}"
        );
        assert!(
            message.contains("SAE_TEST_PROBE_CAUSE: fixture backend failed"),
            "child stderr cause must be visible: {message}"
        );
    }

    #[test]
    fn isolated_probe_drains_large_stderr_without_deadlock_and_retains_cause() {
        let mut command = child_test("model::tests::isolated_probe_child");
        command.env("__SAE_TEST_PROBE_MODE", "large-stderr");
        let started = Instant::now();
        let error = super::run_probe(&mut command, Duration::from_secs(5)).unwrap_err();
        let message = format!("{error:#}");
        assert!(started.elapsed() < Duration::from_secs(10));
        assert!(
            !message.contains("timed out"),
            "child must finish after writing stderr"
        );
        assert!(message.contains("17"), "exit status must be visible");
        assert!(
            message.contains("SAE_TEST_PROBE_CAUSE: fixture backend failed"),
            "diagnostic tail must survive output larger than pipe capacity"
        );
    }

    #[cfg(unix)]
    #[test]
    fn isolated_probe_signal_failure_reports_termination_and_cause() {
        // Terminate a real child by signal while the instrumented Rust test exits normally.
        let mut command = Command::new("/bin/sh");
        command.args([
            "-c",
            "printf '%s\\n' 'SAE_TEST_PROBE_CAUSE: fixture received termination' >&2; kill -TERM \"$$\"",
        ]);
        let error = super::run_probe(&mut command, Duration::from_secs(5)).unwrap_err();
        let message = format!("{error:#}");
        assert!(
            message.contains("signal") || message.contains("SIGTERM"),
            "signal termination must be visible: {message}"
        );
        assert!(
            message.contains("fixture received termination"),
            "{message}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn timed_out_probe_is_killed_and_reaped() {
        let dir = tempdir().unwrap();
        let pid_path = dir.path().join("child.pid");
        let mut command = child_test("model::tests::isolated_probe_child");
        command
            .env("__SAE_TEST_PROBE_MODE", "sleep")
            .env("__SAE_TEST_PROBE_PID", &pid_path);
        let started = Instant::now();
        let error = super::run_probe(&mut command, Duration::from_secs(2)).unwrap_err();
        assert!(error.to_string().contains("isolated model probe timed out"));
        assert!(started.elapsed() < Duration::from_secs(10));
        let pid = fs::read_to_string(pid_path).expect("child must have started before timeout");
        let status = Command::new("kill")
            .args(["-0", pid.trim()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(
            !status.success(),
            "timed out child must no longer exist, including as a zombie"
        );
    }

    #[test]
    fn cache_path_child() {
        if let Some(expected) = env::var_os("__SAE_TEST_EXPECTED_CACHE") {
            assert_eq!(super::cache_dir(), PathBuf::from(expected));
        }
    }

    #[test]
    fn cache_location_precedence_is_isolated_from_other_tests() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let identity = super::selected().identity();
        for (model, hf, xdg, home, expected) in [
            (
                Some("explicit"),
                Some("hf"),
                Some("xdg"),
                Some("home"),
                root.join("explicit"),
            ),
            (
                None,
                Some("hf"),
                Some("xdg"),
                Some("home"),
                root.join("hf/sae/models").join(&identity),
            ),
            (
                None,
                None,
                Some("xdg"),
                Some("home"),
                root.join("xdg/sae/models").join(&identity),
            ),
            (
                None,
                None,
                Some("relative"),
                Some("home"),
                root.join("home/.cache/sae/models").join(&identity),
            ),
            (
                None,
                None,
                None,
                None,
                PathBuf::from(".cache/sae/models").join(&identity),
            ),
        ] {
            let mut command = child_test("model::tests::cache_path_child");
            for key in ["SAE_MODEL_DIR", "HF_HOME", "XDG_CACHE_HOME", "HOME"] {
                command.env_remove(key);
            }
            for (key, value) in [
                ("SAE_MODEL_DIR", model),
                ("HF_HOME", hf),
                ("XDG_CACHE_HOME", xdg),
                ("HOME", home),
            ] {
                if let Some(value) = value {
                    if value == "relative" {
                        command.env(key, value);
                    } else {
                        command.env(key, root.join(value));
                    }
                }
            }
            let status = command
                .env("__SAE_TEST_EXPECTED_CACHE", &expected)
                .status()
                .unwrap();
            assert!(
                status.success(),
                "cache precedence must resolve to {}",
                expected.display()
            );
        }
    }

    #[test]
    fn model_spec_rejects_invalid_pins_artifacts_and_encoder_contract() {
        let base = serde_json::to_value(super::ModelSpec::default()).unwrap();
        super::ModelSpec::default().validate().unwrap();
        for (field, value, reason) in [
            (
                "model_sha256",
                serde_json::json!("0".repeat(63)),
                "artifact SHA256",
            ),
            (
                "tokenizer_sha256",
                serde_json::json!("A".repeat(64)),
                "artifact SHA256",
            ),
            (
                "revision",
                serde_json::json!("a".repeat(39)),
                "pinned full commit",
            ),
            (
                "tokenizer_revision",
                serde_json::json!("g".repeat(40)),
                "pinned full commit",
            ),
            ("model_id", serde_json::json!(""), "model_id"),
            ("dimensions", serde_json::json!(1024), "dimensions or token"),
            ("max_tokens", serde_json::json!(8193), "dimensions or token"),
            (
                "overlap_tokens",
                serde_json::json!(0),
                "dimensions or token",
            ),
            (
                "pooling",
                serde_json::json!("last"),
                "implementation or pooling",
            ),
            (
                "implementation",
                serde_json::json!("other"),
                "implementation or pooling",
            ),
            (
                "backend_revision",
                serde_json::json!("0".repeat(40)),
                "different encoder backend",
            ),
            (
                "bindings_revision",
                serde_json::json!("0".repeat(40)),
                "different encoder backend",
            ),
            (
                "model_url",
                serde_json::json!("http://example.invalid/model"),
                "HTTPS",
            ),
            (
                "tokenizer_url",
                serde_json::json!("file:///tokenizer.json"),
                "HTTPS",
            ),
        ] {
            let mut altered = base.clone();
            altered[field] = value;
            let spec: super::ModelSpec = serde_json::from_value(altered).unwrap();
            let error = spec.validate().unwrap_err().to_string();
            assert!(error.contains(reason), "{field}: {error}");
        }
        let mut unknown = base;
        unknown["unused_backend"] = serde_json::json!(true);
        assert!(serde_json::from_value::<super::ModelSpec>(unknown).is_err());
    }

    fn artifact_server(
        replies: Vec<(&'static str, u16, &'static [u8], usize)>,
    ) -> (String, JoinHandle<()>) {
        use std::io::{BufRead, BufReader};
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = format!("http://{}", listener.local_addr().unwrap());
        let worker = thread::spawn(move || {
            for (path, status, body, missing_bytes) in replies {
                let deadline = Instant::now() + Duration::from_secs(10);
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                            assert!(Instant::now() < deadline, "expected request {path}");
                            thread::sleep(Duration::from_millis(5));
                        }
                        Err(error) => panic!("accept: {error}"),
                    }
                };
                // macOS can inherit the listener's nonblocking mode; request reads need a blocking timeout.
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut request = BufReader::new(&stream);
                let mut line = String::new();
                request.read_line(&mut line).unwrap();
                assert_eq!(line.split_whitespace().nth(1), Some(path));
                loop {
                    line.clear();
                    request.read_line(&mut line).unwrap();
                    if line == "\r\n" || line.is_empty() {
                        break;
                    }
                }
                drop(request);
                use std::io::Write;
                write!(
                    stream,
                    "HTTP/1.1 {status} Test\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len() + missing_bytes
                )
                .unwrap();
                stream.write_all(body).unwrap();
            }
        });
        (address, worker)
    }

    fn local_artifact_spec(address: &str) -> super::ModelSpec {
        use sha2::{Digest, Sha256};
        super::ModelSpec {
            model_url: format!("{address}/model"),
            tokenizer_url: format!("{address}/tokenizer"),
            model_sha256: super::digest_hex(&Sha256::digest(b"verified model")),
            tokenizer_sha256: super::digest_hex(&Sha256::digest(b"verified tokenizer")),
            ..super::ModelSpec::default()
        }
    }

    #[test]
    fn artifact_transport_fetches_repairs_and_reuses_only_checksum_verified_cache() {
        let (address, server) = artifact_server(vec![
            ("/model", 200, b"verified model", 0),
            ("/tokenizer", 200, b"verified tokenizer", 0),
            ("/model", 200, b"verified model", 0),
        ]);
        let spec = local_artifact_spec(&address);
        let dir = tempdir().unwrap();
        let cache = dir.path().join("new/cache");
        super::download_artifacts_at(&cache, &spec).unwrap();
        assert_eq!(
            fs::read(cache.join("model.gguf")).unwrap(),
            b"verified model"
        );
        assert_eq!(
            fs::read(cache.join("tokenizer.json")).unwrap(),
            b"verified tokenizer"
        );
        // The server accepts only a model repair next, so cache hits must not fetch either artifact.
        super::download_artifacts_at(&cache, &spec).unwrap();
        fs::write(cache.join("model.gguf"), b"corrupt cache").unwrap();
        super::download_artifacts_at(&cache, &spec).unwrap();
        server.join().unwrap();
        assert_eq!(
            fs::read(cache.join("model.gguf")).unwrap(),
            b"verified model"
        );
        assert_eq!(
            fs::read(cache.join("tokenizer.json")).unwrap(),
            b"verified tokenizer"
        );
        assert_eq!(fs::read_dir(cache).unwrap().count(), 2);
    }

    #[test]
    fn artifact_transport_rejects_http_failure_corruption_and_truncation_preserving_old_bytes() {
        for (status, body, missing_bytes, expected_error) in [
            (404, &b"not found"[..], 0, "404"),
            (200, &b"wrong checksum"[..], 0, "checksum"),
            (200, &b"partial"[..], 100, ""),
        ] {
            let (address, server) = artifact_server(vec![("/model", status, body, missing_bytes)]);
            let spec = local_artifact_spec(&address);
            let dir = tempdir().unwrap();
            let model = dir.path().join("model.gguf");
            fs::write(&model, b"retained model").unwrap();
            let error = super::download_artifacts_at(dir.path(), &spec).unwrap_err();
            server.join().unwrap();
            assert!(error.to_string().contains(expected_error), "{error}");
            assert_eq!(fs::read(model).unwrap(), b"retained model");
            assert!(!dir.path().join("tokenizer.json").exists());
            assert_eq!(
                fs::read_dir(dir.path()).unwrap().count(),
                1,
                "no incomplete files survive"
            );
        }
    }

    #[test]
    fn download_verification_reports_backend_and_probe_failures_distinctly() {
        super::verify_download(|| Ok(true)).unwrap();
        assert!(matches!(
            super::verify_download(|| Ok(false)),
            Err(super::ModelDownloadError::BackendUnavailable)
        ));
        let error = super::verify_download(|| Err(anyhow::anyhow!("child crashed"))).unwrap_err();
        assert!(
            matches!(error, super::ModelDownloadError::ProbeFailed(reason) if reason == "child crashed")
        );
    }

    #[test]
    fn file_digest_matches_sha256_reference_vector_and_missing_files_fail() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sample");
        fs::write(&path, b"abc").unwrap();
        assert_eq!(
            hash_file(&path).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert!(hash_file(&dir.path().join("missing")).is_err());
    }

    #[test]
    fn degraded_model_reasons_give_actionable_text_search_guidance() {
        use super::{DegradedReason, EmbedderDegraded};
        assert_eq!(
            EmbedderDegraded(DegradedReason::NotInstalled)
                .user_note("sae model download")
                .as_deref(),
            Some(
                "embedding model not installed; run `sae model download` to enable semantic search"
            )
        );
        for reason in [
            DegradedReason::ProbeFailed,
            DegradedReason::BackendUnavailable,
        ] {
            assert_eq!(
                EmbedderDegraded(reason).user_note("unused").as_deref(),
                Some("embedding model unavailable; results from text search only")
            );
        }
        assert!(
            EmbedderDegraded(DegradedReason::Disabled)
                .user_note("unused")
                .is_none()
        );
        for (reason, expected) in [
            (DegradedReason::NotInstalled, "model not installed"),
            (DegradedReason::ProbeFailed, "model probe failed"),
            (DegradedReason::BackendUnavailable, "backend unavailable"),
            (DegradedReason::Disabled, "model disabled"),
        ] {
            assert_eq!(reason.to_string(), expected);
            super::record_degraded(reason, "test");
            assert_eq!(
                super::degrade_with_warn("test", reason)("probe failed"),
                reason
            );
        }
    }

    #[test]
    fn artifact_install_activates_only_verified_complete_download() {
        let dir = tempdir().unwrap();
        let sample = dir.path().join("sample");
        fs::write(&sample, b"new complete model").unwrap();
        let expected = hash_file(&sample).unwrap();
        fs::remove_file(sample).unwrap();
        let target = dir.path().join("model.gguf");
        fs::write(&target, b"old retained model").unwrap();
        install_artifact(
            dir.path(),
            "model.gguf",
            &expected,
            &mut Cursor::new(b"new complete model"),
        )
        .unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"new complete model");
        assert_eq!(
            fs::read_dir(dir.path()).unwrap().count(),
            1,
            "no temporary download remains"
        );
    }

    #[test]
    fn artifact_checksum_failure_preserves_existing_model_and_cleans_partial_file() {
        let dir = tempdir().unwrap();
        let target = dir.path().join("model.gguf");
        fs::write(&target, b"old retained model").unwrap();
        let expected = hash_file(&target).unwrap();
        let error = install_artifact(
            dir.path(),
            "model.gguf",
            &expected,
            &mut Cursor::new(b"corrupt replacement"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("checksum"));
        assert_eq!(fs::read(target).unwrap(), b"old retained model");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn interrupted_download_preserves_existing_model_and_cleans_partial_file() {
        struct Interrupted {
            read_once: bool,
        }
        impl Read for Interrupted {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                if self.read_once {
                    Err(io::Error::new(
                        io::ErrorKind::ConnectionReset,
                        "connection lost",
                    ))
                } else {
                    self.read_once = true;
                    buffer[..7].copy_from_slice(b"partial");
                    Ok(7)
                }
            }
        }
        let dir = tempdir().unwrap();
        let target = dir.path().join("model.gguf");
        fs::write(&target, b"old retained model").unwrap();
        let expected = hash_file(&target).unwrap();
        let result = install_artifact(
            dir.path(),
            "model.gguf",
            &expected,
            &mut Interrupted { read_once: false },
        );
        assert!(result.is_err());
        assert_eq!(fs::read(target).unwrap(), b"old retained model");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }
}
