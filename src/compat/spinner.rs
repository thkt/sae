use std::io::{IsTerminal, Write, stderr};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::Duration;

const FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const TICK_MS: u64 = 80;

pub struct Spinner {
    done: Arc<AtomicBool>,
    message: Arc<Mutex<String>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Spinner {
    pub(crate) fn new(msg: &str) -> Self {
        Self::new_with_tty(msg, stderr().is_terminal())
    }

    fn new_with_tty(msg: &str, is_tty: bool) -> Self {
        let done = Arc::new(AtomicBool::new(false));
        let message = Arc::new(Mutex::new(msg.to_owned()));

        let thread = if is_tty {
            let done = Arc::clone(&done);
            let message = Arc::clone(&message);
            Some(thread::spawn(move || {
                let mut err = stderr();
                let mut i = 0;
                loop {
                    if done.load(Ordering::Relaxed) {
                        break;
                    }
                    let msg = message.lock().map(|m| m.clone()).unwrap_or_default();
                    let _ = write!(err, "\r\x1b[2K{} {}", FRAMES[i % FRAMES.len()], msg);
                    let _ = err.flush();
                    thread::sleep(Duration::from_millis(TICK_MS));
                    i += 1;
                }
            }))
        } else {
            None
        };

        Self {
            done,
            message,
            thread,
        }
    }

    pub(crate) fn set_message(&self, msg: &str) {
        if let Ok(mut m) = self.message.lock() {
            *m = msg.to_owned();
        }
    }

    pub(crate) fn finish(self, msg: &str) {
        self.finish_with_detail(msg, None);
    }

    pub(crate) fn finish_with_detail(self, main: &str, detail: Option<&str>) {
        // Drop first so the frame-thread clears the spinner line before `done` writes `✓`.
        drop(self);
        done(main);
        if let Some(d) = detail {
            eprintln!("  {d}");
        }
    }

    pub(crate) fn cancel(self) {}
}

pub(crate) fn done(msg: &str) {
    eprintln!("\x1b[32m✓\x1b[0m {msg}");
}

impl Drop for Spinner {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
            eprint!("\r\x1b[2K");
            let _ = stderr().flush();
        }
    }
}

pub(crate) fn embed_with_spinners<M, R, E>(
    pending: u32,
    load_model: impl FnOnce(&dyn Fn(&str)) -> Result<M, E>,
    finish_msg: impl FnOnce(&R) -> String,
    run_embed: impl FnOnce(M, &dyn Fn(&str)) -> Result<R, E>,
) -> Result<Option<R>, E> {
    if pending == 0 {
        return Ok(None);
    }
    let model = with_spinner("Loading model...", |_| "Model ready".to_owned(), load_model)?;
    let result = with_spinner(
        &format!("Embedding... 0/{pending} chunks"),
        finish_msg,
        |update| run_embed(model, update),
    )?;
    Ok(Some(result))
}

pub(crate) fn with_spinner<T, E>(
    start: &str,
    finish_msg: impl FnOnce(&T) -> String,
    work: impl FnOnce(&dyn Fn(&str)) -> Result<T, E>,
) -> Result<T, E> {
    let spinner = Spinner::new(start);
    let result = {
        let update = |msg: &str| spinner.set_message(msg);
        work(&update)
    };
    match result {
        Ok(val) => {
            let msg = finish_msg(&val);
            spinner.finish(&msg);
            Ok(val)
        }
        Err(e) => {
            spinner.cancel();
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Spinner, embed_with_spinners, with_spinner};
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    #[test]
    fn zero_pending_skips_model_and_work_and_failure_stops_following_stage() {
        let result: Result<Option<()>, &str> = embed_with_spinners(
            0,
            |_| panic!("no model needed"),
            |_| panic!("no finish message needed"),
            |(), _| panic!("no embedding needed"),
        );
        assert_eq!(result, Ok(None));
        let result = embed_with_spinners(
            1,
            |_| Err("load failed"),
            |_: &()| panic!("no completion on failure"),
            |(), _| panic!("no work after failed model load"),
        );
        assert_eq!(result, Err("load failed"));
        let result = with_spinner(
            "work",
            |_: &()| panic!("no completion on failure"),
            |update| {
                update("started");
                Err("inference failed")
            },
        );
        assert_eq!(result, Err("inference failed"));
    }

    #[test]
    fn spinner_updates_shared_message_and_stops_worker_before_finishing() {
        let spinner = Spinner::new_with_tty("initial", true);
        let done: Arc<AtomicBool> = Arc::clone(&spinner.done);
        spinner.set_message("updated");
        assert_eq!(*spinner.message.lock().unwrap(), "updated");
        spinner.finish_with_detail("finished", Some("details"));
        assert!(
            done.load(Ordering::Relaxed),
            "completion must stop the worker"
        );
    }
}
