use crate::{
    embedding::{ChunkedEmbedding, EMBEDDING_DIMS, Embed, EmbedError, EmbedOptions},
    model::ModelSpec,
};
use anyhow::{Context, Result, ensure};
use llama_cpp_2::{
    context::params::{LlamaAttentionType, LlamaContextParams, LlamaPoolingType},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{LlamaModel, params::LlamaModelParams},
    token::LlamaToken,
};
use std::{
    num::NonZeroU32,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
    thread,
};
use tokenizers::Tokenizer;

pub const MAX_TOKENS: usize = 8192;
static BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();
fn backend() -> Result<&'static LlamaBackend> {
    BACKEND
        .get_or_init(|| {
            LlamaBackend::init()
                .map(|mut b| {
                    b.void_logs();
                    b
                })
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| anyhow::anyhow!(e.clone()))
}
pub fn gpu_available() -> Result<bool> {
    Ok(backend()?.supports_gpu_offload())
}

pub fn capacity_for(tokens: usize) -> Result<u32> {
    ensure!(
        (1..=MAX_TOKENS).contains(&tokens),
        "context outside 1..8192 tokens"
    );
    Ok(u32::try_from(tokens.max(512).next_power_of_two())?)
}

pub struct Gemma {
    model: Mutex<LlamaModel>,
    tokenizer: OnceLock<Result<Tokenizer, String>>,
    tokenizer_path: PathBuf,
    spec: ModelSpec,
}
struct DocumentWindow {
    tokens: Vec<LlamaToken>,
    byte_start: usize,
    byte_end: usize,
}
impl Gemma {
    pub fn load(path: &Path, tokenizer: &Path, spec: ModelSpec) -> Result<Self> {
        spec.validate()?;
        let b = backend()?;
        let model = LlamaModel::load_from_file(
            b,
            path,
            &LlamaModelParams::default().with_n_gpu_layers(1000),
        )?;
        ensure!(
            model.n_embd_out() == i32::try_from(EMBEDDING_DIMS)?,
            "expected projected 768 dimensions"
        );
        Ok(Self {
            model: Mutex::new(model),
            tokenizer: OnceLock::new(),
            tokenizer_path: tokenizer.to_path_buf(),
            spec,
        })
    }

    fn offset_tokenizer(&self) -> Result<&Tokenizer> {
        self.tokenizer
            .get_or_init(|| Tokenizer::from_file(&self.tokenizer_path).map_err(|e| e.to_string()))
            .as_ref()
            .map_err(|e| anyhow::anyhow!(e.clone()))
    }

    #[allow(clippy::cast_possible_truncation)]
    fn encode(
        model: &LlamaModel,
        inputs: &[Vec<LlamaToken>],
        options: &EmbedOptions,
    ) -> Result<Vec<Vec<f32>>> {
        let total = inputs.iter().map(Vec::len).sum::<usize>();
        ensure!(
            !inputs.is_empty() && inputs.len() <= 32 && inputs.iter().all(|s| !s.is_empty()),
            "invalid embedding batch"
        );
        let capacity = capacity_for(total)?;
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(capacity))
            .with_n_batch(capacity)
            .with_n_ubatch(capacity)
            .with_n_seq_max(32)
            .with_n_threads(4)
            .with_n_threads_batch(4)
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Mean)
            .with_attention_type(LlamaAttentionType::NonCausal);
        let mut ctx = model.new_context(backend()?, params)?;
        let mut batch = LlamaBatch::new(total, 1);
        for (i, tokens) in inputs.iter().enumerate() {
            batch.add_sequence(tokens, i32::try_from(i)?, true)?;
        }
        ctx.encode(&mut batch)?;
        let result = (0..inputs.len())
            .map(|i| {
                let v = ctx.embeddings_seq_ith(i32::try_from(i)?)?;
                ensure!(
                    v.len() == EMBEDDING_DIMS && v.iter().all(|x| x.is_finite()),
                    "invalid embedding"
                );
                let norm = v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
                ensure!(norm > 0., "zero embedding");
                Ok(v.iter().map(|x| (f64::from(*x) / norm) as f32).collect())
            })
            .collect();
        if let Some(pause) = options.forward_pause {
            thread::sleep(pause);
        }
        result
    }

    fn query_tokens(
        &self,
        model: &LlamaModel,
        text: &str,
        prefix: &str,
    ) -> Result<Vec<LlamaToken>> {
        let mut t = model
            .vocab()
            .tokenize(format!("{prefix}{text}").as_bytes(), true, false);
        ensure!(!t.is_empty(), "empty token sequence");
        if t.len() > MAX_TOKENS {
            t.truncate(MAX_TOKENS);
            let eos = model.vocab().eos();
            ensure!(eos.0 >= 0, "missing EOS token");
            t[MAX_TOKENS - 1] = eos;
        }
        Ok(t)
    }

    fn document_tokens(&self, model: &LlamaModel, text: &str) -> Result<Vec<DocumentWindow>> {
        let tokenize = |s: &str| {
            model.vocab().tokenize(
                format!("{}{s}", self.spec.document_prefix).as_bytes(),
                true,
                false,
            )
        };
        let full = tokenize(text);
        if full.len() <= MAX_TOKENS {
            ensure!(!full.is_empty(), "empty token sequence");
            return Ok(vec![DocumentWindow {
                tokens: full,
                byte_start: 0,
                byte_end: text.len(),
            }]);
        }
        let raw = self
            .offset_tokenizer()?
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let offsets = raw.get_offsets();
        let count = offsets.len();
        let budget = MAX_TOKENS
            .checked_sub(tokenize("").len())
            .context("prefix exceeds token limit")?;
        ensure!(
            budget > self.spec.overlap_tokens && count > 0,
            "invalid token split budget"
        );
        let mut result = Vec::new();
        let mut start = 0;
        // Re-tokenize offset windows with the prefix to enforce the native token budget.
        while start < count {
            let mut end = (start + budget).min(count);
            let lo = if start == 0 { 0 } else { offsets[start].0 };
            let (hi, tokens) = loop {
                ensure!(end > start, "document cannot fit");
                let hi = if end == count {
                    text.len()
                } else {
                    offsets[end - 1].1
                };
                let slice = text
                    .get(lo..hi)
                    .context("tokenizer offset is not UTF-8 aligned")?;
                let t = tokenize(slice);
                if t.len() <= MAX_TOKENS {
                    break (hi, t);
                }
                end -= 1;
            };
            result.push(DocumentWindow {
                tokens,
                byte_start: lo,
                byte_end: hi,
            });
            if end == count {
                break;
            }
            let next = end.saturating_sub(self.spec.overlap_tokens);
            ensure!(next > start, "token splitting made no progress");
            start = next;
        }
        Ok(result)
    }
}

impl Embed for Gemma {
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.embed_text(text, &self.spec.query_prefix)
    }
    fn embed_text(&self, text: &str, prefix: &str) -> Result<Vec<f32>, EmbedError> {
        let m = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("model lock poisoned"))?;
        let t = self.query_tokens(&m, text, prefix)?;
        Ok(Self::encode(&m, &[t], &EmbedOptions::default())?.remove(0))
    }
    fn embed_document(&self, text: &str) -> Result<ChunkedEmbedding, EmbedError> {
        Ok(self.embed_documents_batch(&[text])?.remove(0))
    }
    fn embed_documents_batch(&self, texts: &[&str]) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        self.embed_documents_batch_with_options(texts, &EmbedOptions::default())
    }
    fn embed_documents_batch_with_options(
        &self,
        texts: &[&str],
        options: &EmbedOptions,
    ) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        let m = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("model lock poisoned"))?;
        let mut out: Vec<Vec<Vec<f32>>> = texts.iter().map(|_| Vec::new()).collect();
        let budget = options
            .token_budget
            .unwrap_or(MAX_TOKENS)
            .clamp(1, MAX_TOKENS);
        let mut batch = Vec::new();
        let mut owners = Vec::new();
        let mut count = 0;
        let flush = |batch: &mut Vec<Vec<LlamaToken>>,
                     owners: &mut Vec<usize>,
                     out: &mut Vec<Vec<Vec<f32>>>|
         -> Result<()> {
            if batch.is_empty() {
                return Ok(());
            }
            for (owner, vector) in owners.drain(..).zip(Self::encode(&m, batch, options)?) {
                out[owner].push(vector);
            }
            batch.clear();
            Ok(())
        };
        for (i, text) in texts.iter().enumerate() {
            for window in self.document_tokens(&m, text)? {
                text.get(window.byte_start..window.byte_end)
                    .context("invalid window coverage")?;
                let tokens = window.tokens;
                if !batch.is_empty() && (batch.len() == 32 || count + tokens.len() > budget) {
                    flush(&mut batch, &mut owners, &mut out)?;
                    count = 0;
                }
                count += tokens.len();
                batch.push(tokens);
                owners.push(i);
            }
        }
        flush(&mut batch, &mut owners, &mut out)?;
        out.into_iter().map(ChunkedEmbedding::try_new).collect()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn native_context_capacity_handles_limit_and_rejects_outside_contract() {
        use super::capacity_for;
        for (tokens, expected) in [
            (1, 512),
            (511, 512),
            (512, 512),
            (513, 1024),
            (4097, 8192),
            (8192, 8192),
        ] {
            assert_eq!(capacity_for(tokens).unwrap(), expected);
        }
        assert!(capacity_for(0).is_err());
        assert!(capacity_for(8193).is_err());
        assert!(capacity_for(usize::MAX).is_err());
    }

    #[test]
    #[ignore = "requires pinned model/tokenizer in SAE_MODEL_DIR; no inference or downloads"]
    fn real_document_windows_preserve_utf8_coverage_overlap_and_token_boundaries() {
        use super::{Gemma, MAX_TOKENS};
        use crate::model::{cached_artifacts, selected};
        let root = cached_artifacts()
            .unwrap()
            .expect("pinned artifacts required");
        let gemma = Gemma::load(
            &root.join("model.gguf"),
            &root.join("tokenizer.json"),
            selected().clone(),
        )
        .unwrap();
        let model = gemma.model.lock().unwrap();
        for target in [8191_usize, 8192, 8193] {
            let text = (target - 40..=target)
                .map(|count| "a ".repeat(count))
                .find(|text| {
                    model
                        .vocab()
                        .tokenize(
                            format!("{}{text}", selected().document_prefix).as_bytes(),
                            true,
                            false,
                        )
                        .len()
                        == target
                })
                .expect("fixture reaches exact tokenizer boundary");
            let windows = gemma.document_tokens(&model, &text).unwrap();
            assert_eq!(
                windows.len(),
                if target <= MAX_TOKENS { 1 } else { 2 },
                "boundary {target}"
            );
            assert_eq!(windows.first().unwrap().byte_start, 0);
            assert_eq!(windows.last().unwrap().byte_end, text.len());
            for window in &windows {
                assert!(window.tokens.len() <= MAX_TOKENS);
            }
        }

        for target in [8191_usize, 8192, 8193] {
            let text = (target - 40..=target)
                .map(|count| "a ".repeat(count))
                .find(|text| {
                    model
                        .vocab()
                        .tokenize(
                            format!("{}{text}", selected().query_prefix).as_bytes(),
                            true,
                            false,
                        )
                        .len()
                        == target
                })
                .expect("query fixture reaches exact tokenizer boundary");
            let query = gemma
                .query_tokens(&model, &text, &selected().query_prefix)
                .unwrap();
            assert_eq!(query.len(), target.min(MAX_TOKENS));
            if target > MAX_TOKENS {
                assert_eq!(
                    query.last(),
                    Some(&model.vocab().eos()),
                    "truncated query retains EOS"
                );
            }
        }

        let long = format!(
            "  {} 終端TAIL\n",
            "日本語🙂 e\u{301} Rust token ".repeat(2500)
        );
        let windows = gemma.document_tokens(&model, &long).unwrap();
        assert!(windows.len() > 1);
        let raw = gemma
            .offset_tokenizer()
            .unwrap()
            .encode(long.as_str(), false)
            .unwrap();
        let offsets = raw.get_offsets();
        let mut covered_until = 0;
        for (index, window) in windows.iter().enumerate() {
            assert!(
                window.byte_start <= covered_until,
                "no source bytes may be skipped"
            );
            assert!(
                window.byte_end > covered_until,
                "every window must make progress"
            );
            let text = long
                .get(window.byte_start..window.byte_end)
                .expect("window boundaries preserve UTF-8");
            assert_eq!(
                window.tokens,
                model.vocab().tokenize(
                    format!("{}{text}", selected().document_prefix).as_bytes(),
                    true,
                    false
                ),
                "each source window must retain its document prefix"
            );
            assert!((1..=MAX_TOKENS).contains(&window.tokens.len()));
            if index > 0 {
                let previous = &windows[index - 1];
                assert!(
                    window.byte_start < previous.byte_end,
                    "adjacent windows overlap"
                );
                let previous_end = offsets
                    .iter()
                    .rposition(|(_, end)| *end == previous.byte_end)
                    .expect("previous window ends on tokenizer offset")
                    + 1;
                let required_start = offsets[previous_end - selected().overlap_tokens].0;
                assert_eq!(
                    window.byte_start, required_start,
                    "overlap must retain the configured 2048 source tokens"
                );
            }
            covered_until = window.byte_end;
        }
        assert_eq!(
            covered_until,
            long.len(),
            "leading whitespace and final marker are preserved"
        );
    }
}
