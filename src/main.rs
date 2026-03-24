use clap::{Parser, Subcommand};
use sae::client::EsaClient;
use sae::config::Config;
use sae::embedder::{Embed, Embedder, ModelPaths};

const HF_REPO: &str = "cl-nagoya/ruri-v3-310m";

#[derive(Parser)]
#[command(name = "sae", about = "esa semantic search CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Fetch and index esa posts
    Harvest {
        /// Team name
        team: String,
        /// Re-fetch all posts (ignore sync state)
        #[arg(long)]
        full: bool,
    },
    /// Semantic search over indexed posts
    Search {
        /// Search query
        query: String,
        /// Team name
        #[arg(long)]
        team: Option<String>,
        /// Max results (1-100)
        #[arg(long, default_value = "10")]
        limit: u32,
    },
    /// Get a post by number
    Get {
        /// Post number
        number: u32,
        /// Team name
        #[arg(long)]
        team: Option<String>,
    },
    /// Create a new post
    Create {
        /// Post title
        #[arg(long)]
        name: String,
        /// Post body (Markdown)
        #[arg(long)]
        body: Option<String>,
        /// Category path
        #[arg(long)]
        category: Option<String>,
        /// Tags
        #[arg(long)]
        tag: Vec<String>,
        /// Mark as WIP
        #[arg(long)]
        wip: bool,
        /// Team name
        #[arg(long)]
        team: Option<String>,
    },
    /// Update a post
    Update {
        /// Post number
        number: u32,
        /// New title
        #[arg(long)]
        name: Option<String>,
        /// New body (Markdown)
        #[arg(long)]
        body: Option<String>,
        /// New category path
        #[arg(long)]
        category: Option<String>,
        /// New tags (replaces existing)
        #[arg(long)]
        tag: Vec<String>,
        /// Team name
        #[arg(long)]
        team: Option<String>,
    },
    /// Archive a post
    Archive {
        /// Post number
        number: u32,
        /// Team name
        #[arg(long)]
        team: Option<String>,
    },
    /// Ship a WIP post (set wip=false)
    Ship {
        /// Post number
        number: u32,
        /// Team name
        #[arg(long)]
        team: Option<String>,
    },
    /// Download model and embed all chunks
    Embed {
        /// Team name
        team: String,
    },
    /// Show sync status
    Status {
        /// Team name (omit to show all teams)
        #[arg(long)]
        team: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("sae=info".parse()?),
        )
        .init();

    let cli = Cli::parse();
    let config = Config::load()?;

    match cli.command {
        Command::Harvest { team, full } => {
            let team = config.resolve_team(Some(&team))?;
            let client = EsaClient::from_env()?;
            let db_path = config.team_db_path(team)?;
            let db = sae::storage::Db::open(&db_path)?;
            let result = sae::sync::harvest(&client, &db, team, full).await?;
            println!("{result}");
        }
        Command::Search {
            query,
            team,
            limit,
        } => {
            let team = config.resolve_team(team.as_deref())?;
            let db_path = config.team_db_path(team)?;
            if !db_path.exists() {
                eprintln!("No data for team '{team}'. Run `sae harvest {team}` first.");
                std::process::exit(1);
            }
            let db = sae::storage::Db::open(&db_path)?;
            let mut embedder = try_load_embedder();
            let query_embedding = embedder.as_mut().and_then(|e| {
                match e.embed_query(&query) {
                    Ok(v) => Some(v),
                    Err(e) => {
                        eprintln!("Warning: embed_query failed: {e}");
                        None
                    }
                }
            });
            let results = sae::storage::hybrid_search(
                db.conn(),
                &query,
                query_embedding.as_deref(),
                limit,
            )?;
            if results.is_empty() {
                println!("No results for '{query}'");
            } else {
                for r in &results {
                    let section = r
                        .section_title
                        .as_deref()
                        .map(|s| format!(" > {s}"))
                        .unwrap_or_default();
                    println!(
                        "[{:.4}] {}{} (#{})  {}",
                        r.score, r.post_name, section, r.post_number, r.post_url
                    );
                    if !r.snippet.is_empty() {
                        println!("  {}", r.snippet.replace('\n', " "));
                    }
                }
            }
        }
        Command::Get { number, team } => {
            let team = config.resolve_team(team.as_deref())?;
            let client = EsaClient::from_env()?;
            let post = client.get_post(team, number).await?;
            if post.wip {
                println!("[WIP] {}", post.full_name);
            } else {
                println!("{}", post.full_name);
            }
            println!("{}", post.url);
            println!("---");
            println!("{}", post.body_md.as_deref().unwrap_or("(empty)"));
        }
        Command::Create {
            name,
            body,
            category,
            tag,
            wip,
            team,
        } => {
            let team = config.resolve_team(team.as_deref())?;
            let client = EsaClient::from_env()?;
            let post = client
                .create_post(team, &name, body.as_deref(), category.as_deref(), tag, wip)
                .await?;
            println!("Created: {} (#{}) {}", post.name, post.number, post.url);
        }
        Command::Update {
            number,
            name,
            body,
            category,
            tag,
            team,
        } => {
            let team = config.resolve_team(team.as_deref())?;
            let client = EsaClient::from_env()?;
            let tags = if tag.is_empty() { None } else { Some(tag) };
            let post = client
                .update_post(
                    team,
                    number,
                    name.as_deref(),
                    body.as_deref(),
                    category.as_deref(),
                    tags,
                    None,
                )
                .await?;
            println!("Updated: {} (#{}) {}", post.name, post.number, post.url);
        }
        Command::Embed { team } => {
            let team = config.resolve_team(Some(&team))?;
            let db_path = config.team_db_path(team)?;
            if !db_path.exists() {
                eprintln!("No data for team '{team}'. Run `sae harvest {team}` first.");
                std::process::exit(1);
            }
            let db = sae::storage::Db::open(&db_path)?;

            eprintln!("Checking model...");
            let paths = ensure_model()?;
            let mut embedder = Embedder::new(&paths)
                .map_err(|e| format!("Failed to load model: {e}"))?;
            eprintln!("Model ready");

            const BATCH_SIZE: u32 = 500;
            let mut total_added = 0u32;
            let mut total_failed = 0u32;
            let mut done = 0u32;
            loop {
                let batch = sae::storage::get_unembedded_chunks(db.conn(), BATCH_SIZE)?;
                if batch.is_empty() {
                    break;
                }
                if done == 0 {
                    eprintln!("Embedding chunks...");
                }
                let mut embeddings = Vec::with_capacity(batch.len());
                for (chunk_id, content) in &batch {
                    match embedder.embed_document(content) {
                        Ok(embedding) => embeddings.push((*chunk_id, embedding)),
                        Err(e) => {
                            eprintln!("Warning: chunk {chunk_id} failed: {e}");
                            total_failed += 1;
                        }
                    }
                    done += 1;
                    if done % 100 == 0 {
                        eprintln!("  {done} chunks processed");
                    }
                }
                if embeddings.is_empty() && !batch.is_empty() {
                    eprintln!("Error: all chunks in batch failed. Aborting.");
                    std::process::exit(1);
                }
                total_added += sae::storage::add_embeddings(db.conn(), &embeddings)?;
            }
            if done == 0 {
                println!("All chunks already embedded");
            } else {
                println!("Embedded {total_added} chunks ({total_failed} failed)");
            }
        }
        Command::Archive { number, team } => {
            let team = config.resolve_team(team.as_deref())?;
            let client = EsaClient::from_env()?;
            let post = client.get_post(team, number).await?;
            let current_category = post.category.as_deref().unwrap_or("");
            if current_category.starts_with("Archived/") || current_category == "Archived" {
                println!("Already archived: {} (#{}) {}", post.name, post.number, post.url);
            } else {
                let archived_category = if current_category.is_empty() {
                    "Archived".to_string()
                } else {
                    format!("Archived/{current_category}")
                };
                let post = client
                    .update_post(team, number, None, None, Some(&archived_category), None, None)
                    .await?;
                println!("Archived: {} (#{}) {}", post.name, post.number, post.url);
            }
        }
        Command::Ship { number, team } => {
            let team = config.resolve_team(team.as_deref())?;
            let client = EsaClient::from_env()?;
            let post = client
                .update_post(team, number, None, None, None, None, Some(false))
                .await?;
            println!("Shipped: {} (#{}) {}", post.name, post.number, post.url);
        }
        Command::Status { team } => {
            let teams: Vec<&str> = if let Some(ref t) = team {
                vec![config.resolve_team(Some(t))?]
            } else {
                config.teams.iter().map(String::as_str).collect()
            };
            for t in &teams {
                println!("--- {t} ---");
                match config.team_db_path(t) {
                    Ok(path) if path.exists() => {
                        let db = sae::storage::Db::open(&path)?;
                        let count = sae::storage::count_posts(db.conn())?;
                        let state = sae::storage::get_sync_state(db.conn())?;
                        println!("  Posts: {count}");
                        if let Some(s) = state {
                            println!(
                                "  Last sync: {} (total: {}, local: {})",
                                s.updated_at, s.total_count, s.local_count
                            );
                            if let Some(pg) = s.last_page {
                                println!("  Checkpoint: page {pg} (interrupted)");
                            }
                        } else {
                            println!("  Not yet synced");
                        }
                    }
                    Ok(path) => {
                        println!("  Not yet synced (no DB at {})", path.display());
                    }
                    Err(e) => {
                        println!("  Error: {e}");
                    }
                }
            }
        }
    }

    Ok(())
}

fn try_load_embedder() -> Option<Embedder> {
    let paths = cached_model_paths().or_else(|| {
        eprintln!("Note: embedding model not cached. Run `sae embed <team>` to enable semantic search.");
        None
    })?;
    match Embedder::new(&paths) {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!("Warning: failed to load embedding model: {e}");
            None
        }
    }
}

fn ensure_model() -> Result<ModelPaths, Box<dyn std::error::Error>> {
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(HF_REPO.to_string());
    let model = repo.get("model.safetensors")?;
    let config = repo.get("config.json")?;
    let tokenizer = repo.get("tokenizer.json")?;
    Ok(ModelPaths {
        model,
        config,
        tokenizer,
    })
}

fn cached_model_paths() -> Option<ModelPaths> {
    let cache = hf_hub::Cache::from_env();
    let repo = cache.repo(hf_hub::Repo::model(HF_REPO.to_string()));
    Some(ModelPaths {
        model: repo.get("model.safetensors")?,
        config: repo.get("config.json")?,
        tokenizer: repo.get("tokenizer.json")?,
    })
}
