use clap::{Parser, Subcommand};
use rurico::embed::{Embed, Embedder};
use sae::client::{CreatePostParams, EsaClient, UpdatePostParams};
use sae::config::Config;

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

type AppError = Box<dyn std::error::Error>;

fn resolve_client<'a>(config: &'a Config, team: Option<&'a str>) -> Result<(&'a str, EsaClient), AppError> {
    let team = config.resolve_team(team)?;
    let client = EsaClient::from_env()?;
    Ok((team, client))
}

fn require_db(config: &Config, team: &str) -> Result<sae::storage::Db, AppError> {
    let db_path = config.team_db_path(team)?;
    if !db_path.exists() {
        return Err(format!("No data for team '{team}'. Run `sae harvest {team}` first.").into());
    }
    Ok(sae::storage::Db::open(&db_path)?)
}

fn archive_category(current: Option<&str>) -> Option<String> {
    let current = current.unwrap_or("");
    if current.starts_with("Archived/") || current == "Archived" {
        return None;
    }
    Some(if current.is_empty() {
        "Archived".to_string()
    } else {
        format!("Archived/{current}")
    })
}

#[tokio::main]
async fn main() -> Result<(), AppError> {
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
            let (team, client) = resolve_client(&config, Some(&team))?;
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
            let db = require_db(&config, team)?;
            let embedder = try_load_embedder();
            let query_embedding = embedder.as_ref().and_then(|e| {
                match e.embed_query(&query) {
                    Ok(v) => Some(v),
                    Err(e) => {
                        tracing::warn!(error = %e, "embed_query failed, falling back to FTS");
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
            let (team, client) = resolve_client(&config, team.as_deref())?;
            let post = client.get_post(team, number).await?;
            print_post_yaml(&post);
        }
        Command::Create {
            name,
            body,
            category,
            tag,
            wip,
            team,
        } => {
            let (team, client) = resolve_client(&config, team.as_deref())?;
            let params = CreatePostParams {
                name: &name,
                body_md: body.as_deref(),
                category: category.as_deref(),
                tags: tag,
                wip,
            };
            let post = client.create_post(team, &params).await?;
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
            let (team, client) = resolve_client(&config, team.as_deref())?;
            let tags = if tag.is_empty() { None } else { Some(tag) };
            let params = UpdatePostParams {
                name: name.as_deref(),
                body_md: body.as_deref(),
                category: category.as_deref(),
                tags,
                ..Default::default()
            };
            let post = client.update_post(team, number, &params).await?;
            println!("Updated: {} (#{}) {}", post.name, post.number, post.url);
        }
        Command::Embed { team } => {
            let team = config.resolve_team(Some(&team))?;
            let db = require_db(&config, team)?;

            eprintln!("Checking model...");
            let paths = rurico::embed::download_model()
                .map_err(|e| format!("Failed to download model: {e}"))?;
            let embedder = Embedder::new(&paths)
                .map_err(|e| format!("Failed to load model: {e}"))?;
            eprintln!("Model ready");

            const BATCH_SIZE: u32 = 500;
            let mut total_added = 0u32;
            let mut done = 0u32;
            loop {
                let batch = sae::storage::get_unembedded_chunks(db.conn(), BATCH_SIZE)?;
                if batch.is_empty() {
                    break;
                }
                if done == 0 {
                    eprintln!("Embedding chunks...");
                }
                let texts: Vec<&str> = batch.iter().map(|(_, content)| content.as_str()).collect();
                let batch_len = batch.len() as u32;
                let embs = embedder
                    .embed_documents_batch(&texts)
                    .map_err(|e| format!("Batch embedding failed: {e}"))?;
                let embeddings: Vec<(i64, Vec<f32>)> = batch
                    .iter()
                    .map(|(id, _)| *id)
                    .zip(embs)
                    .collect();
                total_added += sae::storage::add_embeddings(db.conn(), &embeddings)?;
                done += batch_len;
                eprintln!("  {done} chunks processed");
            }
            if done == 0 {
                println!("All chunks already embedded");
            } else {
                println!("Embedded {total_added} chunks");
            }
        }
        Command::Archive { number, team } => {
            let (team, client) = resolve_client(&config, team.as_deref())?;
            let post = client.get_post(team, number).await?;
            match archive_category(post.category.as_deref()) {
                None => {
                    println!("Already archived: {} (#{}) {}", post.name, post.number, post.url);
                }
                Some(new_category) => {
                    let params = UpdatePostParams {
                        category: Some(&new_category),
                        ..Default::default()
                    };
                    let post = client.update_post(team, number, &params).await?;
                    println!("Archived: {} (#{}) {}", post.name, post.number, post.url);
                }
            }
        }
        Command::Ship { number, team } => {
            let (team, client) = resolve_client(&config, team.as_deref())?;
            let params = UpdatePostParams {
                wip: Some(false),
                ..Default::default()
            };
            let post = client.update_post(team, number, &params).await?;
            println!("Shipped: {} (#{}) {}", post.name, post.number, post.url);
        }
        Command::Status { team } => {
            let target_teams: Vec<&str> = if let Some(ref t) = team {
                vec![config.resolve_team(Some(t))?]
            } else {
                config
                    .teams
                    .iter()
                    .filter(|t| sae::config::validate_team_name(t).is_ok())
                    .map(String::as_str)
                    .collect()
            };
            for t in &target_teams {
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

fn print_post_yaml(post: &sae::client::EsaPost) {
    fn yaml_escape(s: &str) -> String {
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\t', "\\t")
    }
    println!("---");
    println!("title: \"{}\"", yaml_escape(&post.full_name));
    if let Some(ref cat) = post.category {
        if !cat.is_empty() {
            println!("category: \"{}\"", yaml_escape(cat));
        }
    }
    if !post.tags.is_empty() {
        let tags: Vec<String> = post.tags.iter().map(|t| format!("\"{}\"", yaml_escape(t))).collect();
        println!("tags: [{}]", tags.join(", "));
    }
    println!("author: \"@{}\"", post.created_by.screen_name);
    if post.updated_by.screen_name != post.created_by.screen_name {
        println!("updated_by: \"@{}\"", post.updated_by.screen_name);
    }
    println!("updated_at: \"{}\"", post.updated_at);
    if post.wip {
        println!("wip: true");
    }
    println!("number: {}", post.number);
    println!("url: {}", post.url);
    println!("---");
    println!();
    println!("{}", post.body_md.as_deref().unwrap_or("(empty)"));
}

fn try_load_embedder() -> Option<Embedder> {
    let paths = match rurico::embed::download_model() {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!(error = %e, "embedding model not available");
            eprintln!("Note: embedding model not available. Run `sae embed <team>` to enable semantic search.");
            return None;
        }
    };
    match Embedder::new(&paths) {
        Ok(e) => Some(e),
        Err(e) => {
            tracing::warn!(error = %e, "failed to load embedding model");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archive_no_category() {
        assert_eq!(archive_category(None), Some("Archived".into()));
        assert_eq!(archive_category(Some("")), Some("Archived".into()));
    }

    #[test]
    fn archive_with_category() {
        assert_eq!(
            archive_category(Some("dev/guide")),
            Some("Archived/dev/guide".into())
        );
    }

    #[test]
    fn archive_already_archived() {
        assert_eq!(archive_category(Some("Archived")), None);
        assert_eq!(archive_category(Some("Archived/dev")), None);
    }

    #[test]
    fn archive_not_prefix_match() {
        assert_eq!(
            archive_category(Some("ArchivedData")),
            Some("Archived/ArchivedData".into())
        );
    }
}
