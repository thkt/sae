use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    pub teams: Vec<String>,
    pub default_team: Option<String>,
    pub embed_budget: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            teams: Vec::new(),
            default_team: None,
            embed_budget: 50,
        }
    }
}

impl Config {
    pub fn load() -> Result<Self, ConfigError> {
        let path = config_dir()?.join("config.json");
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(&path)
            .map_err(|e| ConfigError::ReadFailed(path.clone(), e))?;
        let config: Self =
            serde_json::from_str(&content).map_err(|e| ConfigError::ParseFailed(path, e))?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        for team in &self.teams {
            if team.is_empty() {
                return Err(ConfigError::InvalidValue(
                    "team name must not be empty".into(),
                ));
            }
        }
        if let Some(ref default) = self.default_team {
            if !self.teams.contains(default) {
                return Err(ConfigError::InvalidValue(format!(
                    "default_team '{default}' not found in teams list"
                )));
            }
        }
        Ok(())
    }

    pub fn resolve_team<'a>(&'a self, team: Option<&'a str>) -> Result<&'a str, ConfigError> {
        if let Some(t) = team {
            validate_team_name(t)?;
            if !self.teams.is_empty() && !self.teams.iter().any(|s| s == t) {
                return Err(ConfigError::UnknownTeam(t.to_string()));
            }
            return Ok(t);
        }
        if let Some(ref default) = self.default_team {
            return Ok(default.as_str());
        }
        if self.teams.len() == 1 {
            return Ok(&self.teams[0]);
        }
        Err(ConfigError::NoTeamSpecified)
    }

    pub fn team_db_path(&self, team: &str) -> Result<PathBuf, ConfigError> {
        Ok(data_dir()?.join(format!("{team}.db")))
    }
}

pub(crate) fn data_dir() -> Result<PathBuf, ConfigError> {
    data_dir_with(|k| std::env::var(k))
}

fn data_dir_with(
    get_var: impl Fn(&str) -> Result<String, std::env::VarError>,
) -> Result<PathBuf, ConfigError> {
    let base = get_var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|_| {
            get_var("HOME")
                .map(|h| PathBuf::from(h).join(".local").join("share"))
                .map_err(|_| ConfigError::HomeDirNotFound)
        })?;
    Ok(base.join("sae"))
}

fn config_dir() -> Result<PathBuf, ConfigError> {
    let base = std::env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|_| dirs_fallback_config_home())?;
    Ok(base.join("sae"))
}

fn dirs_fallback_config_home() -> Result<PathBuf, ConfigError> {
    Ok(home_dir()?.join(".config"))
}

/// esa team name: lowercase alphanumeric + hyphen, must start with alphanumeric.
fn validate_team_name(name: &str) -> Result<(), ConfigError> {
    if name.is_empty() {
        return Err(ConfigError::InvalidValue(
            "team name must not be empty".into(),
        ));
    }
    let valid = name
        .bytes()
        .enumerate()
        .all(|(i, b)| b.is_ascii_lowercase() || b.is_ascii_digit() || (b == b'-' && i > 0));
    if !valid {
        return Err(ConfigError::InvalidValue(format!(
            "invalid team name '{name}': must be lowercase alphanumeric and hyphens, starting with a letter or digit"
        )));
    }
    Ok(())
}

fn home_dir() -> Result<PathBuf, ConfigError> {
    std::env::var("HOME")
        .map(PathBuf::from)
        .map_err(|_| ConfigError::HomeDirNotFound)
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Unknown team: {0}")]
    UnknownTeam(String),

    #[error("No team specified and no default_team configured")]
    NoTeamSpecified,

    #[error("Failed to read config at {0}: {1}")]
    ReadFailed(PathBuf, std::io::Error),

    #[error("Failed to parse config at {0}: {1}")]
    ParseFailed(PathBuf, serde_json::Error),

    #[error("HOME environment variable not set")]
    HomeDirNotFound,

    #[error("Invalid config: {0}")]
    InvalidValue(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = Config::default();
        assert!(config.teams.is_empty());
        assert!(config.default_team.is_none());
        assert_eq!(config.embed_budget, 50);
    }

    #[test]
    fn resolve_explicit_team_in_allowlist() {
        let config = Config {
            teams: vec!["alpha".into(), "beta".into()],
            ..Default::default()
        };
        assert_eq!(config.resolve_team(Some("alpha")).unwrap(), "alpha");
    }

    #[test]
    fn resolve_explicit_team_empty_allowlist() {
        let config = Config::default();
        assert_eq!(config.resolve_team(Some("any")).unwrap(), "any");
    }

    #[test]
    fn resolve_default_team() {
        let config = Config {
            teams: vec!["alpha".into(), "beta".into()],
            default_team: Some("beta".into()),
            ..Default::default()
        };
        assert_eq!(config.resolve_team(None).unwrap(), "beta");
    }

    #[test]
    fn resolve_single_team_no_default() {
        let config = Config {
            teams: vec!["only".into()],
            ..Default::default()
        };
        assert_eq!(config.resolve_team(None).unwrap(), "only");
    }

    #[test]
    fn resolve_unknown_team_returns_error() {
        let config = Config {
            teams: vec!["alpha".into()],
            ..Default::default()
        };
        let err = config.resolve_team(Some("nope")).unwrap_err();
        assert!(matches!(err, ConfigError::UnknownTeam(_)));
    }

    #[test]
    fn resolve_no_team_no_default() {
        let config = Config {
            teams: vec!["a".into(), "b".into()],
            ..Default::default()
        };
        let err = config.resolve_team(None).unwrap_err();
        assert!(matches!(err, ConfigError::NoTeamSpecified));
    }

    #[test]
    fn validate_rejects_empty_team_name() {
        let config = Config {
            teams: vec!["".into()],
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_default_not_in_teams() {
        let config = Config {
            teams: vec!["alpha".into()],
            default_team: Some("beta".into()),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn data_dir_respects_xdg() {
        let dir = data_dir_with(|key| match key {
            "XDG_DATA_HOME" => Ok("/tmp/test-xdg".into()),
            _ => Err(std::env::VarError::NotPresent),
        })
        .unwrap();
        assert_eq!(dir, PathBuf::from("/tmp/test-xdg/sae"));
    }

    #[test]
    fn data_dir_falls_back_to_home() {
        let dir = data_dir_with(|key| match key {
            "HOME" => Ok("/home/testuser".into()),
            _ => Err(std::env::VarError::NotPresent),
        })
        .unwrap();
        assert_eq!(dir, PathBuf::from("/home/testuser/.local/share/sae"));
    }

    #[test]
    fn validate_team_name_valid() {
        assert!(validate_team_name("gaji").is_ok());
        assert!(validate_team_name("my-team").is_ok());
        assert!(validate_team_name("team123").is_ok());
        assert!(validate_team_name("a").is_ok());
    }

    #[test]
    fn validate_team_name_rejects_path_traversal() {
        assert!(validate_team_name("../etc").is_err());
        assert!(validate_team_name("team/..").is_err());
        assert!(validate_team_name("./team").is_err());
    }

    #[test]
    fn validate_team_name_rejects_special_chars() {
        assert!(validate_team_name("").is_err());
        assert!(validate_team_name("-starts-with-hyphen").is_err());
        assert!(validate_team_name("has spaces").is_err());
        assert!(validate_team_name("UPPER").is_err());
        assert!(validate_team_name("under_score").is_err());
        assert!(validate_team_name("dot.name").is_err());
    }

    #[test]
    fn resolve_team_rejects_invalid_name() {
        let config = Config::default();
        let err = config.resolve_team(Some("../evil")).unwrap_err();
        assert!(matches!(err, ConfigError::InvalidValue(_)));
    }

    #[test]
    fn team_db_path_format() {
        let config = Config {
            teams: vec!["myteam".into()],
            ..Default::default()
        };
        if let Ok(path) = config.team_db_path("myteam") {
            assert!(path.to_string_lossy().ends_with("myteam.db"));
        }
    }
}
