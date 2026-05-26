use std::fs;
use std::io;
use std::path::PathBuf;

use amici::cli::env_lookup;
use serde::Deserialize;

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
        let content =
            fs::read_to_string(&path).map_err(|e| ConfigError::ReadFailed(path.clone(), e))?;
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
        if let Some(ref default) = self.default_team
            && !self.teams.contains(default)
        {
            return Err(ConfigError::InvalidValue(format!(
                "default_team '{default}' not found in teams list"
            )));
        }
        Ok(())
    }

    pub fn resolve_team<'a>(&'a self, team: Option<&'a str>) -> Result<&'a str, ConfigError> {
        if let Some(t) = team {
            validate_team_name(t)?;
            if !self.teams.is_empty() && !self.teams.iter().any(|s| s == t) {
                return Err(ConfigError::UnknownTeam(t.to_owned()));
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
    data_dir_with(env_lookup())
}

fn data_dir_with(get_var: impl Fn(&str) -> Option<String>) -> Result<PathBuf, ConfigError> {
    let base = match get_var("XDG_DATA_HOME") {
        Some(v) => PathBuf::from(v),
        None => {
            let home = get_var("HOME").ok_or(ConfigError::HomeDirNotFound)?;
            PathBuf::from(home).join(".local").join("share")
        }
    };
    Ok(base.join("sae"))
}

fn config_dir() -> Result<PathBuf, ConfigError> {
    config_dir_with(env_lookup())
}

fn config_dir_with(get_var: impl Fn(&str) -> Option<String>) -> Result<PathBuf, ConfigError> {
    let base = match get_var("XDG_CONFIG_HOME") {
        Some(v) => PathBuf::from(v),
        None => home_dir_with(&get_var)?.join(".config"),
    };
    Ok(base.join("sae"))
}

/// esa team name: lowercase alphanumeric + hyphen, must start with alphanumeric.
pub fn validate_team_name(name: &str) -> Result<(), ConfigError> {
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

fn home_dir_with(get_var: impl Fn(&str) -> Option<String>) -> Result<PathBuf, ConfigError> {
    get_var("HOME")
        .map(PathBuf::from)
        .ok_or(ConfigError::HomeDirNotFound)
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Unknown team: {0}")]
    UnknownTeam(String),

    #[error("No team specified and no default_team configured")]
    NoTeamSpecified,

    #[error("Failed to read config at {0}: {1}")]
    ReadFailed(PathBuf, io::Error),

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

    // T-219: resolve_team returns the team name when it is in the allowlist
    #[test]
    fn resolve_explicit_team_in_allowlist() {
        let config = Config {
            teams: vec!["alpha".into(), "beta".into()],
            ..Default::default()
        };
        assert_eq!(config.resolve_team(Some("alpha")).unwrap(), "alpha");
    }

    // T-220: resolve_team accepts any team name when allowlist is empty
    #[test]
    fn resolve_explicit_team_empty_allowlist() {
        let config = Config::default();
        assert_eq!(config.resolve_team(Some("any")).unwrap(), "any");
    }

    // T-221: resolve_team returns default_team when no team argument is given
    #[test]
    fn resolve_default_team() {
        let config = Config {
            teams: vec!["alpha".into(), "beta".into()],
            default_team: Some("beta".into()),
            ..Default::default()
        };
        assert_eq!(config.resolve_team(None).unwrap(), "beta");
    }

    // T-222: resolve_team picks the sole team when there is one team and no default
    #[test]
    fn resolve_single_team_no_default() {
        let config = Config {
            teams: vec!["only".into()],
            ..Default::default()
        };
        assert_eq!(config.resolve_team(None).unwrap(), "only");
    }

    // T-223: resolve_team returns UnknownTeam error for a team not in the allowlist
    #[test]
    fn resolve_unknown_team_returns_error() {
        let config = Config {
            teams: vec!["alpha".into()],
            ..Default::default()
        };
        let err = config.resolve_team(Some("nope")).unwrap_err();
        assert!(matches!(err, ConfigError::UnknownTeam(_)));
    }

    // T-224: resolve_team returns NoTeamSpecified error when multiple teams and no default
    #[test]
    fn resolve_no_team_no_default() {
        let config = Config {
            teams: vec!["a".into(), "b".into()],
            ..Default::default()
        };
        let err = config.resolve_team(None).unwrap_err();
        assert!(matches!(err, ConfigError::NoTeamSpecified));
    }

    // T-225: validate returns error when a team entry is an empty string
    #[test]
    fn validate_rejects_empty_team_name() {
        let config = Config {
            teams: vec!["".into()],
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    // T-226: validate returns error when default_team is not listed in teams
    #[test]
    fn validate_rejects_default_not_in_teams() {
        let config = Config {
            teams: vec!["alpha".into()],
            default_team: Some("beta".into()),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    // T-227: data_dir uses XDG_DATA_HOME when that environment variable is set
    #[test]
    fn data_dir_respects_xdg() {
        let dir = data_dir_with(|key| match key {
            "XDG_DATA_HOME" => Some("/tmp/test-xdg".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(dir, PathBuf::from("/tmp/test-xdg/sae"));
    }

    // T-228: data_dir falls back to ~/.local/share/sae when XDG_DATA_HOME is unset
    #[test]
    fn data_dir_falls_back_to_home() {
        let dir = data_dir_with(|key| match key {
            "HOME" => Some("/home/testuser".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(dir, PathBuf::from("/home/testuser/.local/share/sae"));
    }

    // T-234: data_dir returns HomeDirNotFound when neither XDG_DATA_HOME nor HOME is set
    #[test]
    fn data_dir_missing_home_errors() {
        let err = data_dir_with(|_| None).unwrap_err();
        assert!(matches!(err, ConfigError::HomeDirNotFound));
    }

    // T-235: config_dir uses XDG_CONFIG_HOME when that environment variable is set
    #[test]
    fn config_dir_respects_xdg() {
        let dir = config_dir_with(|key| match key {
            "XDG_CONFIG_HOME" => Some("/tmp/test-config".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(dir, PathBuf::from("/tmp/test-config/sae"));
    }

    // T-236: config_dir falls back to $HOME/.config/sae when XDG_CONFIG_HOME is unset
    #[test]
    fn config_dir_falls_back_to_home() {
        let dir = config_dir_with(|key| match key {
            "HOME" => Some("/home/testuser".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(dir, PathBuf::from("/home/testuser/.config/sae"));
    }

    // T-229: validate_team_name accepts lowercase alphanumeric names with hyphens
    #[test]
    fn validate_team_name_valid() {
        assert!(validate_team_name("gaji").is_ok());
        assert!(validate_team_name("my-team").is_ok());
        assert!(validate_team_name("team123").is_ok());
        assert!(validate_team_name("a").is_ok());
    }

    // T-230: validate_team_name rejects names containing path traversal sequences
    #[test]
    fn validate_team_name_rejects_path_traversal() {
        assert!(validate_team_name("../etc").is_err());
        assert!(validate_team_name("team/..").is_err());
        assert!(validate_team_name("./team").is_err());
    }

    // T-231: validate_team_name rejects empty, uppercase, spaces, and special chars
    #[test]
    fn validate_team_name_rejects_special_chars() {
        assert!(validate_team_name("").is_err());
        assert!(validate_team_name("-starts-with-hyphen").is_err());
        assert!(validate_team_name("has spaces").is_err());
        assert!(validate_team_name("UPPER").is_err());
        assert!(validate_team_name("under_score").is_err());
        assert!(validate_team_name("dot.name").is_err());
    }

    // T-232: resolve_team returns InvalidValue error for a team name with ".."
    #[test]
    fn resolve_team_rejects_invalid_name() {
        let config = Config::default();
        let err = config.resolve_team(Some("../evil")).unwrap_err();
        assert!(matches!(err, ConfigError::InvalidValue(_)));
    }

    // T-233: team_db_path returns a path ending with "<team>.db"
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
