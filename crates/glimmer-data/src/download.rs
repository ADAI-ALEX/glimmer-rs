//! HTTP downloads with byte limits, SHA-256 checksums and provenance records.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::Path;

pub const USER_AGENT: &str = concat!(
    "glimmer-rs/",
    env!("CARGO_PKG_VERSION"),
    " (dataset fetcher; +https://github.com/ADAI-ALEX/glimmer-rs)"
);

/// What was downloaded, from where, when, and its checksum.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DownloadRecord {
    pub url: String,
    pub file: String,
    pub bytes: u64,
    pub sha256: String,
    /// True when the file was deliberately truncated with a range request.
    pub truncated: bool,
    pub fetched_unix: u64,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Download `url` to `dest`, reading at most `max_bytes` if given. Writes to a `.part` file and
/// renames on success so interrupted downloads never look complete. `progress(bytes_so_far)` is
/// called periodically.
pub fn fetch(
    url: &str,
    dest: &Path,
    max_bytes: Option<u64>,
    mut progress: impl FnMut(u64),
) -> Result<DownloadRecord> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut req = ureq::get(url).header("User-Agent", USER_AGENT);
    if let Some(max) = max_bytes {
        req = req.header("Range", &format!("bytes=0-{}", max.saturating_sub(1)));
    }
    let mut resp = req.call().with_context(|| format!("GET {url}"))?;
    let status = resp.status().as_u16();
    if !(200..300).contains(&status) {
        bail!("GET {url} returned HTTP {status}");
    }
    let mut reader = resp.body_mut().with_config().limit(u64::MAX).reader();
    let part = dest.with_extension("part");
    let mut out = std::io::BufWriter::new(std::fs::File::create(&part)?);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    let mut total = 0u64;
    let mut last_report = 0u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        let take = match max_bytes {
            Some(max) => ((max - total) as usize).min(n),
            None => n,
        };
        out.write_all(&buf[..take])?;
        hasher.update(&buf[..take]);
        total += take as u64;
        if total - last_report >= 4 << 20 {
            progress(total);
            last_report = total;
        }
        if max_bytes.is_some_and(|m| total >= m) {
            break;
        }
    }
    out.flush()?;
    drop(out);
    std::fs::rename(&part, dest)?;
    progress(total);
    Ok(DownloadRecord {
        url: url.to_string(),
        file: dest
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default(),
        bytes: total,
        sha256: hex(&hasher.finalize()),
        truncated: max_bytes.is_some_and(|m| total >= m),
        fetched_unix: glimmer_core::time::now_unix(),
    })
}

/// SHA-256 of a file on disk.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_value() {
        let dir = std::env::temp_dir().join(format!("glimmer-dl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x.txt");
        std::fs::write(&p, b"abc").unwrap();
        assert_eq!(
            sha256_file(&p).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        std::fs::remove_dir_all(dir).ok();
    }
}
