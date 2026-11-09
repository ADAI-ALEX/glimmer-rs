//! Flat token files for pretraining: a 16-byte header followed by little-endian `u16` ids.
//!
//! Files are memory-mapped, so a multi-gigabyte corpus costs no heap memory and random-window
//! sampling touches only the pages it reads.

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use std::io::Write;
use std::path::Path;

const MAGIC: &[u8; 8] = b"GLMTOK1\0";
const HEADER: usize = 16;

pub fn write_tokens(path: &Path, tokens: &[u16]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(MAGIC)?;
    f.write_all(&(tokens.len() as u64).to_le_bytes())?;
    let mut buf = Vec::with_capacity(tokens.len() * 2);
    for t in tokens {
        buf.extend_from_slice(&t.to_le_bytes());
    }
    f.write_all(&buf)?;
    f.flush()?;
    Ok(())
}

/// Read-only, memory-mapped token file.
pub struct TokenFile {
    mmap: Mmap,
    len: usize,
}

impl TokenFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| format!("opening token file {}", path.display()))?;
        // SAFETY: the file is opened read-only and GLIMMER never mutates token files after they
        // are written; concurrent external modification would be a user error.
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < HEADER || &mmap[..8] != MAGIC {
            bail!("{} is not a GLIMMER token file", path.display());
        }
        let len = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        if mmap.len() != HEADER + 2 * len {
            bail!(
                "{} is truncated: header says {len} tokens, file holds {}",
                path.display(),
                (mmap.len() - HEADER) / 2
            );
        }
        Ok(Self { mmap, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn get(&self, i: usize) -> u16 {
        let o = HEADER + 2 * i;
        u16::from_le_bytes([self.mmap[o], self.mmap[o + 1]])
    }

    /// Copy tokens `[start, start + n)` into `out` as `u32`.
    pub fn read_into(&self, start: usize, n: usize, out: &mut Vec<u32>) {
        let bytes = &self.mmap[HEADER + 2 * start..HEADER + 2 * (start + n)];
        out.extend(bytes.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes([c[0], c[1]]) as u32));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_validation() {
        let dir = std::env::temp_dir().join(format!("glimmer-tok-file-{}", std::process::id()));
        let p = dir.join("t.bin");
        let toks: Vec<u16> = (0..1000).map(|i| (i * 7 % 65_000) as u16).collect();
        write_tokens(&p, &toks).unwrap();
        let f = TokenFile::open(&p).unwrap();
        assert_eq!(f.len(), 1000);
        assert_eq!(f.get(10), toks[10]);
        let mut out = Vec::new();
        f.read_into(5, 4, &mut out);
        assert_eq!(out, toks[5..9].iter().map(|&t| t as u32).collect::<Vec<_>>());
        std::fs::write(&p, b"garbage").unwrap();
        assert!(TokenFile::open(&p).is_err());
        std::fs::remove_dir_all(dir).ok();
    }
}
