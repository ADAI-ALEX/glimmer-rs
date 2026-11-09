//! Binary storage for supervised fine-tuning examples (token ids + per-token target mask).

use anyhow::{Context, Result, bail};
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"GLMSFT1\0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SftExample {
    pub tokens: Vec<u32>,
    /// 1 where `tokens[i]` is a supervised target (assistant content or its closing `<|end|>`).
    pub mask: Vec<u8>,
}

impl SftExample {
    pub fn supervised(&self) -> usize {
        self.mask.iter().filter(|&&m| m == 1).count()
    }
}

pub fn write_examples(path: &Path, examples: &[SftExample]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(MAGIC)?;
    f.write_all(&(examples.len() as u64).to_le_bytes())?;
    for ex in examples {
        assert_eq!(ex.tokens.len(), ex.mask.len());
        f.write_all(&(ex.tokens.len() as u32).to_le_bytes())?;
        for &t in &ex.tokens {
            f.write_all(&u16::try_from(t).expect("token id exceeds u16").to_le_bytes())?;
        }
        f.write_all(&ex.mask)?;
    }
    f.flush()?;
    Ok(())
}

pub fn read_examples(path: &Path) -> Result<Vec<SftExample>> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .with_context(|| format!("opening SFT file {}", path.display()))?
        .read_to_end(&mut bytes)?;
    if bytes.len() < 16 || &bytes[..8] != MAGIC {
        bail!("{} is not a GLIMMER SFT file", path.display());
    }
    let n = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let mut off = 16;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if off + 4 > bytes.len() {
            bail!("{} is truncated", path.display());
        }
        let len = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        if off + 3 * len > bytes.len() {
            bail!("{} is truncated", path.display());
        }
        let tokens = bytes[off..off + 2 * len]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as u32)
            .collect();
        off += 2 * len;
        let mask = bytes[off..off + len].to_vec();
        off += len;
        out.push(SftExample { tokens, mask });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let dir = std::env::temp_dir().join(format!("glimmer-sft-{}", std::process::id()));
        let p = dir.join("x.sft");
        let ex = vec![
            SftExample { tokens: vec![3, 100, 5, 4, 200, 5], mask: vec![0, 0, 0, 0, 1, 1] },
            SftExample { tokens: vec![1], mask: vec![0] },
        ];
        write_examples(&p, &ex).unwrap();
        assert_eq!(read_examples(&p).unwrap(), ex);
        std::fs::remove_dir_all(dir).ok();
    }
}
