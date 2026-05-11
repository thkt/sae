use std::io::{self, Write};

/// Write `output` to `w`, ensuring exactly one trailing newline.
pub fn write_output<W: Write>(w: &mut W, output: &str) -> io::Result<()> {
    w.write_all(output.as_bytes())?;
    if !output.ends_with('\n') {
        w.write_all(b"\n")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::{self, Write};

    use super::write_output;

    /// T-W001: write_output appends newline when output lacks trailing newline
    #[test]
    fn write_output_appends_newline_when_missing() {
        let mut buf = Vec::new();
        write_output(&mut buf, "hello").unwrap();
        assert_eq!(&buf, b"hello\n");
    }

    /// T-W002: write_output preserves single trailing newline
    #[test]
    fn write_output_preserves_existing_newline() {
        let mut buf = Vec::new();
        write_output(&mut buf, "hello\n").unwrap();
        assert_eq!(&buf, b"hello\n");
    }

    /// T-W003: write_output propagates BrokenPipe error from writer
    #[test]
    fn write_output_propagates_broken_pipe() {
        struct BrokenPipeWriter;
        impl Write for BrokenPipeWriter {
            fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
                Err(io::Error::from(io::ErrorKind::BrokenPipe))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let mut w = BrokenPipeWriter;
        let err = write_output(&mut w, "hello").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::BrokenPipe);
    }
}
