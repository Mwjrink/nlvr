use std::{io::Cursor, path::Path};

pub fn load<P: AsRef<Path,>,>(path: P,) -> Cursor<Vec<u8,>,> {
    use std::{fs::File, io::Read};

    let mut buf = Vec::new();
    let fullpath = &Path::new("assets",).join(&path,);
    let mut file = File::open(&fullpath,).unwrap();
    file.read_to_end(&mut buf,).unwrap();
    Cursor::new(buf,)
}
