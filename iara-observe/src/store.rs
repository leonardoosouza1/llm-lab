use std::path::PathBuf;
use std::io::Write;
use parking_lot::Mutex;
use crate::{TensorSnapshot, NeuronSnapshot, FlowEvent};

pub struct ObserveStore {
    dir: PathBuf,
    tensor_file: Mutex<Option<std::io::BufWriter<std::fs::File>>>,
    neuron_file: Mutex<Option<std::io::BufWriter<std::fs::File>>>,
    flow_file: Mutex<Option<std::io::BufWriter<std::fs::File>>>,
    session_id: String,
}

impl ObserveStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        let dir = dir.into();
        let session_id = chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string();
        Self {
            dir,
            tensor_file: Mutex::new(None),
            neuron_file: Mutex::new(None),
            flow_file: Mutex::new(None),
            session_id,
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    fn ensure_dir(&self) {
        let session_dir = self.dir.join(&self.session_id);
        std::fs::create_dir_all(&session_dir).ok();
    }

    fn open_file(&self, name: &str) -> std::io::BufWriter<std::fs::File> {
        self.ensure_dir();
        let path = self.dir.join(&self.session_id).join(format!("{name}.jsonl"));
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("Failed to open observe file");
        std::io::BufWriter::new(file)
    }

    pub fn record_tensor(&self, snap: &TensorSnapshot) {
        let mut guard = self.tensor_file.lock();
        if guard.is_none() {
            *guard = Some(self.open_file("tensors"));
        }
        if let Some(ref mut f) = *guard {
            let _ = serde_json::to_writer(&mut *f, snap);
            let _ = f.write_all(b"\n");
        }
    }

    pub fn record_neuron(&self, snap: &NeuronSnapshot) {
        let mut guard = self.neuron_file.lock();
        if guard.is_none() {
            *guard = Some(self.open_file("neurons"));
        }
        if let Some(ref mut f) = *guard {
            let _ = serde_json::to_writer(&mut *f, snap);
            let _ = f.write_all(b"\n");
        }
    }

    pub fn record_flow(&self, event: &FlowEvent) {
        let mut guard = self.flow_file.lock();
        if guard.is_none() {
            *guard = Some(self.open_file("flow"));
        }
        if let Some(ref mut f) = *guard {
            let _ = serde_json::to_writer(&mut *f, event);
            let _ = f.write_all(b"\n");
        }
    }

    pub fn flush(&self) {
        if let Some(ref mut f) = *self.tensor_file.lock() { let _ = f.flush(); }
        if let Some(ref mut f) = *self.neuron_file.lock() { let _ = f.flush(); }
        if let Some(ref mut f) = *self.flow_file.lock() { let _ = f.flush(); }
    }
}
