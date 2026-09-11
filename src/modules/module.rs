use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use crate::tensor::Tensor;

pub trait Module {
    fn forward(&self, _input: &Tensor) -> Tensor {
        panic!("Forward not implemented for base module trait");
    }

    fn direct_parameters(&self) -> Vec<(String, Tensor)> {
        vec![]
    }

    fn modules(&self) -> Vec<(String, &dyn Module)> {
        vec![]
    }

    fn parameters(&self) -> Vec<(String, Tensor)> {
        let mut params = self.direct_parameters();
        for (mod_name, m) in self.modules() {
            for (param_name, param) in m.parameters() {
                params.push((format!("{}.{}", mod_name, param_name), param));
            }
        }
        params
    }

    fn zero_grad(&self) {
        for (_name, p) in self.parameters() {
            p.zero_grad();
        }
    }

    fn requires_grad(&self, requires_grad: bool) {
        for (_name, mut p) in self.parameters() {
            p.set_requires_grad(requires_grad);
        }
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut state: HashMap<String, Tensor> = HashMap::new();
        for (name, param) in self.parameters() {
            state.insert(name, param);
        }
        state
    }

    fn load_state_dict(&self, state_dict: &HashMap<String, Tensor>) {
        for (name, param) in self.parameters() {
            if let Some(loaded) = state_dict.get(&name) {
                param.set_data(loaded.data());
            }
        }
    }

    fn save(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        let state = self.state_dict();
        writer.write_all(&(state.len() as u32).to_le_bytes())?;

        for (name, tensor) in &state {
            let name_bytes = name.as_bytes();
            writer.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
            writer.write_all(name_bytes)?;

            let shape = tensor.shape();
            writer.write_all(&(shape.len() as u32).to_le_bytes())?;
            for &dim in &shape {
                writer.write_all(&(dim as u64).to_le_bytes())?;
            }

            let data = tensor.data();
            writer.write_all(&(data.len() as u64).to_le_bytes())?;
            for &val in &data {
                writer.write_all(&val.to_le_bytes())?;
            }
        }
        writer.flush()?;
        Ok(())
    }

    fn load(&self, path: &str) -> Result<(), String> {
        let file = File::open(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;
        let mut reader = BufReader::new(file);

        let mut u32_buf = [0u8; 4];
        let mut u64_buf = [0u8; 8];

        reader.read_exact(&mut u32_buf).map_err(|e| e.to_string())?;
        let param_count = u32::from_le_bytes(u32_buf) as usize;

        let mut state_dict = HashMap::new();

        for _ in 0..param_count {
            reader.read_exact(&mut u32_buf).map_err(|e| e.to_string())?;
            let name_len = u32::from_le_bytes(u32_buf) as usize;
            let mut name_bytes = vec![0u8; name_len];
            reader.read_exact(&mut name_bytes).map_err(|e| e.to_string())?;
            let name = String::from_utf8(name_bytes).map_err(|e| e.to_string())?;

            reader.read_exact(&mut u32_buf).map_err(|e| e.to_string())?;
            let shape_len = u32::from_le_bytes(u32_buf) as usize;
            let mut shape = Vec::with_capacity(shape_len);
            for _ in 0..shape_len {
                reader.read_exact(&mut u64_buf).map_err(|e| e.to_string())?;
                shape.push(u64::from_le_bytes(u64_buf) as usize);
            }

            reader.read_exact(&mut u64_buf).map_err(|e| e.to_string())?;
            let data_len = u64::from_le_bytes(u64_buf) as usize;
            let mut data = Vec::with_capacity(data_len);
            let mut f32_buf = [0u8; 4];
            for _ in 0..data_len {
                reader.read_exact(&mut f32_buf).map_err(|e| e.to_string())?;
                data.push(f32::from_le_bytes(f32_buf));
            }

            state_dict.insert(name, Tensor::new(data, shape));
        }

        self.load_state_dict(&state_dict);
        Ok(())
    }
}