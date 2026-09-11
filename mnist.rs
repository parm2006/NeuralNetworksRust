use NeuralNetworksRust::{
    AdamW, CrossEntropyLoss, Flatten, Linear, Loss, Module, Optimizer, Relu, Sequential, Tensor,
};
use std::env;
use std::fs;
use std::path::Path;

fn load_mnist(img_path: &str, lbl_path: &str, count: usize) -> (Tensor, Tensor) {
    let raw_imgs = fs::read(img_path).expect("Failed to read images");
    let raw_lbls = fs::read(lbl_path).expect("Failed to read labels");

    let imgs: Vec<f32> = raw_imgs[16..16 + count * 784]
        .iter()
        .map(|&b| b as f32 / 255.0)
        .collect();
    let lbls: Vec<f32> = raw_lbls[8..8 + count].iter().map(|&b| b as f32).collect();

    (
        Tensor::new(imgs, vec![count, 28, 28]),
        Tensor::new(lbls, vec![count]),
    )
}

fn build_model() -> Sequential {
    Sequential::new(vec![
        Box::new(Flatten::new(1, None)),
        Box::new(Linear::new(784, 128)),
        Box::new(Relu::new()),
        Box::new(Linear::new(128, 64)),
        Box::new(Relu::new()),
        Box::new(Linear::new(64, 10)),
    ])
}

fn train_and_save_model(model_path: &str) -> Sequential {
    let train_count = 60000;
    println!("Loading all {} MNIST training samples...", train_count);
    let (train_x, train_y) = load_mnist(
        "data/MNIST/raw/train-images-idx3-ubyte",
        "data/MNIST/raw/train-labels-idx1-ubyte",
        train_count,
    );

    let model = build_model();
    let criterion = CrossEntropyLoss::new();
    let mut optimizer = AdamW::new(model.parameters(), 0.002);

    let batch_size = 64;
    let n_batches = train_count / batch_size;

    println!(
        "Training model for 5 epochs ({} batches/epoch, {} images/epoch)...",
        n_batches, train_count
    );
    let train_start = std::time::Instant::now();

    for epoch in 1..=5 {
        let epoch_start = std::time::Instant::now();
        let mut total_loss = 0.0;
        for b in 0..n_batches {
            let start = b * batch_size;
            let end = start + batch_size;

            let bx = Tensor::new(
                train_x.data()[start * 784..end * 784].to_vec(),
                vec![batch_size, 28, 28],
            );
            let by = Tensor::new(train_y.data()[start..end].to_vec(), vec![batch_size]);

            optimizer.zero_grad();
            let logits = model.forward(&bx);
            let loss = criterion.forward(&logits, &by);
            total_loss += loss.item();

            loss.backward();
            optimizer.step();

            if (b + 1) % 250 == 0 || b + 1 == n_batches {
                println!(
                    "  Epoch [{epoch}/5] - Batch [{:>3}/{n_batches}] - Loss: {:.4}",
                    b + 1,
                    loss.item()
                );
            }
        }
        println!(
            "Epoch [{epoch}/5] Done - Avg Loss: {:.4} (took {:.2?})",
            total_loss / n_batches as f32,
            epoch_start.elapsed()
        );
    }
    println!("Training completed in {:.2?}!\n", train_start.elapsed());

    println!("Saving trained model weights to '{}'...", model_path);
    model.save(model_path).expect("Failed to save model");
    println!("Model saved successfully.\n");
    model
}

fn print_ascii_digit(pixels: &[f32]) {
    println!("+----------------------------+");
    for r in 0..28 {
        print!("|");
        for c in 0..28 {
            let val = pixels[r * 28 + c];
            let ch = if val > 0.75 {
                '#'
            } else if val > 0.4 {
                '*'
            } else if val > 0.1 {
                '.'
            } else {
                ' '
            };
            print!("{ch}");
        }
        println!("|");
    }
    println!("+----------------------------+");
}

fn run_single_inference(index: usize, model_path: &str) {
    let (test_x, test_y) = load_mnist(
        "data/MNIST/raw/t10k-images-idx3-ubyte",
        "data/MNIST/raw/t10k-labels-idx1-ubyte",
        10000,
    );

    if index >= test_x.shape()[0] {
        eprintln!("Index out of range (max: {})", test_x.shape()[0] - 1);
        return;
    }

    let model = build_model();
    if !Path::new(model_path).exists() {
        println!("No saved model found. Training one first...");
        train_and_save_model(model_path);
    }
    model
        .load(model_path)
        .expect("Failed to load model weights");

    // Extract single sample [1, 28, 28]
    let sample_data = test_x.data()[index * 784..(index + 1) * 784].to_vec();
    let actual_label = test_y.data()[index] as usize;
    let sample_tensor = Tensor::new(sample_data.clone(), vec![1, 28, 28]);

    // Forward pass through Rust network
    let logits = model.forward(&sample_tensor);
    let raw_scores = logits.data();

    // Softmax probabilities
    let max_score = raw_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_scores: Vec<f32> = raw_scores.iter().map(|&z| (z - max_score).exp()).collect();
    let sum_exp: f32 = exp_scores.iter().sum();
    let probs: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

    let mut predicted_class = 0;
    let mut max_prob = 0.0;
    for (c, &p) in probs.iter().enumerate() {
        if p > max_prob {
            max_prob = p;
            predicted_class = c;
        }
    }

    // Display ASCII image
    println!("\nSample #{index} (Actual: {actual_label}):");
    print_ascii_digit(&sample_data);

    println!("\nClass Probabilities:");
    for c in 0..10 {
        let bar_len = (probs[c] * 25.0) as usize;
        let bar: String = "█".repeat(bar_len);
        let marker = if c == predicted_class {
            " <-- PREDICTION"
        } else {
            ""
        };
        println!("  Digit {c}: {:5.1}% | {bar}{marker}", probs[c] * 100.0);
    }

    println!("\nPREDICTION: {predicted_class}");
    println!("ACTUAL: {actual_label}");
    println!("CONFIDENCE: {:.1}%", max_prob * 100.0);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_path = "parm_mnist_model.bin";

    if args.len() > 1 && args[1] != "--train" {
        // If an integer index is provided, run single inference on that sample
        if let Ok(idx) = args[1].parse::<usize>() {
            run_single_inference(idx, model_path);
            return;
        }
    }

    // Default mode: Train and evaluate
    let model = train_and_save_model(model_path);

    let test_count = 10000;
    println!("Evaluating on all {} test samples...", test_count);
    let (test_x, test_y) = load_mnist(
        "data/MNIST/raw/t10k-images-idx3-ubyte",
        "data/MNIST/raw/t10k-labels-idx1-ubyte",
        test_count,
    );

    let criterion = CrossEntropyLoss::new();
    let eval_batch = 500;
    let eval_batches = test_count / eval_batch;
    let mut total_test_loss = 0.0;
    let mut correct = 0;

    for eb in 0..eval_batches {
        let start = eb * eval_batch;
        let end = start + eval_batch;

        let bx = Tensor::new(
            test_x.data()[start * 784..end * 784].to_vec(),
            vec![eval_batch, 28, 28],
        );
        let by = Tensor::new(test_y.data()[start..end].to_vec(), vec![eval_batch]);

        let logits = model.forward(&bx);
        let loss = criterion.forward(&logits, &by);
        total_test_loss += loss.item();

        let preds = logits.data();
        let targets = by.data();
        for i in 0..eval_batch {
            let row = &preds[i * 10..(i + 1) * 10];
            let pred_class = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            if pred_class == targets[i] as usize {
                correct += 1;
            }
        }
    }

    println!("--------------------------------------------------");
    println!("Avg Test Loss: {:.4}", total_test_loss / eval_batches as f32);
    println!(
        "Test Accuracy: {:.2}% ({correct}/{test_count})",
        (correct as f32) / (test_count as f32) * 100.0
    );
    println!("--------------------------------------------------");
    println!(
        "\n💡 Tip: Run 'cargo run --release --bin mnist -- <index>' or 'uv run python infer.py <index>' to inspect any sample!"
    );
}
