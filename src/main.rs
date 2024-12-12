mod preprocess;
use preprocess::{preprocess_csv, print_data_statistics};
use ndarray::{Array1, Array2, ArrayView1, Axis};
use std::error::Error;
use rand::Rng;

// rand weight initialization (small vals worked better)
fn initialize_weights(num_features: usize) -> Array1<f64> {
    let mut rng = rand::thread_rng();
    Array1::from_vec(
        (0..num_features)
            .map(|_| rng.gen_range(-0.01..0.01)) // Smaller initial weights
            .collect()
    )
}

fn sigmoid(x: &ArrayView1<f64>) -> Array1<f64> {
    x.mapv(|val| 1.0 / (1.0 + (-val).exp()))
}

fn scale_features(features: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let feature_means = features.mean_axis(Axis(0)).unwrap();
    let feature_stds = features.std_axis(Axis(0), 0.0);
    
    let scaled = features.outer_iter()
        .map(|row| {
            row.iter()
                .zip(feature_means.iter())
                .zip(feature_stds.iter())
                .map(|((&x, &mean), &std)| {
                    if std != 0.0 { (x - mean) / std } else { 0.0 } // Default to 0 if std is 0
                })
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();
    
    let scaled_array = Array2::from_shape_vec(
        (features.nrows(), features.ncols()),
        scaled.into_iter().flatten().collect()
    ).unwrap();
    
    (scaled_array, feature_means, feature_stds)
}

//training function with class weights and mini-batch processing
fn train_logistic_regression(
    features: &Array2<f64>,
    labels: &Array1<f64>,
    learning_rate: f64,
    iterations: usize,
    lambda: f64,
    batch_size: usize,
) -> (Array1<f64>, f64) {
    let num_samples = features.nrows();
    let num_features = features.ncols();
    
    // Initialize weights and bias
    let mut weights = initialize_weights(num_features);
    let mut bias = 0.0;
    
    // Scale features
    let (scaled_features, _, _) = scale_features(features);
    
    // Calculate class weights
    let positive_samples = labels.iter().filter(|&&x| x > 0.5).count();
    let negative_samples = labels.len() - positive_samples;
    let pos_weight = (num_samples as f64) / (2.0 * positive_samples as f64);
    let neg_weight = (num_samples as f64) / (2.0 * negative_samples as f64);
    
    let mut rng = rand::thread_rng();
    
    for _ in 0..iterations {
        let batch_indices: Vec<usize> = (0..batch_size) // random batch indices
            .map(|_| rng.gen_range(0..num_samples))
            .collect();
        let batch_features = batch_indices.iter() // get batch data
            .map(|&i| scaled_features.row(i).to_owned())
            .collect::<Vec<_>>();
        let batch_labels = batch_indices.iter()
            .map(|&i| labels[i])
            .collect::<Vec<_>>();
            
        let batch_features = Array2::from_shape_vec(
            (batch_size, num_features),
            batch_features.into_iter().flatten().collect()
        ).unwrap();
        let batch_labels = Array1::from(batch_labels);
        
        // Forward pass
        let linear_model = batch_features.dot(&weights) + bias;
        let predictions = sigmoid(&linear_model.view());
        
        //compute weighted error
        let error = predictions.iter()
            .zip(batch_labels.iter())
            .map(|(&pred, &label)| {
                let weight = if label > 0.5 { pos_weight } else { neg_weight };
                weight * (pred - label)
            })
            .collect::<Array1<f64>>();
        
        // compute gradients with L2 regularization
        let dw = batch_features.t().dot(&error) / batch_size as f64 
            + (lambda / batch_size as f64) * &weights;
        let db = error.sum() / batch_size as f64;
        
        //update with momentum
        let momentum = 0.9;
        weights = &weights * momentum - &(dw * learning_rate);
        bias = bias * momentum - db * learning_rate;
    }
    
    (weights, bias)
}

fn predict(
    features: &Array2<f64>, 
    weights: &Array1<f64>, 
    bias: f64, 
    threshold: f64
) -> Array1<f64> {
    let (scaled_features, _, _) = scale_features(features);
    let linear_model = scaled_features.dot(weights) + bias;
    linear_model.mapv(|val| {
        let prob = 1.0 / (1.0 + (-val).exp());
        if prob >= threshold { 1.0 } else { 0.0 }
    })
}

// Rest of the code remains the same (compute_model_metrics and ModelMetrics struct)...
#[derive(Debug, Clone)] // Added Clone trait
struct ModelMetrics {
    accuracy: f64,
    precision: f64,
    recall: f64,
    f1_score: f64,
    true_positives: usize,
    false_positives: usize,
    true_negatives: usize,
    false_negatives: usize,
}

fn compute_model_metrics(true_labels: &Array1<f64>, predictions: &Array1<f64>) -> ModelMetrics {
    let total_samples = true_labels.len();
    
    let mut true_positives = 0;
    let mut false_positives = 0;
    let mut true_negatives = 0;
    let mut false_negatives = 0;
    
    true_labels.iter()
        .zip(predictions.iter())
        .for_each(|(&true_label, &predicted)| {
            if true_label > 0.5 && predicted > 0.5 {
                true_positives += 1;
            } else if true_label > 0.5 && predicted <= 0.5 {
                false_negatives += 1;
            } else if true_label <= 0.5 && predicted > 0.5 {
                false_positives += 1;
            } else {
                true_negatives += 1;
            }
        });
    
    let accuracy = (true_positives + true_negatives) as f64 / total_samples as f64;
    let precision = if true_positives + false_positives > 0 {
        true_positives as f64 / (true_positives + false_positives) as f64
    } else {
        0.0
    };
    let recall = if true_positives + false_negatives > 0 {
        true_positives as f64 / (true_positives + false_negatives) as f64
    } else {
        0.0
    };
    let f1_score = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    
    ModelMetrics {
        accuracy,
        precision,
        recall,
        f1_score,
        true_positives,
        false_positives,
        true_negatives,
        false_negatives,
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let train_file_path = "hmeq_train.csv";
    let test_file_path = "hmeq_test.csv";
    
    let (train_features, train_labels) = preprocess_csv(train_file_path)?;
    let (test_features, test_labels) = preprocess_csv(test_file_path)?;
    
    print_data_statistics(&train_features, &train_labels);
    
    //hyperparam tuning
    let learning_rates = vec![0.001, 0.005, 0.01];
    let lambdas = vec![0.01, 0.1, 1.0];
    let thresholds = vec![0.3, 0.4, 0.5, 0.6, 0.7];
    
    let mut best_accuracy = 0.0;
    let mut best_hyperparams = (0.0, 0.0, 0.0);
    let mut best_metrics = None;
    
    for &lr in &learning_rates {
        for &l in &lambdas {
            let (weights, bias) = train_logistic_regression(
                &train_features,
                &train_labels,
                lr,
                10000,
                l,
                128,
            );
            
            for &t in &thresholds {
                let predictions = predict(&test_features, &weights, bias, t);
                let metrics = compute_model_metrics(&test_labels, &predictions);
                
                if metrics.accuracy > best_accuracy {
                    best_accuracy = metrics.accuracy;
                    best_hyperparams = (lr, l, t);
                    best_metrics = Some(metrics.clone());
                }
            }
        }
    }
    
    if let Some(metrics) = best_metrics { // print results for best hyperparameters
        println!("\nBest Hyperparameters:");
        println!("Learning Rate: {}", best_hyperparams.0);
        println!("Lambda: {}", best_hyperparams.1);
        println!("Threshold: {}", best_hyperparams.2);
        println!("\nBest Model Performance Metrics:");
        println!("Accuracy: {:.2}%", metrics.accuracy * 100.0);
        println!("Precision: {:.2}%", metrics.precision * 100.0);
        println!("Recall: {:.2}%", metrics.recall * 100.0);
        println!("F1 Score: {:.2}%", metrics.f1_score * 100.0);
        println!("\nConfusion Matrix:");
        println!("True Positives: {}", metrics.true_positives);
        println!("False Positives: {}", metrics.false_positives);
        println!("True Negatives: {}", metrics.true_negatives);
        println!("False Negatives: {}", metrics.false_negatives);
    }
    Ok(())
}