use ndarray::{Array1, Array2, ArrayView1, Axis};
use rand::Rng;

#[derive(Debug)]
pub struct LogisticRegression {
    weights: Option<Array1<f64>>,
    bias: f64,
    feature_means: Option<Array1<f64>>,
    feature_stds: Option<Array1<f64>>,
    learning_rate: f64,
    lambda: f64,
    iterations: usize,
    batch_size: usize,
    threshold: f64,
}

#[derive(Debug, Clone)]
pub struct ModelMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub true_positives: usize,
    pub false_positives: usize,
    pub true_negatives: usize,
    pub false_negatives: usize,
}

impl LogisticRegression {
    pub fn new(learning_rate: f64, lambda: f64, iterations: usize, batch_size: usize, threshold: f64) -> Self {
        Self {
            weights: None,
            bias: 0.0,
            feature_means: None,
            feature_stds: None,
            learning_rate,
            lambda,
            iterations,
            batch_size,
            threshold,
        }
    }

    fn initialize_weights(num_features: usize) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        Array1::from_vec(
            (0..num_features)
                .map(|_| rng.gen_range(-0.01..0.01))
                .collect()
        )
    }

    fn sigmoid(x: &ArrayView1<f64>) -> Array1<f64> {
        x.mapv(|val| 1.0 / (1.0 + (-val).exp()))
    }

    fn scale_features(&self, features: &Array2<f64>) -> Array2<f64> {
        let feature_means = self.feature_means.as_ref().unwrap();
        let feature_stds = self.feature_stds.as_ref().unwrap();
        
        let scaled = features.outer_iter()
            .map(|row| {
                row.iter()
                    .zip(feature_means.iter())
                    .zip(feature_stds.iter())
                    .map(|((&x, &mean), &std)| {
                        if std != 0.0 { (x - mean) / std } else { 0.0 }
                    })
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>();
        
        Array2::from_shape_vec(
            (features.nrows(), features.ncols()),
            scaled.into_iter().flatten().collect()
        ).unwrap()
    }

    pub fn fit(&mut self, features: &Array2<f64>, labels: &Array1<f64>) {
        let num_samples = features.nrows();
        let num_features = features.ncols();
        
        // init weights and compute feature statistics
        let mut weights = Self::initialize_weights(num_features);
        self.feature_means = Some(features.mean_axis(Axis(0)).unwrap());
        self.feature_stds = Some(features.std_axis(Axis(0), 0.0));
        
        let scaled_features = self.scale_features(features);
        
        // calculate class weights
        let positive_samples = labels.iter().filter(|&&x| x > 0.5).count();
        let negative_samples = labels.len() - positive_samples;
        let pos_weight = (num_samples as f64) / (2.0 * positive_samples as f64);
        let neg_weight = (num_samples as f64) / (2.0 * negative_samples as f64);
        
        let mut rng = rand::thread_rng();
        let momentum = 0.9;
        let mut bias = 0.0;
        
        for _ in 0..self.iterations {
            let batch_indices: Vec<usize> = (0..self.batch_size)
                .map(|_| rng.gen_range(0..num_samples))
                .collect();
                
            let batch_features = batch_indices.iter()
                .map(|&i| scaled_features.row(i).to_owned())
                .collect::<Vec<_>>();
            let batch_labels = batch_indices.iter()
                .map(|&i| labels[i])
                .collect::<Vec<_>>();
                
            let batch_features = Array2::from_shape_vec(
                (self.batch_size, num_features),
                batch_features.into_iter().flatten().collect()
            ).unwrap();
            let batch_labels = Array1::from(batch_labels);
            
            let linear_model = batch_features.dot(&weights) + bias;
            let predictions = Self::sigmoid(&linear_model.view());
            
            let error = predictions.iter()
                .zip(batch_labels.iter())
                .map(|(&pred, &label)| {
                    let weight = if label > 0.5 { pos_weight } else { neg_weight };
                    weight * (pred - label)
                })
                .collect::<Array1<f64>>();
            
            let dw = &batch_features.t().dot(&error) / self.batch_size as f64 
                + &weights * (self.lambda / self.batch_size as f64);
            let db = error.sum() / self.batch_size as f64;
            
            weights = &weights * momentum - &(dw * self.learning_rate);
            bias = bias * momentum - db * self.learning_rate;
        }
        
        self.weights = Some(weights);
        self.bias = bias;
    }

    pub fn predict(&self, features: &Array2<f64>) -> Array1<f64> {
        assert!(self.weights.is_some(), "Model must be trained before prediction");
        
        let scaled_features = self.scale_features(features);
        let linear_model = scaled_features.dot(self.weights.as_ref().unwrap()) + self.bias;
        linear_model.mapv(|val| {
            let prob = 1.0 / (1.0 + (-val).exp());
            if prob >= self.threshold { 1.0 } else { 0.0 }
        })
    }

    pub fn predict_proba(&self, features: &Array2<f64>) -> Array1<f64> {
        assert!(self.weights.is_some(), "Model must be trained before prediction");
        
        let scaled_features = self.scale_features(features);
        let linear_model = scaled_features.dot(self.weights.as_ref().unwrap()) + self.bias;
        linear_model.mapv(|val| 1.0 / (1.0 + (-val).exp()))
    }

    pub fn evaluate(&self, features: &Array2<f64>, labels: &Array1<f64>) -> ModelMetrics {
        let predictions = self.predict(features);
        self.compute_metrics(labels, &predictions)
    }

    fn compute_metrics(&self, true_labels: &Array1<f64>, predictions: &Array1<f64>) -> ModelMetrics {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn generate_test_data(samples: usize, features: usize) -> (Array2<f64>, Array1<f64>) {
        let mut rng = rand::thread_rng();
        
        // generate random features
        let mut feature_data = Vec::with_capacity(samples * features);
        for _ in 0..samples * features {
            feature_data.push(rng.gen_range(-1.0..1.0));
        }
        
        // create features matrix
        let features = Array2::from_shape_vec((samples, features), feature_data).unwrap();
        
        let weights = Array1::from_vec(vec![0.5; features.ncols()]);
        let linear_combination = features.dot(&weights);
        let labels = linear_combination.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        
        (features, labels)
    }

    #[test]
    fn test_model_initialization() {
        let model = LogisticRegression::new(0.01, 0.1, 100, 32, 0.5);
        assert!(model.weights.is_none());
        assert_eq!(model.bias, 0.0);
        assert_eq!(model.learning_rate, 0.01);
        assert_eq!(model.lambda, 0.1);
        assert_eq!(model.threshold, 0.5);
    }

    #[test]
    fn test_model_training() {
        let (features, labels) = generate_test_data(100, 5);
        let mut model = LogisticRegression::new(0.01, 0.1, 100, 32, 0.5);
        
        model.fit(&features, &labels);
        
        assert!(model.weights.is_some());
        assert_eq!(model.weights.as_ref().unwrap().len(), 5);
        assert!(model.feature_means.is_some());
        assert!(model.feature_stds.is_some());
    }

    #[test]
    fn test_prediction_shape() {
        let (features, labels) = generate_test_data(100, 5);
        let mut model = LogisticRegression::new(0.01, 0.1, 100, 32, 0.5);
        
        model.fit(&features, &labels);
        
        let predictions = model.predict(&features);
        assert_eq!(predictions.len(), features.nrows());
        
        let probabilities = model.predict_proba(&features);
        assert_eq!(probabilities.len(), features.nrows());
    }

    #[test]
    fn test_prediction_bounds() {
        let (features, labels) = generate_test_data(100, 5);
        let mut model = LogisticRegression::new(0.01, 0.1, 100, 32, 0.5);
        
        model.fit(&features, &labels);
        
        let predictions = model.predict(&features);
        for pred in predictions.iter() {
            assert!(*pred == 0.0 || *pred == 1.0);
        }
        
        let probabilities = model.predict_proba(&features);
        for prob in probabilities.iter() {
            assert!(*prob >= 0.0 && *prob <= 1.0);
        }
    }

    #[test]
    fn test_metrics_calculation() {
        let true_labels = Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0, 1.0]);
        let predictions = Array1::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0]);
        
        let model = LogisticRegression::new(0.01, 0.1, 100, 32, 0.5);
        let metrics = model.compute_metrics(&true_labels, &predictions);
        
        assert_eq!(metrics.true_positives, 2);
        assert_eq!(metrics.false_positives, 1);
        assert_eq!(metrics.true_negatives, 1);
        assert_eq!(metrics.false_negatives, 1);
        assert!((metrics.accuracy - 0.6).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "Model must be trained before prediction")]
    fn test_predict_before_training() {
        let model = LogisticRegression::new(0.01, 0.1, 100, 32, 0.5);
        let features = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        model.predict(&features);
    }

    #[test]
    fn test_model_convergence() {
        let (features, labels) = generate_test_data(1000, 2);
        let mut model = LogisticRegression::new(0.01, 0.1, 1000, 32, 0.5);
        
        model.fit(&features, &labels);
        let metrics = model.evaluate(&features, &labels);
        
        assert!(metrics.accuracy > 0.6);  // Model should perform better than random on training data
    }

    #[test]
    fn test_different_thresholds() {
        let (features, labels) = generate_test_data(100, 2);
        
        let mut high_threshold_model = LogisticRegression::new(0.01, 0.1, 100, 32, 0.8);
        let mut low_threshold_model = LogisticRegression::new(0.01, 0.1, 100, 32, 0.2);
        
        high_threshold_model.fit(&features, &labels);
        low_threshold_model.fit(&features, &labels);
        
        let high_predictions = high_threshold_model.predict(&features);
        let low_predictions = low_threshold_model.predict(&features);
        
        let high_positives = high_predictions.iter().filter(|&&x| x > 0.5).count();
        let low_positives = low_predictions.iter().filter(|&&x| x > 0.5).count();
        
        assert!(high_positives <= low_positives);
    }

    #[test]
    fn test_feature_scaling() {
        let mut features = Array2::zeros((100, 2));
        features.column_mut(0).fill(1000.0); // First feature has large values
        features.column_mut(1).fill(0.1);    // Second feature has small values
        
        let labels = Array1::zeros(100);
        
        let mut model = LogisticRegression::new(0.01, 0.1, 100, 32, 0.5);
        model.fit(&features, &labels);
        
        let predictions = model.predict_proba(&features);
        for prob in predictions.iter() {
            assert!(!prob.is_nan());
            assert!(*prob >= 0.0 && *prob <= 1.0);
        }
    }
}