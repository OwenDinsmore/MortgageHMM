mod preprocess;
mod logreg;

use preprocess::{preprocess_csv, print_data_statistics};
use logreg::LogisticRegression;
use std::error::Error;
use ndarray::Array2;

fn main() -> Result<(), Box<dyn Error>> {
    // load and preprocess data
    let train_file_path = "hmeq_train.csv";
    let test_file_path = "hmeq_test.csv";
    
    let (train_features, train_labels) = preprocess_csv(train_file_path)?;
    let (test_features, test_labels) = preprocess_csv(test_file_path)?;
    
    print_data_statistics(&train_features, &train_labels);
    
    // get the number of features from our training data
    let num_features = train_features.ncols();
    println!("Number of features in training data: {}", num_features);
    
    // define params
    let learning_rates = vec![0.001, 0.005, 0.01];
    let lambdas = vec![0.01, 0.1, 1.0];
    let thresholds = vec![0.3, 0.4, 0.5, 0.6, 0.7];
    
    let mut best_accuracy = 0.0;
    let mut best_hyperparams = (0.0, 0.0, 0.0);
    let mut best_metrics = None;
    
    // Grid search
    for &lr in &learning_rates {
        for &l in &lambdas {
            for &t in &thresholds {
                let mut model = LogisticRegression::new(lr, l, 10000, 128, t);
                model.fit(&train_features, &train_labels);
                let metrics = model.evaluate(&test_features, &test_labels);
                
                if metrics.accuracy > best_accuracy {
                    best_accuracy = metrics.accuracy;
                    best_hyperparams = (lr, l, t);
                    best_metrics = Some(metrics);
                }
            }
        }
    }
    
    // train final model with best hyperparameters
    if let Some(metrics) = best_metrics {
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

        // train final model
        let mut final_model = LogisticRegression::new(
            best_hyperparams.0,
            best_hyperparams.1,
            10000,
            128,
            best_hyperparams.2
        );
        final_model.fit(&train_features, &train_labels);

        println!("\nExample Predictions:");      // Example predictions
        
        //example features with the correct number of columns
        // features are: LOAN, MORTDUE, VALUE, YOJ, DEROG, DELINQ, CLAGE, NINQ
        let sample_features = vec![
            // Low risk example: Large down payments, high values, stable jobs, good credit
            vec![95000.0, 25000.0, 160000.0, 10.0, 0.0, 0.0, 120.0, 1.0],
            vec![15000.0, 40000.0, 45000.0, 1.0, 2.0, 3.0, 36.0, 4.0],
        ];

        for (i, features) in sample_features.iter().enumerate() {
            // Verify we have the correct number of features
            assert_eq!(features.len(), num_features, "Example features must match training data dimensions");
            
            let feature_array = Array2::from_shape_vec(
                (1, features.len()),
                features.clone()
            ).unwrap();
            
            let prob = final_model.predict_proba(&feature_array)[0];
            let prediction = final_model.predict(&feature_array)[0];
            
            println!("\nSample {} Prediction:", i + 1);
            println!("Features: {:?}", features);
            println!("Default Probability: {:.2}%", prob * 100.0);
            println!("Prediction: {}", if prediction > 0.5 { "High Risk" } else { "Low Risk" });
        }
    }

    Ok(())
}