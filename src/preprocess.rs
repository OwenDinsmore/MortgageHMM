use ndarray::{Array1, Array2, Axis};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn preprocess_csv(file_path: &str) -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    
    let mut data: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<f64> = Vec::new();
    
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        let values: Vec<&str> = line.split(',')
            .map(|s| s.trim())
            .collect();
        
        if i == 0 && values.iter().any(|v| v.parse::<f64>().is_err()) {
            continue;
        }
        
        let parsed_values: Vec<f64> = values.iter()
            .map(|&v| v.parse::<f64>().unwrap_or(0.0))
            .collect();
        
        if parsed_values.len() > 1 {
            let label = parsed_values[0];
            let features = parsed_values[1..].to_vec();
            
            data.push(features);
            labels.push(label);
        }
    }
    
    if data.is_empty() {
        return Err("No valid data found in the CSV".into());
    }
    
    let num_features = data[0].len();
    let data_array = Array2::from_shape_vec(
        (data.len(), num_features),
        data.into_iter().flatten().collect()
    )?;
    let data_array_cleaned = replace_nan_with_mean(data_array);
    
    Ok((data_array_cleaned, Array1::from(labels)))
}

///replaces nan vals in a 2D array with the mean of each column
fn replace_nan_with_mean(mut array: Array2<f64>) -> Array2<f64> {
    // calc column means, ignoring NaNs and zeros
    let mean_values: Array1<f64> = array
        .axis_iter(Axis(1))
        .map(|col| {
            let valid_values: Vec<f64> = col.iter()
                .cloned()
                .filter(|&x| x > 0.0 && !x.is_nan())
                .collect();
            
            if valid_values.is_empty() {
                0.0
            } else {
                valid_values.iter().sum::<f64>() / valid_values.len() as f64
            }
        })
        .collect();

    // replace nan with mean
    array.mapv_inplace(|val| {
        if val <= 0.0 || val.is_nan() {
            // find the corresponding column mean
            let col_mean = mean_values[0]; // Use the first column's mean as default
            col_mean
        } else {
            val
        }
    });

    array
}

// print data statistics to help diagnose issues
pub fn print_data_statistics(features: &Array2<f64>, _labels: &Array1<f64>) {
    println!("Data Shape: {} rows, {} columns", features.nrows(), features.ncols());
    println!("Label Distribution:");
    let positive_labels = _labels.iter().filter(|&&x| x > 0.5).count();
    let negative_labels = _labels.len() - positive_labels;
    println!("Positive Labels: {}", positive_labels);
    println!("Negative Labels: {}", negative_labels);
    println!("\nColumn Means:");
    for (i, mean) in features.mean_axis(Axis(0)).unwrap().iter().enumerate() {
        println!("Column {}: {}", i + 1, mean);
    }
}