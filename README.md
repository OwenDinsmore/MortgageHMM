

The main.rs file is where everything comes together. Think of it as the control center where we load our mortgage data, train our model, and test how well it works. It starts by reading in two sets of data - one for training and one for testing. Then it tries out different settings (like how fast the model learns or how strict it should be with predictions) to find what works best. Once it finds the best settings, 
it shows us how accurate the predictions are and gives us some example predictions with real numbers. This makes it easy to see how the model would work in real life, like checking if someone asking for a $95,000 loan with a good job history might default.

The final output has a 70-80% accuracy on the testing data. We could increase that accuracy with more features, but for the purpose of this finding that data with a larger shape proved to be difficult. It's important to note that I created a 75/25 train/test split on the data using python/ pandas



The logreg.rs file is how we implement the logistic regression. it takes all those numbers about someone's loan and credit history and turns them into a yes/no prediction (0/1) about whether they might default. 
it learns gradually from the data, can handle when our data isn't perfectly balanced (like having more successful loans than defaults), and includes some safeguards to prevent it from being too rigid in its predictions. The code also includes a bunch of tests to make sure everything works correctly, checking things like whether the predictions make sense and the math is working/ model is working

weights and bias: The trainable parameters (weights is Optional since it's initialized during training)
feature_means and feature_stds: Used for feature scaling/normalization
Hyperparameters: learning_rate, lambda (L2 regularization), iterations, batch_size, and threshold

The ModelMetrics struct tracks model performance with:
Standard metrics: accuracy, precision, recall, F1 score
Confusion matrix components: true/false positives/negatives

Key method implementations:

train piplline:

Initializes weights randomly in small range (-0.01 to 0.01)
Computes feature statistics (means and standard deviations)
Implements mini-batch gradient descent with:

Momentum (set at 0.9) for faster convergence
Class weighting to handle imbalanced data
L2 regularization to prevent overfinding


updates weights and bias iteratively using computed gradients


prediction methods:

predict: Returns binary classifications (0 or 1) based on threshold
predict_proba: Returns raw probabilities
Both methods scale input features using stored means/stds


featture scaling:

Standardizes features using (x - mean) / std
Handles zero standard deviation case by defaulting to 0
Applied consistently in both training and prediction


model evalution:

Calculates  metrics for model performance
Handles edge cases (zero denominators)
Provides detailed breakdown of prediction outcomes

The preprocess.rs file does all cleanup work. Raw mortgage data can be messy - missing information, numbers that are all over the place in terms of scale, and so on. This file handles all of that.
It reads in the CSV files where our mortgage data is stored, fills in any missing numbers with sensible estimates (using averages), and makes sure all our numbers are on similar scales (scaling so that a difference in loan amount doesn't cause overfitting). 

