## -Predicting-Loan-Default-ML

 
## Abstract:

This project undertakes the task of predicting loan defaults by employing various machine learning models applied to data sourced from a peer-to-peer lending platform. The models utilized include Linear Regression, Ridge Regression, Lasso Regression, Random Forest, and Neural Network. The dataset encompasses a diverse set of borrower attributes, such as loan amount, interest rate, employment duration, homeownership status, annual income, and credit history. The primary objective is to analyze how these variables correlate with the loan status and identify the 10 most correlated and the 10 least correlated features. Through comprehensive analysis and model evaluation, the project aims to determine the most effective model for predicting loan defaults. The selected model will undergo thorough evaluation and comparison with other models to assess its efficacy in risk assessment and decision-making within the lending platform.

## Task 1(Evaluating Models for Loan Default Prediction):

This task focuses on the evaluation of multiple models to determine the most effective approach for predicting loan defaults. The process begins with the creation of a new variable, denoted as "y," derived from the training dataset. This variable serves as an indicator, assigning a value of 1 to defaulted loans and 0 to non-defaulted ones. Furthermore, the exploration of various transformations, such as converting categorical data into continuous variables, is conducted to enhance the predictive accuracy of the models.

## 1.1	Linear Regression Model:

A linear regression model was applied to the train data, with the target variable "y" and all predictors. The model was trained and used to predict the target variable for both training and testing datasets. The Mean Squared Error (MSE) for the training data was found to be 0.0685, and for the testing data, it was also 0.0685. This indicates a consistent performance of the model on both training and testing datasets.

## 1.2	Ridge Regression Model

The Ridge Regression model showed decent performance in predicting loan defaults using our dataset. We found that setting the regularization parameter (alpha) to 0.08 gave us the best results. However, even with this adjustment, our model's prediction errors, as measured by Mean Squared Error (MSE), were still quite high around 3.92 for the training data and 3.83 for the testing data.

## 1.3	Lasso Regresion Model:

This task was to use a Lasso regression model to forecast loan defaults. We systematically examined how various λ values affected the model's performance, mirroring the range used in ridge regression. After analyzing the results, we found that the optimal Lasso model, characterized by a λ value of 0.01, yielded a Mean Squared Error (MSE) of train dataset is 0.068598 and test dataset is 0.069170.
 
## 1.4	Random Forest:

In this task, we utilized a Random Forest model to forecast loan defaults. The model was trained on the train data, where the target variable (y) indicated loan defaults, and the predictors included borrower attributes.. Upon evaluation, the Mean Squared Error (MSE) for the best model on the training data was 0.00348, while on the test data, it is 0.0261.


## Variable Importance Analysis:

Analyzing variable importance in making well-informed decisions concerning risk assessment and loan approval. The Random Forest model's ability to achieve low Mean Squared Error (MSE) values for both training and testing data underscores its robust predictive capabilities. Analyzing variable importance provides valuable insights into the key factors influencing the prediction of loan defaults.

## 1.5	Neural Network:
For this task, we utilized a Multi-layer Perceptron (MLP) Neural Network model. This model was configured with a single hidden layer containing 50 neurons and utilized the Rectified Linear Unit activation function. Additionally, we set the random state to 42 for consistency in the results. Upon evaluation, the Mean Squared Error (MSE) for the training data was measured at 616021.57835395, while for the testing data, it was 41099.05548659612.

## 1.6	Evaluation:

## Comparison of Approaches:

We used five different methods to evaluate the predictive capability in forecasting loan defaults from the provided dataset: Linear Regression, Ridge Regression, Lasso Regression, Random Forest, and a Multi-layer Perceptron (MLP) Neural Network. Each of these models underwent training and assessment using the train Data, where y served as the target variable alongside a collection of predictors.

## Linear Regression Model:

Starting with Linear Regression model, produced a mean squared error (MSE) of 0.0685 for the training and testing sets of data. This shows that the model performed consistently on both sets of data.

## Ridge Regression Model:

In the Ridge Regression model, we investigated various hyperparameters (λ) and determined the optimal alpha value to be 0.08. Nonetheless, even after fine-tuning, the model exhibited relatively elevated mean squared errors (MSEs), registering at 3.92 for the training dataset and 3.83 for the testing dataset.
Lasso Regression Model:
we employed Lasso Regression, examining different alpha values and determining the ideal alpha to be 0.001. Although the MSE values were marginally lower than those obtained with Ridge Regression, the errors remained very high, with training data MSEs of 0.0686 and testing data MSEs of 0.0692.
 
## Random Forest Model:

Moving to Random Forest, this group learning technique exhibited encouraging results, showcasing remarkably low MSE values of 0.00348 for training data and 0.02609 for testing data.

## Neural Network Model:

Finally, we utilized an MLP Neural Network, setting it up with a single hidden layer consisting of 50 neurons and ReLU activation. Despite its intricacy, the Neural Network model displayed elevated MSE values of 616021.58 for training data and 41099.06 for testing data, suggesting considerable prediction inaccuracies.

## Conclusion:

To summarize, the Random Forest model proved to be the most proficient method for forecasting loan defaults based on the provided dataset, exhibiting the lowest MSE values across both training and testing datasets. Its ensemble learning characteristic likely bolstered its robust predictive capability, surpassing the accuracy and generalization of other regression and neural network models. Therefore, we identify the Random Forest model as the best model for this prediction task.
