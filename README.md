💳 Credit Scoring & Fraud Detection Using SOM
📌 Project Overview

This project was developed to identify customers' credit risk and potential fraudulent behavior.

Credit scoring is used in the financial and banking sectors to assess the likelihood that a customer will fulfill their credit obligations. A customer's income, debts, credit history, and other financial indicators can be analyzed to evaluate their level of risk.

However, in real-world datasets, not all risky or fraudulent customers may be labeled in advance. Therefore, this project uses a Self-Organizing Map (SOM) to identify normal and unusual behavioral patterns among customers.

The main purpose of SOM is to group customers based on their similarities without requiring predefined fraud labels and to detect observations that significantly differ from general customer behavior.

This approach allows us to:

analyze customer behavioral patterns;
group similar customers into the same or neighboring clusters;
identify customers whose behavior differs from normal patterns;
detect potential fraud/anomalies;
create a new is_fraud target variable based on the detected anomalies.

📊 Data Preprocessing

Several preprocessing steps were performed on the dataset before building the model.

1. Data Loading

The dataset was loaded from an Excel file:

df = pd.read_excel('scoring.xlsx')

The dataset contains financial and other characteristics of customers.

2. Missing Values

First, missing values in the dataset were checked.

Missing values in numerical variables were replaced with the median value of the corresponding column.

The median was used because it is less sensitive to extreme values compared to the mean.

3. Outlier Detection

Boxplot visualization was used to identify potential outliers in numerical variables.

The IQR (Interquartile Range) method was then applied to identify outliers.

IQR = Q3 - Q1

Lower Limit = Q1 - 1.5 × IQR
Upper Limit = Q3 + 1.5 × IQR

Instead of removing the outliers from the dataset, they were capped at the calculated limits.

This approach helps preserve the data while reducing the impact of extreme values on the model.

4. Feature Selection

The input variables to be used for SOM were prepared.

Customer_ID and Credit_Score were removed from the model inputs.

inputs = df.drop(['Customer_ID', 'Credit_Score'], axis=1)
target = df['Credit_Score']

As a result, SOM learned customer patterns using only the available customer characteristics.

5. Feature Scaling

Since SOM is a distance-based model, it is important for the features to be on a similar scale.

Therefore, MinMaxScaler was used to scale all variables between 0 and 1.

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
inputs_scaled = sc.fit_transform(inputs)

🗺️ SOM Model Construction
Self-Organizing Map

SOM is an unsupervised neural network model that analyzes data without requiring a predefined target.

The model places each customer onto the neuron in the SOM grid that best represents their characteristics.

Customers with similar characteristics are placed on the same or nearby neurons, while customers with significantly different patterns may be located in other areas of the map.

The main purpose of SOM in this project is:

to identify observations that significantly differ from normal customer patterns.

⚙️ SOM Hyperparameter Optimization

The SOM parameters were not selected manually.

Optuna was used to find the optimal parameters.

The following parameters were optimized:

x — X dimension of the SOM grid
y — Y dimension of the SOM grid
sigma — neighborhood radius of the neurons
learning_rate — learning rate of the model
num_iteration — number of training iterations

Optuna tested different parameter combinations and determined the SOM configuration that provided the best result.

📈 SOM Evaluation — Silhouette Score

The Silhouette Score was used to evaluate the SOM results.

The Silhouette Score measures how well the customer groups are separated from each other.

The objective of the Optuna optimization was to:

maximize the Silhouette Score.

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

As a result, the best SOM parameters were selected and the final SOM model was built using these parameters.

📊 SOM Visualization

The results of the optimized SOM were visualized using a Distance Map.

The Distance Map represents the distances between neurons.

In general:

Low distance → similar patterns
High distance → different patterns and potential anomaly areas

The neurons containing customers were also displayed on the SOM.

This visualization helps understand how customers are distributed across the SOM grid and where different groups are located.

Anomaly / Fraud Detection

Neurons with high distance values were identified using the SOM Distance Map.

In this project, the threshold was set to:

threshold = 0.9

Therefore:

Distance >= 0.9

was considered a potential anomaly area.

Customers assigned to these neurons were then identified.

mappings = som_opt.win_map(inputs_scaled)

These customers were considered observations that differed from normal customer behavior according to the SOM model.

Fraud Label Creation

The potential fraud observations identified by SOM were merged with the original dataset.

A new is_fraud variable was then created:

is_fraud = 0 → Normal customer
is_fraud = 1 → Potential fraud / anomaly

Therefore, even though the original dataset did not contain a predefined fraud label, a new fraud target was created based on the anomaly detection results of SOM.

This represents the main output of the SOM stage.

🧠 ANN Model

After completing the SOM stage, a second-stage Artificial Neural Network (ANN) model was built using the is_fraud target created by SOM.

The main purpose of the ANN is to learn the fraud patterns identified by SOM and predict the potential fraud probability for new customers.

In other words, SOM identifies anomalies, while ANN learns these patterns and applies them to new data

⚙️ ANN Data Preparation

Customer_ID was removed from the model, and the remaining features were standardized using StandardScaler.

The dataset was then split into:

80% → Training
20% → Testing

ANN Model Architecture

The ANN model consists of two hidden layers and one output layer:

Input Layer
     ↓
Dense Layer 1 — ReLU
     ↓
Dense Layer 2 — ReLU
     ↓
Output Layer — Sigmoid

A sigmoid activation function was used in the output layer because is_fraud is a binary classification problem.

The model returns a fraud probability between 0 and 1 for each customer.

ANN Hyperparameter Optimization

Optuna was also used to find the optimal parameters for the ANN model.

The following parameters were optimized:

First hidden layer neurons
Second hidden layer neurons
Optimizer
Learning rate
Epochs
Batch size

The following optimizers were tested:

Adam
SGD
RMSprop
Adagrad

ROC-AUC was used as the model selection criterion.

📊 ANN Model Evaluation

The final ANN model was evaluated on both the training and testing datasets.

The main evaluation metrics were:

ROC-AUC
Gini Coefficient

The Gini coefficient was calculated using the following formula:

Gini = 2 × ROC-AUC − 1

Comparing the Train and Test results allows us to evaluate both the model's performance and its generalization ability.

💾 Final Model

The ANN model built using the best parameters was saved as:

som_ann_model.h5

The model can later be loaded without retraining.

New Customer Prediction

New customer data was loaded from the test_practice.xlsx file.

The same preprocessing steps used during training were applied to the new data, and the data was scaled using the same StandardScaler.

The saved ANN model was then used to generate predictions.

As a result, the prediction column contains the probability of each customer being potentially fraudulent.

Customer_ID was not used as a model feature because it is only used for customer identification.

Credit_Score was separated from the input variables during the initial stage
