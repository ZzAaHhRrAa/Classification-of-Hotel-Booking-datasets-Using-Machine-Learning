# Classification of Hotel Booking Datasets Using Machine Learning

## Objective
This project aims to develop a machine learning model to predict the booking status (target value) of rooms in a hotel using a dataset consisting of fifteen features. The main objective is to find the best-performing model, perform hyperparameter tuning, and identify the most important features toward predicting the target value.

## Dataset
The dataset contains 36,285 rows and 17 columns, covering various features related to maintaining a hotel booking or canceling it. Some of the main features include:

- Number of adults
- Number of children
- Number of weekend nights
- Number of weeknights
- Type of meal
- Car parking space
- Room type
- Lead time
- Market segment type
- Repeated: if the reservation is repetitive
- P-C: if the reservation is repetitive, the number of the last cancellation
- P-not-C: if the reservation is repetitive, the number of the last non-cancellation
- Average price
- Special requests
- Date of reservation

**Target:**

- Booking status: if the booking was canceled anytime before the arrival date.

## Project Workflow

### 1. Data Preprocessing
- **Handling Missing Values:** The dataset was almost clean, with only 37 rows missing dates, which were deleted.
- **Feature Scaling:** `StandardScaler` was used to normalize the features.
- **Outlier Detection:** The dataset had neither outliers nor abnormal values.

### 2. Feature Engineering
- `Total individuals`: Created by summing the number of adults and children.
- `Total nights`: Calculated by adding the number of weekend and weeknights.
- `Customer Type`: Generated based on the number of adults and children in each group of guests.
- `Cancel ratio by stay length`: Used during exploratory data analysis (EDA) but excluded from the model to avoid deployment limitations.

### 3. Model Building
The following machine learning algorithms were explored:
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Decision Tree**
- **Gradient Boosting**
- **XGBoost**
- **Logistic Regression**
- **Naive Bayes**

Among them, **Random Forest** achieved the best performance with an F1-score of 0.8926 and test accuracy of 0.8937.

### 4. Hyperparameter Tuning
Both **Grid Search** and **Random Search** were used to optimize the Random Forest model. The difference between the results was negligible. The best parameters that maximized accuracy and F1-score were:

- `Bootstrap: True`
- `max_depth: 20`
- `max_features: sqrt`
- `min_samples_leaf: 1`
- `min_samples_split: 2`
- `n_estimators: 400`

### 5. Feature Importance
The most influential features were identified using Random Forest's built-in feature importance. The top four important features were:

- Lead time
- Average price
- Special requests
- Arrival Month

Least important features:

- Previous Cancellation (P-C)
- Previous not Cancellation (P-not-C)
- Repeated reservation

After selecting the most important features (threshold = 0.2), the model accuracy improved to 0.8948.

### 6. Evaluation
The final Random Forest model achieved:
- F1-score: 0.8937
- Test accuracy: 0.8948

**Confusion matrix** and **classification reports** were generated to gain deeper insights into the model's performance.

## Results
- The Random Forest model outperformed other models with an accuracy of 0.8937.
- Hyperparameter tuning slightly improved the accuracy by less than 0.1%.
- Feature selection improved accuracy to 0.8948.

## Key Takeaways
- **Model Performance:** Random Forest emerged as the best-performing model.
- **Model Improvements:** Fine-tuning hyperparameters and using important features improved the accuracy marginally (~0.1%).

## Conclusion
This project demonstrated the use of machine learning techniques to predict hotel booking status. The **Random Forest** model, enhanced by hyperparameter tuning, was the most effective. Future work could explore more advanced feature selection techniques or dimensionality reduction approaches.

## Instructions for Use
1. Clone this repository.
2. Install the required libraries listed in `requirements.txt`.
3. Run the Jupyter notebook `EDA_Booking_Hotel.ipynb` to view the EDA process, from data cleaning to feature heatmap.
4. Run `Model_Booking_Hotel.ipynb` to see the modeling process, testing different ML models, hyperparameter tuning, and feature importance.
5. Run `Predict_Booking_Hotel.ipynb` to predict the booking status for unseen data.

## Libraries Used
- `pandas`
- `scikit-learn`
- `matplotlib`/`seaborn`
- `numpy`
- `sklearn`
- `xgboost`
- `Datetime`
- `time`
- `joblib`

## Challenges
Some challenges faced during the project include:
- Handling missing data.
- Fine-tuning the models for optimal performance.
