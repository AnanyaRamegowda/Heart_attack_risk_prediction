# Heart_attack_risk_prediction
## Introduction
This project predicts the risk of a heart attack based on various health parameters using machine learning models. The aim is to leverage predictive analytics to identify individuals at high risk for heart disease, allowing for early intervention and prevention. The dataset includes features such as age, cholesterol levels, blood pressure, and lifestyle factors to train and evaluate the models.
### Objectives
*Create a model that predicts the risk of a heart attack using patient health data.
*Help doctors and patients take early action to prevent heart attacks.
*Understand which health factors are most important for predicting heart attacks.
*Make healthcare decisions easier with accurate, automated predictions.
#### Brief Description About The Project
This project involves the analysis of a dataset containing various health parameters and a binary classification task to predict the presence or absence of heart disease in individuals. The project consists of the following major steps:

    1. Importing libraries and datasets.
    2. Data understanding and exploration.
    3. Data preprocessing and cleaning.
    4. Exploratory Data Analysis (EDA) to gain insights into the dataset.
    5. Feature selection and handling multicollinearity.
    6. Splitting the data into training and testing sets.
    7. Hyperparameter tuning for machine learning models.
    8. Model training and evaluation using three different algorithms: Logistic Regression, Decision Tree, and Support Vector Machine (SVM).
## Importing Libraries and Dataset
    1.pandas.
         ```bash 
            import pandas as pd
    2.Numpy.
         ``` bash
            import numpy as np
 
  
      *Seaborn
      *Scikit-learn
      *Matplotlib
      *Statsmodels
The dataset used in this project is stored in a CSV file named 'heart.csv'.
###### Factors or Parameters Considered from the CSV File
The following factors or parameters are considered from the CSV file:
```bash
    - 'age': Age of the patient.
    - 'sex': Gender of the patient.
    - 'cp': Chest pain type.
    - 'trtbps': Resting blood pressure in mm Hg.
    - 'chol': Cholesterol level in mg/dL.
    - 'exng': Exercise-induced angina.
    - 'fbs': Fasting blood sugar level.
    - 'restecg': Resting electrocardiographic results.
    - 'thalachh': Maximum heart rate achieved.
    - 'slp': Slope.
    - 'caa': Number of major vessels.
    - 'thall': Thallium stress test result.
    - 'output': Target variable (0 for less chance of heart attack, 1 for more chance of heart attack).
####### Steps Included in this Project
```bash
    1. Data loading and exploration.
    2. Data preprocessing, including handling outliers, missing values, and duplicates.
    3. Exploratory data analysis (EDA) to gain insights into the dataset.
    4. Feature selection based on correlation.
    5. Train-test split of the dataset.
    6. Hyperparameter tuning for multiple machine learning models:
    7. Logistic Regression
    8. Decision Tree
    9. Support Vector Machine (SVM)
    10. Model training and evaluation.
    11. Display of confusion matrices and classification reports for model performance.
######## Brief Description and Insight
Description: This project predicts the risk of a heart attack based on patient data such as cholesterol levels, blood pressure, and age. Machine learning algorithms analyze patterns in the data to provide predictions. Insight: Health-related machine learning models can have a profound impact by providing early warning signs and assisting healthcare professionals in diagnosing potential risks. This project could use logistic regression, decision trees, or neural networks to predict cardiovascular risks.
```bash
Three modelling procedures are employed in this project:
1. **Logistic Regression**: A widely used classification algorithm that estimates the probability of a binary outcome. Logistic Regression is used to predict the likelihood of heart disease. The model is trained with hyperparameters optimized through grid search.
2. **Decision Tree**: A tree-structured model that makes decisions based on the input features. A Decision Tree model is employed for heart disease prediction. The model's hyperparameters are fine-tuned for optimal performance.
######### Conclusion
The heart attack risk prediction model offers a promising tool for identifying individuals at higher risk of heart disease, enabling early interventions. With accurate predictions, it can significantly aid healthcare providers in preventive care, ultimately reducing mortality rates. Further refinement of the model through feature selection, more comprehensive datasets, and real-time integration into healthcare systems could enhance its clinical applicability and impact.
Thank you for visiting the Heart Disease Prediction project repository! Feel free to drop a star if you like it.
