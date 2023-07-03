# diabetes-prediction-machine-learning
This project showcases the application of machine learning classifiers for predicting diabetes. By leveraging a diverse dataset and employing various classifiers. we aim to accurately identify individuals at risk of diabetes. 

Here is a summary of the key steps and results:

Data Preprocessing:
The dataset is loaded from a CSV file using pandas.
The first 20 rows of the dataset are displayed using db.head(20).
The shape of the dataset is displayed using db.shape.
The summary statistics of the dataset are displayed using db.describe().T.
Information about the dataset, including column names and data types, is displayed using db.info().
The 'age' column is converted to integer type using db.age = db.age.astype(int).

Data Splitting:
The dataset is split into two subsets based on the 'diabetes' column: one for positive cases (diabetes = 1) and one for negative cases (diabetes = 0).
The negative cases subset is sampled to match the number of positive cases, resulting in equal-sized subsets.
The two subsets are concatenated to create the final dataset for prediction.

Data Encoding:
Categorical variables 'gender' and 'smoking_history' are encoded using predefined mappings.
The encoded dataset is displayed.

Machine Learning Classifiers:
The dataset is split into training and testing sets using train_test_split.

The dataset is scaled using StandardScaler.

Several classifiers are trained and evaluated:

Decision Tree Classifier:
The decision tree classifier is created using DecisionTreeClassifier.
Accuracy score and confusion matrix are computed and displayed.

Random Forest Classifier:
The random forest classifier is created using RandomForestClassifier.
Accuracy score and confusion matrix are computed and displayed.

SVM Classifier:
Four SVM classifiers with different kernels (sigmoid, linear, poly, rbf) are created using SVC.
Accuracy score and confusion matrix are computed and displayed for each kernel.

Gradient Boosting Classifier:
The gradient boosting classifier is created using GradientBoostingClassifier.
Accuracy score and confusion matrix are computed and displayed.

Naive Bayes:
The Gaussian Naive Bayes classifier is created using GaussianNB.
Accuracy score and confusion matrix are computed and displayed.

K-Nearest Neighbors (KNN) Classifier:
The KNN classifier is created using KNeighborsClassifier.
Accuracy score and confusion matrix are computed and displayed.

Conclusion:
The accuracy scores and confusion matrices of all classifiers are presented.
The Random Forest Classifier and Gradient Boosting Classifier achieved the highest accuracy scores (90% and 91% respectively) on the given dataset.

#pandas #project #testing #statistics #training #python #dataanalyst #diabetes #diabetesmanagement #MachineLearning #HealthcareAI #DiabetesPrediction #PersonalizedHealthcare #TransformingLives
