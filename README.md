# Data Science 101
Repository and detailed plan to become an applied data scientist within 2 months

**Pre-requisites**:
1. Basics of Python
2. Handling data using Pandas, Numpy
3. Basic data visualization using Matplotlib
4. SQL and ETL process which involves joining, merging, conditional queries etc.

## Week 1 - EDA and Data Processing
**Assignment 1: Big Mart III Data- EDA and Feature Engineering**

**Link**: [BigMart Sales Data | Kaggle](https://www.kaggle.com/brijbhushannanda1979/bigmart-sales-data)

**Description:** Pre-process the dataset by doing data cleaning, handling null/missing values, etc and then perform Exploratory Data Analysis (EDA) on the Big Mart Sales Dataset  (It’s up to you to explore different ways in which you can describe and visually represent the dataset)

**Dataset:** Big Mart III Dataset

**Assignment 2: Movie Genre Data- EDA and One Hot Encoding**

**Link**: [Movie Genre | Kaggle](https://www.kaggle.com/neha1703/movie-genre-from-its-poster)

**Description:** Although the actual dataset contains lot of images, but for the ETL work we will use the MovieGenre.csv which is under the Data section. The main problem is about Movie Genre classification from image posters.But for this assignment, from the actual dataset (MovieGenre.csv) you would have to create a One Hot Encoded processed dataset (please search online what one-hot encoding is). So, the output should look like the processed_data.csv as attached. You would have to do exploratory data analysis (EDA) on the dataset and finally, on the processed dataset after the ETL process, you would be plotting the distribution of the entire dataset.
In short, three broad tasks for the assignment:
1.	From MovieGenre.csv, pre-process the dataset to address missing values, null values, etc and then do EDA to understand the data
2.	Transform MovieGenre format to ProcessData format ( So if each movie has multiple genres, it should create the specific genre as a column and mark 1 for that, otherwise 0 for other genres)
3.	Visualization on distribution of the processed dataset using bar charts

**Dataset:** Movie Genre Dataset


## Week 2 - Supervised Machine Learning

**Assignment 3: Big Mart III Regression Problem**

**Link**: [BigMart Sales Data | Kaggle](https://www.kaggle.com/brijbhushannanda1979/bigmart-sales-data)

**Description:** Regression Modelling – Using assignment 1, do Regression modelling on the Big Mart III Data Set. You must cover below models and concepts  
•	Linear Regression 
•	Why do we use Ridge and Lasso Regression  
•	PCA and Cross-Validation  
•	Regression Boosting techniques  
•	Bias Variance trade-off

**Dataset:** Big Mart III Dataset

**Assignment 4: Loan Prediction Classification Problem**

**Description:** In this assignment, you will be building end to end Classification model on the Loan Prediction dataset using the following algorithms:
•	Logistic regression, Boosting 
•	Decision tree, Random forest  
•	Bagging, boosting and stacking  
Note: Use appropriate model evaluation metrics used to test models  

**Dataset:** Loan Prediction Dataset

## Week 3 - Tuning Supervised Machine Learning Models

**Assignment 5: Heart Disease Classification Problem**

**Description:** In this assignment, you will be building end to end Classification model on the heart disease dataset using the following algorithms:
1.	K- Nearest Neighbours
2.	Support Vector Machines
3.	Decision Trees
4.	Random Forests
5.	Logistic Regression
Apply your existing knowledge on EDA. Explore and focus on feature engineering for this assignment. So, for this you need to know what is feature engineering and why do we need it. You need to know various ways of finding feature importance. (Previously we have explored ETL and one hot encoding which are some of the feature engineering techniques used, now explore other techniques as well). Note: Split your dataset into 70% train 20% dev and 10% test set.
Explore on the following classification model evaluation metrics:
1.	Accuracy
2.	Precision
3.	Recall
4.	AUC-ROC score
5.	F1 score 
6.	Confusion Matrix
Based on these metrics, you would have to report the result for your Train Set, Dev Set and Test Set.
Using the results, you would have to tell whether the model is going through Over-Fitting or Under-fitting and explain these two phenomena in terms of Bias and Variance.
After that you would have to suggest various ways to fight over-fitting and under-fitting, which ever you have in your solution.

**Dataset:** Heart Disease UCI Dataset

**Assignment 6: Hand digit classification problem**

**Link**: MNIST Digit Dataset

**Description:** A multi-class classification problem to train a machine learning model to predict hand written digits from images. Experiment your results with stacking various models like XGBoost, Random Forest and try to tune your model to get a training, testing and validation accuracy over 99%.

**Dataset:** MNIST Digit Dataset


## Week 4 - Unsupervised Machine Learning

**Assignment 7: **

**Link**: 

**Description:** 

**Dataset:** 


**Assignment 8: **

**Link**: 

**Description:** 

**Dataset:** 

## Week 5 - Time Series Data Analysis

**Assignment 9: Univariate Time Series Analysis**

**Description:** This time you are helping out Unicorn Investors with your data hacking skills. They are considering making an investment in a new form of transportation called JetRail. JetRail uses Jet propulsion technology to run rails and move people at a high speed! While JetRail has mastered the technology and they hold the patent for their product, the investment would only make sense, if they can get more than 1 Million monthly users with in next 18 months.You need to help Unicorn ventures with the decision. They usually invest in B2C start-ups less than 4 years old looking for pre-series A funding. To help Unicorn Ventures in their decision, you need to forecast the traffic on JetRail for the next 7 months. You are provided with traffic data of JetRail since inception in the train file.

Note: Make sure you cover the following concepts 
1)	Why and how is time series different from Regression
2)	What are different types of time series predictions 
3)	What is Hypothesis Generation
4)	Understanding the Dataset Structure and Content
5)	Feature Extraction
6)	Exploratory Analysis
7)	Splitting the data into training and validation part
8)	Modelling techniques
9)	Holt’s Linear Trend Model on daily time series
10)	Holt Winter’s Model on daily time series
11)	ARIMA model
12)	Parameter tuning for ARIMA model
13)	SARIMAX model on daily time series
14)	Calculate error using different evaluation metrics 


**Dataset:** Jet Rail Time Series Data


**Assignment 10: Multivariate Time Series Analysis**

**Link**: 

**Description:** 

**Dataset:** 

## Week 6 - Deep Learning

**Assignment 11: **

**Link**: 

**Description:** 

**Dataset:** 

**Assignment 12: **

**Link**: 

**Description:** 

**Dataset:** 

## Week 7 - Productionizing ML models using Microsoft Azure

**Assignment 13: ** Deploying ML models as API services

**Link**: 

**Description:** 

**Dataset:** 

**Assignment 14: ** Azure Machine Learning and Cognitive Services

**Link**: 

**Description:** 

**Dataset:** 


## Week 8 - Data Reporting and Visualization

**Assignment 15: ** Dashboarding using PowerBi

**Link**: 

**Description:** 

**Dataset:** 

**Assignment 16: ** Dashboarding using PowerBi

**Link**: 

**Description:** 

**Dataset:** 



