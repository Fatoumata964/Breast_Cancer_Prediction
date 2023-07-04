# Breast_Cancer_Prediction
This code helps you classify breast cancer(malignant and benign tumors) using Pyspark with 4 
different models: DecisionTreeClassifier, LinearSVC, LogisticRegression and 
RandiomForestClassifier from MLlib. We also used built-in cross-validation to optimize 
hyperparameters in our algorithms and pipelines.

 Pyspark-3.3.1

 Py4j-0.10.9.5

The data used during this project is available on kaggle in csv format:
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

1. ID number 
2. Diagnosis (M = malignant, B = benign) 
3. -32

The project is directly made under Colab https:/colab.research.google.com/ but can be done with 
jupyter by installing pyspark locally.

Under colab : !pip install pyspark , will do all the job.

from pyspark.sql import SparkSession 

spark = SparkSession.builder.getOrCreate() 

df = spark.read.csv(path, inferSchema = True, header = True) 

 Visualisation :

import vizualiz 

vizualiz.pca_viz(df)

vizualiz.plot_corr_matrix(df)

 Data processing :

import mylib 

train, test = mylib.process1(spark, path)

train.show(3)

 Predict with one model :

import logisticReg

model = logisticReg.logisticRegress(train) 

pred = model.transform(test) 

pred.show(3)

 The project is finished.

 Try other classification models like MultilayerPerceptronClassifier

 Or Pyspark Gradient-boosted Tree Classifier

 This project was carried out for the validation of the Big Data Algorithm course 
(Paris8 University)

 Many thanks to our teacher Rakia Jaziri
