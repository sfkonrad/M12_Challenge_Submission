# [M12_Challenge_Submission](https://github.com/sfkonrad/M12_Challenge_Submission/blob/main/M12_Challenge_Submission/M12_Challenge_KonradK_credit_risk_resampling.ipynb)




#### Konrad Kozicki
### UCB-VIRT-FIN-PT-12-2020-U-B-TTH
---

# Credit Risk Classification and Reporting 


---
---

### Write a Credit Risk Analysis Report

For this challenge, we were tasked with composing a brief report that presents a summary and an analysis of the performance of two machine learning models that we compared in this challenge. 

We were instructed to produce our report by using the template provided in the `Starter_Code.zip` directory, ensuring that our summary report includes the following:

1. An Overview of the Analysis: 
> - Explaining the purpose of this analysis.
 
2. The Results of Our Analysis: 
> - Using bulleted lists for describing the balanced accuracy scores and the precision and recall scores of both machine learning models.

3. A Summary of Our Analysis: 
> - Summarizing the results from the machine learning models. 
> - Comparing the two versions of the dataset predictions. 
> - Including our recommendation for the model to use, if any, on the original vs. the resampled data. 

---
---
# CREDIT RISK ANALYSIS REPORT

## Introduction
Credit risk poses a classification problem that is inherently imbalanced. This is caused by healthy loans far outnumbering risky loans. For this Project, we’ve applied various techniques to **train** and **evaluate** models with **imbalanced classes**.  A dataset composed of historical lending activity from a peer-to-peer lending services company was employed to build a model that can identify the creditworthiness of borrowers.

The following sections constitute the scope of our analysis:

* Split the Data into Training and Testing Sets

* Create a Logistic Regression Model with the Original Data

* Predict a Logistic Regression Model with Resampled Training Data 

---

# OVERVIEW
### Split the Data into Training and Testing Sets

1. Read the `lending_data.csv` data from the `Resources` folder into a Pandas DataFrame.

2. Create the labels set (`y`)  from the “loan_status” column, and then create the features (`X`) DataFrame from the remaining columns.

    > **Note** A value of `0` in the “loan_status” column means that the loan is healthy. A value of `1` means that the loan has a high risk of defaulting.  

3. Check the balance of the labels variable (`y`) by using the `value_counts` function.

4. Split the data into training and testing datasets by using `train_test_split`.

### Create a Logistic Regression Model with the Original Data

Employ your knowledge of logistic regression to complete the following steps:

1. Fit a logistic regression model by using the training data (`X_train` and `y_train`).

2. Save the predictions on the testing data labels by using the testing feature data (`X_test`) and the fitted model.

3. Evaluate the model’s performance by doing the following:

    * Calculate the accuracy score of the model.

    * Generate a confusion matrix.

    * Print the classification report.

4. Answer the following question: How well does the logistic regression model predict both the `0` (healthy loan) and `1` (high-risk loan) labels?

### Predict a Logistic Regression Model with Resampled Training Data

Did you notice the small number of high-risk loan labels? Perhaps, a model that uses resampled data will perform better. You’ll thus resample the training data and then reevaluate the model. Specifically, you’ll use `RandomOverSampler`.

To do so, complete the following steps:

1. Use the `RandomOverSampler` module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points. 

2. Use the `LogisticRegression` classifier and the resampled data to fit the model and make predictions.

3. Evaluate the model’s performance by doing the following:

    * Calculate the accuracy score of the model.

    * Generate a confusion matrix.

    * Print the classification report.
    
4. Answer the following question: How well does the logistic regression model, fit with oversampled data, predict both the `0` (healthy loan) and `1` (high-risk loan) labels?

---

