### [M12_Challenge_Submission](https://github.com/sfkonrad/M12_Challenge_Submission/blob/main/M12_Challenge_Submission/M12_Challenge_KonradK_credit_risk_resampling.ipynb)




#### Konrad Kozicki
### UCB-VIRT-FIN-PT-12-2020-U-B-TTH
---

## Credit Risk Classification and Reporting 


---

### Write a `Credit Risk Analysis Report`

For this challenge, we were tasked with composing a brief report that presents a summary and an analysis of the performance of two machine learning models that we compared in this challenge. 

We were instructed to produce our report by using the template provided in the `Starter_Code.zip` directory, ensuring that our summary report includes the following:

> 1. An Overview of the Analysis: 
>     - Explaining the purpose of this analysis.
> 
> 2. The Results of Our Analysis: 
>     - Using bulleted lists for describing the balanced accuracy scores and the precision and recall scores of both machine learning models.
>
> 3. A Summary of Our Analysis: 
>     - Summarizing the results from the machine learning models. 
>     - Comparing the two versions of the dataset predictions. 
>     - Including our recommendation for the model to use, if any, on the original vs. the resampled data. 

---
---
# `CREDIT RISK ANALYSIS REPORT`

### Introduction
Credit risk poses a classification problem that is inherently imbalanced. This is caused by healthy loans far outnumbering risky loans. For this Project, we’ve applied various techniques to **train** and **evaluate** models with **imbalanced classes**.  A dataset composed of historical lending activity from a peer-to-peer lending services company was employed to build a model that can identify the creditworthiness of borrowers.

The following sections constitute the scope of our analysis:

* Splitting the Data into Training and Testing Sets

* Creating a Logistic Regression Model with the Original Data

* Predicting a Logistic Regression Model with Resampled Training Data 



---

## OVERVIEW

To build a model that identifies borrowers' creditworthiness, we used our knowledge of the **imbalanced-learn** library to develop a logistic regression model for comparing two versions of the historical lending activity datasets from a peer-to-peer lending services company:
* First, we used the original dataset.
* Second, we resample the data by using the `RandomOverSampler` module from the imbalanced-learn library.

For both cases, we: 
   > - obtained the count of the target classes
   >
   > - trained a logistic regression classifier
   >
   > - calculated the balanced accuracy score
   >
   > - generated a confusion matrix
   >
   > - generated a classification report

### 1) Splitting the Data into Training and Testing Sets
1. We read the `lending_data.csv` data from the `Resources` folder into a Pandas DataFrame.
2. Created the labels set (`y`)  from the “loan_status” column, and then created the features (`X`) DataFrame from the remaining columns.
    **`Note`** 
    > - A value of `0` in the “loan_status” column indicates that the loan is healthy. 
    > - A value of `1` indicates that the loan has a high risk of defaulting.  
3. We checked the balance of the labels variable (`y`) by using the `value_counts` function.
4. Split the data into training and testing datasets by using `train_test_split`.

### 2) Creating a Logistic Regression Model with the Original Data
Employing our logistic regression skillz to complete the following steps:
1. Fit a logistic regression model by using the training data (`X_train` and `y_train`).
2. Save the predictions on the testing data labels by using the testing feature data (`X_test`) and the fitted model.
3. Evaluate the model’s performance by doing the following:
    * Calculate the accuracy score of the model.
    * Generate a confusion matrix.
    * Print the classification report.

### 3) Predicting a Logistic Regression Model with Resampled Training Data
As expected, the small number of loans labeled `high-risk` requires us to develop a model that uses resampled data to help increase its performance. We did this by using `RandomOverSampler` to resample the training data and then reevaluate the model. 
To do so, we completed the following steps:
1. Used the `RandomOverSampler` module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points. 
2. Used the `LogisticRegression` classifier and the resampled data to fit the model and make predictions.
3. Evaluated the model’s performance by doing the following:
    * Calculate the accuracy score of the model.
    * Generate a confusion matrix.
    * Print the classification report.
    


---

## ANALYSIS RESULTS


* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
   > - count of the target classes
   > ![image.png](attachment:b839f64d-86ea-4060-aeda-e87254a231b2.png)
   > ![image.png](attachment:56330d55-43ab-443d-8a66-d37035fb5aa3.png)



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
   > - count of the target classes
   > ![image.png](attachment:113afb25-69ad-444e-9c40-dca1414c2280.png)
   > ![image.png](attachment:5a79d0af-b642-40c6-89de-d9881f3e7195.png)



---

## SUMMARY

* compare the count of the target classes
   > ![image.png](attachment:f9d75c20-613d-40cb-bd4d-63a18a6ca86d.png)
   
   
* compare the balanced accuracy scores
   > ![image.png](attachment:309a7eda-7ec2-45c2-b796-ae613e0d946e.png)
   
   
* compare the confusion matrixis
   > ![image.png](attachment:e38c78d1-b3ca-4969-be78-83c82adaefaf.png)
   
   
* compare the classification reports
   > ![image.png](attachment:66a303e7-09d9-4dee-8092-e0c25ded6323.png)
   
   




> 3. A summary: Summarize the results from the machine learning models. Compare the two versions of the dataset predictions. Include your recommendation for the model to use, if any, on the original vs. the resampled data. If you don’t recommend either model, justify your reasoning.
>
>
>
> Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
> * Which one seems to perform best? How do you know it performs best?
> * Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
> 
> 
> 
> Summarize the results from the machine learning models. Compare the two versions of the dataset predictions. Include your recommendation, if any, for the model to use the original vs. the resampled data. If you don’t recommend either model, justify your reasoning. (4 points)






>  Q4. FIT/TRAIN: Answer the following question: How well does the logistic regression model predict both the `0` (healthy loan) and `1` (high-risk loan) labels?

> Q4. PREDICT: Answer the following question: How well does the logistic regression model, fit with oversampled data, predict both the `0` (healthy loan) and `1` (high-risk loan) labels?
