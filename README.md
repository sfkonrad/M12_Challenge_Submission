### [M12_Challenge_Submission](https://github.com/sfkonrad/M12_Challenge_Submission/blob/main/M12_Challenge_Submission/M12_Challenge_KonradK_credit_risk_resampling.ipynb)




#### Konrad Kozicki
### UCB-VIRT-FIN-PT-12-2020-U-B-TTH
---

## Credit Risk Classification and Reporting 


---

### Write a [`CREDIT RISK ANALYSIS REPORT`](https://github.com/sfkonrad/M12_Challenge_Submission#credit-risk-analysis-report)

For this challenge, we were tasked with composing a brief report that presents a summary and an analysis of the performance of two machine learning models that we compared. 

We were instructed to produce our report by using the template provided in the `Starter_Code.zip` directory, ensuring that our summary report includes the following:

> 1. An [Overview](https://github.com/sfkonrad/M12_Challenge_Submission#overview) of the Analysis: 
>     - Explaining the purpose of this analysis.
> 
> 2. The [Results](https://github.com/sfkonrad/M12_Challenge_Submission#analysis-results) of Our Analysis: 
>     - Using bulleted lists for describing the balanced accuracy scores and the precision and recall scores of both machine learning models.
>
> 3. A [Summary](https://github.com/sfkonrad/M12_Challenge_Submission#summary) of Our Analysis: 
>     - Summarizing the results from the machine learning models. 
>     - Comparing the two versions of the dataset predictions. 
>     - Including our recommendation for the model to use, if any, on the original vs. the resampled data. 

---
---
# `CREDIT RISK ANALYSIS REPORT`

### Introduction
Credit risk poses a classification problem that is inherently imbalanced. This is caused by healthy loans far outnumbering risky loans. For this assignment, we’ve applied various techniques to **train** and **evaluate** models with **imbalanced classes**.  A dataset composed of historical lending activity from a peer-to-peer lending services company was employed to build a model that can identify the creditworthiness of borrowers.

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


### **Original Machine Learning Model 1:**
  * Description of Model 1's Target Class Count, Accuracy, Precision, and Recall scores:
   > ![image.png](https://github.com/sfkonrad/M12_Challenge_Submission/blob/main/M12_Challenge_Submission/Documentation/Images/M12C-MLM1_target_class_count.jpg?raw=true)
   > ![image.png](https://github.com/sfkonrad/M12_Challenge_Submission/blob/main/M12_Challenge_Submission/Documentation/Images/M12C-MLM1_baso_confusion_matrix_classification_report.jpg?raw=true)



### **Resampled Machine Learning Model 2:**
  * Description of Model 2's Target Class Count, Accuracy, Precision, and Recall scores:
   > ![image.png](https://github.com/sfkonrad/M12_Challenge_Submission/blob/main/M12_Challenge_Submission/Documentation/Images/M12C-MLM2_target_class_count.jpg?raw=true)
   > ![image.png](https://github.com/sfkonrad/M12_Challenge_Submission/blob/main/M12_Challenge_Submission/Documentation/Images/M12C-MLM2_basr_confusion_matrix_classification_report.jpg?raw=true)



---

## SUMMARY

### Target Class Count Comparison
Reveals that the **2500** 'Bad' loans represent approx 3% of the Original data.  Whereas the `RandomOverSampler` populates **56271** entries, or exactly 50% of the Resampled loan data, as 'Bad' loans.
   > ![image.png](https://github.com/sfkonrad/M12_Challenge_Submission/blob/main/M12_Challenge_Submission/Documentation/Images/M12C-MLM1vM2_target_class_count.jpg?raw=true)
   
   
### Balanced Accuracy Score Comparison
The **Resampled Model 2 appears to demonstrate the better performance** for overall accuracy with a Balanced Accuracy Score of **0.9937**.  Whereas the Original Model produced a score of **0.9520**.  Which implies that the Resampled Model is more accurate.  We'll take a closer look at the Precision of this accuracy below.
   > ![image.png](https://github.com/sfkonrad/M12_Challenge_Submission/blob/main/M12_Challenge_Submission/Documentation/Images/M12C-MLM1vM2_baso_basr.jpg?raw=true)
   
   
### Confusion Matrix Comparison
Best Performance:
> True Positive (TP):  **Original 18663** v 18649 (bad as bad // denying bad loans)
>
> True Negative (TN):  **Resample 615** v 563 (good as good // approving good loans)
>
> False Positive (FP): **Resample 4** v 56 (good as bad // losing out on good loans)
>
> False Negative (FN): **Original 102** v 116 (bad as good // assuming more risky loans)
   > ![image.png](https://github.com/sfkonrad/M12_Challenge_Submission/blob/main/M12_Challenge_Submission/Documentation/Images/M12C-MLM1vM2_confusion_matrix.jpg?raw=true)
   >>>>>> ![image.png](https://github.com/sfkonrad/M12_Challenge_Submission/blob/main/M12_Challenge_Submission/Documentation/Images/Un-Confusion_Matrix_by_Khaled_Karman.jpg?raw=true)
   >>>>>> © 2021 Khaled Karman
   
   
### Classification Report Comparison

   > ![image.png](https://github.com/sfkonrad/M12_Challenge_Submission/blob/main/M12_Challenge_Submission/Documentation/Images/M12C-MLM1vM2_classification_report.jpg?raw=true)
   

























>> 3. A summary: Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
>> * Which one seems to perform best? How do you know it performs best?
>> * Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )... the `1`'s' are rare, but critical to identify and predict
>> 
>> 
>> Compare the two versions of the dataset predictions. Include your recommendation for the model to use, if any, on the original vs. the resampled data. If you don’t recommend either model, justify your reasoning. (4 points)




>>  Q4. FIT/TRAIN: Answer the following question: How well does the logistic regression model predict both the `0` (healthy loan) and `1` (high-risk loan) labels?
>>
>> Q4. PREDICT: Answer the following question: How well does the logistic regression model, fit with oversampled data, predict both the `0` (healthy loan) and `1` (high-risk loan) labels?









Since the present objective is risk mitigation, we closely observe the (FN) to better distinguish the bad borrowers from the good.

We mitigate more risk by not classifying 'Bad' borrowers as 'Good' and therefore avoid lending to bad borrowers

What's most important in this analysis is to make certain all the `1`'s, or 'Bad' borrowers can be identified and declined the loan.

Rather than ensuring that every `0`, or 'Good' borrower, is approved for their loan, as might be considered in a reckless growth strategy. 


If growth is the main objective, observing (FPs) to avoid losing out on business by classifying more borrowers as 'Good' 
at the risk of declining some 'Good' borrowers their loans


If the goal is risk mitigation, then observing the FN avoid bad loans. to better id bad borrowrs
mitigate more risk by not classifying Bad borrowers as Good
to avoid lending to bad borrowers






