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
   > 
> `1` Classification 
> - The Original Model correctly identifies the `1` Class with only 85% `pre` and 91% `rec`  
> - The model incorrectly classified 10-15% of the '1' Class predictions
> - Which is critical for us to model correctly in the context of this RA.
> 
> `0` Classification 
> - The Original Model correctly identifies the `0` Class with 100% `pre` and 99% `rec`  
> - The model correctly classified effectively 100% of the '0' Class predictions
> - Which doesn't matter much to us in the context of this RA.



### **Resampled Machine Learning Model 2:**
  * Description of Model 2's Target Class Count, Accuracy, Precision, and Recall scores:
   > ![image.png](https://github.com/sfkonrad/M12_Challenge_Submission/blob/main/M12_Challenge_Submission/Documentation/Images/M12C-MLM2_target_class_count.jpg?raw=true)
   > ![image.png](https://github.com/sfkonrad/M12_Challenge_Submission/blob/main/M12_Challenge_Submission/Documentation/Images/M12C-MLM2_basr_confusion_matrix_classification_report.jpg?raw=true)
   > 
> `1` Classification 
> - The Resampled Model correctly identifies the `1` Class with only 84% `pre` and 99% `rec`  
> - The model incorrectly classified 10-15% of the '1' Class predictions
> - Which is critical for us to model correctly in the context of this RA.
> 
> `0` Classification 
> - The Resampled Model correctly identifies the `0` Class with 100% `pre` and 99% `rec`  
> - The model correctly classified effectively 100% of the '0' Class predictions
> - Which doesn't matter much to us in the context of this RA.




---

## SUMMARY

### Target Class Count Comparison
Reveals that the **2500** 'high-risk' loans represent approx 3% of the Original data.  Whereas the `RandomOverSampler` populates **56271** entries, or exactly 50% of the Resampled loan data, as 'high-risk' loans.
   > ![image.png](https://github.com/sfkonrad/M12_Challenge_Submission/blob/main/M12_Challenge_Submission/Documentation/Images/M12C-MLM1vM2_target_class_count.jpg?raw=true)
   
   
### Balanced Accuracy Score Comparison
The **Resampled Model 2 appears to demonstrate the better performance** for overall accuracy with a Balanced Accuracy Score of **0.9937**.  Whereas the Original Model produced a score of **0.9520**.  Which implies that the Resampled Model is more accurate.  We'll take a closer look at the Precision of this accuracy and Recall below.
   > ![image.png](https://github.com/sfkonrad/M12_Challenge_Submission/blob/main/M12_Challenge_Submission/Documentation/Images/M12C-MLM1vM2_baso_basr.jpg?raw=true)
   
   
### Confusion Matrix Comparison
Best Performance:
> True Positive (TP):  **Original 18663** vs. 18649 (high-risk as high-risk // denying high-risk loans)
>
> True Negative (TN):  **Resampled 615** vs. 563 (low-risk as low-risk // approving low-risk loans)
>
> False Positive (FP): **Resampled 4** vs. 56 (low-risk as high-risk // denying low-risk loans)
>
> False Negative (FN): **Original 102** vs. 116 (high-risk as low-risk // approving high-risk loans)
   > ![image.png](https://github.com/sfkonrad/M12_Challenge_Submission/blob/main/M12_Challenge_Submission/Documentation/Images/M12C-MLM1vM2_confusion_matrix.jpg?raw=true)
   >>>>>> ![image.png](https://github.com/sfkonrad/M12_Challenge_Submission/blob/main/M12_Challenge_Submission/Documentation/Images/Un-Confusion_Matrix_by_Khaled_Karman.jpg?raw=true)
   >>>>>> © 2021 Khaled Karman
   
   
### Classification Report Comparison

   > ![image.png](https://github.com/sfkonrad/M12_Challenge_Submission/blob/main/M12_Challenge_Submission/Documentation/Images/M12C-MLM1vM2_classification_report.jpg?raw=true)
> #### Best Performance for type `1` Classification:
>
> * Precision (`pre`):  **Original 0.85** vs. 0.84 (how many are correctly classified among that class)
>
> * Recall (`rec`):     **Resampled 0.99** vs. 0.91 (the ability of the classifier to distinguish all the positive samples)
>
> * f1-score (`f1`): **Resampled 0.91** vs. 0.88 (the harmonic mean between `pre` & `rec`)
>
> * Geometric Mean (`geo`): **Resampled 0.99** vs. 0.95 ([Geometric Mean - How to Find, Formula, & Definition](https://tutors.com/math-tutors/geometry-help/geometric-mean))
>
> * Index of Balanced Accuracy (`iba`): **Resampled 0.99** vs. 0.90 ([used to evaluate learning processes in two-class imbalanced domains](https://link.springer.com/chapter/10.1007/978-3-642-02172-5_57#:~:text=This%20paper%20introduces%20a%20new,the%20highest%20individual%20accuracy%20rate.))
>
> * Support (`sup`): **Same 619** vs. 619. (the number of occurence of the given class(es) in the dataset)
---

Since the present objective is risk mitigation, the most prevalent concern in this analysis is to make certain that all the `1`'s, or 'high-risk' borrowers can be identified and declined the loan.  We must closely evaluate and work to reduce the False Negative (FN) predictions to more accurately distinguish the 'high-risk' borrowers from the 'low-risk'.  We mitigate more risk by not classifying 'high-risk' borrowers as 'low-risk' and therefore avoid lending on loans that are likely to default.

Although Precision is slightly better in the Original Model (0.85 vs. 0.84), the Resampled Model appears to outperform the Original across the remainder of our classification metrics.

Notably, the Resampled Model correctly **recalled** the results classified as `1` nearly 10% better (0.99 vs. 0.91) than the Original Model.

The recommendation is to initially employ the 'Resampled Model 2' to classify risk among new applicants due to its superior `rec` score. It's further recommended that engineers continue refining the regression model to continue reducing instances of a False Negative prediction. Perhaps evolving into a model that more closely resembles **neural net** frameworks.


---
---
---
---

### Citations
>
> * [Mr. Khaled Karman]() for ['Class Predictions v Actual' Graphic]() from [Class Activity Day 13.2]()
>
> * [Scikit-learn's Documentatinon on `recall`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
>
> * ["How to interpret classification report of scikit-learn?"](https://datascience.stackexchange.com/a/64443) by [bradS](https://datascience.stackexchange.com/users/45374/brads) from [Stack Exchange](https://datascience.stackexchange.com/)
>
> * García V., Mollineda R.A., Sánchez J.S. (2009) [Index of Balanced Accuracy: A Performance Measure for Skewed Class Distributions](https://link.springer.com/chapter/10.1007/978-3-642-02172-5_57#:~:text=This%20paper%20introduces%20a%20new,the%20highest%20individual%20accuracy%20rate). In: Araujo H., Mendonça A.M., Pinho A.J., Torres M.I. (eds) Pattern Recognition and Image Analysis. IbPRIA 2009. Lecture Notes in Computer Science, vol 5524. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-02172-5_57
