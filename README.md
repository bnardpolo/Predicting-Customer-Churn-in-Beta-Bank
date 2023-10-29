# Predicting-Customer-Churn-in-Beta-Bank
Predicting Customer Churn in Beta Bank: A Data Science Approach to Retention
# Predicting Customer Churn in Beta Bank: A Data Science Approach to Retention

## Introduction:

Customer churn, the loss of customers over time, is a significant concern for businesses. In the case of Beta Bank, customers have been leaving gradually, impacting the bank's bottom line. Recognizing the cost-effectiveness of retaining existing customers compared to acquiring new ones, Beta Bank aims to predict customer churn to proactively address the issue.

This data science project focuses on building a predictive model to identify customers who are likely to leave the bank soon. By analyzing historical customer behavior and contract termination data, the goal is to develop a model with a high F1 score, indicating a balance between precision and recall in predicting customer churn.

The dataset used for this project, sourced from Kaggle, contains various features such as customer demographics, credit score, account balance, product usage, and more. Leveraging this data, the project aims to identify key factors that contribute to customer churn and develop an accurate prediction model.

To achieve the project's objectives, the approach involves several stages. Initially, the data will be downloaded and prepared, followed by an examination of class balance to understand the distribution of churned and retained customers. The project will then proceed with model development, starting with training the model without considering class imbalance and evaluating the F1 score.

Recognizing the impact of class imbalance on model performance, the project will further enhance the model's quality by considering techniques to address the imbalance issue. Various classification algorithms, including logistic regression, decision trees, random forests, and gradient boosting, will be explored. Hyperparameter tuning and model selection will be performed to identify the best combination of algorithms and parameters to maximize the F1 score.

## Class Balance Examination:

To check the class balance, we analyzed the target variable "Exited" which represents whether a customer has left the bank or not. We labeled the observations as follows:

- Positive Class (1): If the scale is balanced, indicating the customer has left the bank.
- Negative Class (0): If the scale is not balanced, indicating the customer has not left the bank.

Results: After labeling the observations, we obtained the following distribution:

- 0 (Negative Class): 7963
- 1 (Positive Class): 2037

Based on the distribution, we can observe that there is a class imbalance issue in the dataset. The negative class (0) representing customers who have not left the bank is significantly larger than the positive class (1) representing customers who have left the bank. This class imbalance can potentially affect the model's performance, as it may be biased towards predicting the majority class.

## Implications:

The class imbalance has important implications for model training and evaluation. Since the F1 score is the evaluation metric specified in the project, it is essential to consider the imbalance and employ techniques to address it. A high F1 score indicates a balance between precision and recall, which is crucial for accurately predicting customer churn.

Based on the provided F1 score of 0.1033, it suggests that the model's performance is poor and is not able to accurately predict the positive class. This is confirmed based on the previous Class Balance Examination. A higher F1 score implies that your model is performing well in both correctly identifying customers who are likely to churn (high recall) and accurately classifying customers as churned or retained (high precision). This can provide Beta Bank with valuable insights and actionable information to implement effective retention strategies and reduce customer churn.

## Model Improvement through Upsampling:

By performing upsampling on the minority class, you were able to improve the F1 score to 0.6202. This indicates that the upsampling technique helped the model better capture the patterns and characteristics of the churned customers, resulting in improved predictive performance.

Upsampling the minority class provides the model with more representative samples, which can help balance the class distribution and mitigate the impact of class imbalance. This approach allows the model to learn from a more diverse set of data and make better predictions for both the minority and majority classes.

## Final Testing:

An AUC-ROC score of 0.800 indicates that the model has good discriminative power and is able to effectively distinguish between churned and retained customers. The AUC-ROC (Area Under the Receiver Operating Characteristic Curve) is a popular evaluation metric for binary classification models.

A higher AUC-ROC score suggests that your model has a higher probability of assigning a higher predicted churn probability to actual churned customers compared to non-churned customers. In other words, the model can accurately rank the customers based on their likelihood of churn.

With an AUC-ROC score of 0.7880, your model performs significantly better than a random classifier (with an AUC-ROC score of 0.5) and demonstrates its effectiveness in predicting customer churn for Beta Bank.

The AUC-ROC score provides valuable insights into the overall performance of your model and complements the F1 score. While the F1 score focuses on balancing precision and recall, the AUC-ROC score evaluates the model's ability to rank the predicted probabilities correctly.

## Conclusion:

In this data science project, we aimed to predict customer churn for Beta Bank and develop a model with a high F1 score. By analyzing customers' past behavior and contract termination data, we built a predictive model using various techniques and evaluated its performance.

Initially, we examined the class balance and found a significant imbalance, with a majority of retained customers and a smaller number of churned customers. To address this issue, we employed undersampling techniques to balance the classes and improve the model's performance.

We trained and evaluated several classification algorithms, including logistic regression, decision trees, random forests, and gradient boosting. Through model selection and hyperparameter tuning, we identified the best combination of algorithms and parameters that maximized the F1 score.

Upon evaluating the final model, we achieved an F1 score of 0.6147, exceeding the project requirement. This indicates a good balance between precision and recall in predicting customer churn. Additionally, the AUC-ROC score of 0.7880 demonstrated the model's ability to effectively rank the predicted probabilities.

By leveraging the developed model, Beta Bank can proactively identify customers at risk of churning and implement targeted retention strategies. This can lead to improved customer satisfaction, reduced churn rate, and enhanced overall performance for the bank.

It's important to note that there is still potential for further improvement. Additional feature engineering, exploring other algorithms, and handling class imbalance with advanced techniques like oversampling can enhance the model's performance and increase the accuracy of churn predictions.

Overall, this project highlights the significance of data-driven approaches in customer churn prediction and retention efforts. By leveraging data science techniques, Beta Bank can take proactive measures to retain customers, optimize resources, and achieve long-term success in customer relationship management.

**For detailed instructions on using the code and replicating the project, please refer to the project's Jupyter Notebook and associated files.**
