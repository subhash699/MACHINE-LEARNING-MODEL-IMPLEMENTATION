# MACHINE-LEARNING-MODEL-IMPLEMENTATION

COMPANY: CODTECH IT SOLUCTIONS

NAME: SUBHASH K

INTERN ID: CT04DG769

DOMAIN: PYTHON PROGRAMMING

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH


As part of my internship at **CodTech IT Solutions Pvt. Ltd.**, Task 4 focused on building a **predictive machine learning model** using the **scikit-learn** library. The objective was to implement a classification model that could accurately predict outcomes from a dataset. For this task, I chose to build a **Spam Email Detection system**, a classic problem in natural language processing (NLP) and machine learning. The goal of the system was to classify email messages as either **"ham" (not spam)** or **"spam"**, using a supervised learning approach.

To carry out this project, I used **Jupyter Notebook**, a powerful platform for writing and running Python code, especially suited for data science workflows. The programming was done in **Python 3.12.2**, and all required libraries such as **scikit-learn**, **pandas**, **numpy**, **matplotlib**, and **seaborn** were installed using `pip`. These libraries were essential for handling data, training the model, and evaluating performance through visualizations and metrics.

The dataset used was the **spam.csv** file, a well-known dataset containing thousands of email messages labeled as either spam or ham. This file was loaded into a pandas DataFrame, and only the relevant columns were selected and renamed for simplicity. The categorical labels ("ham" and "spam") were encoded into binary format (0 and 1) to be compatible with machine learning models.

After loading the data, the next crucial step was **data preprocessing and feature extraction**. Textual email content cannot be fed directly into machine learning models, so I used **CountVectorizer** from scikit-learn to convert the messages into a numerical format. This technique transformed each email into a vector of word counts, making it suitable for training.

Once the data was preprocessed, I split it into training and testing sets using an 80-20 ratio. I then trained a **Multinomial Naive Bayes classifier**, which is particularly effective for text classification problems. After the model was trained on the training data, predictions were made on the test set.

For evaluating the model's performance, several metrics were calculated using scikit-learn functions. The **accuracy score** provided an overall performance measure, and in this case, the model achieved over **97% accuracy**, indicating a high level of correctness in its predictions. Additionally, a **classification report** was generated to provide insights into **precision**, **recall**, and **F1-score** for each class (ham and spam). These metrics gave a detailed view of how well the model distinguished between spam and non-spam messages.

To visually interpret the model's performance, a **confusion matrix** was plotted using the seaborn library. This matrix gave a snapshot of the model's true positives, false positives, true negatives, and false negatives. A heatmap representation of the confusion matrix made it easy to understand how often the model was misclassifying emails and where it was most accurate.

This task was highly relevant to real-world applications, as spam detection is a critical component of **email systems**, **messaging platforms**, and **digital security infrastructures**. The techniques learned and applied here—such as data preprocessing, vectorization, model training, and evaluation—are foundational to many machine learning applications. The project demonstrated the power and simplicity of scikit-learn for building robust machine learning models with minimal code and high accuracy.

In conclusion, **Task 4** enabled me to practically implement a full machine learning pipeline, from reading the dataset to making predictions and visualizing the results. It enhanced my understanding of classification techniques, model evaluation, and the importance of preprocessing textual data. This experience has equipped me with the knowledge and confidence to explore more advanced models and datasets in the future.

OUTPUT:

