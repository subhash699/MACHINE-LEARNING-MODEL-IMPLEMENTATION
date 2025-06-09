# MACHINE-LEARNING-MODEL-IMPLEMENTATION

COMPANY: CODTECH IT SOLUCTIONS

NAME: SUBHASH K

INTERN ID: CT04DG769

DOMAIN: PYTHON PROGRAMMING

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH


As part of my internship at CodTech IT Solutions Pvt. Ltd., Task 4 was centered around the development of a predictive machine learning model using the scikit-learn library in Python. The goal of this task was to implement a classification model that could analyze input data and accurately predict outcomes. For this project, I chose to build a Spam Email Detection system, which is a well-known and practical application of machine learning, particularly in the field of natural language processing. The task involved training a supervised learning model that could classify messages as either ham (not spam) or spam, based on the content of the email.

To implement this project, I used Jupyter Notebook as the development environment. Jupyter is widely used in the data science community due to its ability to support interactive coding and visualization. The programming was done using Python version 3.12.2. I installed and used several essential libraries for the task, including scikit-learn for building and evaluating the model, pandas and numpy for data manipulation, and matplotlib and seaborn for data visualization and performance analysis.

The dataset used for this task was a file named spam.csv, which is a publicly available and commonly used dataset for spam detection projects. It consists of thousands of labeled email messages, categorized as either ham or spam. After loading the data using pandas, I selected the necessary columns and renamed them for ease of use. The textual labels were converted into numerical format using simple mapping, with ham labeled as 0 and spam as 1, which is required for classification algorithms in machine learning.

The next step involved preprocessing and transforming the text data into a suitable format for model training. Since machine learning models cannot work directly with raw text, I used the CountVectorizer class from scikit-learn to convert the emails into a matrix of token counts. This approach turns each message into a vector that represents the frequency of words in the text, making it possible to apply mathematical models for prediction.

Following feature extraction, the dataset was split into training and testing sets in an 80 to 20 ratio. The training data was used to train a Multinomial Naive Bayes classifier, which is particularly efficient and effective for text classification problems. Once the model was trained, it was tested on the unseen data from the test set to evaluate its performance.

To assess how well the model performed, I used several metrics. The accuracy score was calculated to measure the overall effectiveness of the model, which in this case was over 97 percent, indicating strong predictive capabilities. Additionally, a classification report was generated to provide more detailed insights into precision, recall, and F1-score, offering a deeper understanding of the model’s performance for both spam and ham classes.

A confusion matrix was also created and visualized using seaborn to better understand the number of true and false positives and negatives. This graphical representation allowed for a quick and effective evaluation of where the model performed well and where it could improve.

This project demonstrated real-world relevance, as spam email detection is critical in digital communication and cybersecurity. Completing this task helped me understand the end-to-end process of building a machine learning pipeline, from loading and preprocessing data to model training and evaluation. It improved my skills in using scikit-learn and solidified my understanding of classification algorithms, making this a valuable and practical learning experience during the internship.


OUTPUT:

![Image](https://github.com/user-attachments/assets/1229da23-cd18-4da0-a84b-00762fc560a4)

![Image](https://github.com/user-attachments/assets/7d34100e-9a15-4687-bead-5a37f0780596)
