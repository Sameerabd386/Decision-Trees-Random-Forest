**Decision Tree vs. Random Forest: Wine Classification**

A hands-on project demonstrating and comparing the performance of two powerful classification algorithms—Decision Trees and Random Forests—on the classic Wine dataset.

**📋 Table of Contents**



Project Overview


Core Concepts Explained


What is a Decision Tree?


What is a Random Forest?


Dataset Used


Technologies & Libraries


Step-by-Step Workflow


Results & Conclusion


How to Run This Project


**Project Overview**
This project serves as a clear and simple introduction to two fundamental machine learning algorithms for classification. We tackle a real-world problem: classifying wine into one of three types based on its chemical properties.

The main objective is to build and train both a Decision Tree and a Random Forest model on the same data, and then compare their accuracy to see which one performs better. This highlights the power of ensemble methods (like Random Forest) over individual models.

Core Concepts Explained


**What is a Decision Tree?**
A Decision Tree is one of the most intuitive machine learning models. It works like a flowchart of yes/no questions to arrive at a decision.

Analogy: Think of it as a single, highly methodical expert. It starts with the entire dataset and asks the best possible question to split the data into cleaner, more distinct groups. It repeats this process until it reaches a final prediction.
Pros: Easy to understand and visualize. The logic is transparent.
Cons: A single tree can easily "memorize" the training data (overfit) and may not generalize well to new, unseen data.


**What is a Random Forest?**

A Random Forest is a more advanced model that leverages the "wisdom of the crowd." Instead of relying on a single expert, it builds an entire forest of Decision Trees and combines their outputs.

Analogy: Think of it as a team of experts. Each expert (tree) is trained on a slightly different random subset of the data. To make a final prediction, the forest takes a majority vote from all the trees.
Pros: Generally much more accurate and robust than a single Decision Tree. It corrects for the overfitting issues of individual trees.
Cons: It's more of a "black box," as you can't easily visualize the logic of hundreds of trees at once.


**Dataset Used**

We used the Wine Dataset, a classic and well-understood dataset available directly from scikit-learn.

Source: sklearn.datasets.load_wine


Description: The data is the result of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars.

Features (Inputs): 13 chemical properties, such as alcohol, malic_acid, ash, and color_intensity.


Target (Output): The class of the wine, which can be class_0, class_1, or class_2.
Instances: 178 samples in total.


**🛠️ Technologies & Libraries**
This project is built entirely in Python and relies on the following core data science libraries:

Scikit-learn: For loading the dataset, splitting the data, and implementing the DecisionTreeClassifier and RandomForestClassifier models.
Pandas: For creating and manipulating DataFrames to easily view and handle our data.
📈 Step-by-Step Workflow
The project follows a standard machine learning pipeline:

Load Data: The Wine dataset is loaded from scikit-learn.

Data Preparation: The features (X) and target (y) are separated and viewed using a Pandas DataFrame.

Train-Test Split: The dataset is split into an 80% training set and a 20% testing set. This ensures our models are evaluated on data they have never seen before.

Model 1 (Decision Tree):
A DecisionTreeClassifier is instantiated.
The model is trained on the training data (X_train, y_train).
Predictions are made on the test data (X_test).




Model 2 (Random Forest):
A RandomForestClassifier (with 100 trees) is instantiated.
The model is trained on the same training data.
Predictions are made on the same test data.
Evaluation: The performance of both models is measured using the Accuracy Score, which calculates the percentage of correct predictions.

Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)




**📊 Results & Conclusion**
After training and evaluating both models on the test set, we obtained the following results:

Model	Accuracy on Test Set
Decision Tree	94.44%
Random Forest	100.00%
<br>
Conclusion: The Random Forest model significantly outperformed the single Decision Tree, achieving perfect accuracy on the test set.

This result clearly demonstrates the power of ensemble learning. By combining the predictions of 100 slightly different trees, the Random Forest was able to create a more robust and accurate model, effectively eliminating the errors that a single Decision Tree might make.

**▶️ How to Run This Project**
To run this project on your own machine, follow these steps:

Clone the repository:

Bash

git clone https://github.com/Sameerabd386/Decision-Trees-Random-Forest.git
cd Decision-Trees-Random-Forest
Ensure you have the required libraries:

Bash

pip install scikit-learn pandas
Run the Python script or Jupyter/Colab notebook:
The entire code is contained in a single file. Simply execute it to see the training process and the final accuracy comparison printed to the console.
