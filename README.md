# ACM 30-Day Machine Learning Challenge – Kilari Vamsika

Hi! I'm Vamsika, a data science enthusiast eager to solve real-world problems using ML, statistics, and analytical tools. This repository is a curated journey through multiple challenge cycles focusing on preprocessing, modeling, and evaluation tasks to build solid data intuition and technical fluency.

---

## Cycle 1

### 📅 Daily Progress

| Day | Task Summary |
|-----|--------------|
| **Day 1** | Performed basic data cleaning and exploratory data analysis (EDA) on a burnout dataset. Handled missing values, explored distributions, and generated summary statistics. |
| **Day 2** | Applied preprocessing techniques for machine learning. Encoded categorical variables using OneHotEncoder and combined them with scaled numerical features to prepare the dataset for modeling. Performed Regression and decided which is best among three the types. |
| **Day 3** | Preprocessed data using one-hot encoding and scaling. Trained Logistic Regression and LDA models to classify burnout, then evaluated them using Accuracy, Confusion Matrix, and ROC-AUC with ROC curve visualization. |
| **Day 4** | Trained Decision Tree, Random Forest, and k-NN models. Used Mutual Information to pick top 3 features and compared model accuracy before and after feature selection. |
| **Day 5** | A Random Forest model predicts burnout risk after encoding and scaling the data. The top 3 features were used to build a simpler model without losing accuracy. A heatmap shows how these features relate to each other. |
| **Main Challenge** | This model predicts medical insurance costs after cleaning and preprocessing the data. Tested different regression methods to find the best fit. Using mutual info, we picked the most important features and trained a Random Forest model for better accuracy. The final model gives reliable predictions with a solid R² score. It's efficient yet powerful enough for real-world use. |

### 📁 Repository Contents – Cycle 1
- `Day1.ipynb` – Data cleaning and exploration
- `Day2.ipynb` – Feature encoding and preprocessing
- `Day3.ipynb` – Classifier Arena
- `Day4.ipynb` – Tree-Based Models + k-NN + Feature Selection
- `Day5.ipynb` – 3-Feature Showdown
- `MAIN_CHALLENGE_1.ipynb` – Medical cost regression project

---

## Cycle 2

### 📅 Daily Progress

| Day | Task Summary |
|-----|--------------|
| **Phase 1** | Explored ensemble learning with Bagging and Boosting using the Breast Cancer dataset. Trained and compared Random Forest, AdaBoost, and XGBoost classifiers. Accuracy was measured for all models and the best-performing one was highlighted. Label encoding and feature scaling were applied, followed by model evaluation using classification report. |
| **Phase 2** | Implemented SVM classification on the Breast Cancer dataset using Linear, RBF, and Polynomial kernels. Applied outlier removal (IQR), label encoding, feature scaling, and PCA for 2D visualization. Compared models using accuracy and classification reports. |
| **Phase 3** | Applied KMeans clustering on the Iris dataset to uncover hidden groupings in the data. Preprocessing steps included label encoding, outlier removal using IQR, and feature scaling. Principal Component Analysis (PCA) was used to reduce dimensionality and visualize clusters in 2D. The Elbow Method was used to determine the optimal number of clusters. Real-world applications of clustering were also discussed. |
| **Phase 4** | Performed dimensionality reduction on the 20 Newsgroups dataset using TF-IDF followed by Truncated SVD. The high-dimensional TF-IDF matrix was reduced to two components to visualize document similarity. A scatter plot was generated to show separability of newsgroups. Optionally applied KMeans clustering with 20 clusters, and evaluated grouping performance using silhouette score and visual comparison to true labels. |
| **Phase 5** | Explored model validation techniques using the Breast Cancer dataset. Random Forest was chosen to demonstrate K-Fold Cross-Validation and the bias-variance trade-off. Learning curves were plotted to analyze training vs validation accuracy across varying dataset sizes, enabling identification of overfitting or underfitting behaviors. |
| **Main Challenge** | Applied sentiment analysis on the Sentiment140 dataset to classify tweets into Negative, Neutral, and Positive categories. Performed text cleaning (removing URLs, mentions, special characters), label mapping, and TF-IDF vectorization for feature extraction. Trained a classification model (e.g., Logistic Regression or SVM), evaluated it using accuracy, confusion matrix, and classification report. |


---

### 📁 Repository Contents – Cycle 2
- `Phase1.ipynb` – Bagging vs Boosting (Random Forest, AdaBoost, XGBoost)
- `Phase2.ipynb` – SVM Classification with Linear, RBF, and Polynomial kernels + PCA visualization
- `Phase3.ipynb` – Unsupervised Learning (Clusters)
- `Phase4.ipynb` – SVD + PCA
- `Phase5.ipynb` –  Model Validation & Selection
- `MAIN_CHALLENGE_2.ipynb` – Tweet Sentiment Analysis

---

## 🧰 Tools & Libraries Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn

