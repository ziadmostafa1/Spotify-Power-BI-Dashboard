### Music Genre Classification Notebook Documentation

This notebook explores the process of building a classification model to predict music genres based on various audio features.

---

### 1. **Introduction**

The goal of this project is to develop a machine learning model that can accurately classify music genres. We will use a dataset containing various audio features of songs, such as danceability, energy, loudness, and more. The target variable is the genre of the music, represented by the 'Class' column.

---

### 2. **Data Import and Initial Inspection**

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
```

* The necessary libraries for data manipulation, visualization, and modeling are imported.
* The dataset is loaded into two DataFrames: `train` and `test`.

```python
test = pd.read_csv('test (2).csv')
train = pd.read_csv('train (1).csv')
```

---

### 3. **Data Exploration**

#### 3.1 Initial Data Inspection

```python
train.info()
```

* The training dataset contains 14,396 entries and 18 columns. The features include both numerical and categorical data, with some missing values.

#### 3.2 Data Visualization

##### 3.2.1 Distribution of Danceability

```python
plt.figure(figsize=(8, 6))
sns.histplot(train['danceability'], kde=True)
plt.title('Distribution of Danceability')
plt.show()
```

* The distribution of the 'danceability' feature is visualized to understand its spread and central tendency.

##### 3.2.2 Correlation Matrix

```python
plt.figure(figsize=(12, 10))
correlation_matrix = train.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
```

* A heatmap is generated to visualize the correlations between different features, helping to identify relationships that might be useful for feature selection.

##### 3.2.3 Class Distribution

```python
plt.figure(figsize=(10, 6))
sns.countplot(x='Class', data=train)
plt.title('Class Distribution')
plt.show()
```

* The distribution of the target variable 'Class' is plotted to understand the balance of genres in the dataset.

---

### 4. **Data Preprocessing**

#### 4.1 Data Cleaning

```python
train.drop({'Id','Artist Name','Track Name'}, axis=1, inplace=True)
train.fillna(train.mean(), inplace=True)
```

* Columns 'Id', 'Artist Name', and 'Track Name' are dropped as they are not useful for the classification task.
* Missing values are filled with the mean of their respective columns.

#### 4.2 Feature Engineering

```python
train['energy_valence_interaction'] = train['energy'] * train['valence']
train['tempo_binned'] = pd.cut(train['tempo'], bins=5, labels=False)
```

* Two new features are created:
  - `energy_valence_interaction`: Interaction term between 'energy' and 'valence'.
  - `tempo_binned`: Binned version of the 'tempo' feature.

---

### 5. **Modeling**

#### 5.1 Train-Test Split

```python
X = train.drop(columns='Class')
y = train['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

* The dataset is split into training and testing sets, with 80% of the data used for training and 20% for testing.

#### 5.2 Model Training

##### 5.2.1 Bagging Classifier

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, oob_score=True, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
bag_clf.oob_score_
```

* A Bagging Classifier with a Decision Tree as the base estimator is trained and evaluated using out-of-bag (OOB) samples.

##### 5.2.2 Voting Classifier

```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
svc_model = SVC(probability=True, random_state=42)
lr_model = LogisticRegression(random_state=42)

voting_clf = VotingClassifier(estimators=[('rf', rf_model), ('gb', gb_model), ('svc', svc_model)], voting='soft')
voting_clf.fit(X_train, y_train)
voting_pred = voting_clf.predict(X_test)
```

* A Voting Classifier is trained, combining the predictions of three models: Random Forest, Gradient Boosting, and Support Vector Classifier.

##### 5.2.3 Stacking Classifier

```python
stacking_clf = StackingClassifier(estimators=[('rf', rf_model), ('gb', gb_model), ('svc', svc_model), ('lr', lr_model)], final_estimator=lr_model, cv=5)
stacking_clf.fit(X_train, y_train)
stacking_pred = stacking_clf.predict(X_test)
```

* A Stacking Classifier is trained, which combines multiple models by feeding their outputs into a Logistic Regression model for final predictions.

---

### 6. **Model Evaluation**

#### 6.1 Evaluation Metrics

```python
print("Voting Classifier Results")
print("Confusion Matrix:\n", confusion_matrix(y_test, voting_pred))
print("\nClassification Report:\n", classification_report(y_test, voting_pred))
print("\nAccuracy Score:\n", accuracy_score(y_test, voting_pred))
```

* The Voting Classifier's performance is evaluated using a confusion matrix, classification report, and accuracy score.

#### 6.2 Stacking Classifier Performance

```python
print("Stacking Classifier Results")
print("Confusion Matrix:\n", confusion_matrix(y_test, stacking_pred))
print("\nClassification Report:\n", classification_report(y_test, stacking_pred))
print("\nAccuracy Score:\n", accuracy_score(y_test, stacking_pred))
```

* Similarly, the Stacking Classifier's performance is assessed.

---

### 7. **Prediction on Test Data**

```python
ID = test['Id']
test.drop({'Id', 'Artist Name', 'Track Name'}, axis=1, inplace=True)
test.fillna(test.mean(), inplace=True)

test['energy_valence_interaction'] = test['energy'] * test['valence']
test['tempo_binned'] = pd.cut(test['tempo'], bins=5, labels=False)

predictions = stacking_clf.predict(test)
submission = pd.DataFrame({'Id': ID, 'Class': predictions})
submission.to_csv('sub.csv', index=False)
```

* The final model (Stacking Classifier) is used to predict the genres for the test dataset.
* Predictions are saved to a CSV file for submission.

---

This notebook demonstrates the process of building and evaluating multiple machine learning models to classify music genres. The final model, a Stacking Classifier, is used to generate predictions on the test dataset.