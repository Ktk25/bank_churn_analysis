## Bank Customer Churn Analysis

This project analyzes and predicts churn for bank customers using their historical data. It incorporates data preprocessing, exploratory data analysis, feature selection, model building, evaluation, and monitoring.

### Exploratory Data Analysis:

We begin by identifying the key numeric and categorical variables:

```python
num_variables = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
cat_variables = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
```

For better insights into the distribution of these variables, we create buckets:

```python
data['Credit_Score_Bucket'] = pd.qcut(data['CreditScore'], q=10, duplicates='drop')
data['Age_Bucket'] = pd.qcut(data['Age'], q=10, duplicates='drop')
data['Salary_Bucket'] = pd.qcut(data['EstimatedSalary'], q=10, duplicates='drop')
data['Balance_Bucket'] = pd.qcut(data['Balance'], q=10, duplicates='drop')
```

We then visualize the churn rates across different categories:

```python
def plot_event_rate(data, event_label=None, feat_label=None):
    ...
for var in cat_variables:
    plot_event_rate(data, event_label='Exited', feat_label=var)
```

### Feature Selection:

**Information Value (IV)** is calculated for categorical variables.

```python
def get_IV(data, feature=None, target=None):
    ...
iv_df = pd.DataFrame(iv_dict).sort_values(by='IV', ascending=False)
```

ANOVA F-test scores are calculated for numerical variables:

```python
f_scores = pd.DataFrame(f_classif(data[num_variables], data['Exited']), columns=num_variables)
```

### Model Building and Evaluation:

The dataset is split into training and test sets. Three models are trained: a basic logistic regression, a basic random forest, and a tuned random forest.

```python
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, stratify=Y, random_state=1)
...

clf_lr = LogisticRegression(random_state=7).fit(X_train, y_train)
...
clf_rf = RandomForestClassifier(random_state=7).fit(X_train, y_train)
...
clf_rf = RandomForestClassifier(n_estimators=1000, ...).fit(X_train, y_train)
...
```

### Model Monitoring:

The performance of the model is monitored using an ROC curve.

```python
import sklearn.metrics as metrics
...
plt.title('Receiver Operating Characteristic')
...
plt.show()
```



