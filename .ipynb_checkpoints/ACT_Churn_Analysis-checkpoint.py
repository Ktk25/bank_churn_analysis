import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('C:/Users/karti/Data Science/ACT_Churn_Analysis/Bank Customer Historic Data.csv')
# data.head()
# data.info()

#-----------------------------------------EDA----------------------------------#########

num_variables = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
cat_variables = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']

# data[num_variables].describe()
# data[cat_variables].describe()
data['Credit_Score_Bucket'] = pd.qcut(data['CreditScore'], q=10, duplicates='drop')
data['Age_Bucket'] = pd.qcut(data['Age'], q=10, duplicates='drop')
data['Salary_Bucket'] = pd.qcut(data['EstimatedSalary'], q=10, duplicates='drop')
data['Balance_Bucket'] = pd.qcut(data['Balance'], q=10, duplicates='drop')

cat_variables.extend(['Credit_Score_Bucket', 'Age_Bucket', 'Salary_Bucket','Balance_Bucket','Tenure', 'NumOfProducts'])

def plot_event_rate(data, event_label=None, feat_label=None):
    cross = pd.crosstab(data[feat_label], data[event_label])
    cross['Event_Rate'] = round(cross[1]/(cross[0] + cross[1])*100, 2)
    cross['Cat'] = cross.index
    cross = cross[['Event_Rate', 'Cat']]
    return cross.plot.bar('Cat', 'Event_Rate', title=feat_label+' Churn Rates')

plot_lists = []
for var in cat_variables:
    plot_lists.append(plot_event_rate(data, event_label='Exited', feat_label=var))
    
    
#-----------------------------------------Feature Selection-------------------------------#########

def get_IV(data, feature=None, target=None):
    df_woe_iv = (pd.crosstab(data[feature],data[target],
                          normalize='columns')
                 .assign(woe=lambda dfx: np.log(dfx[1] / dfx[0]))
                 .assign(iv=lambda dfx: np.sum(dfx['woe']*(dfx[1]-dfx[0]))))
    
    return df_woe_iv['iv'][0]

iv_dict = {'features': [], 'IV':[]}
for var in cat_variables:
    try:
        iv_dict['IV'].append(get_IV(data, feature=var, target='Exited'))
        iv_dict['features'].append(var)
    except:
        print(var)
        pass
# iv = get_IV(data, feature='Balance_Bucket', target='Exited')
iv_df = pd.DataFrame(iv_dict).sort_values(by='IV', ascending=False)
# iv_df

f_scores = pd.DataFrame(f_classif(data[num_variables], data['Exited']), columns=num_variables)
# f_scores

final_features = ['Age', 'Geography', 'IsActiveMember', 'Balance', 'Gender', 'CreditScore', 'EstimatedSalary']


#-----------------------------------------Model Building and Evalutaiton-------------------------######### 

X = data[final_features]
X = pd.get_dummies(X)
Y = data['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, stratify=Y,random_state=1)


clf_lr = LogisticRegression(random_state=7).fit(X_train, y_train)
y_pred_train = clf_lr.predict_proba(X_train)[:, 1]
y_pred_test  = clf_lr.predict_proba(X_test)[:, 1]

train_auc_lr = roc_auc_score(y_train, y_pred_train)
test_auc_lr  = roc_auc_score(y_test, y_pred_test)
print("TRAIN", train_auc_lr, "TEST", test_auc_lr)

clf_rf = RandomForestClassifier(random_state=7).fit(X_train, y_train)
y_pred_train = clf_rf.predict_proba(X_train)[:, 1]
y_pred_test  = clf_rf.predict_proba(X_test)[:, 1]

train_auc_lr = roc_auc_score(y_train, y_pred_train)
test_auc_lr  = roc_auc_score(y_test, y_pred_test)
print("TRAIN", train_auc_lr, "TEST", test_auc_lr)

clf_rf = RandomForestClassifier(n_estimators=1000,min_samples_split=10,
 min_samples_leaf=2, max_features='sqrt',max_depth = 10, random_state=7).fit(X_train, y_train)
y_pred_train = clf_rf.predict_proba(X_train)[:, 1]
y_pred_test  = clf_rf.predict_proba(X_test)[:, 1]

train_auc_lr = roc_auc_score(y_train, y_pred_train)
test_auc_lr  = roc_auc_score(y_test, y_pred_test)
print("TRAIN", train_auc_lr, "TEST", test_auc_lr)


import sklearn.metrics as metrics
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_test)
roc_auc = metrics.auc(fpr, tpr)


#-----------------------------------------Model Monitoring-------------------------######### 

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()