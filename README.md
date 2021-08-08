# GIT-Research
GIT Research


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, model_selection, svm
import statsmodels.api as sm
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from statistics import mean
from statistics import stdev
from sklearn.metrics import r2_score 
from google.colab import drive
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
drive.mount('/content/drive')

#load CSV file
df=pd.read_csv('/content/drive/My Drive/Correct_Dataset.csv')
x_input=df.copy()
x_input_chol=df.copy()
y_vals=df['Target']

df.head()
#this shows basic stats of the dataframe
#df.describe()

# convert 1-4 targets into just 1's
target_binarized_chol=[]
for i in y_vals:
  if i==0:
    target_binarized_chol.append(0)
  else:
    target_binarized_chol.append(1)



target_binarized_chol=np.array(target_binarized_chol)
target_binarized_chol=target_binarized_chol.reshape(-1,1)
#print(target_binarized_chol)

#histogram of cholesterol values with (green) & without (red) heart disease
# x-axis cholesterol
# do it here^^
#Hypothesis: mean chol values in disease vs no-disease should be different

target_val=df['Target']
zeros_list=[]
ones_list=[]
index=0
for i in  target_val:
  if i==0:
    zeros_list.append(index)
  else:
    ones_list.append(index)
  index +=1;

# make dataframe of only target value 0
zeros_df=df.drop(labels=ones_list,axis=0)

# make dataframe of only target value 1,2,3
ones_df=df.drop(labels=zeros_list,axis=0)


# plot the histogram(s)
sns.set_theme(); np.random.seed(0)
ax = sns.histplot(zeros_df['Colestrol'],color='green')
ax = sns.histplot(ones_df['Colestrol'],color='red')

ax.set(title='Cholesterol to Heart Disease\n\nMean Cholesterol (0) = 242.640244\nMean Cholesterol (1)= 251.474820')
ax.set(xlabel="\nCholesterol values with (red) & without (green) heart disease", ylabel = "Count")

# Average Cholesterol is slightly higher for folks with heart disease
# Hypothesis is true.



# based only off of cholesterol

x_chol=np.array(df['Colestrol'])
x_chol=x_chol.reshape(-1,1)

#fit model (only Cholesterol)
reg_1chol=LogisticRegression().fit(x_chol,target_binarized_chol)
y_predictions_1factor=reg_1chol.predict(x_chol)

#R^2 value (only cholesterol)
rsquared_chol=reg_1chol.score(x_chol,target_binarized_chol)
print(reg_1chol.score(x_chol,target_binarized_chol))

#plot target predictions only off cholesterol (0 or 1)

# y (predicted target) vs. x (cholesterol)
plt.scatter(x_chol,y_predictions_1factor, s=14,color='lightgreen')

#plot true cholesterol values
#plt.scatter(x_chol,y_vals, s=14,color='red')

plt.title('predicted target\nbased only on Cholesterol',fontsize=25) #plot title

#sort x_chol first
# generate 50 samples?
x_sortedSample=np.linspace(0,500,num=50)
x_vals_sorted=x_sortedSample.reshape(-1,1)


#plot curve of best fit
plt.plot(x_vals_sorted,reg_1chol.predict_proba(x_vals_sorted)[:,1],color='blue',linewidth=1)
#sort x_chol

#or make my own list cholesterol range (5,10,15,20...)


plt.xlabel('\nCholesterol',fontsize=20)  # x-axis name
plt.ylabel('\nPredicted Target',fontsize=20)  # y-axis name

# put R^2 on the plot
#plt.annotate(("r^2 = {:.4f}".format(r2_score(y_vals, y_predictions_1factor))), (12, 90))

plt.show()



# calculate these 3: accuracy, matthew's coefficient, AUROC

# accuracy
targets=['0=No Disease', '1=Disease']
print(classification_report(target_binarized_chol,y_predictions_1factor,target_names=targets))
# ^^ macro avg? weighted avg?


# matthew's correlation coefficient
print("Matthew's coefficient (only cholesterol) is: ", matthews_corrcoef(target_binarized_chol,y_predictions_1factor))

# AUROC
#print("\nAUROC (true vs. predicted) is: ", roc_auc_score(target_binarized_chol,y_predictions_1factor)) << incorrect

# is it true y vs. confidence score?
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html 
# it say: y_true vs. y_score
print("\nAUROC (true vs. probability is) : ",roc_auc_score(target_binarized_chol,reg_1chol.predict_proba(x_chol)[:,1]))
#same as decision_function
print("\nAUROC (true vs. confidence score) is: ", roc_auc_score(target_binarized_chol,reg_1chol.decision_function(x_chol)))

# accuracy & f1-score manually:

#accuracy
print("accuracy: ",accuracy_score(target_binarized_chol,y_predictions_1factor))

#f1 scores
print("f1 score (macro): ",f1_score(target_binarized_chol,y_predictions_1factor,average='macro'))
print("f1 score (weighted): ",f1_score(target_binarized_chol,y_predictions_1factor,average='weighted'))

# generate the receiver-operating curve for cholesterol model
metrics.plot_roc_curve(reg_1chol, x_chol, target_binarized_chol,color='green')  
plt.title('ROC Curve \nfor Cholesterol Model',fontsize=20)
plt.xlabel('\nFalse Positive Rate\n(Positive label=1)',fontsize=13)  
plt.ylabel('\nTrue Positive Rate\n(Positive label=1)',fontsize=13)
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='red', label='Random')
plt.show()

# manually calculate sensitivity & specificity

true_y_list=[] #hold the true 0 or 1
pred_y_list=[] #hold the predicted 0 or 1

true_positives=0
false_positives=0
false_negatives=0
true_negatives=0

y_predictions_1factor=y_predictions_1factor.reshape(-1,1)


for i in y_vals:
  if i==0:
    true_y_list.append(0)
  else:
    true_y_list.append(1)

for i in y_predictions_1factor:
  if i==0:
    pred_y_list.append(0)
  else:
    pred_y_list.append(1)

for index in range(len(pred_y_list)):
  if true_y_list[index]==1 and pred_y_list[index]==1:
    # true positive (both true and predicted are 1)
    true_positives +=1
  elif true_y_list[index]==0 and pred_y_list[index]==1:
    # false positive (true is 0 but pred is 1)
    false_positives +=1
  elif true_y_list[index]==1 and pred_y_list[index]==0:
    # false negative (true is 1 but pred is 0)
    false_negatives +=1
  elif true_y_list[index]==0 and pred_y_list[index]==0:
    # true negative (both true and pred are 0)
    true_negatives +=1

print("total values:",len(true_y_list),'\n')
print('true positives:',true_positives)
print('false positives:',false_positives)
print('false negatives:',false_negatives)
print('true negatives:',true_negatives)

## MULTI-variable model starts here
## indicate which input cols can use one hot encoding
input_cols_oneHotEncoding = []
input_cols_others = []
x_post=pd.DataFrame()

## Data analysis to see which inputs can use one hot encoding
for i in df:
  if(i == 'Target'):
    pass;
  elif(len(df[i].unique()) < 10): # category has less than 10 distinct values
    print('Unique values in {} are: {}'.format(i, df[i].unique()))

    # append the column name
    input_cols_oneHotEncoding.append(i)
  else: 
    # append the column name
    input_cols_others.append(i)
    


## indicates string cols ##
strClasses = input_cols_oneHotEncoding

for i in strClasses:
  # one hot encoding
  enc = OneHotEncoder(handle_unknown='ignore')

  # get Column
  input_data = df[i].values.reshape(-1,1)

  # Fit
  enc.fit(input_data)
  output_array = enc.transform(input_data).toarray()
  
  # this will only work with 2 values #
  # X_postprocessing[i+'_one_hot_encoder'], X_postprocessing[i+'one_hot_encoder'] = output_array.T 

  # for more than 2 values
  for j in range(output_array.shape[1]):
    x_post[i+'_one_hot_encoder_'+str(j)] = output_array[:,j]

x_post.head()


## Add the remaning columns (non one-hot)
strClasses1 = input_cols_others
for i in strClasses1:

  ## get Column ##
  x_post[i] = df[i].copy()

x_post.head() # <-- final input dataset

# Logistic Regression for all variables model
clf_allVars=LogisticRegression().fit(x_post,target_binarized_chol)
y_predictions_allfactor=clf_allVars.predict(x_post)


# matthew's correlation coefficient (all variables)
print("Matthew's coefficient (all vars) is: ", matthews_corrcoef(target_binarized_chol,y_predictions_allfactor))


print("\nAUROC (true vs. probability is) : ",roc_auc_score(target_binarized_chol,clf_allVars.predict_proba(x_post)[:,1]))
#same as decision_function
#print("\nAUROC (true vs. confidence score) is: ", roc_auc_score(target_binarized_chol,clf_allVars.decision_function(x_post)))

# accuracy & f1-score manually:

#accuracy
print("accuracy: ",accuracy_score(target_binarized_chol,y_predictions_allfactor))

#f1 scores
print("f1 score (macro): ",f1_score(target_binarized_chol,y_predictions_allfactor,average='macro'))
print("f1 score (weighted): ",f1_score(target_binarized_chol,y_predictions_allfactor,average='weighted'))

# generate the receiver-operating curve for multi-var model
metrics.plot_roc_curve(clf_allVars, x_post, target_binarized_chol,color='green')  
plt.title('ROC Curve \nfor Multi-Variable Model',fontsize=20)
plt.xlabel('\nFalse Positive Rate\n(Positive label=1)',fontsize=13)  
plt.ylabel('\nTrue Positive Rate\n(Positive label=1)',fontsize=13)
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='red', label='Random')
plt.show()


# manually calculate sensitivity & specificity (all-vars model)

true_y_list=[] #hold the true 0 or 1
pred_y_list=[] #hold the predicted 0 or 1

true_positives=0
false_positives=0
false_negatives=0
true_negatives=0

y_predictions_allfactor=y_predictions_allfactor.reshape(-1,1)


for i in y_vals:
  if i==0:
    true_y_list.append(0)
  else:
    true_y_list.append(1)

for i in y_predictions_allfactor:
  if i==0:
    pred_y_list.append(0)
  else:
    pred_y_list.append(1)

for index in range(len(pred_y_list)):
  if true_y_list[index]==1 and pred_y_list[index]==1:
    # true positive (both true and predicted are 1)
    true_positives +=1
  elif true_y_list[index]==0 and pred_y_list[index]==1:
    # false positive (true is 0 but pred is 1)
    false_positives +=1
  elif true_y_list[index]==1 and pred_y_list[index]==0:
    # false negative (true is 1 but pred is 0)
    false_negatives +=1
  elif true_y_list[index]==0 and pred_y_list[index]==0:
    # true negative (both true and pred are 0)
    true_negatives +=1

print("total values:",len(true_y_list),'\n')
print('true positives:',true_positives)
print('false positives:',false_positives)
print('false negatives:',false_negatives)
print('true negatives:',true_negatives)


