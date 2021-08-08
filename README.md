# GIT-Research
GIT Research


# Test-Score-Predictions
Predicting post-test scores based off students' pretest scores, as well as additional independent features

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statistics import mean
from statistics import stdev
from sklearn.metrics import r2_score 
from google.colab import drive
drive.mount('/content/drive')

#load CSV
df=pd.read_csv('/content/drive/My Drive/test_scores.csv')
x_input=df.copy()
x_input.pop('student_id')

y_vals=x_input.pop('posttest')
x_input.head()

strClasses=['classroom']
for i in strClasses:
  x_input.pop(i)

x_input.head()

#one hot encoding
from sklearn.preprocessing import OneHotEncoder

strClasses = ['school_setting']
for i in strClasses:
  enc = OneHotEncoder(handle_unknown='ignore')
  input_data = x_input[i].values.reshape(-1,1)

  #fit data
  enc.fit(input_data)
  output_array = enc.transform(input_data).toarray()
  for j in range(output_array.shape[1]):
    x_input[i+'_oneHot_'+str(j)] = output_array[:,j]

  x_input.pop(i)

x_input.head()

#remove all columns except one-hot for school type & pretest scores

# not school type instead using school_setting for screenshot
strClasses_remove=['school','school_type','teaching_method','n_student','gender','lunch','pretest']

for i in strClasses_remove:  #removing the extraneous columns
  x_input.pop(i)


#x_input.head()
print(x_input.loc[39:42]) #print specific rows from the dataframe

print(x_input.loc[39:39],'\n')
print(x_input.loc[41:41],'\n')
print(x_input.loc[460:460])

setting=df['school_setting']
print(setting.loc[39:39],'\n')
print(setting.loc[41:41],'\n')
print(setting.loc[460:460])

# based only off of pretest

x_pretest=np.array(df['pretest'])
x_pretest=x_pretest.reshape(-1,1)
reg_1factor=LinearRegression().fit(x_pretest,y_vals)
y_predictions_1factor=reg_1factor.predict(x_pretest)

#R^2 value (only pretest)
print(r2_score(y_vals,y_predictions_1factor))
print(reg_1factor.score(x_pretest,y_vals))

#plot predictions only off pretest

plt.scatter(y_predictions_1factor,y_vals, color='teal')
plt.title('predicted post-test based only on pretest',fontsize=25) #plot title

plt.xlabel('predicted post-test',fontsize=20)  # x-axis name
plt.ylabel('real post-test',fontsize=20)  # y-axis name
#plt.annotate(("r^2 = {:.4f}".format(r2_score(y_vals, y_predictions_1factor))), (12, 90))
plt.show()

#prediction on both school type and pretest
plt.plot(x_pretest, reg_2factors.coef_[0] * x_pretest + reg_2factors.intercept_, color='magenta'); # plot prediction line (mx + b)
plt.scatter(x_pretest, y_vals, s=15,color='lightblue')
plt.title('predicted post-test based on \nboth pretest and school type',fontsize=20) #plot title

plt.xlabel('pretest scores',fontsize=15)  # x-axis name
plt.ylabel('predicted post-test scores',fontsize=15)  # y-axis name
plt.annotate(("r^2 = {:.3f}".format(r2_score(y_vals, y_predictions))), (25, 100))


#prediction based only on pretest
plt.plot(x_pretest, reg_1factor.coef_[0] * x_pretest + reg_1factor.intercept_, color='red') # plot prediction line (mx + b)
plt.scatter(x_pretest, y_vals, s=15,color='lightgreen')
plt.title('predicted post-test \nbased only on pretest',fontsize=20) #plot title

plt.xlabel('pretest scores',fontsize=15)  # x-axis name
plt.ylabel('predicted post-test scores',fontsize=15)  # y-axis name
plt.annotate(("r^2 = {:.3f}".format(r2_score(y_vals, y_predictions_1factor))), (25,100))


#count the school_setting category

sett=df['school_setting']
count_u=0
count_s=0
count_r=0
for i in sett:
  if i=='Urban':
    count_u+=1
  elif i=='Suburban':
    count_s+=1
  elif i=='Rural':
    count_r+=1

print("Urban: ",count_u)
print("Suburban: ",count_s)
print("Rural: ",count_r)
print(count_u+count_s+count_r)



