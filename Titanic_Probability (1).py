#!/usr/bin/env python
# coding: utf-8

# ## Women and Children First: Probability Aboard the Titanic , Logistic Regression
# 

# ###### https://makeschool.org/mediabook/oa/tutorials/titanic-dataset-tutorial-an-intro-to-data-analysis-and-statistics-n40/titanic-mean-median-mode/

# #### https://www.youtube.com/watch?v=eGaImwD8fPQ

# ##### https://www.analyticsvidhya.com/blog/2021/07/titanic-survival-prediction-using-machine-learning/

# In[335]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# In[336]:


df = pd.read_csv("/Users/vineethaalexander/Downloads/titanic/train.csv")


# In[337]:


df

PassengerId: A unique identifier or reference number for each passenger.
Survived: Indicates whether the passenger survived or not (0 for No, 1 for Yes).
Pclass: The passenger's class or ticket class (1st, 2nd, or 3rd class).
Name: The passenger's name.
Sex: The gender of the passenger (Male or Female).
Age: The age of the passenger.
SibSp: The number of siblings or spouses aboard the Titanic.
Parch: The number of parents or children aboard the Titanic.
Ticket: The ticket number.
Fare: The fare paid for the ticket.
Cabin: The cabin number or location on the ship.
Embarked: The port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).
# In[338]:


df[df['Survived']==1].count()


# In[339]:


df[df['Survived']==0].count()


# In[340]:


df.isnull().sum()


# In[341]:


df=df.drop('Cabin',axis=1)
df


# In[342]:


df.isnull().sum()


# #Replacing the missing values in the “Age” column with the mean value

# In[343]:


df['Age'].fillna(df['Age'].mean(),inplace=True)
df


# In[344]:


df.isnull().sum()


# In[345]:


df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
df


# In[346]:


df.isnull().sum()


# In[347]:


df['Embarked'].unique()


# In[348]:


df.replace({'Embarked':{'S':0, 'C':1, 'Q':2 }}, inplace=True)
df


# In[349]:


df.replace({'Sex': {'male':1 , 'female':0}}, inplace = True)
df


# In[350]:


df.replace({'Sex':{'male':1,'female':0}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
df


# In[351]:


df.drop('Ticket', axis=1, inplace=True)
df


# In[352]:


X=df.drop( ['Name','Survived','PassengerId'],axis=1,)
X


# In[353]:


Y=df['Survived']
Y


# In[354]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  


# In[355]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)


# In[356]:


model = LogisticRegression()


# In[357]:


model.fit(X_train, Y_train)


# In[358]:


X_train


# In[359]:


X_train_prediction=model.predict(X_train)
X_train_prediction


# In[360]:


#The Accuracy score is calculated by dividing the number of correct predictions by the total prediction number.
#Accuracy score is used to measure the model performance in terms of measuring the ratio of sum of true positive 
#... and true negatives out of all the predictions made.


# In[361]:


from sklearn.metrics import accuracy_score


# In[362]:


X_train_prediction=model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ',training_data_accuracy)


# In[363]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)


# In[364]:


input_data = (3,0,35,0,0,8.05,0)  # Note that these datas exclude the Survived data, as it is to be determined from the model itself


# In[369]:


input_data = (1,0,35,0,0,21.05,0)  # Note that these datas exclude the Survived data, as it is to be determined from the model itself


# In[370]:


input_data_as_numpy_array = np.asarray(input_data)
input_data_as_numpy_array 


# In[371]:


input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
input_data_reshaped


# In[372]:


prediction = model.predict(input_data_reshaped)
print(prediction)
if prediction[0]==0:
    print("Dead")
if prediction[0]==1:
    print("Alive")


# In[ ]:





# ## Quest : Which gender had a better chance of survival?
# 

# In[373]:


w=df[(df['Sex'] == 0)]
w
w.count()


# In[374]:


m=df[(df['Sex'] == 1)]
m
m.count()


# In[375]:


survived_women=df[(df['Sex'] == 0) & (df['Survived'] == 1)]
survived_women
survived_women.count()


# In[376]:


survived_men=df[(df['Sex'] == 1) & (df['Survived'] == 1)]
survived_men
survived_men.count()


# ##### Ans : Survived female are 233 while Survived men ar 109 in count. Hence, Survival rate is higher for women
# 

# ## Ques: Which social class had a better chance of survival?
# 

# In[428]:


df = pd.read_csv("/Users/vineethaalexander/Downloads/titanic/train.csv")
df


# In[429]:


survived_1class = df[(df['Survived']==1) & (df['Pclass'] == 1)]
survived_1class
survived_1class.count()


# In[430]:


survived_2class = df[(df['Survived']==1) & (df['Pclass'] == 2)]
survived_2class
survived_2class.count()


# In[431]:


survived_3class = df[(df['Survived']==1) & (df['Pclass'] == 3)]
survived_3class
survived_3class.count()


# ### Quest: Just from the two simple calculations: (mean, median, mode), and (histogram), what can you concluded from the data? What sorts of statements can you confidently say about the Titanic and the people aboard?          
# 

# ### Were most people old or young? Was it a ship full of children or the elderly?
# 

# In[ ]:





# In[432]:


df.mean()


# In[433]:


df.mode()


# In[434]:


df.median()


# In[435]:


df['Age'].mean()


# In[436]:


df['Age'].mode()


# In[437]:


df['Age'].median()


# ##### Ans: Hence most ppl were Young of avg age 28-29  yrs

# In[438]:


df['Age'].count()


# #### Children means age<16 

# In[439]:


children = df[df['Age']<16]
children


# In[440]:


children.count()


# In[441]:


old = df[df['Age']>65]
old


# In[442]:


old.count()


# In[443]:


adults = df[(df['Age']>=16) & (df['Age']<=65)]
adults


# In[444]:


adults.count()


# ##### children(Age<16) is 83
# ##### old(Age>65) is 8
# ##### adults(Age>=16 and Age<=65) is 623

# In[445]:


df['Age'].hist()


# In[446]:


df.hist(column='Age')


# In[447]:


df['Fare'].hist()


# In[448]:


df.hist(column='Fare')


# ### Quest: Who was the oldest passenger aboard the ship?

# In[449]:


df['Age'].max()


# In[450]:


df[df['Age']==df['Age'].max()]


# ###### Ans: Barkworth, Mr. Algernon Henry Wilson is the oldest. 80 years old.

# ### Quest: Who was the youngest passenger aboard the ship who did not survive ?

# In[451]:


df[df['Survived']==0]


# In[452]:


Not_survived= df[df['Survived']==0]
Not_survived


# In[453]:


Not_survived['Age'].min()


# In[454]:


Not_survived['Age'].max()


# In[455]:


df[df['Age']== Not_survived['Age'].max()]


# In[456]:


df[df['Age']==Not_survived['Age'].min()]


# ### Quest: Who was the oldest passenger aboard the ship who did not survive ?
# 

# In[457]:


df[df['Age']== Not_survived['Age'].max()]


# ### Quest: How much did the cheapest ticket cost? 
# 

# In[409]:


min_fare=df['Fare'].min()
min_fare


# In[410]:


df[df['Fare']==min_fare]


# ### Quest: What was the range of ticket prices? (Hint max - min)
# 

# In[411]:


df['Fare'].max() - df['Fare'].min()


# In[412]:


df.var()


# In[413]:


df.std()


# #### We can use the NumPy function .var(arr) to find the variance score of an array. The .var(arr) function uses "the average of the squared deviations from the mean"—which in english reads: The average of the square of the absolute difference between each element and the average of the whole dataset.

# #### Variance is about how far away elements are from the average, so let's pull up the averagse of these two features with the .describe() method:

# In[414]:


df.describe()


# In[415]:


plt.hist(df.Age)


# In[416]:


df['Age'].hist()
plt.xlabel('Age')
plt.ylabel('Count')
plt.axvline(df.Age.mean(), color='w', linestyle='dashed', linewidth=2)
plt.title('Ages of Passengers on Titanic')


# In[417]:


df['Age'].hist()
plt.axvline(df.Age.mean(), color='w', linestyle='dashed', linewidth=2)
plt.xlabel('Age')
plt.ylabel('Count')

#standard deviation
plt.axvline(df.Age.mean()+df.Age.std(), color='k', linestyle='dashed', linewidth=2)
plt.axvline(df.Age.mean()-df.Age.std(), color='k', linestyle='dashed', linewidth=2)


# In[418]:


plt.axvline(df['Age'].mean()+df['Age'].std(), color='k', linestyle='dashed', linewidth=2)
plt.axvline(df['Age'].mean()-df['Age'].std(), color='k', linestyle='dashed', linewidth=2)


# In[419]:


df.Age.mean()


# In[420]:


df.Age.std()


# In[421]:


df.Age.mean()-df.Age.std()


# In[422]:


df.Age.mean()+df.Age.std()


# #### Is there a correlation between age and fare price?
# #### Is there a correlation between class and survival? Did rich people survive more than the working people?
# #### Was there a correlation between age and survival? Did they really allow "women and children first"?
# #### What about the passenger's sex and survival?

# In[458]:


df['Age'].corr(df['Fare'])


# In[459]:


df['Pclass'].corr(df['Survived'])


# In[460]:


df['Age'].corr(df['Survived'])


# In[461]:


df['Sex'].str.get_dummies().corrwith(df['Survived'])


# ###### –0.54. A moderate negative (downhill sloping) relationship
# ##### +0.54. A moderate positive (upward sloping) linear relationship
# 

# In[462]:


df.corr() #correlation matrix


# In[463]:


import seaborn as sn
sn.heatmap(df.corr(),annot = True)


# In[464]:


df.drop('PassengerId', axis=1) #By default, the axis argument is set to 0 which refers to rows. 
#You must specify axis=1 to tell pandas to look at the columns.


# In[465]:


genders= {"male":1,"female":0}
for dataset in [df]:
    dataset['Sex']=dataset['Sex'].map(genders)


# In[466]:


df


# ### Women and Children First—Probability Aboard the Titanic
# 

# #### Was it really women and children first on the Titanic?
# 
# #### Women and Children vs. Grown Men
# 
# #### What was the probability of survival for a child? P(Survived=true | Age<16)
# #### What was the probability of survival for a woman? P(Survived= true | Sex=female)
# #### What was the probability of survival for a man? P(Survived= true | Sex=male)
# 
# #### Which was more important to survival aboard the Titanic? Your class, your being a child, or your sex?
# #### Lets generate a similar comparison between first class women and third class women on survival rate?
# 
# 

# In[467]:


women_and_children = df[(df['Age']<16) | (df['Sex']== 0)]
women_and_children.count()


# In[468]:


living_children = df[(df['Age'] < 16) & (df['Survived'] == 1)]
living_children.count()


# In[469]:


living_women = df[(df['Sex'] == 0) & (df['Survived'] == 1)]
living_women.count()


# In[470]:


living_children.hist()


# ###### Probability = Ways / Outcomes
# 

# In[471]:


df.value_counts('Sex')


# In[472]:


w_a_c_survival_rate = women_and_children['Survived'].value_counts(normalize=True) * 100 
#With normalize set to True , returns the relatie frequency by dividing all values by the sum of values.
w_a_c_survival_rate


# In[473]:


adults_men_survived = df[(df['Age']>16) & (df['Sex'] == 1)]
adults_men_survival_rate = adults_men_survived['Survived'].value_counts(normalize=True) * 100
adults_men_survival_rate


# In[474]:


71.751412/17.661692 #Survival rate of w_a_c is 4 times that of adults


# In[475]:


women_and_children['Survived'].value_counts(normalize=False)


# In[476]:


women_and_children


# #### 1. What was the probability of survival for a child? P(Survived=true | Age<16)
# #### 2. What was the probability of survival for a woman? P(Survived= true | Sex=female)
# #### 3. What was the probability of survival for a man? P(Survived= true | Sex=male)

# In[477]:


#child_chance_of_survival= format((df[df['Age']<16].shape[0])/(df['Age'].shape[0]),".0%")
#child_chance_of_survival


# In[478]:


children = df[df['Age'] < 16]
surviving_children = df[(df['Age'] < 16) & (df['Survived'] == 1)]
child_chance_of_survival = format((surviving_children.shape[0] / children.shape[0]),".0%")
child_chance_of_survival


# In[479]:


women_chance_of_survival= format((df[(df['Survived']== 1) & (df['Sex']== 0)].shape[0])/(df[(df['Sex']== 0)].shape[0]),".0%")
women_chance_of_survival


# In[480]:


#format(df[(df['Survived']== 1) & (df['Sex']== 0)].shape[0]),"0%"


# In[481]:


men_chance_of_survival= format(df[((df['Survived']==1) & (df['Sex']==1))].shape[0]/df[(df['Sex']==1)].shape[0],".0%")
men_chance_of_survival


# In[482]:


import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
x_axis = [ "Men","Children", "Women"]
data = [men_chance_of_survival,child_chance_of_survival,women_chance_of_survival]
ax.bar(x_axis, data)
plt.show()


# In[483]:


# Importing library
import matplotlib
  
# Create figure() objects
# This acts as a container
# for the different plots
fig = matplotlib.pyplot.figure()
  
# Creating axis
# add_axes([xmin,ymin,dx,dy])
#axes = fig.add_axes([0.5, 1, 0.5, 1])
axes = fig.add_axes([0.5, 1, 0.5, 1])

  
# Depict illustration
fig.show()


# In[484]:


surviving_men = df[(df['Sex'] == 1) & (df['Age'] > 16) & (df['Survived'] == 1)]
surviving_men.describe()


# In[485]:


dead_men = df[(df['Sex'] == 1) & (df['Age'] > 16) & (df['Survived'] == 0)]
dead_men.describe()


# #### First vs. Third Class Men
# 

# In[ ]:






# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[486]:


df = pd.read_csv("/Users/vineethaalexander/Downloads/titanic/train.csv")


# In[487]:


women = df[(df['Sex'] == 'female') & (df['Age'] > 16)]
women


Child Survival Rate

# In[488]:


children = df[df['Age'] < 16]
children.shape


# In[489]:


children


# In[490]:


living_children = df[(df['Age'] < 16) & (df['Survived'] == 1)]
living_children.hist()


# In[491]:


not_surviving_children = df[(df['Age'] < 16) & (df['Survived'] == 0)]
not_surviving_children.hist()


Women and Children vs. Grown Men

# ###### What is normalize True in pandas?
# ##### With normalize set to True , returns the relative frequency by dividing all values by the sum of values.

# In[492]:


women_and_children = df[(df['Sex'].str.match("female")) | (df['Age'] < 16)]
w_a_c_survival_rate = women_and_children['Survived'].value_counts(normalize=True) * 100
w_a_c_survival_rate


# In[493]:


adult_men = df[(df['Sex'].str.match('male')) & (df['Age'] > 16)]
a_m_survival_rate = adult_men['Survived'].value_counts(normalize=True) * 100
a_m_survival_rate


Women and children had a more than 4x better chance of survival than men.
Conditional Probability and Percentage

Let's ask some simple probability questions:

What was the probability of survival for a child? P(Survived=true | Age<16)
What was the probability of survival for a woman? P(Survived= true | Sex=female)
What was the probability of survival for a man? P(Survived= true | Sex=male)
# In[494]:


children = df[df['Age'] < 16]
surviving_children = df[(df['Age'] < 16) & (df['Survived'] == 1)]
child_chance_of_survival = surviving_children.shape[0] / children.shape[0]
format(child_chance_of_survival, ".0%")


# In[495]:


women = df[(df['Sex'] == 'female') & (df['Age'] > 16)]
surviving_women = df[(df['Sex'] == 'female') & (df['Age'] > 16) & (df['Survived'] == 1)]
women_chance_of_survival = surviving_women.shape[0] / women.shape[0]
format(women_chance_of_survival, ".0%")


# In[505]:


men = df[(df['Sex']=='male') & (df['Age']>16)]
surviving_men = df[(df['Sex']=='male') & (df['Age']>16) & (df['Survived']==1)]
men_chance_of_survival = surviving_men.shape[0]/men.shape[0]
format(men_chance_of_survival, ".0%")


# In[518]:


import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
x_axis = ["Children", "Women", "Men"]
data = [child_chance_of_survival, women_chance_of_survival, men_chance_of_survival]
ax.bar(x_axis, data)
plt.show()

Less than 1 in 5 men on the Titanic survived.

Almost 4 in 5 women on the Titanic survived.The Men Who Lived
# In[497]:


surviving_men = df[(df['Sex'] == "male") & (df['Age'] > 16) & (df['Survived'] == 1)]
surviving_men.describe()


# In[189]:


the_dead = df[df["Survived"] == 0]
the_dead.hist()


Multiple Factors: Class

# In[190]:


surviving_men = df[(df['Sex'] == "male") & (df['Age'] > 16) & (df['Survived'] == 1)]
surviving_men.describe()


# In[191]:


dead_men = df[(df['Sex'] == "male") & (df['Age'] > 16) & (df['Survived'] == 0)]
dead_men.describe()


# In[192]:


third_class_adult_men = df[(df['Sex'] == "male") & (df['Age'] > 16) & (df['Pclass'] == 3)]
thrird_class_adult_men_survival_rate = third_class_adult_men['Survived'].value_counts(normalize=True) * 100
thrird_class_adult_men_survival_rate


# In[193]:


first_class_adult_men = df[(df['Sex'] == "male") & (df['Age'] > 16) & (df['Pclass'] == 1)]
first_class_adult_men_survival_rate = first_class_adult_men['Survived'].value_counts(normalize=True) * 100
first_class_adult_men_survival_rate


Third class men had a 12% chance of survival, and first class men had a 37% chance of survival. From these we can say:
An adult man in first class on the Titanic had 3x better chance of survival than an adult man in third class


# In[194]:


the_dead = df[df["Survived"] == 0]
the_dead.hist()


# In[195]:


import seaborn as sn
sn.barplot(x='Pclass', y='Survived', data=women)


# In[196]:


sn.barplot(x='Pclass', y='Survived', hue="Sex", data=df)


# In[197]:


sn.barplot(x='Sex', y='Survived', hue="Pclass", data=df)


Representing Probabilities with PDFs and CDFs Aboard the Titanic

# In[198]:


import seaborn as sns
# create a list of Age values not including N/A values
ls_age = df['Age'].dropna()
# Now plot the data in this list into a histogram!
sns.distplot(ls_age, hist=True, kde=False, bins=16)


# In[199]:


import seaborn as sns
# Notice only the KDE parameter is different!
# What does kde stand for? https://seaborn.pydata.org/generated/seaborn.distplot.html
sns.distplot(df['Age'].dropna(), hist=True, kde=True, bins=16)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




