#!/usr/bin/env python
# coding: utf-8

# In[55]:


# import necessary library 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
get_ipython().run_line_magic('matplotlib', 'inline')


# In[56]:


#Read the data
df_2019 = pd.read_csv("./world_happiness/2019.csv")
df_2018 = pd.read_csv("./world_happiness/2018.csv")
df_2017 = pd.read_csv("./world_happiness/2017.csv")
df_2016 = pd.read_csv("./world_happiness/2016.csv")
df_2015 = pd.read_csv("./world_happiness/2015.csv")


# In[57]:


# Check how 2019 data looks like
df_2019.info()


# In[58]:


# Add year attribute to data
df_2019['year'] = 2019
df_2019.head(10)


# In[59]:


# Check how 2018 data looks like
df_2018.info()


# In[60]:


# Add year attribute to data
df_2018['year'] = 2018


# In[61]:


df_2018.head(10)


# In[62]:


# Check how 2017 data looks like
df_2017.info()


# In[63]:


# Now we have found that we had to rename certain countries by checking World Happiness Report Website
# Since we will use 2019 as test set, we will use the columns that exist in 2019
# Add a year column
df_2017['year'] = 2017
df_2017.rename(columns={"Country": "Country or region", 
 "Happiness.Rank": "Overall rank",
  'Happiness.Score': 'Score',
 'Health..Life.Expectancy.' : 'Healthy life expectancy', 
  'Freedom':'Freedom to make life choices',
 'Trust..Government.Corruption.' :'Perceptions of corruption',
   'Economy..GDP.per.Capita.' : 'GDP per capita',    
     'Family':'Social support'                   
 },inplace = True)
df_2017.columns


# In[64]:


df_2017.head(10)


# In[65]:


# Check how 2016 data looks like
df_2016.info()


# In[66]:


# Add a year column
df_2016['year'] = 2016
df_2016.rename(columns={
    "Country": "Country or region", 
 "Happiness Rank": "Overall rank",
  'Happiness Score': 'Score',
 'Health (Life Expectancy)' : 'Healthy life expectancy', 
  'Freedom':'Freedom to make life choices',
 'Trust (Government Corruption)' :'Perceptions of corruption',
  'Economy (GDP per Capita)' : 'GDP per capita',
    'Family':'Social support'
 },inplace = True)
df_2016.columns


# In[67]:


df_2016.head(5)


# In[68]:


df_2015.info()


# In[69]:


# Add a year column
df_2015['year'] = 2015
# Rename columns
df_2015.rename(columns={
    "Country": "Country or region", 
 "Happiness Rank": "Overall rank",
  'Happiness Score': 'Score',
 'Health (Life Expectancy)' : 'Healthy life expectancy', 
  'Freedom':'Freedom to make life choices',
 'Trust (Government Corruption)' :'Perceptions of corruption',
  'Economy (GDP per Capita)' : 'GDP per capita',
 'Trust (Government Corruption)' :'Perceptions of corruption',
    'Family':'Social support'
 },inplace = True)
df_2015.columns


# In[70]:


df_2015.head(5)


# In[71]:


dfs = [df_2015, df_2016, df_2017, df_2018]
df = pd.concat(dfs)


# # Q1
# Merge

# In[72]:


# Select attributes and target variable from our merged data of 2015 to 2018
train = pd.DataFrame(df,columns = ['Overall rank', 'Country or region', 'Score', 'GDP per capita',
        'Healthy life expectancy', 'Social support',
       'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption','year'])
# Selected data:
train.sample(8)


# In[73]:


# Select attributes and target variable from our merged data in 2019
test =  pd.DataFrame(df_2019,columns = ['Overall rank', 'Country or region', 'Score', 'GDP per capita',
        'Healthy life expectancy', 'Social support',
       'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption','year'])


# # Q2

# We will take different countries to see the trend

# In[74]:


# Select some example countries or regions prepared for later analysis
d1 = train[train['Country or region']=='Denmark']
d2 = train[train['Country or region']=='Finland']
d3 = train[train['Country or region']=='United States']
d4 = train[train['Country or region']=='China']
d5 = train[train['Country or region']=='Australia']
d6 = train[train['Country or region']=='Canada']
d7 = train[train['Country or region']=='New Zealand']
d8 = train[train['Country or region']=='United Kingdom']
d9 = train[train['Country or region']=='Norway']


# In[75]:


# Plot the trend of our selected examples's Scores 
plt.figure(figsize=(10, 10))
plt.style.use('seaborn')
plt.plot(d1['year'],d1['Score'], label="Vietnam")
plt.plot(d2['year'],d2['Score'], label="Finland")
plt.plot(d3['year'],d3['Score'], label="United States")
plt.plot(d4['year'],d4['Score'], label="China")
plt.plot(d5['year'],d5['Score'], label="Australia")
plt.plot(d6['year'],d6['Score'], label="Canada")
plt.plot(d7['year'],d7['Score'], label="New Zealand")
plt.plot(d8['year'],d8['Score'], label="United Kingdom")
plt.legend(loc=0)
plt.xlabel('year')
plt.ylabel('country or region')


# In[76]:


# Calculate the mean(of all countries or regions) in each year
mean_score = train.groupby("year").mean()["Score"]
mean_score 


# In[77]:


# Plot the mean of Score of all countries or regions from 2015 to 2018
plt.figure(figsize=(6, 6))
plt.xlabel("year")
plt.ylabel("mean happiness score of all countries")
plt.style.use('seaborn')
plt.plot(mean_score, label="mean score")
plt.legend(loc = 0)
plt.xticks([2015, 2016, 2017, 2018])


# In[78]:


d3
# The happiness score seems to decrease for United States


# In[79]:


d6
# The happiness score decreased from 2015 to 2017 and then increased in 2018 for Canada


# In[80]:


d8
# The happiness score decreased from 2015 to 2018 and then incresed in 2018 for United Kingdom


# # Q3

# Stable rankings (They had the least standard deviation in Overall Rankings)

# In[81]:


train.groupby('Country or region').std().nsmallest(10, 'Overall rank')
# These countries have rather stable rannkings


# In[82]:


train.groupby('Country or region').mean().nsmallest(10, 'Overall rank')
# These countries are ranked very high on average from 2015 tto 2018


# Improved countries

# In[83]:


d2
# Finland increased its rank from 6th in 2015 to 1st in 2018


# In[84]:


d9
# Norway increased its rank from 4th in 2015 to 2nd in 2018


# In[85]:


d8
# UK increased its rank from 21st in 2015 to 11st in 2018


# # Q4

# In[86]:


# make a heatmap tp show the correlation between happiness score and other numerical variables such as GDP per capita, life expectancy
plt.figure(figsize=(10, 10))
sns.heatmap(train.corr(),cmap = 'coolwarm',annot = True, square = True) # plot the heatmap
bot, top = plt.ylim() 
bot+=0.5
top-=0.5
plt.ylim(bot, top) 


# From the heatmap above we can see that the happniess score has a strong correlation with GDP per capita, life expectancy, social support, and freedom

# In[87]:


# visulize the relation between GDP per capita and happiness score, we can see there is a strong correlattion
sns.jointplot(x="GDP per capita", y="Score", data=train, kind="reg", scatter_kws={'s':15})


# In[88]:


# visulize the relation between life expectancy and happiness score, we can see there is a strong correlattion
sns.jointplot(x="Healthy life expectancy", y="Score", data=train, kind="reg", scatter_kws={'s':15})


# In[89]:


# visulize the relation between social support and happiness score, we can see there is a strong correlattion
sns.jointplot(x="Social support", y="Score", data=train, kind="reg", scatter_kws={'s':15})


# In[90]:


# visulize the relation between freedom and happiness score, we can see there is a strong correlattion
sns.jointplot(x="Freedom to make life choices", y="Score", data=train, kind="reg", scatter_kws={'s':15})


# ## Q5
# From the visualization we can see that GDP per capita, life expectancy, social support, and freedom play an importanct role in countries' happiness scores. As a result, If I were a president, I would develop the economy (GDP per capita), build a strong health care system & encourage people to do exercise frequently (Life expectancy), encourage people to support the needed people (social support), and create a free environment where people can make life dicision freely in order to make my citizens happier.

# # Modeling and Analysis

# In[91]:


X_train = train[["GDP per capita", "Healthy life expectancy", 'Social support', 'Freedom to make life choices']]
X_train 


# In[92]:


y_train = train["Score"]
y_train


# In[93]:


X_test = df_2019[["GDP per capita", "Healthy life expectancy", 'Social support', 'Freedom to make life choices']]
X_test


# In[94]:


y_test = df_2019["Score"]
y_test


# In[95]:


score_original = df_2019["Score"] #Original Score
rank_original = df_2019["Overall rank"] #Original Rank


# # Model 1: Multiple Linear Regression
# Multiple linear regression is used to estimate the relationship between two or more independent variables and one dependent variable. It uses Least Squares Estimation to estimate the coefficient of each variable. The least squares provides a way of choosing the coefficients effectively by minimizing the sum of the squared errors.

# In[96]:


# Build the model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
regression = lm.fit(X_train, y_train)
score_predict1 = lm.predict(X_test) # predictted score for 2019
regression.score(X_test, y_test)


# In[97]:


fig, ax = plt.subplots()
plt.scatter(y_test, score_predict1.flatten(),s = 10)
plt.xlabel('Original 2019 Score')
plt.ylabel('Predicted 2019 Score')
plt.title('Multiple Linear Regression')
# Adding a blue line which indicates: Original 2019 Score = Predicted 2019 Score
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'b-')
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)


# In[98]:


from sklearn.metrics import mean_squared_error
# Here we compute the root mean squared error to evaluate the multiple linear regression model.
RMSE = np.sqrt(mean_squared_error(score_original, score_predict1)) 
mean = np.mean(score_original)
print(RMSE)
print(mean)
# Evaluating the performance by score
# The RMSE of our predicted score is quite small, so we can say that the multiple linear regression model did well.


# In[99]:


rank_predict1 = pd.Series(score_predict1).rank(ascending = 0) # predicted rank for 2019
RMSE = np.sqrt(mean_squared_error(df_2019["Overall rank"], rank_predict1)) 
print(RMSE)
# Evaluating the performance by rank
# The RMSE of our predicted rank is small, so we can say that the multiple linear regression model did well.


# # Model 2: Epsilon-Support Vector Regression.
# The second model is called support-vector machines. This algorithm deals with both classfication and regression problems. Here we used the regression version of SVM:SVR. SVR allows us to decide how much error is acceptable in our model and finds hyperplanes to fit the data. The algorithm sets up contraints to minimize error: making it less than or equal to a margin or maximum error ϵ (epsilon). In this way, it sets up the hyperplane that maximizes the margin.

# In[100]:


# Build the model
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X = train[['GDP per capita', 'Social support',
       'Healthy life expectancy', 'Freedom to make life choices',]]
y = train['Score']
svr = make_pipeline(StandardScaler(), SVR(C=1, epsilon=0.1, kernel = 'rbf'))
svr.fit(X_train, y_train)


# In[101]:


score_predict2 = svr.predict(X_test) # predicted score for year 2019
svr.score(X_test, y_test)# The best possible value is 1, might be negative


# In[102]:


fig, ax = plt.subplots()
plt.scatter(y_test, score_predict2.flatten(),s = 10)
plt.xlabel('Original 2019 Score')
plt.ylabel('Predicted 2019 Score')
plt.title('Epsilon-Support Vector Regression.')
# Adding a blue line which indicates: Original 2019 Score = Predicted 2019 Score
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'b-')
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)


# In[103]:


RMSE = np.sqrt(mean_squared_error(score_original, score_predict2)) 
mean = np.mean(score_original)
print(RMSE)
print(mean)
# Evaluating the performance by score
# The RMSE of our predicted score is quite small, so we can say that the multiple linear regression model did well.


# In[104]:


rank_predict2 = pd.Series(score_predict2).rank(ascending = 0) # predicted rank for 2019
RMSE = np.sqrt(mean_squared_error(df_2019["Overall rank"], rank_predict2)) 
print(RMSE)
# Evaluating the performance by rank
# The RMSE of our predicted rank is small, so we can say that the multiple linear regression model did well.


# # Model 3: Neural Network
# The third model we used is Neural Network. Multi-Layer Perceptron is one example of feedforward neural networks. It has three layers: Input layer, Hidden layer, Output layer. Each layer has many neurons. Neurons are the most fundamental units of a neural network. A neuron takes in data, processes the information, and produces output. The idea behinds the algorithm is: feed the input, then use some math to minimize the loss in the neural network, finally we get the output at the end.

# In[105]:


# Build the model
from sklearn.neural_network import MLPRegressor
neural_net = MLPRegressor(hidden_layer_sizes=500, max_iter=1000)
net = neural_net.fit(X_train, y_train)
score_predict3 = neural_net.predict(X_test) # predicted score for 2019
net.score(X_test, y_test)# The best possible value is 1, might be negative


# In[106]:


fig, ax = plt.subplots()
plt.scatter(y_test, score_predict3.flatten(),s = 10)
plt.xlabel('Original 2019 Score')
plt.ylabel('Predicted 2019 Score')
plt.title('Neural Network')
# Adding a blue line which indicates: Original 2019 Score = Predicted 2019 Score
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'b-')
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)


# In[107]:


RMSE = np.sqrt(mean_squared_error(score_original, score_predict3)) 
mean = np.mean(score_original)
print(RMSE)
print(mean)
# Evaluating the performance by score
# The RMSE of our predicted score is quite small, so we can say that the multiple linear regression model did well.


# In[108]:


rank_predict3 = pd.Series(score_predict3).rank(ascending = 0) # predicted rank for 2019
RMSE = np.sqrt(mean_squared_error(df_2019["Overall rank"], rank_predict3)) 
print(RMSE)
# Evaluating the performance by rank
# The RMSE of our predicted rank is small, so we can say that the multiple linear regression model did well.


# # Fomula:

# In[109]:


regression.intercept_
# Multiple linear regression intercept


# In[110]:


regression.coef_
# Multiple linear regression coeffcients


# # Score =  2.29937623994713 + GDP per capita * 1.14588287 +Social support * 1.16559883 + Healthy life expectancy * 0.5446503 +Freedom to make life choices * 1.85021572
