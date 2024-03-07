#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

data=pd.read_csv("D:/projects/records.csv")
print(data)


# In[4]:


data.info()


# In[4]:


missing_val=data.isna().sum()
missing_val


# In[5]:


data.nunique()


# In[6]:


data.describe()


# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
seed = 42  
np.random.seed(seed)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)


it20_data = pd.read_csv('D:/projects/records1.csv')
innings2 = it20_data.loc[it20_data['Innings'] == 2]


innings2 = innings2.drop(innings2[innings2['Balls Remaining'] == 0].index)
innings2.reset_index(drop=True, inplace=True)

print('No of Matches: ', innings2['Match ID'].nunique())
print('Data Frame Length: ', len(innings2.index))

innings2.head()


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


features = ['Runs From Ball', 'Innings Runs', 'Innings Wickets', 'Balls Remaining', 'Target Score', 'Total Batter Runs', 'Total Non Striker Runs', 'Batter Balls Faced', 'Non Striker Balls Faced']


cutoff_date = '2020-01-01'


train_data = innings2[innings2['Date'] < cutoff_date]
test_data = innings2[innings2['Date'] >= cutoff_date]

X_train = train_data[features]
y_train = train_data['Chased Successfully']
X_test = test_data[features]
y_test = test_data['Chased Successfully']

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)
print(X_test)


# In[19]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = innings2
phases = ['Powerplay', 'Middle Overs', 'Final Overs']
for i, phase in enumerate(phases):
    print('Phase of Play: ', phase)
    if i == 0:
        data = df[df['Balls Remaining'] > 84]
    elif i == 1:
        data = df[(df['Balls Remaining'] > 30) & (df['Balls Remaining'] <= 84)]
    else:
        data = df[df['Balls Remaining'] <= 30]


    cutoff_date = '2018-01-01'
    train_data = data[data['Date'] < cutoff_date]
    test_data = data[data['Date'] >= cutoff_date]

    X_train = train_data[features]
    y_train = train_data['Chased Successfully']
    X_test = test_data[features]
    y_test = test_data['Chased Successfully']

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(C=1)
    model.fit(X_train, y_train)

    LR_score = model.score(X_test, y_test)
    print("Accuracy of {} Classifier:".format(phase), LR_score)


# In[14]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

random_state = 42

classifiers = [
    ("Logistic Regression", LogisticRegression(random_state=random_state)),
    ("Random Forest", RandomForestClassifier(random_state=random_state)),
    
    ("Gradient Boosting", GradientBoostingClassifier(random_state=random_state)),
   
]

for name, model in classifiers:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name} Accuracy: {score:.3f}")


# In[15]:


import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('D:/projects/records1.csv') 
venue_counts = data['Venue'].value_counts()
plt.figure(figsize=(10, 6))
plt.bar(venue_counts.index, venue_counts.values)
plt.xlabel('Venue')
plt.ylabel('Number of Matches')
plt.title('Match Venue Distribution')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[17]:


import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('D:/projects/records1.csv')

bowler_wickets = data.groupby('Bowler')['Innings Wickets'].sum().reset_index()

bowler_wickets = bowler_wickets.sort_values(by='Innings Wickets', ascending=False)
plt.figure(figsize=(12, 6))
plt.bar(bowler_wickets['Bowler'], bowler_wickets['Innings Wickets'])
plt.xlabel('Bowler')
plt.ylabel('Total Wickets Taken')
plt.title('Bowler Performance: Total Wickets Taken')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[5]:


data["Runs From Ball"].value_counts().sort_index().plot.line(color=["blue"])


# In[18]:


import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('D:/projects/records1.csv')
bowler_wickets = data.groupby('Bowler')['Innings Wickets'].sum().reset_index()
top_10_bowlers = bowler_wickets.sort_values(by='Innings Wickets', ascending=False).head(10)
plt.figure(figsize=(12, 6))
plt.bar(top_10_bowlers['Bowler'], top_10_bowlers['Innings Wickets'])
plt.xlabel('Bowler')
plt.ylabel('Total Wickets Taken')
plt.title('Top 10 Bowler Performances (Wickets Taken)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('D:/projects/records1.csv') 
x_column = 'Innings Runs'  
y_column = 'Innings Wickets' 
plt.figure(figsize=(10, 6))
plt.scatter(data[x_column], data[y_column], alpha=0.5)
plt.xlabel(x_column)
plt.ylabel(y_column)
plt.title(f'Scatter Plot: {x_column} vs. {y_column}')
plt.grid(True) 
plt.tight_layout()
plt.show()


# In[5]:


import pandas as pd
data = pd.read_csv('D:/projects/records1.csv')
selected_columns = ['Date', 'Innings Runs', 'Innings Wickets']
inning_data = data[data['Innings'] == 1]
inning_data['Date'] = pd.to_datetime(inning_data['Date'])
monthly_data = inning_data.resample('M', on='Date').mean()

plt.figure(figsize=(10, 6))
plt.plot(monthly_data.index, monthly_data['Innings Runs'], label='Innings Runs', marker='o')
plt.plot(monthly_data.index, monthly_data['Innings Wickets'], label='Innings Wickets', marker='x')

plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Innings Runs and Wickets Over Time (Monthly Aggregation)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('D:/projects/records1.csv') 
selected_columns = ['Innings Runs', 'Innings Wickets', 'Target Score', 'Runs to Get',
                    'Balls Remaining', 'Total Batter Runs', 'Total Non Striker Runs',
                    'Batter Balls Faced', 'Non Striker Balls Faced', 'Player Out Runs',
                    'Player Out Balls Faced', 'Bowler Runs Conceded']
subset_data = data[selected_columns]
correlation_matrix = subset_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[29]:


import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('D:/projects/records1.csv')  
column1 = 'Innings Runs' 
column2 = 'Bowler Runs Conceded'  

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(data[column1], bins=20, color='skyblue', edgecolor='black')
plt.title(f'Histogram of {column1}')
plt.xlabel(column1)
plt.ylabel('Frequency')
plt.subplot(1, 2, 2)
plt.hist(data[column2], bins=20, color='lightcoral', edgecolor='black')
plt.title(f'Histogram of {column2}')
plt.xlabel(column2)
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[21]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('D:/projects/records1.csv')  
selected_column = 'Innings Runs'
category_column = 'Bat First' 

plt.figure(figsize=(12, 6)) 
sns.boxplot(x=category_column, y=selected_column, data=data, palette='Set2')
plt.title(f'Box Plot of {selected_column} by {category_column}')
plt.xlabel(category_column)
plt.ylabel(selected_column)
plt.xticks(rotation=45, ha="right") 
plt.tight_layout()

plt.show()


# In[32]:


import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('D:/projects/records1.csv')  
extra_runs_distribution = data.groupby('Extra Type')['Extra Runs'].sum()

plt.figure(figsize=(6, 4))
plt.pie(extra_runs_distribution, labels=extra_runs_distribution.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Extra Runs in Matches')

plt.axis('equal')  # Equal aspect ratio ensures that the pie chart is circular.

plt.show()


# In[15]:


import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('D:/projects/records1.csv')  
chased_successfully = len(data[data['Chased Successfully'] == 1])
not_chased_successfully = len(data[data['Chased Successfully'] == 0])
plt.figure(figsize=(4, 4))
labels = ['Chased Successfully', 'Not Chased Successfully']
sizes = [chased_successfully, not_chased_successfully]
colors = ['lightcoral', 'lightgreen']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Outcome of Matches: Chased Successfully vs. Not Chased Successfully')

plt.axis('equal')

plt.show()


# In[27]:



columns = ['Batter Runs', 'Extra Runs', 'Runs From Ball', 'Ball Rebowled', 'Innings Runs', 'Innings Wickets',
           'Target Score', 'Runs to Get', 'Balls Remaining', 'Total Batter Runs', 'Total Non Striker Runs', 
           'Batter Balls Faced', 'Non Striker Balls Faced', 'Player Out Runs', 'Player Out Balls Faced',
           'Bowler Runs Conceded']
sns.set(style="ticks")
sns.pairplot(data[columns], diag_kind='kde')
plt.show()


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





# In[ ]:





# #    
