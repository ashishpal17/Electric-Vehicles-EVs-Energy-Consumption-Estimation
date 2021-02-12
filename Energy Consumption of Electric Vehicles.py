#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pprint


# In[2]:


df = pd.read_csv("C:/Users/ashis/Desktop/Dataset/VED.csv")


# In[3]:


df


# In[4]:


print("="*50)
print("Input Data Information","\n")
print(df.info())


# In[64]:


plt.figure(figsize=(18,9))
plt.plot( df.HV_Battery_SOC,  df.HV_Battery_Current)
plt.xlabel("HV_Battery_SOC")
plt.ylabel("HV_Battery_Current")
plt.show()


# In[70]:


from matplotlib import style
fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0))
style.use('ggplot')
sns.lineplot(x=df["HV_Battery_SOC"], y=df["HV_Battery_Current"], data=df)
sns.set(rc={'figure.figsize':(10,8)})
plt.xlabel("HV_Battery_SOC in %")
plt.ylabel("HV_Battery_Current")
plt.grid(True)
plt.legend()
for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(90)
plt.title("Energy Consumption")


# In[7]:


from matplotlib import style

fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0))

style.use('ggplot')

sns.lineplot(x=df["HV_Battery_SOC"], y=df["VehId"], data=df)
sns.set(rc={'figure.figsize':(15,6)})

plt.title("Energy consumptionnin")
plt.xlabel("HV_Battery_SOC in %")
plt.ylabel("Vehicle Id")
plt.grid(True)
plt.legend()

for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(90)


plt.title("Energy Consumption")


# In[8]:


sns.distplot(df["HV_Battery_SOC"])
plt.title("Energy Distribution for State of Charge")


# In[9]:


sns.distplot(df["HV_Battery_Current"])
plt.title("Energy Distribution for Current Battery State")


# In[10]:


# Energy with Respect to Time
fig = plt.figure()
ax1= fig.add_subplot(111)
sns.lineplot(x=df["HV_Battery_Voltage"],y=df["HV_Battery_SOC"], data=df)
plt.title("Battery Voltage of Car vs State of Charge of Battery Voltage")
plt.xlabel("HV_Battery_Voltage")
plt.grid(True, alpha=1)
plt.legend()

for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(90)


# In[11]:


print("Dataset ",df.shape )


# In[12]:


TestData = df.tail(100)

Training_Set = df.iloc[:,0:1]

Training_Set = Training_Set[:-60]


# In[13]:


print("Training Set Shape ", Training_Set.shape)
print("Test Set Shape ", TestData.shape)


# In[14]:


from sklearn.preprocessing import MinMaxScaler


# In[15]:



sc = MinMaxScaler(feature_range=(0, 1))
Train = sc.fit_transform(Training_Set)


# In[16]:


Training_Set = Training_Set.values


# In[17]:


Training_Set


# In[18]:


Train


# In[19]:


X_Train = []
Y_Train = []

# Range should be fromm 60 Values to END 
for i in range(60, Train.shape[0]):
    
    # X_Train 0-59 
    X_Train.append(Train[i-60:i])
    
    # Y Would be 60 th Value based on past 60 Values 
    Y_Train.append(Train[i])

# Convert into Numpy Array
X_Train = np.array(X_Train)
Y_Train = np.array(Y_Train)

print(X_Train.shape)
print(Y_Train.shape)


# In[20]:


# Shape should be Number of [Datapoints , Steps , 1 )
# we convert into 3-d Vector or #rd Dimesnsion
X_Train = np.reshape(X_Train, newshape=(X_Train.shape[0], X_Train.shape[1], 1))
X_Train.shape


# In[21]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Dropout


# In[23]:


regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_Train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[24]:


regressor.fit(X_Train, Y_Train, epochs = 50, batch_size = 32)


# In[25]:


TestData.head(2)


# In[26]:


TestData.shape


# In[27]:


df.shape


# In[28]:


Df_Total = pd.concat((df[["HV_Battery_SOC"]], TestData[["HV_Battery_SOC"]]), axis=0)


# In[29]:


Df_Total.shape


# In[30]:


inputs = Df_Total[len(Df_Total) - len(TestData) - 60:].values
inputs.shape


# In[44]:


inputs = Df_Total[len(Df_Total) - len(TestData) - 60:].values

# We need to Reshape
inputs = inputs.reshape(-1,1)

# Normalize the Dataset
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 160):
    X_test.append(inputs[i-60:i])
    
# Convert into Numpy Array
X_test = np.array(X_test)

# Reshape before Passing to Network
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Pass to Model 
predicted_Soc_ = regressor.predict(X_test)

# Do inverse Transformation to get Values 
predicted_Soc_ = sc.inverse_transform(predicted_Soc_)


# In[58]:


True_Soc = TestData["HV_Battery_SOC"].to_list()
True_Soc  = predicted_Soc_ 
dates = TestData.index.to_list()


# In[71]:


Machine_Df = pd.DataFrame(data={
    "TrueSoc": True_Soc,
    "PredictedSoc":[x[0] for x in Predicted_Soc]
})


# In[72]:


# Future Predicted
Machine_Df


# In[61]:


True_Soc


# In[62]:


Predicted_Soc


# In[63]:


fig = plt.figure()

ax1= fig.add_subplot(111)

x = dates
y = True_Soc

y1 = Predicted_Soc

plt.plot(x,y, color="green")
plt.plot(x,y1, color="red")
# beautify the x-labels
plt.gcf().autofmt_xdate()
plt.xlabel('Dates')
plt.ylabel("HV_Battery_SOC")
plt.title("Machine Learned the Pattern Predicting Future Values ")
plt.legend()


# In[ ]:




