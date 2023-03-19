#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install keras


# In[2]:


pip install tensorflow==2.10.1 --ignore-installed


# In[3]:


pip install --upgrade tensorflow --user


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation


# In[2]:


from sklearn.preprocessing import StandardScaler
from sklearn import metrics


# In[4]:


infy = pd.read_csv('C:/Users/jayas/Downloads/INFY.NS.csv')


# In[5]:


tcs= pd.read_csv('C:/Users/jayas/Downloads/TCS.NS.csv')
itc= pd.read_csv('C:/Users/jayas/Downloads/ITC.NS.csv')
cipla= pd.read_csv('C:/Users/jayas/Downloads/CIPLA.NS.csv')
nestle= pd.read_csv('C:/Users/jayas/Downloads/NESTLEIND.NS.csv')


# In[3]:


wipro= pd.read_csv('C:/Users/jayas/Downloads/WIPRO.NS.csv')
ongc= pd.read_csv('C:/Users/jayas/Downloads/ONGC.NS.csv')


# In[4]:


asp= pd.read_csv('C:/Users/jayas/Downloads/ASIANPAINT.NS.csv')
hdfc= pd.read_csv('C:/Users/jayas/Downloads/HDFCBANK.NS (1).csv')


# In[5]:


reliance= pd.read_csv('C:/Users/jayas/Downloads/RELIANCE.NS.csv')


# In[7]:


ongc.head()


# In[8]:


ongc.info()


# In[9]:


ongc.shape


# In[10]:


splitted = ongc['Date'].str.split('-', expand=True)
 
ongc['month'] = splitted[1].astype('int')
ongc['year'] = splitted[0].astype('int')
ongc['day'] = splitted[2].astype('int')
 
ongc.tail()


# In[12]:


plt.figure(figsize=(16,6))
plt.title('Close Price History')
plt.plot(ongc['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price INR', fontsize=18)
plt.show()


# In[14]:


ongc.isnull().sum()


# In[16]:


ongc = ongc.dropna()


# In[45]:


X = ongc[['Open']]
Y = ongc[['Close']]
factor = 0.80
length = X.shape[0]
total_for_train = int(length*factor)
X_train = X[:total_for_train]
Y_train = Y[:total_for_train]
X_test = X[total_for_train:]
Y_test = Y[total_for_train:]


# In[19]:


print("X_train", X_train.shape)
print("y_train", Y_train.shape)
print("X_test", X_test.shape)
print("y_test", Y_test.shape)


# In[20]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, Y_train)


# In[21]:


print("Performance (R^2): ", lr.score(X_train, Y_train))


# In[22]:


def get_mape(y_true, y_pred): 
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[23]:


Y_train_pred = lr.predict(X_train)
Y_test_pred = lr.predict(X_test)


# In[24]:


print("Training R-squared: ",round(metrics.r2_score(Y_train,Y_train_pred),2))
print("Training Explained Variation: ",round(metrics.explained_variance_score(Y_train,Y_train_pred),2))
print('Training MAPE:', round(get_mape(Y_train,Y_train_pred), 2)) 
print('Training Mean Squared Error:', round(metrics.mean_squared_error(Y_train,Y_train_pred), 2)) 
print("Training RMSE: ",round(np.sqrt(metrics.mean_squared_error(Y_train,Y_train_pred)),2))
print("Training MAE: ",round(metrics.mean_absolute_error(Y_train,Y_train_pred),2))

print(' ')


# In[25]:


print("Test R-squared: ",round(metrics.r2_score(Y_test,Y_test_pred),2))
print("Test Explained Variation: ",round(metrics.explained_variance_score(Y_test,Y_test_pred),2))
print('Test MAPE:', round(get_mape(Y_test,Y_test_pred), 2)) 
print('Test Mean Squared Error:', round(metrics.mean_squared_error(Y_test,Y_test_pred), 2)) 
print("Test RMSE: ",round(np.sqrt(metrics.mean_squared_error(Y_test,Y_test_pred)),2))
print("Test MAE: ",round(metrics.mean_absolute_error(Y_test,Y_test_pred),2))


# In[26]:


df_pred = pd.DataFrame(Y_test.values, columns=['Actual'], index=Y_test.index)
df_pred['Predicted'] = Y_test_pred
df_pred = df_pred.reset_index()

df_pred


# In[27]:


Y_test


# In[28]:


df_pred[['Actual', 'Predicted']].plot()


# In[29]:


##LSTM


# In[30]:


ongc.dropna()


# In[34]:


data = ongc.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .80 ))

training_data_len


# In[35]:


dataset


# In[36]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data


# In[37]:


# Create the training data set 
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# In[38]:


x_train


# In[39]:


x_train.shape


# In[40]:


from keras.models import Sequential
from keras.layers import Dense, LSTM

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[41]:


# Create the testing data set 
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
predictions


# In[42]:


y_test


# In[43]:


rmse


# In[44]:


train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price INR', fontsize=18)
# plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


# In[42]:


valid


# In[43]:


print("Test R-squared: ",round(metrics.r2_score(Y_test,predictions),2))
print("Test Explained Variation: ",round(metrics.explained_variance_score(Y_test,predictions),2))
print('Test MAPE:', round(get_mape(Y_test,predictions), 2)) 
print('Test Mean Squared Error:', round(metrics.mean_squared_error(Y_test,predictions), 2)) 
print("Test RMSE: ",round(np.sqrt(np.mean(((predictions - y_test) ** 2)))))
print("Test MAE: ",round(metrics.mean_absolute_error(Y_test,predictions),2))


# In[1]:


#DTR


# In[48]:


from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state = 1)
dtr.fit(X_train,Y_train)
print(dtr.score(X_test,Y_test))


# In[49]:


predicted_dtr=dtr.predict(X_test)


# In[50]:


parameters={"splitter":["best","random"],
            "max_depth" : [1,5],
           "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
           "max_features":["auto","log2","sqrt"], #auto and None work the same
           "max_leaf_nodes":[None,10,20,30,40] }


# In[51]:


from sklearn.model_selection import GridSearchCV
tuning_model=GridSearchCV(dtr,param_grid=parameters,scoring='neg_mean_squared_error',cv=3,verbose=0)
tuning_model.fit(X_train,Y_train)

print(tuning_model.best_params_)
print(tuning_model.best_estimator_.score(X_test,Y_test))


# In[52]:


import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# In[53]:


predicted_dtr = tuning_model.predict(X_test)
predicted_dtr


# In[54]:


df_preds = pd.DataFrame(Y_test.values, columns=['Actual'], index=Y_test.index)
df_preds['predicted'] = predicted_dtr
df_preds = df_preds.reset_index()

df_preds


# In[55]:


df_preds[['Actual', 'predicted']].plot()


# In[56]:


print("Test R-squared: ",round(metrics.r2_score(Y_test,predicted_dtr),2))
print("Test Explained Variation: ",round(metrics.explained_variance_score(Y_test,predicted_dtr),2))
print('Test MAPE:', round(get_mape(Y_test,predicted_dtr), 2)) 
print('Test Mean Squared Error:', round(metrics.mean_squared_error(Y_test,predicted_dtr), 2)) 
print("Test RMSE: ",round(np.sqrt(metrics.mean_squared_error(Y_test,predicted_dtr)),2))
print("Test MAE: ",round(metrics.mean_absolute_error(Y_test,predicted_dtr),2))


# In[ ]:




