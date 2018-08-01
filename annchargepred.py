import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse

#Importing Dataset
df=pd.read_csv('insurance.csv')

#Converting Categorical variable into categorical datatype
df.sex=df.sex.astype('category')
df.smoker=df.smoker.astype('category')
df.children=df.children.astype('category')

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder
labenc=LabelEncoder()
df['sex']=labenc.fit_transform(df['sex'])
df['smoker']=labenc.fit_transform(df['smoker'])
df['region']=labenc.fit_transform(df['region'])

#Encoding with dummies for variable region and age
col=pd.get_dummies(df['region'],prefix='region')
col=col.iloc[:,1:]
df=pd.concat([df,col],axis=1)
df.drop('region',axis=1,inplace=True)

#Removing outliers in charges
x=[]
for i in range(0,len(df)):
    if (df.loc[i,'charges']>48500 and df.loc[i,'smoker']==1):
       x.append(i)
    elif(df.loc[i,'charges']>15500 and df.loc[i,'smoker']==0):
        x.append(i)
df=df.drop(df.index[x]).reset_index(drop=True)

#Feature engineering
x=[]
for i in range(0,len(df)):
    if(df.iloc[i,2]>30):
        x.append(1)
    else:
        x.append(0)
df['obesity']=x

#FeatureScaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df.iloc[:,[0,2]]=sc.fit_transform(df.iloc[:,[0,2]])

sc1=StandardScaler()
df.iloc[:,[5]]=sc1.fit_transform(df.iloc[:,[5]])

#Filtering out y variable and dropping it from the dataframe
y=df['charges']
df=df.drop(['charges'],axis=1)

#Checking for variable eligibility
import statsmodels.formula.api as sm
from sklearn.preprocessing import Imputer

x=df.iloc[:,[0,3,4,8]]
x=Imputer().fit_transform(x)
x_opt=np.append(arr=np.ones((len(df),1)).astype(int),values=x,axis=1)
x=x_opt[:,[0,1,2,3,4]]
regsm=sm.OLS(endog=y,exog=x).fit()
regsm.summary()

#Keeping only relavent variables
x=df.iloc[:,[0,3,4,8]]


#Splitting Data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#ANN
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(30,activation='relu',input_dim=len(x_train.T),kernel_initializer='normal'))
model.add(Dense(30,activation='relu',kernel_initializer='normal'))
model.add(Dense(20,activation='relu',kernel_initializer='normal'))
model.add(Dense(10,activation='relu',kernel_initializer='normal'))
model.add(Dense(1,kernel_initializer='normal'))

model.compile(loss='mse',optimizer='adam')

model.fit(x_train,y_train,epochs=70,batch_size=30)

ypred=model.predict(x_test)
ypred=np.ravel(ypred)

#Inverse Transforming
ypred=sc1.inverse_transform(ypred)
y_test=sc1.inverse_transform(y_test)

ydiff=y_test-ypred

#Checking Data Validity.
rmse=np.sqrt(mse(y_test,ypred))
print('\nRoot Mean Square: ',rmse)