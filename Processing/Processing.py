#Importing the required packages
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from statistics import mode

from sklearn import preprocessing

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss

from sklearn.model_selection import train_test_split


#Creating the EDA class
class EDA:

    
    #Introduction
    def intro():
        
        print("EDA:")

    
    #Taking the dataframe as input
    def data_input():
        
        print("\n")
        
        name=input("Enter the name of the dataset with the file extension")
        
        df=pd.read_csv(name)
        
        
        print("\n")

        print("\n")

        print("Basic information about the dataset:")

        
        print("\n")
        
        print("\n")
        
        print("Head:")
        
        print("\n")
        
        df.head()
        
        
        print("\n")
        
        print("\n")
        
        print("Info:")
        
        print("\n")
        
        df.info()
        
        
        print("\n")
        
        print("\n")
        
        print("Description:")
        
        print("\n")
        
        df.describe(include="object")
        
        
        print("\n")
        
        print("\n")
        
        print("Shape:")
        
        print("\n")

        df.shape
        
        
        X=df.copy()
        
        Y=X['binaryClass'].copy()
        
        col=X.columns
        
        
        return X,Y,col

    
    
    #Filling Missing Values
    def filling_values(X,col):

        print("\n")

        print("\n")

        print("Filling in Missing Values")
        
        
        print("\n")

        print("\n")
        
        print("Count of each value in a column:")
        
        print("\n")
        
        for x in range(len(col)):
            print(X[col[x]].value_counts())
                
        
        print("\n")
        
        print("\n")
        
        c=X['age'].loc[X.index[0]]
        
        print("Datatype of values stored in the age column:")
        
        print("\n")
        
        type(c)

        
        X=X.replace('?',np.nan)
        
        print("\n")
        
        print("\n")
        
        print("Count of each value in a column after replcing '?' with nan:") 
        
        print("\n")

        for x in range(len(col)):
            print(X[col[x]].value_counts())
        
        
        print("\n")
        
        print("\n")
        
        print("Number of missing values in each column:")
        
        print("\n")
        
        for x in range(len(col)):
            print(col[x])
            print(X[col[x]].isna().sum())
        
        
        X['age']=X['age'].astype('str').astype('float')
        X['age']=X['age'].replace(np.nan,X['age'].mean())
        
        X['sex']=X['sex'].replace(np.nan,mode(X['sex']))
        
        X['TSH']=pd.to_numeric(X["TSH"])
        X['TSH']=X['TSH'].replace(np.nan,X['TSH'].mean())
        
        X['T3']=pd.to_numeric(X['T3'])
        X['T3']=X['T3'].replace(np.nan,X['T3'].mean())
        
        X['TT4']=pd.to_numeric(X['TT4'])
        X['TT4']=X['TT4'].replace(np.nan,X['TT4'].mean())
        
        X['T4U']=pd.to_numeric(X['T4U'])
        X['T4U']=X['T4U'].replace(np.nan,X['T4U'].mean())
        
        X['FTI']=pd.to_numeric(X['FTI'])
        X['FTI']=X['FTI'].replace(np.nan,X['FTI'].mean())
        
        X=X.drop('TBG measured',axis=1)
        col=col.drop('TBG measured')
        
        X=X.drop('TBG',axis=1)
        col=col.drop('TBG')

        
        print("\n")
        
        print("\n")

        print("Displaying Info of the dataset after filling in missing values:")
        
        print("\n")
        
        X.info()
        
        
        return X,col
        
    
    #Converting Object Columns to Numeric Columns
    def converting_columns(X,col):
        
        print("\n")

        print("\n")

        print("Converting Object Columns to Numeric Columns")
        
        
        print("\n")

        print("\n")
        
        print("Count of each value in a column:")
        
        print("\n")
        
        for x in range(len(col)):
            print(X[col[x]].value_counts())
            
            
        lenc=preprocessing.LabelEncoder()
            
        X['sex']=lenc.fit_transform(X['sex'])
            
        X['on thyroxine']=lenc.fit_transform(X['on thyroxine'])
            
        X['query on thyroxine']=lenc.fit_transform(X['query on thyroxine'])
            
        X['on antithyroid medication']=lenc.fit_transform(X['on antithyroid medication'])
            
        X['sick']=lenc.fit_transform(X['sick'])
            
        X['pregnant']=lenc.fit_transform(X['pregnant'])
            
        X['thyroid surgery']=lenc.fit_transform(X['thyroid surgery'])
            
        X['I131 treatment']=lenc.fit_transform(X['I131 treatment'])
            
        X['query hypothyroid']=lenc.fit_transform(X['query hypothyroid'])
            
        X['query hyperthyroid']=lenc.fit_transform(X['query hyperthyroid'])
            
        X['lithium']=lenc.fit_transform(X['lithium'])
            
        X['goitre']=lenc.fit_transform(X['goitre'])
            
        X['tumor']=lenc.fit_transform(X['tumor'])
            
        X['hypopituitary']=lenc.fit_transform(X['hypopituitary'])
            
        X['psych']=lenc.fit_transform(X['psych'])
            
        X['TSH measured']=lenc.fit_transform(X['TSH measured'])
            
        X['T3 measured']=lenc.fit_transform(X['T3 measured'])
            
        X['TT4 measured']=lenc.fit_transform(X['TT4 measured'])
            
        X['T4U measured']=lenc.fit_transform(X['T4U measured'])
            
        X['FTI measured']=lenc.fit_transform(X['FTI measured'])
            
        X['binaryClass']=lenc.fit_transform(X['binaryClass'])
            
            
        X=X.drop('referral source',axis=1)
            
        col=col.drop('referral source')

            
        print("\n")
        
        print("\n")

        print("Displaying Info of the dataset after converting object columns to numeric columns:")
            
        print("\n")
        
        X.info()
            
            
        return X,col

    
    #Outlier Detection and Removal
    def outlier_detection(X,col):

        print("\n")

        print("\n")
        
        print("Outlier Detection and Removal")
        
        
        print("\n")

        print("\n")
        
        print("Boxplots of the column:")
        
        print("\n")
        
        for x in range(len(col)):
            if len(X[col[x]].unique())>5:
                sns.boxplot(X[col[x]])
                plt.show()


        print("\n")
        
        print("\n")
        
        print("Boxplots after outlier removal:")
        
        print("\n")
        
        for x in range(len(col)):
            if len(X[col[x]].unique())>5:
                u=X[col[x]].mean()+3*X[col[x]].std()
                l=X[col[x]].mean()-3*X[col[x]].std()
                X[col[x]]=np.where(X[col[x]]>u,u,X[col[x]])
                X[col[x]]=np.where(X[col[x]]<l,l,X[col[x]])
                sns.boxplot(X[col[x]])
                plt.show()
        
        
        return X,col
        
    
    #Correlation Analysis
    def correlation_analysis(X,col):

        print("\n")

        print("\n")
        
        print("Correlation Analysis")
        
        
        c=X.corr()
        
        print("\n")

        print("\n")
        
        print("Correlation among the columns:")
        
        print("\n")
        
        c

        
        x=c['binaryClass']
        
        print("\n")

        print("\n")
        
        print("Correlation between each column and the target column:")
        
        print("\n")
        
        print(x)

        
        X=X.drop('age',axis=1)
        col=col.drop('age')
        
        X=X.drop('sex',axis=1)
        col=col.drop('sex')
        
        X=X.drop('on thyroxine',axis=1)
        col=col.drop('on thyroxine')
        
        X=X.drop('query on thyroxine',axis=1)
        col=col.drop('query on thyroxine')
        
        X=X.drop('on antithyroid medication',axis=1)
        col=col.drop('on antithyroid medication')
        
        X=X.drop('sick',axis=1)
        col=col.drop('sick')
        
        X=X.drop('pregnant',axis=1)
        col=col.drop('pregnant')
        
        X=X.drop('thyroid surgery',axis=1)
        col=col.drop('thyroid surgery')
        
        X=X.drop('I131 treatment',axis=1)
        col=col.drop('I131 treatment')
        
        X=X.drop('query hypothyroid',axis=1)
        col=col.drop('query hypothyroid')
        
        X=X.drop('query hyperthyroid',axis=1)
        col=col.drop('query hyperthyroid')
        
        X=X.drop('lithium',axis=1)
        col=col.drop('lithium')
        
        X=X.drop('goitre',axis=1)
        col=col.drop('goitre')
        
        X=X.drop('tumor',axis=1)
        col=col.drop('tumor')
        
        X=X.drop('hypopituitary',axis=1)
        col=col.drop('hypopituitary')
        
        X=X.drop('psych',axis=1)
        col=col.drop('psych')
        
        X=X.drop('TSH measured',axis=1)
        col=col.drop('TSH measured')
        
        X=X.drop('T3 measured',axis=1)
        col=col.drop('T3 measured')
        
        X=X.drop('TT4 measured',axis=1)
        col=col.drop('TT4 measured')
        
        X=X.drop('T4U measured',axis=1)
        col=col.drop('T4U measured')
        
        X=X.drop('T4U',axis=1)
        col=col.drop('T4U')
        
        X=X.drop('FTI measured',axis=1)
        col=col.drop('FTI measured')
        
        X=X.drop('binaryClass',axis=1)
        col=col.drop('binaryClass')
        
        
        print("\n")
        
        print("\n")
        
        print("Info of the dataset after dropping unnecessary columns:")
        
        print("\n")
        
        X.info()
        
        
        print("\n")
        
        print("\n")
        
        print("Description of the dataset after dropping unnecessary:")
        
        print("\n")
        
        X.describe()

        
        return X,col
        
    
    #Data Balancing
    def data_balancing(X,Y):
        
        print("\n")

        print("\n")

        print("Data Balancing")
        
        
        print("\n")

        print("\n")
        
        print("Count of each value in the target column:")
        
        print("\n")
        
        Y.value_counts()
        
        
        sm=SMOTE(random_state=43)
        
        nr=NearMiss()
        
        X21,Y21=sm.fit_resample(X,Y)
        
        X22,Y22=nr.fit_resample(X,Y)


        print("\n")
        
        print("\n")

        print("Count of each value in the target column after applying oversampling on the column:")
        
        print("\n")
        
        print("Y21:",Y21.value_counts())

        
        print("\n")
        
        print("\n")
        
        print("Count of each value in the target column after applying undersampling on the column:")
        
        print("\n")
        
        print("Y22:",Y22.value_counts())
        
        
        return X21,Y21,X22,Y22
        
    
    #Data Normalization
    def normalization(X21,X22):
        
        print("\n")

        print("\n")

        print("Data Normalization")

        
        X21=preprocessing.normalize(X21)
        
        X22=preprocessing.normalize(X22)
        
        
        return X21,X22
        
    
    #Splitting the 2nd Set's Data
    def splitting_data(X21,Y21,X22,Y22):
        
        print("\n")

        print("\n")

        print("Splitting the Data")
        
        
        X21_train,X21_test,Y21_train,Y21_test=train_test_split(X21,Y21,test_size=0.2,random_state=42)
        
        X22_train,X22_test,Y22_train,Y22_test=train_test_split(X22,Y22,test_size=0.2,random_state=42)
        
        
        return X21_train,X21_test,Y21_train,Y21_test,X22_train,X22_test,Y22_train,Y22_test

# import sys

# print(sys.path)