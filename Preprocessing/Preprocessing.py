<<<<<<< HEAD
#Importing the required packages
import sys

sys.path.insert(0, 'C:/Users/Admin/Desktop/Project/Processing')

import Processing


#Creating object of the required modules
p=Processing.EDA


#Performing EDA on the dataset
def processing_call():
    
    p.intro()
    
    X,Y,col=p.data_input()
    
    X,col=p.filling_values(X,col)
    
    X,col=p.converting_columns(X,col)
    
    X,col=p.outlier_detection(X,col)
    
    X,col=p.correlation_analysis(X,col)
    
    X21,Y21,X22,Y22=p.data_balancing(X,Y)
    
    X21_train,X21_test,Y21_train,Y21_test,X22_train,X22_test,Y22_train,Y22_test=p.splitting_data(X21,Y21,X22,Y22)

    return X21_train,X21_test,Y21_train,Y21_test,X22_train,X22_test,Y22_train,Y22_test
# import sys

# print(sys.path)
=======
#Importing the required packages
import sys

sys.path.insert(0, 'C:/Users/Admin/Desktop/Project/Processing')

import Processing


#Creating object of the required modules
p=Processing.EDA


#Performing EDA on the dataset
def processing_call():
    
    p.intro()
    
    X,Y,col=p.data_input()
    
    X,col=p.filling_values(X,col)
    
    X,col=p.converting_columns(X,col)
    
    X,col=p.outlier_detection(X,col)
    
    X,col=p.correlation_analysis(X,col)
    
    X21,Y21,X22,Y22=p.data_balancing(X,Y)
    
    X21_train,X21_test,Y21_train,Y21_test,X22_train,X22_test,Y22_train,Y22_test=p.splitting_data(X21,Y21,X22,Y22)

    return X21_train,X21_test,Y21_train,Y21_test,X22_train,X22_test,Y22_train,Y22_test
# import sys

# print(sys.path)
>>>>>>> 36faa9a68a31c5013a3921f15370a6d87c3edd13
