#Importing the required packages
import pickle

import numpy as np


class model_call:

    #Introduction
    def intro():
        
        print("\n")
        
        print("\n")
        
        print("Accessing the pickle files:")


    #Accessing the Naive Bayes algorithm with oversampling's pickle files
    def Naive_Bayes1():
        
        filepath='C:/Users/Admin/Desktop/Project/Model/NaiveBayes1.pkl'

        with open(filepath,'rb') as file1:

            clf1=pickle.load(file1)
        
        
        print("\n")
        
        print("The Naive Bayes algorithm with oversampling")

        
        print("\n")
        
        features=np.array([[4.2,2.2,120,125]])
        
        pred1=clf1.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred1)
        
        
        print("\n")
        
        features=np.array([[60,1,45,27]])
        
        pred2=clf1.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred2)

        
        print("\n")
        
        a=float(input("Enter TSH:"))
        
        b=float(input("Enter T3:"))
        
        c=float(input("Enter TT4:"))
        
        d=float(input("Enter FTI:"))
        
        data=np.array([[a,b,c,d]])

        
        print("\n")
        
        pred3=clf1.predict(data)
        
        print("Features:",data)
        
        print("Prediction:",pred3)


    #Accessing the Naive Bayes algoirhtm with undersampling's pickle files
    def Naive_Bayes2():
        
        filepath='C:/Users/Admin/Desktop/Project/Model/NaiveBayes2.pkl'

        with open(filepath,'rb') as file2:

            clf2=pickle.load(file2)
            
        
        print("\n")
        
        print("The Naive Bayes algorithm with undersampling")

        
        print("\n")
        
        features=np.array([[4.2,2.2,120,125]])
        
        pred1=clf2.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred1)
        
        
        print("\n")
        
        features=np.array([[60,1,45,27]])
        
        pred2=clf2.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred2)
        
        
        print("\n")
        
        a=float(input("Enter TSH:"))
        
        b=float(input("Enter T3:"))
            
        c=float(input("Enter TT4:"))
            
        d=float(input("Enter FTI:"))
        
        data=np.array([[a,b,c,d]])
        
        
        print("\n")
        
        pred3=clf2.predict(data)
        
        print("Features:",data)
        
        print("Prediction:",pred3)


    #Accessing the KNN algorihtm with oversampling's pickle files
    def KNN1():
        
        filepath='C:/Users/Admin/Desktop/Project/Model/KNN1.pkl'

        with open(filepath,'rb') as file3:

            clf3=pickle.load(file3)
            

        print("\n")
        
        print("The KNN algorithm with oversampling")
        
        
        print("\n")
        
        features=np.array([[4.2,2.2,120,125]])
        
        pred1=clf3.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred1)
        
        
        print("\n")
        
        features=np.array([[60,1,45,27]])
        
        pred2=clf3.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred2)
        
        
        print("\n")
        
        a=float(input("Enter TSH:"))
            
        b=float(input("Enter T3:"))
            
        c=float(input("Enter TT4:"))
            
        d=float(input("Enter FTI:"))
        
        data=np.array([[a,b,c,d]])
        
        
        print("\n")
        
        pred3=clf3.predict(data)
        
        print("Features:",data)
        
        print("Prediction:",pred3)

        
    #Accessing the KNN algorithm with undersampling's pickle files
    def KNN2():
        
        filepath='C:/Users/Admin/Desktop/Project/Model/KNN2.pkl'

        with open(filepath,'rb') as file4:

            clf4=pickle.load(file4)
        
        
        print("\n")

        print("The KNN algorithm with undersampling")
        
        
        print("\n")
        
        features=np.array([[60,1,45,27]])
        
        pred1=clf4.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred1)
        
        
        print("\n")
        
        features=np.array([[2.5,1.3,90,95]])
        
        pred2=clf4.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred2)

        
        print("\n")
        
        a=float(input("Enter TSH:"))
        
        b=float(input("Enter T3:"))
        
        c=float(input("Enter TT4:"))
        
        d=float(input("Enter FTI:"))

        data=np.array([[a,b,c,d]])

        
        print("\n")
        
        pred3=clf4.predict(data)
        
        print("Features:",data)
        
        print("Prediction:",pred3)


    #Accessing the SVM alogrithm with oversampling's pickle files
    def SVM1():
        
        filepath='C:/Users/Admin/Desktop/Project/Model/SVM1.pkl'

        with open(filepath,'rb') as file5:

            clf5=pickle.load(file5)

        
        print("\n")

        print("The SVM algorithm with oversampling")

        
        print("\n")
        
        features=np.array([[4.2,2.2,120,125]])
        
        pred1=clf5.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred1)

        
        print("\n")
        
        features=np.array([[60,1,45,27]])
        
        pred2=clf5.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred2)

        
        print("\n")
        
        a=float(input("Enter TSH:"))
    
        b=float(input("Enter T3:"))
        
        c=float(input("Enter TT4:"))
        
        d=float(input("Enter FTI:"))

    
        data=np.array([[a,b,c,d]])

        
        print("\n")
        
        pred3=clf5.predict(data)
        
        print("Features:",data)
        
        print("Prediction:",pred3)


    #Accesing the SVM algorithm with undersampling's pickle files
    def SVM2():
        
        filepath='C:/Users/Admin/Desktop/Project/Model/SVM2.pkl'

        with open(filepath,'rb') as file6:

            clf6=pickle.load(file6)

        
        print("\n")

        print("The SVM algorithm with undersampling")
        
        
        print("\n")
        
        features=np.array([[4.2,2.2,120,125]])
        
        pred1=clf6.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred1)

        
        print("\n")
        
        features=np.array([[60,1,45,27]])
        
        pred2=clf6.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred2)

        
        print("\n")
        
        a=float(input("Enter TSH:"))
        
        b=float(input("Enter T3:"))
    
        c=float(input("Enter TT4:"))
        
        d=float(input("Enter FTI:"))

        data=np.array([[a,b,c,d]])

        
        print("\n")
        
        pred3=clf6.predict(data)
        
        print("Features:",data)
        
        print("Prediction:",pred3)


    #Accessing the Decision Tree alogrithm with oversmapling's pickle files
    def Decision_Tree1():
        
        filepath='C:/Users/Admin/Desktop/Project/Model/DecisionTree1.pkl'

        with open(filepath,'rb') as file7:

            clf7=pickle.load(file7)

        
        print("\n")

        print("The Decision Tree algorithm with oversampling")
        
        
        print("\n")
        
        features=np.array([[4.2,2.2,120,125]])
        
        pred1=clf7.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred1)

        
        print("\n")
        
        features=np.array([[60,1,45,27]])
        
        pred2=clf7.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred2)

        
        print("\n")
        
        a=float(input("Enter TSH:"))
        
        b=float(input("Enter T3:"))
        
        c=float(input("Enter TT4:"))
        
        d=float(input("Enter FTI:"))

        data=np.array([[a,b,c,d]])

        
        print("\n")
        
        pred3=clf7.predict(data)
        
        print("Features:",data)
        
        print("Prediction:",pred3)

        
    #Accesing the Decision Tree algorithm with undersampling's pickle files
    def Decision_Tree2():
        
        filepath='C:/Users/Admin/Desktop/Project/Model/DecisionTree2.pkl'

        with open(filepath,'rb') as file8:

            clf8=pickle.load(file8)

        
        print("\n")

        print("The Decision Tree algorithm with undersampling")
        
        
        print("\n")
        
        features=np.array([[4.2,2.2,120,125]])
        
        pred1=clf8.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred1)

        
        print("\n")
        
        features=np.array([[60,1,45,27]])
        
        pred2=clf8.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred2)

        
        print("\n")
        
        a=float(input("Enter TSH:"))
    
        b=float(input("Enter T3:"))
    
        c=float(input("Enter TT4:"))
    
        d=float(input("Enter FTI:"))

        data=np.array([[a,b,c,d]])

        
        print("\n")
        
        pred3=clf8.predict(data)
            
        print("Features:",data)
        
        print("Prediction:",pred3)


    #Accessing the Decision Tree algorithm with oversampling after hyperparameter tuning's pickle files
    def Decision_Tree_Hyp1():
        
        filepath='C:/Users/Admin/Desktop/Project/Model/DTHyperparameter1.pkl'

        with open(filepath,'rb') as file9:

            clf9=pickle.load(file9)

        
        print("\n")

        print("The Decision Tree algorithm with hyperparameter tuning and oversampling")
        
        
        print("\n")
        
        features=np.array([[4.2,2.2,120,125]])
        
        pred1=clf9.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred1)

        
        print("\n")
        
        features=np.array([[60,1,45,27]])
        
        pred2=clf9.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred2)

        
        print("\n")
        
        a=float(input("Enter TSH:"))
    
        b=float(input("Enter T3:"))
        
        c=float(input("Enter TT4:"))
        
        d=float(input("Enter FTI:"))

        data=np.array([[a,b,c,d]])

        
        print("\n")
        
        pred3=clf9.predict(data)
        
        print("Features:",data)
        
        print("Prediction:",pred3)


    #Accessing the Decision Tree algorithm with undersampling after hyperparameter tuning's pickle files
    def Decision_Tree_Hyp2():
        
        filepath='C:/Users/Admin/Desktop/Project/Model/DTHyperparameter2.pkl'

        with open(filepath,'rb') as file10:

            clf10=pickle.load(file10)

        
        print("\n")

        print("The Decision Tree algorithm with hyperparameter tuning and undersampling")
        
        
        print("\n")
        
        features=np.array([[4.2,2.2,120,125]])
        
        pred1=clf10.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred1)

        
        print("\n")
        
        features=np.array([[60,1,45,27]])
        
        pred2=clf10.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred2)

        
        print("\n")
        
        a=float(input("Enter TSH:"))
        
        b=float(input("Enter T3:"))
        
        c=float(input("Enter TT4:"))
        
        d=float(input("Enter FTI:"))

        data=np.array([[a,b,c,d]])

        
        print("\n")
        
        pred3=clf10.predict(data)
        
        print("Features:",data)
        
        print("Prediction:",pred3)


    #Accessig the Random Forest algorithm with oversampling's pickle files 
    def Random_Forest1():
        
        filepath='C:/Users/Admin/Desktop/Project/Model/RandomForest1.pkl'

        with open(filepath,'rb') as file11:

            clf11=pickle.load(file11)

        
        print("\n")

        print("The Random Forest algorithm with oversampling")
        
        
        print("\n")
        
        features=np.array([[4.2,2.2,120,125]])
        
        pred1=clf11.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred1)

        
        print("\n")
        
        features=np.array([[60,1,45,27]])
        
        pred2=clf11.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred2)

        
        print("\n")
        
        a=float(input("Enter TSH:"))
    
        b=float(input("Enter T3:"))
    
        c=float(input("Enter TT4:"))
    
        d=float(input("Enter FTI:"))

        data=np.array([[a,b,c,d]])

        
        print("\n")
        
        pred3=clf11.predict(data)
        
        print("Features:",data)
        
        print("Prediction:",pred3)


    #Accessing the Random Forest algorithm with undersmapling's pickle files
    def Random_Forest2():
        
        filepath='C:/Users/Admin/Desktop/Project/Model/RandomForest2.pkl'

        with open(filepath,'rb') as file12:

            clf12=pickle.load(file12)


        print("\n")

        print("The Random Forest algorithm with undersampling")
        
        
        print("\n")
        
        features=np.array([[4.2,2.2,120,125]])
        
        pred1=clf12.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred1)


        print("\n")
        
        features=np.array([[60,1,45,27]])
        
        pred2=clf12.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred2)


        print("\n")
        
        a=float(input("Enter TSH:"))
        
        b=float(input("Enter T3:"))
        
        c=float(input("Enter TT4:"))
    
        d=float(input("Enter FTI:"))

        data=np.array([[a,b,c,d]])


        print("\n")
        
        pred3=clf12.predict(data)
        
        print("Features:",data)
        
        print("Prediction:",pred3)


    #Accessing the Random Forest algorithm with oversampling after hyperparameter tuning's pickle files
    def Random_Forest_Hyp1():
        
        filepath='Model/RFHyperparameter1.pkl'

        with open(filepath,'rb') as file13:

            clf13=pickle.load(file13)


        print("\n")
        
        print("The Random Forest algorithm with hyperparameter tuning and oversampling")
        
        
        print("\n")
        
        features=np.array([[4.2,2.2,120,125]])
        
        pred1=clf13.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred1)


        print("\n")
        
        features=np.array([[60,1,45,27]])
        
        pred1=clf13.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred1)


        print("\n")

        try:
            
            a=float(input("Enter TSH:"))
    
            b=float(input("Enter T3:"))
    
            c=float(input("Enter TT4:"))
    
            d=float(input("Enter FTI:"))

        except EOFError:

            a = 1.2
            
            b = 15

            c = 80

            d = 82

        data=np.array([[a,b,c,d]])


        print("\n")
        
        pred3=clf13.predict(data)
        
        print("Features:",data)
        
        print("Prediction:",pred3)


    #Accessing the Random Forest alogrithm with undersampling after hyperparameter tuning's pickle files
    def Random_Forest_Hyp2():
        
        filepath='C:/Users/Admin/Desktop/Project/Model/RFHyperparameter2.pkl'

        with open(filepath,'rb') as file14:

            clf14=pickle.load(file14)


        print("\n")
        
        print("The Random Forest algorithm with hyperparameter tuning and undersampling")
        
        
        print("\n")
        
        features=np.array([[4.2,2.2,120,125]])
        
        pred1=clf14.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred1)


        print("\n")
        
        features=np.array([[60,1,45,27]])
        
        pred2=clf14.predict(features)
        
        print("Features:",features)
        
        print("Prediction:",pred2)


        print("\n")
        
        a=float(input("Enter TSH:"))
    
        b=float(input("Enter T3:"))
        
        c=float(input("Enter TT4:"))
        
        d=float(input("Enter FTI:"))

        data=np.array([[a,b,c,d]])


        print("\n")
        
        pred3=clf14.predict(data)
        
        print("Features:",data)
        
        print("Prediction:",pred3)

    print("\n")


