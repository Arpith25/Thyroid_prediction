#Importing the required packages
import numpy as np

import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB

from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

import pickle

from sklearn import preprocessing

import sys

sys.path.insert(0, 'C:/Users/Admin/Desktop/Project/Preprocessing')

import Preprocessing


#Performing EDA
X21_train, X21_test, Y21_train, Y21_test, X22_train, X22_test, Y22_train, Y22_test=Preprocessing.processing_call()


#Creating the ML_models class
class ML_models:

    #Introduction
    print("Creating predictive models:")
    
    
    #Applying the Naive Bayes Algorithm
    gn1=GaussianNB()
        
    gn1.fit(X21_train,Y21_train)
        
    pred1=gn1.predict(X21_test)
        
    predtrain1=gn1.predict(X21_train)
    
    
    gn2=GaussianNB()
    
    gn2.fit(X22_train,Y22_train)
    
    pred2=gn2.predict(X22_test)
    
    predtrain2=gn2.predict(X22_train)
    
    
    print("\n")
    
    print("\n")
    
    print("The Naive Bayes algorithm with oversampling:")
    
    print("\n")
    
    print("Y21 test:",Y21_test)

    print("\n")
    
    print("Pred1:",pred1)

    
    print("\n")

    print("Confusion Matrix:")
    
    conf1=metrics.confusion_matrix(Y21_test,pred1)
    
    cmd1=metrics.ConfusionMatrixDisplay(confusion_matrix=conf1,display_labels=['No','Yes'])
    
    cmd1.plot()
    
    plt.show()

    
    print("Training accuracy:",metrics.accuracy_score(Y21_train,predtrain1))
    
    print("Test:")
    
    print("Accuracy:",metrics.accuracy_score(Y21_test,pred1))
    
    print("Precision:",metrics.precision_score(Y21_test,pred1,pos_label="P"))
    
    print("Recall:",metrics.recall_score(Y21_test,pred1,pos_label="P"))
    
    print("Sensitivity:",metrics.recall_score(Y21_test,pred1,pos_label="N"))
    
    print("F1 score:",metrics.f1_score(Y21_test,pred1,pos_label="P"))

    
    print("\n")

    print("Testing predictive accuracy:")

    
    print("\n")
    
    features=np.array([[3.3,1.8,109,119]])
    
    npred1=gn1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred1)

    
    print("\n")
    
    features=np.array([[2.8,1.7,97,107]])
    
    npred2=gn1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred2)

    
    print("\n")       
    
    features=np.array([[45,1.4,39,33]])        
    
    npred3=gn1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred3)

    
    print("\n")
    
    features=np.array([[14.8,1.5,61,72]])
    
    npred4=gn1.predict(features)        
    
    print("Fetaures:",features)        
    
    print("Prediction:",npred4)
    
    
    print("\n")
    
    features=np.array([[4.2,2.2,120,125]])
    
    npred5=gn1.predict(features)        
    
    print("Features:",features)        
    
    print("Prediction:",npred5)

    
    print("\n")
    
    features=([[2.5,1.3,90,95]])
    
    npred6=gn1.predict(features)        
    
    print("Features:",features)        
    
    print("Prediction:",npred6)

    
    print("\n")
    
    features=([[60,1,45,27]])
    
    npred7=gn1.predict(features)
    
    print("Features:",features)        
    
    print("Prediction:",npred7)

    
    print("\n")
    
    features=np.array([[20,1.5,70,65]])
    
    npred8=gn1.predict(features)
    
    print("Features:",features)        
    
    print("Prediction:",npred8)
    
    
    print("\n")
    
    print("\n")

    print("The Naive Bayes algorithm with undersampling")
    
    print("\n")
    
    print("Y22 test:",Y22_test)
    
    print("\n")
    
    print("Pred2:",pred2)

    
    print("\n")

    print("Confusion Matrix:")
    
    conf2=metrics.confusion_matrix(Y22_test,pred2)
    
    cmd2=metrics.ConfusionMatrixDisplay(confusion_matrix=conf2,display_labels=["No","Yes"])
    
    cmd2.plot()
    
    plt.show()

    
    print("Training accuracy:",metrics.accuracy_score(Y22_train,predtrain2))
    
    print("Test")
    
    print("Accuracy:",metrics.accuracy_score(Y22_test,pred2))
    
    print("Precision:",metrics.precision_score(Y22_test,pred2,pos_label="P"))
    
    print("Recall:",metrics.recall_score(Y22_test,pred2,pos_label="P"))
    
    print("Sensitivity:",metrics.recall_score(Y22_test,pred2,pos_label="N"))
    
    print("F1 score:",metrics.f1_score(Y22_test,pred2,pos_label="P"))
    
    
    print("\n")

    print("Testing predictive accuracy:")


    print("\n")
    
    features=np.array([[3.3,1.8,109,119]])
    
    npred21=gn2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred21)
    
    
    print("\n")
    
    features=np.array([[2.8,1.7,97,107]])
    
    npred22=gn2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred22)
    
    
    print("\n")
    
    features=np.array([[45,1.4,39,33]])
    
    npred23=gn2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred23)


    print("\n")
    
    features=np.array([[14.8,1.5,61,72]])
    
    npred24=gn2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred24)

    
    print("\n")
    
    features=np.array([[4.2,2.2,120,125]])
    
    npred25=gn2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred25)

    
    print("\n")
    
    features=([[2.5,1.3,90,95]])
    
    npred26=gn2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred26)


    print("\n")
    
    features=([[60,1,45,27]])
    
    npred27=gn2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred27)

    
    print("\n")
    
    features=np.array([[20,1.5,70,65]])
    
    npred28=gn2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred28)

    
    #Applying KNN        
    knn1=KNeighborsClassifier()
    
    knn1.fit(X21_train,Y21_train)
    
    pred1=knn1.predict(X21_test)
    
    predtrain1=knn1.predict(X21_train)
    
    
    knn2=KNeighborsClassifier()
    
    knn2.fit(X22_train,Y22_train)
    
    pred2=knn2.predict(X22_test)
    
    predtrain2=knn2.predict(X22_train)
    

    print("\n")
    
    print("\n")
    
    print("The KNN algorithm wiht oversampling")
    
    print("\n")
    
    print("Y21 test:",Y21_test)

    print("\n")
    
    print("Pred1:",pred1)

    
    print("\n")

    print("Confusion Matrix:")
    
    conf1=metrics.confusion_matrix(Y21_test,pred1)
    
    cmd1=metrics.ConfusionMatrixDisplay(confusion_matrix=conf1,display_labels=["No","Yes"])
    
    cmd1.plot()
    
    plt.show()

    
    print("Training accuracy:",metrics.accuracy_score(Y21_train,predtrain1))
    
    print("Test")
    
    print("Accuracy:",metrics.accuracy_score(Y21_test,pred1))
    
    print("Precision:",metrics.precision_score(Y21_test,pred1,pos_label="P"))
    
    print("Recall:",metrics.recall_score(Y21_test,pred1,pos_label="P"))
    
    print("Sensitivity:",metrics.recall_score(Y21_test,pred1,pos_label="N"))
    
    print("F1 score:",metrics.f1_score(Y21_test,pred1,pos_label="P"))
    
    
    print("\n")

    print("Testing predictive accuracy:")

    
    print("\n")
    
    features=np.array([[3.3,1.8,109,119]])
    
    npred1=knn1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred1)
    
    
    print("\n")
    
    features=np.array([[2.8,1.7,97,107]])
    
    npred2=knn1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred2)
    
    
    print("\n")
    
    features=np.array([[45,1.4,39,33]])
    
    npred3=knn1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred3)
    
    
    print("\n")
    
    features=np.array([[14.8,1.5,61,72]])
    
    npred4=knn1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred4)
    
    
    print("\n")
    
    features=np.array([[4.2,2.2,120,125]])
    
    npred5=knn1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred5)
    
    
    print("\n")
    
    features=([[2.5,1.3,90,95]])
    
    npred6=knn1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred6)
    
    
    print("\n")
    
    features=([[60,1,45,27]])
    
    npred7=knn1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred7)
    
    
    print("\n")
    
    features=np.array([[20,1.5,70,65]])
    
    npred8=knn1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred8)
    
    
    print("\n")
    
    print("\n")

    print("The KNN algorithm with undersampling")
    
    print("\n")
    
    print("Y22 test:",Y22_test)

    print("\n")
    
    print("Pred 2:",pred2)

    
    print("\n")

    print("Confusion Matrix:")
    
    conf2=metrics.confusion_matrix(Y22_test,pred2)
    
    cmd2=metrics.ConfusionMatrixDisplay(confusion_matrix=conf2,display_labels=["No","Yes"])
    
    cmd2.plot()
    
    plt.show()
    
    
    print("Training accuracy:",metrics.accuracy_score(Y22_train,predtrain2))
    
    print("Test")
    
    print("Accuracy:",metrics.accuracy_score(Y22_test,pred2))
    
    print("Precision:",metrics.precision_score(Y22_test,pred2,pos_label="P"))
    
    print("Recall:",metrics.recall_score(Y22_test,pred2,pos_label="P"))
    
    print("Sensitivity:",metrics.recall_score(Y22_test,pred2,pos_label="N"))
    
    print("F1 score:",metrics.f1_score(Y22_test,pred2,pos_label="P"))
    
    
    print("\n")

    print("Testing predictive accuracy:")


    print("\n")
    
    features=np.array([[3.3,1.8,109,119]])
    
    npred21=knn2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred21)
    

    print("\n")
    
    features=np.array([[2.8,1.7,97,107]])
    
    npred22=knn2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred22)
    

    print("\n")
    
    features=np.array([[45,1.4,39,33]])
    
    npred23=knn2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred23)
    

    print("\n")
    
    features=np.array([[14.8,1.5,61,72]])
    
    npred24=knn2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred24)
    
    
    print("\n")
    
    features=np.array([[4.2,2.2,120,125]])
    
    npred25=knn2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred25)
    
    
    print("\n")
    
    features=([[2.5,1.3,90,95]])
    
    npred26=knn2.predict(features)
    
    print("Features:",features)
    
    print("Predicition:",npred26)
    

    print("\n")
    
    features=([[60,1,45,27]])
    
    npred27=knn2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred27)
    

    print("\n")
    
    features=np.array([[20,1.5,70,65]])
    
    npred28=knn2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred28)


    #Applying the SVM algorithm
    cls1=svm.SVC()
    
    cls1.fit(X21_train,Y21_train)
    
    pred1=cls1.predict(X21_test)
    
    predtrain1=cls1.predict(X21_train)
    
    
    cls2=svm.SVC()
    
    cls2.fit(X22_train,Y22_train)
    
    pred2=cls2.predict(X22_test)
    
    predtrain2=cls2.predict(X22_train)
    
    
    print("\n")
    
    print("\n")
    
    print("The SVM algorithm with oversampling")
    
    print("\n")
    
    print("Y21 test:",Y21_test)
    
    print("\n")
        
    print("Pred 1:",pred1)
    
    
    print("\n")

    print("Confusion Matrix:")
    
    conf1=metrics.confusion_matrix(Y21_test,pred1)
    
    cmd1=metrics.ConfusionMatrixDisplay(confusion_matrix=conf1,display_labels=["No","Yes"])
    
    cmd1.plot()
    
    plt.show()
    
    
    print("Training accuracy:",metrics.accuracy_score(Y21_train,predtrain1))
    
    print("Test")
    
    print("Accuracy:",metrics.accuracy_score(Y21_test,pred1))
    
    print("Precision:",metrics.precision_score(Y21_test,pred1,pos_label="P"))
    
    print("Recall:",metrics.recall_score(Y21_test,pred1,pos_label="P"))
    
    print("Sensitivity:",metrics.recall_score(Y21_test,pred1,pos_label="N"))
    
    print("F1 score:",metrics.f1_score(Y21_test,pred1,pos_label="P"))
    
    
    print("\n")

    print("Testing predictive accuracy:")
    

    print("\n")
    
    features=np.array([[3.3,1.8,109,119]])
    
    npred1=cls1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred1)
    
    
    print("\n")
    
    features=np.array([[2.8,1.7,97,107]])
    
    npred2=cls1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred2)
    
    
    print("\n")
    
    features=np.array([[45,1.4,39,33]])
    
    npred3=cls1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred3)
    

    print("\n")
    
    features=np.array([[14.8,1.5,61,72]])
    
    npred4=cls1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred4)
    

    print("\n")
    
    features=np.array([[4.2,2.2,120,125]])
    
    npred5=cls1.predict(features)
    
    print("Features:",features)
    
    print("Predicition:",npred5)
    

    print("\n")
    
    features=([[2.5,1.3,90,95]])
    
    npred6=cls1.predict(features)
    
    print("Features:",features)
    
    print("Predicition:",npred6)
    
    
    print("\n")
    
    features=([[60,1,45,27]])
    
    npred7=cls1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred7)
    

    print("\n")
    
    features=np.array([[20,1.5,70,65]])
    
    npred8=cls1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred8)
    
    
    print("\n")
    
    print("\n")

    print("The SVM algorithm with undersampling")
    
    print("\n")
    
    print("Y22 test:",Y22_test)

    print("\n")
    
    print("Pred 2:",pred2)

    
    print("\n")

    print("Confusion Matrix:")
    
    conf2=metrics.confusion_matrix(Y22_test,pred2)
    
    cmd2=metrics.ConfusionMatrixDisplay(confusion_matrix=conf2,display_labels=["No","Yes"])
    
    cmd2.plot()
    
    plt.show()
    
    
    print("Training accuracy:",metrics.accuracy_score(Y22_train,predtrain2))
    
    print("Test")
    
    print("Accuracy:",metrics.accuracy_score(Y22_test,pred2))
    
    print("Precision:",metrics.precision_score(Y22_test,pred2,pos_label="P"))
    
    print("Recall:",metrics.recall_score(Y22_test,pred2,pos_label="P"))
    
    print("Sensitivity:",metrics.recall_score(Y22_test,pred2,pos_label="N"))
    
    print("F1 score:",metrics.f1_score(Y22_test,pred2,pos_label="P"))
    

    print("\n")
    
    print("Testing predictive accuracy:")


    print("\n")
    
    features=np.array([[3.3,1.8,109,119]])
    
    npred21=cls2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred21)
    

    print("\n")
    
    features=np.array([[2.8,1.7,97,107]])
    
    npred22=cls2.predict(features)
    
    print("Fetures:",features)
    
    print("Prediction:",npred22)
    

    print("\n")
    
    features=np.array([[45,1.4,39,33]])
    
    npred23=cls2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred23)


    print("\n")
    
    features=np.array([[14.8,1.5,61,72]])
    
    npred24=cls2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred24)
    

    print("\n")
    
    features=np.array([[4.2,2.2,120,125]])
    
    npred25=cls2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred25)
    

    print("\n")
    
    features=([[2.5,1.3,90,95]])
    
    npred26=cls2.predict(features)
    
    print("Features:",features)
    
    print("Predicton:",npred26)
    

    print("\n")
    
    features=([[60,1,45,27]])
    
    npred27=cls2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred27)
    

    print("\n")
    
    features=np.array([[20,1.5,70,65]])
    
    npred28=cls2.predict(features)
    
    print("Features:",features)
    
    print("Predicton:",npred28)
        
    
    #Applying the Decision Tree algorithm
    dtree1=DecisionTreeClassifier()
    
    dtree1.fit(X21_train,Y21_train)
    
    pred1=dtree1.predict(X21_test)
    
    predtrain1=dtree1.predict(X21_train)
    
    
    dtree2=DecisionTreeClassifier()
    
    dtree2.fit(X22_train,Y22_train)
    
    pred2=dtree2.predict(X22_test)
    
    predtrain2=dtree2.predict(X22_train)
    

    print("\n")
    
    print("\n")
    
    print("The Decision Tree algorithm with oversampling")
    
    print("\n")
    
    print("Y21_test:",Y21_test)
    
    print("\n")
    
    print("Pred 1:",pred1)
    
    
    print("\n")

    print("Confusion Matrix")
    
    conf1=metrics.confusion_matrix(Y21_test,pred1)
    
    cmd1=metrics.ConfusionMatrixDisplay(confusion_matrix=conf1,display_labels=["No","Yes"])
    
    cmd1.plot()
    
    plt.show()
    
    
    print("Training accuracy:",metrics.accuracy_score(Y21_train,predtrain1))
    
    print("Test")
    
    print("Accuracy:",metrics.accuracy_score(Y21_test,pred1))
    
    print("Precision:",metrics.precision_score(Y21_test,pred1,pos_label="P"))
    
    print("Recall:",metrics.recall_score(Y21_test,pred1,pos_label="P"))
    
    print("Specifity:",metrics.recall_score(Y21_test,pred1,pos_label="N"))
    
    print("F1 score:",metrics.f1_score(Y21_test,pred1,pos_label="P"))

    
    print("\n")

    print("Testing predictive accuracy:")

    
    print("\n")
    
    features=np.array([[3.3,1.8,109,119]])
    
    npred1=dtree1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred1)
    
    
    print("\n")
    
    features=np.array([[2.8,1.7,97,107]])
    
    npred2=dtree1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred2)
    
    
    print("\n")
    
    features=np.array([[45,1.4,39,33]])
    
    npred3=dtree1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred3)
    
    
    print("\n")
    
    features=np.array([[14.8,1.5,61,72]])
    
    npred4=dtree1.predict(features)
    
    print("Features:",features)
    
    print("Predicition:",npred4)
    
    
    print("\n")
    
    features=np.array([[4.2,2.2,120,125]])
    
    npred5=dtree1.predict(features)
    
    print("Features:",features)
    
    print("prediction:",npred5)
    
    
    print("\n")
    
    features=([[2.5,1.3,90,95]])
    
    npred6=dtree1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred6)

    
    print("\n")
    
    features=([[60,1,45,27]])
    
    npred7=dtree1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred7)
    
    
    print("\n")
    
    features=np.array([[20,1.5,70,65]])
    
    npred8=dtree1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred8)
    
    
    print("\n")
    
    print("\n")

    print("The Decision Tree algorithm with undersampling")
    
    print("\n")
    
    print("Y22 test:",Y22_test)
    
    print("\n")
    
    print("Pred 2:",pred2)
    
    
    print("\n")

    print("Confusion Matrix")
    
    conf2=metrics.confusion_matrix(Y22_test,pred2)
    
    cmd2=metrics.ConfusionMatrixDisplay(confusion_matrix=conf2,display_labels=["No","Yes"])
    
    cmd2.plot()
    
    plt.show()
    
    
    print("Training accuracy:",metrics.accuracy_score(Y22_train,predtrain2))
    
    print("Test")
    
    print("Accuracy:",metrics.accuracy_score(Y22_test,pred2))
    
    print("Precision:",metrics.precision_score(Y22_test,pred2,pos_label="P"))
    
    print("Recall:",metrics.recall_score(Y22_test,pred2,pos_label="P"))
    
    print("Specifity:",metrics.recall_score(Y22_test,pred2,pos_label="N"))
    
    print("F1 score:",metrics.f1_score(Y22_test,pred2,pos_label="P"))
    
    
    print("\n")

    print("Testing predictive accuracy:")
    
    
    print("\n")
    
    features=np.array([[3.3,1.8,109,119]])
    
    npred21=dtree2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred21)
    
    
    print("\n")
    
    features=np.array([[2.8,1.7,97,107]])
    
    npred22=dtree2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred22)
    
    
    print("\n")
    
    features=np.array([[45,1.4,39,33]])
    
    npred23=dtree2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred23)
    
    
    print("\n")
    
    features=np.array([[14.8,1.5,61,72]])
    
    npred24=dtree2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred24)
    
    
    print("\n")
    
    features=np.array([[4.2,2.2,120,125]])
    
    npred25=dtree2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred25)
    
    
    print("\n")
    
    features=([[2.5,1.3,90,95]])
    
    npred26=dtree2.predict(features)
    
    print("Features:",features)
    
    print("Prediction",npred26)
    
    
    print("\n")
    
    features=([[60,1,45,27]])
    
    npred27=dtree2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred27)
    
    
    print("\n")
    
    features=np.array([[20,1.5,70,65]])
    
    npred28=dtree2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred28)

    
    print("\n")

    print("\n")
        
    
    #Applying Decision Trees with Hyperparameter Tuning
    para={'max_depth':[2,4,6,8,10],'max_features':['sqrt','log2',None],'criterion':['gini','entropy'],'splitter':['best','random'],'min_samples_split':[1,2,5,10]}
    
    dtree3=DecisionTreeClassifier()
    
    gs1=GridSearchCV(dtree3,param_grid=para,cv=10,scoring='accuracy')
    
    gs1.fit(X21_train,Y21_train)
    
    print("\n")
    
    print("The Decision Tree algorithm with oversampling's ideal hyperparameters")

    print("\n")
    
    print(gs1.best_params_)
    
    dtree32=DecisionTreeClassifier(criterion='gini', max_depth=4, max_features=None,  min_samples_split=10, splitter='best')
    
    dtree32.fit(X21_train,Y21_train)
    
    pred3=dtree32.predict(X21_test)
    
    predtrain3=dtree32.predict(X21_train)

    print("\n")

    print("\n")
    
    
    dtree4=DecisionTreeClassifier()
    
    gs2=GridSearchCV(dtree4,param_grid=para,cv=10,scoring='accuracy')
    
    gs2.fit(X22_train,Y22_train)
    
    print("\n")
    
    print("The Decision Tree algorithm with undersampling's ideal hyperparameters")

    print("\n")
    
    print(gs2.best_params_)
    
    dtree42=DecisionTreeClassifier(criterion='gini', max_depth=2, max_features=None, min_samples_split=10, splitter='best')
    
    dtree42.fit(X22_train,Y22_train)
    
    pred4=dtree42.predict(X22_test)
    
    predtrain4=dtree42.predict(X22_train)

    
    print("\n")
    
    print("\n")
    
    print("The Deecision Tree algorithm with hyperparameter tuning and oversampling")
    
    print("\n")
    
    print("Y21 test:",Y21_test)
    
    print("\n")
    
    print("Pred 3:",pred3)
    
    
    print("\n")

    print("Confusion Matrix")
    
    conf1=metrics.confusion_matrix(Y21_test,pred3)
    
    cmd1=metrics.ConfusionMatrixDisplay(confusion_matrix=conf1,display_labels=["No","Yes"])
    
    cmd1.plot()
    
    plt.show()
    
    
    print("Training accuracy:",metrics.accuracy_score(Y21_train,predtrain3))
    
    print("Test")
    
    print("Accuracy:",metrics.accuracy_score(Y21_test,pred3))
    
    print("Precision:",metrics.precision_score(Y21_test,pred3,pos_label="P"))
    
    print("Recall:",metrics.recall_score(Y21_test,pred3,pos_label="P"))
    
    print("Specificity:",metrics.recall_score(Y21_test,pred3,pos_label="N"))
    
    print("F1 score:",metrics.f1_score(Y21_test,pred3,pos_label="P"))


    print("\n")
    
    print("Testing prediction accuracy:")
    
    
    print("\n")
    
    features=np.array([[3.3,1.8,109,119]])
    
    npred1=dtree32.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred1)
    
    
    print("\n")
    
    features=np.array([[2.8,1.7,97,107]])
    
    npred2=dtree32.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred2)

    
    print("\n")
    
    features=np.array([[45,1.4,39,33]])
    
    npred3=dtree32.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred3)
    
    
    print("\n")
    
    features=np.array([[14.8,1.5,61,72]])
    
    npred4=dtree32.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred4)
    
    
    print("\n")
    
    features=np.array([[4.2,2.2,120,125]])
    
    npred5=dtree32.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred5)
    
    
    print("\n")
    
    features=([[2.5,1.3,90,95]])
    
    npred6=dtree32.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred6)
    
    
    print("\n")
    
    features=([[60,1,45,27]])
    
    npred7=dtree32.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred7)
    
    
    print("\n")
    
    features=np.array([[20,1.5,70,65]])
    
    npred8=dtree32.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred8)

    
    print("\n")
    
    print("\n")

    print("The Decision Tree algorithm with hyperparameter tuning and undersampling")
    
    print("\n")
    
    print("Y22 test:",Y22_test)
    
    print("\n")
    
    print("Pred 4:",pred4)

    
    print("\n")

    print("Confusion Matrix")
    
    conf2=metrics.confusion_matrix(Y22_test,pred4)
    
    cmd2=metrics.ConfusionMatrixDisplay(confusion_matrix=conf2,display_labels=["No","Yes"])
    
    cmd2.plot()
    
    plt.show()
    
    
    print("Training accuracy:",metrics.accuracy_score(Y22_train,predtrain4))
    
    print("Test")
    
    print("Accuracy:",metrics.accuracy_score(Y22_test,pred4))
    
    print("Precision:",metrics.precision_score(Y22_test,pred4,pos_label="P"))
    
    print("Recall:",metrics.recall_score(Y22_test,pred4,pos_label="P"))
    
    print("Specificity:",metrics.recall_score(Y22_test,pred4,pos_label="N"))
    
    print("F1 score:",metrics.f1_score(Y22_test,pred4,pos_label="P"))
    
    
    print("\n")
    
    print("Testing predictive accuracy:")
            
    
    print("\n")
    
    features=np.array([[3.3,1.8,109,119]])
    
    npred21=dtree42.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred21)
    
    
    print("\n")
    
    features=np.array([[2.8,1.7,97,107]])
    
    npred22=dtree42.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred22)
    
    
    print("\n")
    
    features=np.array([[45,1.4,39,33]])
    
    npred23=dtree42.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred23)
    
    
    print("\n")
    
    features=np.array([[14.8,1.5,61,72]])
    
    npred24=dtree42.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred24)
    
    
    print("\n")
    
    features=np.array([[4.2,2.2,120,125]])
    
    npred25=dtree42.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred25)
    
    
    print("\n")
    
    features=([[2.5,1.3,90,95]])
    
    npred26=dtree42.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred26)
    
    
    print("\n")
    
    features=([[60,1,45,27]])
    
    npred27=dtree42.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred27)
    
    
    print("\n")
    
    features=np.array([[20,1.5,70,65]])
    
    npred28=dtree42.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred28)


    #Applying the Random Forest algorithm        
    rf1=RandomForestClassifier()
    
    rf1.fit(X21_train,Y21_train)
    
    pred1=rf1.predict(X21_test)
    
    predtrain1=rf1.predict(X21_train)
    
    
    rf2=RandomForestClassifier()
    
    rf2.fit(X22_train,Y22_train)
    
    pred2=rf2.predict(X22_test)
    
    predtrain2=rf2.predict(X22_train)
    

    print("\n")
    
    print("\n")

    print("The Random Forest algorithm with oversampling")
    
    print("\n")
    
    print("Y21 test:",Y21_test)
    
    print("\n")
    
    print("Pred 1:",pred1)
    
    
    print("\n")

    print("Confusion Matrix")
    
    conf1=metrics.confusion_matrix(Y21_test,pred1)
    
    cmd1=metrics.ConfusionMatrixDisplay(confusion_matrix=conf1,display_labels=["No","Yes"])
    
    cmd1.plot()
    
    plt.show()
    
    
    print("Training accuracy:",metrics.accuracy_score(Y21_train,predtrain1))
    
    print("Test")
    
    print("Accuracy:",metrics.accuracy_score(Y21_test,pred1))
    
    print("Precision:",metrics.precision_score(Y21_test,pred1,pos_label="P"))
    
    print("Recall:",metrics.recall_score(Y21_test,pred1,pos_label="P"))
    
    print("Specificity:",metrics.recall_score(Y21_test,pred1,pos_label="N"))
    
    print("F1 score:",metrics.f1_score(Y21_test,pred1,pos_label="P"))
    
    
    print("\n")

    print("Testing predictive accuracy:")
    
    
    print("\n")
    
    features=np.array([[3.3,1.8,109,119]])
    
    npred1=rf1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred1)
    
    
    print("\n")
    
    features=np.array([[2.8,1.7,97,107]])
    
    npred2=rf1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred2)
    
    
    print("\n")
    
    features=np.array([[45,1.4,39,33]])
    
    npred3=rf1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred3)
    
    
    print("\n")
    
    features=np.array([[14.8,1.5,61,72]])
    
    npred4=rf1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred4)
    
    
    print("\n")
    
    features=np.array([[4.2,2.2,120,125]])
    
    npred5=rf1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred5)
    
    
    print("\n")
    
    features=([[2.5,1.3,90,95]])
    
    npred6=rf1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred6)
    
    
    print("\n")
    
    features=([[60,1,45,27]])
    
    npred7=rf1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred7)
    
    
    print("\n")
    
    features=np.array([[20,1.5,70,65]])
    
    npred8=rf1.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred8)
    
    
    print("\n")
    
    print("\n")

    print("The Random Forest algorithm with undersampling")
    
    print("\n")
    
    print("Y22 test:",Y22_test)
    
    print("\n")
    
    print("Pred 2:",pred2)
    
    
    print("\n")

    print("Confusion Matrix")
    
    conf2=metrics.confusion_matrix(Y22_test,pred2)
    
    cmd2=metrics.ConfusionMatrixDisplay(confusion_matrix=conf2,display_labels=["No","Yes"])
    
    cmd2.plot()
    
    plt.show()
    
    
    print("Training accuracy:",metrics.accuracy_score(Y22_train,predtrain2))
    
    print("Test")
    
    print("Accuracy:",metrics.accuracy_score(Y22_test,pred2))
    
    print("Precision:",metrics.precision_score(Y22_test,pred2,pos_label="P"))
    
    print("Recall:",metrics.recall_score(Y22_test,pred2,pos_label="P"))
    
    print("Specificity:",metrics.recall_score(Y22_test,pred2,pos_label="N"))
    
    print("F1 score:",metrics.f1_score(Y22_test,pred2,pos_label="P"))
    

    print("\n")

    print("Testing predictive accuracy:")
    
    
    print("\n")
    
    features=np.array([[3.3,1.8,109,119]])
    
    npred21=rf2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred21)
    
    
    print("\n")
    
    features=np.array([[2.8,1.7,97,107]])
    
    npred22=rf2.predict(features)

    print("Features:",features)
    
    print("Prediction:",npred22)

    
    print("\n")
    
    features=np.array([[45,1.4,39,33]])
    
    npred23=rf2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred23)
    
    
    print("\n")
    
    features=np.array([[14.8,1.5,61,72]])
    
    npred24=rf2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred24)
    
    
    print("\n")
    
    features=np.array([[4.2,2.2,120,125]])
    
    npred25=rf2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred25)
    
    
    print("\n")
    
    features=([[2.5,1.3,90,95]])
    
    npred26=rf2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred26)
    
    
    print("\n")
    
    features=([[60,1,45,27]])
    
    npred27=rf2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred27)
    
    
    print("\n")
    
    features=np.array([[20,1.5,70,65]])
    
    npred28=rf2.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred28)
        
    
    #Applying the Random Forest Algorithm with hyper parameter tuning
    para={'n_estimators':[90,110], 'max_depth':[2,6,10], 'max_features':['sqrt','log2',None], 'min_samples_split':[2,5,10], 'bootstrap':[True,False]}
    
    
    rf3=RandomForestClassifier()
    
    g3=GridSearchCV(rf3, param_grid=para, cv=10, scoring='accuracy')
    
    g3.fit(X21_train,Y21_train)
    
    print("\n")
    
    print("\n")
    
    print("The Random Forest algorithm with oversampling's ideal hyperparameters")

    print("\n")
    
    print(g3.best_params_)
    
    rf32=RandomForestClassifier(bootstrap=False, max_depth=10, max_features='sqrt', min_samples_split=2, n_estimators=90)
    
    rf32.fit(X21_train,Y21_train)
    
    pred3=rf32.predict(X21_test)
    
    predtrain3=rf32.predict(X21_train)
    
    
    rf4=RandomForestClassifier()
    
    g4=GridSearchCV(rf4, param_grid=para, cv=10, scoring='accuracy')
    
    g4.fit(X22_train,Y22_train)
    
    print("\n")
    
    print("The Random Forest algorithm with undersampling's ideal hyperparameters")

    print("\n")
    
    print(g4.best_params_)
    
    rf42=RandomForestClassifier(bootstrap=False, max_depth=2, max_features=None, min_samples_split=2, n_estimators=100)
    
    rf42.fit(X22_train,Y22_train)
    
    pred4=rf42.predict(X22_test)
    
    predtrain4=rf42.predict(X22_train)
    

    print("\n")
    
    print("\n")
    
    print("The Random Forest algorithm with hyperparameter tuning and oversampling")
    
    print("\n")
    
    print("Y21 test:",Y21_test)
    
    print("\n")
    
    print("Pred 3:",pred3)
    
    
    print("\n")

    print("Confusion Matrix")
    
    conf1=metrics.confusion_matrix(Y21_test,pred3)
    
    cmd1=metrics.ConfusionMatrixDisplay(confusion_matrix=conf1, display_labels=['No','Yes'])
    
    cmd1.plot()
    
    plt.show()
    
    
    print("Traing accuracy:",metrics.accuracy_score(Y21_train,predtrain3))
    
    print("Test")
    
    print('Accuracy:',metrics.accuracy_score(Y21_test,pred3))
    
    print('Precision:',metrics.precision_score(Y21_test,pred3,pos_label='P'))
    
    print('Recall:',metrics.recall_score(Y21_test,pred3,pos_label='P'))
    
    print('Sensitivity:',metrics.recall_score(Y21_test,pred3,pos_label='N'))
    
    print('F1 score:',metrics.f1_score(Y21_test,pred3,pos_label='P'))
    
    
    print("\n")

    print("Testing predictive accuracy:")

    
    print("\n")
    
    features=np.array([[3.3,1.8,109,119]])
    
    npred1=rf32.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred1)

    
    print("\n")
    
    features=np.array([[2.8,1.7,97,107]])
    
    npred2=rf32.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred2)
    
    
    print("\n")
    
    features=np.array([[45,1.4,39,33]])
    
    npred3=rf32.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred3)
    
    
    print("\n")
    
    features=np.array([[14.8,1.5,61,72]])
    
    npred4=rf32.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred4)
    
    
    print("\n")
    
    features=np.array([[4.2,2.2,120,125]])
    
    npred5=rf32.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred5)
    
    
    print("\n")
    
    features=([[2.5,1.3,90,95]])
    
    npred6=rf32.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred6)
    
    
    print("\n")
    
    features=([[60,1,45,27]])
    
    npred7=rf32.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred7)
    
    
    print("\n")
    
    features=np.array([[20,1.5,70,65]])
    
    npred8=rf32.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred8)
    
    
    print("\n")
    
    print("\n")

    print("The Random Forest algorithm with hyperparaemeter tuning and undersampling")
    
    print("\n")
    
    print("Y22 test:",Y22_test)
    
    print("\n")
    
    print("Pred 4:",pred4)
    
    
    print("\n")

    print("Confusion Matrix")
    
    conf2=metrics.confusion_matrix(Y22_test,pred4)
    
    cmd2=metrics.ConfusionMatrixDisplay(confusion_matrix=conf2,display_labels=['No','Yes'])
    
    cmd2.plot()
    
    plt.show()
    
    
    print("Training accuracy:",metrics.accuracy_score(Y22_train,predtrain4))
    
    print("Test")
    
    print("Accuracy:",metrics.accuracy_score(Y22_test,pred4))
    
    print("Precision:",metrics.precision_score(Y22_test,pred4,pos_label="P"))
    
    print("Recall:",metrics.recall_score(Y22_test,pred4,pos_label="P"))
    
    print("Sensitivity:",metrics.recall_score(Y22_test,pred4,pos_label="N"))
    
    print("F1 score:",metrics.f1_score(Y22_test,pred4,pos_label="P"))
    
    
    print("\n")

    print("Testing predictive accuracy:")

    
    print("\n")
    
    features=np.array([[3.3,1.8,109,119]])
    
    npred21=rf42.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred21)
    
    
    print("\n")
    
    features=np.array([[2.8,1.7,97,107]])
    
    npred22=rf42.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred22)
    
    
    print("\n")
    
    features=np.array([[45,1.4,39,33]])
    
    npred23=rf42.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred23)
    
    
    print("\n")
    
    features=np.array([[14.8,1.5,61,72]])
    
    npred24=rf42.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred24)
    
    
    print("\n")
    
    features=np.array([[4.2,2.2,120,125]])
    
    npred25=rf42.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred25)
    
    
    print("\n")
    
    features=([[2.5,1.3,90,95]])
    
    npred26=rf42.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred26)
    
    
    print("\n")
    
    features=([[60,1,45,27]])
    
    npred27=rf42.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred27)

    
    print("\n")
    
    features=np.array([[20,1.5,70,65]])
    
    npred28=rf42.predict(features)
    
    print("Features:",features)
    
    print("Prediction:",npred28)
    

    #Creating the Naive Bayes algorithm's pickle files 
    filepath='C:/Users/Admin/Desktop/Project/Model/NaiveBayes1.pkl'
    
    with open(filepath,'wb') as file1:
    
        pickle.dump(gn1, file1)
    
    
    filepath='C:/Users/Admin/Desktop/Project/Model/NaiveBayes2.pkl'
    
    with open(filepath,'wb') as file2:
    
        pickle.dump(gn2, file2)
        
    
    #Creating the KNN algorithm's pickle files
    filepath='C:/Users/Admin/Desktop/Project/Model/KNN1.pkl'
    
    with open(filepath,'wb') as file3:
        
        pickle.dump(knn1, file3)
        
    
    filepath='C:/Users/Admin/Desktop/Project/Model/KNN2.pkl'

    with open(filepath,'wb') as file4:

        pickle.dump(knn2, file4)
        
    
    #Creating the SVM algorithm's pickle files
    filepath='C:/Users/Admin/Desktop/Project/Model/SVM1.pkl'

    with open(filepath,'wb') as file5:

        pickle.dump(cls1, file5)
        
    
    filepath='C:/Users/Admin/Desktop/Project/Model/SVM2.pkl'

    with open(filepath,'wb') as file6:

        pickle.dump(cls2, file6)
        

    #Creating the Decision Tree algorithm's pickle files
    filepath='C:/Users/Admin/Desktop/Project/Model/DecisionTree1.pkl'

    with open(filepath,'wb') as file7:

        pickle.dump(dtree1, file7)
        
    
    filepath='C:/Users/Admin/Desktop/Project/Model/DecisionTree2.pkl'

    with open(filepath,'wb') as file8:

        pickle.dump(dtree2, file8)
        
    
    #Creating the Decision Tree algorithm after hyperparameter tuning's pickle files
    filepath='C:/Users/Admin/Desktop/Project/Model/DTHyperparameter1.pkl'

    with open('DTHyperparameter1.pkl','wb') as file9:

        pickle.dump(dtree32, file9)
        

    filepath='C:/Users/Admin/Desktop/Project/Model/DTHyperparameter2.pkl' 

    with open(filepath,'wb') as file10:

        pickle.dump(dtree42, file10)
        
    
    #Creating the Random Forest algorithm's pickle files
    filepath='C:/Users/Admin/Desktop/Project/Model/RandomForest1.pkl'

    with open(filepath,'wb') as file11:

        pickle.dump(rf1, file11)
        
    
    filepath='C:/Users/Admin/Desktop/Project/Model/RandomForest2.pkl'

    with open(filepath,'wb') as file12:

        pickle.dump(rf2, file12)
        
    
    #Creating the Random Forest algorithm after hyperparameter tuning's pickle files
    filepath='C:/Users/Admin/Desktop/Project/Model/RFHyperparameter1.pkl'
    
    with open(filepath,'wb') as file13:
    
        pickle.dump(rf32, file13)
        
    
    filepath='C:/Users/Admin/Desktop/Project/Model/RFHyperparameter2.pkl'

    with open(filepath,'wb') as file14:

        pickle.dump(rf42, file14)
