import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
import scipy
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning) #remove warning

def load_dataset(path_to_file):
    df=pd.read_csv(path_to_file)
    return df

def standardize(df):
    scaler=StandardScaler()
    df_std=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
    df_std[df.columns[-1]]=df[df.columns[-1]]
    return df_std

def train_test(df):
    X=df.drop(df.columns[-1],axis=1)
    y=df[df.columns[-1]]
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=42)
    return([XTrain,XTest,yTrain,yTest])

def knn_classifier(k,XTrain,yTrain,XTest):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(XTrain, yTrain)
    yPred = knn.predict(XTest)
    return yPred

def percentage_accuracy(yPred,yTest):
    return(accuracy_score(yTest, yPred))

def confusion_matrixp(yPred,yTest):
    return(confusion_matrix(yTest,yPred))

def naive_bayes_classifier(XTrain,yTrain,XTest):
    gnb = GaussianNB()
    gnb.fit(XTrain, yTrain)
    yPred = gnb.predict(XTest)
    return yPred

def pca(df,n):
    x=df.drop(df.columns[-1],axis=1)
    pca = PCA(n_components=n)
    pComps = pca.fit_transform(x)
    pDf = pd.DataFrame(data = pComps)
    pDf[df.columns[-1]]=df[df.columns[-1]]
    return pDf

def bayes(X,y,yName):
     XTrain, XTest, YTrain, YTest =train_test_split(X,y, test_size=0.3, random_state=42,shuffle=True)
     XTrain0=XTrain[XTrain[yName] == 0]
     XTrain1=XTrain[XTrain[yName] == 1]
     meanXTrain0=XTrain0.mean(axis = 0)
     meanXTrain1=XTrain1.mean(axis = 0)
     covXTrain0=XTrain0.cov()
     covXTrain1=XTrain1.cov()
     YPred=[]
     for i in range(len(XTest)):
         p1=multivariate_normal.pdf(XTest.iloc[i], mean=meanXTrain1,cov= covXTrain1,allow_singular=True)
         p0=multivariate_normal.pdf(XTest.iloc[i], mean=meanXTrain0,cov= covXTrain0,allow_singular=True)
         if(p1>p0):
             YPred.append(1)
         else:
             YPred.append(0)
     return YPred

def responsibily(x, w, mean, cov):
    r = 0
    for i in range(len(w)):
        r += w[i] * scipy.stats.multivariate_normal.pdf(x, mean[i], cov[i], allow_singular=True)
    return r

def bayesGMM(k,df):
    df_0 = df[df['Z_Scratch']==0]
    df_1 = df[df['Z_Scratch']==1]
    X_train0, X_test0, y_train0, y_test0 = train_test_data(df_0)
    X_train1, X_test1, y_train1, y_test1 = train_test_data(df_1)

    XTest = np.concatenate((X_test0, X_test1))
    yTest = np.concatenate((y_test0, y_test1))

    gmm = GaussianMixture(n_components=k,reg_covar=1e-4)
    gmm.fit(X_train0)

    gmm2 = GaussianMixture(n_components=k,reg_covar=1e-4)
    gmm2.fit(X_train1)
    
    yPred = []
    for i in XTest:
        res0 = res(i, gmm.weights_, gmm.means_, gmm.covariances_)
        res1 = res(i, gmm2.weights_, gmm2.means_, gmm2.covariances_)
        if res0>res1:
            yPred.append(0)
        else:
            yPred.append(1)
    print("Accuracy for GMM  Bayes Classifier, k = : ",k)
    print(percentage_accuracy(yPred, yTest))
    print(confusion_matrixp(yPred, yTest))
    print()

#knn

df=load_dataset("../inLab/SteelPlateFaults-2class.csv")

dfStd=standardize(df)
XTrain=train_test(dfStd)[0]
XTest=train_test(dfStd)[1]
yTrain=train_test(dfStd)[2]
yTest=train_test(dfStd)[3]
X = df
y = df['Z_Scratch']
yTest = train_test(df)[3]

k = range(1,22,2)
accuracies=[]
confusion=[]
for i in k:
    print("K = ",i)
    yPredKnn=knn_classifier(i,XTrain,yTrain,XTest)
    print(confusion_matrixp(yPredKnn,yTest))
    confusion.append(confusion_matrixp(yPredKnn,yTest))
    print(percentage_accuracy(yPredKnn,yTest),"\n")
    accuracies.append(percentage_accuracy(yPredKnn,yTest))

print("Confusion Matrix by KNN Classifier: \n",sum(confusion)/len(confusion),"\n")
print("Accuracy by KNN Classifier:",sum(accuracies)/len(accuracies))
plt.plot(range(1,22,2),accuracies,color='b')
plt.xlabel("Value of k")
plt.ylabel("Accuracy")
plt.show()

#pca then knn

#l=[i for i in range(1,len(df.columns))]
l=[1,2,3]

for i in l:
    print("VALUE OF N: ",i,"\n")
    pdf = pca(df,i)
    #print(pdf.head())
    
    dfStd=standardize(pdf)
    XTrain=train_test(dfStd)[0]
    XTest=train_test(dfStd)[1]
    yTrain=train_test(dfStd)[2]
    yTest=train_test(dfStd)[3]
    X = df
    y = df['Z_Scratch']
    yTest = train_test(df)[3]

    k = range(1,22,2)
    accuracies=[]
    confusion=[]
    for i in k:
        print("K = ",i)
        yPredKnn=knn_classifier(i,XTrain,yTrain,XTest)
        print(confusion_matrixp(yPredKnn,yTest))
        confusion.append(confusion_matrixp(yPredKnn,yTest))
        print(percentage_accuracy(yPredKnn,yTest),"\n")
        accuracies.append(percentage_accuracy(yPredKnn,yTest))
    #yPredBayes = bayes(X,y)
    print("Confusion Matrix by KNN Classifier: \n",sum(confusion)/len(confusion),"\n")
    #print("Confusion Matrix by Bayes Classifier: \n",confusion_matrixp(yPredBayes,yTest),"\n")
    print("Accuracy by KNN Classifier:",sum(accuracies)/len(accuracies))
    #print("Accuracy by Bayes Classifier :",percentage_accuracy(yPredBayes,yTest))
    plt.plot(range(1,22,2),accuracies,color='b')
    plt.xlabel("Value of k")
    plt.ylabel("Accuracy")
    plt.show()
    print("##############################\n")

#GMM

df=load_dataset("../inLab/SteelPlateFaults-2class.csv")

dfX_test,dfX_train,dfY_test,dfY_train=train_test_data(df)

dfh= pd.concat([dfX_train,dfY_train], axis=1)
dft= pd.concat([dfX_test,dfY_test],axis=1)

ls = [2,4,8,16]
for k in ls:
    bayesGMM(k,df)

#pca then gmm

df=load_dataset("../inLab/SteelPlateFaults-2class.csv")

#l=[i for i in range(1,len(df.columns))]
l=[1,2,3]

for i in l:
    print("VALUE OF N: ",i,"\n")
    pdf = pca(df,i)

    dfX_test,dfX_train,dfY_test,dfY_train=train_test_data(df)

    dfh= pd.concat([dfX_train,dfY_train], axis=1)
    dft= pd.concat([dfX_test,dfY_test],axis=1)

    ls = [2,4,8,16]
    for k in ls:
        bayesGMM(k,df)
    print("##############################\n")

