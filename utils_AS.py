import numpy as np
np.set_printoptions(threshold=10000,suppress=True)
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, auc, precision_score,precision_recall_curve,make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import shap
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
import pickle


SEED=1

classifieurs = {
'DT' : DecisionTreeClassifier(criterion='gini',random_state=SEED),
'kNN' : KNeighborsClassifier(n_neighbors=5,n_jobs=-1),
'MLP' : MLPClassifier(hidden_layer_sizes=(40,20),random_state=SEED)
}

parametres={
    'MLP' : {'hidden_layer_sizes' : [(10),(20),(30),(20,10),(30,10),(30,20),(40,20)],
            'activation' : ['identity', 'logistic', 'tanh', 'relu']
            }
}

def myscore(Ytest,Ypred):
    return (accuracy_score(Ytest,Ypred)+precision_score(Ytest,Ypred))/2

monscore=make_scorer(myscore,greater_is_better=True)

def run_classifieurs_train_test(Xtrain,Xtest,Ytrain,Ytest,classifieurs):
    for i in classifieurs:
        clf=classifieurs[i]
        clf.fit(Xtrain,Ytrain)
        Ypred=clf.predict(Xtest)

        print("###### {0} ######".format(i))
        print(confusion_matrix(Ytest,Ypred))
        accuracy=accuracy_score(Ytest,Ypred)
        precision=precision_score(Ytest,Ypred)
        moyenne=(accuracy+precision)/2
        print("Accuracy : {0:.2f}%, Precision : {1:.2f}%, Moyenne : {2:.2f}%".format(accuracy*100,precision*100, moyenne*100))    


def importances_variables(Xtrain,Ytrain,features):
    RF = RandomForestClassifier(n_estimators=1000,random_state=SEED,n_jobs=-1) 
    RF.fit(Xtrain, Ytrain)
    importances=RF.feature_importances_
    std = np.std([tree.feature_importances_ for tree in RF.estimators_],axis=0)
    sorted_idx = np.argsort(importances)[::-1]
    padding = np.arange(Xtrain.size/len(Xtrain)) + 0.5
    plt.barh(padding, importances[sorted_idx],xerr=std[sorted_idx], align='center') 
    plt.yticks(padding, features[sorted_idx])
    plt.xlabel("Relative Importance")
    plt.title("Variable Importance")
    plt.show()
    return RF,sorted_idx

def explicabilite_variables(clf,Xtest,features):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(Xtest)
    shap.summary_plot(shap_values[:,:,1], Xtest,feature_names=features)

def evolution_features(Xtrain,Xtest,Ytrain,Ytest,clf,sorted_idx,score):
    scores=np.zeros(Xtrain.shape[1]) 
    for f in np.arange(0, Xtrain.shape[1]): 
        X1_f = Xtrain[:,sorted_idx[:f+1]]
        X2_f = Xtest [:,sorted_idx[:f+1]]
        clf.fit(X1_f,Ytrain)
        Ypred=clf.predict(X2_f)
        scores[f]=np.round(score(Ytest,Ypred),3)
    plt.plot(scores)
    plt.xlabel("Nombre de Variables")
    plt.ylabel("Score")
    plt.title("Evolution de votre score en fonction des variables")
    plt.show()
    selected_features=sorted_idx[:np.argmax(scores)+1]

    return scores,selected_features


def recherche_parametres(clf,param,scoring,X,Y):
    GS=GridSearchCV(clf,param,cv=5,n_jobs=-1,scoring=scoring)
    GS.fit(X,Y)
    return GS.best_estimator_


def creation_pipeline(classifieur,nbfeatures,X,Y):
    P=Pipeline([('SS',StandardScaler()),
                ('RFE',RFE(RandomForestClassifier(n_estimators=1000,n_jobs=-1,random_state=SEED),n_features_to_select=nbfeatures)),
                ('clf', classifieur)
    ])

    P.fit(X,Y)
    pickle.dump(P,open('./Models/model_banque.pkl','wb'))

def model_final(model,X,threshold):
    return model.predict_proba(X)[:,1]>=threshold


def recherche_meilleur_seuil_decision(X,Y,clf):
    prediction_proba_1=clf.predict_proba(X)[:,1]
    precision,recall,threshold=precision_recall_curve(Y,prediction_proba_1)
    scores = np.array([myscore(Y,model_final(clf,X,t)) for t in threshold])
    best_index = np.argmax(scores)
    best_threshold = threshold[best_index]
    best_score = scores[best_index]
    plt.plot(threshold, precision[:-1], label='Precision(y=1)')
    plt.plot(threshold, recall[:-1], label='Rappel(y=1)')
    plt.plot(threshold, scores, label='Mon score')
    plt.scatter(best_threshold,best_score,zorder=5)
    plt.axvline(best_threshold,linestyle='--',label=f'Seuil optimal = {best_threshold:.3f}')

    plt.annotate(
        f'Score max = {best_score:.3f}',
        xy=(best_threshold, best_score),
        xycoords='data',
        textcoords='offset points',
        xytext=(10, 10),
        arrowprops=dict(arrowstyle='->')
    )

    plt.xlabel("Seuil de décision")
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid(True)
    plt.show()
    return best_threshold
