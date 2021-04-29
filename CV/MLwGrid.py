# Adapted, non-cluster, version of script for finding optimal ML hyper-parameters 
# for a give ML algorithm, feature set, and optimization metric. All combinations tested 
# script requires following parameters:
# 1. 0-13 which index corresponding ML algorithms
# 2. feature set name (complete list of options can be found in publications supplement)
# 3. optimization metric (Prec, Acc, MCC, Multi)
# This script was tested using:
# > python MLwGrid.py 0 AllSumSph MCC
# All inputs in MLCombosAll.txt were used for publication

#general requirements
import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings(action="ignore")

#preprocessing stuff
from sklearn.model_selection import train_test_split, GridSearchCV, GroupShuffleSplit, StratifiedShuffleSplit, cross_validate, StratifiedKFold
from sklearn import preprocessing
from sklearn import impute
#classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
#process results
from sklearn.metrics import roc_curve, auc, recall_score, accuracy_score, precision_score, confusion_matrix, make_scorer, matthews_corrcoef, jaccard_score

# custom script
import GetFeatureSet as GetFeatureSet

# correspinding ML algorithm for a given index
names = ["LogRegr", "Ridge", "PassAggr",
        "QDA", "LDA", "NaiveBayes",
        "NearNeigh",
        "LinSVM", "RBFSVM", "SigSVM",
        "RandomForest", "ExtraTrees", "GradBoost", 
        "NeurNet"]

classifiers = [
    LogisticRegression(solver="liblinear", tol = 1e-4), 
    RidgeClassifier(solver="auto", tol = 1e-4),
    PassiveAggressiveClassifier(tol = 1e-4),
    QuadraticDiscriminantAnalysis(priors=None), #also almost no real parameters to adjust
    LinearDiscriminantAnalysis(solver = "lsqr"),
    GaussianNB(priors=None), #only one real parameter to adjust! 
    KNeighborsClassifier(algorithm = "ball_tree", weights = 'distance'), 
    SVC(kernel="linear", max_iter=10000, tol = 1e-4), 
    SVC(kernel="rbf", class_weight = "balanced", tol = 1e-4), 
    SVC(kernel="sigmoid", class_weight = "balanced", tol = 1e-4),
    RandomForestClassifier(class_weight = "balanced", bootstrap = True, n_estimators=500, max_features='sqrt', criterion='entropy'),
    ExtraTreesClassifier(n_estimators=500, min_samples_split=3, max_depth=None, criterion="gini"),
    GradientBoostingClassifier(criterion = 'friedman_mse', min_samples_split = 3, n_estimators=1000, loss='deviance'),
    MLPClassifier(learning_rate_init = 0.01, activation='relu'), 
]

parameter_space = [
    { "penalty": ["l2", 'l1'], "C":[1.0, 0.01, 0.001], "class_weight":['balanced', None]}, #LogRegr
    { "alpha": np.logspace(-4, 1, 6), "class_weight":['balanced', None]}, #Ridge 
    { "C": np.logspace(-1, 3, 5), "class_weight":['balanced', None] }, #PassAggr
    { "reg_param": np.linspace(0.5, 1, 7) }, #QDA
    { "shrinkage": ["auto", 0, 0.1, 0.25, 0.5, 0.75, 1] }, #LDA
    { "var_smoothing": np.logspace(-9, 0, 10) }, #Gauss
    [ {"metric": ["minkowski"], "p":[2, 3], "n_neighbors": [5, 8, 10, 15]}, {"metric": ["chebyshev"], "n_neighbors": [5, 8, 10, 15]} ], #kNN
    { "C": np.logspace(-3, 1, 5), "class_weight":['balanced', None] }, #SVC lin
    { "C": np.logspace(-3, 1, 5), "gamma": ["scale", "auto"] }, #SVC rbf
    { "C": np.logspace(-3, 1, 5),"gamma": ["scale", "auto"] }, #SVC sig
    { "max_depth": [6, 10, 20, None], 'min_samples_split': [5, 25, 50] }, #RF
    { "class_weight": ["balanced", None], "max_features": ['log2', None], "bootstrap" : [True, False] }, #ExtraTrees
    { "learning_rate": [0.1, 0.01], "max_features" : ['log2', None], "subsample":[0.5, 0.65, 0.8]},#GBClassifier
    { "hidden_layer_sizes": [(50,), (100,), (200,)], "alpha": [0.1, 0.01, 0.001] } #MLPClass
]



name_index = int(sys.argv[1]) # number for given ML algorithm
feature_set = str(sys.argv[2]) #All_Sph, All_Shell, Gen, etc
opt_type = str(sys.argv[3]) #Prec Acc MCC Multi

name = names[name_index]


## read ion all sites/features
sites = pd.read_csv("../../data/publication_sites/sites_calculated_features_scaled.txt", sep=',')
sites = sites.set_index('SITE_ID',drop=True)

## get training/kfold sites, random under sample, and split out target value ("Catalytic")
X = sites.loc[sites.Set == "data"].copy()
X_Cat = X[X['Catalytic']==True]
X_nonCat = X[X['Catalytic']==False]
# the following line controls under sampling
X_nonCat = X_nonCat.sample(n=len(X_Cat)*3, axis=0, random_state=1)
X = X_Cat.append(X_nonCat)
y = X['Catalytic']; del X['Catalytic']

## get test sites and split out target value ("Catalytic")
testX = sites.loc[sites.Set == "test"].copy()
testY = testX['Catalytic']; del testX['Catalytic']

#split into features and classification
X = GetFeatureSet.feature_subset(X, feature_set, noBSA=True)
print("DataSet entries: %s \t features: %s"%(X.shape[0], X.shape[1]))
testX = GetFeatureSet.feature_subset(testX, feature_set, noBSA=True)
print("TestSet entries: %s \t features: %s"%(testX.shape[0], testX.shape[1]))

def setDisplay(X, x, Y, y):
    print("\nTRAIN entries: %s \t features: %s"%(X.shape[0], X.shape[1]))
    print("\tNum catalytic: %s \n\tNum non-catalytic: %s"%(len(Y[Y==1]),len(Y[Y==0])))
    print("CV entries: %s \t features: %s"%(x.shape[0], x.shape[1]))
    print("\tNum catalytic: %s \n\tNum non-catalytic: %s"%(len(y[y==1]),len(y[y==0])))
 

this_clf = classifiers[name_index]
num_jobs = 15
inner_cv_type = StratifiedShuffleSplit(n_splits=7)
these_params = parameter_space[name_index]


def prec_score_custom(y_true, y_pred, this_label = True):
    return( precision_score(y_true, y_pred, pos_label= this_label) )

def mcc_score(y_true, y_pred):
    return( matthews_corrcoef(y_true, y_pred))
def jac_score(y_true, y_pred, this_label = True):
    return( jaccard_score(y_true, y_pred, pos_label=this_label))


if opt_type == "Prec": 
    this_scoring = make_scorer(prec_score_custom, greater_is_better = True)
elif opt_type == "Acc":
    this_scoring = "accuracy"
elif opt_type == "MCC":
    this_scoring = make_scorer(mcc_score, greater_is_better = True)
elif opt_type == "Multi":
    this_scoring = {"Acc":'accuracy', "MCC": make_scorer(mcc_score, greater_is_better = True), "Jaccard": make_scorer(jac_score, greater_is_better = True) }
else:
    print("Invalid scoring term")
    sys.exit()

outer_cv_type=StratifiedKFold(n_splits=7)
outer_cv_results = []
outer_coeffs = []
outer_params = []
outer_feat_imp = []
for i, (train_idx, test_idx) in enumerate(outer_cv_type.split(X,y)):
    print("OUTER LOOP NUMBER:", i)
    X_train, X_outerCV = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_outerCV = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    print("outer CV display:")
    setDisplay(X_train, X_outerCV, y_train, y_outerCV)

    print("post add_oversampling display:")
    setDisplay(X_train, X_outerCV, y_train, y_outerCV)
    
    #run feature selection and CV
    clf = GridSearchCV(estimator = this_clf, cv=inner_cv_type, param_grid = these_params, scoring = this_scoring, iid = True, refit = False, verbose=100, n_jobs = num_jobs)

    clf.fit(X_train.reset_index(drop=True), y_train.reset_index(drop=True))
    results = clf.cv_results_
    
    #somehow get best combination of multiple scoring terms
    print(results)
    ranks = []
    for key in results:
        if "rank_test_" in key:
            ranks.append(results[key])
    #best params will have to be identified for full data set for final model building after best model is selected
    best_params = results['params'][np.argmin(np.sum(np.asarray(ranks), axis = 0))]
    print(best_params)
    outer_params.append(best_params)
    
    ## set the new classifier to these parameters
    outer_clf = this_clf.set_params(**best_params)
    ## fit on all training data - this is what GridSearchCV(refit = True) will do anyways,
    ## but its selection of params is not necessary Meghans
    outer_clf.fit(X_train.reset_index(drop=True), y_train.reset_index(drop=True))
    
    outerCV = pd.DataFrame(y_outerCV, columns=['Catalytic'])
    
    #predict based on fitted outer CV model
    outerCV_preds =  pd.DataFrame(outer_clf.predict(X_outerCV.reset_index(drop=True)), columns=['Prediction'])
    outerCV_preds['SITE_ID']=X_outerCV.index
    outerCV_preds = outerCV_preds.set_index('SITE_ID', drop=True)
    
    outerCV = pd.merge(outerCV, outerCV_preds, left_index=True, right_index=True)
    
    ## calculate stats
    accuracy = accuracy_score(outerCV.Catalytic, outerCV.Prediction)
    recall = recall_score(outerCV.Catalytic, outerCV.Prediction) 
    precision = precision_score(outerCV.Catalytic, outerCV.Prediction)
    true_neg_rate = len( outerCV[(outerCV.Catalytic == 0) & (outerCV.Prediction == 0)] )/ len(outerCV[(outerCV.Catalytic == 0)])
    mcc = matthews_corrcoef(outerCV.Catalytic, outerCV.Prediction)
    dist_rand = (recall + -1*(1-true_neg_rate)) / np.sqrt(2)
    TN, FP, FN, TP = confusion_matrix(outerCV.Catalytic, outerCV.Prediction ).ravel()
    outer_cv_results.append([ accuracy, precision, recall, true_neg_rate, mcc, dist_rand, TP, TN, FP, FN ])
    print(mcc, accuracy, recall, dist_rand)
    
## check that all hyperparmeters match up for each inner CV run => model stability
outer_params_df = pd.DataFrame(outer_params)
outer_params_df.to_csv("kfold/params/%s_%s_%s_params.txt"%(name, feature_set, opt_type))

stable = True
if max(outer_params_df.nunique())>1:
    stable = False

scores_filename = "kfold/scores/%s_%s_%s_scores.txt"%(name, feature_set, opt_type)
with open(scores_filename, "w+") as outData: 
    outData.write("Classifier\tDataSet\tOptType\tstable\tAccuracy\tPrecision\tRecall\tTrueNegRate\tMCC\tDistRand\tTP\tTN\tFP\tFN\tAccuracy_std\tPrecision_std\tRecall_std\tTrueNegRate_std\tMCC_std\tDistRand_std\t\TN_std\tTN_std\tFP_std\tFN_std\n")
    outData.write("\n")
    outData.write("%s\t%s\t%s\t%s\t"%(name, feature_set, opt_type, stable)+"\t".join(map(str, np.mean(outer_cv_results, axis = 0)))+"\t"+"\t".join(map(str, np.std(outer_cv_results, axis = 0))))

try:
    outer_cv_results_df = pd.DataFrame(outer_cv_results)
    outer_cv_results_df.to_csv("kfold/allScores/%s_%s_%s_allScores.txt"%(name, feature_set, opt_type))
except:
    hmm=33
overall_best_params={}
for param in outer_params_df.columns:
    most_pop_param_val = outer_params_df[param].iloc[0]
    for cur_param_val in outer_params_df[param].unique():
        num_most_vals = len(outer_params_df[outer_params_df[param]==most_pop_param_val])
        num_cur_vals = len(outer_params_df[outer_params_df[param]==cur_param_val])
        if num_cur_vals>num_most_vals:
            most_pop_param_val=cur_param_val
    
    overall_best_params[param]=most_pop_param_val
print(overall_best_params)