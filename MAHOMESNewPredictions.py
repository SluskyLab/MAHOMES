# Takes a saved dataframe of calculated features for site(s) and uses MAHOMES to make Enzyme or Non-Enzyme prediction
# requires name of job which matches folder in data containing calculated feature file

# libraries
import numpy as np
import pandas as pd
import joblib

# scale features
from sklearn import preprocessing
from sklearn import impute
# classifier
from sklearn.ensemble import ExtraTreesClassifier

# custom scripts
import sys
sys.path.insert(0, "%s" % "CV/")
import GetFeatureSet as GetFeatureSet

job_name = str(sys.argv[1])

def scale_features(feat_df):
     ## load saved scalers and imputers which were fit to data-set
     cont_scaler = joblib.load("ModelsForMAHOMES/ContVarScaler.pkl")
     cont_imputer = joblib.load("ModelsForMAHOMES/ContVarImpute.pkl")
     ctg_scaler = joblib.load("ModelsForMAHOMES/CtgVarScaler.pkl")
     ctg_imputer = joblib.load("ModelsForMAHOMES/CtgVarImpute.pkl")
     
     #split for scaling into categorical and not categorical
     not_ctg_geom = ("geom_gRMSD", "geom_MaxgRMSDDev","geom_val", "geom_nVESCUM","geom_AtomRMSD", "geom_AvgO", "geom_AvgN", "geom_AvgS", "geom_AvgOther", "geom_Charge")
     geom = [name for name in feat_df if name.startswith("geom")]
     ctg_data = [x for x in geom if not x in not_ctg_geom]

     # scale and fill in missing cont. features
     cont_feat = feat_df[feat_df.columns.difference(ctg_data)]
     feat_scaled = pd.DataFrame(cont_scaler.transform(cont_feat), columns=cont_feat.columns, index=cont_feat.index)
     feat_scaled = pd.DataFrame(cont_imputer.transform(feat_scaled), columns=feat_scaled.columns, index=feat_scaled.index)

     # scale and fill in missing discrete features
     if len(feat_df.columns.intersection(ctg_data)) > 0:
          dis_feat = feat_df[feat_df.columns.intersection(ctg_data)]
          dis_feat_scld = pd.DataFrame(ctg_scaler.fit_transform(dis_feat), columns=dis_feat.columns, index=dis_feat.index)
          dis_feat_scld = pd.DataFrame(ctg_imputer.transform(dis_feat_scld), columns=dis_feat_scld.columns, index=dis_feat_scld.index)
          feat_scaled = pd.merge(feat_scaled, dis_feat_scld, left_index=True, right_index=True)
          
     return(feat_scaled)

## number of iterations to improve reproducability
num_rand_seeds = 10 # needs to match number used in MAHOMES_eval_T-metal-site notebook
def make_predictions(feat_df):
     feat_df = scale_features(feat_df)
     # get correct features in feature set used by MAHOMES
     X = GetFeatureSet.feature_subset(feat_df, "AllMeanSph", noBSA=True)

     # load and use MAHOMES classifiers (different random seeds for clf and under-sampling)
     site_preds =  pd.DataFrame(index=feat_df.index) #{'actual': pd.Series("NaN", index=feat_df.index)}
     for rand_seed in range(0,num_rand_seeds):
          clf = joblib.load("ModelsForMAHOMES/MAHOMES%s.pkl"%(rand_seed))
          new_preds = clf.predict(X)
          site_preds['prediction_%s'%(rand_seed)]= pd.Series(new_preds, index=feat_df.index)
     
     ## calcualte the average of all random seed predictions
     # #site_preds = pd.DataFrame(site_preds)
     site_preds['prediction']=0
     for rand_seed in range(0,num_rand_seeds):
          site_preds['prediction']+=site_preds['prediction_%s'%(rand_seed)] 
     site_preds['prediction']=site_preds['prediction']/num_rand_seeds
          
     ## make final prediction
     site_preds['final_prediction']="Non-Enzyme"
     site_preds.loc[site_preds['prediction']>=0.5, 'final_prediction']="Enzyme"
     return(site_preds)

site_features = pd.read_csv("../data/%s/sites_calculated_features.txt"%(job_name))
site_features = site_features.set_index('SITE_ID',drop=True)
site_predictions = make_predictions(site_features)
site_predictions.to_csv("../data/%s/sites_predictions.txt"%(job_name))

