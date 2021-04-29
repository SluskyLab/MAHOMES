
import pandas as pd
import numpy as np


def feature_subset(df, subset, other_bad_terms = [], noBSA=False):
    X = df.copy()
    X = X.fillna(0)
    not_needed = ("Catalytic", "SITE_ID", "Set", "ValidSet", 'NewSet', 'cath_class', 'cath_arch', 'scop_class', 'scop_fold', 'ECOD_arch', 'ECOD_x_poshom', 'ECOD_hom')
    X = X.drop(columns = [term for term in X if term.startswith(not_needed)])
    bad_terms = ("hbond_lr_", 'dslf_fa13', 'pro_close', 'ref', 'fa_sol_', 'MetalCodes', 'MetalAtoms', 'SEPocket', 'geom_gRMSD', 'geom_MaxgRMSDDev', 'geom_AtomRMSD')
    X = X.drop(columns = [term for term in X if term.startswith(bad_terms)])
    #print(X.shape)#, list(X))
    #general terms
    gen_set = ['Depth', 'Vol', "SITEDistCenter", "SITEDistNormCenter"]
    gen_terms = ("BSA", 'SASA')
    all_gen_set = [ term for term in X if term.startswith(gen_terms) ]
    gen_shell = [name for name in all_gen_set if "_S" in name]
    gen_sph = list(set(all_gen_set).difference(gen_shell))
    gen_shell += gen_set
    gen_shell += ["BSA_3.5", "SASA_3.5"]
    gen_sph += gen_set
    all_gen_set += gen_set
    all_gen_set = sorted(set(all_gen_set))
    #Rosetta terms only
    ros_sum_sph0 = list(set([name for name in X if name.endswith("_Sum_3.5")]).difference(all_gen_set))
    ros_sum_sph1 = list(set([ name for name in X if name.endswith("_Sum_5") ]).difference(all_gen_set))
    ros_sum_sph2 = list(set([ name for name in X if name.endswith("_Sum_7.5") ]).difference(all_gen_set))
    ros_sum_sph3 = list(set([ name for name in X if name.endswith("_Sum_9") ]).difference(all_gen_set))
    ros_sum_shell1 = list(set([ name for name in X if name.endswith("_Sum_S5") ]).difference(all_gen_set))
    ros_sum_shell2 = list(set([ name for name in X if name.endswith("_Sum_S7.5") ]).difference(all_gen_set))
    ros_sum_shell3 = list(set([ name for name in X if name.endswith("_Sum_S9") ]).difference(all_gen_set))
    ros_sum_shell = ros_sum_sph0 + ros_sum_shell1 + ros_sum_shell2 + ros_sum_shell3
    ros_sum_sph = ros_sum_sph0 + ros_sum_sph1 + ros_sum_sph2 + ros_sum_sph3
    
    ros_mean_sph0 = list(set([name for name in X if name.endswith("_Mean_3.5")]).difference(all_gen_set))
    ros_mean_sph1 = list(set([ name for name in X if name.endswith("_Mean_5") ]).difference(all_gen_set))
    ros_mean_sph2 = list(set([ name for name in X if name.endswith("_Mean_7.5") ]).difference(all_gen_set))
    ros_mean_sph3 = list(set([ name for name in X if name.endswith("_Mean_9") ]).difference(all_gen_set))
    ros_mean_shell1 = list(set([ name for name in X if name.endswith("_Mean_S5") ]).difference(all_gen_set))
    ros_mean_shell2 = list(set([ name for name in X if name.endswith("_Mean_S7.5") ]).difference(all_gen_set))
    ros_mean_shell3 = list(set([ name for name in X if name.endswith("_Mean_S9") ]).difference(all_gen_set))
    ros_mean_shell = ros_mean_sph0 + ros_mean_shell1 + ros_mean_shell2 + ros_mean_shell3
    ros_mean_sph = ros_mean_sph0 + ros_mean_sph1 + ros_mean_sph2 + ros_mean_sph3
    
    electro = [name for name in X if name.startswith("Elec")]
    geom = [name for name in X if name.startswith("geom")]
    findgeo_geoms = ("lin", "trv", "tri", "tev", "spv", 
        "tet", "spl", "bva", "bvp", "pyv", 
        "spy", "tbp", "tpv", 
        "oct", "tpr", "pva", "pvp", "cof", "con", "ctf", "ctn",
        "pbp", "coc", "ctp", "hva", "hvp", "cuv", "sav",
        "hbp", "cub", "sqa", "boc", "bts", "btt", 
        "ttp", "csa")
    geom = [name for name in geom if not name.endswith(findgeo_geoms)] #remove the individual geom types
    #pocket features only
    pocket_set = ['Depth', 'Vol', "SITEDistCenter", "SITEDistNormCenter", 'LongPath', 'farPtLow', 'PocketAreaLow', 'OffsetLow', 'LongAxLow', 'ShortAxLow', 'farPtMid', 'PocketAreaMid', 'OffsetMid', 'LongAxMid', 'ShortAxMid', 'farPtHigh', 'PocketAreaHigh', 'OffsetHigh', 'LongAxHigh', 'ShortAxHigh']
    pocket_set = list(set(pocket_set).difference(other_bad_terms))
    #pocket lining only
    lining_set = ['num_pocket_bb', 'num_pocket_sc', 'avg_eisen_hp', 'min_eisen', 'max_eisen', 'skew_eisen', 'std_dev_eisen', 'avg_kyte_hp', 'min_kyte', 'max_kyte', 'skew_kyte', 'std_dev_kyte', 'occ_vol', 'NoSC_vol', 'SC_vol_perc', 'LiningArea']
    lining_set = list(set(lining_set).difference(other_bad_terms))
    
    #print( len(all_gen_set), len(sorted(set(ros_sum_sph+ros_sum_shell+ros_mean_shell+ros_mean_sph))), len(electro), len(geom), len(pocket_set), len(lining_set))
    #print(len(sorted(set(ros_sum_sph+ros_sum_shell+ros_mean_shell+ros_mean_sph+all_gen_set+electro+geom+pocket_set+lining_set))))
    
    subset_list = ["AllSumSph", "AllMeanSph", "AllSumShell", "AllMeanShell", 
                    "GenSph", "GenShell", "Pocket", "Lining", 
                    'RosSumSph', 'RosSumSph0', 'RosSumSph1', 'RosMeanSph', 'RosMeanSph0', 'RosMeanSph1', "RosSumSphInner2", "RosMeanSphInner2",
                    'RosSumShell', 'RosSumShell1', 'RosMeanShell', 'RosMeanShell1',"RosSumShellInner2", "RosMeanShellInner2",
                    "LinPocket", "LinRosSumSph", "LinRosMeanSph", "LinRosSumShell", "LinRosMeanShell",
                    "PocketRosSumSph", "PocketRosMeanSph", "PocketRosSumShell", "PocketRosMeanShell",
                    "Geom", "LinPocketGeom", "GeomElectro", "GeomRosSumSph", "GeomRosSumShell", "GeomRosMeanSph", "GeomRosMeanShell",
                    
                    "Electro", "LinPocketElectro", "LinPocketElectroGeom", "ElectroRosSumSph", "ElectroRosSumShell", "ElectroRosMeanSph", "ElectroRosMeanShell", 
                    "AllSumSphMinusGen", "AllSumSphMinusLin", "AllSumSphMinusPocket", 
                    "AllSumSphMinusGeom", "AllSumSphMinusElectro", 
                    "AllMeanSphMinusGen", "AllMeanSphMinusLin", "AllMeanSphMinusPocket", 
                    "AllMeanSphMinusGeom", "AllMeanSphMinusElectro", "AllMinusRosSph",
                    
                    "AllSumShellMinusGen", "AllSumShellMinusLin", "AllSumShellMinusPocket", 
                    "AllSumShellMinusGeom", "AllSumShellMinusElectro", 
                    "AllMeanShellMinusGen", "AllMeanShellMinusLin", "AllMeanShellMinusPocket", 
                    "AllMeanShellMinusGeom", "AllMeanShellMinusElectro", "AllMinusRosShell",
                    ]
    column_subsets = [  sorted(set(gen_sph+ros_sum_sph+pocket_set+lining_set+electro+geom)),#AllSumSph  GSP
                        sorted(set(gen_shell+ros_mean_sph+pocket_set+lining_set+electro+geom)),#AllMeanSph  GSP
                        sorted(set(gen_sph+ros_sum_shell+pocket_set+lining_set+electro+geom)),#AllSumShell  GSH
                        sorted(set(gen_shell+ros_mean_shell+pocket_set+lining_set+electro+geom)), #AllMeanShell GSH
                        gen_sph,#GenSph GSH
                        gen_shell,#GenShell GPH
                      
                        pocket_set,#Pocket
                        lining_set,#Lining
                        ros_sum_sph,#RosSumSph
                        ros_sum_sph0,#RosSumSph0
                        ros_sum_sph1,#RosSumSph1
                        ros_mean_sph,#RosMeanSph
                        ros_mean_sph0,#RosMeanSph0
                        ros_mean_sph1,#RosMeanSph1
                        sorted(set(ros_sum_sph0+ros_sum_sph1)),#RosSumSphInner2
                        sorted(set(ros_mean_sph0+ros_mean_sph1)),#RosMeanSphInner2
                        ros_sum_shell, #RosSumShell
                        ros_sum_shell1, #RosSumShell1
                        ros_mean_shell, #RosMeanShell
                        ros_mean_shell1, #RosMeanShell1
                        sorted(set(ros_sum_sph0 + ros_sum_shell1)), #RosSumShellInner2
                        sorted(set(ros_mean_sph0 + ros_mean_shell1)), #RosMeanShellInner2
                        lining_set+pocket_set, #LinPocket
                        lining_set+ros_sum_sph, #LinRosSumSph
                        lining_set+ros_mean_sph, #LinRosMeanSph
                        lining_set+ros_sum_shell, #LinRosSumShell
                        lining_set+ros_mean_shell, #LinRosMeanShell
                        pocket_set+ros_sum_sph, #PocketRosSumSph
                        pocket_set+ros_mean_sph, #PocketRosMeanSph
                        pocket_set+ros_sum_shell, #PocketRosSumShell
                        pocket_set+ros_mean_shell, #PocketRosMeanShell
                        geom, #Geom
                        lining_set+pocket_set+geom, #LinPocketGeom
                        geom+electro, #GeomElectro
                        geom+ros_sum_sph, #GeomRosSumSph
                        geom+ros_sum_shell,#GeomRosSumShell
                        geom+ros_mean_sph, #GeomRosMeanSph
                        geom+ros_mean_shell,#GeomRosMeanShell
                      
                        electro,#Electro
                        lining_set+pocket_set+electro,#LinPocketElectro
                        lining_set+pocket_set+electro+geom,#LinPocketElectroGeom
                        electro+ros_sum_sph,#ElectroRosSumSph
                        electro+ros_sum_shell,#ElectroRosSumShell
                        electro+ros_mean_sph,#ElectroRosMeanSph
                        electro+ros_mean_shell,#ElectroRosMeanShell
                        sorted(set(ros_sum_sph+pocket_set+lining_set+electro+geom)),#AllSumSphMinusGen
                        sorted(set(gen_sph+ros_sum_sph+pocket_set+electro+geom)),#AllSumSphMinusLin   GSP
                        sorted(set(gen_sph+ros_sum_sph+lining_set+electro+geom)),#AllSumSphMinusPocket  GSP
                        sorted(set(gen_sph+ros_sum_sph+pocket_set+lining_set+electro)),#AllSumSphMinusGeom  GSP
                        sorted(set(gen_sph+ros_sum_sph+pocket_set+lining_set+geom)), #AllSumSphMinusElectro  GSP
                        sorted(set(ros_mean_sph+pocket_set+lining_set+electro+geom)),#AllMeanSphMinusGen
                        sorted(set(gen_sph+ros_mean_sph+pocket_set+electro+geom)),#AllMeanSphMinusLin   GSP
                        sorted(set(gen_sph+ros_mean_sph+lining_set+electro+geom)),#AllMeanSphMinusPocket  GSP
                        sorted(set(gen_sph+ros_mean_sph+pocket_set+lining_set+electro)),#AllMeanSphMinusGeom  GSP
                        sorted(set(gen_sph+ros_mean_sph+pocket_set+lining_set+geom)),#AllMeanSphMinusElectro  GSP
                        sorted(set(gen_sph+pocket_set+lining_set+electro+geom)),#AllMinusRosSph  GSP
                      
                        sorted(set(ros_sum_shell+pocket_set+lining_set+electro+geom)),#AllSumShellMinusGen
                        sorted(set(gen_shell+ros_sum_shell+pocket_set+electro+geom)),#AllSumShellMinusLin  GSH
                        sorted(set(gen_shell+ros_sum_shell+lining_set+electro+geom)),#AllSumShellMinusPocket  GSH
                        sorted(set(gen_shell+ros_sum_shell+pocket_set+lining_set+electro)),#AllSumShellMinusGeom  GSH
                        sorted(set(gen_shell+ros_sum_shell+pocket_set+lining_set+geom)), #AllSumShellMinusElectro  GSH
                        sorted(set(ros_mean_shell+pocket_set+lining_set+electro+geom)),#AllMeanShellMinusGen  
                        sorted(set(gen_shell+ros_mean_shell+pocket_set+electro+geom)),#AllMeanShellMinusLin  GSH
                        sorted(set(gen_shell+ros_mean_shell+lining_set+electro+geom)),#AllMeanShellMinusPocket  GSH
                        sorted(set(gen_shell+ros_mean_shell+pocket_set+lining_set+electro)), #AllMeanShellMinusGeom  GSH
                        sorted(set(gen_shell+ros_mean_shell+pocket_set+lining_set+geom)),#AllMeanShellMinusElectro   GSH
                        sorted(set(gen_shell+pocket_set+lining_set+electro+geom)),#AllMinusRosShell  GSH
                        ]
    #print(column_subsets[subset_list.index(data_subset)] )
    if subset in subset_list:
        X = X[ column_subsets[subset_list.index(subset)] ] 
    else:
        print("Not a subset in list; defaulting to AllSph")
        X = X[ column_subsets[0] ] #this is all for usage with PCA/UMAP; it uses the rosetta sphere terms plus all the non-rosetta terms
    if 'groupID' in df.columns:
        X=pd.merge(X,df['groupID'], left_index=True, right_index=True)
    ## added to remove BSA terms for undersampling DataSet using BSA
    if noBSA==True:
        X = X.drop(columns = [term for term in X if term.startswith("BSA")])
        X = X.drop(columns = [term for term in X if term.startswith("SASA")])    
    return(X)
