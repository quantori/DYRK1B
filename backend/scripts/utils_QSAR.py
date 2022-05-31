import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score

import utils_opt

#%% QSAR defs:
def build_QSAR( df_train_fps, y_NAME = "Potency", runRFEselector = True, clf_type = 'NuSVR', iter_MAX = 0 ):
    '''Presumes that one of the inputs (df_train) contains fields ['Structure','ID','Potency']
       iter_MAX == 0 or 1 would result in only ONE iteration!
    '''
    CONVERGED = False
    r2_BEST   = 0
    iter_CURRENT = 0
    
    while not CONVERGED:
        iter_CURRENT += 1
        print("\nFeat_Sel Optimization: Iter#", iter_CURRENT)
        
        df_train_fps_SEL, model_SEL = SelectFeatures(
            df_train_fps.drop(["Structure", "ID"], axis=1), 
            y_name = y_NAME, 
            VarThresh=0, 
            thresh_Rank=0.005, # thresh_Rank=0.005,
            runRFEselector=runRFEselector 
        ) 
        
        MB = BuildModel( df_train_fps_SEL, y_name = y_NAME, clf_type = clf_type, opt_type='Optimized' ) # 'Default' 'Optimized' 'OptimizedWithFeatureSelection'
        
        r2_RFE = model_SEL['rank_vs_perf']['best']['r2']
        r2_opt = MB['clf_summary']['clf_opt_score']
        r2_CURRENT = 0.5 * ( r2_RFE + r2_opt )
                
        if iter_MAX <= 1: # use whatever 1st iter gives
            CONVERGED = True
            BEST_model_SEL = model_SEL
            BEST_df_train_fps_SEL = df_train_fps_SEL.copy()
            BEST_MB = MB
            
        else: # check if obtained better model
            print("\n  r2_CURRENT =", r2_CURRENT)
            if r2_CURRENT > r2_BEST:
                r2_BEST = r2_CURRENT
                BEST_model_SEL = model_SEL
                BEST_df_train_fps_SEL = df_train_fps_SEL.copy()
                BEST_MB = MB
                print("  r2_BEST =", r2_BEST)
                
            if iter_CURRENT >= iter_MAX:
                CONVERGED = True
                
    print("\n    BEST CONVERGED r2_MEAN =", round(r2_BEST, 4) )
    print("    BEST Nfeats set =", int( BEST_model_SEL['rank_vs_perf']['best']['Nfeats']) )
    print("    BEST r2_RFE =", round(BEST_model_SEL['rank_vs_perf']['best']['r2'], 4) )
    print("    BEST r2_opt =", round(BEST_MB['clf_summary']['clf_opt_score'], 4) )
    
    # put back ["Structure", "ID"]
    BEST_df_train_SEL = pd.concat( 
        [
            df_train_fps[ ["Structure", "ID"] ], 
            BEST_df_train_fps_SEL 
        ], 
        axis = 1, 
        join='inner', 
        ignore_index=False 
    )
    
    model_QSAR = {
            'df_train_sel' : BEST_df_train_SEL,
            'y_name'       : y_NAME,
            'clf_summary'  : BEST_MB['clf_summary']
    }
            
    return model_QSAR

def apply_QSAR( model_QSAR = None, df_test = None ):
    if model_QSAR is None or df_test is None:
        raise ValueError("MUST SUPPLY BOTH model_QSAR AND df_test")
        
    MB = ApplyModel(model_QSAR, df_test)
    
    return MB


def SelectFeatures( df_in, y_name = None, VarThresh = 0, runRFEselector=False, thresh_Rank=0.1, clf_type = "NuSVR" ) :
    if y_name is None:
        raise ValueError("Provide the column name of the response variable (y_name)")
    
    y_train_all = np.array( df_in[ y_name ].values, dtype=float ) # received from the caller
    X_train_all = np.array( df_in.drop([ y_name ], axis=1).values, dtype=float )

    FPnames_ALL = list( df_in.drop([ y_name ], axis=1).columns.values )
    
    print("Applying VarThresh filter...")

    x_std = StandardScaler()
    X_train_all_scaled = x_std.fit_transform( X_train_all )
    
    y_mms = MinMaxScaler()
    y_train_all_scaled = y_mms.fit_transform( y_train_all.reshape(-1,1) )

    ind_stdGT0 = np.std(X_train_all_scaled, axis=0) > VarThresh
    idx_stdGT0 = np.arange(0, X_train_all_scaled.shape[1])[ind_stdGT0]

    import itertools
    FPnames_std = list( itertools.compress(FPnames_ALL, ind_stdGT0) ) # compress works ONLY for True/False vals
    X_train_all_scaled_selected = X_train_all_scaled[:, idx_stdGT0]    
    print("Reduced dim using VarThresh=" + str(VarThresh) + " from " + str(X_train_all.shape[1]) + " => " + str(X_train_all_scaled_selected.shape[1]))

    if( not runRFEselector ) :
        print("Requested NOT to run RFE selector. Returning pruned FPs...")
        df_sel = pd.concat([df_in[ y_name ], df_in[FPnames_std]], axis=1, join='inner')   
        return df_sel, None
    

    print("Calling feature ranker / optimal selector (CAN BE SLOW!!)...")

    clf_type_chosen = clf_type

    X_train_all = np.array( X_train_all_scaled_selected, dtype="float64" )    
    y_train_all = np.array( y_train_all_scaled, dtype="float64" ).reshape(-1,1) # received from the caller
    
    model_opt = utils_opt.ClfTrainFinalOptRFE( 
        np.array( X_train_all_scaled_selected, dtype="float64" ), 
        np.array( y_train_all_scaled, dtype="float64" ).reshape(-1,1), 
        clf_type = clf_type_chosen
    )
    print("Ranking calcs are finished.")
    
    ind_selected_features = model_opt['idx_features_selected']
    print("ind_selected_features =", ind_selected_features)
    
    FPnames_std_selected = [FPnames_std[i] for i in ind_selected_features] # list( itertools.compress(FPnames_std, ind_selected_features) )

    print("Feature selection (" + str(X_train_all_scaled_selected.shape[1]) + " => " + str( len(ind_selected_features) ) + ") finished...")
        
    df_sel = pd.concat([df_in[y_name], df_in[FPnames_std_selected]], axis=1, join='inner')

    return df_sel, model_opt

def BuildModel( df_train, y_name = None, clf_type = "NuSVR", opt_type='Default') :
    clf_types={'NuSVR', 'GPy', 'RFR'}

    if clf_type not in clf_types:
        raise ValueError("UNEXPECTED clf_type -->" + str(clf_type))

    if y_name is None:
        raise ValueError("Missing y_name (column name of the response variable)")
        
    y_train_all = np.array( df_train[ y_name ].values, dtype="float64" ) # received from the caller
    X_train_all = np.array( df_train.drop([ y_name ], axis=1).values, dtype="float64" )

    MB = dict()

    MB['clf_summary'] = {}
    MB['y_name'] = y_name

    if( opt_type == 'Default' ) : # LADD_config.model_type['chosen'] ) :        
        mdl = utils_opt.ClfTrainFinalDefault(X_train_all, y_train_all, clf_type=clf_type)
        # y_pred_chk_default = utils_opt.ClfPredictFinal( model_default, X_test_all )

    elif( opt_type == 'Optimized' ) : # LADD_config.model_type['chosen'] ) :                    
        mdl = utils_opt.ClfTrainFinalOpt(X_train_all, y_train_all, clf_type=clf_type)
        # y_pred_chk_default = utils_opt.ClfPredictFinal( model_default, X_test_all )            
        
    elif( opt_type == 'OptimizedWithFeatureSelection' ) : # LADD_config.model_type['chosen'] ) :            
        mdl = utils_opt.ClfTrainFinalOptRFE(X_train_all, y_train_all, clf_type=clf_type)
        # y_pred_chk_default = utils_opt.ClfPredictFinal( model_default, X_test_all )            
        
    else :
        raise ValueError("UNDEFINED LEVEL OF MODEL TRAINING -->" + str(opt_type))

    mdl['df_train'] = df_train.copy()
    ID_mdl = clf_type + "@" + opt_type

    MB['clf_summary'] = mdl
    MB['ID'] = ID_mdl

    return MB

def ApplyModel(model_QSAR, df_test) :
    '''Scores df_test with all models available in model,
       If df_test contains y_name, generates a side-by-side 2-column df for r2 scoring later
    '''
    if model_QSAR['y_name'] in df_test:
        df_test_y = df_test[ model_QSAR['y_name'] ]
    else:
        df_test_y = None
    
    yy_ret = []

    clf_summary = model_QSAR['clf_summary']
    
    if model_QSAR['y_name'] in df_test.columns:
        df_test_sel = df_test[ model_QSAR['df_train_sel'].columns.values ] # only retain "train" features (test usually contains a superset)
        list2drop = [ model_QSAR['y_name'] ]
        for col2drop in ['Structure', 'ID']:
            if col2drop in df_test_sel.columns:
                list2drop.append( col2drop )

        df_test_sel_vals = df_test_sel.drop( list2drop, axis=1 ).values
        XX_test_unscaled = np.array( df_test_sel_vals, dtype='float64' )
    else:
        XX_test_unscaled = np.array( df_test.values, dtype='float64' )

    yy_pred_test_unscaled = utils_opt.ClfPredictFinal(clf_summary, XX_test_unscaled)

    df_pred = pd.DataFrame(yy_pred_test_unscaled, columns=[ 'Predicted' ], index = df_test.index)
    
    if df_test_y is not None:
        df_pred_vs_GT = pd.concat([df_test_y, df_pred], axis=1, join='inner')
        r2_val = r2_score(df_test_y, yy_pred_test_unscaled)
        print( 'R^2 test set = %.3f' % r2_val )            
    else:
        df_pred_vs_GT = None

    yy_ret = {
        'clf_summary'   : clf_summary,
        'df_pred_vs_GT' : df_pred_vs_GT,
        'y_pred_UnScaled' : yy_pred_test_unscaled
        }

    return yy_ret

