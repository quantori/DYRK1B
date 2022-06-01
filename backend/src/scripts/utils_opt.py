import numpy as np
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def RFErank( model ) :
    supp_vect = model.support_vectors_.T
    alpha_Y = model.dual_coef_

    N_feat, N_sv = supp_vect.shape

    H = np.ones( [N_sv, N_sv], dtype = float)
    H = supp_vect.T.dot( supp_vect )

    DJ_1_mat = 0.5 * alpha_Y.dot( H ).dot(alpha_Y.T)
    DJ_1 = DJ_1_mat.diagonal()

    DJ_2_mat = []
    
    K = np.ones([N_sv, N_sv])

    for i in range( N_feat ) : 
        K = H - np.outer(supp_vect[i,:], supp_vect[i,:])
        temp_mat = 0.5 * alpha_Y.dot( K ).dot(alpha_Y.T)    
        temp = temp_mat.diagonal()
        DJ_2_mat.append( temp )
    
    DJ_2 = np.array( DJ_2_mat ).T

    if np.min( alpha_Y.shape ) > 1 :
        DJ = np.mean( DJ_2 + np.repeat(DJ_1, DJ_2.shape[1], axis=0).reshape(temp_mat.shape[1],-1), axis=0 )
    else :
        DJ = DJ_1 + DJ_2
        DJ = DJ.flatten()
       
    ind_feat  = np.argsort( DJ )
    rfc = DJ[ ind_feat ] / np.max( DJ )

    rank_feat = 1 - (rfc - np.min(rfc)) / (np.max(rfc) - np.min(rfc))

    return rank_feat, ind_feat

def rank_feats(X, y, SVR=True, C=None, g=None, nu=None):
    '''Ranks features from feature matrix X and response variable y via building X->y SVM/SVR model 
       and then performing sensitivity analysis on the built model. Continuous SVR model is built by default
    '''
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    
    if X.shape[1] == 1:
        rank_feat = np.array([1.0]).ravel()
        ind_feat  = np.array([0]).ravel()
        return rank_feat, ind_feat
                                       
    if C is None:
        C = 1.0
        
    if g is None:
        g = 1/X.shape[1]
        
    if nu is None:
        nu = 1.0
        
    sc_minmax = MinMaxScaler(feature_range= [-1, 1])
    sc_std = StandardScaler()
        
    XX = sc_std.fit_transform( X )
    yy = sc_minmax.fit_transform( y.reshape(-1,1) ).ravel()

    if SVR:
        model = svm.NuSVR(kernel='rbf', C=C, gamma=g, nu=1.0)
        model.fit(XX, yy)
    else:
        model = svm.SVC(kernel = 'rbf', C=C, gamma=g)
        model.fit(XX, y)
    
    rank_feat, ind_feat = RFErank( model )
    
    return rank_feat, ind_feat


def ClfTrainDefault(XX_train, yy_train, clf_type='NuSVR', return_best_score=False) :
    """
    Presumes XX_train, yy_train to be already properly scaled
    """ 
    store_clf=[]
    store_score=[]
    cv_num = 5    

    if( clf_type == 'NuSVR' ) :
        print("Building DEFAULT NuSVR model...")
        from sklearn import svm
        C = 1.0
        nVar = 1
        if( XX_train.ndim > 1 ) :
            nVar = XX_train.shape[1] 
        G = 1.0 / nVar       
        
        if( return_best_score ) :     
            kf = KFold(n_splits=cv_num, shuffle=True)
            cv_count = 1
            for idx_train, idx_test in kf.split(XX_train, yy_train):

                XX_train_selected = XX_train[idx_train, :]
                yy_train_selected = yy_train[idx_train].ravel().reshape(-1,1)
                XX_test_selected = XX_train[idx_test, :]
                yy_test_selected = yy_train[idx_test].ravel().reshape(-1,1)
                            
                clf = svm.NuSVR(kernel='rbf', C=C, gamma=G, nu=1.0)        
                clf.fit(XX_train_selected, yy_train_selected)
                            
                store_clf.append(clf)
                yy_test_predicted = clf.predict(XX_test_selected)
                r2_current = r2_score(yy_test_selected, yy_test_predicted)
                store_score.append(r2_current)
                
                cv_count = cv_count + 1
                        
            score_mean = np.mean( store_score )
            print("Mean NuSVR cv_score = " + str(score_mean))
            print("Building final DEFAULT classifier...")
            
            clf_default = svm.NuSVR(kernel='rbf', C=C, gamma=G, nu=1.0)        
            clf_default.fit(XX_train, yy_train) # print( clf.score(X_train, y_train) )                            
            
            print("Final DEFAULT classifier was successfully built.")            
            return clf_default, score_mean
            
        else :
            clf = svm.NuSVR(kernel='rbf', C=C, gamma=G, nu=1.0)        
            clf.fit(XX_train, yy_train.ravel()) # print( clf.score(X_train, y_train) )
        
    elif( clf_type == 'RFR' ) :
        from sklearn.ensemble import RandomForestRegressor
        print("Building DEFAULT Random Forest Regressor model...")        
        
        if( return_best_score ) :     
            kf = KFold(n_splits=cv_num, shuffle=True)
            cv_count = 1
            for idx_train, idx_test in kf.split(XX_train, yy_train):

                XX_train_selected = XX_train[idx_train, :]
                yy_train_selected = yy_train[idx_train].ravel().reshape(-1,1)
                XX_test_selected = XX_train[idx_test, :]
                yy_test_selected = yy_train[idx_test].ravel().reshape(-1,1)
                                            
                clf = RandomForestRegressor(n_estimators=10000, max_depth=1000, n_jobs=-1)
                clf.fit(XX_train_selected, yy_train_selected.ravel())                
                            
                store_clf.append(clf)
                yy_test_predicted = clf.predict(XX_test_selected)
                r2_current = r2_score(yy_test_selected, yy_test_predicted)

                store_score.append(r2_current)
                
                cv_count = cv_count + 1
                        
            score_mean = np.mean( store_score )
            print("Mean RFR cv_score = " + str(score_mean))
            print("Building final DEFAULT classifier...")
                        
            clf_default = RandomForestRegressor(n_estimators=10000, max_depth=1000, n_jobs=-1)
            clf_default.fit(XX_train, yy_train.ravel())                
                        
            print("Final DEFAULT classifier was successfully built.")            
            return clf_default, score_mean        
        else :
            from sklearn.ensemble import RandomForestRegressor
            clf = RandomForestRegressor(n_estimators=10000, max_depth=1000, n_jobs=-1)
            clf.fit(XX_train, yy_train.ravel())
                    
    elif( clf_type == 'GPy' ) :
        print("Building DEFAULT GPy model...")
        import GPy        
                
        if( return_best_score ) :     
            kf = KFold(n_splits=cv_num, shuffle=True)
            cv_count = 1
            for idx_train, idx_test in kf.split(XX_train, yy_train):

                XX_train_selected = XX_train[idx_train, :]
                yy_train_selected = yy_train[idx_train].ravel().reshape(-1,1)
                XX_test_selected = XX_train[idx_test, :]
                yy_test_selected = yy_train[idx_test].ravel().reshape(-1,1)
            
                kernel = GPy.kern.RBF(input_dim=XX_train_selected.shape[1])
                clf = GPy.models.GPRegression(XX_train_selected, yy_train_selected, kernel)
                clf.optimize(messages=False, max_f_eval = 10000)
                            
                store_clf.append(clf)
                yy_test_predicted, _ = clf.predict(XX_test_selected)
                r2_current = r2_score(yy_test_selected, yy_test_predicted)
                store_score.append(r2_current)
                
                cv_count = cv_count + 1
                        
            score_mean = np.mean( store_score )
            print("Mean GPy cv_score = " + str(score_mean))
            print("Building final DEFAULT classifier...")
            kernel = GPy.kern.RBF(input_dim=XX_train.shape[1])
            clf_default = GPy.models.GPRegression(XX_train, yy_train, kernel)
            clf_default.optimize(messages=False, max_f_eval = 10000)
            print("Final DEFAULT classifier was successfully built.")                    
            return clf_default, score_mean
        else :
            kernel = GPy.kern.RBF(input_dim=XX_train.shape[1])
            clf = GPy.models.GPRegression(XX_train, yy_train.reshape(-1,1), kernel)
            clf.optimize(messages=False, max_f_eval = 10000)            
        
    else :
        raise ValueError('UNKNOWN clf_type -->' + clf_type)
     
    print("DEFAULT " + clf_type + " model has been successfully built.")    
    return clf
    
    
def ClfTrainOpt(XX_train, yy_train, clf_type='NuSVR', return_best_score=False, cv_num = 5) :
    """
    Presumes XX_train, yy_train to be already properly scaled
    """

    if( clf_type == 'NuSVR' ) :
        print("Building OPTIMAL NuSVR model...")

        G = 1.0 / XX_train.shape[1]

        params2opt = [{'gamma': [G, G*2.0, G*0.5, G*5.0, G*0.2, G*10.0, G*0.1],
                     'C': [0.5, 0.75, 0.9, 1.0, 10.0, 100.0]}
                    ]

        # clf_cv = GridSearchCV( svm.NuSVR(kernel='rbf'), params2opt, cv=5, n_jobs=-1)
        clf_cv = GridSearchCV( svm.NuSVR(kernel='rbf', nu=1.0), params2opt, cv=KFold(n_splits=cv_num, shuffle=True), n_jobs=-1)
        clf_cv.fit(XX_train, yy_train.ravel()) # print( clf.score(X_train, y_train) )        

        clf_opt = clf_cv.best_estimator_
        if return_best_score :
            return clf_opt, clf_cv.best_score_
        else :    
            return clf_opt
        
    elif( clf_type == 'RFR' ) :
        print("Building OPTIMAL Random Forest Regressor model...")
        from sklearn.ensemble import RandomForestRegressor
        
        params2opt = [{'max_depth': [1000, 500, 100],
                     'n_estimators': [10000, 5000, 1000]}
                    ]

        clf_cv = GridSearchCV( RandomForestRegressor(n_jobs=-1), params2opt, cv=KFold(n_splits=cv_num, shuffle=True), n_jobs=-1)
        clf_cv.fit(XX_train, yy_train.ravel())

        clf_opt = clf_cv.best_estimator_            
        if return_best_score :            
            return clf_opt, clf_cv.best_score_
        else :    
            return clf_opt        

    elif( clf_type == 'GPy' ) :
        print("Building OPTIMAL GPy model...")
        import GPy
        store_clf=[]
        store_score=[]

        kf = KFold(n_splits=cv_num, shuffle=True)
        cv_count = 1
        for idx_train, idx_test in kf.split(XX_train, yy_train):

            XX_train_selected = XX_train[idx_train, :]
            yy_train_selected = yy_train[idx_train].ravel().reshape(-1,1)
            XX_test_selected = XX_train[idx_test, :]
            yy_test_selected = yy_train[idx_test].ravel().reshape(-1,1)
        
            kernel = GPy.kern.RBF(input_dim=XX_train_selected.shape[1])
            clf = GPy.models.GPRegression(XX_train_selected, yy_train_selected, kernel)
            clf.optimize(messages=False, max_f_eval = 10000)
                        
            store_clf.append(clf)
            yy_test_predicted, _ = clf.predict(XX_test_selected)
            r2_current = r2_score(yy_test_selected, yy_test_predicted)

            store_score.append(r2_current)
            
            cv_count = cv_count + 1
                    
        score_mean = np.mean( store_score )
        print("Mean GPy cv_score = " + str(score_mean))

        kernel = GPy.kern.RBF(input_dim=XX_train.shape[1])
        clf_opt = GPy.models.GPRegression(XX_train, yy_train, kernel)
        clf_opt.optimize(messages=False, max_f_eval = 10000)

        if return_best_score :
            return clf_opt, score_mean
        else :    
            return clf_opt                             
            
    else :
        raise ValueError('UNKNOWN clf_type -->' + clf_type)
        
    return clf
    
def ClfTrainDefaultRFE(XX_train_all, yy_train_all, clf_type='NuSVR', return_idx_selected_features=True) :
    """
    Presumes XX_train_all, yy_train_all to be already properly scaled
    NOTE: returned clf would probably be USELESS WITHOUT indices of selected features!!!
    """   
    print("Building feature-ranking curve...\n")

    maxFeats = 1000000
    from sklearn import svm
    C = 1.0
    G = 1.0 / XX_train_all.shape[1]    
    clf4RFErank = svm.NuSVR(kernel='rbf', C=C, gamma=G, nu=1.0) 
    clf4RFErank.fit(XX_train_all, yy_train_all.ravel())
    
    rank_feat, ind_feat = RFErank( clf4RFErank )
    
    thresh_Ranks = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]

    flen_perf = np.zeros_like(thresh_Ranks)
    rank_perf = np.zeros_like(thresh_Ranks)
    store_ind_selected = {}
    store_clf = {}
    
    for idx_rank, thresh_Rank in enumerate(thresh_Ranks) :    
        ind_selected = ind_feat[ rank_feat >= thresh_Rank ][:maxFeats]
        store_ind_selected[idx_rank] = ind_selected
        
        XX_train_all_selected = XX_train_all[:, ind_selected]
        
        clf_opt, clf_opt_score = ClfTrainDefault(XX_train_all_selected, yy_train_all, clf_type=clf_type, return_best_score=True)
        store_clf[idx_rank] = clf_opt
        rank_perf[idx_rank] = clf_opt_score
        flen_perf[idx_rank] = len(ind_selected)
        print('For rank_thresh = %.4f (Nfeats = %d) => R^2 cv_opt_mean = %.3f' % (thresh_Rank, len(ind_selected), rank_perf[idx_rank]) )
    
    thresh_Rank_sorted_idx = np.argsort(rank_perf)[::-1]
    thresh_Rank_best_idx = thresh_Rank_sorted_idx[0]
    thresh_Rank_best = thresh_Ranks[thresh_Rank_best_idx]
    
    print('\nbest_thresh_Rank = %.3f, Nfeats = %d, best_r2 = %.3f' % (thresh_Rank_best, flen_perf[thresh_Rank_best_idx], rank_perf[thresh_Rank_best_idx]))
    
    if return_idx_selected_features :
        rank_vs_perf = {'thresh_Ranks' : thresh_Ranks,
                        'flen_perf' : flen_perf, 
                        'rank_perf' : rank_perf}                        
        return store_clf[thresh_Rank_best_idx], store_ind_selected[thresh_Rank_best_idx], rank_vs_perf
    else :    
        return store_clf[thresh_Rank_best_idx]          
        
def ClfTrainOptRFE(XX_train_all, yy_train_all, clf_type='NuSVR', return_idx_selected_features=True) :
    """
    Presumes XX_train_all, yy_train_all to be already properly scaled
    NOTE: returned clf would probably be USELESS WITHOUT indices of selected features!!!
    """   
    print("Building feature-ranking curve...\n")
    import numpy as np
    
    maxFeats = 1000000
    from sklearn import svm
    C = 1.0
    G = 1.0 / XX_train_all.shape[1]    
    clf4RFErank = svm.NuSVR(kernel='rbf', C=C, gamma=G, nu=1.0) 
    clf4RFErank.fit(XX_train_all, yy_train_all.ravel())
    
    rank_feat, ind_feat = RFErank( clf4RFErank )
    
    thresh_Ranks = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35] # [0.0001, 0.0005, 0.001, 0.005, 0.01]

    flen_perf = np.zeros_like(thresh_Ranks)
    rank_perf = np.zeros_like(thresh_Ranks)
    store_ind_selected = {}
    store_clf = {}
    
    for idx_rank, thresh_Rank in enumerate(thresh_Ranks) :    
        ind_selected = ind_feat[ rank_feat >= thresh_Rank ][:maxFeats]
        store_ind_selected[idx_rank] = ind_selected
        
        XX_train_all_selected = XX_train_all[:, ind_selected]
        
        clf_opt, clf_opt_score = ClfTrainOpt(XX_train_all_selected, yy_train_all, clf_type=clf_type, return_best_score=True) 
        store_clf[idx_rank] = clf_opt            
        rank_perf[idx_rank] = clf_opt_score
        flen_perf[idx_rank] = len(ind_selected)
        print('For rank_thresh = %.4f (Nfeats = %d) => R^2 cv_opt_mean = %.3f' % (thresh_Rank, len(ind_selected), rank_perf[idx_rank]) )
    
    thresh_Rank_sorted_idx = np.argsort(rank_perf)[::-1]
    thresh_Rank_best_idx = thresh_Rank_sorted_idx[0]
    thresh_Rank_best = thresh_Ranks[thresh_Rank_best_idx]
    
    print('\nbest_thresh_Rank = %.3f, Nfeats = %d, best_r2 = %.3f' % (thresh_Rank_best, flen_perf[thresh_Rank_best_idx], rank_perf[thresh_Rank_best_idx]))
    
    if return_idx_selected_features :
        rank_vs_perf = {
            'thresh_Ranks' : thresh_Ranks,
            'flen_perf' : flen_perf, 
            'rank_perf' : rank_perf,
            'best' : {
                'thresh_Rank' : thresh_Rank_best,
                'Nfeats' : flen_perf[thresh_Rank_best_idx],
                'r2' : rank_perf[thresh_Rank_best_idx]
                }
            }                        
        return store_clf[thresh_Rank_best_idx], store_ind_selected[thresh_Rank_best_idx], rank_vs_perf
    else :    
        return store_clf[thresh_Rank_best_idx]      

# Models below will contain features/response-scaling info and selected_features indices
def ClfTrainFinalDefault(X_train_all, y_train_all, clf_type='NuSVR', return_ranked_features=False) :

    import numpy as np
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    y_mms = MinMaxScaler().fit( y_train_all.reshape(-1,1) )
    x_std = StandardScaler().fit( X_train_all )
    
    XX_train_all = x_std.transform( X_train_all )
    yy_train_all = y_mms.transform( y_train_all.reshape(-1,1) )
    
    clf_default = ClfTrainDefault(XX_train_all, yy_train_all, clf_type=clf_type)
    
    if return_ranked_features : 
        C = 1.0
        G = 1.0 / XX_train_all.shape[1]    
        clf4RFErank = svm.NuSVR(kernel='rbf', C=C, gamma=G, nu=1.0) 
        clf4RFErank.fit(XX_train_all, yy_train_all.ravel())    
        rank_feat, ind_feat = RFErank( clf4RFErank )
        ind_selected_features = ind_feat
    else :
        ind_selected_features = np.arange(X_train_all.shape[1])    
    
    model_final = {}
    model_final['clf_type'] = clf_type
    model_final['clf'] = clf_default
    model_final['opt_level'] = 'Default'
    model_final['idx_features_selected'] = ind_selected_features
    model_final['transform_x'] = x_std
    model_final['transform_y'] = y_mms
    
    return model_final    

def ClfTrainFinalOpt(X_train_all, y_train_all, clf_type='NuSVR', return_ranked_features=False) :
    x_std = StandardScaler()
    XX_train_all = x_std.fit_transform( X_train_all )
    
    y_mms = MinMaxScaler()
    yy_train_all = y_mms.fit_transform( y_train_all.reshape(-1,1) )      
    
    clf_opt, clf_opt_score = ClfTrainOpt(XX_train_all, yy_train_all, clf_type=clf_type, return_best_score=True)
    print("Final Opt model (all features) best cv_score = " + str(clf_opt_score))
    
    if return_ranked_features : 
        C = 1.0
        G = 1.0 / XX_train_all.shape[1]    
        clf4RFErank = svm.NuSVR(kernel='rbf', C=C, gamma=G, nu=1.0) 
        clf4RFErank.fit(XX_train_all, yy_train_all.ravel())    
        rank_feat, ind_feat = RFErank( clf4RFErank )
        ind_selected_features = ind_feat
    else :
        ind_selected_features = np.arange(X_train_all.shape[1])
    
    model_final = {}
    model_final['clf_type'] = clf_type
    model_final['clf'] = clf_opt
    model_final['opt_level'] = 'Optimized'
    model_final['clf_opt_score'] = clf_opt_score    
    model_final['idx_features_selected'] = ind_selected_features
    model_final['transform_x'] = x_std
    model_final['transform_y'] = y_mms
    
    return model_final


def ClfTrainFinalOptRFE(X_train_all, y_train_all, clf_type='NuSVR') :
_type='GPy' # 'GPy' 'NuSVR' 'RFR'
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
            
    y_mms = MinMaxScaler().fit( y_train_all.reshape(-1,1) )
    x_std = StandardScaler().fit( X_train_all )
    
    XX_train_all = x_std.transform( X_train_all )
    yy_train_all = y_mms.transform( y_train_all.reshape(-1,1) )
    
    clf_opt_RFE, ind_selected_features, rank_vs_perf = ClfTrainOptRFE(XX_train_all, yy_train_all, clf_type=clf_type)
    
    model_final = {}
    model_final['clf_type'] = clf_type
    model_final['clf'] = clf_opt_RFE
    model_final['opt_level'] = 'OptRFE'
    model_final['idx_features_selected'] = ind_selected_features
    model_final['transform_x'] = x_std
    model_final['transform_y'] = y_mms
    model_final['rank_vs_perf'] = rank_vs_perf
    
    return model_final

    
def ClfTrainFinalDefaultRFE(X_train_all, y_train_all, clf_type='NuSVR') :
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
            
    y_mms = MinMaxScaler().fit( y_train_all.reshape(-1,1) )
    x_std = StandardScaler().fit( X_train_all )
    
    XX_train_all = x_std.transform( X_train_all )
    yy_train_all = y_mms.transform( y_train_all.reshape(-1,1) )
    
    clf_opt_RFE, ind_selected_features, rank_vs_perf = ClfTrainDefaultRFE(XX_train_all, yy_train_all, clf_type=clf_type)
    
    model_final = {}
    model_final['clf_type'] = clf_type
    model_final['clf'] = clf_opt_RFE
    model_final['opt_level'] = 'DefaultRFE'
    model_final['idx_features_selected'] = ind_selected_features
    model_final['transform_x'] = x_std
    model_final['transform_y'] = y_mms
    model_final['rank_vs_perf'] = rank_vs_perf
    
    return model_final    

    
def ClfPredict(clf_reduced, clf_type, XX_test_selected) :
    """
    Presumes XX_train, yy_train to be already properly scaled
    """    

    if( clf_type == 'NuSVR' ) :
        y_pred_test = clf_reduced.predict(XX_test_selected)        
    elif( clf_type == 'RFR' ) :
        y_pred_test = clf_reduced.predict(XX_test_selected)
    elif( clf_type == 'GPy' ) :
        y_pred_test, _  = clf_reduced.predict(XX_test_selected)
    else :
        raise ValueError('UNKNOWN clf_type -->' + clf_type)
        
    return y_pred_test    

def ClfPredictFinal( model_final, X_test_all, return_unscaled=True ) :
    """
    NOTE: returns UNscaled prediction!    
    """
    X_test_scaled_reduced = model_final['transform_x'].transform( X_test_all )[:, model_final['idx_features_selected']]

    y_pred_chk = ClfPredict( model_final['clf'], model_final['clf_type'], X_test_scaled_reduced )
    if return_unscaled :
        return model_final['transform_y'].inverse_transform( y_pred_chk.reshape(-1,1) )
    else :
        return y_pred_chk
