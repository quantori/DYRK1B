#%% imports, declarations:
import pandas as pd
import utils_FPS
from sklearn.metrics import r2_score
from utils_QSAR import apply_QSAR, build_QSAR

fname_TRAIN = "DrugsTrain.csv"
fname_TEST  = "DrugsTest.csv"

#%% Read in the Train set (Structures, IDs, Modeled Var)
train_df = pd.read_csv( fname_TRAIN )

#%% Fingerprint the Train Set:
train_df_fps = utils_FPS.get_SD_fps(train_df)

#%% Build Best CV-optimized QSAR model:
model_QSAR = build_QSAR( train_df_fps, clf_type = "NuSVR", iter_MAX = 2 ) # make iter_MAX = 100 for VERY small (N=15-30) datasets!

# pd.to_pickle(model_QSAR, "model_QSAR.pkl")
# model_QSAR = pd.read_pickle( "model_QSAR.pkl" )

#%% Check how the model performs on a Test Set:

test_df = pd.read_csv( fname_TEST )
test_df_fps = utils_FPS.get_SD_fps(test_df)

test_df_eval = apply_QSAR( model_QSAR = model_QSAR, df_test = test_df_fps )

r2_val = r2_score(test_df_eval['df_pred_vs_GT'].iloc[:,0], test_df_eval['df_pred_vs_GT'].iloc[:,1])
print( 'R^2 test set = %.3f' % r2_val )

#%% END

