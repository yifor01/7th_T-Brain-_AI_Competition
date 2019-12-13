import os
import gc
import pandas as pd 
import numpy as np 
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
#import pandas_profiling 
import matplotlib
import warnings
matplotlib.style.use('ggplot')
warnings.filterwarnings("ignore")
from IPython.display import display
import random
import time
from sklearn.model_selection import train_test_split,StratifiedKFold,TimeSeriesSplit,GridSearchCV
from model import *
from tool import *
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
import xgboost as xgb
from sklearn.metrics import make_scorer


# Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
df = train.merge(test,how='outer')
del train,test
df = reduce_mem_usage(df)
gc.collect()


print(pd.concat([df[df['fraud_ind'] == 1]['conam']\
                 .quantile([.01, .1, .25, .5, .75, .9, .99])\
                 .reset_index(), 
                 df[df['fraud_ind'] == 0]['conam']\
                 .quantile([.01, .1, .25, .5, .75, .9, .99])\
                 .reset_index()],
                axis=1, keys=['Fraud', "No Fraud"]))



# Feature Extending
df['locdt_dtran'] = df['locdt'] % 7 
df['locdt_mtran'] = df['locdt'] % 30
df['loctm'] = df['loctm']//10000 + (df['loctm'] - (df['loctm']//10000)*10000 )/6000
df['loctm_sec'] = df['loctm']%100
df['loctm_hour'] = df['loctm']//10000
df['conam'] = (df.conam/1943452)
df['bacno_return'] = df.sort_values(by=['bacno','locdt']).groupby('bacno').locdt.diff().sort_index()

cat_feature = [x for x in df.columns if x not in ['conam','loctm','txkey','fraud_ind','loctm_sec','locdt_mtran'] ]
# Check train category not use in test
print(f'Before: {df.shape}')
for i in set(cat_feature).difference(['cano','locdt','bacno']) :
    if len(set(df[df.fraud_ind.notnull()][i].unique()).difference(set(df[df.fraud_ind.isnull()][i].unique()) ))>0 :
        drop_target = list(set(df[df.fraud_ind.notnull()][i].unique()).difference(set(df[df.fraud_ind.isnull()][i].unique()) ))
        print(  f'{i}: {len(drop_target)}' )
        df = df.drop(df[df[i].isin(drop_target)].index)
print(f'After: {df.shape}')

#target_mean_feature = ['csmcu','etymd','mcc','mchno','scity','stocn','stscd']
target_mean_feature = ['csmcu','etymd','mcc','scity','stocn','stscd']
onehot_feature = ['contp','flbmk','ecfg','flg_3dsmk','hcefg','insfg','ovrlt','stscd','locdt_dtran','iterm']
freq_feature = ['csmcu','etymd','mcc','mchno','acqic','bacno','cano','scity','stocn']
print('Target mean encoding ...')
#for k in target_mean_feature:
#    df[k+'_tm'] = df[k].map(df.groupby(k).fraud_ind.mean())
#    df[k+'_ts'] = df[k].map(df.groupby(k).fraud_ind.std())
#    df[k+'_txkeym'] = df[k].map(df.groupby(k).txkey.mean())
#    df[k+'_txkeys'] = df[k].map(df.groupby(k).txkey.std())

# Might be overfit feature
df['test1'] = df.cano.map( df.groupby('cano').txkey.median())
#df['test2'] = df.bacno.map( df.groupby('bacno').txkey.median())

for k in freq_feature:
    df[k+'_f'] = df[k].map(df[k].value_counts(normalize=True) )
print('One hot encoding ...')
for k in onehot_feature:
    add_dumy = pd.get_dummies(df[k])
    add_dumy.columns = [ k+ "_{}".format(x)  for x in add_dumy.columns ]
    if add_dumy.shape[0] < 2:
        add_dumy = add_dumy.iloc[:,0]
    df = pd.concat([df,add_dumy],axis=1)


print('Feature mean encoding ...')
# conam 相關
#for k in cat_feature:
    #df[k+"_conam_min"] =  df[k].map(df.groupby(k).conam.min())
    #df[k+"_conam_max"] =  df[k].map(df.groupby(k).conam.max())
    #df[k+"_conam_med"] =  df[k].map(df.groupby(k).conam.median())
    #df[k+"_conam_mean"] = df[k].map(df.groupby(k).conam.mean())
    #df[k+"_conam_std"] =  df[k].map(df.groupby(k).conam.std())
    #df[k+"_conam_mean_ratio"] = df.conam / df[k+"_conam_mean"] 
    #df[k+"_conam_std_ratio"] =  df.conam / df[k+"_conam_std"] 
    #del df[k+"_conam_mean"] ,df[k+"_conam_std"]


# txkey 相關
add_feature = ['contp','hcefg','stscd','locdt_dtran','iterm']

#for k in add_feature:
#    #df[k+"_txkey_min"] =  df[k].map(df.groupby(k).txkey.min())
#    #df[k+"_txkey_max"] =  df[k].map(df.groupby(k).txkey.max())
#    #df[k+"_txkey_med"] =  df[k].map(df.groupby(k).txkey.median())
#    df[k+"_txkey_mean"] = df[k].map(df.groupby(k).txkey.mean())
#    df[k+"_txkey_std"] =  df[k].map(df.groupby(k).txkey.std())
#    df[k+"_txkey_mean_ratio"] = df.txkey / df[k+"_txkey_mean"] 
#    df[k+"_txkey_std_ratio"] =  df.txkey / df[k+"_txkey_std"] 
#    del df[k+"_txkey_mean"] ,df[k+"_txkey_std"]
    
df['digital'] = (df.conam - df.conam.astype('int'))*1000

# stocn target mean (train: 103, all:109,test=87)
#df['stocn_tm1'] = df.stocn.map(df.groupby('stocn').fraud_ind.mean())

# mcc target mean (train:434 , all:460,test=372)
#df['mcc_tm1'] = df.mcc.map(df.groupby('mcc').fraud_ind.mean())

# csmcu target mean  (train:72 , all:76, test=56)
#df['csmcu_tm1'] = df.csmcu.map(df.groupby('csmcu').fraud_ind.mean())

# card use ratio 
df['use_card'] = df['cano'].map(df['cano'].value_counts() )/ df['bacno'].map(df['bacno'].value_counts() )
print('Kaggle feature ...')
# Kaggle feature
df['mean_last'] = df['conam'] - df.groupby('cano')['conam'].transform(lambda x: x.rolling(10, 1).mean())
df['min_last'] = df.groupby('cano')['conam'].transform(lambda x: x.rolling(10, 1).min())
df['max_last'] = df.groupby('cano')['conam'].transform(lambda x: x.rolling(10, 1).max())
df['std_last'] = df['mean_last'] / df.groupby('cano')['conam'].transform(lambda x: x.rolling(10, 1).std())
df['count_last'] = df.groupby('cano')['conam'].transform(lambda x: x.rolling(30, 1).count())
df['mean_last'].fillna(0, inplace=True, )
df['std_last'].fillna(0, inplace=True)

# Feature selection
threshold = 0.98
# Absolute value correlation matrix
corr_matrix = df[df['fraud_ind'].notnull()].corr().abs()
# Getting the upper triangle of correlations
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Select columns with correlations above threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
df = df.drop(columns=to_drop)

temp1 = df.groupby(['insfg','iterm']).fraud_ind.mean().reset_index()
temp1.columns.values[2] = 'insfg_iterm_tm'
df = df.merge(temp1,how='left')
df = reduce_mem_usage(df)
print(f'Data size: {df.shape}')
###################################################################
###################################################################
###################################################################
'''
id_cols = [c for c in df.columns if c not in ['locdt'] ]
error_col = []
for i in id_cols:
    try:
        df[df.fraud_ind.notnull()].set_index('locdt')[i].plot(style='.', title=i, figsize=(15, 3), alpha=0.01)
        df[df.fraud_ind.isnull()].set_index('locdt')[i].plot(style='.', title=i, figsize=(15, 3), alpha=0.01)
        plt.show()
    except TypeError:
        error_col.append(i)
        pass
print(error_col)

'''
###################################################################
###################################################################
# model
#import lightgbm as lgb
from sklearn.metrics import f1_score
#import xgboost as xgb
#from sklearn.model_selection import GridSearchCV 
from xgboost.sklearn import XGBClassifier


def split_data(df,method=1):
    # random split (X)
    if method == 1:
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y,random_state = 123)
    # by cano (O)
    elif method == 2:
        q_index = df[df.fraud_ind.notnull()].groupby('cano').fraud_ind.mean()
        q_index[q_index>0] = 1
        tt_idx = random.choices(q_index[q_index==0].index , k = np.int_(len(q_index[q_index==0])*0.8)) +\
            random.choices(q_index[q_index==1].index , k = np.int_(len(q_index[q_index==1])*0.8)) 
        vv_idx = list(set(q_index.index).difference(tt_idx))
        X_train = df[(df.fraud_ind.notnull()) & (df.cano.isin(tt_idx)) & (df.locdt<=66) ][feature]
        y_train = df[(df.fraud_ind.notnull()) & (df.cano.isin(tt_idx)) & (df.locdt<=66) ]['fraud_ind'].values.astype('int') 
        X_test = df[(df.fraud_ind.notnull()) & (df.cano.isin(vv_idx))  & (df.locdt> 66)][feature]
        y_test = df[(df.fraud_ind.notnull()) & (df.cano.isin(vv_idx))  & (df.locdt> 66)]['fraud_ind'].values.astype('int') 
    # by locdt day (X)
    elif method==3:
        tt_idx = []
        for i in range(1,91):
            num = df[df.locdt==i].shape[0]
            cand = random.sample( list(df[df.locdt==i].index) , np.int_(num*0.8) )
            tt_idx+= cand
        vv_idx = list(set(df[df.fraud_ind.notnull()].index).difference(tt_idx))
        X_train = df[(df.fraud_ind.notnull()) & (df.index.isin(tt_idx)) ][feature]
        y_train = df[(df.fraud_ind.notnull()) & (df.index.isin(tt_idx)) ]['fraud_ind'].values.astype('int') 
        X_test = df[(df.fraud_ind.notnull()) & (df.index.isin(vv_idx)) ][feature]
        y_test = df[(df.fraud_ind.notnull()) & (df.index.isin(vv_idx)) ]['fraud_ind'].values.astype('int') 
    # sliding windows
    elif method==4: 
        X_train = df[(df.fraud_ind.notnull()) & (df.locdt<=66) ][feature]
        y_train = df[(df.fraud_ind.notnull()) & (df.locdt<=66) ]['fraud_ind'].values.astype('int') 
        X_test = df[(df.fraud_ind.notnull()) & (df.locdt>66) ][feature]
        y_test = df[(df.fraud_ind.notnull()) & (df.locdt>66) ]['fraud_ind'].values.astype('int') 
        
    return X_train, X_test, y_train, y_test


def model_(x_train,y_train,x_test,y_test,boost_type='lgb'):
    tStart = time.time()
    if boost_type=='lgb':
        model = lgb.LGBMClassifier(
            boosting_type='gbdt',
            objective='binary',
            learning_rate=0.01, 
            n_estimators= 9000, 
            max_depth = 8, 
            min_child_weight = 5,       
            scale_pos_weight = 9, # refer: 70
            subsample = 0.7,
            colsample_bytree = 0.7,
            subsample_freq =1,
            n_jobs=-1)

    elif boost_type=='xgb':
        model = XGBClassifier(
            learning_rate = 0.025 , 
            tree_method = 'gpu_hist',
            n_estimators=6000, 
            max_depth=9,
            min_child_weight=1, 
            gamma=0, 
            subsample=0.8, 
            colsample_bytree=0.8,
            objective= 'binary:logistic', 
            nthread=-1, 
            scale_pos_weight=11, 
            seed=27
            )
    print('--'*25)
    print('Start training ...')
    model.fit(x_train,y_train)
    yp_train = model.predict_proba(x_train)[:,1]
    yp_valid = model.predict_proba(x_test)[:,1]
    print(f'Use time: { np.int_((time.time()-tStart)/60)  } mins\nCaluate prob ...')
    
    ## probability tune
    mat = np.zeros([5,100])
    for threshold in range(100):
        y_pred_train = np.int_( yp_train > threshold*0.01)
        y_pred_valid = np.int_( yp_valid > threshold*0.01)
        mat[0,threshold] = round(threshold*0.01,2)
        mat[1,threshold] = f1_score(y_train,y_pred_train)
        mat[2,threshold] = f1_score(y_test,y_pred_valid) 
        mat[3,threshold] = (y_train==y_pred_train).mean()
        mat[4,threshold] = (y_test==y_pred_valid).mean()
        
    # Fig1 for F1
    sns.pointplot( x= mat[0,:],y= mat[1,:],color='r')
    sns.pointplot( x= mat[0,:],y= mat[2,:],color='b')
    plt.title(f'{boost_type} F1 performance',color='r')
    plt.show()
    
    # Fig2 for acc
    sns.pointplot( x= mat[0,10:],y= mat[3,10:],color='r')
    sns.pointplot( x= mat[0,10:],y= mat[4,10:],color='b')
    plt.title(f'{boost_type} Acc performance',color='r')
    plt.show()
    print('--'*20)
    
    # reult for best probalility
    best_prob = round(np.argmax(mat[2,:])*0.01,2)
    print('Valid Result:\nprob: {}, F1 : {}, acc : {}'.\
          format(best_prob,max(mat[2,:]).round(3), mat[4,:][np.argmax(mat[2,:])].round(3)))
    print('--'*20)
    
    # confusion matrix
    y_pred_train = np.int_( yp_train > best_prob) 
    y_pred_valid = np.int_( yp_valid > best_prob) 
    print('Train confusion matrix')
    display(pd.crosstab(y_train, y_pred_train,margins=True, margins_name="Total" ))
    print('--'*20)
    print('Valid confusion matrix')
    display(pd.crosstab(y_test,y_pred_valid,margins=True, margins_name="Total" ))
    print('--'*20)
    
    print('Feature Importance (Top 10)')
    display(pd.DataFrame({'feature':feature,'gain':model.feature_importances_}).\
        sort_values(by='gain',ascending=False).iloc[0:10,:])
    print('--'*25)
    return model,best_prob

# xgboost parameter
space = {
    'max_depth': hp.quniform('max_depth', 3, 15, 1),
    'reg_alpha':  hp.uniform('reg_alpha', 0.01, 0.4),
    'reg_lambda': hp.uniform('reg_lambda', 0.01, .4),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 0.9),
    'gamma': hp.uniform('gamma', 0.01, 0.7),
    'num_leaves': hp.choice('num_leaves', list(range(20, 250, 10))),
    'min_child_samples': hp.choice('min_child_samples', list(range(100, 250, 10))),
    'subsample': hp.choice('subsample', [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    'feature_fraction': hp.uniform('feature_fraction', 0.4, 0.8),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.4, 0.9)
}



feature = [x for x in df.columns if x not in \
            onehot_feature + freq_feature+['count_1','fraud_ind','locdt','locdt_tran']]
X = df[df.fraud_ind.notnull()][feature]
y = df[df.fraud_ind.notnull()]['fraud_ind'].values.astype('int') 

X_train, X_test, y_train, y_test = split_data(df,method = 4 )

## CV
print('CV ...')
gc.collect()
final_model,best_prob = model_(x_train=X_train,y_train=y_train,x_test=X_test,y_test=y_test,boost_type='xgb')
print(f'Best prob is {best_prob}')


# final model
print('Training ...')
tStart = time.time()
final_model.fit(X,y)
#final_model.fit(new_X,new_y)
print(f'Use time: { np.int_((time.time()-tStart)/60)  } mins\nCaluate prob ...')


y_prob = final_model.predict_proba( df[df.fraud_ind.isnull()][feature])[:,1]
sns.distplot(y_prob)
submit = pd.read_csv('submission_test.csv')

submit.fraud_ind = np.int_(y_prob >best_prob)
submit.fraud_ind.value_counts()


submit.fraud_ind = np.int_(y_prob >0.4)
submit.fraud_ind.value_counts()


VIP = pd.DataFrame([(x,y) for (x,y) in zip(X.columns,final_model.feature_importances_)]).sort_values(by=1,ascending=False)
display(VIP[VIP.iloc[:,1]>0])


submit.to_csv('submit_xgq6461.csv',index = False)


# record 
print(f'# {final_model.learning_rate}, {final_model.max_depth} ,{final_model.scale_pos_weight}   ,{final_model.min_child_weight}    ,{final_model.n_estimators}  ,  ')




# eta ,dep,scale,child,nround, cv result                             , csv          , lb        , meth, feature  
# 0.02,15 ,15   ,1    ,5000  , (prob: 0.77, F1 : 0.781, acc : 0.994) , submit_l7698 , 0.527928  , 1   , test 1,2,5
# 0.02,15 ,15   ,1    ,5000  , (prob: 0.78, F1 : 0.751, acc : 0.993) , submit_l5647 , 0.567554  , 1   , test 5
# 0.03,15 ,10   ,1    ,9000  , (prob: 0.68, F1 : 0.784, acc : 0.994) , submit_l4835 , 0.544638  , 1   , test 5
# 0.01, 5 ,10   ,3    ,9000  , (prob: 0.74, F1 : 0.730, acc : 0.993) , submit_l6260 , 0.566163  , 1   , test 5
# 0.01, 5 ,10   ,3    ,9000  , (prob: 0.77, F1 : 0.714, acc : 0.993) , submit_l6524 , 0.574345  , 1   , -txkey + test 5
# ------------------------------------------------  fix cv split  -------------------------------------------------------------
# 0.01, 5 ,10   ,3    ,9000  , (prob: 0.77, F1 : 0.605, acc : 0.990) , submit_l6524 , 0.574345  , 2   , -txkey + test 5
# 0.03, 5 ,10   ,5    ,3000  , (prob: 0.71, F1 : 0.601, acc : 0.990) , submit_x7809 , 0.579507  , 2   , -txkey + test 5
# 0.01, 5 ,10   ,3    ,9000  , (prob: 0.81, F1 : 0.621, acc : 0.991) , submit_l5831 , 0.578424  , 2   , -txkey + test 5 + new
# 0.01, 7 ,10   ,1    ,5000  , (prob: 0.70, F1 : 0.613, acc : 0.991) , submit_x7161 , 0.587242  , 2   , -txkey + test 5
# -------------------------------------------------  fix loctm  ---------------------------------------------------------------
# 0.01, 5 ,10   ,3    ,9000  , (prob: 0.79, F1 : 0.615, acc : 0.991) , submit_l6246 , 0.582314  , 2   , -txkey + test 5 + new
# 0.01, 7 ,10   ,1    ,5000  , (prob: 0.71, F1 : 0.622, acc : 0.991) , submit_x7084 , 0.584780  , 2   , -txkey + test 5 + new2
# ------------------------------------------------  fix cv split  -------------------------------------------------------------
# 0.01, 5 ,10   ,3    ,9000  , (prob: 0.69, F1 : 0.466, acc : 0.990) , submit_l6246 , 0.582314  , 4   , -txkey + test 5 + new
# 0.01, 7 ,20   ,3    ,9000  , (prob: 0.81, F1 : 0.478, acc : 0.991) , submit_l6556 , 0.562601  , 4   , test 5 + new3
# 0.02, 5 ,10   ,5    ,5000  , (prob: 0.63, F1 : 0.463, acc : 0.989) , submit_x9470 , 0.567734  , 4   , -txkey + test 5 + new2
# 0.01, 15,10   ,3    ,9000  , (prob: 0.63, F1 : 0.492, acc : 0.991) , submit_l7607 , 0.572729  , 4   , test 5 + new2
# 0.01, 10,10   ,1    ,9000  , (prob: 0.22, F1 : 0.466, acc : 0.990) , submit_x9847 , 0.558851  , 4   , -txkey + test 5 + new2
# 0.01, 8 ,10   ,1    ,9000  , (prob: 0.65, F1 : 0.494, acc : 0.991) , submit_l7228 , 0.572884  , 4   , -txkey + test 5 + new2
# 0.01, 8 ,9    ,5    ,9000  , (prob: 0.49, F1 : 0.508, acc : 0.991) , submit_l8952 , 0.574535  , 4   , -txkey + new999
# 0.01, 8 ,9    ,5    ,9000  , (prob: 0.60, F1 : 0.508, acc : 0.991) , submit_lq7189, 0.590527  , 4   , -txkey + new999
# 0.01, 8 ,9    ,5    ,9000  , (prob: 0.68, F1 : 0.508, acc : 0.991) , submit_lq6489, 0.596744  , 4   , -txkey + new999
# 0.03, 9 ,11   ,1    ,2000  , (prob: 0.22, F1 : 0.523, acc : 0.991) , submit_xg8437, 0.570375  , 4   , -txkey + new999 
# 0.03, 9 ,11   ,1    ,5000  , (prob: 0.22, F1 : 0.531, acc : 0.992) , submit_xg6177, 0.592118  , 4   , -txkey + new999 
# 0.03, 9 ,11   ,1    ,5000  , (prob: 0.11, F1 : 0.544, acc : 0.992) , submit_xg9081, 0.548502  , 4   , new

        


'''
def objective(params,X_train=X_train,y_train=y_train,FOLDS = 5,cvmethod='ts'):
    X_train = pd.DataFrame(X_train)
    y_train = pd.DataFrame(y_train)
    time1 = time.time()
    params = {
        'max_depth': int(params['max_depth']),
        'gamma': "{:.3f}".format(params['gamma']),
        'subsample': "{:.2f}".format(params['subsample']),
        'reg_alpha': "{:.3f}".format(params['reg_alpha']),
        'reg_lambda': "{:.3f}".format(params['reg_lambda']),
        'learning_rate': "{:.3f}".format(params['learning_rate']),
        'num_leaves': '{:.3f}'.format(params['num_leaves']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
        'min_child_samples': '{:.3f}'.format(params['min_child_samples']),
        'feature_fraction': '{:.3f}'.format(params['feature_fraction']),
        'bagging_fraction': '{:.3f}'.format(params['bagging_fraction'])
    }

    print("\n############## New Run ################")
    print(f"params = {params}")
    count=1
    
    if cvmethod =='ts':
        cvsplit = TimeSeriesSplit(n_splits=FOLDS)
    elif cvmethod =='random':
        cvsplit = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

    #y_preds = np.zeros(sample_submission.shape[0])
    #y_oof = np.zeros(X_train.shape[0])
    score_mean = 0
    score_mean_test = 0
    for tr_idx, val_idx in cvsplit.split(X_train, y_train):
        clf = xgb.XGBClassifier(
            n_estimators=1500, random_state=4, verbose=True, 
            tree_method='gpu_hist', 
            **params
        )

        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        clf.fit(X_tr, y_tr)
        #y_pred_train = clf.predict_proba(X_vl)[:,1]
        #print(y_pred_train)
        score = make_scorer(f1_score, needs_proba=False)(clf, X_vl, y_vl)
        score_test = make_scorer(f1_score, needs_proba=False)(clf, pd.DataFrame(X_test), pd.DataFrame(y_test))
        score_mean_test += score_test
        # plt.show()
        score_mean += score
        print(f'{count} CV - score: {round(score, 4)}')
        print(f'{count} CV - score: {round(score_test, 4)}')
        count += 1
    time2 = time.time() - time1
    print(f"Total Time Run: {round(time2 / 60,2)} mins")
    gc.collect()
    print(f'Mean F1 : {score_mean_test / FOLDS}')
    del X_tr, X_vl, y_tr, y_vl, clf, score
    return (-( score_mean / FOLDS) -( score_mean_test/ FOLDS) )/2

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10)
# Print best parameters
best_params = space_eval(space, best)
best_params






clf = xgb.XGBClassifier(bagging_fraction= 0.7721447342857187,
                        colsample_bytree= 0.8673257590608814,
                        feature_fraction= 0.45256780044353817,
                        gamma= 0.301448930827109,
                        learning_rate= 0.04523906242170635,
                        max_depth= 22,
                        min_child_samples= 220,
                        num_leaves= 50,
                        reg_alpha= 0.03820567101015298,
                        reg_lambda= 0.39803679318040613,
                        subsample= 0.9,
                        n_estimators=2000,
                        random_state=4, 
                        verbose=True, 
                        tree_method='gpu_hist')
clf.fit(X, y)






y_prob = clf.predict_proba( df[df.fraud_ind.isnull()][feature])[:,1]
sns.distplot(y_prob)
submit = pd.read_csv('submission_test.csv')

submit.fraud_ind = np.int_(y_prob >0.1)
submit.fraud_ind.value_counts()

VIP = pd.DataFrame([(x,y) for (x,y) in zip(X.columns,clf.feature_importances_)]).sort_values(by=1,ascending=False)
display(VIP[VIP.iloc[:,1]>0])


submit.to_csv('submit_xgq6575.csv',index = False)
'''
