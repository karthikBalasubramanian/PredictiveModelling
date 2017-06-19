#!/home/bks4line/miniconda3/bin/python
# Author: Karthik Balasubramanian
import sys
import pandas as pd
import sklearn
import sklearn.preprocessing
import numpy as np
import seaborn as sns
import operator
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn import linear_model
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
#%matplotlib inline

seed = 7
np.random.seed(seed)



def get_processed_df(dataFrame):


    df = dataFrame.copy()
    
    angle_df = pd.get_dummies(df["device_angle"])
    angle_df.rename(columns={1:"less_thirty",2:"thirty_sixty",3:"sixty_ninety"},inplace=True)
    
    distance_df = pd.get_dummies(df["distance_to_door"])
    distance_df.rename(columns={1:"less_two_m",2:"two_four_m",3:"gt_four_m"},inplace=True)
    
    period_df = pd.get_dummies(df["AM_or_PM"])
    period_df.rename(columns={0:"AM",1:"PM"},inplace=True)
    
    place_df = pd.get_dummies(df["mall_or_street"])
    place_df.rename(columns={1:"Mall",2:"Street"},inplace=True)
    
    new_df = pd.concat([df,angle_df,distance_df,period_df,place_df],axis=1)
    
    new_df.drop("device_angle",inplace=True,axis=1)
    new_df.drop("distance_to_door",inplace=True,axis=1)
    new_df.drop("AM_or_PM",inplace=True,axis=1)
    new_df.drop("mall_or_street",inplace=True,axis=1)
    
    new_df['average_person_size'] =  sklearn.preprocessing.scale(df['average_person_size']/(480*320))
    new_df['video_walkin'] =  sklearn.preprocessing.scale(df['video_walkin'])
    new_df['video_walkout'] = sklearn.preprocessing.scale(df['video_walkout'])
    new_df['predict_walkin'] = sklearn.preprocessing.scale(df['predict_walkin'])
    new_df['predict_walkout'] = sklearn.preprocessing.scale(df['predict_walkout'])
    new_df['wifi_walkin'] = sklearn.preprocessing.scale(df['wifi_walkin'])
    new_df['wifi_walkout'] = sklearn.preprocessing.scale(df['wifi_walkout'])
    new_df['sales_in_next_15_min'] = sklearn.preprocessing.scale(df['sales_in_next_15_min'])
    new_df['sales_in_next_15_to_30_min'] = sklearn.preprocessing.scale(df['sales_in_next_15_to_30_min'])
    
    
    return new_df



def train_test_splits(df):
    Y = np.array(df['groundtruth_walkin'])
    X = df.ix[:, df.columns != 'groundtruth_walkin'].as_matrix()
    return train_test_split(X, Y, test_size=0.3, random_state=0)



def all_linear_Models(X_train,y_train,X_cv,y_cv,case):
    m_dict = {}
    baseline_mean_train = np.mean(y_train)* np.ones(len(y_train))
    baseline_mean_cv =  np.mean(y_train)* np.ones(len(y_cv))
    
    
    case = case
    reg =  None
    # Lasso Regression => Avoid features when they are not significant, Modeled with Default params
    # Has L1 regulariser. Does not have L2 Norm involved. Generally has no effect on 
    if case==1:
        reg = linear_model.LassoCV()
        reg.fit(X_train,y_train)
        m_dict["Lasso_cv_r2"] =  reg.score(X_cv,y_cv)
        m_dict["Lasso_train_r2"] =reg.score(X_train,y_train)
        case+=1
        reg=None
    
    # ElasticNet => Utilises both L1 and L2 penalty as prior regularizer constant.
    # ElasticNetCV => decides which Regulariser to weigh based on the l1_ratio_ value
    # if l1_ratio_== 0 then the model will be devoid of l1 penalty
    # else if l1_ratio ==1 then model will be devoid of l2 penalty.
    
    if case==2:
        reg = linear_model.ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1])
        reg.fit(X_train,y_train)
        m_dict["ElasticNet_cv_r2"] =  reg.score(X_cv,y_cv)
        m_dict["ElasticNet_train_r2"] =reg.score(X_train,y_train)
        case+=1
        reg = None
    
    # SGD Regressor => Works for Many datapoints with multiple features ( for a n*p matrix both n and p has to be high)
    # Loss function used => Huber loss, which means to calculate L2 loss till a threshold (epsilon =0.8) and calculate L1 after
    # the threshold to avoid the effect of outliers in the model
    if case==3:
        reg = linear_model.SGDRegressor(loss='huber',epsilon=0.8)
        reg.fit(X_train,y_train)
        m_dict["SGD_cv_r2"] =  reg.score(X_cv,y_cv)
        m_dict["SGD_train_r2"] =reg.score(X_train,y_train)
        case+=1
        reg = None
        
    
    # Huber Regressor => is better than SGD regressor with lesser huber loss for lesser number of datapoints
    # Its scale invariant. Hence using Huber Regressor
    if case==4:
        reg = linear_model.HuberRegressor()
        reg.fit(X_train,y_train)
        m_dict["Huber_cv_r2"] =  reg.score(X_cv,y_cv)
        m_dict["Huber_train_r2"] =reg.score(X_train,y_train)
        case+=1
        reg = None
    
    # Trying Ensemble Weak learners => Gradient Boosting Regressor
    if case==5:
        reg = GradientBoostingRegressor(n_estimators=200)
        reg.fit(X_train,y_train)
        
        kfold = KFold(n_splits=5, random_state=seed)
        m_dict['Avg_5_fold_GBR_r2']= np.mean(cross_val_score(reg,X_train,y_train,cv=kfold))
        m_dict["GBR_cv_r2"] =  reg.score(X_cv,y_cv)
        m_dict["GBR_train_r2"] =reg.score(X_train,y_train)
        df = pd.DataFrame({"gbr_predict_cv":reg.predict(X_cv),"actual":y_cv})
        case+=1
        
        
        
        
    
    return (m_dict,df,reg)




def baseline_nn_model():
    
    # create model
    model = Sequential()
    model.add(Dense(25, input_dim=18, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1,kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='rmsprop',metrics=['mse','accuracy'])
    model.save("baseline_nn.h5")
    return model



def deep_nn_model():
    
    # create model
    model = Sequential()
    model.add(Dense(25, input_dim=18, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1 ,kernel_initializer='normal'))
    # Compile model
    
    model.compile(loss='mean_squared_error', optimizer='rmsprop',metrics=['mse','accuracy'])
    model.save("dense_model.h5")
    return model



def evaluate_nn_model(model_obj,x_tr,x_cv,y_tr,y_cv,t_df=pd.DataFrame()):
    y_test = None
    estimator = KerasRegressor(build_fn=model_obj,epochs=300, batch_size=32, verbose=0)
    estimator.fit(x_tr,y_tr,validation_split=0.2) 
    x_train_pred =  estimator.predict(x_tr)
    x_cv_pred = estimator.predict(x_cv)
    if not t_df.empty:
        
        test_df_pred = t_df.values
        y_test = np.floor(estimator.predict(test_df_pred))
        
    return(r2_score(y_tr,x_train_pred),r2_score(y_cv,x_cv_pred),y_test)
    



if __name__=="__main__":
    
    with warnings.catch_warnings():
        
        warnings.simplefilter("ignore")
        train_df = pd.read_csv(sys.argv[-2])
        test_df = pd.read_csv(sys.argv[-1])
        normalised_tdf = get_processed_df(train_df)
        norm_test_df =  get_processed_df(test_df)
        X_train, X_cv, y_train, y_cv = train_test_splits(normalised_tdf)
        model_dict,df,model = all_linear_Models(X_train,y_train,X_cv,y_cv,1)
        print("Mean Squared Error for Gradient Boosting is {0} ".format(sklearn.metrics.mean_squared_error(df.actual,df.gbr_predict_cv)))
        new_model_list = sorted(model_dict.items(), key=operator.itemgetter(1), reverse=True)[:2]
        print("Best Model - Gradient Boosting Train set R2 score {0} and holdout set R2 score {1}".format(new_model_list[0][1],new_model_list[1][1]))
        test_df["walk_in_pred_GBR"] =np.floor(model.predict(norm_test_df))

        r2_train, r2_holdout, y_test = evaluate_nn_model(baseline_nn_model,X_train,X_cv,y_train,y_cv)
        print("R2 for Baseline NN Train is {0} and R2 for Baseline NN holdout is {1}".format(r2_train,r2_holdout))
        
        r2_deep_train, r2_deep_holdout,y_test = evaluate_nn_model(deep_nn_model,X_train,X_cv,y_train,y_cv,t_df=norm_test_df)
        print("R2 for deep NN Train is {0} and R2 for Baseline NN holdout is {1}".format(r2_deep_train,r2_deep_holdout))
        test_df["walk_in_pred_NN"] = y_test
        test_df.to_csv("test_data_solution.csv")



