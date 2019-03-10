from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook, tnrange
import pandas as pd
import numpy as np
import warnings
import datetime
import gc

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype(np.str)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df
def downCast_dtype(df):
    float_cols = [c for c in df if df[c].dtype == 'float64']
    int_cols = [c for c in df if df[c].dtype == 'int64']
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int32)
    return df

# def dateUtils(df=None,timeCol='purchase_date'):
#     dateHandle = pd.to_datetime(df[timeCol])
#     df['week'] = dateHandle.dt.week
#     df['year'] = dateHandle.dt.year
#     df['month_gap'] = (dateHandle.dt.date - datetime.date(2018,2,28)).dt.days//30
#     df['day_gap'] = (dateHandle.dt.date - datetime.date(2018,2,28)).dt.days
#     #cardid用户连续购买之间的时间差
#     roll = df.groupby(['card_id'])['day_gap'].apply(lambda series:series.diff(1))
#     df['day_diff'] = roll.values
#     return df

def label_encoding(df,encodCols):
    for col in tqdm_notebook(encodCols):
        lbl = LabelEncoder()
        lbl.fit(list(df[col].values.astype('str')))
        df[col] = lbl.transform(list(df[col].values.astype('str')))
    return df
def getMeanStaticsFeatures(df_data,df_feature,group,fea='',name=''):
    df_temp = df_feature.groupby(group)[fea].mean().reset_index()
    df_temp.rename(columns={fea:name},inplace=True)
    df_data = df_data.merge(df_temp,on=group,how='left')
    return df_data
def getStdStaticsFeatures(df_data,df_feature,group,fea='',name=''):
    df_temp = df_feature.groupby(group)[fea].std().reset_index()
    df_temp.rename(columns={fea:name},inplace=True)
    df_data = df_data.merge(df_temp,on=group,how='left')
    return df_data
def getMaxStaticsFeatures(df_data,df_feature,group,fea='',name=''):
    df_temp = df_feature.groupby(group)[fea].max().reset_index()
    df_temp.rename(columns={fea:name},inplace=True)
    df_data = df_data.merge(df_temp,on=group,how='left')
    return df_data
def getMedianStaticsFeatures(df_data,df_feature,group,fea='',name=''):
    df_temp = df_feature.groupby(group)[fea].median().reset_index()
    df_temp.rename(columns={fea:name},inplace=True)
    df_data = df_data.merge(df_temp,on=group,how='left')
    return df_data
def getMinStaticsFeatures(df_data,df_feature,group,fea='',name=''):
    df_temp = df_feature.groupby(group)[fea].min().reset_index()
    df_temp.rename(columns={fea:name},inplace=True)
    df_data = df_data.merge(df_temp,on=group,how='left')
    return df_data
def getSumStaticsFeatures(df_data,df_feature,group,fea='',name=''):
    df_temp = df_feature.groupby(group)[fea].sum().reset_index()
    df_temp.rename(columns={fea:name},inplace=True)
    df_data = df_data.merge(df_temp,on=group,how='left') 
    return df_data
def getQuantileStaticsFeatures(df_data,df_feature,group,fea='',name=''):
    df_temp = df_feature.groupby(group)[fea].quantile(0.2).reset_index()
    df_temp.rename(columns={fea:name},inplace=True)
    df_data = df_data.merge(df_temp,on=group,how='left')
    return df_data
def getCountsStaticsFeatures(df_data,df_feature,group,fea='',name=''):
    df_temp = df_feature.groupby(group)[fea].count().reset_index()
    df_temp.rename(columns={fea:name},inplace=True)
    df_data = df_data.merge(df_temp,on=group,how='left')
    return df_data
#统计用户刷信用卡的主要商店和商店类型
def getCategoryFrequenceMax(df_data,df_feature,group,fea='',name=''):
    df_temp = df_feature.groupby(group)[fea].apply(lambda series:series.value_counts(dropna=False).index[0]).reset_index()
    df_temp.rename(columns={fea:name},inplace=True)
    df_data = df_data.merge(df_temp,on=group,how='left')
    return df_data
def getCategoryCounts(df_data,df_feature,group,fea='',name=''):
    df_temp = df_feature.groupby(group)[fea].apply(lambda series:len(series.unique())).reset_index()
    df_temp.rename(columns={fea:name},inplace=True)
    df_data = df_data.merge(df_temp,on=group,how='left')
    return df_data
#历史访问最多的店的次数所占比例
def getCategoryFrequenceMaxRatio(df_data,df_feature,group,fea='',name=''):
    df_temp = df_feature.groupby(group)[fea].apply(lambda series:list(series.value_counts(dropna=False).values)[0]/series.shape[0]).reset_index()
    df_temp.rename(columns={fea:name},inplace=True)
    df_data = df_data.merge(df_temp,on=group,how='left')
    return df_data
#历史访问的多类点所占的比例
def getCategoryCountsRatio(df_data,df_feature,group,fea='',name=''):
    df_temp = df_feature.groupby(group)[fea].apply(lambda series:len(series.unique())/series.shape[0]).reset_index()
    df_temp.rename(columns={fea:name},inplace=True)
    df_data = df_data.merge(df_temp,on=group,how='left')
    return df_data

