
# coding: utf-8

# In[33]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.datasets import make_classification

class FFMFormatPandas:
    def __init__(self):
        self.field_index_ = None
        self.feature_index_ = None
        self.y = None

    def fit(self, df, y=None):
        self.y = y
        df_ffm = df[df.columns.difference([self.y])]
        if self.field_index_ is None:
            self.field_index_ = {col: i for i, col in enumerate(df_ffm)}

        if self.feature_index_ is not None:
            last_idx = max(list(self.feature_index_.values()))

        if self.feature_index_ is None:
            self.feature_index_ = dict()
            last_idx = 0

        for col in df.columns:
            vals = df[col].unique()
            for val in vals:
                if pd.isnull(val):
                    continue
                name = '{}_{}'.format(col, val)
                if name not in self.feature_index_:
                    self.feature_index_[name] = last_idx
                    last_idx += 1
            self.feature_index_[col] = last_idx
            last_idx += 1
        return self

    def fit_transform(self, df, y=None):
        self.fit(df, y)
        return self.transform(df)

    def transform_row_(self, row, t):
        ffm = []
        if self.y != None:
            ffm.append(str(row.loc[row.index == self.y][0]))
        if self.y is None:
            ffm.append(str(0))

        for col, val in row.loc[row.index != self.y].to_dict().items():
            col_type = t[col]
            name = '{}_{}'.format(col, val)
            if col_type.kind ==  'O':
                ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
#             elif col_type.kind == 'i':
            else:
                ffm.append('{}:{}:{}'.format(self.field_index_[col], self.feature_index_[col], val))
        return ' '.join(ffm)

    def transform(self, df):
        t = df.dtypes.to_dict()
        return pd.Series({idx: self.transform_row_(row, t) for idx, row in df.iterrows()})

def dtype_to_str(df,categoryCols):
    for col in categoryCols:
        df[col] = df[col].map(str)
    return df
print(".....Start FFM...")

len_train = df_data[df_data.is_test==0].shape[0]

df_data_ffm = df_data.copy()

dropCols = ['card_id','first_active_month','is_test','is_outlier']
tr_features = [_f for _f in df_data_ffm.columns if _f not in dropCols]
categoryCols = ['feature_1','feature_2','feature_3','start_year','start_month','feature_1_2','merchant_category_id_frequenceMax_byCardId'
               ,'category_3_frequenceMax_byCardId','category_2_frequenceMax_byCardId','city_id_frequenceMax_byCardId',
               'category_1_frequenceMax_byCardId','authorized_flag_frequenceMax_byCardId','monthLageFrequenceMax_byCardId']
numCols = [_f for _f in tr_features if _f not in categoryCols]

df_data_ffm = dtype_to_str(df_data_ffm,categoryCols)
df_data_ffm = df_data_ffm[tr_features]

df_data_ffm[numCols] = df_data_ffm[numCols].apply(lambda series:series-np.min(series)/(np.max(series)-np.min(series)))

ffm_utils = FFMFormatPandas()
ffm_data = ffm_utils.fit_transform(df_data_ffm,y='target')

ffm_data.to_csv(DATA_PATH+'ffm_data.csv',index=False)
print('......done！....')
#文件保存
with open(DATA_PATH+'ffm_data.csv') as fin:
    df_ffm_train = open(DATA_PATH+'df_ffm_train.csv','w')
    df_ffm_test = open(DATA_PATH+'df_ffm_test.csv','w')
    
    for (i,line) in enumerate(fin):
        if i<len_train:
            df_ffm_train.write(line)
        else:
            df_ffm_test.write(line)
    df_ffm_train.close()
    df_ffm_test.close()


# In[ ]:


from sklearn import preprocessing
import xlearn as xl
ffm_model = xl.create_ffm()
ffm_model.setTrain('./datasets/df_ffm_train.csv')
ffm_model.setTest('./datasets/df_ffm_test.csv')

params = {'task':'reg','lr':0.2,'metric':'rmse'}
ffm_model.fit(params,'./ffm_model.out')

ffm_model.predict('./ffm_model.out','./submission/ffm_predict.txt')

# create submission file
df_sub = pd.read_csv('./datasets/sample_submission.csv')
df_sub['target'] = np.loadtxt('./submission/ffm_predict.txt')
df_sub.fillna(method='ffill',inplace=True)
# df_sub.HasDetections = pred
df_sub.to_csv('./submission/df_ffm_submission.csv', index=False)

