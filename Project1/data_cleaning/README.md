```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import random
random.seed(1234)
```


```python
def get_summary(dat):

    dtypes_dict = dat.dtypes.apply(lambda x: x.name).to_dict()

    numeric = [x for x in dtypes_dict.keys() if dtypes_dict[x] == 'float64' or dtypes_dict[x] == 'int64']
    chars = [x for x in dtypes_dict.keys() if dtypes_dict[x] == 'object']

    type_numeric = ['numeric'] * len(numeric)
    type_chars = ['char'] * len(chars)

    numeric_counts = dat.loc[:, numeric].count().to_dict()
    numeric_populated = list(numeric_counts.values())
    numeric_pct = [str((round(x / len(dat), 2)) * 100) + '%' for x in numeric_populated]
    numeric_zero = [sum(dat.loc[:, x] == 0) for x in numeric]

    numeric_s = pd.DataFrame(list(zip(numeric, type_numeric, numeric_populated, numeric_pct, numeric_zero)),
                             columns=['Name', 'Type', 'Non_NA_Values', 'Pct_Non_NA', 'Num_Zeros'])

    numeric_cols_df = pd.concat([numeric_s,
                                 dat.describe().transpose().reset_index().drop(['count', 'index'], axis=1)],
                                axis=1)

    char_counts = dat.loc[:, chars].count().to_dict()
    char_populated = list(char_counts.values())
    char_pct = [str((round(x / len(dat), 3)) * 100) + '%' for x in char_populated]
    char_zero = [sum(dat.loc[:, x] == 0) for x in chars]

    chars_unique = [len(dat.loc[:, x].unique()) for x in chars]
    chars_val_max = [dat.loc[:, x].value_counts().idxmax() for x in chars]

    chars_df = pd.DataFrame(
        list(zip(chars, type_chars, char_populated, char_pct, char_zero, chars_unique, chars_val_max)),
        columns=['Name', 'Type', 'Non_NA_Values', 'Pct_Non_NA', 'Num_Zeros', 'Unique', 'Most_Common'])

    which = input('Would you like a table of numeric or char columns? (N/C)')
    if which == 'N':
        return numeric_cols_df
    elif which == 'C':
        return chars_df
```


```python
df = pd.read_csv('Project1_NY property data.csv')
```


```python
# Repace NAN's with 0's for ZIP only
df['ZIP'] = df['ZIP'].replace(np.nan,0 )

# Groups by Boro and Lot and get the mode
df['c4'] = df.groupby(['B', 'BLOCK'])['ZIP'].transform(lambda x: pd.Series.mode(x)[0])
# Groups by Boro to get MODE
df['c5'] = df.groupby(['B'])['ZIP'].transform(lambda x: pd.Series.mode(x)[0])

# Replaces 0 values with C4 then C5
df['ZIP'] = np.where(df['ZIP'] == 0, 
                      np.where(df['c4'] == 0, df['c5'], df['c4']),
                      df['ZIP'])
# Drops created columns
df = df.drop(['c4', 'c5'], axis = 1)

# Change to categorical
df['ZIP'] = df["ZIP"].astype('int').astype('category')

# Create ZIP 3 as Category
df['ZIP3'] = df['ZIP'].astype(str).str[0:3].astype('category')

```


```python
# Replaces 0's with NAN for mean gathering. 
df["FULLVAL"] = df["FULLVAL"].replace(0, np.nan)
df["AVLAND"] = df["AVLAND"].replace(0, np.nan)
df["AVTOT"] = df["AVTOT"].replace(0, np.nan)
df["STORIES"] = df["STORIES"].replace(0, np.nan)
df["LTFRONT"] = df["LTFRONT"].replace(0, np.nan)
df["LTDEPTH"] = df["LTDEPTH"].replace(0, np.nan)
df["BLDFRONT"] = df["BLDFRONT"].replace(0, np.nan)
df["BLDDEPTH"] = df["BLDDEPTH"].replace(0, np.nan)

# Fill in 0's for FULLVAL, AVLAND, AVTOT, STORIES, LTFRONT, LTDEPTH, BLDFRONT, BLDDEPTH with mean of group
df["FULLVAL"] = df.groupby(['B', 'ZIP'])['FULLVAL'].transform(lambda x: x.fillna(x.mean()))
df["AVLAND"] = df.groupby(['B', 'ZIP'])['AVLAND'].transform(lambda x: x.fillna(x.mean()))
df["AVTOT"] = df.groupby(['B', 'ZIP'])['AVTOT'].transform(lambda x: x.fillna(x.mean()))
df["STORIES"] = df.groupby(['B','ZIP'])['STORIES'].transform(lambda x: x.fillna(x.median()))
df["STORIES"] = df.groupby(['B'])['STORIES'].transform(lambda x: x.fillna(x.median()))
df["LTFRONT"] = df.groupby(['B', 'ZIP'])['LTFRONT'].transform(lambda x: x.fillna(x.median()))
df["LTDEPTH"] = df.groupby(['B', 'ZIP'])['LTDEPTH'].transform(lambda x: x.fillna(x.median()))
df["BLDFRONT"] = df.groupby(['ZIP'])['BLDFRONT'].transform(lambda x: x.fillna(x.median()))
df["BLDDEPTH"] = df.groupby([ 'ZIP'])['BLDDEPTH'].transform(lambda x: x.fillna(x.median()))

#Replaces last 0's with BLDGCL global mean

df["BLDFRONT"] = df.groupby(['BLDGCL'])['BLDFRONT'].transform(lambda x: x.fillna(x.median()))
df["BLDDEPTH"] = df.groupby([ 'BLDGCL'])['BLDDEPTH'].transform(lambda x: x.fillna(x.median()))
```


```python
# Create lotarea, bldarea, bldvol
df['lotarea'] = df['LTFRONT'] * df['LTDEPTH']
df['bldarea'] = df['BLDFRONT'] * df['BLDDEPTH']
df['bldvol'] = df['bldarea'] * df['STORIES']
```


```python
# Create 9 variables

df['FULLVAL_lotarea'] = df['FULLVAL'] / df['lotarea']
df['FULLVAL_bldarea'] = df['FULLVAL'] / df['bldarea']
df['FULLVAL_bldvol'] = df['FULLVAL'] / df['bldvol']
df['AVLAND_Lotarea'] = df['AVLAND'] / df['lotarea']
df['AVLAND_bldarea'] = df['AVLAND'] / df['bldarea']
df['AVLAND_bldvol'] = df['AVLAND'] / df['bldvol']
df['AVTOT_lotarea'] = df['AVTOT'] / df['lotarea']
df['AVTOT_bldarea'] = df['AVTOT'] / df['bldarea']
df['AVTOT_bldvol'] = df['AVTOT'] / df['bldvol']
```


```python
columns = ['FULLVAL_lotarea','FULLVAL_bldarea', 'FULLVAL_bldvol', 'AVLAND_Lotarea', 'AVLAND_bldarea', \
           'AVLAND_bldvol', 'AVTOT_lotarea', 'AVTOT_bldarea', 'AVTOT_bldvol']
```


```python
# Loops to create all variable combinations 
for col in columns: 
    df['zip5_' + col] = df[col] / df.groupby('ZIP')[col].transform(lambda x: x.mean())

for col in columns: 
    df['zip3_' + col] = df[col] / df.groupby('ZIP3')[col].transform(lambda x: x.mean())

for col in columns: 
    df['taxclass_' + col] = df[col] / df.groupby('TAXCLASS')[col].transform(lambda x: x.mean())
    
for col in columns: 
    df['borough_' + col] = df[col] / df.groupby('B')[col].transform(lambda x: x.mean())

for col in columns: 
    df['all_' + col] = df[col] / df[col].mean()
```


```python
df.columns
```




    Index(['RECORD', 'BBLE', 'B', 'BLOCK', 'LOT', 'EASEMENT', 'OWNER', 'BLDGCL',
           'TAXCLASS', 'LTFRONT', 'LTDEPTH', 'EXT', 'STORIES', 'FULLVAL', 'AVLAND',
           'AVTOT', 'EXLAND', 'EXTOT', 'EXCD1', 'STADDR', 'ZIP', 'EXMPTCL',
           'BLDFRONT', 'BLDDEPTH', 'AVLAND2', 'AVTOT2', 'EXLAND2', 'EXTOT2',
           'EXCD2', 'PERIOD', 'YEAR', 'VALTYPE', 'ZIP3', 'lotarea', 'bldarea',
           'bldvol', 'FULLVAL_lotarea', 'FULLVAL_bldarea', 'FULLVAL_bldvol',
           'AVLAND_Lotarea', 'AVLAND_bldarea', 'AVLAND_bldvol', 'AVTOT_lotarea',
           'AVTOT_bldarea', 'AVTOT_bldvol', 'zip5_FULLVAL_lotarea',
           'zip5_FULLVAL_bldarea', 'zip5_FULLVAL_bldvol', 'zip5_AVLAND_Lotarea',
           'zip5_AVLAND_bldarea', 'zip5_AVLAND_bldvol', 'zip5_AVTOT_lotarea',
           'zip5_AVTOT_bldarea', 'zip5_AVTOT_bldvol', 'zip3_FULLVAL_lotarea',
           'zip3_FULLVAL_bldarea', 'zip3_FULLVAL_bldvol', 'zip3_AVLAND_Lotarea',
           'zip3_AVLAND_bldarea', 'zip3_AVLAND_bldvol', 'zip3_AVTOT_lotarea',
           'zip3_AVTOT_bldarea', 'zip3_AVTOT_bldvol', 'taxclass_FULLVAL_lotarea',
           'taxclass_FULLVAL_bldarea', 'taxclass_FULLVAL_bldvol',
           'taxclass_AVLAND_Lotarea', 'taxclass_AVLAND_bldarea',
           'taxclass_AVLAND_bldvol', 'taxclass_AVTOT_lotarea',
           'taxclass_AVTOT_bldarea', 'taxclass_AVTOT_bldvol',
           'borough_FULLVAL_lotarea', 'borough_FULLVAL_bldarea',
           'borough_FULLVAL_bldvol', 'borough_AVLAND_Lotarea',
           'borough_AVLAND_bldarea', 'borough_AVLAND_bldvol',
           'borough_AVTOT_lotarea', 'borough_AVTOT_bldarea',
           'borough_AVTOT_bldvol', 'all_FULLVAL_lotarea', 'all_FULLVAL_bldarea',
           'all_FULLVAL_bldvol', 'all_AVLAND_Lotarea', 'all_AVLAND_bldarea',
           'all_AVLAND_bldvol', 'all_AVTOT_lotarea', 'all_AVTOT_bldarea',
           'all_AVTOT_bldvol'],
          dtype='object')




```python
get_summary(df.iloc[:, -45:])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Type</th>
      <th>Non_NA_Values</th>
      <th>Pct_Non_NA</th>
      <th>Num_Zeros</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>zip5_FULLVAL_lotarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>5.755436</td>
      <td>1.715544e-06</td>
      <td>0.452184</td>
      <td>0.891784</td>
      <td>1.203736</td>
      <td>2506.256950</td>
    </tr>
    <tr>
      <th>1</th>
      <td>zip5_FULLVAL_bldarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>9.999132</td>
      <td>2.400407e-06</td>
      <td>0.378315</td>
      <td>0.842002</td>
      <td>1.143872</td>
      <td>4534.557129</td>
    </tr>
    <tr>
      <th>2</th>
      <td>zip5_FULLVAL_bldvol</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>11.192967</td>
      <td>5.600924e-07</td>
      <td>0.346875</td>
      <td>0.809131</td>
      <td>1.127279</td>
      <td>5365.503929</td>
    </tr>
    <tr>
      <th>3</th>
      <td>zip5_AVLAND_Lotarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>13.342164</td>
      <td>1.911487e-06</td>
      <td>0.345289</td>
      <td>0.719050</td>
      <td>1.061286</td>
      <td>9305.941252</td>
    </tr>
    <tr>
      <th>4</th>
      <td>zip5_AVLAND_bldarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>21.930206</td>
      <td>7.255254e-07</td>
      <td>0.181404</td>
      <td>0.540334</td>
      <td>0.840824</td>
      <td>9686.446562</td>
    </tr>
    <tr>
      <th>5</th>
      <td>zip5_AVLAND_bldvol</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>24.193031</td>
      <td>2.670986e-07</td>
      <td>0.109533</td>
      <td>0.464009</td>
      <td>0.779872</td>
      <td>9247.088104</td>
    </tr>
    <tr>
      <th>6</th>
      <td>zip5_AVTOT_lotarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>11.535567</td>
      <td>1.029946e-06</td>
      <td>0.370859</td>
      <td>0.628945</td>
      <td>0.936608</td>
      <td>7191.342564</td>
    </tr>
    <tr>
      <th>7</th>
      <td>zip5_AVTOT_bldarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>19.066755</td>
      <td>3.178907e-06</td>
      <td>0.303617</td>
      <td>0.561409</td>
      <td>0.810192</td>
      <td>8928.919663</td>
    </tr>
    <tr>
      <th>8</th>
      <td>zip5_AVTOT_bldvol</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>20.568301</td>
      <td>1.044773e-06</td>
      <td>0.245034</td>
      <td>0.518043</td>
      <td>0.776108</td>
      <td>8604.935283</td>
    </tr>
    <tr>
      <th>9</th>
      <td>zip3_FULLVAL_lotarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>8.726385</td>
      <td>2.234429e-06</td>
      <td>0.429538</td>
      <td>0.846049</td>
      <td>1.200425</td>
      <td>6182.459163</td>
    </tr>
    <tr>
      <th>10</th>
      <td>zip3_FULLVAL_bldarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>15.490933</td>
      <td>1.134648e-06</td>
      <td>0.376550</td>
      <td>0.812228</td>
      <td>1.085280</td>
      <td>8253.802100</td>
    </tr>
    <tr>
      <th>11</th>
      <td>zip3_FULLVAL_bldvol</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>17.168823</td>
      <td>3.691279e-07</td>
      <td>0.327216</td>
      <td>0.801075</td>
      <td>1.100635</td>
      <td>9071.485398</td>
    </tr>
    <tr>
      <th>12</th>
      <td>zip3_AVLAND_Lotarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>25.180457</td>
      <td>1.647286e-06</td>
      <td>0.328324</td>
      <td>0.669091</td>
      <td>0.977569</td>
      <td>19607.668268</td>
    </tr>
    <tr>
      <th>13</th>
      <td>zip3_AVLAND_bldarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>95.859638</td>
      <td>1.002650e-06</td>
      <td>0.178230</td>
      <td>0.352340</td>
      <td>0.505491</td>
      <td>66140.696486</td>
    </tr>
    <tr>
      <th>14</th>
      <td>zip3_AVLAND_bldvol</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>93.465381</td>
      <td>2.486360e-07</td>
      <td>0.123178</td>
      <td>0.334733</td>
      <td>0.491166</td>
      <td>65825.583553</td>
    </tr>
    <tr>
      <th>15</th>
      <td>zip3_AVTOT_lotarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>17.247984</td>
      <td>9.220929e-07</td>
      <td>0.345366</td>
      <td>0.549733</td>
      <td>0.826116</td>
      <td>8448.886499</td>
    </tr>
    <tr>
      <th>16</th>
      <td>zip3_AVTOT_bldarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>79.691699</td>
      <td>2.242359e-06</td>
      <td>0.258196</td>
      <td>0.387610</td>
      <td>0.548704</td>
      <td>62719.566676</td>
    </tr>
    <tr>
      <th>17</th>
      <td>zip3_AVTOT_bldvol</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>76.033856</td>
      <td>7.985365e-07</td>
      <td>0.240355</td>
      <td>0.384925</td>
      <td>0.554924</td>
      <td>48567.365978</td>
    </tr>
    <tr>
      <th>18</th>
      <td>taxclass_FULLVAL_lotarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>7.297981</td>
      <td>1.854789e-06</td>
      <td>0.391202</td>
      <td>0.669578</td>
      <td>1.051511</td>
      <td>4118.254885</td>
    </tr>
    <tr>
      <th>19</th>
      <td>taxclass_FULLVAL_bldarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>19.118790</td>
      <td>3.461134e-06</td>
      <td>0.458105</td>
      <td>0.739149</td>
      <td>1.002372</td>
      <td>14394.758771</td>
    </tr>
    <tr>
      <th>20</th>
      <td>taxclass_FULLVAL_bldvol</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>63.853033</td>
      <td>1.189520e-06</td>
      <td>0.401414</td>
      <td>0.743226</td>
      <td>1.031437</td>
      <td>64313.363402</td>
    </tr>
    <tr>
      <th>21</th>
      <td>taxclass_AVLAND_Lotarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>11.235392</td>
      <td>9.289344e-07</td>
      <td>0.323034</td>
      <td>0.694833</td>
      <td>1.084974</td>
      <td>6668.244369</td>
    </tr>
    <tr>
      <th>22</th>
      <td>taxclass_AVLAND_bldarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>48.928628</td>
      <td>2.803143e-06</td>
      <td>0.259111</td>
      <td>0.763407</td>
      <td>1.056271</td>
      <td>47215.684196</td>
    </tr>
    <tr>
      <th>23</th>
      <td>taxclass_AVLAND_bldvol</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>124.912342</td>
      <td>5.866613e-07</td>
      <td>0.169176</td>
      <td>0.742718</td>
      <td>1.062068</td>
      <td>128461.164383</td>
    </tr>
    <tr>
      <th>24</th>
      <td>taxclass_AVTOT_lotarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>11.137646</td>
      <td>4.440048e-07</td>
      <td>0.391830</td>
      <td>0.718102</td>
      <td>1.094067</td>
      <td>6916.554637</td>
    </tr>
    <tr>
      <th>25</th>
      <td>taxclass_AVTOT_bldarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>22.526818</td>
      <td>4.458508e-06</td>
      <td>0.417788</td>
      <td>0.781227</td>
      <td>1.051203</td>
      <td>15019.675816</td>
    </tr>
    <tr>
      <th>26</th>
      <td>taxclass_AVTOT_bldvol</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>67.808826</td>
      <td>1.550303e-06</td>
      <td>0.351555</td>
      <td>0.768337</td>
      <td>1.059818</td>
      <td>67893.921021</td>
    </tr>
    <tr>
      <th>27</th>
      <td>borough_FULLVAL_lotarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>8.817915</td>
      <td>2.568530e-06</td>
      <td>0.416886</td>
      <td>0.821405</td>
      <td>1.207733</td>
      <td>6182.459163</td>
    </tr>
    <tr>
      <th>28</th>
      <td>borough_FULLVAL_bldarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>16.109654</td>
      <td>1.160949e-06</td>
      <td>0.379380</td>
      <td>0.805204</td>
      <td>1.083881</td>
      <td>8254.638298</td>
    </tr>
    <tr>
      <th>29</th>
      <td>borough_FULLVAL_bldvol</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>17.618766</td>
      <td>3.805181e-07</td>
      <td>0.325220</td>
      <td>0.788992</td>
      <td>1.105492</td>
      <td>9072.138306</td>
    </tr>
    <tr>
      <th>30</th>
      <td>borough_AVLAND_Lotarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>25.205630</td>
      <td>1.647286e-06</td>
      <td>0.318810</td>
      <td>0.645694</td>
      <td>0.984901</td>
      <td>19609.594021</td>
    </tr>
    <tr>
      <th>31</th>
      <td>borough_AVLAND_bldarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>94.546138</td>
      <td>1.042758e-06</td>
      <td>0.178578</td>
      <td>0.341024</td>
      <td>0.491653</td>
      <td>66157.545401</td>
    </tr>
    <tr>
      <th>32</th>
      <td>borough_AVLAND_bldvol</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>95.595089</td>
      <td>2.621305e-07</td>
      <td>0.124206</td>
      <td>0.323458</td>
      <td>0.476183</td>
      <td>65842.300581</td>
    </tr>
    <tr>
      <th>33</th>
      <td>borough_AVTOT_lotarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>17.014372</td>
      <td>9.220929e-07</td>
      <td>0.327414</td>
      <td>0.529325</td>
      <td>0.826850</td>
      <td>8229.533555</td>
    </tr>
    <tr>
      <th>34</th>
      <td>borough_AVTOT_bldarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>79.187910</td>
      <td>2.308863e-06</td>
      <td>0.255751</td>
      <td>0.378122</td>
      <td>0.540363</td>
      <td>58322.066084</td>
    </tr>
    <tr>
      <th>35</th>
      <td>borough_AVTOT_bldvol</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>74.535832</td>
      <td>8.358420e-07</td>
      <td>0.241290</td>
      <td>0.382618</td>
      <td>0.554511</td>
      <td>43325.764569</td>
    </tr>
    <tr>
      <th>36</th>
      <td>all_FULLVAL_lotarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>6.917336</td>
      <td>2.040932e-06</td>
      <td>0.357497</td>
      <td>0.692680</td>
      <td>1.102103</td>
      <td>3791.597998</td>
    </tr>
    <tr>
      <th>37</th>
      <td>all_FULLVAL_bldarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>16.613984</td>
      <td>1.119600e-06</td>
      <td>0.374978</td>
      <td>0.797909</td>
      <td>1.093024</td>
      <td>8615.797998</td>
    </tr>
    <tr>
      <th>38</th>
      <td>all_FULLVAL_bldvol</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>17.382911</td>
      <td>1.906288e-07</td>
      <td>0.316904</td>
      <td>0.840862</td>
      <td>1.183965</td>
      <td>10306.658823</td>
    </tr>
    <tr>
      <th>39</th>
      <td>all_AVLAND_Lotarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>18.677222</td>
      <td>7.397596e-07</td>
      <td>0.236949</td>
      <td>0.420306</td>
      <td>0.658175</td>
      <td>13677.027694</td>
    </tr>
    <tr>
      <th>40</th>
      <td>all_AVLAND_bldarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>91.512042</td>
      <td>1.646531e-06</td>
      <td>0.168626</td>
      <td>0.322974</td>
      <td>0.436387</td>
      <td>51316.564389</td>
    </tr>
    <tr>
      <th>41</th>
      <td>all_AVLAND_bldvol</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>96.665020</td>
      <td>2.748050e-07</td>
      <td>0.113485</td>
      <td>0.330743</td>
      <td>0.466543</td>
      <td>60174.029512</td>
    </tr>
    <tr>
      <th>42</th>
      <td>all_AVTOT_lotarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>13.080188</td>
      <td>2.894529e-07</td>
      <td>0.207416</td>
      <td>0.323550</td>
      <td>0.528072</td>
      <td>7068.398161</td>
    </tr>
    <tr>
      <th>43</th>
      <td>all_AVTOT_bldarea</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>78.348308</td>
      <td>4.531719e-06</td>
      <td>0.236990</td>
      <td>0.331537</td>
      <td>0.444289</td>
      <td>62560.435256</td>
    </tr>
    <tr>
      <th>44</th>
      <td>all_AVTOT_bldvol</td>
      <td>numeric</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1.0</td>
      <td>79.740141</td>
      <td>8.940413e-07</td>
      <td>0.235138</td>
      <td>0.382511</td>
      <td>0.532735</td>
      <td>53483.082742</td>
    </tr>
  </tbody>
</table>
</div>




```python
get_summary(df.iloc[:, :-45])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Type</th>
      <th>Non_NA_Values</th>
      <th>Pct_Non_NA</th>
      <th>Num_Zeros</th>
      <th>Unique</th>
      <th>Most_Common</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BBLE</td>
      <td>char</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1070994</td>
      <td>5051470022</td>
    </tr>
    <tr>
      <th>1</th>
      <td>EASEMENT</td>
      <td>char</td>
      <td>4636</td>
      <td>0.4%</td>
      <td>0</td>
      <td>13</td>
      <td>E</td>
    </tr>
    <tr>
      <th>2</th>
      <td>OWNER</td>
      <td>char</td>
      <td>1039251</td>
      <td>97.0%</td>
      <td>0</td>
      <td>863349</td>
      <td>PARKCHESTER PRESERVAT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BLDGCL</td>
      <td>char</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>200</td>
      <td>R4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TAXCLASS</td>
      <td>char</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>EXT</td>
      <td>char</td>
      <td>354305</td>
      <td>33.1%</td>
      <td>0</td>
      <td>4</td>
      <td>G</td>
    </tr>
    <tr>
      <th>6</th>
      <td>STADDR</td>
      <td>char</td>
      <td>1070318</td>
      <td>99.9%</td>
      <td>0</td>
      <td>839281</td>
      <td>501 SURF AVENUE</td>
    </tr>
    <tr>
      <th>7</th>
      <td>EXMPTCL</td>
      <td>char</td>
      <td>15579</td>
      <td>1.5%</td>
      <td>0</td>
      <td>15</td>
      <td>X1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>PERIOD</td>
      <td>char</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1</td>
      <td>FINAL</td>
    </tr>
    <tr>
      <th>9</th>
      <td>YEAR</td>
      <td>char</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1</td>
      <td>2010/11</td>
    </tr>
    <tr>
      <th>10</th>
      <td>VALTYPE</td>
      <td>char</td>
      <td>1070994</td>
      <td>100.0%</td>
      <td>0</td>
      <td>1</td>
      <td>AC-TR</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.to_csv('features.csv')
```
