# Project 1 - Fraud Detection algorithms developed for the New York Property dataset


## Background
The dataset for this fraud detection project was a publically available dataset of New York property valuations from 2010/2011, available through the New York Department of Finance. In total, the dataset contains valuations for 1,070,994 properties (dataset not included for size reasons). The dataset contained 32 features. 


## The Plan
In order to accurately catch anomalies in this dataset, the following plan was determined
1. Complete exploratory analysis of the dataset
2. Using the results of the exploratory analysis, determine data cleaning tasks that should be completed 
3. Create custom variables for statistical models 
3. Design two models - a heuristic model using principal components of the feature variables and an autoencoder trained on the full dataset. A "fraud score" would be developed from the results of these two models to find valuation record that could be fraudulent. 


### Step 1 - Exploratory Analysis 
[Numeric Variable Summary](imgs/data_summary.png)
[Character Variable Summary](imgs/data_summary2.png)

The two screenshots above show the results of a summary function applied to the dataset:
```
def get_summary(dat):

    dtypes_dict = dat.dtypes.apply(lambda x: x.name).to_dict

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

    return pd.concat([numeric_df, chars_df], ignore_index=True)
```

After creating the summary tables above, all the variables in the dataset were visualized using the 'seaborn' package. In the interest of document length, only a count plot of the 'BLOCK' variable and the distribution of the 'LTFRONT' or lot frontage variable are included:

[BLOCK](imgs/block_count_plot.png)
[LTFRONT](imgs/LTFRONT.png)

Based on the distributions of the variables visualized, initial data transformations were applied. For example, and numerical variable that was found to be skewed could be 'adjusted' using a log transformation in order to have the variable's distribution more closely follow the normal distribution. This is one possible way to deal with outliers in a dataset. 

### Step 2 - Data Cleaning

As mentioned above, exploratory analysis reveals any variables in our dataset that may have missing values that need to be filled in, or are skewed and could require some kind of transformation. The summary table above shows that the following variables would need to be adjusted for missing values:

* ZIP - zip code
* FULLVAL - full market value
* AVLAND - final market value of land
* AVTOT - total market value of land
* STORIES - number of stories
* LTFRONT - lot frontage
* LTDEPTH - lot depth
* BLDFRONT - building frontage
* BLDDEPTH - building depth
