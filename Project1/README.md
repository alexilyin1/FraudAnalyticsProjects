# Project 1 - Fraud Detection algorithms developed for the New York Property dataset


## Background
The dataset for this fraud detection project was a publically available dataset of New York property valuations from 2010/2011, available through the New York Department of Finance. In total, the dataset contains valuations for 1,070,994 properties (dataset not included for size reasons). The dataset contained 32 features. 


## The Plan
In order to accurately catch anomalies in this dataset, the following plan was determined
1. Complete exploratory analysis of the dataset
2. Using the results of the exploratory analysis, determine data cleaning tasks that should be completed 
3. Create custom variables for statistical models 
4. Select a final feature set - dimensionality reduction
5. Design two models - a heuristic model using principal components of the feature variables and an autoencoder trained on the full dataset. A "fraud score" would be developed from the results of these two models to find valuation record that could be fraudulent. 


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

While the approach to replacing NULL/missing values for numerical features can be as straightforward as using the mean of a column, missing values for character variables requires some more thought. I will directly address the strategy for filling in values for the ZIP and FULLVAL columns. 

To fill in the ZIP column (treated as a string), two additional variables were used that are closely related to ZIP in their relation to a properties' geography: 'BOROUGH' and 'BLOCK'. For each missing value, the mode of the group  a record is in would replace the missing value. Since BOROUGH and BLOCK represent the location of a property, it would make sense that the ZIP would follow those two features.

To fill in the FULLVAL column, BOROUGH and newly filled in ZIP variable were used to fill in missing values. Similarly to the above approach, for each combination of BOROUGH and ZIP, the mean was found and missing values of FULLVAL were filled in based on the group they belonged in. 

### Step 3 - Variable Creation

After the data cleaning process, new variables could be created to actually complete the fraud detection. While the original dataset provided a fair amount of features, it would be beneifical to have a wide range of features to make the final feature selection process more in depth. Some examples of variables that were created include:

* Lot area = lot front * lot depth
* Building area = building front * building depth
* Building volume = building area * stories

Using these three variables, an additional nine variables were created on the basis of the FULLVAL, AVLAND and AVTOT variables. These three existing variables were normalized by the three new variables above. For each variable, we now had variables for price per volume/square foot. The next step in the variable creation process was to combine these normalized variables with different property identifiers:

[Expert Vars](imgs/expert_vars.png)

Zip5 and Zip3 represent the full zipcode and the first 3 digits of the zipcode, respectively (3 digits chosen to give a wider area of properties). Taxclass refers to the NY tax class each property belongs in, Boro refers to the borough a property is located in and All is a grouping of the above variables. After applying models to these 45 variables, the expected result is that outliers (fraudulent records) would be identified based on their outlier status with respect to the distribution of these new variables. Summary stats for the new variables were:

[Expert Vars Stats](imgs/expert_vars_summary.png)

### Step 4 - Feature Selection 
Before applying any models, feature selection was needed to select the most important features from the set above. However, with so many variables, the issue of multicollinearity appears. This is an issue in statistical modeling, as relationship between a feature and the outcome variable could be hidden in the correlation between two variables. To find any multicollinearity, a correlation matrix was created:

[Correlation Matrix)(imgs/corr_matrix.png)

With many of the new variables correlated, Principal Component Analysis is a possible dimensionality reduction algorithm that can be applied in order to reduce the complexity of our feature set and reduce multicollinearity that exists. PCA can help us create linear combinations of the features created, with new combinations created orthogonal to each other. This approach reduces correlation. After applying the PCA algorithm in sklearn, a scree plot can be used to determine with principal components to keep. The principal components to keep are determined by their respective explained variance. It is generally accepted that we want to keep components that combine to explain around ~90% of the variance in our data. The final step is to apply Z-score scaling to the remaining principal components. This will reduce the chances that the first principal component in our feature set will take up the most importance (due to it having the highest level of variance explained). 

What we are left with is a feature set of four principal components. While we have dratistically reduced the dimension of our dataset, what we are left with is a feature set that is uncorrelated and scaled. We can now begin applying models. 

### Step 5a - Heuristic Distance Measure

This first model is a heuristic model that will be applied to the remaining four features. 
