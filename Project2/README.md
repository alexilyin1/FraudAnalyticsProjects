# Project 2 - Fraud Detection algorithms developed for a dataset of credit card transactions

## File Directory

```
Project2
│   README.md
│
└───models
│   │   README.md
│   │   alex_nb.html
|   |   alex_nb.ipynb
│   │
│   └───alex_nb_files
│
└───variable_creation
│   │   README.md
│   │   cc_variable_creation.html
|   |   cc_variable_creation.ipynb
│   │
│   └───cc_variable_creation_files
│
└───variable_selection
│   │   README.md
│   │   credit_card_variable_selection.html
|   |   credit_card_variable_selection.ipynb
│   │
│   └───credit_card_variable_selection_files

```


## Background
The dataset for this fraud detection project was a dataset of real-world credit card transactions made in the year 2010. In total, 96,753 transactions were included with 9 columns. The final column identified whether or not the transaction was found to be fraudulent. Other columns included the transaction amount, geographic transaction identifiers such as State and Zip, as well as the transaction number. The goal of this fraud detection assignment is to build an optimal model for deployment as a real time fraud detection algorithm. This means, as transactions are happening in real time, a trained model should be able to detect fraudulent transactions and prevent them. 


## The Plan
In order to accurately catch anomalies in this dataset, the following plan was determined
1. Complete exploratory analysis of the dataset
2. Using the results of the exploratory analysis, determine data cleaning tasks that should be completed 
3. Create custom variables for statistical models 
4. Select a final feature set
5. Design a set of classification models that would be able to detect future fraudulent transactions in real time


### Step 1 - Exploratory Analysis 
![Numeric Variable Summary](imgs/summary_amount.png)
![Character Variable Summary](imgs/summary_char.png)

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

After creating the summary tables above, all the variables in the dataset were visualized using the 'seaborn' package. In the interest of document length, a count plot of the 'Merchstate' variable as well as a distribution of transaction amount are included below:

![Amount](imgs/amount_dist.png)
![Merchant States](imgs/merch_states.png)

Additionally, the 'fraud' outcome variable was used to observe when and where fraud is more likely to occur. To accomplish this, 'Day of the Week' and 'State' were used to show where fraud occurred the most. This produced interesting results that would drive the rest of our analysis:

![Fraud by Day](imgs/fraud_day.png)
![Fraud by State](imgs/fraud_state.png)

### Step 2 - Data Cleaning

Upon completion of an initial data exploration, it was found that three of the possible independent variables (transaction merchant identification number, state and zip) contained missing values. Since these variables are categorical, thinking outside of the box was required to fill in the missing values. 

The first variable filled in was merchant number. A three-phased approach was used so all missing values could be filled in - first, the column 'Merchant description' was aggregated and the most common matching merchant number for the missing record's merchant description was used. For the remaining missing values, the most common merchant numbers by State and by Zip were used in a similar fashion. 

To fill in merchant state and zip, other geographical identifiers could be used to replace missing values. For example, to fill in merchant state, each record was aggregated by State-Zip and the most common values were used to fill in missing values. Any remaining values were filled in by merchant description and merchant number. 

### Step 3 - Variable Creation

After filling in missing values in the dataset, the next step was to create a set of candidate dependent variables that could be used in the model building process. Again, with not many variables available, creativity was needed for creation of a decently sized set of variables. Variables were created based on the following criteria:

* Variables related to a transaction's amount, i.e. Average/max/median/total transaction amount by this card/card at a particular merchant over the past 0/1/3/7 days
* Variables related to frequency of transactions by each entity, i.e. Number of transactions with this card over the past 0/1/3/7 days
* Variables related to the last time an entity made a purchase, i.e. the current date minus the date of the most recent transaction with the same card
* Transaction velociy variables - Number of transactions with the same card/merchant over the past 1 day divided by the average daily transactions with the same card/merchant over the past 7/14/30 days
* Transaction difference variables - Take the difference between the current transaction amount and the average/max/median transaction amount with the same card/merchant over the past 7/14/30 days

The total set of variables was now 399. Obviously, all of these variables would not be used in the final feature set, but this set of variables had potential to inlcude variables that would be better at detecting fraud, compared to the original set of 9 variables. 

### Step 4 - Feature Selection 

[variable selection](variable_selection)

In order to select a final set of features, two feature selection methods would be applied. The first method would be using statistical tests and Fraud Detection Rate and sorting by those two metrics to find variables that served as good predictors of fraud. 

The statistical test used was the Kolmogorov-Smirnov test, or the KS test for short. The KS test is a hypothesis test, with the null hypothesis being that two distributions are identical/the difference of their integrals is 0 and the alternative hypothesis is that the difference is not 0. With a binary classification task such as this one, for each possible dependent variable, two distributions would be created - one for fraudulent transactions and the other for non-fraudulent transactions. The KS test would then test to see if there was a statistically significant difference between the two distributions. If there is, the resulting KS score would be high, and it would tell us that the particular variable serves as a good predictor/differentiator between the two types of transactions. 

Fraud Detection Rate (FDR) is a filter method that looks at a variables ability to detect fraud within the first 3% of a particular population, in this case the sample of credit card transactions. 

After calculating both the KS score and FDR for each variable, the variables were sorted in a table with both scores in descending order. In order to come to a final feature set, the second feature selection method used was Recursive Feature Elimination (RFE). RFE uses a simple classification algorithm (in our case, logistic regression) to find the optimal subset in terms of a particular classification metric (we chose ROC-AUC). The algorithm starts with the full subset of training features and repeatedly removes features. We can then create a plot that graphs the # of features against the classification error metric to give us an idea of which subset of features is most optimal:

![RFE Plot](imgs/rfe_graph.png)

Additionally, Sklearn's RFECV function returns a "rferanking" variable that gives a ranking from 1 to the number of subsets tried. This makes it easy to chose our optimal feature subset. This feature selection algorithm was run on top 80 features and again with top 50 features, from which we chose our final 30 features. 

Before applying any models, feature selection was needed to select the most important features from the set above. However, with so many variables, the issue of multicollinearity appears. This is an issue in statistical modeling, as relationship between a feature and the outcome variable could be hidden in the correlation between two variables. To find any multicollinearity, a correlation matrix was created:

![Correlation Matrix)(imgs/corr_matrix.png)

With many of the new variables correlated, Principal Component Analysis is a possible dimensionality reduction algorithm that can be applied in order to reduce the complexity of our feature set and reduce multicollinearity that exists. PCA can help us create linear combinations of the features created, with new combinations created orthogonal to each other. However, before applying the PCA algorithm, each variable should be scaled to increase the validity of the PCA results. 

The approach of PCA reduces correlation. After applying the PCA algorithm in sklearn, a scree plot can be used to determine with principal components to keep. The principal components to keep are determined by their respective explained variance. It is generally accepted that we want to keep components that combine to explain around ~90% of the variance in our data. The final step is to apply Z-score scaling to the remaining principal components. This will reduce the chances that the first principal component in our feature set will take up the most importance (due to it having the highest level of variance explained). 

What we are left with is a feature set of four principal components. While we have dratistically reduced the dimension of our dataset, what we are left with is a feature set that is uncorrelated and scaled. We can now begin applying models. 

### Step 5a - Heuristic Distance Measure

This first model is a heuristic model that will be applied to the remaining four features. For each of the records in the dataset, a score would be determined by finding the distance of the records' feature values from the origin point of that variable on a 2 dimensional plane (in other words, the euclidean distance of each feature). With our features already undergoing scaling, it was guaranteed that any irregularities in dimensions could be avoided. This unsupervised heuristic approach would tell us which records appear to be outliers. With one of the steps in our feature selection process involving standardizing the feature variables  Since there was not an actual model being built, we could simple apply this score to the whole dataset. The score was evaluated this way - if a variable had a score of 500 - this meant it was 500 standard deviations away from the mean - this would definitely mean that the record could be fraudulent.

### Step 5b - Autoencoder

An autoencoder is a type of Neural Net that converts its input to a compressed latent representation. The version of the input variables is then recreated and the distance between the resulting output vector and the input vector would be the basis of the 2nd model's results. In a sense, what the autoencoder is doing is training on the variance of the input data to recreate it. The difference between what the autoencoder predicts as the input vector and what the actual input vector is could serve as an outlier detection value. This value can be referred to as a reconstruction error. With the nature fo this model, it would be beneficial to train on the WHOLE dataset, rather than splitting it up into training/test/validation sets. The parameters of the final model were:

* Keras neural net
* reLu activation
* 25 epochs 

The final score, or the reconstruction error, was calculated using the commonly used vector distance measure, or the Minkowski score. The reasoning for choosing such a method was, since the first method used a linear distance measure, the autoencoder would attempt to capture non-linearities among our feautres. 

### Step 5c - Final Score 

With two scores not available, one from the heuristic Euclidean distance model and the other from the autoencoder's reconstruction error, we could combine them directly in order to get a final score. An approach called Extreme Quantile Binning, in which scores from each method are "binned" and are ordered based on their "rank" of score in an ascending order. The number of bins used equaled the number of records in the data, and the resulting score was an average of both of the ranks. Finding the average of rank ordered values would give us a way to evaluate the results of these unsupervised algorithms.

### Step 6 - Results 

Below are distributions for the two model scores in their respective order:

![Heuristic Model](imgs/model1_results.png)
![Autoencoder Model](imgs/model2_results.png)

Here are the results of the extreme quantile binning approach:

![EQB Model](imgs/eqb_results.png)

We can explore the top 10 results from each score to find validate them directly based on our knowledge of the New York Housing Dataset

For example, from model 1, we can take some of the housing records for which high fraud scores were found. The method we'll use for validation is whether or not the value is 3 standard deviations outside of the mean of a particular feature.

* Building 917942 - The value for FULLVAL is an outlier
* Building 1067360 - The value for LTFRONT and LTDEPTH are both outliers
etc...

After listing out the top 10 most "fraudulent" records found in each model, we can compare the results to the quantile binning approach. Using this approach, we find that the top 10 most fraudulent records from the quantile binning approach directly match the top 10 most fraudulent records from the heuristic euclidean distance model. 

It was important to add the last step to this project - validating the results of the models. Since unsupervised models were applied to the results of a PCA algorithm, the results of the models could not be used at face value. It was necessary to actually examine some of the most fraudulent records to see if their actual feature values were outliers.

Finally, these results can be presented to members of NY Housing/Tax committees, to show fraudulent housing records as well as characteristics of records found to be fraudulent. 

