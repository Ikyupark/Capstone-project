# Introduction

Data and the age of technology has nonetheless benefitted society a great deal. However, it is not to say that such advancements hasn't been accompanied by down-sides. One of which is identity fraud. The ability for fraudsters, scammers, and hackers to access information as they please is unprecedented and unsuspecting victims often succumb to methods such as phone calls, emails or even fake job postings! In an effort to prevent identity theft, using a data set acquired from [kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction), we aim to create a machine learning model that can predict fraud vs non-fraud jobs based on several parameters.

## Outline

In data analysis, there are several important steps to follow to ensure a good result which allows us to break our project down into 5 segments:

1. Initial Data Analysis
2. Data Preprocessing
3. Feature Engineering
4. Train/Test split data for Machine learning
5. Validate
6. Final result

![Process Outline](https://github.com/Ikyupark/Capstone-project/blob/27d1306cb9c31c0ca6d9e3d5b2847f8c5f33eef3/Resources/New%20ML%20Diagram.png)

### Executive Presentation/Dashboard

To view an executive presentation our project please visit [this link](https://docs.google.com/presentation/d/1v73JqSy9JSMub6i1UreA3L4CD9q1c_3ZiC92hr3AVbY/edit#slide=id.p)  
To view a Tableau dashboard of our project please visit [this link](https://public.tableau.com/app/profile/ikyu.park/viz/capstone_16594856522150/Dashboard2?publish=yes)


## Initial Exploratory Data Analysis

Much of the data will be summarized with figures, refer to the cleaning [file](Uncleaned_Data_Analysis.ipynb) for more detail.

The data imported from the csv was a 17880x18 dataframe and contained various forms of data types as noted below:

![dtypes](https://user-images.githubusercontent.com/100324759/182724852-ebcc0c92-ab1c-4fc3-a90b-ebdbdc867353.PNG)

Some columns also had a staggering number of null values with 'salary_range' sitting at 15012 nulls

![count_null](https://user-images.githubusercontent.com/100324759/182725135-b4e05257-8133-49fe-ae52-68cb1d6e1a84.PNG)

Further exploration also revealed no correlation as shown in the heatmap

![heatmap](https://user-images.githubusercontent.com/100324759/182725759-5a461302-0b12-4871-810d-ff0fdd35fa83.PNG)

### Summaries

Columns job_id, description, and requirements were dropped for the summaries which will be explained in future sections.

The overall fraud rate is relatively low at 95.2% and varied greatly between requirements. As we can see from the figures below, fraud rates are the highest for jobs with the follow attributes: entry level jobs, high school level education, full-time, oil & energy industry jobs, and administrative/engineering jobs.

Aside from the just counts, we should also consider the percentages which gave us the following highest: Administrative - 18.9%, Oil & Energy 38.0%, Full-time 8.5%, and Highschool Ed. 2.0%.

![graphs](https://user-images.githubusercontent.com/100324759/182728992-071a9604-9e23-4d5b-a752-bef23f18eab8.PNG)

## Data Preprocessing

We first dropped the 3 columns job_id, salary, and title due to either too many nulls or all unique values. We were still left with a large amount of nulls but as the dataset was skewed with only 5% fraudulent data entries we did not want to remove the data from the dataset so they were filled with 'not specified'. The location column was then split so that we would only keep the country for consistency. Further modifications include removing punctuation and various characters and then lower casing all characters using the following code
```
for col in clean_cols:
    dataset_df[col] = dataset_df[col].replace(r'[^a-zA-Z0-9\s]', '',regex=True)
    dataset_df[col] = dataset_df[col].replace(r'\s{2,}', '',regex=True)
    
string_cols = list(dataset_df.select_dtypes(include='object'))
for col in string_cols:
    dataset_df[col] = dataset_df[col].str.lower()
```
## Feature Engineering
### Feature Encoding

The data was sub divided into nominal ('department', 'industry', 'function', 'Country') columns and ordinal columns ('employment_type','required_experience','required_education').
Given this:
Target encoding was used on ordinal columns in order to convert the categorical values into integers which are related to the mean of the fraudulent target.

```
Targetenc = TargetEncoder()
for col in nom_cols:
    values = Targetenc.fit_transform(X = dataset_df[col], y = dataset_df['fraudulent'])
    dataset_df[col] = values[col]
```

Label encoding was applied to nominal columns in order to normalize the data into integers for the machine learning model. 
```
le = LabelEncoder()
for col in ord_cols:
    dataset_df[col] = le.fit_transform(dataset_df[col])
```

After encoding the data the resulting dataset was made up of primarily numerical data that could be used in the machine learning model

![cleaned_dtypes](https://user-images.githubusercontent.com/100324759/182904854-016c9b4b-b6ec-41fe-af45-ee91833976ca.PNG)

### Tokenizing

Four columns with large complex strings (requirements, description, company, and benefits) we used NLTK and sklearn.  
  
First, the stop-words are removed using a lamba function on the columns and then stemming and lemmatization was applied. Finally, the columns were then combined into a single column and then tokenized.  
![Tokenizing](https://user-images.githubusercontent.com/100324759/182908382-a62976ee-2d7c-4c1d-b966-226e97c855d5.png)

Once these columns were tokenized term frequency-inverse document frequency (TF-IDF) was used to determine the relevance of each word for each job posting and the relevance of ach word to the whole dataset
![TF-IDF](https://github.com/Ikyupark/Capstone-project/blob/fa2a45ffe612d86c225320187ac75dc69daba40d/Resources/TF-IDF.PNG)


## Machine Learning Model

### Train and Test Split
Initialization of the ML model followed standard procedure with splitting the data accordingly into test and train data sets with a 10% test size of the overall dataset.

### Model Choice
The machine learning model chosen is LightGBM which is a gradient boosting framework that uses tree based learning algorithms. Having 250 iterations and a learning rate of 0.08 takes ~18 seconds to run and achieved an accuracy of ~98% on the testing dataset.

![Model Parameters](https://github.com/Ikyupark/Capstone-project/blob/main/Resources/Model%20parameters.PNG)
![Model Testing Accuracy](https://github.com/Ikyupark/Capstone-project/blob/main/Resources/LightGBM%20testing%20accuracy.PNG)

The LightGBM model has the following benefits:
- faster training speed with higher accuracy compared to other models
- lower memory usage
- better compatibility with large datasets
    
The LightGBM model has the following limitations:
- sensitive to overfitting due to producing more complex trees compared to other models
- sensitive to overfitting on small datasets making LightGBM incompatible with smaller datasets
    
    
 ### Model Tuning
Looking deeper into our ML model revealed that the industry feature was the most important feature and remained consistent even after using XGBOOST instead of LightGBM. Interestingly, changing the learning rate for our ML model didn't appear to change the results overall in terms of accuracy. Trying various learning rates between 0.05 and 1 along with altering the iterations and increasing and decreasing the test and training dataset sizes all yielded a relatively stable accuracy rating.

 ### Model Results
Taking a look at our confusion matrix, non-fraudulent jobs were predicted extremely accurately with only a < 1% of non-fraud being predicted as fraud. However, the number of fraudulent job posts predicted as non job fraud was 37.6%

![Confusion Matrix](https://github.com/Ikyupark/Capstone-project/blob/main/Resources/model%20results.PNG)


# Conclusion

### ML Model
Overall, the machine learning model for detection job fraud proved to be a success. We were able to achieve an accuracy rating of > 98% using the LightGBM model and even with fluctuations in learning rate, the model proved to be quite resilient. We were able to determine that certain sectors of the employment field were more susceptible to job fraud postings such as those with lower requirements (education/experience), administrative/engineering positions.

### Improvements
While the project was a success, the limited availability of posting data in csv format limits the amount of data this machine learning model used for training. Future projects should look to scrape more data from web or use a larger source of data to improve the ML model as the sample size of fraudulent jobs is quite low. Future improvements can enable implementation of this model into various job boards to prevent identity theft.

