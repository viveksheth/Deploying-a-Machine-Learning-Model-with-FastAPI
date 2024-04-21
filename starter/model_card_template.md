# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Prediction task is to determine whether a person makes over 50K a year.
We use a GradientBoostingClassifier using the optimized hyperparameters in scikit-learn 1.2.0.
Hyperparameters tuning was realized using GridSearchCV.

Optimal parameters used are:
- learning_rate: 1.0
- max_depth: 5
- min_samples_split: 100
- n_estimators: 10
Model is saved as a pickle file in the model folder. All training steps and metrics are logged in the file "journal.log".

## Intended Use
This model can be used to predict the salary level of an individual based off a handful of attributes. The usage is meant for students, academics or research purpose.

## Training Data
The Census Income Dataset was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income) as a csv file.
The original data set has 32,561 rows and 15 columns composed of the target label "salary", 8 categorical features and 6 numerical features.
Details on each of the features ae available at the UCI link above.
Target label "salary" has two classes ('<=50K', '>50K') and shows class imbalance with a ratio of circa 75% / 25%.
A simple data cleansing was performed on the original dataset to remove leading and trailing whitespaces. Data cleaning steps are in data_clean.ipynb notebook for data exploration and clearning. 


## Evaluation Data

## Metrics
_Please include the metrics used and your model's performance on those metrics._

## Ethical Considerations

## Caveats and Recommendations
