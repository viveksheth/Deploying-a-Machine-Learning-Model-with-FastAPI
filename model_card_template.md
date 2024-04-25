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
This model can be used to predict the salary level of an individual beased on different attributes.

## Training Data
The Census Income Dataset was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income) as a csv file.
Extraction was done by Barry Becker from the 1994 Census database.
Prediction task is to determine whether a person makes over 50K a year.

The original data set has 32,561 rows and 15 columns composed of the target label "salary", 8 categorical features and 6 numerical features.
Details on each of the features ae available at the UCI link above.
Target label "salary" has two classes ('<=50K', '>50K') and shows class imbalance with a ratio of circa 75% / 25%.
A simple data cleansing was performed on the original dataset to remove leading and trailing whitespaces. Data cleaning steps are in data_clean.ipynb notebook for data exploration and clearning. 

Model features: 
- age: continuous.
 - workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
 - fnlwgt: continuous.
 - education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
 - education-num: continuous.
 - marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
 - occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
 - relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
 - race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
 - sex: Female, Male.
 - capital-gain: continuous.
 - capital-loss: continuous.
 - hours-per-week: continuous.
 - native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.


## Evaluation Data
The original dataset is first preprocessed and then split into training and evaluation data with evaluation data size of 20%


## Metrics
The classification performance is evaluated using precision, recall and fbeta metrics.
The confusion matrix is also calculated.

The model achieves below scores using the test set:
- precision:0.759
- recall:0.643
- fbeta:0.696
- Confusion matrix:
[[4625  320]
 [ 560 1008]]

## Ethical Considerations
This model is trained on census data. The model is not biased towards any particular group of people.

## Caveats and Recommendations
I recommend that checks are included upstream of any decision-making points to ensure that bias is minimized. Extraction was done from 1994 Census database. This dataset is outdated and sample cannot adequentely be use as representation of the population. 
