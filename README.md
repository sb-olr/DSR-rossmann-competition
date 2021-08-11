# DSR Rossman Mini-Competition
This mini competition is adapted from the Kaggle Rossman challenge.

## Setup environment
Create a new environment and install the required libraries.
```angular2html
pip install -r requirements.txt
```
Note: If you are working on a new Macbook with M1 chip, it is possible that you have to additionally run `conda install -c conda-forge py-xgboost` to make xgboost work.

## Getting the data
```angular2html
#  during the competition run
python data.py

#  at test time run
python data.py --test 1
```

## Dataset
The dataset during model development is made of two csvs: 
```angular2html
#  store.csv
['Store', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

#  train.csv
['Date', 'Store', 'DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo','StateHoliday', 'SchoolHoliday']
```
More information from Kaggle
```angular2html
Id - an Id that represents a (Store, Date) duple within the test set

Store - a unique Id for each store

Sales - the turnover for any given day (this is what you are predicting)

Customers - the number of customers on a given day

Open - an indicator for whether the store was open: 0 = closed, 1 = open

StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None

SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools

StoreType - differentiates between 4 different store models: a, b, c, d

Assortment - describes an assortment level: a = basic, b = extra, c = extended

CompetitionDistance - distance in meters to the nearest competitor store

CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened

Promo - indicates whether a store is running a promo on that day

Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating

Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2

PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store
```
The holdout test period is from 2014-08-01 to 2015-07-31 - the holdout test dataset is the same format as train.csv, and is called holdout.csv.

## Doing a full run of training models
```angular2html
python src/data/make_dataset.py
python src/features/build_features.py
python src/models/train_model.py
```

## Making predictions with test data
```angular2html
python src/models/predict_model.py
```

## Project organization
```angular2html
├── README.md         <- The top-level README for developers using this project.
├── data              <- Data folder with raw data and saved files.
├── models             <- Folder with saved models.
├── notebooks          <- Jupyter notebooks.
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use final model to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
└──
```

## Scoring metric
The task is to predict the Sales of a given store on a given day. Model performance is evaluated on the root mean square percentage error (RMSPE).



