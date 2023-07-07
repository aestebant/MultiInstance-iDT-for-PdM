# MultiInstance iDT for PdM: *Multi-Instance Learning and Incremental Decision Trees for failure detection in industrial equipment*

Associated repository with complementary material to the manuscript *Multi-Instance Learning and Incremental Decision Trees for failure detection in industrial equipment* [under review in *Expert Systems with Applications*]:

* Source code of the MI-IDT proposals.
* Datasets used in the experimentation.
* Complete tables of results.

## Source code

The aim of the proposals developed in this work is the Time Series Classification in a context of system degradation and weak labeling environment. The proposals have been conceptualized under the paradigm of Multi-Instance Learning, using Incremental Decision Trees for the classification at instance level. The code is structured under the [src](src/) folder with the following structure:
```
src
│   requirements.yml
│   tutorial.ipynb
│
└───mi-idt
│   │   milidt_tsc.py > Implementation of MI-HT, MI-HATT, MI-HAT.
│   
└───dataprep > Auxiliary functions to prepare the datasets for MIL
```

The development environment is based on Python >=3.9, with a special mention to the library [`scikit-multiflow`](https://scikit-multiflow.github.io) as the base for the implementation of the incremental decision trees. The complete list of libraries to replicate the environment is available in [requirements.yml](requirements.yml).

The Jupyter notebook [tutorial.ipnb](tutorial.ipynb) describes a complete tutorial for using the presented library, including loading data, building and testing the classification model, and accessing to the interpretability resources.

## Datasets

This work tests its results in two popular Predictive Maintenance problems: [NASA Ames Turbofan Engine Degradation Dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan) and [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter). They have been adapted to perform time series classification, with these modified versions available to download under the [datasets](datasets/) folder. Each dataset is further divided in four problems with different characteristics, but in all of them the objective is to identify the grade of degradation of the industrial system based on the record of its internal measures during a working sequence of variable duration. Bellow they are shown the characteristics of each dataset in terms of states to classify, number of sequences belonging to each state and average duration of sequences per each state. Finally, it is shown the file of the given dataset. 

**NASA Ames Turbofan engine degradation dataset:**
| Dataset | Train | | | Test | | |
|---|---|---|---|---|---|---|
| | **Sequences per class** | **Average duration** | **File** | **Sequences per class** | **Average duration** | **File** | 
| FD001 | 33/34/33 | 112.1/143.8/209.1 | train_FD001.csv|  39/32/29 | 102.3/91.6/174.6 | test_FD001.csv |
| FD002 | 86/88/86 | 107.8/144.4/201.7 | train_FD002.csv |  87/82/86 | 82.8/96.5/179.7 | test_FD002.csv |
| FD003 | 33/34/33 | 152.8/177.6/257.8 | train_FD003.csv |  35/32/33 | 122.3/117.8/230.1 | test_FD003.csv |
| FD004 | 83/83/83 | 128.4/158.5/253.8 | train_FD004.csv | 62/104/82 | 109.8/134.6/219.9 | test_FD004.csv |



**Case Western Reserve University Bearing dataset:**

| Dataset | Train | | | Test | | |
|---|---|---|---|---|---|---|
| | **Sequences per class** | **Average duration** | **File** | **Sequences per class** | **Average duration** | **File** | 
| DE-IR | 313/317/307/307/304 | 1205.4 | train_drive_innerrace.csv | 95/88/97/97/97 | 1205.4 | test_drive_innerrace.csv |
| DE-BB | 321/312/323/327/323 | 1204.7 | train_drive_ball.csv | 87/93/81/77/80 | 1204.7 | test_drive_ball.csv |
| FE-IR | 340/329/325/305 | 1204.8 | train_fan_innerrace.csv | 68/75/78/95 | 1205.0 | test_fan_innerrace.csv |
| FE-BB | 319/338/315/304 | 1205.8 | train_fan_ball.csv | 89/63/88/99 | 1205.9 | test_fan_ball.csv


### Datasets transformation

Both NASA and CWRU datasets have required transformations to work in a multi-instance fault-detection framework. The transformation code is available under the fold [src/dataprep](src/dataprep).

The original target of the NASA Ames Turbofan Engine Degradation dataset is the remaining useful life estimation, i.e., a regression task. Thus, the transformations go in the direction of generating a ground truth that associates each time series with a health state degradation state.

In the case of the CWRU bearing dataset, the target is the fault identification directly, but the working sequences are not ready to use in a machine learning model because they are in separate files corresponding to a single complete experiment. Thus, the transformations consist of splitting the sequences into separate time series, mixing all the degradation states in a single pool, and creating separate sets for training and testing the machine learning models.

## Results

The final results obtained for the proposed MI-IDT in the test partition of each dataset are the following.

| Dataset | MI-HT | | MI-HATT | | MI-HAT | |
|--|--:|--:|--:|--:|--:|--:|
| | **Acc** | **F1-score** | **Acc** | **F1-score** | **Acc** | **F1-score** |
| FD001 | 90.00 | 90.38 | 95.00 | 95.02 | 92.00 | 92.26 |
| FD002 | 87.84 | 87.82 | 87.84 | 87.83 | 82.35 | 82.85 |
| FD003 | 77.00 | 76.22 | 88.00 | 87.44 | 87.00 | 86.91 |
| FD004 | 85.48 | 85.15 | 81.05 | 80.40 | 83.87 | 84.07 |
| |
| DE-BB | 93.33 | 93.79 | 93.10 | 93.18 | 60.48 | 62.41 |
| DE-IR | 100.00 | 100.00 | 94.82 | 94.92 | 42.57 | 35.51 |
| FE-BB | 100.00 | 100.00 | 94.39 | 93.37 | 55.75 | 54.34 |
| FE-IR | 100.00 | 100.00 | 100.00 | 100.00 | 66.77 | 66.34 |

Which have been obtained with the following configurations for each of the proposed methods:

| Dataset | MI-HT | | | | | MI-HATT | | | | | MI-HAT | | | | |
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
| | $\omega$ | $\lambda$ | *k* | *grace period* | *split conf.* | $\omega$ | $\lambda$ | *k* | *grace period* | *split conf.* | $\omega$ | $\lambda$ | *k* | *grace period* | *split conf.* |
| FD001 | 10 | 5 | 10 | 4e3 | 1e-5 | 5 | 5| 6 | 6e2 | 1e-7 | 15 | 5 | 10 | 1e3 | 1e-9 |
| FD002 | 15 | 5 | 6 | 1e3 | 1e-7 | 15 | 3 | 10 | 1e3 | 1e-5 | 5 | 1 | 6 | 6e2 | 1e-5 |
| FD003 | 10 | 5 | 6 | 4e3 | 1e-9 | 15 | 3 | 6 | 6e2 | 1e-7 | 5 |1 | 10 | 4e3 | 1e-7 |
| FD004 | 15 | 5 | 10 | 6e2 | 1e-9 | 5 | 3 | 10 | 1e3 | 1e-5 | 5 | 5 | 6 | 6e2 | 1e-9 |
| |
| DE-BB | 300 | 10 | 10 | 1e4 | 1e-9 | 300 | 10 | 6 |2e4 | 1e-5 | 100 | 20 | 8 | 2e4 | 1e-9 |
| DE-IR | 100 | 30 | 6 | 1e4 | 1e-9 | 300 | 10 | 6 | 1e4 | 1e-9 | 100 | 30 | 6 | 1e4 | 1e-5 |
| FE-BB | 300 | 10 | 6 | 1e4 | 1e-9 | 300 | 30 | 8 | 3e4 | 1e-7 | 100 | 10 | 6 | 1e4 | 1e-5 |
| FE-IR | 100 | 10 | 6 | 1e4 | 1e-9 | 300 | 30 | 8 | 2e4 | 1e-9 | 100 | 30 | 10 | 3e4 | 1e-9|

The results associated to the complete experimentation carried out in this work are available in the [results](results/) folder. There are organized in spreadsheets for each MI-IDT model, with a page for a dataset and a row for each configuration tested. The columns show the performance in both train and test for accuracy, macro-F1, and per-class-F1 metrics.

* miht_report.xlsx: complete experimentation for MI-HT.
* mihatt_report.xlsx: complete experimentation for MI-HATT.
* mihat_report.xlsx: complete experimentation for MI-HAT.
* sil_idt_report.xlsx: complete experimentation for HT, HATT and HAT from the single-instance learning approach carried out for comparative purposes.
* mil_dl_report.xlsx: complete experimentation for deep learning models from multi-instance learning carried out for comparative purposes.
* sil_fe_report.xlsx: complete experimentation for classic machine learning models using feature-extraction statistical methods over the temporal series, carried out for comparative purposes.
