# MI-IDTforTSC: *Multi-Instance Learning and Incremental Decision Trees for failure detection in industrial equipment*: additional materials

Associated repository to the paper <span style="color:red">[FULL CITE]</span> with complementary material:

* Source code of the MI-IDT proposals.
* Datasets used in the experimentation.
* Complete tables of results.

## Source code

The aim of the proposals developed in this work is the Time Series Classification in a context of system degradation and weak labeling environment. The proposals have been conceptualized under the paradigm of Multi-Instance Learning, using Incremental Decision Trees for the classification at instance level. The code is structured under the `src` folder with the following structure:
```
src
│   requeriments.yml    
│
└───mi-idt
│   │   mil_idt_tsc.py > MI-HT, MI-HATT, MI-HAT.
│   
└───comparison
│    │   sil_idt_tsc.py > Infraestructure for training/testing HT, HATT, HAT.
│    │   sil_fe_tsc.py > Classic ML algorithms based on feature extraction over temporal data.
│    │   mil_dl_tsc.py > Deep learning from MI approach.
│   
└───misc > Auxiliary functions for information printing.
```

The development environment is based on Python >=3.9, with a special mention to the library [`scikit-multiflow`](https://scikit-multiflow.github.io) as the base for the implementation of the incremental decision trees. The complete list of libraries to replicate the environment is available in [requeriments.yml](https://github.com/aestebant/MI-IDTforTSC/blob/master/src/requirements.yml).

## Datasets

This work tests its results in two popular Predictive Maintenance problems: [NASA Ames Turbofan Engine Degradation Dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan) and [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter). They have been properly adapted to perform time series classification (more details bellow) and available for download in the [dataset](https://github.com/aestebant/MI-IDTforTSC/blob/master/datasets) folder. Each dataset is further divided in four problems with different characteristics (more information in <span style="color:red">[REF TO OUR WORK]</span>), and each one is separated in both train and test sets formated in CSV. The files are organized in ZIP files that package the following files:

* NASA Ames Turbofan Engine Degradation dataset:
    * FD001 train and test files: train_FD001.csv, test_FD001.csv.
    * FD002 train and test files: train_FD002.csv, test_FD002.csv.
    * FD003 train and test files: train_FD003.csv, test_FD003.csv.
    * FD004 train and test files: train_FD004.csv, test_FD004.csv.
* Case Western Reserve University Bearing dataset:
    * Train and test files for ball fault in engine drive: train_drive_ball.csv, test_drive_ball.csv.
    * Train and test files for inner race fault in engine drive: train_drive_innerrace.csv, test_drive_innerrace.csv.
    * Train and test files for ball fault in engine fan: train_fan_ball.csv, test_fan_ball.csv.
    * Train and test files for inner race fault in engine fan: train_fan_innerrace.csv, test_fan_innerrace.csv.

### Datasets transformation

<span style="color:red"> \#ToDo </span>

## Results

The results associated to the complete experimentation carried out in this work are available in the [results](https://github.com/aestebant/MI-IDTforTSC/blob/master/results) folder. There are organized in spreadsheets for each MI-IDT model, whith a page for a dataset and a row for each configuration tested. The columns show the performance in both train and test for accuracy, macro-F1, and per-class-F1 metrics.

* miht_report.xlsx: complete experimentation for MI-HT.
* mihatt_report.xlsx: complete experimentation for MI-HATT.
* mihat_report.xlsx: complete experimentation for MI-HAT.
* sil_idt_report.xlsx: complete experimentation for HT, HATT and HAT from the single-instance learning approach carried out for comparative purposes.
* mil_dl_report.xlsx: complete experimentation for deep learning models from multi-instance learning carrid out for comparative purposes.
* sil_fe_report.xlsx: complete experimentation for classic machine learning models using feature-extraction statistical methods over the temporal series, carried out for comparative purposes.