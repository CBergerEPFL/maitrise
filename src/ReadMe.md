# Master Project Charles Berger
# Project : Improving ECG automatic classifications in the context of noisy or corrupted lead

## Introduction

In this folder, you can find all the code made during the project. This includes codes :
- for the Signal Quality Assesment index. All the index used tested in this study are given as well as the one we used for the report
- for ECG lead reconstruction.
- for classification after reconstruction and SQA

## Files present:

Python code presents are :
    - In the **notebooks** folder :
        - **create_attractor.ipynb** : This Jupyter notebook allows to create csv file containing the time duration and 3D coordinates time evolution of a 3D chaotic system. The chaotic systems studied can be selected in the last cells as well as the number of simulation desired. The output are individual folders  (named using the following convention : *name_attractor*) containing all the csv files of each simulation. These folders are saved in the *data* folder. WARNING : To run this code, you must have, in the *shared_utils* folder, the python file name *utils_parallel.py* to run this code.


        - ** test_sample.ipynb ** : This notebook contain the codes for all SQA indexes tested during this study. It takes as input a formated ECG dataset and extract one recording of random patient's ECG . It organized each leads recording into dictionnaries which are afterwards used by each SQA indexes implemented. Cells containing each SQA methods are indicated.

        - ** Test_TSD_article.ipynb ** : This notebook contain codes to reproduce the Time Series Dimensions or TSD SQA index, as presented in the following article : [Estimating the level of dynamical noise in time series by using fractal dimensions](https://www.sciencedirect.com/science/article/pii/S0375960116000177). Each cells indicate which function and which part of the article is reproduced. The TSD is applied on real ECG data. Notes : in the *shared_utils* folder, you must have the *Time_series_dimensions_calculus.py* to run this jupyter notebook.

        - **Attractor_shuffling.ipynb** : This notebook applies the TSD to attractors. To run this, you must run before the **create_attractor.ipynb** notebook. It then apply  all the methods in the previous article to attractors selected and attempt to reproduce the results obtained in the latter. Note : though functions necessary to run this notebook is already present in it, we recommend that you use *Time_series_dimensions_calculus.py* in the *shared_utils* folder.

    - in the **shared_utils** folder :
        - **utils_parallel.py** : This python file allows the computation of 2D and 3D chaotic system at specific initial conditions. The attractors that are computed are indicated in the file.

        -**Time_series_dimensions_calculus.py** : This python file coontains all the functions for the implementation of the TSD methods proposed by the previous.
