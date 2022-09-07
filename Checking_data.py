from petastorm import make_reader
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd

##Read the dataset
path_formatted_glasgow = "/workspaces/maitrise/data/20220902_data_physio_formatted_merged/merged/dataParquet"
path_petastorm = f"file:///{path_formatted_glasgow}"
with make_reader(path_petastorm) as reader:
    for sample in reader:
        print(sample)
        break
#data = pd.read_parquet(path_formatted_glasgow,engine ="pyarrow")
#print(data)
