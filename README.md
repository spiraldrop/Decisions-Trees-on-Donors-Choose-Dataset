# Decisions-Trees-on-Donors-Choose-Dataset

DonorsChoose dataset is a dataset provided by  DonorsChoose.org.  DonorsChoose empowers public school teachers from across the country to request much-needed materials and experiences for their students. DonorsChoose receives hundreds of thousands of project proposals each year for classroom projects in need of funding. In this problem, we try to predict whether or not a DonorsChoose.org project proposal submitted by a teacher will be approved, using the text of project descriptions as well as additional metadata about the project, teacher, and school.

### About this project

I completed this assignment as a part of my Machine Learning course at Applied AI. In this assignment, I have utilized two sets of vectorization mutations for our feature set, one with ```TF-IDF```(set 1) and the other with ```TF-IDF weighted W2V``` (set 2). The performance metric used is the AUC score and Confusion Matrix, where``` y = 'project_is_approved'```. For hyperparameter tuning, I've chosen ```RandomizedSearchCV```, although computationally expensive, it bore good results. Here, the parameter distributions included ```'max_depth'``` and ```'min_samples_split'```. The procedure conducted for both sets is the same. Moreover, I have also applied the classifier on an additional set of False Positive data points, set 3, in the test dataset extracted at the end of set 1. For visualization of the performance metrics, you can plot a Confusion Matrix heatmap or a 3D plot from ```plotly```,  both are included in the 'Assignment_DT_Instructions' file. The initial section of the file consists of the vectorizations of all given features, categorical and numerical into ```TF-IDF``` and ```TF-IDF weighted W2V``` form. Then comes the hyperparameter tuning and finally, applying DT on both sets. Almost all piping with set 3 is similar to the previous 2 sets and is in the 'Task 2' section towards the end of the file. In addition to the ROC_AUC curve, I have also printed the ```WordCloud``` on both sets (on 'essay') as well as the ```PDF``` of the False Positives.
### Link to the files 
https://drive.google.com/drive/folders/165O0Nc_m2Q4qGm8hS5anFooKT1s2dzAA

### Libraries needed
You need to install the following methods and libraries: 
```
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.sparse import hstack

import re
# Tutorial about Python regular expressions: https://pymotw.com/2/re/

import pickle
from tqdm import tqdm
import os

from chart_studio import plotly
import plotly.offline as offline
import plotly.graph_objs as go
offline.init_notebook_mode()
from collections import Counter
```

#### Link to the course
https://www.appliedaicourse.com/course/11/Applied-Machine-learning-course 

