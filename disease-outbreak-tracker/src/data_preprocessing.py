import pandas as pds
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import kagglehub

# Download latest version
path = kagglehub.dataset_download("imdevskp/corona-virus-report")

df = pd.read_csv('C:/projects/disease-outbreak-tracker/disease-outbreak-tracker/data/raw/covid_19_clean_complete.csv')
df.head()

# General Info About data
df.info()
df.describe()

print(df.describe())

warnings.filterwarnings('ignore')