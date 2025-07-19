# importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

from sklearn.preprocessing import MinMaxScaler
import joblib
# Load the dataset (update the filename accordingly)
df = pd.read_csv("irrigation_machine.csv")
# first 5 rows to be printed, df.tail()
df.head()
df.info()
df.columns
df = df.drop('Unnamed: 0', axis=1)
df.head()
df.describe() # Statistics of the dataset
# -------------------------------
# STEP 2: DEFINE FEATURES AND LABELS
# -------------------------------

X = df.iloc[:, 0:20]   # This gives you columns 0 to 19 (sensor_0 to sensor_19)
y = df.iloc[:, 20:]
X.sample(10)
y.sample(10)
X.info()
y.info()
X
X.shape, y.shape