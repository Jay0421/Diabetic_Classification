
# Import Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Loading dataset

data = pd.read_csv("diabetes.csv")
data.head()


# Handling Missing Values

data.isnull().sum()
data = data.replace(0, np.nan)
data = data.fillna(data.mean())

