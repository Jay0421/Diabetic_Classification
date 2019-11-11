
# Import Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Loading dataset

data = pd.read_csv("diabetes.csv")
data.head()


# Handling Missing Values

data.isnull().sum()
data = data.replace(0, np.nan)
data = data.fillna(data.mean())


# Label Encoding

label = LabelEncoder()
data["is_diabetic"] = label.fit_transform(data["is_diabetic"])
data.head()

# Train Test Split

x = data.iloc[:, 0:8].values
y = data.iloc[:, 8].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 421)

# Build the model

classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)

