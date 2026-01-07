import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Load dataset and parse data to change type
dataset = pd.read_csv("data/gold_price_data.csv", parse_dates=["Date"])
# dataset.info()

# Processing missing values/null values:
nulls = dataset.isna().sum().sort_values(ascending=False)
# print(nulls)

# Checking & visualizing correlation:
correlation = dataset.corr()
sns.heatmap(correlation, cmap="coolwarm", center=0, annot=True)
plt.title("Correlation Matrix Heatmap")
plt.xlabel("Features")
plt.ylabel("Features")
plt.show()

# Dropping SLV due to high correlation
dataset.drop("SLV", axis=1, inplace=True)

# Setting Date:
dataset.set_index("Date", inplace=True)

dataset["EUR/USD"].plot()
plt.title("Change in price of gold")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()
