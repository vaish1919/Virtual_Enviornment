import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("insurance.csv")

lb = LabelEncoder()

df['sex'] = lb.fit_transform(df['sex'])
df['region'] = lb.fit_transform(df['region'])
df['smoker'] = lb.fit_transform(df['smoker'])

X = df.drop(columns=['charges'])
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

r2 = r2_score(y_test, y_pred)

print(f"{r2}")