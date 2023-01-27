import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('../DATA/Advertising.csv')
#print(df.head())
X = df.drop('sales',axis=1)
y = df['sales']

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
# X_validation, X_holdout_test, y_validation, y_holdout_test = train_test_split(X_test, y_test, test_size=0.5, random_state=101)

model = RandomForestRegressor(n_estimators=30,random_state=101)
model.fit(X,y)

import joblib
joblib.dump(model,'RFR_model.pkl')
joblib.dump(list(X.columns),'col_names.pkl')