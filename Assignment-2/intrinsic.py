import pandas as pd
import numpy as np
import time as t

from sklearn.model_selection import train_test_split
from interpret.glassbox import LinearRegression,ExplainableBoostingRegressor
from interpret import show,set_show_addr,preserve

from sklearn.metrics import r2_score
SEED=42
df=pd.read_csv("data/LengthOfStay.csv")
df["total_diseases"]=np.sum(df.iloc[:,4:14],axis=1)
#print(df["total_diseases"])
columns=["total_diseases","dialysisrenalendstage","asthma","irondef","pneum","substancedependence","psychologicaldisordermajor","depress","psychother","fibrosisandother","malnutrition","hemo","hematocrit","neutrophils","sodium","glucose","bloodureanitro","creatinine","bmi","pulse","respiration","secondarydiagnosisnonicd9"]

X=df[columns]
y=df["lengthofstay"].apply(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=SEED)

#Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred=lin_reg.predict(X_test)
print(f"LINEAR REGRESSION R2:{r2_score(y_test,y_pred)}")

#show(lin_reg.explain_global(),open_browser=True)


#Boosting Regressor
boosting_reg=ExplainableBoostingRegressor(learning_rate=0.01,
                                          n_jobs=-1,
                                          max_rounds=700,
                                          early_stopping_rounds=40,
                                          random_state=SEED)
boosting_reg.fit(X_train,y_train)

y_pred=boosting_reg.predict(X_test)
print(f"BOOSTING REGRESSOR R2:{r2_score(y_test,y_pred)}")

#show(boosting_ref.explain_global(),open_browser=True)

explanations = [
    lin_reg.explain_global(),
    boosting_reg.explain_global()
]

#Open server http://127.0.0.1:7002
set_show_addr(("127.0.0.1",7002))
show(explanations,open_browser=True)

#Save as html
preserve(lin_reg.explain_global(), file_name="Assignment-2/LR_results.html")
preserve(boosting_reg.explain_global(), file_name="Assignment-2/EBR_results.html")
input("Press Enter Close Server")

