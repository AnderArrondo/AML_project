import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#DATA LOAD
df=pd.read_csv("data/stroke.csv")

#TODO HAY QUE QUITAR LOS NULLS

#DATA TYPE CLEAN
df=df.drop(columns=["id"])
df["age"]=df["age"].apply(int)
df["ever_married"] = df["ever_married"].map({"Yes":1, "No":0})
df=df.rename(columns={"Residence_type":"is_rural"})
df["is_rural"]=df["is_rural"].map({"Rural":1,"Urban":0})
df_dummies= pd.get_dummies(df,columns=["work_type","gender","smoking_status"])

for column in df.columns:
    print(column,":")
    print(df[column].unique())


#PLOTS Distribution

fig, axes = plt.subplots(3,2, figsize=(12,11))
fig.suptitle("Data Distribution")

order = df["gender"].value_counts().index

plot1=sns.countplot(data=df, x="gender", ax=axes[0,0], order=order)
plot1.bar_label(plot1.containers[0])
plot1.set_title("Gender")
plot1.set_xlabel("")
plot1.set_ylim(0, 28000)

order = df["work_type"].value_counts().index

plot2=sns.countplot(data=df, x="work_type", ax=axes[0,1], order=order)
plot2.bar_label(plot2.containers[0])
plot2.set_title("Work Type")
plot2.set_xlabel("")
plot2.set_ylim(0, 27000)




order = df["is_rural"].value_counts().index

plot3=sns.countplot(data=df, x="is_rural", ax=axes[1,0], order=order)
plot3.bar_label(plot3.containers[0])
plot3.set_title("Rural Residence")
plot3.set_xlabel("")
plot3.set_ylim(0, 23000)




order = df["smoking_status"].value_counts().index

plot4=sns.countplot(data=df, x="smoking_status", ax=axes[1,1], order=order)
plot4.bar_label(plot4.containers[0])
plot4.set_title("Smoking Status")
plot4.set_xlabel("")
plot4.set_ylim(0, 18000)



plot5=sns.histplot(data=df, x="avg_glucose_level", ax=axes[2,0],binwidth=5)
plot5.set_title("Glucose Level")
plot5.set_xlabel("")



plot6=sns.histplot(data=df, x="bmi", ax=axes[2,1],binwidth=2)
plot6.set_title("BMI")
plot6.set_xlabel("")

plt.tight_layout()
#plt.show()

#Heatmap

corr_matrix=df_dummies.corr()

fig = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale="RdBu_r"
)

fig.write_html("corr_matrix.html")