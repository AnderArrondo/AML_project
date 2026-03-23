import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# CONFIG
csv_path = "./data/stroke.csv"
stroke_palette = {0: "#198038", 1: "#ff832b"}
work_palette = {
    "Private": "#fd7f6f",
    "Self-employed": "#b2e061",
    "Govt_job": "#ffb55a",
    "children": "#8bd3c7",
    "Never_worked": "#ffa600"
}
# DATA COLLECTION
df=pd.read_csv(csv_path)

#TODO HAY QUE QUITAR LOS NULLS
# interpolation no es en esta entrega no?

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


# DISTRIBUTION PLOTS
fig, axes = plt.subplots(3,2, figsize=(12,11))
fig.suptitle("Data Distribution")

order = df["gender"].value_counts().index
plot1=sns.countplot(
    data=df,
    x="gender",
    ax=axes[0,0],
    order=order
)
plot1.bar_label(plot1.containers[0])
plot1.set_title("Gender")
plot1.set_xlabel("")
plot1.set_ylim(0, 28000)

order = df["work_type"].value_counts().index
plot2=sns.countplot(
    data=df, 
    x="work_type",
    hue="work_type",
    palette=work_palette,
    ax=axes[0,1],
    order=order
)
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
plt.show()

# CORRELATION HEATMAP
corr_matrix=df_dummies.corr()

fig = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale="RdBu_r"
)

fig.write_html("corr_matrix.html")

# RELATIONSHIP BETWEEN VARIABLES
fig, axes = plt.subplots(1, 3, figsize=(16, 8))
fig.suptitle("Comparison of Health Metrics by Stroke Status", fontsize=16)

sns.kdeplot(
    data=df[df['stroke'] == 0], x="age", 
    fill=True, color=stroke_palette[0],
    label="Total Population", alpha=0.3, ax=axes[0]
)
sns.kdeplot(
    data=df[df['stroke'] == 1], x="age", 
    fill=True, color=stroke_palette[1],
    label="Stroke Cases", alpha=0.6, ax=axes[0]
)
axes[0].set_title("Age: Global vs. Stroke")
axes[0].legend()

sns.kdeplot(
    data=df[df['stroke'] == 0], x="bmi", 
    fill=True, color=stroke_palette[0],
    label="Total Population", alpha=0.3, ax=axes[1]
)
sns.kdeplot(
    data=df[df['stroke'] == 1], x="bmi", 
    fill=True, color=stroke_palette[1],
    label="Stroke Cases", alpha=0.6, ax=axes[1]
)
axes[1].set_title("BMI: Global vs. Stroke")
axes[1].legend()

sns.violinplot(
    data=df, x=[""] * len(df), y="avg_glucose_level", 
    hue="stroke", split=True, inner="quart", 
    palette=stroke_palette, ax=axes[2]
)
axes[2].set_title("Glucose Levels")

plt.tight_layout()
plt.show()

pivot_table = df.groupby(['work_type', 'is_rural'])['stroke'].mean().unstack()

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Work Type Analysis: Stroke Risk vs. Age Distribution", fontsize=16)

pivot_table = df.groupby(['work_type', 'is_rural'])['stroke'].mean().unstack()
sns.heatmap(pivot_table, annot=True, fmt=".1%", cmap="YlOrRd", ax=axes[0])
axes[0].set_title("Stroke Risk: Work vs Residence")

sns.kdeplot(
    data=df, x="age", hue="work_type", 
    fill=True, common_norm=False, alpha=0.3, 
    palette=work_palette, ax=axes[1]
)
axes[1].set_title("Age Density by Work Type")

sns.kdeplot(
    data=df, 
    x="age", 
    hue="work_type", 
    fill=True, 
    common_norm=False, 
    alpha=0.2,
    ax=axes[1]
)
axes[1].set_title("Age Density by Work Type")
axes[1].set_xlabel("Age")
axes[1].set_ylabel("Density")

plt.tight_layout()
plt.show()

pct_table = pd.crosstab(df['hypertension'], df['stroke'], normalize='index')

color_list = [stroke_palette[0], stroke_palette[1]]

ax = pct_table.plot(kind='bar', stacked=True, color=color_list, figsize=(8, 5))

for container in ax.containers:
    labels = [f'{val*100:.1f}%' for val in container.datavalues]
    ax.bar_label(container, labels=labels, label_type='center', color='white', fontweight='bold')

plt.title("Stroke Risk: Hypertension Impact")
plt.xlabel("Hypertension (0 = No, 1 = Yes)")
plt.legend(title="Stroke", labels=['No', 'Yes'], loc='upper left', bbox_to_anchor=(1, 1))
plt.xticks(rotation=0)
plt.show()
