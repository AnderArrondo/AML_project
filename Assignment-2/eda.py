import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from matplotlib import use
use('Agg')
#Don't show 

df = pd.read_csv("Assignment-2/data/LengthOfStay.csv")
folder="Assignment-2/images"
img_counter=1

# CLEANING
order=["A","B","C","D","E"]
df['facid'] = pd.Categorical(df['facid'], categories=order, ordered=True)

order=["0","1","2","3","4","5+"]
df['rcount'] = pd.Categorical(df['rcount'], categories=order, ordered=True)


# 1. HEXBIN PLOT (For relationship between two numerical variables)
# Replaces the Scatter Plot. Groups points into hexagons and uses color for density.
plt.figure(figsize=(10, 6))
plt.hexbin(df['glucose'], df['lengthofstay'], gridsize=30, cmap='Blues')
plt.colorbar(label='Patient Count')
plt.title('Density of Glucose vs Length of Stay')
plt.xlabel('Glucose')
plt.ylabel('Length of Stay (LOS)')
plt.savefig(f"Assignment-2/images/plot_{img_counter}");img_counter+=1

# 5. RIDGE PLOT (Joyplot)
# Perfect for comparing distributions of a numerical variable across many categories.
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='hematocrit', hue='facid', fill=True, common_norm=False, alpha=0.25, palette="tab10")
plt.title('Hematocrit Comparison across different Facilities (FACID)')
plt.savefig(f"Assignment-2/images/plot_{img_counter}");img_counter+=1

# 2. KDE PLOT (Kernel Density Estimate)
# Shows the "shape" of the distribution of a variable without showing individual points.
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='hematocrit', hue='gender', fill=True, common_norm=False, palette='viridis')
plt.title('Hematocrit Density Distribution by Gender')
plt.savefig(f"Assignment-2/images/plot_{img_counter}");img_counter+=1

# 3. VIOLIN PLOTS
# Combines a Boxplot with the density of the distribution. Ideal for comparing categories.
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='facid', y='lengthofstay', inner="quart")
plt.title('Stay Distribution by Hospital Facility (FACID)')
plt.savefig(f"Assignment-2/images/plot_{img_counter}");img_counter+=1

# 4. CORRELATION HEATMAP (Matrix)
# Summarizes millions of relationships in a single color-coded table.
plt.figure(figsize=(12, 10))
corr = df.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(corr, annot=False, cmap='RdBu_r', center=0)
plt.title('General Correlation of Clinical Variables')
plt.savefig(f"Assignment-2/images/plot_{img_counter}");img_counter+=1

# 1. JOINTPLOT WITH DENSITY (KDE)
# Shows the relationship between two variables and their marginal distributions at the same time.
# Ideal for seeing where the patient mass "clusters".
sns.jointplot(data=df, x='bmi', y='glucose', kind="kde", cmap="magma", fill=True)
plt.suptitle('Joint Density: BMI vs Glucose', y=1.02)
plt.savefig(f"Assignment-2/images/plot_{img_counter}");img_counter+=1

# 2. BOXENPLOT (Letter-Value Plot)
# An improved version of the Boxplot for "Big Data".
# Shows more quantiles, allowing for a better view of the distribution in the dataset "tails".
plt.figure(figsize=(12, 6))

sns.boxenplot(data=df, x='rcount', y='lengthofstay', legend=False)
plt.title('Length of Stay by Number of Readmissions (Scaled for Big Data)')
plt.savefig(f"Assignment-2/images/plot_{img_counter}");img_counter+=1

# 3. CUMULATIVE DISTRIBUTION PLOT (ECDF)
# Shows what percentage of the population is below a certain value.
# Useful for answering: "What % of patients stay less than 5 days?"
plt.figure(figsize=(10, 6))
sns.ecdfplot(data=df, x='lengthofstay', hue='gender')
plt.title('Cumulative Distribution of Length of Stay')
plt.grid(True, alpha=0.3)
plt.savefig(f"Assignment-2/images/plot_{img_counter}");img_counter+=1

# 4. CLUSTERMAP
# Automatically groups variables and observations that behave similarly.
# Note: In giant datasets, it is usually applied to a sample or to correlations.
plt.figure(figsize=(10, 10))
sns.clustermap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='vlag')
plt.savefig(f"Assignment-2/images/plot_{img_counter}");img_counter+=1


plt.figure(figsize=(10, 8))

df_sample = df.sample(frac=0.1, random_state=42) 

# Use df_sample for your plots
#sns.kdeplot(data=df_sample, x='glucose', y='lengthofstay')

sns.kdeplot(
    data=df_sample, 
    x='bmi', 
    y='glucose', 
    hue='gender', 
    fill=True,       # Fills the contours
    alpha=0.4,       # Transparency to see overlaps
    thresh=0.05,     # Minimum density level to display
    cmap='viridis'   # Attractive color palette
)
plt.title('Joint Density of BMI and Glucose by Gender\n(Summary for Big Data)')
plt.xlabel('Body Mass Index (BMI)')
plt.ylabel('Glucose Level')
plt.grid(True, alpha=0.2)
plt.savefig(f"Assignment-2/images/plot_{img_counter}");img_counter+=1


# 2. Hybrid Plot: Quantity + Distribution
# Compares Hospital Length of Stay by Facility (FACID).
# The boxplot gives the statistical summary; the swarmplot shows the real shape.
plt.figure(figsize=(12, 7))

# Layer 1: Swarmplot (showing compacted individual points)
# Note: In GIANT datasets (>50k rows), the swarmplot can be slow. 
# If so, reduce 'size' or use only 'stripplot'.
sns.swarmplot(
    data=df_sample, 
    x='facid', 
    y='lengthofstay', 
    hue='gender', 
    dodge=True,       # Separates points by gender
    size=2.5,         # Small size to handle more points
    alpha=0.4,        # Transparency
    palette='pastel'
)

sns.boxplot(
    data=df_sample, 
    x='facid', 
    y='lengthofstay', 
    hue='gender', 
    dodge=True,       # Separates by gender (matching swarmplot)
    width=0.4,        # Narrower so as not to cover points
    color='grey',     # Neutral color to avoid distraction
    boxprops=dict(alpha=0.3) # Semi-transparent boxplot
)

plt.title('Detailed Stay Distribution by Facility and Gender')
plt.xlabel('Hospital Facility (FACID)')
plt.ylabel('Length of Stay (LOS)')
# Unifying legends
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[0:2], labels[0:2], title='Gender', loc='upper right')

plt.tight_layout()
plt.savefig(f"Assignment-2/images/plot_{img_counter}");img_counter+=1