import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from matplotlib import use
use('Agg')
#Don't show 

df = pd.read_csv("Assignment-2/LengthOfStay.csv")
folder="Assignment-2/images"
img_counter=1

# 1. HEXBIN PLOT (Para relación entre dos variables numéricas)
# Reemplaza el Scatter Plot. Agrupa puntos en hexágonos y usa color para la densidad.
plt.figure(figsize=(10, 6))
plt.hexbin(df['glucose'], df['lengthofstay'], gridsize=30, cmap='Blues')
plt.colorbar(label='Conteo de pacientes')
plt.title('Densidad de Glucosa vs Estancia Hospitalaria')
plt.xlabel('Glucosa')
plt.ylabel('Días de Estancia (LOS)')
plt.savefig(f"Assignment-2/images/plot_{img_counter}");img_counter+=1

# 5. RIDGE PLOT (Joyplot)
# Perfecto para comparar distribuciones de una variable numérica a través de muchas categorías.
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='hematocrit', hue='facid', fill=True, common_norm=False, alpha=0.25, palette="tab10")
plt.title('Comparación de Hematocrito entre diferentes Instalaciones (FACID)')
plt.savefig(f"Assignment-2/images/plot_{img_counter}");img_counter+=1

# 2. KDE PLOT (Kernel Density Estimate)
# Muestra la "forma" de la distribución de una variable sin mostrar puntos individuales.
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='hematocrit', hue='gender', fill=True, common_norm=False, palette='viridis')
plt.title('Distribución de Densidad del Hematocrito por Género')
plt.savefig(f"Assignment-2/images/plot_{img_counter}");img_counter+=1

# 3. VIOLIN PLOTS
# Combina un Boxplot con la densidad de la distribución. Ideal para comparar categorías.
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='facid', y='lengthofstay', inner="quart")
plt.title('Distribución de Estancia por Centro Hospitalario (FACID)')
plt.savefig(f"Assignment-2/images/plot_{img_counter}");img_counter+=1

# 4. MAPA DE CALOR DE CORRELACIÓN (Matriz)
# Resume millones de relaciones en una sola tabla de colores.
plt.figure(figsize=(12, 10))
corr = df.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(corr, annot=False, cmap='RdBu_r', center=0)
plt.title('Correlación General de Variables Clínicas')
plt.savefig(f"Assignment-2/images/plot_{img_counter}");img_counter+=1

# 1. JOINTPLOT CON DENSIDAD (KDE)
# Muestra la relación entre dos variables y sus distribuciones marginales al mismo tiempo.
# Ideal para ver dónde se "amontona" la masa de pacientes.
sns.jointplot(data=df, x='bmi', y='glucose', kind="kde", cmap="magma", fill=True)
plt.suptitle('Densidad Conjunta: BMI vs Glucosa', y=1.02)
plt.savefig(f"Assignment-2/images/plot_{img_counter}");img_counter+=1

# 2. BOXENPLOT (Letter-Value Plot)
# Es una versión mejorada del Boxplot para "Big Data". 
# Muestra más cuantiles, lo que permite ver mejor la distribución en las "colas" del dataset.
plt.figure(figsize=(12, 6))
sns.boxenplot(data=df, x='rcount', y='lengthofstay', palette="viridis")
plt.title('Estancia Hospitalaria por Número de Readmisiones (Escalado para grandes datos)')
plt.savefig(f"Assignment-2/images/plot_{img_counter}");img_counter+=1

# 3. CUMULATIVE DISTRIBUTION PLOT (ECDF)
# Muestra qué porcentaje de la población está por debajo de cierto valor.
# Útil para responder: "¿Qué % de pacientes se queda menos de 5 días?"
plt.figure(figsize=(10, 6))
sns.ecdfplot(data=df, x='lengthofstay', hue='gender')
plt.title('Distribución Acumulada de la Estancia Hospitalaria')
plt.grid(True, alpha=0.3)
plt.savefig(f"Assignment-2/images/plot_{img_counter}");img_counter+=1

# 4. CLUSTERMAP
# Agrupa automáticamente variables y observaciones que se comportan de forma similar.
# Nota: En datasets gigantes, se suele aplicar a una muestra o a las correlaciones.
plt.figure(figsize=(10, 10))
sns.clustermap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='vlag')
plt.savefig(f"Assignment-2/images/plot_{img_counter}");img_counter+=1


plt.figure(figsize=(10, 8))

df_sample = df.sample(frac=0.1, random_state=42) 

# Usa df_sample para tus plots
#sns.kdeplot(data=df_sample, x='glucose', y='lengthofstay')

sns.kdeplot(
    data=df_sample, 
    x='bmi', 
    y='glucose', 
    hue='gender', 
    fill=True,       # Rellena los contornos
    alpha=0.4,       # Transparencia para ver superposiciones
    thresh=0.05,     # Nivel mínimo de densidad para mostrar
    cmap='viridis'   # Paleta de colores atractiva
)
plt.title('Densidad Conjunta de BMI y Glucosa por Género\n(Resumen para Grandes Datos)')
plt.xlabel('Índice de Masa Corporal (BMI)')
plt.ylabel('Nivel de Glucosa')
plt.grid(True, alpha=0.2)
plt.savefig(f"Assignment-2/images/plot_{img_counter}");img_counter+=1


# 2. Gráfico Híbrido: Cantidad + Distribución
# Compara la Estancia Hospitalaria por Instalación (FACID).
# El boxplot da el resumen estadístico; el swarmplot muestra la forma real.
plt.figure(figsize=(12, 7))

# Capa 1: Swarmplot (mostrando los puntos individuales compactados)
# Nota: En datasets GIGANTES (>50k filas), el swarmplot puede ser lento. 
# Si es así, reduce 'size' o usa solo 'stripplot'.
sns.swarmplot(
    data=df_sample, 
    x='facid', 
    y='lengthofstay', 
    hue='gender', 
    dodge=True,       # Separa los puntos por género
    size=2.5,         # Tamaño pequeño para manejar más puntos
    alpha=0.4,        # Transparencia
    palette='pastel'
)

sns.boxplot(
    data=df_sample, 
    x='facid', 
    y='lengthofstay', 
    hue='gender', 
    dodge=True,       # Separa por género (coincidiendo con swarmplot)
    width=0.4,        # Más estrecho para no tapar puntos
    color='grey',     # Color neutro para no distraer
    boxprops=dict(alpha=0.3) # Boxplot semi-transparente
)

plt.title('Distribución Detallada de Estancia por Instalación y Género')
plt.xlabel('Instalación Hospitalaria (FACID)')
plt.ylabel('Días de Estancia (LOS)')
# Unificamos las leyendas
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[0:2], labels[0:2], title='Género', loc='upper right')

plt.tight_layout()
plt.savefig(f"Assignment-2/images/plot_{img_counter}");img_counter+=1

