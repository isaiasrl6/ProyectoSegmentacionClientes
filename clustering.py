import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos y preparar RFM

clientes = pd.read_csv("data_limpio.csv", parse_dates=["fecha_ultima_transaccion"])
fecha_ref = pd.to_datetime("2025-05-13")

# Calcular R, F y M

clientes['R'] = (fecha_ref - clientes['fecha_ultima_transaccion']).dt.days
clientes['F'] = clientes['frecuencia_transacciones']
clientes['M'] = clientes['saldo_promedio']

# Selección y estandarización de variables

features = clientes[['R', 'F', 'M', 'ingreso_mensual', 'usa_banca_digital', 'participa_en_promociones']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Aplicar K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clientes['segmento'] = kmeans.fit_predict(scaled_features)

# Descripción por clúster

cluster_summary = clientes.groupby('segmento')[['R', 'F', 'M', 'ingreso_mensual']].mean().round(1)
print("📊 Resumen por segmento:")
print(cluster_summary)

# Visualización con Seaborn

sns.set(style="whitegrid")
g = sns.pairplot(
    clientes,
    hue='segmento',
    vars=['R', 'F', 'M'],
    palette='tab10',
    corner=True
)
g.fig.suptitle('📌 Distribución de clientes por segmento (K-Means)', y=1.02)
plt.show()
