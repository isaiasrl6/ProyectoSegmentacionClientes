import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# Cargar y preparar datos RFM
# ============================
clientes = pd.read_csv("data_limpio.csv", parse_dates=["fecha_ultima_transaccion"])
fecha_ref = pd.to_datetime("2025-05-13")

clientes['R'] = (fecha_ref - clientes['fecha_ultima_transaccion']).dt.days
clientes['F'] = clientes['frecuencia_transacciones']
clientes['M'] = clientes['saldo_promedio']

# =======================================
# Selección de variables para clustering
# =======================================
features = clientes[['R', 'F', 'M', 'ingreso_mensual', 'usa_banca_digital', 'participa_en_promociones']]

# Eliminar filas con valores nulos (si existen)
features = features.dropna()
clientes = clientes.loc[features.index]

# Estandarización de variables
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# ============================
# Aplicar K-Means Clustering
# ============================
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clientes['segmento'] = kmeans.fit_predict(scaled_features)

# Etiquetas opcionales por tipo de cliente
segment_labels = {
    0: "Dormido",
    1: "Premium",
    2: "Tradicional",
    3: "Multiproducto"
}
clientes['perfil_segmento'] = clientes['segmento'].map(segment_labels)

# ============================
# Resumen de cada segmento
# ============================
cluster_summary = clientes.groupby('perfil_segmento')[['R', 'F', 'M', 'ingreso_mensual']].mean().round(1)
print("Resumen por segmento:")
print(cluster_summary)

# ============================
# Visualización de resultados
# ============================
sns.set(style="whitegrid")
g = sns.pairplot(
    clientes,
    hue='perfil_segmento',
    vars=['R', 'F', 'M'],
    palette='tab10',
    corner=True
)
g.fig.suptitle('Distribución de clientes por segmento (K-Means)', y=1.02)
plt.show()

# ============================
# Exportar resultados
# ============================
clientes.to_csv("clientes_clusterizados.csv", index=False)
print("Archivo exportado: clientes_clusterizados.csv")
