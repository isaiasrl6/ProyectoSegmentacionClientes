import pandas as pd
from datetime import datetime

# Cargar el dataset
clientes = pd.read_csv("data_limpio.csv", parse_dates=["fecha_ultima_transaccion"])

# Definir fecha de referencia para Recency
fecha_ref = datetime(2025, 5, 13)

# Calcular variables RFM
clientes['R'] = (fecha_ref - clientes['fecha_ultima_transaccion']).dt.days
clientes['F'] = clientes['frecuencia_transacciones']
clientes['M'] = clientes['saldo_promedio']

# Asignar cuartiles (etiquetas ordenadas del peor al mejor cliente)
clientes['R_quartile'] = pd.qcut(clientes['R'], 4, labels=[4, 3, 2, 1])
clientes['F_quartile'] = pd.qcut(clientes['F'].rank(method="first"), 4, labels=[1, 2, 3, 4])
clientes['M_quartile'] = pd.qcut(clientes['M'], 4, labels=[1, 2, 3, 4])

# Generar RFM Score como string concatenado
clientes['RFM_Score'] = (
    clientes['R_quartile'].astype(str) +
    clientes['F_quartile'].astype(str) +
    clientes['M_quartile'].astype(str)
)

# Mostrar muestra de resultados
print("\n📊 Muestra de clientes con RFM:")
print(clientes[['cliente_id', 'R', 'F', 'M', 'R_quartile', 'F_quartile', 'M_quartile', 'RFM_Score']].head(10))
