import pandas as pd
from datetime import datetime

# ====================================
# Cargar el dataset limpio
# ====================================
clientes = pd.read_csv("data_limpio.csv", parse_dates=["fecha_ultima_transaccion"])
fecha_ref = datetime(2025, 5, 13)

# ====================================
# Calcular métricas RFM
# ====================================
clientes['R'] = (fecha_ref - clientes['fecha_ultima_transaccion']).dt.days  # Recency
clientes['F'] = clientes['frecuencia_transacciones']                        # Frequency
clientes['M'] = clientes['saldo_promedio']                                  # Monetary

# ====================================
# Calcular cuartiles (1 = mejor, 4 = peor para R)
# ====================================
clientes['R_quartile'] = pd.qcut(clientes['R'], 4, labels=[4, 3, 2, 1])  # Cuanto más reciente, mejor
clientes['F_quartile'] = pd.qcut(clientes['F'].rank(method="first"), 4, labels=[1, 2, 3, 4])
clientes['M_quartile'] = pd.qcut(clientes['M'], 4, labels=[1, 2, 3, 4])

# ====================================
# Crear código RFM combinado
# ====================================
clientes['RFM_Score'] = (
    clientes['R_quartile'].astype(str) +
    clientes['F_quartile'].astype(str) +
    clientes['M_quartile'].astype(str)
)

# ====================================
# Clasificación por tipo de cliente (opcional)
# ====================================
def clasificar_cliente(score):
    if score == '111':
        return 'Cliente VIP'
    elif score[0] == '1' and score[2] in ['1', '2']:
        return 'Cliente Leal'
    elif score[0] in ['3', '4']:
        return 'Cliente en Riesgo'
    else:
        return 'Cliente Promedio'

clientes['tipo_cliente'] = clientes['RFM_Score'].apply(clasificar_cliente)

# ====================================
# Resultado
# ====================================
print("\nResumen de Clientes con RFM Score:")
print(clientes[['cliente_id', 'R', 'F', 'M', 'RFM_Score', 'tipo_cliente']].head(10))

# ====================================
# Exportar para usar en clustering o dashboard
# ====================================
clientes.to_csv("clientes_segmentados_rfm.csv", index=False)
print("\n✅ Archivo exportado: clientes_segmentados_rfm.csv")
