import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

# Cargar datos y calcular RFM
clientes = pd.read_csv("data_limpio.csv", parse_dates=["fecha_ultima_transaccion"])
fecha_ref = pd.to_datetime("2025-05-13")

clientes['R'] = (fecha_ref - clientes['fecha_ultima_transaccion']).dt.days
clientes['F'] = clientes['frecuencia_transacciones']
clientes['M'] = clientes['saldo_promedio']

# Simular columna de churn (clientes con alto riesgo de fuga)

clientes['churn'] = (
    (clientes['R'] > 180) &                          # No transaccionan hace más de 6 meses
    (clientes['usa_banca_digital'] == 0) &           # No usan banca digital
    (clientes['productos_contratados'] <= 2)         # Tienen pocos productos
).astype(int)

# Selección de variables predictoras

X = clientes[['R', 'F', 'M', 'productos_contratados', 'ingreso_mensual', 'usa_banca_digital', 'participa_en_promociones']]
y = clientes['churn']

# Entrenamiento del modelo

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Evaluación del modelo

y_pred = modelo.predict(X_test)

print("📊 Matriz de Confusión:\n")
print(confusion_matrix(y_test, y_pred))

print("\n📈 Reporte de Clasificación:\n")
print(classification_report(y_test, y_pred))
