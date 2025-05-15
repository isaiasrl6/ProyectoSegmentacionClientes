import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# =========================================
# Cargar datos y calcular métricas RFM
# =========================================
clientes = pd.read_csv("data_limpio.csv", parse_dates=["fecha_ultima_transaccion"])
fecha_ref = pd.to_datetime("2025-05-13")

clientes['R'] = (fecha_ref - clientes['fecha_ultima_transaccion']).dt.days
clientes['F'] = clientes['frecuencia_transacciones']
clientes['M'] = clientes['saldo_promedio']

# =========================================
# Simular columna de churn (cliente en riesgo)
# =========================================
clientes['churn'] = (
    (clientes['R'] > 180) &                          # No transaccionan hace más de 6 meses
    (clientes['usa_banca_digital'] == 0) &           # No usan banca digital
    (clientes['productos_contratados'] <= 2)         # Pocos productos contratados
).astype(int)

# =========================================
# Selección de variables predictoras
# =========================================
X = clientes[['R', 'F', 'M', 'productos_contratados', 'ingreso_mensual', 'usa_banca_digital', 'participa_en_promociones']]
y = clientes['churn']

# =========================================
# Separar datos en entrenamiento y prueba
# =========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================
# Entrenar modelo Random Forest
# =========================================
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# =========================================
# Evaluación del modelo
# =========================================
y_pred = modelo.predict(X_test)

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Visualización de la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=modelo.classes_)
disp.plot(cmap='Blues')
plt.title("Matriz de Confusión - Churn Prediction")
plt.show()

# =========================================
# Importancia de variables
# =========================================
importancia = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nImportancia de variables:")
print(importancia)

# =========================================
# Guardar resultado completo con predicción
# =========================================
clientes['churn_predicho'] = modelo.predict(X)
clientes.to_csv("clientes_con_churn.csv", index=False)
print("\n✅ Archivo exportado: clientes_con_churn.csv")
