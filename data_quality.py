import pandas as pd
import numpy as np

# Cargar el CSV sin errores
df = pd.read_csv("data_limpio.csv")

#CARGA DEL DATASET 
df['fecha_ultima_transaccion'] = pd.to_datetime(df['fecha_ultima_transaccion'], errors='coerce', dayfirst=True)

# CONFIGURACIÓN DE REGLAS DE CALIDAD 
reglas = {
    'frecuencia_transacciones': {'min': 1, 'max': 50},
    'saldo_promedio': {'min': 1000},
    'productos_contratados': {'min': 1, 'max': 6},
    'canal_favorito': {'not_null': True},
    'provincia': {'not_null': True},
    'fecha_ultima_transaccion': {'not_null': True}
}

# CREACIÓN DEL REPORTE DE CALIDAD 
reporte = []

for col in df.columns:
    datos_col = df[col]
    tipo = datos_col.dtypes
    total = len(df)
    nulos = datos_col.isnull().sum()
    porcentaje_nulos = round((nulos / total) * 100, 2)
    unicos = datos_col.nunique()
    duplicados = df.duplicated(subset=col).sum() if col != 'cliente_id' else 0
    minimo = datos_col.min() if np.issubdtype(tipo, np.number) or np.issubdtype(tipo, np.datetime64) else None
    maximo = datos_col.max() if np.issubdtype(tipo, np.number) or np.issubdtype(tipo, np.datetime64) else None
    valores_fuera_rango = None

    # Validaciones específicas
    if col in reglas:
        regla = reglas[col]
        if 'min' in regla:
            valores_fuera_rango = df[df[col] < regla['min']].shape[0]
        if 'max' in regla and valores_fuera_rango is not None:
            valores_fuera_rango += df[df[col] > regla['max']].shape[0]
        elif 'max' in regla:
            valores_fuera_rango = df[df[col] > regla['max']].shape[0]

    reporte.append({
        "Columna": col,
        "Tipo de Dato": tipo,
        "Valores Únicos": unicos,
        "Nulos Totales": nulos,
        "% Nulos": porcentaje_nulos,
        "Duplicados": duplicados,
        "Mínimo": minimo,
        "Máximo": maximo,
        "Fuera de Rango (según reglas)": valores_fuera_rango
    })

# CONVERTIR A DATAFRAME Y GUARDAR 
reporte_df = pd.DataFrame(reporte)
reporte_df.to_csv("reporte_calidad_completo.csv", index=False)
print("✅ Reporte de calidad generado: reporte_calidad_completo.csv")
