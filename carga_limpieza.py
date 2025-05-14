import pandas as pd
import numpy as np  


# Cargar el CSV sin errores
df = pd.read_csv("data.csv")

# Convertir fecha a datetime forzando errores (si los hubiera)
df['fecha_ultima_transaccion'] = pd.to_datetime(df['fecha_ultima_transaccion'], errors='coerce', dayfirst=True)

# ========================
# PERFILADO Y VALIDACIÓN
# ========================

# Revisión inicial
print("🔎 Vista previa:")
print(df.head())
print("\n📋 Tipos de datos:")
print(df.dtypes)
print("\n❌ Valores nulos por columna:")
print(df.isnull().sum())

# ========================
# LIMPIEZA DE DATOS
# ========================

# 1. Reemplazar valores vacíos o nulos en 'canal_favorito' con 'Desconocido'
if 'canal_favorito' in df.columns:
    df['canal_favorito'] = df['canal_favorito'].replace('', np.nan)
    df['canal_favorito'] = df['canal_favorito'].fillna('Desconocido')

# 2. Eliminar registros sin 'provincia' si existe
if 'provincia' in df.columns:
    df = df.dropna(subset=['provincia'])

# 3. Reemplazar saldo 0 o nulo por la mediana (si existe la columna)
if 'saldo_promedio' in df.columns:
    df['saldo_promedio'] = df['saldo_promedio'].replace(0, np.nan)
    mediana = df['saldo_promedio'].median()
    df['saldo_promedio'] = df['saldo_promedio'].fillna(mediana)

# 4. Validar 'productos_contratados' entre 1 y 6 si aplica
if 'productos_contratados' in df.columns:
    df = df[df['productos_contratados'].between(1, 6)]

# 5. Validar 'frecuencia_transacciones' entre 1 y 50 si aplica
if 'frecuencia_transacciones' in df.columns:
    df = df[df['frecuencia_transacciones'].between(1, 50)]

# 6. Eliminar duplicados por ID si existe 'cliente_id'
if 'cliente_id' in df.columns:
    df = df.drop_duplicates(subset='cliente_id')


# Exportamos el archivo limpio

df.to_csv("data_limpio.csv", index=False)
print("\n✅ Archivo limpio generado: data_limpio.csv")

