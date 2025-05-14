import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Configuración inicial del dashboard
st.set_page_config(page_title="Segmentación de Clientes", layout="wide")
st.title("Dashboard de Segmentación de Clientes Bancarios")

# Cargar Dataset
clientes = pd.read_csv("data.csv", parse_dates=["fecha_ultima_transaccion"])
fecha_ref = pd.to_datetime("2025-05-13")

# Calcular RFM
clientes['R'] = (fecha_ref - clientes['fecha_ultima_transaccion']).dt.days
clientes['F'] = clientes['frecuencia_transacciones']
clientes['M'] = clientes['saldo_promedio']

# Segmentación K-Means
if 'segmento' not in clientes.columns:
    features = clientes[['R', 'F', 'M', 'ingreso_mensual', 'usa_banca_digital', 'participa_en_promociones']]
    features = features.dropna()
    clientes = clientes.loc[features.index]  # mantener alineado
    scaled = StandardScaler().fit_transform(features)
    clientes['segmento'] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(scaled)

# Asignar etiquetas de perfil a los segmentos
segment_map = {
    0: "Dormidos",
    1: "Premium Digital",
    2: "Tradicional Estable",
    3: "Multiproducto Rentable"
}
clientes['perfil_segmento'] = clientes['segmento'].map(segment_map)

# Simulación de churn
if 'churn' not in clientes.columns:
    clientes['churn'] = (
        (clientes['R'] > 180) &
        (clientes['usa_banca_digital'] == 0) &
        (clientes['productos_contratados'] <= 2)
    ).astype(int)

# Filtros laterales
st.sidebar.header("Filtros")
segmento = st.sidebar.selectbox("Selecciona un segmento", sorted(clientes['segmento'].unique()))
mostrar_churn = st.sidebar.checkbox("Mostrar solo clientes con riesgo de fuga")
df_filtrado = clientes[clientes['segmento'] == segmento]
if mostrar_churn:
    df_filtrado = df_filtrado[df_filtrado['churn'] == 1]

# Tabs de navegación
tab1, tab2, tab3 = st.tabs(["Resumen", "Gráficos", "Detalle"])

with tab1:
    st.subheader("Indicadores del segmento")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clientes", len(df_filtrado))
    col2.metric("Saldo promedio", f"${round(df_filtrado['M'].mean(), 2):,}")
    col3.metric("Frecuencia promedio", round(df_filtrado['F'].mean(), 2))
    churn_pct = round(df_filtrado['churn'].mean() * 100, 2)
    col4.metric("Porcentaje en riesgo", f"{churn_pct}%")

    st.download_button(
        label="Descargar segmento filtrado",
        data=df_filtrado.to_csv(index=False),
        file_name="segmento_filtrado.csv",
        mime="text/csv"
    )

with tab2:
    st.subheader("Análisis Visual por Segmento")
    fig1 = px.histogram(df_filtrado, x="R", nbins=30, title="Distribución de Recencia")
    fig2 = px.box(df_filtrado, y="M", title="Distribución del Saldo Promedio")
    fig3 = px.pie(df_filtrado, names='canal_favorito', title="Canal Favorito en el Segmento")
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.subheader("Detalle de Clientes")
    st.dataframe(df_filtrado[[
        'cliente_id', 'R', 'F', 'M', 'ingreso_mensual',
        'segmento', 'perfil_segmento', 'churn'
    ]])
