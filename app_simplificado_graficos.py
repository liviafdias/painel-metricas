# app_simplificado.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- Funções utilitárias ---
def ajeita_data(t):
    try:
        return pd.to_datetime(t)
    except:
        return pd.NaT

def len_texto(x):
    try:
        return len(str(x).split())
    except:
        return 0

@st.cache_data
def carregar_dados_brasil():
    df = pd.read_csv('TSNE_BR_COM_SENTIMENTO.csv', sep=';')
    df['DT_PUBLICACAO'] = df['DT_PUBLICACAO'].apply(ajeita_data)
    df.dropna(subset=['DT_PUBLICACAO'], inplace=True)
    df.sort_values('DT_PUBLICACAO', inplace=True)
    df.rename(columns={'N_CURTIDAS_NEW': 'CURTIDAS', 'N_COMENTARIOS_NEW': "COMENTARIOS"}, inplace=True)
    return df

@st.cache_data
def carregar_dados_suecia():
    df = pd.read_csv('TSNE_SE_COM_SENTIMENTO.csv', sep=';')
    df['DT_PUBLICACAO'] = df['DT_PUBLICACAO'].apply(ajeita_data)
    df.dropna(subset=['DT_PUBLICACAO'], inplace=True)
    df.sort_values('DT_PUBLICACAO', inplace=True)
    df.rename(columns={'N_CURTIDAS_NEW': 'CURTIDAS', 'N_COMENTARIOS_NEW': "COMENTARIOS"}, inplace=True)
    return df

# --- função auxiliar para escolher a periodicidade (mês/ano) ---
def agrega_periodo(df, col_valores=None):
    if df.empty:
        return pd.Series([0])
    tmp = df.copy()
    if col_valores:  # curtidas ou comentários
        tmp[col_valores] = pd.to_numeric(tmp[col_valores], errors="coerce").fillna(0)
    # aplica filtro mês/ano
    if agrupamento == "ano":
        tmp["PERIODO"] = tmp["DT_PUBLICACAO"].dt.to_period("Y").dt.to_timestamp()
    else:  # default mês
        tmp["PERIODO"] = tmp["DT_PUBLICACAO"].dt.to_period("M").dt.to_timestamp()
    # agrupamento
    if col_valores:
        return tmp.groupby("PERIODO")[col_valores].sum()
    else:
        return tmp.groupby("PERIODO").size()

# máximos globais
def max_publicacoes_global():
    s_br = agrega_periodo(brasil)
    s_se = agrega_periodo(suecia)
    return max(float(s_br.max() or 0), float(s_se.max() or 0))

def max_curtidas_global():
    s_br = agrega_periodo(brasil, "CURTIDAS")
    s_se = agrega_periodo(suecia, "CURTIDAS")
    return max(float(s_br.max() or 0), float(s_se.max() or 0))

def max_comentarios_global():
    s_br = agrega_periodo(brasil, "COMENTARIOS")
    s_se = agrega_periodo(suecia, "COMENTARIOS")
    return max(float(s_br.max() or 0), float(s_se.max() or 0))

# ------------------ Dados base e filtros ------------------
st.set_page_config(page_title="Painel de Métricas", layout="wide", initial_sidebar_state="expanded")

st.markdown('<h1 style="text-align:center;">Painel de Métricas Simplificado</h1>', unsafe_allow_html=True)

# Carregamentos
try:
    brasil = carregar_dados_brasil()
except Exception as e:
    st.warning(f"Não foi possível carregar dados do Brasil: {e}")
    brasil = pd.DataFrame()
try:
    suecia = carregar_dados_suecia()
except Exception as e:
    st.warning(f"Não foi possível carregar dados da Suécia: {e}")
    suecia = pd.DataFrame()

# Sidebar (filtros)
with st.sidebar:
    st.header("Filtros")

    base_marcas = brasil if not brasil.empty else suecia
    todas_marcas = sorted(list(base_marcas["MARCA"].unique())) if "MARCA" in base_marcas.columns else []

    marcas_selecionadas = st.multiselect(
        "Comparar Marcas",
        options=todas_marcas,
        default=[],
        placeholder="+ Adicionar marcas"
    )

    paises_opts = []
    if not brasil.empty: paises_opts.append("BR")
    if not suecia.empty: paises_opts.append("SE")

    paises = st.multiselect(
        "País",
        options=paises_opts or ["BR", "SE"],
        default=[],
        placeholder="Selecionar país(es)"
    )

    anos_br = brasil['DT_PUBLICACAO'].dt.year.unique().tolist() if 'DT_PUBLICACAO' in brasil.columns else []
    anos_se = suecia['DT_PUBLICACAO'].dt.year.unique().tolist() if 'DT_PUBLICACAO' in suecia.columns else []
    anos_disponiveis = sorted(set(anos_br) | set(anos_se)) or list(range(2012, 2026))

    periodo_selecionado = st.multiselect(
        "Período (anos)",
        options=anos_disponiveis,
        default=[],
        placeholder="Selecionar ano(s)"
    )

    agrupamento = st.radio(
        "Agrupar por",
        options=["mês", "ano"],
        index=0,
        horizontal=True
    )

def filtra_dados():
    df_list = []
    if "BR" in paises and not brasil.empty:
        df_br = brasil.copy(); df_br["PAIS"] = "BR"; df_list.append(df_br)
    if "SE" in paises and not suecia.empty:
        df_se = suecia.copy(); df_se["PAIS"] = "SE"; df_list.append(df_se)
    if not df_list:
        return pd.DataFrame()
    df = pd.concat(df_list, ignore_index=True)

    for _c in ["CURTIDAS", "COMENTARIOS"]:
        if _c in df.columns:
            df[_c] = pd.to_numeric(df[_c], errors="coerce").fillna(0)

    if periodo_selecionado:
        df["ANO"] = df["DT_PUBLICACAO"].dt.year
        df = df[df["ANO"].isin(periodo_selecionado)]
    if marcas_selecionadas:
        df = df[df["MARCA"].isin(marcas_selecionadas)]
    df["MARCA_PAIS"] = df["MARCA"] + " - " + df["PAIS"]
    if agrupamento == "ano":
        df['PERIODO'] = df['DT_PUBLICACAO'].dt.to_period("Y").dt.to_timestamp()
    else:
        df['PERIODO'] = df['DT_PUBLICACAO'].dt.to_period("M").dt.to_timestamp()

    return df

dados = filtra_dados()

# ------------------ Métricas e Gráficos Principais ------------------
if dados.empty:
    st.info("Ajuste os filtros na barra lateral para visualizar as métricas.")
else:
    # Cartas rápidas
    colm1, colm2, colm3 = st.columns(3)
    total_publicacoes = len(dados)
    total_curtidas = int(dados['CURTIDAS'].sum())
    total_comentarios = int(dados['COMENTARIOS'].sum())
    colm1.metric("Total de Publicações", f"{total_publicacoes:,}".replace(",", "."))
    colm2.metric("Total de Curtidas", f"{total_curtidas:,}".replace(",", "."))
    colm3.metric("Total de Comentários", f"{total_comentarios:,}".replace(",", "."))

    # --- Gráfico de Publicações ---
    df_pub = dados.groupby(['PERIODO', 'MARCA_PAIS']).size().reset_index(name='PUBLICAÇÕES')
    ymax_pub = max_publicacoes_global()

    fig_pub = px.line(
        df_pub, x='PERIODO', y='PUBLICAÇÕES', color='MARCA_PAIS',
        title=f'Publicações por {agrupamento}', markers=True
    )
    fig_pub.update_layout(
        xaxis_title='Período',
        yaxis_title='Quantidade',
        legend_title='Marca - País',
        hovermode='x unified',
        yaxis_range=[0, ymax_pub],  # <-- agora fixo no máximo global
        xaxis_range=['2012-01-01', df_pub['PERIODO'].max()]
    )
    fig_pub.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>Período: %{x}<br>Publicações: %{y:,}'
    )
    st.plotly_chart(fig_pub, use_container_width=True)

    # --- Gráfico de Curtidas ---
    df_curtidas = dados.groupby(['PERIODO', 'MARCA_PAIS'])['CURTIDAS'].sum().reset_index()
    ymax_curt = max_curtidas_global()

    fig_curtidas = px.bar(
        df_curtidas, x='PERIODO', y='CURTIDAS', color='MARCA_PAIS',
        barmode='group', title=f'Curtidas por {agrupamento}'
    )
    fig_curtidas.update_layout(
        xaxis_title='Período',
        yaxis_title='Quantidade',
        legend_title='Marca - País',
        hovermode='x unified',
        yaxis_range=[0, ymax_curt],                # <-- agora fixa no máximo global (~509k no seu caso)
        xaxis_range=['2012-01-01', df_curtidas['PERIODO'].max()]
    )
    fig_curtidas.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>Período: %{x}<br>Curtidas: %{y:,}'
    )
    st.plotly_chart(fig_curtidas, use_container_width=True)


    # --- Gráfico de Comentários ---
    df_comentarios = dados.groupby(['PERIODO', 'MARCA_PAIS'])['COMENTARIOS'].sum().reset_index()
    ymax_com = max_comentarios_global()

    fig_comentarios = px.bar(
        df_comentarios, x='PERIODO', y='COMENTARIOS', color='MARCA_PAIS',
        barmode='group', title=f'Comentários por {agrupamento}'
    )
    fig_comentarios.update_layout(
        xaxis_title='Período',
        yaxis_title='Quantidade',
        legend_title='Marca - País',
        hovermode='x unified',
        yaxis_range=[0, ymax_com],  # <-- fixo no máximo global
        xaxis_range=['2012-01-01', df_comentarios['PERIODO'].max()]
    )
    fig_comentarios.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>Período: %{x}<br>Comentários: %{y:,}'
    )
    st.plotly_chart(fig_comentarios, use_container_width=True)

    # --- Correlação Curtidas × Comentários ---
    st.subheader("Correlação Curtidas × Comentários")
    if 'CURTIDAS' in dados.columns and 'COMENTARIOS' in dados.columns:
        try:
            corr_val = dados[['CURTIDAS', 'COMENTARIOS']].corr().iloc[0,1]
        except Exception:
            corr_val = float('nan')
        st.write(f"Coeficiente de correlação de Pearson: **{corr_val:.2f}**" if corr_val == corr_val else "Não foi possível calcular a correlação.")

        fig_corr = px.scatter(
            dados, x="CURTIDAS", y="COMENTARIOS", color="MARCA_PAIS",
            hover_data=["DT_PUBLICACAO","LINK_PUBLICACAO","TEXTO"],
            title="Dispersão Curtidas × Comentários"
        )
        fig_corr.update_traces(marker=dict(size=10, line=dict(width=1, color='black')))
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Colunas 'CURTIDAS' e 'COMENTARIOS' não encontradas para calcular correlação.")
