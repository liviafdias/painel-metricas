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

# ------------------ MÁXIMOS PADRONIZADOS (mensal e anual) ------------------
def _serie_periodica(df, pais: str, period_code: str, val_col: str | None):
    """
    Retorna DataFrame com colunas [PERIODO, MARCA_PAIS, VAL] agregando por MARCA e período.
    period_code: 'M' (mensal) ou 'Y' (anual)
    val_col: None -> conta publicações; 'CURTIDAS' ou 'COMENTARIOS' -> soma valores
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["PERIODO", "MARCA_PAIS", "VAL"])
    t = df.copy()
    t["PAIS"] = pais
    if val_col:
        t[val_col] = pd.to_numeric(t[val_col], errors="coerce").fillna(0)
    t["PERIODO"] = t["DT_PUBLICACAO"].dt.to_period(period_code).dt.to_timestamp()
    if val_col:
        g = t.groupby(["PERIODO", "MARCA", "PAIS"])[val_col].sum().reset_index()
        g.rename(columns={val_col: "VAL"}, inplace=True)
    else:
        g = t.groupby(["PERIODO", "MARCA", "PAIS"]).size().reset_index(name="VAL")
    g["MARCA_PAIS"] = g["MARCA"] + " - " + g["PAIS"]
    return g[["PERIODO", "MARCA_PAIS", "VAL"]]

def _max_por_periodo(period_code: str):
    # Publicações
    pub_br = _serie_periodica(brasil, "BR", period_code, None)
    pub_se = _serie_periodica(suecia, "SE", period_code, None)
    pub_max = float(pd.concat([pub_br, pub_se], ignore_index=True)["VAL"].max() or 0)

    # Curtidas
    cur_br = _serie_periodica(brasil, "BR", period_code, "CURTIDAS")
    cur_se = _serie_periodica(suecia, "SE", period_code, "CURTIDAS")
    curt_max = float(pd.concat([cur_br, cur_se], ignore_index=True)["VAL"].max() or 0)

    # Comentários
    com_br = _serie_periodica(brasil, "BR", period_code, "COMENTARIOS")
    com_se = _serie_periodica(suecia, "SE", period_code, "COMENTARIOS")
    coment_max = float(pd.concat([com_br, com_se], ignore_index=True)["VAL"].max() or 0)

    return {"publicacoes": pub_max, "curtidas": curt_max, "comentarios": coment_max}

# Calcula UMA VEZ os limites padronizados globais
MAX_PADRONIZADO_M = _max_por_periodo("M")  # máximos por mês
MAX_PADRONIZADO_Y = _max_por_periodo("Y")  # máximos por ano

# ------------------ Sidebar (filtros) ------------------
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

    # PERIODO conforme o agrupamento escolhido
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

    # Seleciona limites padronizados conforme o agrupamento
    MAX_ATUAL = MAX_PADRONIZADO_Y if agrupamento == "ano" else MAX_PADRONIZADO_M

    # --- Gráfico de Publicações ---
    st.subheader("Evolução de Publicações, Curtidas e Comentários")
    df_pub = dados.groupby(['PERIODO', 'MARCA_PAIS']).size().reset_index(name='PUBLICAÇÕES')
    fig_pub = px.line(df_pub, x='PERIODO', y='PUBLICAÇÕES', color='MARCA_PAIS',
                      title=f'Publicações por {agrupamento}', markers=True)
    fig_pub.update_layout(
        xaxis_title='Período', yaxis_title='Quantidade',
        legend_title='Marca - País', hovermode='x unified',
        yaxis_range=[0, MAX_ATUAL["publicacoes"]],
        xaxis_range=['2012-01-01', df_pub['PERIODO'].max()]
    )
    fig_pub.update_traces(hovertemplate='<b>%{fullData.name}</b><br>Período: %{x}<br>Publicações: %{y:,}')
    st.plotly_chart(fig_pub, use_container_width=True)

    # --- Gráfico de Curtidas ---
    df_curtidas = dados.groupby(['PERIODO', 'MARCA_PAIS'])['CURTIDAS'].sum().reset_index()
    fig_curtidas = px.bar(df_curtidas, x='PERIODO', y='CURTIDAS', color='MARCA_PAIS',
                          barmode='group', title=f'Curtidas por {agrupamento}')
    fig_curtidas.update_layout(
        xaxis_title='Período', yaxis_title='Quantidade',
        legend_title='Marca - País', hovermode='x unified',
        yaxis_range=[0, MAX_ATUAL["curtidas"]],
        xaxis_range=['2012-01-01', df_curtidas['PERIODO'].max()]
    )
    fig_curtidas.update_traces(hovertemplate='<b>%{fullData.name}</b><br>Período: %{x}<br>Curtidas: %{y:,}')
    st.plotly_chart(fig_curtidas, use_container_width=True)

    # --- Gráfico de Comentários ---
    df_comentarios = dados.groupby(['PERIODO', 'MARCA_PAIS'])['COMENTARIOS'].sum().reset_index()
    fig_comentarios = px.bar(df_comentarios, x='PERIODO', y='COMENTARIOS', color='MARCA_PAIS',
                             barmode='group', title=f'Comentários por {agrupamento}')
    fig_comentarios.update_layout(
        xaxis_title='Período', yaxis_title='Quantidade',
        legend_title='Marca - País', hovermode='x unified',
        yaxis_range=[0, MAX_ATUAL["comentarios"]],
        xaxis_range=['2012-01-01', df_comentarios['PERIODO'].max()]
    )
    fig_comentarios.update_traces(hovertemplate='<b>%{fullData.name}</b><br>Período: %{x}<br>Comentários: %{y:,}')
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
