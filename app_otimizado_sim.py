# app_reclameaqui.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from wordcloud import WordCloud
from sklearn.cluster import KMeans

# --- Imports p/ grafo ---
import re
import itertools
from collections import Counter
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
# ------------------------

# === NOVO: imports p/ similaridade ===
from sklearn.metrics.pairwise import cosine_similarity
import ast


st.set_page_config(page_title="Painel de Tendências", layout="wide", initial_sidebar_state="expanded")

# (opcional) ícones
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
""", unsafe_allow_html=True)

# ------------------ Funções utilitárias ------------------
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
    df['X'] = pd.to_numeric(df['X'])
    df['Y'] = pd.to_numeric(df['Y'])
    df['LEN_TEXTO'] = df['TEXTO'].apply(len_texto)
    return df

@st.cache_data
def carregar_dados_suecia():
    df = pd.read_csv('TSNE_SE_COM_SENTIMENTO.csv', sep=';')
    df['DT_PUBLICACAO'] = df['DT_PUBLICACAO'].apply(ajeita_data)
    df.dropna(subset=['DT_PUBLICACAO'], inplace=True)
    df.sort_values('DT_PUBLICACAO', inplace=True)
    df.rename(columns={'N_CURTIDAS_NEW': 'CURTIDAS', 'N_COMENTARIOS_NEW': "COMENTARIOS"}, inplace=True)
    df['X'] = pd.to_numeric(df['X'])
    df['Y'] = pd.to_numeric(df['Y'])
    df['LEN_TEXTO'] = df['TEXTO'].apply(len_texto)
    return df

@st.cache_data
def carregar_stop_words_br():
    stop_df = pd.read_csv('docs/STOP_WORDS_PORTUGUES.csv', sep=';')
    return set(stop_df.STOP_WORDS.values)

@st.cache_data
def carregar_dados_reclame_aqui():
    # Lê o CSV considerando que não há cabeçalho
    try:
        df = pd.read_csv('docs/reclameaqui.csv', sep=',', header=None)  # ajuste sep se necessário
    except Exception as e:
        st.error(f"Erro ao carregar 'reclameaqui.csv': {e}")
        return pd.DataFrame()

    # Define nomes das colunas
    df.columns = [
        'MARCA', 'CATEGORIA',
        'ITEM1', 'PORCENTAGEM1', 'QUANTIDADE1',
        'ITEM2', 'QUANTIDADE2',
        'ITEM3', 'QUANTIDADE3',
        'ITEM4', 'QUANTIDADE4',
        'ITEM5', 'QUANTIDADE5'
    ]

    # Garantir que a coluna PORCENTAGEM1 é string
    df['PORCENTAGEM1'] = df['PORCENTAGEM1'].astype(str)

    return df

# --------- Funções do grafo (baseadas no seu grafo.py) ---------
def find_topics_column(df: pd.DataFrame, hint: str = "TOPICOS") -> str:
    cols = list(df.columns)
    cands = [c for c in cols if str(c).strip().lower() == hint.lower()]
    if cands: return cands[0]
    cands = [c for c in cols if "topico" in str(c).strip().lower()]
    if cands: return cands[0]
    cands = [c for c in cols if str(c).strip().lower() in {"tópicos", "tópico", "topics", "topic"}]
    if cands: return cands[0]
    raise ValueError(f'Não encontrei a coluna "{hint}". Colunas: {cols}')

def parse_topics(cell) -> list[str]:
    if pd.isna(cell): return []
    s = str(cell)
    s = re.sub(r"[|;/]+", ",", s)  # normaliza delimitadores
    parts = [p.strip() for p in s.split(",") if str(p).strip()]
    parts = [re.sub(r"\s+", " ", p).strip() for p in parts]
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p); out.append(p)
    return out

@st.cache_data
def build_cooccurrence(topic_lists: pd.Series):
    all_topics = [t for sub in topic_lists for t in sub]
    node_freq = Counter(all_topics)
    edge_weights = Counter()
    for topics in topic_lists:
        if len(topics) >= 2:
            for a, b in itertools.combinations(sorted(topics), 2):
                edge_weights[(a, b)] += 1
    nodes_df = (pd.DataFrame([{"node": n, "freq": int(f)} for n, f in node_freq.items()])
                .sort_values("freq", ascending=False).reset_index(drop=True))
    edges_df = (pd.DataFrame([{"ORIGEM": s, "DESTINO": t, "W": int(w)} for (s, t), w in edge_weights.items()])
                .sort_values("W", ascending=False).reset_index(drop=True))
    return nodes_df, edges_df


# === NOVO: helpers de embedding/similaridade ===
def _parse_embedding_cell(x):
    """Converte string como '[0.1, -0.2, ...]' em np.array.
       Retorna np.nan se não conseguir parsear."""
    try:
        arr = ast.literal_eval(str(x))
        return np.array(arr, dtype=float)
    except Exception:
        return np.nan

def _ensure_embeddings(df, col='EMBEDDING'):
    """Gera coluna EMBED_VEC a partir de EMBEDDING (se existir)."""
    if col not in df.columns:
        return df.assign(EMBED_VEC=np.nan)
    out = df.copy()
    out['EMBED_VEC'] = out[col].apply(_parse_embedding_cell)
    return out

def _brand_centroids(df, group_col='MARCA'):
    """Calcula o centróide de embedding por grupo (MARCA ou MARCA_PAIS)."""
    base = df.dropna(subset=['EMBED_VEC']).copy()
    if base.empty:
        return [], np.array([])
    grouped = (
        base.groupby(group_col)['EMBED_VEC']
            .apply(lambda xs: np.mean(np.stack(xs), axis=0))
    )
    labels = list(grouped.index)
    M = np.stack(grouped.values)
    return labels, M
def compute_layout(G: nx.Graph, seed=42):
    return nx.spring_layout(G, seed=seed)

def figure_size_from_pos(pos: dict, base_height=14, dpi=300):
    xs = [p[0] for p in pos.values()] or [0, 1]
    ys = [p[1] for p in pos.values()] or [0, 1]
    width = (max(xs) - min(xs)) or 1e-9
    height = (max(ys) - min(ys)) or 1e-9
    aspect = width / height
    fig_h = base_height
    fig_w = max(fig_h * aspect, fig_h * 0.8)
    return (fig_w, fig_h, dpi)
# ---------------------------------------------------------------

# ---------- Coocorrência a partir de TEXTO (NOVA) ----------
import string
def _normalize_text_keep_hashtags(s: str) -> str:
    # remove URLs e menções
    s = re.sub(r"http\S+|www\.\S+", " ", str(s))
    s = re.sub(r"@\w+", " ", s)
    # mantém #, remove demais pontuações
    keep = set("#_")
    s = "".join(ch if ch.isalnum() or ch in keep or ch.isspace() else " " for ch in s)
    s = re.sub(r"\s+", " ", s.lower()).strip()
    return s

def _tokenize_from_text(s: str, stop: set, min_len: int = 3, hashtags_only: bool = False):
    s = _normalize_text_keep_hashtags(s)
    toks = []
    for t in s.split():
        # preserva hashtags, mas compara stop sem '#'
        t_clean = t.lstrip("#")
        if hashtags_only and not t.startswith("#"):
            continue
        if len(t_clean) < min_len:
            continue
        if t_clean.isdigit():
            continue
        if t_clean in stop:
            continue
        toks.append(t_clean)  # guardamos sem o '#'
    # remove duplicatas por post preservando ordem
    return list(dict.fromkeys(toks))

@st.cache_data
def build_cooc_from_text(df_texto: pd.DataFrame,
                         col_texto: str = "TEXTO",
                         stop_words: set = None,
                         min_len: int = 6,
                         min_freq: int = 3,
                         min_w: int = 2,
                         top_nodes: int = 20,
                         hashtags_only: bool = False):
    """
    Retorna (nodes_df, edges_df, G, pos)
    - nodes_df: token, freq
    - edges_df: ORIGEM, DESTINO, W
    - G: grafo NetworkX
    - pos: layout (dict)
    """
    if stop_words is None:
        stop_words = set()

    textos = df_texto[col_texto].dropna().astype(str).tolist()
    tokens_por_post = [_tokenize_from_text(t, stop_words, min_len, hashtags_only) for t in textos]

    # frequência de nós
    all_tokens = list(itertools.chain.from_iterable(tokens_por_post))
    node_freq = Counter(all_tokens)
    # filtra nós por frequência mínima
    tokens_validos = {tok for tok, c in node_freq.items() if c >= min_freq}

    # coocorrência por post (pares no mesmo post)
    edge_weights = Counter()
    for toks in tokens_por_post:
        toks = [t for t in toks if t in tokens_validos]
        if len(toks) >= 2:
            for a, b in itertools.combinations(sorted(set(toks)), 2):
                edge_weights[(a, b)] += 1

    # arestas com peso mínimo
    edges = [(a, b, w) for (a, b), w in edge_weights.items() if w >= min_w]

    nodes_df = (pd.DataFrame([{"token": t, "freq": node_freq[t]} for t in tokens_validos])
                .sort_values("freq", ascending=False).reset_index(drop=True))
    edges_df = (pd.DataFrame(edges, columns=["ORIGEM", "DESTINO", "W"])
                .sort_values("W", ascending=False).reset_index(drop=True))

    # limita nós para visualização
    top_set = set(nodes_df.head(top_nodes)["token"].tolist())
    edges_df = edges_df[edges_df["ORIGEM"].isin(top_set) & edges_df["DESTINO"].isin(top_set)]

    # monta o grafo
    G = nx.from_pandas_edgelist(edges_df, "ORIGEM", "DESTINO", edge_attr="W", create_using=nx.Graph)
    for n in list(G.nodes()):
        G.nodes[n]["freq"] = int(node_freq.get(n, 1))

    pos = compute_layout(G, seed=42)
    return nodes_df, edges_df, G, pos
# -----------------------------------------------------------

# ------------------ Dados base e filtros ------------------
st.markdown('<h1 style="text-align:center;">Painel de Tendências</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:grey;">Explore tendências no Instagram e dados do Reclame Aqui</p>', unsafe_allow_html=True)

# Carregamentos
stop_words = carregar_stop_words_br()
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

# Sidebar (filtros do Instagram) — INICIAM VAZIOS
with st.sidebar:
    st.header("Filtros (Instagram)")

    # Constrói lista de marcas a partir do BR (se vazio, tenta SE)
    base_marcas = brasil if not brasil.empty else suecia
    todas_marcas = sorted(list(base_marcas["MARCA"].unique())) if "MARCA" in base_marcas.columns else []

    marcas_selecionadas = st.multiselect(
        "Comparar Marcas",
        options=todas_marcas,
        default=[],                     # inicia vazio
        placeholder="+ Adicionar marcas",
        key="sb_marcas"
    )

    # Países disponíveis a partir dos dados carregados
    paises_opts = []
    if not brasil.empty: paises_opts.append("BR")
    if not suecia.empty: paises_opts.append("SE")

    paises = st.multiselect(
        "País",
        options=paises_opts or ["BR", "SE"],
        default=[],                     # inicia vazio
        placeholder="Selecionar país(es)",
        key="sb_paises"
    )

    # Anos dinâmicos a partir dos dados disponíveis
    anos_br = brasil['DT_PUBLICACAO'].dt.year.unique().tolist() if 'DT_PUBLICACAO' in brasil.columns else []
    anos_se = suecia['DT_PUBLICACAO'].dt.year.unique().tolist() if 'DT_PUBLICACAO' in suecia.columns else []
    anos_disponiveis = sorted(set(anos_br) | set(anos_se)) or [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

    periodo_selecionado = st.multiselect(
        "Período (anos)",
        options=anos_disponiveis,
        default=[],                     # inicia vazio
        placeholder="Selecionar ano(s)",
        key="sb_anos"
    )

    agrupamento = st.radio(
        "Agrupar por",
        options=["mês", "ano"],
        index=0,
        horizontal=True,
        key="sb_agrupar"
    )

# Constrói dataframe filtrado do Instagram (sem st.stop)
def filtra_instagram():
    df_list = []
    if "BR" in paises and not brasil.empty:
        df_br = brasil.copy(); df_br["PAIS"] = "BR"; df_list.append(df_br)
    if "SE" in paises and not suecia.empty:
        df_se = suecia.copy(); df_se["PAIS"] = "SE"; df_list.append(df_se)
    if not df_list:
        return pd.DataFrame()
    df = pd.concat(df_list, ignore_index=True)

    # --- Garantir tipos numéricos para evitar "str / str" ---
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

    # --- Cálculo da relevância separada (com proteção a divisão por zero) ---
    import numpy as np
    if "CURTIDAS" in df.columns:
        soma_curtidas = df.groupby("MARCA_PAIS")["CURTIDAS"].transform("sum")
        df["RELEVANCIA_CURTIDAS_%"] = np.where(
            soma_curtidas > 0,
            ((df["CURTIDAS"] / soma_curtidas) * 100).round(0).astype("Int64").astype(str) + "%",
            "0%"
        )
    if "COMENTARIOS" in df.columns:
        soma_coment = df.groupby("MARCA_PAIS")["COMENTARIOS"].transform("sum")
        df["RELEVANCIA_COMENTARIOS_%"] = np.where(
            soma_coment > 0,
            ((df["COMENTARIOS"] / soma_coment) * 100).round(0).astype("Int64").astype(str) + "%",
            "0%"
        )

    return df

dados = filtra_instagram()

# ------------------ Abas principais ------------------
tab_instagram, tab_reclameaqui = st.tabs(["Instagram", "Reclame Aqui"])

# ===========================
# TAB: Instagram
# ===========================
with tab_instagram:
    ig_tab1, ig_tab2, ig_tab3, ig_tab4, ig_tab5, ig_tab6, ig_tab7 = st.tabs([
        "Métricas ao Longo do Tempo",
        "Análise de Sentimentos",
        "Visualização Bidimensional (T-SNE)",
        "Nuvem de Palavras",
        "Clusterização KMeans",
        "Grafo Coocorrência",
        "Similaridade Semântica"
    ])

    

    # --- Métricas ao longo do tempo ---
    with ig_tab1:
        st.subheader("Evolução de Publicações, Curtidas e Comentários")
        if dados.empty:
            st.info("Ajuste os filtros na barra lateral para ver as métricas do Instagram.")
        else:
            df_pub = dados.groupby(['PERIODO', 'MARCA_PAIS']).size().reset_index(name='PUBLICAÇÕES')
            fig_pub = px.line(df_pub, x='PERIODO', y='PUBLICAÇÕES', color='MARCA_PAIS',
                              title=f'Publicações por {agrupamento}', markers=True)
            fig_pub.update_layout(xaxis_title='Período', yaxis_title='Quantidade',
                              legend_title='Marca - País', hovermode='x unified',
                              yaxis_range=[0, 250],
                              xaxis_range=['2012-01-01', df_pub['PERIODO'].max()])
            fig_pub.update_traces(hovertemplate='<b>%{fullData.name}</b><br>Período: %{x}<br>Publicações: %{y:,}')
            st.plotly_chart(fig_pub, use_container_width=True)

            df_curtidas = dados.groupby(['PERIODO', 'MARCA_PAIS'])['CURTIDAS'].sum().reset_index()
            fig_curtidas = px.bar(df_curtidas, x='PERIODO', y='CURTIDAS', color='MARCA_PAIS',
                                  barmode='group', title=f'Curtidas por {agrupamento}')
            fig_curtidas.update_layout(xaxis_title='Período', yaxis_title='Quantidade',
                                       legend_title='Marca - País', hovermode='x unified',
                                       yaxis_range=[0, 150000],
                                       xaxis_range=['2012-01-01', df_curtidas['PERIODO'].max()])
            fig_curtidas.update_traces(hovertemplate='<b>%{fullData.name}</b><br>Período: %{x}<br>Curtidas: %{y:,}')
            st.plotly_chart(fig_curtidas, use_container_width=True)

            df_comentarios = dados.groupby(['PERIODO', 'MARCA_PAIS'])['COMENTARIOS'].sum().reset_index()
            fig_comentarios = px.bar(df_comentarios, x='PERIODO', y='COMENTARIOS', color='MARCA_PAIS',
                                     barmode='group', title=f'Comentários por {agrupamento}')
            fig_comentarios.update_layout(xaxis_title='Período', yaxis_title='Quantidade',
                                          legend_title='Marca - País', hovermode='x unified',
                                          yaxis_range=[0, 10000],
                                          xaxis_range=['2012-01-01', df_comentarios['PERIODO'].max()])
            fig_comentarios.update_traces(hovertemplate='<b>%{fullData.name}</b><br>Período: %{x}<br>Comentários: %{y:,}')
            st.plotly_chart(fig_comentarios, use_container_width=True)

            # Cartas rápidas
            colm1, colm2, colm3 = st.columns(3)
            total_publicacoes = len(dados)
            total_curtidas = int(dados['CURTIDAS'].sum())
            total_comentarios = int(dados['COMENTARIOS'].sum())
            colm1.metric("Total de Publicações", f"{total_publicacoes:,}".replace(",", "."))
            colm2.metric("Total de Curtidas", f"{total_curtidas:,}".replace(",", "."))
            colm3.metric("Total de Comentários", f"{total_comentarios:,}".replace(",", "."))

            st.subheader("Top 5 Publicações mais relevantes em Curtidas (%)")
            top_curtidas = dados.sort_values("RELEVANCIA_CURTIDAS_%", ascending=False).head(5)
            st.dataframe(
                top_curtidas[["MARCA_PAIS", "DT_PUBLICACAO", "CURTIDAS", "RELEVANCIA_CURTIDAS_%", "LINK_PUBLICACAO", "TEXTO"]]
            )

            st.subheader("Top 5 Publicações mais relevantes em Comentários (%)")
            top_comentarios = dados.sort_values("RELEVANCIA_COMENTARIOS_%", ascending=False).head(5)
            st.dataframe(
                top_comentarios[["MARCA_PAIS", "DT_PUBLICACAO", "COMENTARIOS", "RELEVANCIA_COMENTARIOS_%", "LINK_PUBLICACAO", "TEXTO"]]
            )

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

            # --- Engajamento Absoluto ---
            st.subheader("Engajamento Absoluto (Curtidas + Comentários por Seguidores)")
            try:
                # Tenta ler com ';' e depois com ','
                try:
                    seguidores_df = pd.read_csv("qtd_seguidores.csv", sep=",")
                except Exception:
                    seguidores_df = pd.read_csv("qtd_seguidores.csv", sep=",")

                seguidores_df.columns = [str(c).strip().upper() for c in seguidores_df.columns]
                col_marca = next((c for c in seguidores_df.columns if c in {"MARCA","BRAND"}), None)
                col_pais  = next((c for c in seguidores_df.columns if c in {"PAIS","PAÍS","COUNTRY"}), None)
                col_segs  = next((c for c in seguidores_df.columns if "SEGUIDORES" in c or "FOLLOW" in c), None)
                if not (col_marca and col_pais and col_segs):
                    raise ValueError("O arquivo 'qtd_seguidores.csv' deve conter colunas de MARCA, PAIS e SEGUIDORES.")

                base_eng = dados.copy()
                base_eng["MARCA"] = base_eng["MARCA"].astype(str).str.upper()
                base_eng["PAIS"]  = base_eng["PAIS"].astype(str).str.upper()

                seguidores_df[col_marca] = seguidores_df[col_marca].astype(str).str.upper()
                seguidores_df[col_pais]  = seguidores_df[col_pais].astype(str).str.upper()

                eng = base_eng.merge(seguidores_df[[col_marca, col_pais, col_segs]], left_on=["MARCA","PAIS"], right_on=[col_marca, col_pais], how="left")
                eng = eng.rename(columns={col_segs: "SEGUIDORES"})
                eng["SEGUIDORES"] = pd.to_numeric(eng["SEGUIDORES"], errors="coerce")
                eng = eng[eng["SEGUIDORES"].notna() & (eng["SEGUIDORES"] > 0)]
                if eng.empty:
                    st.warning("Não há correspondência de seguidores para as marcas/país filtrados.")
                else:
                    eng["ENGAJAMENTO_ABS"] = (eng["CURTIDAS"] + eng["COMENTARIOS"]) / eng["SEGUIDORES"]

                    fig_eng = px.box(
                        eng, x="MARCA_PAIS", y="ENGAJAMENTO_ABS", color="MARCA_PAIS",
                        points="all", title="Distribuição do Engajamento Absoluto"
                    )
                    fig_eng.update_layout(yaxis_title="Engajamento (likes+comentários)/seguidores")
                    st.plotly_chart(fig_eng, use_container_width=True)

                    resumo = eng.groupby("MARCA_PAIS")["ENGAJAMENTO_ABS"].describe()[["mean","50%","max"]].rename(columns={"50%":"median"})
                    st.dataframe(resumo)
            except FileNotFoundError:
                st.info("Adicione o arquivo 'qtd_seguidores.csv' na raiz do app para calcular o engajamento absoluto.")
            except Exception as e:
                st.warning(f"Não foi possível calcular engajamento absoluto: {e}")

            # --- Tipo de postagem (Foto vs Reel) ---
            st.subheader("Contagem de postagens por tipo")
            if not dados.empty and 'LINK_PUBLICACAO' in dados.columns:
                links = dados['LINK_PUBLICACAO'].astype(str)
                dados['TIPO_POST'] = np.select(
                    [
                        links.str.contains('/reel/', na=False, case=False),
                        links.str.contains('/p/', na=False, case=False),
                    ],
                    ['Reel', 'Foto'],
                    default='Outro'
                )

                contagem_tipos = (dados['TIPO_POST']
                                .value_counts(dropna=False)
                                .rename_axis('Tipo de Post')
                                .reset_index(name='Quantidade'))
                st.dataframe(contagem_tipos)

                st.markdown("**Por Marca-País**")
                contagem_por_marca = (dados.groupby(['MARCA_PAIS', 'TIPO_POST'])
                                            .size()
                                            .reset_index(name='Quantidade'))
                fig_tipo_marca = px.bar(
                    contagem_por_marca, x='MARCA_PAIS', y='Quantidade',
                    color='TIPO_POST', barmode='group',
                    title="Tipos por Marca-País"
                )
                fig_tipo_marca.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig_tipo_marca, use_container_width=True)

                st.markdown(f"**Por {agrupamento.capitalize()}**")
                contagem_por_periodo = (dados.groupby(['PERIODO', 'TIPO_POST'])
                                            .size()
                                            .reset_index(name='Quantidade'))
                fig_tipo_tempo = px.line(
                    contagem_por_periodo, x='PERIODO', y='Quantidade',
                    color='TIPO_POST', markers=True,
                    title=f"Evolução de Tipos por {agrupamento}"
                )
                fig_tipo_tempo.update_layout(hovermode='x unified')
                st.plotly_chart(fig_tipo_tempo, use_container_width=True)
            else:
                st.info("Coluna 'LINK_PUBLICACAO' não encontrada ou não há dados após os filtros.")

    # --- Análise de sentimentos ---
    with ig_tab2:
        st.subheader("Distribuição e Evolução dos Sentimentos")
        if dados.empty:
            st.info("Ajuste os filtros para visualizar os sentimentos.")
        elif 'SENTIMENTO' not in dados.columns:
            st.warning("Coluna 'SENTIMENTO' não encontrada nos dados.")
        else:
            dist_sent = dados['SENTIMENTO'].value_counts().reset_index()
            dist_sent.columns = ['Sentimento', 'Quantidade']
            fig1 = px.bar(dist_sent, x='Sentimento', y='Quantidade', color='Sentimento',
                          title="Frequência dos sentimentos")
            st.plotly_chart(fig1, use_container_width=True)

            dist_sent['Percentual'] = (dist_sent['Quantidade'] / dist_sent['Quantidade'].sum()) * 100
            dist_sent['Percentual'] = dist_sent['Percentual'].round(2)
            fig_pizza = px.pie(dist_sent, names='Sentimento', values='Percentual',
                               title='Percentual de Publicações por Sentimento', hole=0.4)
            fig_pizza.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_pizza, use_container_width=True)

            st.subheader("Distribuição de Sentimentos por Marca")
            df_marca_sent = dados.groupby(['MARCA_PAIS', 'SENTIMENTO']).size().reset_index(name='Quantidade')
            fig2 = px.bar(df_marca_sent, x='MARCA_PAIS', y='Quantidade', color='SENTIMENTO',
                          barmode='group', title="Distribuição de sentimentos por Marca e País")
            fig2.update_layout(xaxis_tickangle=-45, hovermode='x unified')
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Evolução temporal por Sentimento")
            sentimentos_disponiveis = sorted(dados['SENTIMENTO'].dropna().unique().tolist())
            sentimentos_selecionados = st.multiselect(
                "Selecione sentimentos:", options=sentimentos_disponiveis,
                default=sentimentos_disponiveis[:2], key="ig_sent_sel")
            paises_disponiveis = sorted(dados['PAIS'].dropna().unique().tolist())
            paises_selecionados = st.multiselect(
                "Selecione países:", options=paises_disponiveis,
                default=paises_disponiveis, key="ig_paises_sel"
            )
            dados['MES_ANO'] = dados['DT_PUBLICACAO'].dt.to_period('M').dt.to_timestamp()
            evolucao = dados[(dados['SENTIMENTO'].isin(sentimentos_selecionados)) &
                             (dados['PAIS'].isin(paises_selecionados))]
            evolucao = evolucao.groupby(['MES_ANO', 'MARCA_PAIS', 'SENTIMENTO']).size().reset_index(name='COUNT')
            if evolucao.empty:
                st.info("Sem dados para os filtros atuais.")
            else:
                fig3 = px.line(evolucao, x='MES_ANO', y='COUNT', color='MARCA_PAIS',
                               line_dash='SENTIMENTO', markers=True,
                               title='Evolução temporal dos sentimentos por Marca e País')
                fig3.update_layout(hovermode='x unified')
                st.plotly_chart(fig3, use_container_width=True)

    # --- T-SNE ---
    with ig_tab3:
        st.subheader("Visualização Bidimensional (T-SNE)")
        if dados.empty:
            st.info("Ajuste os filtros para visualizar o T-SNE.")
        else:
            if len(marcas_selecionadas) > 1:
                marca_visualizada = st.selectbox("Escolha uma marca:", marcas_selecionadas, key="ig_tsne_marca")
                dados_tsne = dados[dados["MARCA"] == marca_visualizada]
            else:
                dados_tsne = dados.copy()
            fig = px.scatter(
                dados_tsne, x="X", y="Y", color="MARCA",
                hover_data=["CURTIDAS", "COMENTARIOS", "DT_PUBLICACAO", "TEXTO"],
                title=f"Visualização T-SNE - {marca_visualizada if len(marcas_selecionadas)>1 else 'marcas selecionadas'}"
            )
            fig.update_traces(marker=dict(line=dict(width=1, color='black'), size=10))
            st.plotly_chart(fig, use_container_width=True)

    # --- Nuvem de palavras ---
    with ig_tab4:
        st.subheader("Nuvem de Palavras")
        if dados.empty:
            st.info("Ajuste os filtros para gerar a nuvem.")
        else:
            if len(marcas_selecionadas) > 1:
                marca_visualizada = st.selectbox("Escolha uma marca:", marcas_selecionadas, key="ig_wc_marca")
                dados_wc = dados[dados["MARCA"] == marca_visualizada]
            else:
                dados_wc = dados.copy()
            textos = ' '.join(dados_wc['TEXTO'].dropna().astype(str).str.lower().tolist())
            if textos.strip():
                wc = WordCloud(max_words=200, colormap='viridis',
                               stopwords=stop_words, width=900, height=420,
                               background_color='white').generate(textos)
                st.image(wc.to_array(), caption="Nuvem de Palavras")
            else:
                st.info("Não há texto disponível com os filtros atuais.")

    # --- Clusterização KMeans ---
    with ig_tab5:
        st.subheader("Clusterização KMeans")
        if dados.empty or dados[['X','Y']].isnull().values.any():
            st.info("Dados insuficientes ou inválidos para clusterização.")
        else:
            if len(marcas_selecionadas) > 1:
                marca_visualizada = st.selectbox("Escolha uma marca:", marcas_selecionadas, key="ig_km_marca")
                dados_cluster = dados[dados["MARCA"] == marca_visualizada].copy()
            else:
                dados_cluster = dados.copy()

            ncluster = st.number_input("Número de clusters (k)", value=6, min_value=1, max_value=30, step=1, key="ig_km_k")
            try:
                matriz = dados_cluster[['X','Y']].values
                kmeans = KMeans(n_clusters=ncluster, n_init=30, random_state=42).fit(matriz)
                dados_cluster['CLUSTER'] = kmeans.labels_.astype(str)

                fig_cluster_2d = px.scatter(
                    dados_cluster, x="X", y="Y", color="CLUSTER",
                    hover_data=["CURTIDAS","COMENTARIOS","DT_PUBLICACAO","MARCA","TEXTO"],
                    title=f"Clusters (k={ncluster})"
                )
                fig_cluster_2d.update_traces(marker=dict(line=dict(width=1, color='black'), size=10))
                st.plotly_chart(fig_cluster_2d, use_container_width=True)

                st.subheader("Distribuição por Cluster")
                if 'LINK_PUBLICACAO' in dados_cluster.columns:
                    st.dataframe(dados_cluster.groupby('CLUSTER')
                                 .agg({'MARCA':'nunique','LINK_PUBLICACAO':'nunique'})
                                 .rename(columns={'MARCA':'Marcas Distintas','LINK_PUBLICACAO':'Total de Publicações'}))
                else:
                    st.dataframe(dados_cluster.groupby('CLUSTER')
                                 .agg({'MARCA':'nunique','X':'count'})
                                 .rename(columns={'MARCA':'Marcas Distintas','X':'Publicações (contagem)'}))

                st.subheader("Nuvem de Palavras por Cluster")
                lista_cluster = sorted(dados_cluster['CLUSTER'].unique())
                cluster_escolhido = st.selectbox("Selecione um cluster:", lista_cluster, key="ig_km_cluster_sel")
                textos_cluster = ' '.join(dados_cluster[dados_cluster.CLUSTER == cluster_escolhido]['TEXTO'].dropna().astype(str).str.lower())
                if textos_cluster.strip():
                    wc_cluster = WordCloud(max_words=200, colormap='viridis',
                                           stopwords=stop_words, width=900, height=420,
                                           background_color='white').generate(textos_cluster)
                    st.image(wc_cluster.to_array(), caption=f"Nuvem - Cluster {cluster_escolhido}")
                else:
                    st.info("Sem texto para a nuvem deste cluster.")
            except Exception as e:
                st.error(f"Erro ao clusterizar: {e}")

    # --- Grafo de coocorrência a partir de TEXTO (NOVA ABA) ---
    with ig_tab6:
        st.subheader("Rede de Coocorrência (base: TEXTO)")

        if dados.empty or "TEXTO" not in dados.columns:
            st.info("Ajuste os filtros e garanta que a coluna 'TEXTO' está disponível.")
        else:
            # Parâmetros interativos
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                min_len = st.number_input("Tamanho mín. token", 2, 10, 3, 1)
            with c2:
                min_freq = st.number_input("Frequência mín. nó", 1, 50, 3, 1)
            with c3:
                min_w = st.number_input("Peso mín. aresta", 1, 50, 2, 1)
            with c4:
                top_nodes = st.number_input("Máx. nós no grafo", 10, 500, 80, 10)
            with c5:
                hashtags_only = st.checkbox("Somente hashtags", value=False)

            # Stopwords combinando seu arquivo + termos triviais do domínio
            brand_words = {
                "byd","bydauto","bydautobrasil","brasil","oficial","eletrico","elétrico","eletricos",
                "elétricos","ev","plugin","plug","hibrido","híbrido","carro","carros","drive","electric",
                "friday","black","week","promo","oferta","ofertas"
            }
            stop_text = set(stop_words) | brand_words

            try:
                nodes_df, edges_df, G, posx = build_cooc_from_text(
                    df_texto=dados,
                    col_texto="TEXTO",
                    stop_words=stop_text,
                    min_len=int(min_len),
                    min_freq=int(min_freq),
                    min_w=int(min_w),
                    top_nodes=int(top_nodes),
                    hashtags_only=bool(hashtags_only),
                )

                if edges_df.empty or G.number_of_edges() == 0:
                    st.warning("Sem coocorrências suficientes com os parâmetros atuais. Tente relaxar os limiares.")
                else:
                    st.markdown("**Top tokens (por frequência)**")
                    st.dataframe(nodes_df.head(50).rename(columns={"token":"TOKEN","freq":"FREQ"}))

                    st.markdown("**Top arestas (por peso de coocorrência)**")
                    st.dataframe(edges_df.head(100))

                    # Figura
                    fig_w, fig_h, dpi = figure_size_from_pos(posx, base_height=12, dpi=240)
                    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
                    ax = plt.gca()
                    ax.set_axis_off()

                    # tamanhos de nós ~ sqrt(freq)
                    node_sizes = []
                    for n in G.nodes():
                        f = G.nodes[n].get("freq", 1)
                        node_sizes.append(100 + 20 * np.sqrt(f))

                    # espessura da aresta ~ log(1+W)
                    edge_widths = [1 + np.log1p(G[u][v]["W"]) for u, v in G.edges()]

                    nx.draw_networkx_edges(G, posx, width=edge_widths, alpha=0.5, ax=ax)
                    nx.draw_networkx_nodes(G, posx, node_size=node_sizes, ax=ax)
                    # rotular apenas os principais
                    top_label_set = set(nodes_df.sort_values("freq", ascending=False).head(20)["token"])
                    labels = {n: n for n in G.nodes() if n in top_label_set or G.number_of_nodes() <= 25}
                    nx.draw_networkx_labels(G, posx, labels=labels, font_size=8, ax=ax)

                    st.pyplot(fig, clear_figure=True)

                    # # Downloads
                    # col_a, col_b = st.columns(2)
                    # with col_a:
                    #     st.download_button(
                    #         "⬇️ Baixar nós (CSV)",
                    #         data=nodes_df.to_csv(index=False).encode("utf-8"),
                    #         file_name="cooc_text_nodes.csv",
                    #         mime="text/csv"
                    #     )
                    # with col_b:
                    #     st.download_button(
                    #         "⬇️ Baixar arestas (CSV)",
                    #         data=edges_df.to_csv(index=False).encode("utf-8"),
                    #         file_name="cooc_text_edges.csv",
                    #         mime="text/csv"
                    #     )
            except Exception as e:
                st.error(f"Erro ao gerar grafo de coocorrência: {e}")

# ===========================
# TAB: Reclame Aqui
# ===========================

# === NOVO: Aba de Similaridade Semântica ===
with ig_tab7:
    st.subheader("Similaridade Semântica entre Montadoras (via embeddings)")

    if dados.empty:
        st.info("Ajuste os filtros da barra lateral para carregar dados.")
    else:
        dados_emb = _ensure_embeddings(dados, col='EMBEDDING')

        if dados_emb['EMBED_VEC'].isna().all():
            st.warning("A coluna 'EMBEDDING' não está disponível nos dados atuais ou não pôde ser interpretada.")
            st.caption("Dica: mescle seus CSVs que já possuem a coluna EMBEDDING nas bases BR/SE usadas pelo app.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                nivel = st.radio(
                    "Nível de comparação dos centróides",
                    options=["MARCA", "MARCA_PAIS"],
                    index=0,
                    horizontal=True
                )
            with c2:
                mostrar_links = st.checkbox("Exibir links e textos nos pares mais similares", value=True)

            labels, M = _brand_centroids(dados_emb, group_col=nivel)
            if len(labels) < 2:
                st.info("É necessário ter pelo menos duas entidades no nível selecionado para comparar.")
            else:
                from plotly import express as px  # já importado acima, mas garante escopo
                S = cosine_similarity(M)
                sim_df = pd.DataFrame(S, index=labels, columns=labels)

                st.markdown("**Matriz de similaridade (cosseno) entre centróides de embeddings**")
                fig_sim = px.imshow(
                    sim_df,
                    text_auto=".2f",
                    aspect="auto",
                    title=f"Similaridade (coseno) por {nivel}",
                    color_continuous_scale="Blues",
                    zmin=0, zmax=1
                )
                fig_sim.update_layout(margin=dict(l=40, r=10, t=60, b=40))
                st.plotly_chart(fig_sim, use_container_width=True)

                # st.download_button(
                #     "⬇️ Baixar matriz de similaridade (CSV)",
                #     data=sim_df.to_csv().encode("utf-8"),
                #     file_name=f"similaridade_{nivel.lower()}.csv",
                #     mime="text/csv"
                # )

                st.divider()
                st.subheader("Pares de posts mais parecidos entre duas entidades")

                col_a, col_b, col_top = st.columns([1, 1, 1])
                with col_a:
                    a_escolha = st.selectbox("Entidade A", options=labels, index=0)
                with col_b:
                    b_escolha = st.selectbox("Entidade B", options=[x for x in labels if x != a_escolha], index=0)
                with col_top:
                    top_n = st.number_input("Top-N pares", min_value=1, max_value=50, value=5, step=1)

                sub_a = dados_emb[(dados_emb[nivel] == a_escolha) & (dados_emb['EMBED_VEC'].notna())]
                sub_b = dados_emb[(dados_emb[nivel] == b_escolha) & (dados_emb['EMBED_VEC'].notna())]

                if sub_a.empty or sub_b.empty:
                    st.info("Faltam embeddings para uma das entidades escolhidas.")
                else:
                    A = np.stack(sub_a['EMBED_VEC'].values)
                    B = np.stack(sub_b['EMBED_VEC'].values)
                    sims = cosine_similarity(A, B)

                    flat_idx = sims.flatten().argsort()[::-1][:int(top_n)]
                    rows = []
                    for k in flat_idx:
                        i, j = divmod(k, sims.shape[1])
                        linha = {
                            f'{nivel} A': a_escolha,
                            f'{nivel} B': b_escolha,
                            'SIM': float(sims[i, j]),
                            'DT_A': sub_a.iloc[i].get('DT_PUBLICACAO', None),
                            'DT_B': sub_b.iloc[j].get('DT_PUBLICACAO', None),
                            'LINK_A': sub_a.iloc[i].get('LINK_PUBLICACAO', None),
                            'LINK_B': sub_b.iloc[j].get('LINK_PUBLICACAO', None),
                            'TEXTO_A': sub_a.iloc[i].get('TEXTO', None),
                            'TEXTO_B': sub_b.iloc[j].get('TEXTO', None),
                            'CURTIDAS_A': sub_a.iloc[i].get('CURTIDAS', None),
                            'CURTIDAS_B': sub_b.iloc[j].get('CURTIDAS', None),
                            'COMENT_A': sub_a.iloc[i].get('COMENTARIOS', None),
                            'COMENT_B': sub_b.iloc[j].get('COMENTARIOS', None),
                        }
                        rows.append(linha)

                    df_pairs = pd.DataFrame(rows).sort_values('SIM', ascending=False)

                    cols_show = [f'{nivel} A', f'{nivel} B', 'SIM', 'DT_A', 'DT_B', 'CURTIDAS_A', 'CURTIDAS_B', 'COMENT_A', 'COMENT_B']
                    if mostrar_links:
                        cols_show += ['LINK_A', 'LINK_B', 'TEXTO_A', 'TEXTO_B']

                    st.dataframe(df_pairs[cols_show], use_container_width=True)

                    # st.download_button(
                    #     "⬇️ Baixar pares similares (CSV)",
                    #     data=df_pairs.to_csv(index=False).encode("utf-8"),
                    #     file_name=f"pares_similares_{a_escolha}_vs_{b_escolha}.csv",
                    #     mime="text/csv"
                    # )
with tab_reclameaqui:
    rq_tab1, rq_tab2 = st.tabs([
        "Análise de Reclamações",
        "Grafo de Coocorrência (PEUGEOT/TOYOTA)"
    ])

    # --- Análise de Reclamações ---
    with rq_tab1:
        st.subheader("Análise de Reclamações por Marca")
        try:
            df_reclame = carregar_dados_reclame_aqui()
        except Exception as e:
            st.error(f"Erro ao carregar 'data/reclameaqui.xlsx': {e}")
            df_reclame = pd.DataFrame()

        if df_reclame.empty:
            st.info("Adicione o arquivo 'data/reclameaqui.xlsx' para visualizar esta aba.")
        else:
            marcas_reclame = sorted(df_reclame['MARCA'].dropna().unique().tolist())
            marca_selecionada = st.selectbox("Selecione a marca:", options=marcas_reclame, index=0, key="rq_marca")

            dados_marca = df_reclame[df_reclame['MARCA'] == marca_selecionada]

            def criar_card(categoria, titulo):
                dados_categoria = dados_marca[dados_marca['CATEGORIA'].astype(str).str.strip().str.lower() == categoria.lower().strip()]
                if len(dados_categoria) == 0:
                    return f"""
                    <div style="background-color:#f8f9fa;padding:15px;border-radius:10px;margin-bottom:15px;">
                        <h4 style="color:#333;margin-top:0;">{titulo}</h4>
                        <p style="color:#666;">Dados não disponíveis para esta marca</p>
                    </div>
                    """
                row = dados_categoria.iloc[0]
                itens = []
                for i in range(1, 6):
                    item_col = f'ITEM{i}'; qtd_col = f'QUANTIDADE{i}'
                    item = str(row.get(item_col, "")).strip()
                    quantidade = str(row.get(qtd_col, "")).strip()
                    if item and item.lower() != 'nan' and quantidade and quantidade.lower() != 'nan':
                        itens.append(f"<p style='margin-bottom:5px;'>{item} ({quantidade})</p>")
                porcentagem = str(row.get('PORCENTAGEM1', 'N/A')).strip()
                return f"""
                <div style="background-color:#f8f9fa;padding:15px;border-radius:10px;margin-bottom:15px;">
                    <h4 style="color:#333;margin-top:0;'>{titulo}</h4>
                    <p style="font-size:24px;font-weight:bold;color:#2c3e50;margin-bottom:5px;">{porcentagem}</p>
                    {''.join(itens)}
                </div>
                """

            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(criar_card('Tipos de problemas', 'Tipos de problemas'), unsafe_allow_html=True)
            with c2: st.markdown(criar_card('Produtos e Serviços', 'Produtos e Serviços'), unsafe_allow_html=True)
            with c3: st.markdown(criar_card('Categorias', 'Categorias'), unsafe_allow_html=True)

            st.subheader("Visualização Gráfica")
            def criar_grafico(categoria, titulo):
                dados_categoria = dados_marca[dados_marca['CATEGORIA'] == categoria]
                if len(dados_categoria) == 0: return None
                row = dados_categoria.iloc[0]
                itens, quantidades = [], []
                for i in range(1, 5+1):
                    item = row.get(f'ITEM{i}')
                    quantidade = row.get(f'QUANTIDADE{i}')
                    if pd.notna(item) and pd.notna(quantidade):
                        itens.append(item); quantidades.append(quantidade)
                if not itens: return None
                df = pd.DataFrame({'Item': itens, 'Quantidade': quantidades})
                fig = px.bar(df, x='Item', y='Quantidade', title=f'{titulo} - {marca_selecionada}', color='Item')
                fig.update_layout(showlegend=False)
                return fig

            g1, g2, g3 = st.columns(3)
            with g1:
                fig1 = criar_grafico('Tipos de problemas', 'Tipos de Problemas')
                if fig1: st.plotly_chart(fig1, use_container_width=True)
            with g2:
                fig2 = criar_grafico('Produtos e Serviços', 'Produtos e Serviços')
                if fig2: st.plotly_chart(fig2, use_container_width=True)
            with g3:
                fig3 = criar_grafico('Categorias', 'Categorias')
                if fig3: st.plotly_chart(fig3, use_container_width=True)

    # --- Grafo PEUGEOT/TOYOTA ---
    with rq_tab2:
        st.subheader("Coocorrência de tópicos")
        marca_escolhida = st.selectbox(
            "Selecione a base de dados do grafo:",
            options=["PEUGEOT", "TOYOTA"],
            index=0,
            key="rq_grafo_base"
        )
        arquivo_map = {"PEUGEOT": "docs/peugeot.csv", "TOYOTA": "docs/toyota.csv"}
        arquivo_escolhido = arquivo_map[marca_escolhida]
        candidatos = [Path(arquivo_escolhido), Path("data") / arquivo_escolhido]
        xlsx_path = next((p for p in candidatos if p.exists()), None)

        if xlsx_path is None:
            st.error(f"Arquivo '{arquivo_escolhido}' não encontrado na raiz nem em 'data/'.")
            st.info("Coloque o arquivo .xlsx no mesmo diretório do app ou dentro da pasta 'data/'.")
        else:
            try:
                df_topics = pd.read_csv(xlsx_path)
                col = find_topics_column(df_topics, "TOPICOS")
                df_topics = df_topics.rename(columns={col: "TOPICOS"})
                topic_lists = df_topics["TOPICOS"].apply(parse_topics)

                nodes_df, edges_df = build_cooccurrence(topic_lists)
                if edges_df.empty or nodes_df.empty:
                    st.warning("Não foram encontradas coocorrências suficientes para montar o grafo.")
                else:
                    G = nx.from_pandas_edgelist(edges_df, "ORIGEM", "DESTINO", edge_attr="W", create_using=nx.Graph)
                    posx = compute_layout(G, seed=42)

                    LABEL_SOME_NODES = True
                    TOP_N_BY_FREQ = 20
                    LABEL_ALL_IF_SMALLER = 25
                    if (LABEL_SOME_NODES and G.number_of_nodes() > LABEL_ALL_IF_SMALLER):
                        listap = set(nodes_df.sort_values("freq", ascending=False).head(TOP_N_BY_FREQ)["node"])
                    else:
                        listap = set(G.nodes())

                    fig_w, fig_h, dpi = figure_size_from_pos(posx, base_height=14, dpi=300)
                    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
                    ax = plt.gca()
                    ax.set_xlabel(" "); ax.set_ylabel(" ")
                    ax.axis("off"); ax.set_aspect("equal")

                    dgrapg = edges_df.sort_values("W", ascending=False).reset_index(drop=True)
                    z = dgrapg["W"].values
                    vmin, vmax = float(np.min(z)), float(np.max(z))
                    if np.isclose(vmin, vmax): vmin, vmax = vmin - 0.5, vmax + 0.5
                    normal = plt.Normalize(vmin, vmax)

                    for i in range(len(dgrapg)):
                        Gi = nx.from_pandas_edgelist(dgrapg.iloc[i:i+1], "ORIGEM", "DESTINO", "W")
                        caux = plt.cm.Oranges(normal(dgrapg.iloc[i]["W"]))
                        nx.draw_networkx_edges(Gi, posx, edge_color=caux, alpha=1.0,
                                               edge_cmap=plt.cm.Oranges, width=4.0, ax=ax)

                    sm = plt.cm.ScalarMappable(cmap="Oranges", norm=normal)
                    cb = plt.colorbar(sm, ax=ax)
                    cb.set_label(r"$F_{ij}$", size=24)
                    for t in cb.ax.get_yticklabels(): t.set_fontsize(6)

                    for n, (x, y) in posx.items():
                        ax.scatter(x, y, s=100, zorder=2, edgecolor="black", lw=1.5, c="#556C8E")
                        if n in listap:
                            ax.annotate(n, xy=(x, y), fontsize=6, ha="center", va="center",
                                        xytext=(0, 10), textcoords="offset points")

                    st.caption(f"Base selecionada: **{marca_escolhida}** ({xlsx_path.as_posix()})")
                    st.pyplot(fig, clear_figure=True)

            except Exception as e:
                st.error(f"Erro ao gerar o grafo para '{marca_escolhida}': {e}")