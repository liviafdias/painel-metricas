import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import re
import math
import unicodedata
import json
from io import BytesIO
from itertools import combinations
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from deep_translator import GoogleTranslator
from spacy.lang.pt.stop_words import STOP_WORDS as _stop_pt
from spacy.lang.sv.stop_words import STOP_WORDS as _stop_sv

# ====================== CONFIGURAÇÃO ======================
st.set_page_config(page_title="Dashboard Instagram", layout="wide")
st.title("Dashboard — Métricas do Instagram")

# ====================== CARREGAMENTO ======================
BASE_DIR = os.path.dirname(__file__)

_br_path = os.path.join(BASE_DIR, "csv", "montadoras", "TSNE_BR_COM_SENTIMENTO_montadoras.csv")
_se_path = os.path.join(BASE_DIR, "csv", "montadoras", "TSNE_SE_COM_SENTIMENTO_montadoras.csv")

@st.cache_data
def carregar_dados(br_path, se_path):
    br = pd.read_csv(br_path, sep=";",
                     usecols=["MARCA", "DT_PUBLICACAO", "CURTIDAS", "COMENTARIOS", "ORIGEM", "TEXTO", "LINK_PUBLICACAO"])
    se = pd.read_csv(se_path, sep=";",
                     usecols=["MARCA", "DT_PUBLICACAO", "CURTIDAS", "COMENTARIOS", "ORIGEM", "TEXTO", "LINK_PUBLICACAO"])
    df = pd.concat([br, se], ignore_index=True)
    df["DT_PUBLICACAO"] = pd.to_datetime(df["DT_PUBLICACAO"], errors="coerce")
    df = df.dropna(subset=["DT_PUBLICACAO"])
    df["ANO"] = df["DT_PUBLICACAO"].dt.year
    df["CURTIDAS"] = pd.to_numeric(df["CURTIDAS"], errors="coerce").fillna(0)
    df["COMENTARIOS"] = pd.to_numeric(df["COMENTARIOS"], errors="coerce").fillna(0)
    df["TIPO"] = df["LINK_PUBLICACAO"].str.contains("/reel/", na=False).map({True: "Reel", False: "Foto"})
    return df

df = carregar_dados(_br_path, _se_path)

@st.cache_data
def carregar_dados_sentimento(br_path, se_path):
    cols_uteis = ["MARCA", "DT_PUBLICACAO", "CURTIDAS", "COMENTARIOS",
                  "ORIGEM", "TEXTO", "LINK_PUBLICACAO", "SENTIMENTO"]
    br = pd.read_csv(br_path, sep=";", usecols=lambda c: c in cols_uteis)
    se = pd.read_csv(se_path, sep=";", usecols=lambda c: c in cols_uteis)
    df_s = pd.concat([br, se], ignore_index=True)
    df_s["DT_PUBLICACAO"] = pd.to_datetime(df_s["DT_PUBLICACAO"], errors="coerce")
    df_s = df_s.dropna(subset=["DT_PUBLICACAO"])
    df_s["ANO"] = df_s["DT_PUBLICACAO"].dt.year
    df_s["CURTIDAS"] = pd.to_numeric(df_s["CURTIDAS"], errors="coerce").fillna(0)
    df_s["COMENTARIOS"] = pd.to_numeric(df_s["COMENTARIOS"], errors="coerce").fillna(0)
    _mapa = {"positive": "positivo", "neutral": "neutro",
             "negative": "negativo", "inconclusivo": "inconclusivo"}
    df_s["SENTIMENTO"] = (
        df_s["SENTIMENTO"]
        .astype(str)
        .str.lower()
        .map(lambda x: _mapa.get(x, x))
    )
    return df_s

# ====================== FILTROS ======================
st.sidebar.header("Filtros")
st.sidebar.caption("Selecione ao menos uma opção em cada filtro para visualizar os gráficos.")

marcas_disponiveis = sorted(df["MARCA"].dropna().unique().tolist())
marcas_sel = st.sidebar.multiselect("Marca", marcas_disponiveis, default=[], placeholder="Escolha")

paises_disponiveis = sorted(df["ORIGEM"].dropna().unique().tolist())
paises_sel = st.sidebar.multiselect("País", paises_disponiveis, default=[], placeholder="Escolha")

anos_disponiveis = [2020, 2021, 2022, 2023, 2024, 2025]
anos_sel = st.sidebar.multiselect("Período", anos_disponiveis, default=[], placeholder="Escolha")

agrupamento = st.sidebar.radio("Visualização", ["Mensal", "Anual"])
tipo_grafico = st.sidebar.radio("Tipo de gráfico", ["Barras", "Linhas"])
escala_log = st.sidebar.toggle("Escala logarítmica")

top_n_palavras = st.sidebar.slider(
    "Top N palavras",
    min_value=1,
    max_value=40,
    value=10,
    help="Número de palavras mais frequentes exibidas no grafo.",
)

# ====================== VALIDAÇÃO DOS FILTROS ======================
filtros_vazios = not marcas_sel or not paises_sel or not anos_sel

if filtros_vazios:
    st.info("Utilize os filtros na barra lateral para selecionar **Marca**, **País** e **Período** e visualizar os gráficos.")
    st.stop()

# ====================== FILTRAGEM ======================
df_filtrado = df[
    df["MARCA"].isin(marcas_sel) &
    df["ORIGEM"].isin(paises_sel) &
    df["ANO"].isin(anos_sel)
].copy()

if df_filtrado.empty:
    st.warning("Nenhum dado encontrado para os filtros selecionados.")
    st.stop()

# ====================== AGRUPAMENTO ======================
if agrupamento == "Mensal":
    df_filtrado["PERIODO"] = df_filtrado["DT_PUBLICACAO"].dt.to_period("M").astype(str)
else:
    df_filtrado["PERIODO"] = df_filtrado["ANO"].astype(str)

df_agrupado = (
    df_filtrado.groupby(["PERIODO", "ORIGEM", "MARCA"])
    .agg(
        PUBLICACOES=("MARCA", "count"),
        CURTIDAS=("CURTIDAS", "sum"),
        COMENTARIOS=("COMENTARIOS", "sum"),
    )
    .reset_index()
    .sort_values("PERIODO")
)

# Paleta de cores por marca
PALETA_MARCAS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD"
]
marcas_unicas = sorted(df_agrupado["MARCA"].unique())
cor_marca = {marca: PALETA_MARCAS[i % len(PALETA_MARCAS)] for i, marca in enumerate(marcas_unicas)}

def _ajustar_cor(hex_cor, fator):
    """fator < 1 escurece a cor, fator > 1 clareia. Mantém a identidade da marca."""
    hex_cor = hex_cor.lstrip("#")
    r, g, b = (int(hex_cor[i:i + 2], 16) for i in (0, 2, 4))
    if fator < 1:
        r, g, b = (int(c * fator) for c in (r, g, b))
    else:
        r, g, b = (int(c + (255 - c) * (fator - 1)) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"

# Fator de cor por país: BR mantém a cor base, SE usa um tom mais escuro
# para que a mesma marca fique visualmente distinta entre os países.
COR_ORIGEM_FATOR = {"BR": 1.0, "SE": 0.55}

def cor_marca_pais(marca, origem):
    return _ajustar_cor(cor_marca[marca], COR_ORIGEM_FATOR.get(origem, 1.0))

OPACIDADE = {"BR": 1.0, "SE": 1.0}

limites_max = {
    col: df_agrupado[col].replace(0, pd.NA).max()
    for col in ["PUBLICACOES", "CURTIDAS", "COMENTARIOS"]
}

def yaxis_log_range(col):
    max_val = limites_max[col]
    if not max_val or max_val <= 0:
        return [0, 3]
    return [0, math.ceil(math.log10(max_val)) + 0.1]

# ====================== STOPWORDS / NLP ======================
_MARCAS = {
    "byd", "fiat", "ford", "nissan", "renault", "toyota", "volkswagen", "volvo",
    "caoa", "chery", "caoachery", "caoacherypremium",
    "vw", "vwbrasil", "vwsverige", "bydautobrasil", "byddautobrasil",
    "codismanchevrolet", "codisman", "chevrolet", "sanautochevrolet", "fordantaresoficial", "antar",
    "hondanovaluzoficial", "honda", "fortnissan", "jangadanissan", "porschecenterfortaleza",
    "newlandtoyota", "newland", "cearamotor", "fazautovw", "fazauto", "volvocarsgncsueciafortaleza",
}

_MODELOS = {
    "han", "tang", "song", "seal", "dolphin", "yuan", "atto",
    "tiggo", "arrizo", "omoda",
    "uno", "argo", "mobi", "toro", "strada", "pulse", "fastback", "cronos",
    "palio", "ducato", "doblo", "fiorino", "abarth", "500",
    "ranger", "bronco", "territory", "maverick", "mustang", "transit",
    "puma", "explorer", "ecosport", "capri", "mondeo",
    "kicks", "versa", "frontier", "sentra", "leaf", "march", "ariya",
    "kwid", "sandero", "logan", "duster", "captur", "oroch",
    "corolla", "hilux", "rav4", "sw4", "yaris", "etios", "prius",
    "gol", "polo", "virtus", "tcross", "taos", "amarok", "saveiro",
    "jetta", "tiguan", "passat", "troc", "id4", "id3",
    "xc40", "xc60", "xc90", "s60", "s90", "v60", "v90", "c40", "ex30", "ex90",
}

_REDES_SOCIAIS = {
    "instagram", "reels", "reel", "post", "posts", "link", "bio",
    "story", "stories", "feed", "publi", "parceria", "sponsored",
}

@st.cache_data
def _stopwords_br():
    _dominio_br = {
        "clique", "acesse", "saiba", "confira", "conheça", "veja", "assista",
        "participe", "aproveite", "garanta", "descubra", "baixe", "cadastre",
        "carro", "carros", "veículo", "veículos", "suv", "sedan", "picape",
        "elétrico", "elétrica", "híbrido", "híbrida", "motor", "potência",
        "lançamento", "novidade", "oferta", "desconto", "promoção", "testdrive",
        "concessionária", "revendedor", "autorizada", "disponível",
        "pra", "pro", "tá", "tô", "né", "aí",
        "novo", "nova", "novos", "novas", "grande", "grandes",
        "dia", "dias", "ano", "anos", "hoje", "agora", "sempre", "ainda",
        "nosso", "nossa", "nossos", "nossas", "todo", "toda", "todos", "todas",
        "cada", "fazer", "poder", "viver", "coisa", "coisas", "forma", "parte",
        "melhor", "melhores", "mundo", "vida", "tempo", "lugar", "gente",
        "life", "day", "new", "car", "drive", "best", "love", "feel", "meet",
        "made", "make", "your", "more", "than", "with", "just", "that",
    }
    return frozenset(_stop_pt) | _MARCAS | _MODELOS | _REDES_SOCIAIS | _dominio_br

@st.cache_data
def _stopwords_se():
    _dominio_se = {
        "klicka", "läs", "utforska", "upptäck", "prenumerera", "delta",
        "besök", "boka", "provkör", "ladda", "följ",
        "bil", "bilar", "elbil", "elbilar", "laddhybrid", "hybrid",
        "suv", "sedan", "motor", "räckvidd", "laddning", "körning",
        "provkörning", "erbjudande", "nyhet", "nyheter", "modell", "modeller",
        "fordon", "återförsäljare", "kampanj", "rabatt",
        "mer", "här", "nu", "nya", "nytt", "länk", "bio",
        "stor", "stort", "stora", "bra", "bäst", "bästa",
        "dag", "dagar", "år", "hela", "varje", "bara", "just",
        "även", "också", "redan", "snart", "igen", "alla", "allt",
        "din", "ditt", "dina", "vårt", "våra", "vår",
        "värld", "livet", "plats", "sätt", "känsla", "tid",
        "life", "new", "car", "drive", "best", "love", "feel", "meet",
        "made", "make", "your", "more", "than", "with", "just", "that",
    }
    return frozenset(_stop_sv) | _MARCAS | _MODELOS | _REDES_SOCIAIS | _dominio_se

@st.cache_resource
def _nlp(lang):
    import spacy
    modelo = "pt_core_news_sm" if lang == "BR" else "sv_core_news_sm"
    return spacy.load(modelo, disable=["parser", "ner"])

def _limpar(texto, sw):
    if not isinstance(texto, str):
        return ""
    texto = unicodedata.normalize("NFKC", texto.lower())
    texto = re.sub(r"#\S+|http\S+|@\w+|[^\w\s]|\d+", " ", texto)
    return " ".join(p for p in texto.split() if p not in sw and len(p) > 3)

_POS_RELEVANTES = {"NOUN", "PROPN", "ADJ"}

@st.cache_data(show_spinner="Traduzindo posts suecos para português…")
def _traduzir_sv_pt(textos_tuple):
    cache_path = os.path.join(BASE_DIR, "traducoes_cache.json")
    try:
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}

    resultado = list(textos_tuple)
    novos = {}
    tradutor = GoogleTranslator(source="sv", target="pt")

    for i, texto in enumerate(textos_tuple):
        if not isinstance(texto, str) or not texto.strip():
            continue
        chave = texto[:4900]
        if chave in cache:
            resultado[i] = cache[chave]
        else:
            try:
                resultado[i] = tradutor.translate(chave) or texto
                novos[chave] = resultado[i]
            except Exception:
                novos[chave] = texto

    if novos:
        cache.update(novos)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)

    return tuple(resultado)

@st.cache_data
def _traduzir_pt_sv(termos_tuple):
    tradutor = GoogleTranslator(source="pt", target="sv")
    resultado = {}
    for termo in termos_tuple:
        try:
            resultado[termo] = tradutor.translate(termo) or termo
        except Exception:
            resultado[termo] = termo
    return resultado

def _prep_textos(df_col_texto, sw, pais):
    if pais == "SE":
        brutos = tuple(df_col_texto.dropna().tolist())
        traduzidos = _traduzir_sv_pt(brutos)
        sw_pt = _stopwords_br()
        textos = tuple(_limpar(t, sw_pt) for t in traduzidos)
        return textos, sw_pt, "BR"
    textos = tuple(df_col_texto.dropna().apply(lambda t: _limpar(t, sw)).tolist())
    return textos, sw, pais

# ====================== GRÁFICOS DE ENGAJAMENTO ======================
def fazer_grafico(y_col, titulo):
    fig = go.Figure()
    usar_linhas = tipo_grafico == "Linhas"
    for origem in sorted(df_agrupado["ORIGEM"].unique()):
        for marca in marcas_unicas:
            df_slice = df_agrupado[
                (df_agrupado["ORIGEM"] == origem) &
                (df_agrupado["MARCA"] == marca)
            ]
            if df_slice.empty:
                continue
            cor = cor_marca_pais(marca, origem)
            opacidade = OPACIDADE.get(origem, 1.0)
            ht = "<b>%{x}</b><br>" + f"{marca} ({origem}) — {titulo}: " + "%{y:,.0f}<extra></extra>"
            if usar_linhas:
                fig.add_trace(go.Scatter(
                    x=df_slice["PERIODO"],
                    y=df_slice[y_col],
                    name=f"{marca} ({origem})",
                    mode="lines+markers",
                    line=dict(color=cor, dash="dot" if origem == "SE" else "solid"),
                    opacity=opacidade,
                    hovertemplate=ht,
                ))
            else:
                fig.add_trace(go.Bar(
                    x=df_slice["PERIODO"],
                    y=df_slice[y_col],
                    name=f"{marca} ({origem})",
                    marker_color=cor,
                    marker_opacity=opacidade,
                    hovertemplate=ht,
                ))

    yaxis = dict(title=titulo)
    if escala_log:
        yaxis.update(
            type="log",
            range=yaxis_log_range(y_col),
            tickformat=".0s",
        )

    fig.update_layout(
        title=titulo,
        xaxis_title=agrupamento,
        yaxis=yaxis,
        barmode="group",
        hovermode="x unified",
        height=400,
        margin=dict(t=50, b=40, l=40, r=20),
        xaxis=dict(tickangle=-45),
        legend=dict(title="Marca (País)", groupclick="toggleitem"),
    )
    return fig

# ====================== ABAS ======================
aba_eng, aba_grafo, aba_sent = st.tabs([
    "Métricas de Engajamento",
    "Grafos de Coocorrência",
    "Análise de Sentimentos",
])

# ====================== ABA 1: MÉTRICAS DE ENGAJAMENTO ======================
with aba_eng:
    total_pub  = int(df_agrupado["PUBLICACOES"].sum())
    total_curt = int(df_agrupado["CURTIDAS"].sum())
    total_com  = int(df_agrupado["COMENTARIOS"].sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Publicações", f"{total_pub:,}".replace(",", "."))
    col2.metric("Total de Curtidas",    f"{total_curt:,}".replace(",", "."))
    col3.metric("Total de Comentários", f"{total_com:,}".replace(",", "."))

    st.divider()

    st.plotly_chart(fazer_grafico("PUBLICACOES", "Total de Publicações"), use_container_width=True)
    st.plotly_chart(fazer_grafico("CURTIDAS",    "Total de Curtidas"),    use_container_width=True)
    st.plotly_chart(fazer_grafico("COMENTARIOS", "Total de Comentários"), use_container_width=True)

    st.divider()
    st.subheader("Contagem de Postagens por Tipo")

    df_tipo = (
        df_filtrado.groupby(["PERIODO", "ORIGEM", "MARCA", "TIPO"])
        .size()
        .reset_index(name="CONTAGEM")
        .sort_values("PERIODO")
    )

    fig_tipo = go.Figure()
    usar_linhas = tipo_grafico == "Linhas"
    for tipo in ["Foto", "Reel"]:
        for origem in sorted(df_tipo["ORIGEM"].unique()):
            for marca in marcas_unicas:
                df_slice = df_tipo[
                    (df_tipo["TIPO"] == tipo) &
                    (df_tipo["ORIGEM"] == origem) &
                    (df_tipo["MARCA"] == marca)
                ]
                if df_slice.empty:
                    continue
                cor = cor_marca_pais(marca, origem)
                opacidade = OPACIDADE.get(origem, 1.0)
                ht = "<b>%{x}</b><br>" + f"{marca} ({origem}) — {tipo}: " + "%{y:,.0f}<extra></extra>"
                if usar_linhas:
                    fig_tipo.add_trace(go.Scatter(
                        x=df_slice["PERIODO"],
                        y=df_slice["CONTAGEM"],
                        name=f"{marca} ({origem}) — {tipo}",
                        mode="lines+markers",
                        line=dict(
                            color=cor,
                            dash="dot" if tipo == "Reel" else ("dash" if origem == "SE" else "solid"),
                        ),
                        marker=dict(symbol="diamond" if tipo == "Reel" else "circle"),
                        opacity=opacidade,
                        hovertemplate=ht,
                    ))
                else:
                    fig_tipo.add_trace(go.Bar(
                        x=df_slice["PERIODO"],
                        y=df_slice["CONTAGEM"],
                        name=f"{marca} ({origem}) — {tipo}",
                        marker_color=cor,
                        marker_opacity=opacidade,
                        marker_pattern_shape="/" if tipo == "Reel" else "",
                        hovertemplate=ht,
                    ))

    fig_tipo.update_layout(
        xaxis_title=agrupamento,
        yaxis_title="Publicações",
        barmode="group",
        hovermode="x unified",
        height=420,
        margin=dict(t=30, b=40, l=40, r=20),
        xaxis=dict(tickangle=-45),
        legend=dict(title="Marca (País) — Tipo", groupclick="toggleitem"),
    )
    st.plotly_chart(fig_tipo, use_container_width=True)

    st.divider()
    st.subheader("Top 5 Publicações")

    _COLS_TOP = ["MARCA", "ORIGEM", "DT_PUBLICACAO", "CURTIDAS", "COMENTARIOS", "TEXTO", "LINK_PUBLICACAO"]

    def _formatar_top5(df_top):
        df_out = df_top[_COLS_TOP].copy()
        df_out["DT_PUBLICACAO"] = df_out["DT_PUBLICACAO"].dt.strftime("%d/%m/%Y")
        df_out["CURTIDAS"]    = df_out["CURTIDAS"].astype(int)
        df_out["COMENTARIOS"] = df_out["COMENTARIOS"].astype(int)
        df_out["TEXTO"] = df_out["TEXTO"].fillna("").str.slice(0, 120) + "…"
        df_out.columns = ["Marca", "País", "Data", "Curtidas", "Comentários", "Texto", "Link"]
        df_out = df_out.reset_index(drop=True)
        df_out.index += 1
        return df_out

    _LINK_COL = st.column_config.LinkColumn("Link", display_text="link")

    col_top1, col_top2 = st.columns(2)
    with col_top1:
        st.markdown("**Top 5 em Curtidas**")
        top_curt = df_filtrado.nlargest(5, "CURTIDAS")
        st.dataframe(_formatar_top5(top_curt), use_container_width=True,
                     column_config={"Link": _LINK_COL})

    with col_top2:
        st.markdown("**Top 5 em Comentários**")
        top_com = df_filtrado.nlargest(5, "COMENTARIOS")
        st.dataframe(_formatar_top5(top_com), use_container_width=True,
                     column_config={"Link": _LINK_COL})

# ====================== ABA 2: GRAFOS DE COOCORRÊNCIA ======================
@st.cache_data
def _coocorrencia(textos_tuple, stopwords_frozenset, lang="BR", top_n=10, min_cooc=2):
    nlp = _nlp(lang)
    _excluir_tokens = _MARCAS | _MODELOS
    doc_lemmas = []
    for doc in nlp.pipe(textos_tuple, batch_size=64):
        lemmas = [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in _POS_RELEVANTES
            and token.lemma_.lower() not in stopwords_frozenset
            and token.text.lower() not in _excluir_tokens
            and len(token.lemma_) > 3
            and not token.is_space
            and not token.is_punct
        ]
        if len(lemmas) >= 2:
            doc_lemmas.append(lemmas)

    freq = Counter(w for doc in doc_lemmas for w in doc)
    top_words = {w for w, _ in freq.most_common(top_n)}

    cooc = Counter()
    for lemmas in doc_lemmas:
        words_in_doc = [w for w in set(lemmas) if w in top_words]
        for pair in combinations(sorted(words_in_doc), 2):
            cooc[pair] += 1

    cooc = {k: v for k, v in cooc.items() if v >= min_cooc}
    return freq, cooc

def _grafico_coocorrencia(freq, cooc, titulo, sv_labels=None):
    if not cooc:
        return None

    G = nx.Graph()
    for (w1, w2), weight in cooc.items():
        G.add_edge(w1, w2, weight=weight)
    for node in G.nodes():
        G.nodes[node]["freq"] = freq.get(node, 1)

    pos = nx.spring_layout(G, k=1.8, seed=42)

    max_weight = max(cooc.values()) if cooc else 1
    edge_traces = []
    for (u, v), weight in cooc.items():
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        opacity = 0.15 + 0.65 * (weight / max_weight)
        width = 0.8 + 4 * (weight / max_weight)
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color=f"rgba(140,140,140,{opacity:.2f})"),
            hoverinfo="none",
            showlegend=False,
        ))

    nodes = list(G.nodes())
    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]
    node_freq = [G.nodes[n].get("freq", 1) for n in nodes]
    max_freq = max(node_freq) if node_freq else 1
    node_sizes = [12 + 32 * (f / max_freq) for f in node_freq]

    if sv_labels:
        tooltip_labels = [sv_labels.get(n, n) for n in nodes]
        hovertemplate = "<b>%{customdata}</b><br>Frequência: %{marker.color}<extra></extra>"
    else:
        tooltip_labels = nodes
        hovertemplate = "<b>%{text}</b><br>Frequência: %{marker.color}<extra></extra>"

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=nodes,
        customdata=tooltip_labels,
        textposition="top center",
        textfont=dict(size=11, family="Arial Black, sans-serif"),
        marker=dict(
            size=node_sizes,
            color=node_freq,
            colorscale="Turbo",
            cmin=1,
            cmax=max_freq,
            showscale=True,
            colorbar=dict(title="Frequência", thickness=12, len=0.6, tickfont=dict(size=10)),
            line=dict(width=1.5, color="rgba(30,30,30,0.6)"),
        ),
        hovertemplate=hovertemplate,
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=titulo,
        height=520,
        margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig

with aba_grafo:
    st.subheader("Grafos de Coocorrência")
    _sw_map_grafo = {"BR": _stopwords_br(), "SE": _stopwords_se()}

    st.markdown("##### Visão geral por país — todas as marcas")
    st.caption("Palavras que co-ocorrem nos posts de todas as marcas filtradas, agrupadas por país.")

    paises_com_dados = [p for p in sorted(paises_sel)
                        if not df_filtrado[df_filtrado["ORIGEM"] == p].empty]

    if paises_com_dados:
        cols_ag = st.columns(len(paises_com_dados))
        for col_ag, pais in zip(cols_ag, paises_com_dados):
            sw = _sw_map_grafo.get(pais, frozenset())
            textos_pais, sw_ef, lang_ef = _prep_textos(
                df_filtrado[df_filtrado["ORIGEM"] == pais]["TEXTO"], sw, pais,
            )
            freq_ag, cooc_ag = _coocorrencia(
                textos_pais, sw_ef, lang=lang_ef, top_n=top_n_palavras, min_cooc=3
            )
            sv_labels_ag = _traduzir_pt_sv(tuple({w for par in cooc_ag for w in par})) if pais == "SE" else None
            fig_ag = _grafico_coocorrencia(freq_ag, cooc_ag, f"Todas as marcas selecionadas — {pais}", sv_labels=sv_labels_ag)
            with col_ag:
                if fig_ag:
                    fig_ag.update_layout(height=620)
                    st.plotly_chart(fig_ag, use_container_width=True, key="cooc_geral_" + pais)
                else:
                    st.caption(f"{pais}: sem coocorrências suficientes.")

    st.divider()
    st.markdown("##### Por marca selecionada")
    st.caption("Palavras que co-ocorrem nos posts de cada marca filtrada, separados por país.")

    for marca in sorted(marcas_sel):
        df_marca = df_filtrado[df_filtrado["MARCA"] == marca]
        paises_marca = [p for p in sorted(paises_sel)
                        if not df_marca[df_marca["ORIGEM"] == p].empty]
        if not paises_marca:
            continue

        st.markdown(f"**{marca}**")
        cols_m = st.columns(len(paises_marca))
        for col_m, pais in zip(cols_m, paises_marca):
            sw = _sw_map_grafo.get(pais, frozenset())
            textos_mp, sw_ef, lang_ef = _prep_textos(
                df_marca[df_marca["ORIGEM"] == pais]["TEXTO"], sw, pais,
            )
            freq_m, cooc_m = _coocorrencia(
                textos_mp, sw_ef, lang=lang_ef, top_n=top_n_palavras, min_cooc=2
            )
            sv_labels_m = _traduzir_pt_sv(tuple({w for par in cooc_m for w in par})) if pais == "SE" else None
            fig_m = _grafico_coocorrencia(freq_m, cooc_m, f"{marca} — {pais}", sv_labels=sv_labels_m)
            with col_m:
                if fig_m:
                    fig_m.update_layout(height=620)
                    st.plotly_chart(fig_m, use_container_width=True, key=f"cooc_{marca}_{pais}")
                else:
                    st.caption(f"{pais}: sem coocorrências suficientes.")

# ====================== ABA 3: ANÁLISE DE SENTIMENTOS ======================
_COR_SENT = {
    "positivo":     "#2ca02c",
    "neutro":       "#1f77b4",
    "negativo":     "#d62728",
    "inconclusivo": "#7f7f7f",
}

with aba_sent:
    st.subheader("Análise de Sentimentos")

    try:
        df_sent_all = carregar_dados_sentimento(_br_path, _se_path)
    except Exception as _e:
        st.error(f"Não foi possível carregar os dados de sentimento: {_e}")
        st.stop()

    df_sent = df_sent_all[
        df_sent_all["MARCA"].isin(marcas_sel) &
        df_sent_all["ORIGEM"].isin(paises_sel) &
        df_sent_all["ANO"].isin(anos_sel)
    ].copy()

    if df_sent.empty:
        st.warning("Nenhum dado de sentimento encontrado para os filtros selecionados.")
    else:
        _sent_disp = sorted(df_sent["SENTIMENTO"].dropna().unique().tolist())
        _pais_disp = sorted(df_sent["ORIGEM"].dropna().unique().tolist())

        _fc1, _fc2 = st.columns(2)
        with _fc1:
            sentimentos_sel_tab = st.multiselect(
                "Sentimento", _sent_disp, default=_sent_disp, key="tab_sent_sentimentos",
            )
        with _fc2:
            paises_sent_tab = st.multiselect(
                "País", _pais_disp, default=_pais_disp, key="tab_sent_paises",
            )

        df_sf = df_sent[
            df_sent["SENTIMENTO"].isin(sentimentos_sel_tab if sentimentos_sel_tab else _sent_disp) &
            df_sent["ORIGEM"].isin(paises_sent_tab if paises_sent_tab else _pais_disp)
        ].copy()

        _sent_ativo = sentimentos_sel_tab if sentimentos_sel_tab else _sent_disp

        st.divider()

        # ---- 1. Frequência dos sentimentos ----
        st.markdown("##### 1. Frequência dos Sentimentos")
        _freq = df_sf["SENTIMENTO"].value_counts().reset_index()
        _freq.columns = ["SENTIMENTO", "CONTAGEM"]

        fig_freq = go.Figure()
        for _, _r in _freq.iterrows():
            fig_freq.add_trace(go.Bar(
                x=[_r["SENTIMENTO"]], y=[_r["CONTAGEM"]],
                name=_r["SENTIMENTO"],
                marker_color=_COR_SENT.get(_r["SENTIMENTO"], "#888"),
                hovertemplate=f"<b>{_r['SENTIMENTO']}</b><br>Publicações: %{{y:,}}<extra></extra>",
            ))
        fig_freq.update_layout(
            xaxis_title="Sentimento", yaxis_title="Publicações",
            showlegend=False, height=380,
            margin=dict(t=20, b=40, l=40, r=20),
        )
        st.plotly_chart(fig_freq, use_container_width=True)

        st.divider()

        # ---- 2. Percentual de publicações por sentimento ----
        st.markdown("##### 2. Percentual de Publicações por Sentimento")
        fig_pie = go.Figure(go.Pie(
            labels=_freq["SENTIMENTO"],
            values=_freq["CONTAGEM"],
            marker_colors=[_COR_SENT.get(s, "#888") for s in _freq["SENTIMENTO"]],
            hole=0.4,
            hovertemplate="<b>%{label}</b><br>%{percent}<br>%{value:,} publicações<extra></extra>",
        ))
        fig_pie.update_layout(height=400, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True)

        st.divider()

        # ---- 3. Distribuição de sentimentos por marca e país ----
        st.markdown("##### 3. Distribuição de Sentimentos por Marca e País")
        _dist = (
            df_sf.groupby(["MARCA", "ORIGEM", "SENTIMENTO"])
            .size().reset_index(name="CONTAGEM")
        )
        _dist["MARCA_PAIS"] = _dist["MARCA"] + " (" + _dist["ORIGEM"] + ")"

        fig_dist = go.Figure()
        for _s in _sent_ativo:
            _d = _dist[_dist["SENTIMENTO"] == _s]
            if _d.empty:
                continue
            fig_dist.add_trace(go.Bar(
                x=_d["MARCA_PAIS"], y=_d["CONTAGEM"],
                name=_s,
                marker_color=_COR_SENT.get(_s, "#888"),
                hovertemplate="<b>%{x}</b><br>" + f"{_s}: " + "%{y:,}<extra></extra>",
            ))
        fig_dist.update_layout(
            xaxis_title="Marca (País)", yaxis_title="Publicações",
            barmode="group", height=460,
            margin=dict(t=20, b=90, l=40, r=20),
            xaxis=dict(tickangle=-45),
            legend=dict(title="Sentimento"),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        st.divider()

        # ---- 4. Evolução temporal dos sentimentos ----
        st.markdown("##### 4. Evolução Temporal dos Sentimentos por Marca e País")

        if agrupamento == "Mensal":
            df_sf["PERIODO_S"] = df_sf["DT_PUBLICACAO"].dt.to_period("M").astype(str)
        else:
            df_sf["PERIODO_S"] = df_sf["ANO"].astype(str)

        _temp = (
            df_sf.groupby(["PERIODO_S", "MARCA", "ORIGEM", "SENTIMENTO"])
            .size().reset_index(name="CONTAGEM")
            .sort_values("PERIODO_S")
        )

        fig_temp = go.Figure()
        for _s in _sent_ativo:
            for _orig in (paises_sent_tab if paises_sent_tab else _pais_disp):
                for _marca in sorted(marcas_sel):
                    _d = _temp[
                        (_temp["SENTIMENTO"] == _s) &
                        (_temp["ORIGEM"] == _orig) &
                        (_temp["MARCA"] == _marca)
                    ]
                    if _d.empty:
                        continue
                    fig_temp.add_trace(go.Scatter(
                        x=_d["PERIODO_S"], y=_d["CONTAGEM"],
                        name=f"{_marca} ({_orig}) — {_s}",
                        mode="lines+markers",
                        line=dict(
                            color=_COR_SENT.get(_s, "#888"),
                            dash="dot" if _orig == "SE" else "solid",
                        ),
                        opacity=OPACIDADE.get(_orig, 1.0),
                        hovertemplate=(
                            "<b>%{x}</b><br>"
                            + f"{_marca} ({_orig}) — {_s}: "
                            + "%{y:,}<extra></extra>"
                        ),
                    ))

        fig_temp.update_layout(
            xaxis_title=agrupamento, yaxis_title="Publicações",
            hovermode="x unified", height=500,
            margin=dict(t=20, b=40, l=40, r=20),
            xaxis=dict(tickangle=-45),
            legend=dict(title="Marca (País) — Sentimento", groupclick="toggleitem"),
        )
        st.plotly_chart(fig_temp, use_container_width=True)