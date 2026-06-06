# Painel de Métricas — Montadoras Automotivas no Instagram (BR × SE)

Dashboard interativo em **Streamlit** para análise comparativa da comunicação digital de oito montadoras automotivas no Instagram, contrastando os mercados **brasileiro (BR)** e **sueco (SE)** no período de **2020 a 2025**, no contexto da eletrificação veicular.

Esta ferramenta é a camada de visualização do Trabalho de Conclusão de Curso *"Relevância midiática e comunicação digital de montadoras automotivas no Instagram: uma análise comparativa entre Brasil e Suécia no contexto da eletrificação veicular (2020–2025), com apoio de Processamento de Linguagem Natural"*, de **Lívia Farias Dias** (Ciência da Computação — UNIFOR). O painel permite explorar visualmente os resultados descritos no relatório científico: métricas de engajamento, análise de sentimento e discurso de marca por coocorrência lexical.

As marcas analisadas são: **BYD, Fiat, Ford, Nissan, Renault, Toyota, Volkswagen e Volvo**, cada uma com seu perfil oficial nos dois mercados.

## Funcionalidades

O painel organiza a análise em três abas, todas respondendo aos filtros globais de marca, país e período da barra lateral.

**1. Métricas de Engajamento** — totais de publicações, curtidas e comentários; evolução temporal (mensal ou anual) em barras ou linhas; contagem de postagens por tipo de mídia (Foto × Reel); opção de escala logarítmica para lidar com a forte assimetria dos dados; e ranking das cinco publicações de maior alcance por curtidas e por comentários.

**2. Grafos de Coocorrência** — rede dos termos mais frequentes e suas conexões, em visão geral por país e por marca. As legendas passam por limpeza, lematização (spaCy) e filtragem por classe gramatical (substantivos, nomes próprios e adjetivos), com lista de exclusão de domínio (marcas, modelos e gírias). Os termos suecos são traduzidos automaticamente para o português, garantindo comparabilidade direta entre os mercados.

**3. Análise de Sentimentos** — frequência e percentual de cada polaridade (positivo, neutro, negativo, inconclusivo), distribuição por marca e país e evolução temporal. A classificação é proveniente do pipeline de processamento, usando o modelo multilíngue `cardiffnlp/twitter-xlm-roberta-base-sentiment`.

## Tecnologias

A interface usa **Python** com **Streamlit** e **Plotly** para os gráficos interativos. O processamento de linguagem natural emprega **spaCy** (modelos `pt_core_news_sm` e `sv_core_news_sm`) para lematização e filtragem por classe gramatical, **NetworkX** para a construção dos grafos de coocorrência e **deep-translator** (Google Translator) para a tradução dos termos suecos. A manipulação de dados é feita com **pandas** e **NumPy**.

## Estrutura do repositório

```
painel-metricas/
├── dashboard_novo.py        # Aplicação Streamlit (ponto de entrada)
├── requirements.txt         # Dependências do projeto
└── csv/
    └── montadoras/
        ├── TSNE_BR_COM_SENTIMENTO_montadoras.csv   # Base consolidada do Brasil (Git LFS)
        ├── TSNE_SE_COM_SENTIMENTO_montadoras.csv   # Base consolidada da Suécia (Git LFS)
        ├── todos_filtrados.csv
        ├── br/              # Arquivos individuais por perfil brasileiro
        └── se/              # Arquivos individuais por perfil sueco
```

> **Atenção — Git LFS:** os dois CSVs consolidados (`TSNE_BR...` e `TSNE_SE...`) são grandes e estão versionados via **Git LFS**. Após clonar o repositório, é necessário rodar `git lfs pull` para baixar os dados reais; sem isso, os arquivos serão apenas ponteiros de texto e o painel não carregará.

### Formato dos dados

O painel lê os dois CSVs consolidados (separador `;`) e espera, no mínimo, as seguintes colunas:

| Coluna | Descrição |
|---|---|
| `MARCA` | Nome da montadora |
| `DT_PUBLICACAO` | Data da publicação |
| `CURTIDAS` | Número de curtidas |
| `COMENTARIOS` | Número de comentários |
| `ORIGEM` | Mercado de origem (`BR` ou `SE`) |
| `TEXTO` | Texto da legenda |
| `LINK_PUBLICACAO` | URL do post (usada para inferir o tipo Foto/Reel) |
| `SENTIMENTO` | Rótulo de sentimento (positivo/neutro/negativo/inconclusivo) |

## Instalação

Requer Python 3.10+ (compatível com `spacy==3.8.0`).

```bash
# 1. Clonar o repositório
git clone https://github.com/liviafdias/painel-metricas.git
cd painel-metricas

# 2. Baixar os dados versionados via Git LFS
git lfs install
git lfs pull

# 3. (recomendado) Criar e ativar um ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 4. Instalar as dependências
pip install -r requirements.txt
pip install deep-translator      # ainda não consta no requirements.txt
```

Os modelos de linguagem do spaCy (`pt_core_news_sm` e `sv_core_news_sm`) já são instalados pelo `requirements.txt`, que aponta para os respectivos wheels.

## Como executar

```bash
streamlit run dashboard_novo.py
```

O painel abrirá no navegador (por padrão em `http://localhost:8501`). Use os filtros da barra lateral para selecionar **Marca**, **País** e **Período** — é necessário escolher ao menos uma opção em cada um para que os gráficos sejam exibidos.

## Autoria

- **Autora:** Lívia Farias Dias
- **Orientador:** Prof. Me. Ronaldo Gonçalves Junior
- **Instituição:** Universidade de Fortaleza (UNIFOR) — Curso de Ciência da Computação
- **Ano:** 2026