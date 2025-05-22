# Bibliotecas Gerais
import pandas as pd
import string

# Análise de sentimentos e NLP
from transformers import pipeline

# Clusterização via K-MEANS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Bibliotecas internas
from etl_data import carregar_dados, extrair_uf
from visualization import sentimentos_por_estado,plot_distribuicao_sentimentos, sentimento_por_servico, printar_clusters_servico

# Caminho local do arquivo
PATH = "C:\\Users\\Leonardo\\PycharmProjects\\A3Challenge\\Fonte de Dados\\beta_churn_com_texto_tratado.csv"

df = carregar_dados(PATH)

# ---------------------------------------
# Fase 1: Análise de sentimentos por NLP
# ---------------------------------------

# Carregar modelo
classifier = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# Referência de classificação
mapa_estrelas = {
    1: "muito ruim",
    2: "ruim",
    3: "neutro",
    4: "bom",
    5: "muito bom"
}

# Função para aplicar análise de sentimento
def analisar_sentimento(texto):
    if pd.isna(texto) or not isinstance(texto, str) or texto.strip() == "":
        return None
    try:
        resultado = classifier(texto[:1000])[0]                    # Limita o texto a 1000 tokens
        estrelas = int(resultado['label'][0])                      # Extrai a classificação por estrelas 1-5
        return mapa_estrelas.get(estrelas, "desconhecido")         # Transforma estrelas em texto
    except Exception as e:
        print(f"Erro ao processar texto: {texto[:30]}... => {e}")
        return None

# 3. Aplicar no DataFrame
df['sentimento_bert'] = df['comentarios'].apply(analisar_sentimento)

# Add after creating the classifier
print(classifier.model.config.id2label)

extrair_uf(df)

print(df.to_string())

sentimentos_por_estado(df)

plot_distribuicao_sentimentos(df)

sentimento_por_servico(df)

# ----------------------------------------------------
# Fase 2: Clusterização não supervisionada com K-Means
# ----------------------------------------------------

def preprocess_text(text):
    """Preprocessa o texto para análise"""
    if pd.isna(text):
        return ""

    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('portuguese') + list(string.punctuation))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

    return " ".join(tokens)

def cluster_por_servico(df, n_clusters=3, max_words=500):

    # Filtro para considerarmos apenas comentários negativos
    df_negativos = df[df['sentimento_bert'].isin(['ruim','muito ruim'])].copy()

    # Pré-processamento
    df_negativos['comentarios_limpos'] = df_negativos['comentarios'].apply(preprocess_text)
    df_negativos['cluster'] = ""

    # Processa cada tipo de serviço separadamente
    for servico in df_negativos['tipo_de_servico'].unique():
        subset = df_negativos[df_negativos['tipo_de_servico'] == servico]

        if len(subset) >= n_clusters:  # Só clusteriza se houver dados suficientes // 3 pelo menos nesse caso
            # Vetorização TF-IDF
            vectorizer = TfidfVectorizer(max_features=max_words, stop_words=stopwords.words('portuguese'))
            X = vectorizer.fit_transform(subset['comentarios_limpos'])

            # Clusterização
            kmeans = KMeans(n_clusters=n_clusters, random_state=7)
            clusters = kmeans.fit_predict(X)

            # Nomeia os clusters no formato SERVICO_NUMERO
            df_negativos.loc[subset.index, 'cluster'] = [f"{servico.upper()}_{i + 1}" for i in clusters]
        else:
            df_negativos.loc[subset.index, 'cluster'] = f"{servico.upper()}_1"

    return df_negativos

# Uso das funções
df_clusterizado = cluster_por_servico(df)
printar_clusters_servico(df_clusterizado)

# Salvar resultados
df_clusterizado.to_csv('comentarios_clusterizados.csv', index=False)