import pandas as pd
import chardet
import unicodedata   # Python
import re            # Python

# Detectando encoding -> utf-8
#with open(PATH, 'rb') as f:
#    raw_data = f.read(1024)
#    result = chardet.detect(raw_data)
#print(result['encoding'])

def carregar_dados(PATH: str) -> pd.DataFrame:

    # Carrega o CSV em um Data Frame local com Pandas
    df = pd.read_csv(PATH, sep=",", encoding="utf-8")

    # Remove espaços extras nos nomes das colunas
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Remove espaços em colunas string e transforma tudo em minúsculas
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Corrige valores "nulos" digitados como strings
    df.replace(['nan','null', '-', ''], pd.NA, inplace=True)

    # Cria uma lista temporária para armazenar os nomes sem acento
    cleaned_names = []
    for name in df.columns:
        # Normaliza a string, removendo acentos
        normalized_name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
        cleaned_names.append(normalized_name)
    column_names = pd.Series(cleaned_names)  # Atualiza a Series de nomes de colunas

    # Aplica a expressão regular em toda a Series de uma vez com .str.replace()
    column_names = column_names.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)

    df.columns = column_names # Atribui os nomes de colunas limpos de volta ao DataFrame

    # Ajuste na coluna [Duração Média das Chamadas] -> dado origem estava em Data
    df['duracao_media_das_chamadas'] = df['duracao_media_das_chamadas'].astype(float)
    df['duracao_media_das_chamadas'].fillna(0, inplace=True)

    #print("Colunas carregadas:", df.columns.tolist())

    return df

def extrair_uf(df, coluna_origem='localizacao'):
    """
    Extrai a sigla do estado (UF) a partir de uma coluna de localização com formato variável.
    Ex: 'São Paulo/SP/São Paulo' → 'SP'
    """
    def extrair_uf_de_linha(local):
        if pd.isna(local):
            return None
        partes = local.strip().split('/')
        if len(partes) >= 2:
            return partes[1].strip().upper()
        return None

    df['estado'] = df[coluna_origem].apply(extrair_uf_de_linha)
    return df

#df = carregar_dados(PATH)

#print(df.head())

#print(df.to_string())
