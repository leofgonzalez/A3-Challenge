import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribuicao_sentimentos(df):

    # Dados pré-formatados
    dados = df['sentimento_bert'].value_counts()

    # Ordem e cores das categorias
    ordem = ['muito ruim', 'ruim', 'neutro', 'bom', 'muito bom']
    cores = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']  # Vermelho a verde
    rotulos = ['Muito ruim', 'Ruim', 'Neutro',
               'Bom', 'Muito bom']

    # Criação do gráfico
    plt.figure(figsize=(10, 6))
    bars = plt.bar(rotulos, [dados.get(cat, 0) for cat in ordem], color=cores)

    # Adiciona os valores nas barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Customização do visual
    plt.title('Distribuição das Avaliações por Sentimento', pad=20, fontsize=14, fontweight='bold')
    plt.ylabel('Quantidade de Clientes', fontsize=12)
    plt.xticks(rotation=25, ha='right', fontsize=11)
    plt.yticks(fontsize=10)

    # Adiciona grid sutil
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # Remove bordas do gráfico
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.show()


# Referência para mapear os sentimentos para classes agregadas
mapa_sentimentos = {
    'muito ruim': 'negativo',
    'ruim': 'negativo',
    'neutro': 'neutro',
    'bom': 'positivo',
    'muito bom': 'positivo'
}

def sentimentos_por_estado(df):
    # Cria cópia e aplica transformações
    df_temp = df.copy()
    df_temp['sentimento_agregado'] = df_temp['sentimento_bert'].map(mapa_sentimentos)
    df_temp = df_temp.dropna(subset=['estado', 'sentimento_agregado'])

    # Calcula proporções garantindo todas as categorias
    contagem = (df_temp.groupby(['estado', 'sentimento_agregado'])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=['positivo', 'neutro', 'negativo'], fill_value=0))

    proporcao = contagem.div(contagem.sum(axis=1), axis=0) * 100

    # Ordena por maior % positivo
    proporcao = proporcao.loc[proporcao['positivo'].sort_values(ascending=False).index]

    # Configuração do gráfico
    cores = {'positivo': '#2ecc71', 'neutro': '#95a5a6', 'negativo': '#e74c3c'}
    ordem = ['positivo', 'neutro', 'negativo']

    # Plot
    ax = proporcao[ordem].plot(
        kind='bar',
        stacked=True,
        color=[cores[c] for c in ordem],
        figsize=(12, 6),
        edgecolor='w',
        width=0.8
    )

    # Formatação
    plt.title('Distribuição Percentual de Sentimentos por Estado', pad=20)
    plt.ylabel('Percentual (%)')
    plt.xlabel('Estado')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Sentimento', bbox_to_anchor=(1.05, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Adiciona rótulos de porcentagem
    for i, estado in enumerate(proporcao.index):
        acumulado = 0
        for sentimento in ordem:
            valor = proporcao.loc[estado, sentimento]
            if valor > 5:  # Só mostra rótulos para valores significativos
                ax.text(
                    i,
                    acumulado + valor / 2,
                    f'{valor:.0f}%',
                    ha='center',
                    va='center',
                    color='white' if sentimento == 'positivo' else 'black'
                )
            acumulado += valor

    plt.tight_layout()
    plt.show()

def sentimento_por_servico(df):
    # Configurações de cores e ordem
    cores = {
        'muito ruim': '#e74c3c',
        'ruim': '#e67e22',
        'neutro': '#f1c40f',
        'bom': '#2ecc71',
        'muito bom': '#27ae60'
    }
    categorias = ['muito bom', 'bom', 'neutro', 'ruim', 'muito ruim']

    # Garante todas as colunas mesmo se não existirem nos dados
    dados = (df.groupby(['tipo_de_servico', 'sentimento_bert'])
             .size()
             .unstack(fill_value=0)
             .reindex(columns=categorias, fill_value=0))

    # Normalização para porcentagem
    dados_percent = dados.div(dados.sum(axis=1), axis=0) * 100

    # Criação do gráfico
    fig, ax = plt.subplots(figsize=(12, 7))

    # Gráfico de barras empilhadas
    dados_percent.plot.barh(
        stacked=True,
        color=[cores[cat] for cat in categorias],
        ax=ax,
        width=0.8,
        edgecolor='white',
        linewidth=0.5
    )

    # Customização
    ax.set_title('Distribuição de Sentimentos por Tipo de Serviço',
                 pad=20, fontsize=16, fontweight='bold')
    ax.set_xlabel('Percentual (%)', fontsize=12)
    ax.set_ylabel('Tipo de Serviço', fontsize=12)
    ax.legend(title='Classificação', bbox_to_anchor=(1.05, 1))
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    # Adiciona rótulos de porcentagem
    for i, (servico, valores) in enumerate(dados_percent.iterrows()):
        acumulado = 0
        for cat in categorias:
            valor = valores[cat]
            if valor > 0:  # Mostra apenas valores positivos
                ax.text(
                    acumulado + valor / 2, i,
                    f'{valor:.0f}%',
                    ha='center', va='center',
                    color='white' if cat in ['bom', 'muito bom'] else 'black',
                    fontsize=10
                )
            acumulado += valor

    # Remove bordas
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.show()

def printar_clusters_servico(df):
    """Analisa e exibe os clusters por tipo de serviço"""
    print("\nAnálise de Clusters por Tipo de Serviço:")
    print("=" * 60)

    for cluster in sorted(df['cluster'].unique()):
        subset = df[df['cluster'] == cluster]
        servico = cluster.split('_')[0]
        num_cluster = cluster.split('_')[1]

        print(f"\nCluster: {cluster} ({len(subset)} comentários)")
        print("-" * 60)

        # Palavras mais frequentes
        all_words = ' '.join(subset['comentarios_limpos']).split()
        freq_words = pd.Series(all_words).value_counts().head(5)
        print("Palavras mais frequentes:", ', '.join(freq_words.index.tolist()))

        # Exemplos de comentários
        print("\nExemplos de Comentários:")
        for i, row in subset.head(2).iterrows():
            print(f"- {row['comentarios'][:150]}...")

        print("\n" + "=" * 60)
