##Importando pandas para preparar os dados para o modelo preditivo KNN
import pandas as pd

#Localização dos datasets
books = pd.read_csv('dados_limpos_6.csv')

#Verificando os dados
books.head()

colunas_chaves = ['User-ID','ISBN']

# Verifica se há duplicatas com base nas colunas-chave
duplicatas = books.duplicated(subset=colunas_chaves, keep=False)

# Filtra as linhas duplicadas
linhas_duplicadas = books[duplicatas].value_counts()

# Se linhas_duplicadas estiver vazio, não há duplicatas
if linhas_duplicadas.empty:
    print("Não há avaliações duplicadas.")
else:
    print("Avaliações duplicadas encontradas:")
    print(linhas_duplicadas)

#Transformando usuarios em variáveis através da função 'pivot'

books_pivot = books.pivot_table(columns='User-ID' , index='Book-Title', values = 'Book-Rating')

#Visualizando o arquivo transformado
books_pivot.head(20)

#Transformando valores nulos em 0
books_pivot.fillna(0,inplace=True)
books_pivot.head(20)

#Importando a biblioteca Scipy para que seja possivel transformar o dataset em matriz sparsa e compactar os zeros existentes
#de uma forma virtualizada para facilitar o processamento da máquina durante a execução do modelo preditivo.
from scipy.sparse import csr_matrix

#Transformando o dataset em matriz sparsa
books_sparse =csr_matrix(books_pivot)

#Visualizando o tipo de objeto
type(books_sparse)

#Importando algoritmo KNN e Scikit Learn
from sklearn.neighbors import NearestNeighbors

#Criando e treinando o modelo preditivo
Knn_Books = NearestNeighbors(algorithm='brute')
Knn_Books.fit(books_sparse)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Interação com o modelo
ultimo_livro_lido = input("Qual o nome do último livro que você leu? ")

# Tratamento de entrada
while ultimo_livro_lido not in books_pivot.index:
    print("Livro não encontrado. Por favor, insira um livro válido.")
    ultimo_livro_lido = input("Qual o nome do último livro que você leu? ")

# Obtendo sugestões do modelo
distancia, sugestoes = Knn_Books.kneighbors(books_pivot.loc[[ultimo_livro_lido]])

# Criando um DataFrame com os livros recomendados e suas distâncias
recomendacoes_df = pd.DataFrame({
    'Livro Recomendado': books_pivot.index[sugestoes[0]],
    'Distância': distancia[0]
})

# Removendo o livro indicado pelo usuário da lista de recomendações
recomendacoes_df = recomendacoes_df[recomendacoes_df['Livro Recomendado'] != ultimo_livro_lido]

# Ordenando o DataFrame pelo ranking (distância)
recomendacoes_df = recomendacoes_df.sort_values(by='Distância')

# Reduzindo as dimensões usando PCA para visualização
pca = PCA(n_components=2)
livros_pca = pca.fit_transform(books_pivot)

# Plotando o gráfico de dispersão com as recomendações e gráfico de barras ao lado
fig, axs = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [3, 2]})

# Scatter plot para todos os livros
axs[0].scatter(livros_pca[:, 0], livros_pca[:, 1], alpha=0.5, label='Outros Livros', color='lightgray', edgecolors='black', linewidths=0.5)

# Destacando o último livro lido
ultimo_livro_pca = pca.transform(books_pivot.loc[[ultimo_livro_lido]])
axs[0].scatter(ultimo_livro_pca[:, 0], ultimo_livro_pca[:, 1], color='red', marker='*', s=200, label='Último Livro Lido')

# Destacando os livros recomendados
recomendacoes_pca = pca.transform(books_pivot.loc[recomendacoes_df['Livro Recomendado']])
axs[0].scatter(recomendacoes_pca[:, 0], recomendacoes_pca[:, 1], color='blue', marker='o', s=100, label='Livros Recomendados')


# Adicionando gráfico de barras ao lado direito
axs[1].barh(recomendacoes_df['Livro Recomendado'], recomendacoes_df['Distância'], color='blue')
axs[1].set_xlabel('Distância')
axs[1].set_title('Distância dos Livros Recomendados')

# Ajustando o layout e adicionando elementos visuais
axs[0].set_title('Visualização do Espaço de Características com Recomendações KNN')
axs[0].set_xlabel('Principal Component 1')
axs[0].set_ylabel('Principal Component 2')
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Imprimindo a recomendação
print(f"Com base em sua última leitura, acredito que gostaria de ler os livros listados abaixo:")
print(recomendacoes_df[['Livro Recomendado', 'Distância']])
