import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ingestão das funções modulares internas
from pre_processamento import limpar_texto, obter_stopwords_portugues

def executar_pipeline_ml():
    # Caminhos de arquivos baseados na raiz do projeto
    caminho_csv = os.path.join('..', 'dataset', 'dataset_epidemias.csv')
    diretorio_resultados = os.path.join('..', 'resultados')
    os.makedirs(diretorio_resultados, exist_ok=True)
    
    print("[INFO] Carregando a base de dados estruturada...")
    df = pd.read_csv(caminho_csv)
    
    # Concatenação do título e texto para extrair maior contexto semântico
    df['conteudo_completo'] = df['titulo'] + " " + df['texto']
    
    print("[INFO] Executando rotina de pré-processamento dos textos...")
    df['conteudo_limpo'] = df['conteudo_completo'].apply(limpar_texto)
    
    # Configuração do Vetorizador Incremental (Hashing Vectorizer)
    # Suporta evolução contínua do vocabulário sem quebras estruturais
    stopwords = obter_stopwords_portugues()
    vetorizador = HashingVectorizer(n_features=2**14, alternate_sign=False, stop_words=stopwords)
    
    X = vetorizador.transform(df['conteudo_limpo'])
    y = df['classificacao']
    
    # Separação clássica em treino (75%) e teste (25%)
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    print("[INFO] Iniciando treinamento incremental do algoritmo Naive Bayes...")
    modelo = MultinomialNB()
    # Uso do partial_fit inicializa a capacidade de aprendizado online contínuo
    modelo.partial_fit(X_treino, y_treino, classes=[0, 1])
    
    # Predições e Validação
    y_pred = modelo.predict(X_teste)
    
    # Extração de Métricas Matemáticas
    acuracia = accuracy_score(y_teste, y_pred)
    matriz = confusion_matrix(y_teste, y_pred)
    relatorio_texto = classification_report(y_teste, y_pred, target_names=['Fake (0)', 'Fato (1)'])
    
    print("[INFO] Exportando relatórios e métricas de desempenho...")
    
    # 1. Salvar o arquivo txt de métricas de avaliação
    caminho_relatorio = os.path.join(diretorio_resultados, 'relatorio_metrica.txt')
    with open(caminho_relatorio, 'w', encoding='utf-8') as f:
        f.write("========================================================\n")
        f.write("      RELATÓRIO DE AVALIAÇÃO ACADÊMICA DO MODELO        \n")
        f.write("========================================================\n")
        f.write(f"Acurácia Geral Obtida: {acuracia * 100:.2f}%\n\n")
        f.write("Relatório Detalhado por Classe:\n")
        f.write(relatorio_texto)
        f.write("\nMatriz de Confusão Numérica:\n")
        f.write(str(matriz))
        
    # 2. Renderizar e exportar a imagem da Matriz de Confusão
    plt.figure(figsize=(6, 4))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Previsto Fake', 'Previsto Fato'], 
                yticklabels=['Real Fake', 'Real Fato'])
    plt.title('Matriz de Confusão - Classificador de Arboviroses')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Predita')
    plt.tight_layout()
    plt.savefig(os.path.join(diretorio_resultados, 'matriz_confusao.png'), dpi=300)
    plt.close()
    
    print(f"[SUCESSO] Pipeline finalizado. Arquivos gerados na pasta '/resultados'.")

if __name__ == "__main__":
    # Garante que a execução se baseie no diretório correto do script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    executar_pipeline_ml()
