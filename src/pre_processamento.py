import re

def limpar_texto(texto):
    """
    Executa a limpeza e padronização estrutural do texto bruto.
    Aplica conversão para minúsculas e remoção de caracteres não alfabéticos.
    """
    if not isinstance(texto, str):
        return ""
    
    # Conversão para minúsculas (Garante uniformidade de features)
    texto_processado = texto.lower()
    
    # Remoção de URLs/Links
    texto_processado = re.sub(r'https?://\s+|www\.\s+', '', texto_processado)
    
    # Remoção de pontuações, numerais e caracteres especiais
    texto_processado = re.sub(r'[^\w\s]', '', texto_processado)
    texto_processado = re.sub(r'\d+', '', texto_processado)
    
    # Eliminação de espaçamentos duplos/indesejados
    texto_processado = " ".join(texto_processado.split())
    
    return texto_processado

def obter_stopwords_portugues():
    """
    Retorna uma lista estática de stopwords em português para evitar a necessidade 
    de downloads externos constantes do NLTK em servidores isolados.
    """
    return [
        'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 
        'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 
        'foi', 'ao', 'ele', 'das', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 
        'era', 'depois', 'sem', 'mesmo', 'aos', 'ter', 'seus', 'quem', 'nas', 'meu'
    ]
