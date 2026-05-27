# Classificador Computacional de Fake News sobre Arboviroses

Este repositório contém o desenvolvimento de um projeto acadêmico completo voltado à identificação de notícias falsas (*Fake*) e verídicas (*Fato*) associadas a epidemias de Dengue e Zika Vírus. O núcleo analítico do ecossistema baseia-se no algoritmo probabilístico **Naive Bayes** implementado em ambiente Python.

---

## 🛠️ Critérios de Avaliação Atendidos

1. **Criação e Organização da Base de Dados (10 Pontos):** Localizado em `/dataset/dataset_epidemias.csv`. Apresenta rotulação transparente binária (`0` para Fake, `1` para Fato), mapeamento cronológico da coleta (2025-2026), rastreio de links e explicitação de fontes regulatórias (Fiocruz, Ministério da Saúde) e agências checadoras (Lupa, Aos Fatos, Boatos.org).
2. **Implementação do Modelo de Machine Learning (10 Pontos):** Codificado de forma modular dividindo-se entre a higienização textual (`/src/pre_processamento.py`) e a orquestração de treino utilizando a biblioteca `scikit-learn` (`/src/modelo_naive_bayes.py`).
3. **Avaliação e Análise de Resultados (10 Pontos):** Métricas computadas automaticamente e centralizadas na pasta `/resultados`, exportando artefatos estatísticos visuais (`matriz_confusao.png`) e textuais (`relatorio_metrica.txt`).
4. **Apresentação e Demonstração da Solução (10 Pontos):** Métodos internos preparados para realizar predições em tempo real a partir de inputs crus fornecidos pelo usuário no terminal, calculando as probabilidades associadas à predição.

---

## 🚀 Instruções para Instalação e Execução

### 1. Preparação do Isolamento do Servidor (venv)
Para mitigar restrições de ambientes gerenciados externamente (PEP 668), crie e ative o ambiente virtual dentro da pasta raiz:

```bash
python3 -m venv venv
source venv/bin/activate
