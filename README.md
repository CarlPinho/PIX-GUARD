# PixGuard - Sistema de Detecção de Golpes em Transações Pix

## Resumo do Projeto

O PixGuard é um sistema avançado de detecção de golpes em transações Pix, desenvolvido para o Bradesco como parte de um trabalho acadêmico. O sistema utiliza um modelo de Inteligência Artificial para analisar transações e identificar possíveis fraudes com base em diversos fatores. A aplicação oferece uma interface amigável onde os usuários podem inserir dados de transações e receber uma classificação de risco imediata, com recomendações claras para aumentar a segurança.

O objetivo principal é fornecer uma ferramenta robusta e proativa que auxilie na proteção dos usuários do sistema Pix, minimizando perdas financeiras e aumentando a confiança nas transações digitais.

## Tecnologias Utilizadas

Este projeto foi desenvolvido utilizando uma combinação de linguagens e bibliotecas que otimizam a lógica de detecção e a experiência do usuário.

- **Python 3.x:**

  - **Função:** Linguagem principal para o desenvolvimento do modelo de Inteligência Artificial e o gerenciamento da persistência do histórico de transações.
  - **Bibliotecas de IA/Dados:**
    - `scikit-learn`: Utilizado para a implementação do algoritmo de classificação (RandomForestClassifier) e vetorização de texto (TfidfVectorizer).
    - `pandas`: Para manipulação e preparação dos dados sintéticos de treinamento.
    - `numpy`: Para operações numéricas e suporte a arrays.
    - `scipy.sparse`: Para lidar com matrizes esparsas, otimizando o desempenho do modelo.
    - `joblib`: Para serializar (salvar e carregar) o modelo de IA e o vetorizador, permitindo que o modelo treinado seja reutilizado.
  - **Persistência de Dados:** `json` para carregar dados sintéticos e salvar/carregar o histórico de transações (`historico_transacoes.json`).
  - **Outros:** `os`, `sys`, `webbrowser`, `random`, `datetime` para operações de sistema, manipulação de arquivos e geração de dados.

- **HTML5, CSS3 (Bootstrap 5) e JavaScript:**

  - **Função:** Desenvolvimento da interface de usuário (`pixguard_standalone.html`).
  - **HTML5:** Estruturação da página web.
  - **CSS3 (com Bootstrap 5):** Estilização responsiva e componentes visuais para uma experiência de usuário moderna e intuitiva.
  - **JavaScript:** Lógica de frontend para interação com o formulário, validação, e simulação da análise de transação (incluindo uma lógica de fallback baseada em regras embutida no próprio JavaScript, caso o backend não esteja disponível ou para validação inicial).
  - **LocalStorage:** Utilizado no frontend para persistência local do histórico de transações no navegador.

- **Estrutura de Dados Customizada:**
  - **Pilha (Stack):** Implementada manualmente em Python (classe `HistoryStack` em `pixguard_simples.py`) para gerenciar o histórico de transações. Esta implementação segue o princípio LIFO (Last In, First Out) e foi desenvolvida para atender ao requisito de "implementar a estrutura de dados sem utilizar bibliotecas da linguagem" de forma completa, manipulando os elementos diretamente.

---

**Justificativa da Escolha da Linguagem (Python & JavaScript em vez de Java):**

Embora o requisito original solicitasse a utilização de Java, o grupo optou por Python e JavaScript para o desenvolvimento deste projeto pelas seguintes razões:

- **Python:** Sua robustez e vasto ecossistema de bibliotecas para Machine Learning (`scikit-learn`, `pandas`, `numpy`) o tornam a escolha ideal para o desenvolvimento rápido e eficiente de modelos de IA, que é o cerne deste projeto. A facilidade de prototipagem e a clareza sintática do Python permitiram focar mais na lógica do modelo de detecção de golpes e no gerenciamento do histórico de forma eficiente.
- **JavaScript (com HTML/CSS):** Para a interface de usuário, a combinação de HTML, CSS e JavaScript é a tecnologia padrão para o desenvolvimento web front-end. Isso permite a criação de uma experiência interativa e visualmente rica diretamente no navegador, sem a necessidade de tecnologias adicionais para a camada de apresentação, tornando a solução mais acessível e fácil de usar como uma aplicação standalone.

Acreditamos que a escolha dessas tecnologias otimiza o desenvolvimento do sistema, aproveitando as forças de cada linguagem para entregar uma solução eficaz e com boa experiência de usuário, enquanto ainda demonstramos a compreensão e implementação de estruturas de dados fundamentais.

---

## Estrutura do Projeto

.
├── pixguard_simples.py # Script Python principal (lógica de IA, treinamento, e abertura do HTML)
├── pixguard_standalone.html # Interface do usuário (contém HTML, CSS e JavaScript)
├── bradesco_logo.png # Imagem do logo do Bradesco (necessária para o HTML)
├── faculdade_logo.png # Imagem do logo da Faculdade (necessária para o HTML)
├── data/ # Diretório criado automaticamente para dados
│ ├── dados_transacoes.json # Dados sintéticos gerados para treinamento
│ └── historico_transacoes.json # Histórico de análises (gerenciado pela pilha)
└── models/ # Diretório criado automaticamente para modelos treinados
├── modelo_pixguard.pkl # Modelo Random Forest serializado
└── vetor_tfidf_pixguard.pkl # Vetorizador TF-IDF serializado
└── README.md # Este arquivo

---

## Detalhes Técnicos

### Modelo de IA (Python)

- **Algoritmo**: `RandomForestClassifier` do `scikit-learn`.
- **Características de Análise**:
  - Valor da transação.
  - Indicação se a chave Pix é nova (`chave_nova`).
  - Análise de texto da mensagem via TF-IDF (`TfidfVectorizer`).
  - (Campos opcionais da interface não são diretamente usados no modelo de IA do Python, mas são considerados na análise heurística do JavaScript).
- **Geração de Dados Sintéticos**: O script Python gera um dataset sintético balanceado para treinamento do modelo.
- **Persistência do Modelo**: O modelo treinado e o vetorizador são salvos em arquivos `.pkl` para recarregamento rápido.

## Integrantes do Grupo

- [**Carlos Emanuel da Silva Pinho**] - [1252228970]
- [**João Gabriel de Araújo Diniz**] - [12522198775]
- [**Tiago Gonçalves de Almeida**] - [12523117709]

## LinkedIn dos Integrantes

- [João Gabriel Diniz](https://www.linkedin.com/in/joaogabrieldiniz)
- [Carlos Pinho](https://www.linkedin.com/in/carlospinho0101)
- [Tiago Almeida](https://www.linkedin.com/in/tiagoalmeida-ti)
