import os
import sys
import json
import webbrowser
import random
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import scipy.sparse
import joblib
from datetime import datetime

class HistoryStack:

    def __init__(self, path='historico_transacoes.json'):

        self.path = path
        self.items = self._load()
        # top_index aponta para o índice do último elemento adicionado (o "topo").
        # -1 indica que a pilha está vazia.
        self.top_index = len(self.items) - 1 

    def _load(self):
        """
        Carrega o histórico de transações do arquivo JSON.
        Se o arquivo não existir ou estiver corrompido, retorna uma lista vazia.
        """
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Garante que o conteúdo carregado seja uma lista
                    if not isinstance(data, list):
                        return []
                    return data
            except json.JSONDecodeError:
                # Retorna lista vazia se o JSON estiver malformado
                return []
        return []

    def _save(self):
        """
        Salva o histórico atual (self.items) no arquivo JSON.
        """
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.items, f, ensure_ascii=False, indent=4)

    def push(self, item):
        """
        Adiciona um novo item ao "topo" da pilha.
        """
        self.items.append(item) # Adiciona ao final da lista, que é o topo da pilha
        self.top_index = len(self.items) - 1 # Atualiza o índice do topo
        self._save() # Salva o estado da pilha após a adição

    def pop(self):
        """
        Remove e retorna o item do "topo" da pilha (o último item adicionado).
        A remoção é feita manualmente, sem usar o método list.pop() nativo.
        """
        if self.top_index == -1:
            # A pilha está vazia
            return None

        # Pega o item do topo antes de removê-lo
        removed_item = self.items[self.top_index]
        
        # Cria uma nova lista contendo todos os elementos, exceto o último.
        # Isso simula a remoção do topo de forma manual.
        self.items = self.items[:self.top_index]
        
        # Atualiza o índice do topo para a nova pilha (ou -1 se estiver vazia)
        self.top_index = len(self.items) - 1 

        self._save() # Salva o estado da pilha após a remoção
        return removed_item

    def get_all(self):
        """
        Retorna uma cópia de todos os itens atualmente na pilha.
        
        Returns:
            list: Uma cópia da lista de itens da pilha.
        """
        return list(self.items)
    
# Configurações do projeto
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_DADOS = os.path.join(DIRETORIO_ATUAL, "data")
DIRETORIO_MODELOS = os.path.join(DIRETORIO_ATUAL, "models")

# Garantir que os diretórios existam
os.makedirs(DIRETORIO_DADOS, exist_ok=True)
os.makedirs(DIRETORIO_MODELOS, exist_ok=True)

# Caminho para o arquivo HTML
ARQUIVO_HTML = os.path.join(DIRETORIO_ATUAL, "pixguard_standalone.html")

# Caminho para as imagens
IMAGEM_BRADESCO = os.path.join(DIRETORIO_ATUAL, "bradesco_logo.png")
IMAGEM_FACULDADE = os.path.join(DIRETORIO_ATUAL, "faculdade_logo.png")

# Verificar se as imagens existem
if not os.path.exists(IMAGEM_BRADESCO):
    print(f"AVISO: Imagem do Bradesco não encontrada em {IMAGEM_BRADESCO}")
    
if not os.path.exists(IMAGEM_FACULDADE):
    print(f"AVISO: Imagem da faculdade não encontrada em {IMAGEM_FACULDADE}")

# Classe para o modelo de IA
class ModeloPixGuard:
    def __init__(self):
        self.historico = HistoryStack(os.path.join(DIRETORIO_DADOS, "historico_transacoes.json"))
        self.modelo = None
        self.vetorizador = None
        
        modelo_path = os.path.join(DIRETORIO_MODELOS, "modelo_pixguard.pkl")
        vetor_path = os.path.join(DIRETORIO_MODELOS, "vetor_tfidf_pixguard.pkl")
        
        if os.path.exists(modelo_path) and os.path.exists(vetor_path):
            print("Carregando modelo existente...")
            self.modelo = joblib.load(modelo_path)
            self.vetorizador = joblib.load(vetor_path)
        else:
            print("Treinando novo modelo...")
            self._treinar_modelo()

    def _treinar_modelo(self):
        """Treina o modelo com separação adequada dos dados para evitar overfitting"""
        print("Gerando dados sintéticos...")
        dados = self._gerar_dados_sinteticos()
        
        # Preparar features
        X_valor = dados[['valor', 'chave_nova']]
        X_mensagem = dados['mensagem']
        y = dados['rotulo']
        
        # ===== SEPARAÇÃO DOS DADOS PARA EVITAR OVERFITTING =====
        print("Separando dados em treino e teste...")
        X_valor_train, X_valor_test, X_msg_train, X_msg_test, y_train, y_test = train_test_split(
            X_valor, X_mensagem, y, 
            test_size=0.2,           # 20% para teste
            random_state=42,         # Reprodutibilidade
            stratify=y              # Manter proporção das classes
        )
        
        print(f"Dados de treino: {len(X_valor_train)} amostras")
        print(f"Dados de teste: {len(X_valor_test)} amostras")
        
        # Treinar vetorizador APENAS com dados de treino
        self.vetorizador = TfidfVectorizer()
        X_msg_train_tfidf = self.vetorizador.fit_transform(X_msg_train)
        
        # Preparar dados de treino
        X_numerico_train_sparse = scipy.sparse.csr_matrix(X_valor_train.values)
        X_train_final = scipy.sparse.hstack((X_numerico_train_sparse, X_msg_train_tfidf))
        
        # Treinar modelo
        print("Treinando modelo Random Forest...")
        self.modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        self.modelo.fit(X_train_final, y_train)
        
        # ===== AVALIAÇÃO NO CONJUNTO DE TESTE =====
        print("Avaliando modelo no conjunto de teste...")
        
        # Transformar dados de teste (SEM fit, apenas transform!)
        X_msg_test_tfidf = self.vetorizador.transform(X_msg_test)
        X_numerico_test_sparse = scipy.sparse.csr_matrix(X_valor_test.values)
        X_test_final = scipy.sparse.hstack((X_numerico_test_sparse, X_msg_test_tfidf))
        
        # Fazer predições no conjunto de teste
        y_pred = self.modelo.predict(X_test_final)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n📊 RESULTADOS DA AVALIAÇÃO:")
        print(f"Acurácia no conjunto de teste: {accuracy:.3f}")
        print(f"\nRelatório detalhado:")
        print(classification_report(y_test, y_pred))
        
        # Salvar modelo
        joblib.dump(self.modelo, os.path.join(DIRETORIO_MODELOS, "modelo_pixguard.pkl"))
        joblib.dump(self.vetorizador, os.path.join(DIRETORIO_MODELOS, "vetor_tfidf_pixguard.pkl"))
        print("✅ Modelo treinado e salvo com sucesso!")

    def _gerar_dados_sinteticos(self):
        mensagens_normais = [
            "Pagamento da pizzaria", "Presente de aniversário", "Transferência para conta própria",
            "Mensalidade da escola", "Aluguel do mês", "Compra no mercado", "Pagamento de academia",
            "Doação para ONG", "Compra online", "Pagamento de serviço de limpeza", "Conta de luz",
            "Conta de água", "Conta de internet", "Pagamento de streaming", "Presente para amigo",
            "Divisão de conta do restaurante", "Pagamento de curso", "Compra de livro",
            "Pagamento de aplicativo", "Transferência para familiar"
        ]
        mensagens_golpes = [
            "Urgente, estou no hospital", "Ajuda, estou preso", "Pague agora para liberar o prêmio",
            "Deposite para regularizar dívida", "Multa judicial, pagamento imediato",
            "Socorro, preciso pagar cirurgia", "Empréstimo aprovado, envie taxa",
            "Atualização de boleto, novo código", "Problema na conta, transfira já",
            "Parente em perigo, mande dinheiro", "Seu cartão foi bloqueado, regularize",
            "Ganhou sorteio, pague taxa de liberação", "Dívida em seu nome, regularize agora",
            "Imposto atrasado, pague com urgência", "Conta será bloqueada, transfira agora",
            "Seu benefício será cancelado, confirme dados", "Promoção exclusiva, pague reserva",
            "Problema com entrega, pague taxa", "Seu cadastro expirou, renove agora",
            "Sua conta foi invadida, transfira saldo"
        ]
        transacoes = []
        for _ in range(140):
            transacoes.append({"valor": random.randint(10, 2000), "chave_nova": random.choice([0, 0, 0, 1]),
                               "mensagem": random.choice(mensagens_normais), "rotulo": "normal"})
        for _ in range(60):
            transacoes.append({"valor": random.randint(500, 10000), "chave_nova": random.choice([0, 1, 1, 1]),
                               "mensagem": random.choice(mensagens_golpes), "rotulo": "golpe"})
        random.shuffle(transacoes)
        df = pd.DataFrame(transacoes)
        df.to_json(os.path.join(DIRETORIO_DADOS, "dados_transacoes.json"), orient="records", force_ascii=False, indent=4)
        return df

    def analisar_transacao(self, dados):
        if self.modelo is None or self.vetorizador is None:
            print("ERRO: Modelo não carregado!")
            return {"predicao": "erro", "probabilidade_golpe": 0.5, "risco": "Médio"}
        try:
            entrada_numerica = [[dados['valor'], int(dados['chave_nova'])]]
            entrada_numerica_sparse = scipy.sparse.csr_matrix(entrada_numerica)
            entrada_mensagem_tfidf = self.vetorizador.transform([dados['mensagem']])
            entrada_final = scipy.sparse.hstack((entrada_numerica_sparse, entrada_mensagem_tfidf))
            predicao = self.modelo.predict(entrada_final)[0]
            probabilidade = self.modelo.predict_proba(entrada_final)[0]
            classes = self.modelo.classes_
            indice_golpe = np.where(classes == 'golpe')[0][0]
            probabilidade_golpe = probabilidade[indice_golpe]
            risco = "Alto" if probabilidade_golpe > 0.7 else "Médio" if probabilidade_golpe > 0.4 else "Baixo"
            self.historico.push({
                'valor': dados['valor'],
                'chave_nova': dados['chave_nova'],
                'mensagem': dados['mensagem'],
                'predicao': predicao,
                'risco': risco,
                'probabilidade': round(float(probabilidade_golpe), 2)
            })
            return {"predicao": predicao, "probabilidade_golpe": float(probabilidade_golpe), "risco": risco}
        except Exception as e:
            print(f"ERRO na análise: {str(e)}")
            return self._analisar_por_regras(dados)

    def _analisar_por_regras(self, dados):
        palavras_suspeitas = ['urgente', 'emergência', 'socorro', 'ajuda', 'hospital', 'acidente', 'prêmio', 'ganhou',
                              'multa', 'judicial', 'bloqueado', 'regularizar', 'dívida', 'imposto', 'taxa',
                              'devolver', 'confirmar', 'dados', 'senha', 'código']
        fatores_risco = 0
        if dados['valor'] > 3000:
            fatores_risco += 2
        elif dados['valor'] > 1000:
            fatores_risco += 1
        if dados['chave_nova'] == 1:
            fatores_risco += 2
        mensagem_lower = dados['mensagem'].lower()
        for palavra in palavras_suspeitas:
            if palavra in mensagem_lower:
                fatores_risco += 1
        max_fatores = 8
        probabilidade_golpe = max(0.05, min(0.95, fatores_risco / max_fatores))
        predicao = 'golpe' if probabilidade_golpe > 0.7 else 'golpe' if probabilidade_golpe > 0.55 else 'normal'
        risco = 'Alto' if probabilidade_golpe > 0.7 else 'Médio' if probabilidade_golpe > 0.4 else 'Baixo'
        return {"predicao": predicao, "probabilidade_golpe": probabilidade_golpe, "risco": risco}


def main():
    print("=" * 60)
    print("PixGuard - Sistema de Detecção de Golpes em Transações Pix")
    print("=" * 60)
    print("\nIniciando sistema...\n")
    try:
        print("Inicializando modelo de IA...")
        modelo = ModeloPixGuard()
        if not os.path.exists(ARQUIVO_HTML):
            print(f"ERRO: Arquivo HTML não encontrado em {ARQUIVO_HTML}")
            return
        print("\nAbrindo navegador...")
        webbrowser.open('file://' + os.path.abspath(ARQUIVO_HTML))
        print("\n" + "=" * 60)
        print("PixGuard iniciado com sucesso!")
        print("=" * 60)
        print("\nO sistema está rodando no seu navegador.")
        print("Para analisar transações, preencha o formulário na interface.")
        print("\nPressione Ctrl+C para encerrar o sistema.")

        while True:
            comando = input("\nDigite 'sair' para encerrar: ")
            if comando.lower() == 'sair':
                break

    except KeyboardInterrupt:
        print("\nEncerrando PixGuard...")
    except Exception as e:
        print(f"\nErro ao iniciar PixGuard: {str(e)}")
    finally:
        print("\nPixGuard encerrado.")

if __name__ == "__main__":
    main()

