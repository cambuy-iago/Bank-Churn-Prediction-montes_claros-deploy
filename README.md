# 🏦 Predição de Churn em Cartões de Crédito – Banco Mercantil (Projeto Aplicado MBA)

Este repositório contém o desenvolvimento completo de um **sistema de predição de churn de clientes de cartão de crédito**, utilizando **Machine Learning (LightGBM)** e um **webapp interativo em Streamlit** para suporte às áreas de **CRM, Risco e Negócios**.

O projeto foi desenvolvido como **Trabalho de Conclusão de Curso (Projeto Aplicado)** do MBA em Ciência de Dados, com foco em um cenário inspirado no **Banco Mercantil**.

---

## 🎯 1. Objetivo de Negócio

Clientes que cancelam seus cartões (**churn**) geram:

- Perda de receita recorrente;
- Desperdício de custo de aquisição (CAC);
- Redução de cross-sell e up-sell;
- Aumento do esforço comercial para repor a carteira.

> **Objetivo:** construir um **modelo preditivo de churn** capaz de identificar, com alta precisão, quais clientes estão em maior risco de cancelar o cartão, permitindo **ações de retenção direcionadas**.

---

## 📊 2. Dados Utilizados

- Base: `data/BankChurners.csv`  
- Origem: conjunto de dados público inspirado em clientes de cartão de crédito.  
- Registros: **10.127 clientes**  
- Target:
  - `Attrition_Flag` → convertido em variável binária `Attrition` (0 = cliente ativo, 1 = churn)

### Principais grupos de variáveis

- **Demográficas**: idade, estado civil, escolaridade, faixa de renda;
- **Relacionamento**: tempo de casa, quantidade de produtos, contatos com o banco;
- **Crédito**: limite, saldo rotativo, utilização do limite;
- **Transações (12 meses)**: quantidade, valor total, variação Q4/Q1.

### Feature Engineering (exemplos)

Foram criadas variáveis derivadas para capturar melhor o comportamento do cliente, entre elas:

- `Ticket_Medio` – valor médio por transação;  
- `Transacoes_por_Mes` – frequência mensal de uso;  
- `Gasto_Medio_Mensal` – intensidade de consumo;  
- `Rotativo_Ratio` – proporção de limite comprometida no rotativo;  
- `Disponibilidade_Relativa` – % do limite ainda disponível;  
- `Score_Relacionamento` – proxy de profundidade de relacionamento;  
- `LTV_Proxy` – gasto médio mensal × tempo de casa;  
- `Caiu_Valor` e `Caiu_Transacoes` – flags de queda de uso;  
- `Faixa_Idade` e `Renda_Class` – variáveis categóricas de segmentação.

---

## 🧠 3. Metodologia e Pipeline

A solução foi estruturada seguindo uma abordagem próxima ao **CRISP-DM**:

1. **Entendimento de Negócio**  
   - Mapeamento do impacto do churn em cartões.  
   - Definição de métricas: **ROC AUC**, **Recall da classe churn**, **Precision da classe churn**, **F1**.

2. **Entendimento dos Dados / EDA**  
   - Análises em `notebooks/01_Análise_Exploratória.ipynb` e `notebooks/01_eda_base_tratada.ipynb`;  
   - Verificação de:
     - Qualidade de dados (nulos, duplicados, colunas constantes);
     - Distribuição das classes (≈ **16% churn**, 84% ativos);
     - Outliers numéricos (ex.: `Credit_Limit`, `Total_Trans_Amt`, etc.);
     - Correlações com a variável `Attrition`;
     - Segmentação com **PCA 2D/3D** e **KMeans (3 clusters)**.

3. **Preparação dos Dados**  
   - Script principal: `src/01_eda_base_tratada.py`;  
   - Criação da `data/base_tratada.csv` e `data/base_modelagem.csv`;  
   - Seleção de **12 features numéricas principais** para o modelo produtivo;  
   - Tratamento de outliers (análise em `outlier_analysis_summary.csv`);  
   - Separação de variáveis numéricas e categóricas.

4. **Modelagem**  
   - Script principal: `src/02_model_training.py`;  
   - Modelos avaliados:
     - Regressão Logística (baseline);
     - Random Forest;
     - XGBoost;
     - **LightGBM (modelo final)**.
   - Estratégias:
     - **Desbalanceamento:** uso de `class_weight='balanced'` (em vez de SMOTE), para evitar overfitting;
     - Validação: holdout + validação cruzada estratificada;
     - Versão final usando **12 features numéricas principais**, mais simples e mais estável para produção.

5. **Avaliação e Comparação de Modelos**  
   - Métricas consolidadas em: `reports/text/metrics_modelos.csv`;  
   - Curvas ROC e matrizes de confusão em `reports/figures/`.

6. **Implantação (Webapp)**  
   - App em **Streamlit**: `src/app_churn_streamlit.py`;  
   - Suporte a:
     - Análise exploratória;
     - Visão executiva das métricas;
     - Simulador individual de clientes;
     - Análise em lote via upload de CSV.

---

## 📈 4. Resultados Principais

### 4.1 Comparação de Modelos (resumo)

Fonte: `reports/text/metrics_modelos.csv`

| Modelo               | Accuracy | ROC AUC | Precision (churn) | Recall (churn) | F1 (churn) |
|----------------------|----------|---------|--------------------|----------------|------------|
| Regressão Logística  | 0.85     | 0.92    | 0.53               | 0.82           | 0.64       |
| **LightGBM (final)** | **0.97** | **0.99**| **0.93**           | **0.87**       | **0.90**   |

### 4.2 Modelo Final – LightGBM

Resumo (exemplo de execução):

- **AUC:** ~**0.99**  
- **Acurácia:** ~**0.97**  
- **Precision (classe churn):** ~**0.93**  
- **Recall (classe churn):** ~**0.87**  
- **F1 (classe churn):** ~**0.90**

> Interpretação: o modelo consegue **identificar a maioria dos clientes que irão churnar**, com **baixo nível de falsos positivos**, o que é essencial para campanhas de retenção com custo controlado.

### 4.3 Versionamento de Modelos

Arquivo: `models/versions_log.csv`

Exemplo de registro:

- `model_lgbm_v1.pkl`  
- Algoritmo: `lgbm`  
- Versão: `v1`  
- AUC: `0.9846` (validação)  
- Notas: _"12-feature baseline with class_weight='balanced'"_

---

## 🧩 5. Arquitetura do Projeto

```bash
Bank-Churn-Prediction-montes_claros/
├── data/
│   ├── BankChurners.csv
│   ├── base_tratada.csv
│   ├── base_modelagem.csv
│   └── features_modelagem.json
├── eda_results/              # Saídas automatizadas de EDA
├── models/
│   ├── model_final.pkl       # Modelo final em produção
│   ├── model_lgbm_v1.pkl     # Versão anterior
│   └── versions_log.csv      # Log de versões e métricas
├── notebooks/
│   ├── 01_eda_base_tratada.ipynb
│   ├── 02_model_training.ipynb
│   └── ...                   # Outros notebooks exploratórios
├── reports/
│   ├── figures/              # PNGs de ROC, matriz de confusão, SHAP, etc.
│   └── text/                 # Relatórios de métricas, AUC, classification_report
├── src/
│   ├── config.py             # Configuração de caminhos
│   ├── eda.py                # Funções auxiliares de EDA
│   ├── features.py           # Feature engineering
│   ├── 01_eda_base_tratada.py
│   ├── 02_model_training.py
│   ├── train_lgbm.py / train_rf.py / train_xgb.py
│   ├── model_versioning.py   # Registro de versões
│   ├── final_model.py        # Funções de carga do modelo
│   └── app_churn_streamlit.py
├── webapp/
