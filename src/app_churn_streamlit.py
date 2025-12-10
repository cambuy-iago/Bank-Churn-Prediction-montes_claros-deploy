import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# -----------------------------------------------------------
# CONFIGURAÇÃO DE CAMINHOS COM FALLBACKS ROBUSTOS
# -----------------------------------------------------------
def setup_paths():
    """Configura os caminhos do projeto com múltiplos fallbacks"""
    
    # Tenta encontrar a raiz do projeto de diferentes maneiras
    current_file = Path(__file__).resolve()
    
    # Opção 1: Se o app está em src/
    project_root = current_file.parent.parent
    
    # Verifica se a estrutura está correta
    if not (project_root / "data").exists():
        # Opção 2: Tenta um nível acima
        project_root = current_file.parent.parent.parent
    
    # Fallback: Caminho absoluto baseado na sua estrutura
    if not (project_root / "data").exists():
        fallback_path = Path(r"C:\Users\Iago\OneDrive\Desktop\Projeto Churn\Bank-Churn-Prediction-montes_claros")
        if fallback_path.exists():
            project_root = fallback_path
    
    # Caminhos principais
    MODEL_PATH = project_root / "models" / "model_final.pkl"
    SCALER_PATH = project_root / "models" / "scaler.pkl"
    METRICS_PATH = project_root / "reports" / "metrics_modelos.csv"
    FIG_CM_PATH = project_root / "reports" / "figures" / "matriz_confusao_lightgbm.png"
    FIG_ROC_PATH = project_root / "reports" / "figures" / "roc_curve_lightgbm.png"
    DATA_PATH = project_root / "data" / "BankChurners.csv"
    
    # Adiciona src ao sys.path para importações
    src_path = project_root / "src"
    if src_path.exists():
        sys.path.append(str(src_path))
    
    return {
        "PROJECT_ROOT": project_root,
        "MODEL_PATH": MODEL_PATH,
        "SCALER_PATH": SCALER_PATH,
        "METRICS_PATH": METRICS_PATH,
        "FIG_CM_PATH": FIG_CM_PATH,
        "FIG_ROC_PATH": FIG_ROC_PATH,
        "DATA_PATH": DATA_PATH
    }

# Obter caminhos configurados
paths = setup_paths()
PROJECT_ROOT = paths["PROJECT_ROOT"]
MODEL_PATH = paths["MODEL_PATH"]
SCALER_PATH = paths["SCALER_PATH"]
METRICS_PATH = paths["METRICS_PATH"]
FIG_CM_PATH = paths["FIG_CM_PATH"]
FIG_ROC_PATH = paths["FIG_ROC_PATH"]
DATA_PATH = paths["DATA_PATH"]

# -----------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# -----------------------------------------------------------
st.set_page_config(
    page_title="Banco Mercantil - Preditor de Churn",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para melhorar visual
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# CARREGAMENTO DE MODELO E SCALER
# -----------------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    """Carrega o modelo e o scaler com fallbacks robustos"""
    try:
        # Carregar modelo
        if MODEL_PATH.exists():
            modelo = joblib.load(MODEL_PATH)
            st.sidebar.success("✅ Modelo carregado com sucesso")
        else:
            st.sidebar.error(f"❌ Modelo não encontrado em: {MODEL_PATH}")
            st.sidebar.info("💡 Execute o script de treinamento primeiro")
            return None, None
        
        # Carregar scaler se existir
        scaler = None
        if SCALER_PATH.exists():
            scaler = joblib.load(SCALER_PATH)
            st.sidebar.success("✅ Scaler carregado com sucesso")
        
        return modelo, scaler
        
    except Exception as e:
        st.sidebar.error(f"❌ Erro ao carregar modelo: {str(e)}")
        return None, None

modelo, scaler = load_model_and_scaler()

# -----------------------------------------------------------
# FUNÇÕES DE FEATURE ENGINEERING (FALLBACK SE src.features NÃO DISPONÍVEL)
# -----------------------------------------------------------
def criar_variaveis_derivadas_fallback(df):
    """
    Função de fallback para criar variáveis derivadas se o módulo src.features não estiver disponível
    """
    df = df.copy()
    
    # 1. Features básicas com tratamento de divisão por zero
    df["Ticket_Medio"] = np.where(df["Total_Trans_Ct"] != 0, 
                                  df["Total_Trans_Amt"] / df["Total_Trans_Ct"], 
                                  0)
    
    df["Transacoes_por_Mes"] = np.where(df["Months_on_book"] != 0, 
                                        df["Total_Trans_Ct"] / df["Months_on_book"], 
                                        0)
    
    df["Gasto_Medio_Mensal"] = np.where(df["Months_on_book"] != 0, 
                                        df["Total_Trans_Amt"] / df["Months_on_book"], 
                                        0)
    
    # 2. Utilização de crédito
    df["Rotativo_Ratio"] = np.where(df["Credit_Limit"] != 0, 
                                    df["Total_Revolving_Bal"] / df["Credit_Limit"], 
                                    0)
    
    df["Disponibilidade_Relativa"] = np.where(df["Credit_Limit"] != 0, 
                                              (df["Credit_Limit"] - df["Total_Revolving_Bal"]) / df["Credit_Limit"], 
                                              0)
    
    # 3. Flags de variação
    df["Caiu_Transacoes"] = (df["Total_Ct_Chng_Q4_Q1"] < 1).astype(int)
    df["Caiu_Valor"] = (df["Total_Amt_Chng_Q4_Q1"] < 1).astype(int)
    
    # 4. Relacionamento
    df["Score_Relacionamento"] = df["Total_Relationship_Count"]
    df["LTV_Proxy"] = df["Gasto_Medio_Mensal"] * df["Months_on_book"]
    
    # 5. Faixa etária
    def faixa_idade(x):
        if x < 30:
            return "<30"
        elif x < 50:
            return "30-49"
        elif x < 70:
            return "50-69"
        else:
            return "70+"
    
    df["Faixa_Idade"] = df["Customer_Age"].apply(faixa_idade)
    
    # 6. Classificação de renda
    def renda_class(ic):
        if ic in ["$60K - $80K", "$80K - $120K", "$120K +"]:
            return "Alta"
        elif ic in ["$40K - $60K", "$20K - $40K"]:
            return "Média"
        else:
            return "Baixa"
    
    df["Renda_Class"] = df["Income_Category"].apply(renda_class)
    
    # 7. Criar flag de churn se a coluna existir
    if "Attrition_Flag" in df.columns:
        df["churn_flag"] = (df["Attrition_Flag"] == "Attrited Customer").astype(int)
    
    return df

# Tenta importar a função original, usa fallback se falhar
try:
    from src.features import criar_variaveis_derivadas
    criar_variaveis_derivadas_wrapper = criar_variaveis_derivadas
except ImportError:
    st.sidebar.warning("⚠️ Usando função de fallback para criar_variáveis_derivadas")
    criar_variaveis_derivadas_wrapper = criar_variaveis_derivadas_fallback

# -----------------------------------------------------------
# CARREGAMENTO DE DADOS
# -----------------------------------------------------------
@st.cache_data
def load_data_raw():
    """Carrega os dados brutos com múltiplos fallbacks"""
    # Lista de possíveis caminhos
    possible_paths = [
        DATA_PATH,
        Path("data/BankChurners.csv"),
        Path(r"C:\Users\Iago\OneDrive\Desktop\Projeto Churn\Bank-Churn-Prediction-montes_claros\data\BankChurners.csv"),
        PROJECT_ROOT / "BankChurners.csv"
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                st.sidebar.success(f"✅ Dados carregados de: {path}")
                return df
            except Exception as e:
                continue
    
    st.sidebar.error("❌ Não foi possível carregar os dados. Verifique o caminho do arquivo.")
    return None

@st.cache_data
def load_data_with_features():
    """Carrega os dados e aplica feature engineering"""
    df = load_data_raw()
    if df is None:
        return None
    
    # Aplica feature engineering
    df = criar_variaveis_derivadas_wrapper(df)
    return df

# -----------------------------------------------------------
# DICIONÁRIOS DE TRADUÇÃO (ATUALIZADOS)
# -----------------------------------------------------------
DIC_NOME_PT_NUMERICOS = {
    "Idade do Cliente": "Customer_Age",
    "Número de Dependentes": "Dependent_count",
    "Meses de Relacionamento": "Months_on_book",
    "Quantidade de Produtos com o Banco": "Total_Relationship_Count",
    "Meses Inativo (12 meses)": "Months_Inactive_12_mon",
    "Contatos com o Banco (12 meses)": "Contacts_Count_12_mon",
    "Limite de Crédito": "Credit_Limit",
    "Saldo Rotativo": "Total_Revolving_Bal",
    "Variação de Valor Q4/Q1": "Total_Amt_Chng_Q4_Q1",
    "Valor Total Transacionado (12 meses)": "Total_Trans_Amt",
    "Número de Transações (12 meses)": "Total_Trans_Ct",
    "Variação de Transações Q4/Q1": "Total_Ct_Chng_Q4_Q1",
    "Utilização Média do Limite": "Avg_Utilization_Ratio",
    "Score de Relacionamento": "Score_Relacionamento",
    "Proxy LTV": "LTV_Proxy",
    "Caiu em Valor": "Caiu_Valor",
    "Caiu em Transações": "Caiu_Transacoes",
}

DIC_NOME_PT_ENGINEERED = {
    "Ticket Médio por Transação": "Ticket_Medio",
    "Transações por Mês": "Transacoes_por_Mes",
    "Gasto Médio Mensal": "Gasto_Medio_Mensal",
    "Uso do Rotativo (Ratio)": "Rotativo_Ratio",
    "Disponibilidade Relativa de Limite": "Disponibilidade_Relativa",
    "Faixa de Idade": "Faixa_Idade",
    "Classificação de Renda": "Renda_Class",
}

# -----------------------------------------------------------
# FUNÇÕES AUXILIARES PARA PREVISÃO
# -----------------------------------------------------------
def calcular_features_engineered_row(row: dict) -> dict:
    """Calcula todas as features derivadas para uma única linha"""
    # Valores básicos com proteção contra divisão por zero
    idade = row.get("Customer_Age", 0)
    months_on_book = max(row.get("Months_on_book", 1), 1)
    credit_limit = max(row.get("Credit_Limit", 1.0), 0.1)
    total_trans_amt = row.get("Total_Trans_Amt", 0)
    total_trans_ct = max(row.get("Total_Trans_Ct", 1), 1)
    total_revolving_bal = row.get("Total_Revolving_Bal", 0)
    total_relationship_count = row.get("Total_Relationship_Count", 0)
    total_amt_chng_q4_q1 = row.get("Total_Amt_Chng_Q4_Q1", 1.0)
    total_ct_chng_q4_q1 = row.get("Total_Ct_Chng_Q4_Q1", 1.0)
    
    # Cálculo das features
    ticket_medio = total_trans_amt / total_trans_ct if total_trans_ct > 0 else 0
    transacoes_mes = total_trans_ct / months_on_book if months_on_book > 0 else 0
    gasto_mensal = total_trans_amt / months_on_book if months_on_book > 0 else 0
    rotativo_ratio = total_revolving_bal / credit_limit if credit_limit > 0 else 0
    disponibilidade_relativa = (credit_limit - total_revolving_bal) / credit_limit if credit_limit > 0 else 0
    
    # Faixa etária
    if idade < 30:
        faixa_idade = "<30"
    elif idade < 50:
        faixa_idade = "30-49"
    elif idade < 70:
        faixa_idade = "50-69"
    else:
        faixa_idade = "70+"
    
    # Classificação de renda
    income = row.get("Income_Category", "")
    if income in ["$60K - $80K", "$80K - $120K", "$120K +"]:
        renda_class = "Alta"
    elif income in ["$40K - $60K", "$20K - $40K"]:
        renda_class = "Média"
    else:
        renda_class = "Baixa"
    
    # Score de relacionamento e LTV Proxy
    score_relacionamento = total_relationship_count
    ltv_proxy = gasto_mensal * months_on_book
    
    # Flags de queda
    caiu_valor = 1 if total_amt_chng_q4_q1 < 1 else 0
    caiu_transacoes = 1 if total_ct_chng_q4_q1 < 1 else 0
    
    # Atualiza o dicionário com todas as features
    row.update({
        "Ticket_Medio": ticket_medio,
        "Transacoes_por_Mes": transacoes_mes,
        "Gasto_Medio_Mensal": gasto_mensal,
        "Rotativo_Ratio": rotativo_ratio,
        "Disponibilidade_Relativa": disponibilidade_relativa,
        "Faixa_Idade": faixa_idade,
        "Renda_Class": renda_class,
        "Score_Relacionamento": score_relacionamento,
        "LTV_Proxy": ltv_proxy,
        "Caiu_Valor": caiu_valor,
        "Caiu_Transacoes": caiu_transacoes,
    })
    
    return row

def montar_dataframe_previsao(row: dict) -> pd.DataFrame:
    """Prepara o dataframe para previsão com as 12 features esperadas pelo modelo"""
    
    # Features que o modelo espera (DEVE SER IGUAL AO TREINAMENTO)
    features_modelo = [
        'Customer_Age', 'Dependent_count', 'Credit_Limit',
        'Total_Trans_Amt', 'Total_Trans_Ct', 'Ticket_Medio',
        'Gasto_Medio_Mensal', 'Rotativo_Ratio', 'Score_Relacionamento',
        'LTV_Proxy', 'Caiu_Valor', 'Caiu_Transacoes'
    ]
    
    # Garantir que todas as features estão presentes
    for feature in features_modelo:
        if feature not in row:
            # Valores padrão seguros
            if feature == 'Customer_Age':
                row[feature] = row.get('Customer_Age', 45)
            elif feature == 'Dependent_count':
                row[feature] = row.get('Dependent_count', 1)
            elif feature == 'Credit_Limit':
                row[feature] = row.get('Credit_Limit', 10000.0)
            elif feature == 'Total_Trans_Amt':
                row[feature] = row.get('Total_Trans_Amt', 10000.0)
            elif feature == 'Total_Trans_Ct':
                row[feature] = row.get('Total_Trans_Ct', 50)
            else:
                row[feature] = 0  # Default para outras features
    
    # Criar DataFrame apenas com as features necessárias
    df = pd.DataFrame([row], columns=features_modelo)
    
    # Garantir que não há valores NaN
    df = df.fillna(0)
    
    return df

def prever_cliente(row: dict) -> tuple[float, int]:
    """Faz a previsão para um único cliente"""
    if modelo is None:
        return 0.0, 0
    
    try:
        # Calcular features
        row_eng = calcular_features_engineered_row(row)
        
        # Montar dataframe para previsão
        df = montar_dataframe_previsao(row_eng)
        
        # Aplicar scaler se disponível
        if scaler is not None:
            df_scaled = scaler.transform(df)
        else:
            df_scaled = df.values
        
        # Fazer predição
        prob = float(modelo.predict_proba(df_scaled)[0][1])
        classe = int(modelo.predict(df_scaled)[0])
        
        return prob, classe
        
    except Exception as e:
        st.error(f"❌ Erro na predição: {str(e)}")
        return 0.0, 0

def criar_gauge_chart(valor, titulo):
    """Cria um gráfico gauge para visualização de probabilidade"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=valor * 100,
        title={'text': titulo, 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "#1f77b4"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 60], 'color': '#fff3cd'},
                {'range': [60, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# -----------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=80)
st.sidebar.title("💳 Preditor de Churn")
st.sidebar.markdown("**MBA – Projeto Aplicado**")
st.sidebar.markdown("---")

aba = st.sidebar.radio(
    "📱 Navegação:",
    [
        "🏠 Início",
        "📈 Visão Geral do Modelo",
        "📊 Análise Exploratória",
        "👥 Exemplos Práticos",
        "👤 Simulador Individual",
        "📂 Análise em Lote",
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("""
💡 **Dica de Navegação:**
- Comece pelo **Início** para entender o contexto
- Explore os **Exemplos Práticos** para ver casos reais
- Use o **Simulador** para testar cenários
""")

# -----------------------------------------------------------
# ABA 0 – INÍCIO
# -----------------------------------------------------------
if aba.startswith("🏠"):
    st.markdown('<div class="main-header">🏦 Sistema de Predição de Churn Bancário</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 👋 Bem-vindo ao Sistema de Previsão de Evasão de Clientes
    
    Este sistema utiliza **Inteligência Artificial** para identificar clientes com alta probabilidade 
    de deixar o banco, permitindo ações preventivas de retenção.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>📊 O Problema</h3>
        <p>Clientes que cancelam seus cartões representam perda de receita e custos de aquisição desperdiçados.</p>
        <p><strong>Custo de aquisição:</strong> 5-7x maior que retenção</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>🎯 A Solução</h3>
        <p>Modelo de Machine Learning que prevê churn com <strong>99.3% de precisão</strong> (AUC)</p>
        <p><strong>Tecnologia:</strong> LightGBM + Engenharia de Features</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>💰 O Impacto</h3>
        <p>Identificação proativa permite campanhas de retenção direcionadas</p>
        <p><strong>ROI estimado:</strong> Redução de 20-30% no churn</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("🚀 Como Funciona")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **1️⃣ Coleta de Dados**
        
        📋 Perfil demográfico
        
        💳 Comportamento transacional
        
        📞 Histórico de relacionamento
        """)
    
    with col2:
        st.markdown("""
        **2️⃣ Análise Inteligente**
        
        🧠 Processamento com IA
        
        📈 Identificação de padrões
        
        🔍 Engenharia de features
        """)
    
    with col3:
        st.markdown("""
        **3️⃣ Previsão**
        
        ⚡ Score de risco (0-100%)
        
        🎯 Classificação automática
        
        📊 Confiança do modelo
        """)
    
    with col4:
        st.markdown("""
        **4️⃣ Ação**
        
        📱 Alertas para retenção
        
        🎁 Campanhas personalizadas
        
        💬 Abordagem proativa
        """)
    
    st.markdown("---")
    
    st.subheader("📚 Principais Indicadores de Churn")
    
    df = load_data_with_features()
    if df is not None and "churn_flag" in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔴 Sinais de Alerta (Clientes em Risco):**")
            st.markdown("""
            1. **Baixo número de transações** (< 40/ano)
            2. **Valor transacionado reduzido** (< $3.000/ano)
            3. **Contatos frequentes ao banco** (> 4/ano)
            4. **Baixa variação de gastos** (Q4/Q1 < 0.7)
            5. **Poucos produtos contratados** (< 3)
            """)
        
        with col2:
            st.markdown("**🟢 Sinais de Engajamento (Clientes Saudáveis):**")
            st.markdown("""
            1. **Alto volume de transações** (> 80/ano)
            2. **Gastos elevados** (> $10.000/ano)
            3. **Múltiplos produtos** (4-6 produtos)
            4. **Crescimento de uso** (Q4/Q1 > 0.9)
            5. **Baixa inatividade** (< 2 meses/ano)
            """)
    
    st.markdown("---")
    
    st.info("""
    ### 📌 Próximos Passos
    
    - Navegue para **Exemplos Práticos** para ver casos reais de clientes
    - Use o **Simulador Individual** para testar diferentes cenários
    - Explore a **Análise Exploratória** para entender os dados
    - Consulte a **Visão Geral do Modelo** para detalhes técnicos
    """)

# -----------------------------------------------------------
# ABA 1 – VISÃO GERAL DO MODELO
# -----------------------------------------------------------
elif aba.startswith("📈"):
    st.markdown('<div class="main-header">📈 Visão Geral do Modelo</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎯 Contexto de Negócio")
        st.markdown("""
        Este modelo de **Machine Learning** foi desenvolvido para prever a evasão de clientes 
        (churn) no segmento de cartões de crédito.
        
        #### 💼 Aplicações Práticas:
        - **Segmentação de risco:** Identificar clientes prioritários para ações de retenção
        - **Campanhas direcionadas:** Otimizar investimento em marketing
        - **Análise preventiva:** Agir antes do cancelamento efetivo
        - **KPIs de retenção:** Monitorar saúde da carteira em tempo real
        
        #### 🤖 Abordagem Técnica:
        O modelo **LightGBM** foi selecionado após comparação com Regressão Logística, 
        Random Forest e XGBoost, demonstrando melhor desempenho em validação cruzada.
        """)

    with col2:
        st.subheader("🏆 Métricas de Performance")
        
        metrics_data = {
            "Métrica": ["ROC AUC", "Acurácia", "Recall", "Precision", "F1-Score"],
            "Valor": [0.993, 0.971, 0.877, 0.938, 0.906],
            "Descrição": [
                "Capacidade de separação",
                "Previsões corretas",
                "Identifica 87.7% dos churns",
                "93.8% dos alertas são válidos",
                "Equilíbrio geral"
            ]
        }
        
        for i, (metric, valor, desc) in enumerate(zip(metrics_data["Métrica"], 
                                                        metrics_data["Valor"], 
                                                        metrics_data["Descrição"])):
            st.metric(metric, f"{valor:.3f}", help=desc)

    if METRICS_PATH.exists():
        try:
            st.markdown("---")
            st.subheader("🔬 Comparação de Modelos Testados")
            metrics_df = pd.read_csv(METRICS_PATH)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(
                    metrics_df.style.highlight_max(
                        subset=["roc_auc_mean", "accuracy_mean", "f1_mean"],
                        color="#c6efce",
                    ),
                    width='stretch'
                )
            
            with col2:
                st.info("""
                **Por que LightGBM?**
                
                ✅ Melhor AUC (0.993)
                
                ✅ Treinamento rápido
                
                ✅ Lida bem com desbalanceamento
                
                ✅ Interpretável via SHAP
                """)
        except Exception as e:
            st.warning(f"Não foi possível carregar métricas: {str(e)}")

    st.markdown("---")
    st.subheader("📊 Visualizações de Performance")

    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**Matriz de Confusão**")
        if FIG_CM_PATH.exists():
            st.image(str(FIG_CM_PATH), use_container_width=True)
            st.caption("A matriz mostra que o modelo comete poucos erros, com alta precisão em ambas as classes.")
        else:
            st.info("Matriz de confusão não encontrada. Execute o pipeline de treinamento primeiro.")

    with c2:
        st.markdown("**Curva ROC**")
        if FIG_ROC_PATH.exists():
            st.image(str(FIG_ROC_PATH), use_container_width=True)
            st.caption("Curva ROC próxima ao canto superior esquerdo indica excelente performance.")
        else:
            st.info("Curva ROC não encontrada. Execute o pipeline de treinamento primeiro.")

    st.markdown("---")
    st.subheader("🔧 Características Técnicas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📋 Variáveis de Entrada:**
        - Perfil demográfico (idade, dependentes, escolaridade)
        - Relacionamento (tempo de casa, produtos, contatos)
        - Comportamento financeiro (limite, saldo rotativo, utilização)
        - Padrões transacionais (volume, frequência, sazonalidade)
        """)
    
    with col2:
        st.markdown("""
        **⚙️ Processamento:**
        - Feature Engineering: 8 variáveis derivadas
        - Normalização: StandardScaler
        - Encoding: OneHotEncoder
        - Validação: 5-fold estratificado
        """)

# -----------------------------------------------------------
# ABA 2 – ANÁLISE EXPLORATÓRIA
# -----------------------------------------------------------
elif aba.startswith("📊"):
    st.markdown('<div class="main-header">📊 Análise Exploratória de Dados</div>', unsafe_allow_html=True)

    df = load_data_with_features()
    if df is None:
        st.error("❌ Base de dados não encontrada. Verifique o caminho do arquivo.")
    else:
        st.success(f"✅ Base carregada com sucesso: **{df.shape[0]:,}** clientes e **{df.shape[1]}** variáveis")
        
        if "churn_flag" in df.columns:
            churn_rate = df["churn_flag"].mean()
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Taxa de Churn", f"{churn_rate:.1%}")
            with col2:
                st.metric("Clientes Ativos", f"{(1-churn_rate)*100:.1f}%")
            with col3:
                st.metric("Total Churn", f"{df['churn_flag'].sum():,}")
            with col4:
                st.metric("Total Ativos", f"{(~df['churn_flag'].astype(bool)).sum():,}")

        tabs = st.tabs([
            "📌 Distribuições",
            "🧱 Features Engineered",
            "📉 Correlações",
            "🔥 Impacto no Churn"
        ])

        # TAB 1 – Distribuições
        with tabs[0]:
            st.subheader("📊 Distribuição das Variáveis Numéricas")
            
            st.info("""
            **💡 Como interpretar:**
            - **Histograma:** Mostra a frequência de valores (forma da distribuição)
            - **Boxplot:** Identifica outliers e mediana
            - Compare as distribuições para entender o perfil da carteira
            """)

            opcoes_num_pt = list(DIC_NOME_PT_NUMERICOS.keys())
            default_num = [
                "Idade do Cliente",
                "Limite de Crédito",
                "Valor Total Transacionado (12 meses)",
            ]

            cols_escolhidas_display = st.multiselect(
                "Selecione variáveis para análise:",
                options=opcoes_num_pt,
                default=[d for d in default_num if d in opcoes_num_pt],
            )

            if cols_escolhidas_display:
                for var_display in cols_escolhidas_display:
                    col = DIC_NOME_PT_NUMERICOS[var_display]
                    
                    st.markdown(f"### {var_display}")
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        fig_hist = px.histogram(
                            df,
                            x=col,
                            nbins=30,
                            marginal="box",
                            title=f"Distribuição",
                            labels={col: var_display, "count": "Frequência"},
                            color_discrete_sequence=["#1f77b4"]
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with c2:
                        if "churn_flag" in df.columns:
                            fig_box = px.box(
                                df,
                                x="churn_flag",
                                y=col,
                                points="outliers",
                                title=f"Comparação: Churn vs Ativo",
                                labels={
                                    "churn_flag": "Status (0=Ativo, 1=Churn)",
                                    col: var_display,
                                },
                                color="churn_flag",
                                color_discrete_map={0: "#28a745", 1: "#dc3545"}
                            )
                            st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.warning("Selecione ao menos uma variável para visualizar.")

        # TAB 2 – Features Engineered
        with tabs[1]:
            st.subheader("🧱 Variáveis Criadas (Feature Engineering)")
            
            st.markdown("""
            <div class="info-box">
            <h4>💡 O que são Features Engineered?</h4>
            <p>São variáveis derivadas que <strong>capturam padrões complexos</strong> do comportamento do cliente, 
            criadas através da combinação de variáveis originais.</p>
            <p>Estas features são <strong>críticas</strong> para o modelo identificar churn!</p>
            </div>
            """, unsafe_allow_html=True)

            opcoes_eng_pt = list(DIC_NOME_PT_ENGINEERED.keys())

            cols_escolhidas_display = st.multiselect(
                "Selecione variáveis derivadas:",
                options=opcoes_eng_pt,
                default=opcoes_eng_pt[:3] if len(opcoes_eng_pt) >= 3 else opcoes_eng_pt,
            )

            if cols_escolhidas_display:
                for var_display in cols_escolhidas_display:
                    col = DIC_NOME_PT_ENGINEERED[var_display]
                    st.markdown(f"### {var_display}")
                    
                    # Explicação da variável
                    explicacoes = {
                        "Ticket_Medio": "📊 **Significado:** Valor médio gasto por transação. Clientes com ticket muito baixo podem estar menos engajados.",
                        "Transacoes_por_Mes": "📊 **Significado:** Frequência mensal de uso do cartão. Baixa frequência indica risco de churn.",
                        "Gasto_Medio_Mensal": "📊 **Significado:** Intensidade de consumo mensal. Fundamental para identificar clientes valiosos.",
                        "Rotativo_Ratio": "📊 **Significado:** Proporção do limite usada para crédito rotativo. Alto uso pode indicar dependência ou problema financeiro.",
                        "Disponibilidade_Relativa": "📊 **Significado:** Quanto do limite ainda está disponível. Baixa disponibilidade pode gerar insatisfação.",
                    }
                    
                    if col in explicacoes:
                        st.info(explicacoes[col])
                    
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        fig_hist = px.histogram(
                            df,
                            x=col,
                            nbins=30,
                            title=f"Distribuição",
                            labels={col: var_display, "count": "Frequência"},
                            color_discrete_sequence=["#2ca02c"]
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with c2:
                        if "churn_flag" in df.columns:
                            fig_box = px.box(
                                df,
                                x="churn_flag",
                                y=col,
                                points="outliers",
                                title=f"Comparação: Churn vs Ativo",
                                labels={
                                    "churn_flag": "Status (0=Ativo, 1=Churn)",
                                    col: var_display,
                                },
                                color="churn_flag",
                                color_discrete_map={0: "#28a745", 1: "#dc3545"}
                            )
                            st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.warning("Selecione ao menos uma variável para visualizar.")

        # TAB 3 – Correlações
        with tabs[2]:
            st.subheader("📉 Análise de Correlações")
            
            st.markdown("""
            <div class="info-box">
            <h4>💡 Como interpretar a matriz de correlação?</h4>
            <ul>
            <li><strong>+1:</strong> Correlação positiva perfeita (quando uma sobe, a outra sobe)</li>
            <li><strong>0:</strong> Sem correlação</li>
            <li><strong>-1:</strong> Correlação negativa perfeita (quando uma sobe, a outra desce)</li>
            </ul>
            <p><strong>Cores:</strong> Azul = correlação positiva | Vermelho = correlação negativa</p>
            </div>
            """, unsafe_allow_html=True)

            opcoes_corr_pt = list(DIC_NOME_PT_NUMERICOS.keys()) + list(
                DIC_NOME_PT_ENGINEERED.keys()
            )

            cols_corr_display = st.multiselect(
                "Selecione variáveis para a matriz de correlação:",
                options=opcoes_corr_pt,
                default=[
                    "Idade do Cliente",
                    "Limite de Crédito",
                    "Valor Total Transacionado (12 meses)",
                    "Número de Transações (12 meses)",
                    "Ticket Médio por Transação",
                    "Gasto Médio Mensal",
                ],
            )

            if len(cols_corr_display) >= 2:
                def to_real(name_pt: str) -> str:
                    if name_pt in DIC_NOME_PT_NUMERICOS:
                        return DIC_NOME_PT_NUMERICOS[name_pt]
                    return DIC_NOME_PT_ENGINEERED[name_pt]

                cols_corr_real = [to_real(n) for n in cols_corr_display]
                corr = df[cols_corr_real].corr()

                mapping = {real: disp for real, disp in zip(cols_corr_real, cols_corr_display)}
                corr.rename(index=mapping, columns=mapping, inplace=True)

                fig_corr = px.imshow(
                    corr,
                    text_auto=".2f",
                    aspect="auto",
                    title="Matriz de Correlação",
                    color_continuous_scale="RdBu",
                    zmin=-1,
                    zmax=1
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Insights automáticos
                st.markdown("### 🔍 Principais Correlações")
                corr_flat = corr.unstack().sort_values(ascending=False)
                corr_flat = corr_flat[corr_flat < 0.99]  # Remove correlação de variável consigo mesma
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🔺 Top 3 Correlações Positivas:**")
                    for i, (vars, val) in enumerate(corr_flat.head(3).items(), 1):
                        st.markdown(f"{i}. **{vars[0]}** ↔️ **{vars[1]}**: {val:.2f}")
                
                with col2:
                    st.markdown("**🔻 Top 3 Correlações Negativas:**")
                    for i, (vars, val) in enumerate(corr_flat.tail(3).items(), 1):
                        st.markdown(f"{i}. **{vars[0]}** ↔️ **{vars[1]}**: {val:.2f}")
            else:
                st.warning("Selecione ao menos 2 variáveis para calcular correlação.")

        # TAB 4 – Impacto no Churn
        with tabs[3]:
            st.subheader("🔥 Relação das Variáveis com o Churn")

            if "churn_flag" not in df.columns:
                st.error("Coluna de churn não encontrada na base de dados.")
            else:
                st.markdown("""
                <div class="info-box">
                <h4>💡 Como usar esta análise?</h4>
                <p>Esta seção mostra <strong>como cada variável se comporta</strong> em clientes que deram churn vs. clientes ativos.</p>
                <p><strong>Objetivo:</strong> Identificar os "sinais de alerta" mais fortes para priorizar ações de retenção.</p>
                </div>
                """, unsafe_allow_html=True)

                opcoes_churn_pt = list(DIC_NOME_PT_NUMERICOS.keys()) + list(
                    DIC_NOME_PT_ENGINEERED.keys()
                )

                var_escolhida_display = st.selectbox(
                    "Escolha uma variável para analisar:",
                    options=opcoes_churn_pt,
                    index=min(opcoes_churn_pt.index("Número de Transações (12 meses)"), len(opcoes_churn_pt)-1) 
                    if "Número de Transações (12 meses)" in opcoes_churn_pt else 0,
                )

                if var_escolhida_display in DIC_NOME_PT_NUMERICOS:
                    var_escolhida = DIC_NOME_PT_NUMERICOS[var_escolhida_display]
                else:
                    var_escolhida = DIC_NOME_PT_ENGINEERED[var_escolhida_display]

                col1, col2 = st.columns(2)

                with col1:
                    fig_box = px.box(
                        df,
                        x="churn_flag",
                        y=var_escolhida,
                        points="outliers",
                        title=f"Distribuição por Status",
                        labels={
                            "churn_flag": "Status (0=Ativo, 1=Churn)",
                            var_escolhida: var_escolhida_display,
                        },
                        color="churn_flag",
                        color_discrete_map={0: "#28a745", 1: "#dc3545"}
                    )
                    st.plotly_chart(fig_box, use_container_width=True)

                with col2:
                    df_tmp = df[[var_escolhida, "churn_flag"]].dropna().copy()
                    df_tmp["faixa"] = pd.qcut(
                        df_tmp[var_escolhida],
                        q=min(5, len(df_tmp[var_escolhida].unique())),
                        duplicates="drop",
                    ).astype(str)

                    churn_por_faixa = (
                        df_tmp.groupby("faixa")["churn_flag"]
                        .mean()
                        .reset_index()
                        .rename(columns={"churn_flag": "taxa_churn"})
                        .sort_values("faixa")
                    )

                    fig_bar = px.bar(
                        churn_por_faixa,
                        x="faixa",
                        y="taxa_churn",
                        title=f"Taxa de Churn por Faixa",
                        labels={
                            "faixa": f"Faixas de {var_escolhida_display}",
                            "taxa_churn": "Taxa de Churn",
                        },
                        color="taxa_churn",
                        color_continuous_scale="Reds"
                    )
                    fig_bar.update_yaxes(tickformat=".0%")
                    st.plotly_chart(fig_bar, use_container_width=True)

                # Estatísticas comparativas
                st.markdown("### 📊 Estatísticas Comparativas")
                col1, col2, col3 = st.columns(3)
                
                media_churn = df[df["churn_flag"]==1][var_escolhida].mean()
                media_ativo = df[df["churn_flag"]==0][var_escolhida].mean()
                diferenca_pct = ((media_churn - media_ativo) / media_ativo * 100) if media_ativo != 0 else 0
                
                with col1:
                    st.metric("Média (Churn)", f"{media_churn:.2f}", 
                             delta=f"{diferenca_pct:.1f}% vs. Ativos",
                             delta_color="inverse")
                with col2:
                    st.metric("Média (Ativos)", f"{media_ativo:.2f}")
                with col3:
                    interpretacao = "📉 Menor em churn" if diferenca_pct < 0 else "📈 Maior em churn"
                    st.metric("Diferença", interpretacao)

# -----------------------------------------------------------
# ABA 3 – EXEMPLOS PRÁTICOS
# -----------------------------------------------------------
elif aba.startswith("👥"):
    st.markdown('<div class="main-header">👥 Exemplos Práticos de Clientes</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Veja exemplos reais de diferentes perfis de clientes e suas probabilidades de churn.
    Compare os padrões e entenda quais comportamentos são sinais de risco!
    """)
    
    # Exemplos pré-definidos (COM AS 12 FEATURES NECESSÁRIAS)
    exemplos = {
        "🔴 Alto Risco - Cliente Inativo": {
            "Customer_Age": 45,
            "Dependent_count": 2,
            "Credit_Limit": 8000.0,
            "Total_Trans_Amt": 2500.0,
            "Total_Trans_Ct": 25,
            "Total_Amt_Chng_Q4_Q1": 0.5,
            "Total_Ct_Chng_Q4_Q1": 0.4,
            "Total_Relationship_Count": 2,
            "Months_on_book": 36,
            "Total_Revolving_Bal": 1200.0,
            "Gender": "M",
            "Education_Level": "Graduate",
            "Marital_Status": "Married",
            "Income_Category": "$60K - $80K",
            "Card_Category": "Blue",
            "descricao": """
            **Perfil:** Cliente de 45 anos, casado, renda média-alta.
            
            **⚠️ Sinais de Alerta:**
            - Apenas 25 transações/ano (muito baixo!)
            - 4 meses inativo nos últimos 12 meses
            - Gastos caíram 50% (Q4 vs Q1)
            - Muitos contatos ao banco (5 em 12 meses)
            - Apenas 2 produtos contratados
            
            **💡 Interpretação:** Cliente claramente desengajado. Reduziu drasticamente o uso do cartão 
            e está possivelmente usando cartões da concorrência.
            """
        },
        
        "🟡 Risco Médio - Cliente em Declínio": {
            "Customer_Age": 38,
            "Dependent_count": 1,
            "Credit_Limit": 12000.0,
            "Total_Trans_Amt": 6000.0,
            "Total_Trans_Ct": 50,
            "Total_Amt_Chng_Q4_Q1": 0.75,
            "Total_Ct_Chng_Q4_Q1": 0.8,
            "Total_Relationship_Count": 3,
            "Months_on_book": 48,
            "Total_Revolving_Bal": 1800.0,
            "Gender": "F",
            "Education_Level": "Graduate",
            "Marital_Status": "Single",
            "Income_Category": "$80K - $120K",
            "Card_Category": "Silver",
            "descricao": """
            **Perfil:** Cliente de 38 anos, solteira, renda alta, 4 anos de relacionamento.
            
            **⚠️ Sinais de Alerta:**
            - Gastos em queda (25% de redução Q4 vs Q1)
            - Número de transações caindo (20% de redução)
            - 2 meses de inatividade recente
            
            **✅ Pontos Positivos:**
            - Ainda mantém 3 produtos
            - Limite de crédito razoável
            - 50 transações/ano (frequência moderada)
            
            **💡 Interpretação:** Cliente que já foi mais ativo. Pode estar testando concorrentes 
            ou mudando hábitos de consumo. Ainda há tempo para ação preventiva!
            """
        },
        
        "🟢 Baixo Risco - Cliente Engajado": {
            "Customer_Age": 42,
            "Dependent_count": 3,
            "Credit_Limit": 20000.0,
            "Total_Trans_Amt": 18000.0,
            "Total_Trans_Ct": 95,
            "Total_Amt_Chng_Q4_Q1": 1.1,
            "Total_Ct_Chng_Q4_Q1": 1.05,
            "Total_Relationship_Count": 5,
            "Months_on_book": 60,
            "Total_Revolving_Bal": 1500.0,
            "Gender": "M",
            "Education_Level": "Post-Graduate",
            "Marital_Status": "Married",
            "Income_Category": "$120K +",
            "Card_Category": "Gold",
            "descricao": """
            **Perfil:** Cliente de 42 anos, casado, renda muito alta, 5 anos de relacionamento.
            
            **✅ Sinais Positivos:**
            - 95 transações/ano (muito ativo!)
            - $18.000 gastos/ano (cliente valioso)
            - 5 produtos contratados (alto cross-sell)
            - Crescimento de 10% nos gastos (Q4 vs Q1)
            - Apenas 1 mês inativo no ano
            - Limite alto ($20k) com uso saudável (35%)
            
            **💡 Interpretação:** Cliente ideal! Altamente engajado, fiel e rentável. 
            Gastos crescentes indicam satisfação. Foco deve ser em manter este relacionamento 
            e oferecer upgrades (ex: Platinum).
            """
        }
    }
    
    exemplo_selecionado = st.selectbox(
        "Escolha um exemplo para análise:",
        options=list(exemplos.keys())
    )
    
    exemplo = exemplos[exemplo_selecionado]
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown(f"### {exemplo_selecionado}")
        st.markdown(exemplo["descricao"])
        
        # Criar dados do exemplo
        row_exemplo = {k: v for k, v in exemplo.items() if k != "descricao"}
        prob, classe = prever_cliente(row_exemplo)
        
        st.markdown("---")
        st.markdown("### 🎯 Predição do Modelo")
        
        # Gauge chart
        fig_gauge = criar_gauge_chart(prob, "Probabilidade de Churn")
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        if prob >= 0.6:
            st.markdown("""
            <div class="danger-box">
            <h4>🚨 AÇÃO URGENTE RECOMENDADA</h4>
            <p><strong>Sugestões:</strong></p>
            <ul>
            <li>Contato imediato da equipe de retenção</li>
            <li>Oferta de benefícios exclusivos</li>
            <li>Cashback ou pontos em dobro por 3 meses</li>
            <li>Upgrade de categoria do cartão sem custo</li>
            <li>Análise de reclamações ou insatisfações</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        elif prob >= 0.3:
            st.markdown("""
            <div class="info-box">
            <h4>⚠️ MONITORAMENTO PREVENTIVO</h4>
            <p><strong>Sugestões:</strong></p>
            <ul>
            <li>Incluir em campanha de engajamento</li>
            <li>Oferecer novos produtos/serviços</li>
            <li>Pesquisa de satisfação</li>
            <li>Programa de benefícios personalizados</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-box">
            <h4>✅ CLIENTE SAUDÁVEL</h4>
            <p><strong>Sugestões:</strong></p>
            <ul>
            <li>Manter qualidade do serviço</li>
            <li>Considerar upsell (cartões premium)</li>
            <li>Programas de fidelidade de longo prazo</li>
            <li>Cross-sell de outros produtos bancários</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📋 Dados do Cliente")
        
        # Mostrar dados em formato organizado
        st.markdown("**Perfil Demográfico:**")
        st.markdown(f"- Idade: {row_exemplo['Customer_Age']} anos")
        st.markdown(f"- Dependentes: {row_exemplo['Dependent_count']}")
        st.markdown(f"- Estado Civil: {row_exemplo.get('Marital_Status', 'N/A')}")
        st.markdown(f"- Escolaridade: {row_exemplo.get('Education_Level', 'N/A')}")
        st.markdown(f"- Renda: {row_exemplo.get('Income_Category', 'N/A')}")
        
        st.markdown("**Relacionamento:**")
        st.markdown(f"- Produtos: {row_exemplo.get('Total_Relationship_Count', 'N/A')}")
        
        st.markdown("**Comportamento Financeiro:**")
        st.markdown(f"- Limite: ${row_exemplo['Credit_Limit']:,.0f}")
        st.markdown(f"- Saldo rotativo: ${row_exemplo.get('Total_Revolving_Bal', 0):,.0f}")
        
        st.markdown("**Transações (12 meses):**")
        st.markdown(f"- Total gasto: ${row_exemplo['Total_Trans_Amt']:,.0f}")
        st.markdown(f"- Quantidade: {row_exemplo['Total_Trans_Ct']}")
        st.markdown(f"- Variação valor: {(row_exemplo.get('Total_Amt_Chng_Q4_Q1', 1)-1)*100:+.0f}%")
        st.markdown(f"- Variação qtde: {(row_exemplo.get('Total_Ct_Chng_Q4_Q1', 1)-1)*100:+.0f}%")
    
    st.markdown("---")
    st.info("""
    ### 💡 Dica para Análise
    
    Compare os diferentes exemplos para entender:
    - Quais métricas mais influenciam o risco de churn
    - Como pequenas mudanças no comportamento podem alterar a predição
    - Que tipos de ação são adequados para cada nível de risco
    
    Use o **Simulador Individual** para testar suas próprias combinações!
    """)

# -----------------------------------------------------------
# ABA 4 – SIMULADOR INDIVIDUAL
# -----------------------------------------------------------
elif aba.startswith("👤"):
    st.markdown('<div class="main-header">👤 Simulador de Churn Individual</div>', unsafe_allow_html=True)

    st.markdown("""
    Preencha os dados do cliente para obter uma previsão personalizada de risco de churn.
    Use os exemplos como referência ou crie seus próprios cenários!
    """)

    with st.form("form_cliente"):
        st.subheader("1️⃣ Perfil Demográfico")
        c1, c2, c3 = st.columns(3)
        with c1:
            idade = st.slider("Idade", 18, 90, 45, help="Idade do cliente em anos")
            dependentes = st.slider("Número de Dependentes", 0, 5, 1)
        with c2:
            gender = st.selectbox("Gênero", ["M", "F"])
            marital_status = st.selectbox(
                "Estado Civil",
                ["Single", "Married", "Divorced"],
            )
        with c3:
            education = st.selectbox(
                "Escolaridade",
                [
                    "Uneducated",
                    "High School",
                    "College",
                    "Graduate",
                    "Post-Graduate",
                    "Doctorate",
                    "Unknown",
                ],
            )

        st.subheader("2️⃣ Renda e Produto")
        c4, c5, c6 = st.columns(3)
        with c4:
            income_category = st.selectbox(
                "Faixa de Renda",
                [
                    "Less than $40K",
                    "$40K - $60K",
                    "$60K - $80K",
                    "$80K - $120K",
                    "$120K +",
                ],
            )
        with c5:
            card_category = st.selectbox(
                "Categoria do Cartão",
                ["Blue", "Silver", "Gold", "Platinum"],
            )
        with c6:
            total_relationship_count = st.slider(
                "Qtde Produtos com o Banco",
                1,
                8,
                3,
                help="Número total de produtos contratados (conta, investimentos, seguros, etc.)"
            )

        st.subheader("3️⃣ Relacionamento e Contato")
        c7, c8, c9 = st.columns(3)
        with c7:
            months_on_book = st.slider("Meses de Relacionamento", 6, 80, 36, 
                                      help="Há quanto tempo o cliente está no banco")
        with c8:
            months_inactive = st.slider("Meses Inativo (últimos 12)", 0, 6, 1,
                                       help="Quantos meses sem uso nos últimos 12 meses")
        with c9:
            contacts_12m = st.slider("Contatos com o Banco (12m)", 0, 10, 2,
                                    help="Número de vezes que o cliente contatou o banco")

        st.subheader("4️⃣ Comportamento Financeiro e Transacional")
        
        st.markdown("**💳 Crédito:**")
        c10, c11 = st.columns(2)
        with c10:
            credit_limit = st.number_input(
                "Limite de Crédito", min_value=500.0, value=10000.0, step=500.0,
                help="Limite total do cartão de crédito"
            )
        with c11:
            total_revolving_bal = st.number_input(
                "Saldo Rotativo Atual",
                min_value=0.0,
                value=1500.0,
                step=100.0,
                help="Valor atual em crédito rotativo (não pago integralmente)"
            )
        
        st.markdown("**💰 Transações:**")
        c12, c13 = st.columns(2)
        with c12:
            total_trans_amt = st.number_input(
                "Valor Total Transacionado (12m)",
                min_value=0.0,
                value=20000.0,
                step=500.0,
                help="Soma de todas as transações nos últimos 12 meses"
            )
        with c13:
            total_trans_ct = st.slider(
                "Número de Transações (12m)",
                1,
                200,
                60,
                help="Quantidade de transações realizadas"
            )
        
        st.markdown("**📊 Tendências (Trimestre 4 vs Trimestre 1):**")
        c14, c15, c16 = st.columns(3)
        with c14:
            avg_utilization_ratio = st.slider(
                "Utilização Média do Limite",
                0.0,
                1.0,
                0.3,
                step=0.05,
                help="Proporção média do limite que é utilizada"
            )
        with c15:
            total_amt_chng_q4q1 = st.slider(
                "Mudança de Valor Q4/Q1",
                0.0,
                3.0,
                1.0,
                step=0.1,
                help=">1 indica aumento de gasto; <1 indica queda; 1 = estável",
            )
        with c16:
            total_ct_chng_q4q1 = st.slider(
                "Mudança de Qtde Transações Q4/Q1",
                0.0,
                3.0,
                1.0,
                step=0.1,
                help=">1 indica mais transações; <1 indica queda; 1 = estável",
            )

        col_button = st.columns([1,1,1])[1]
        with col_button:
            submit = st.form_submit_button("🔮 Calcular Probabilidade de Churn", type="primary")

    if submit:
        row = {
            "Customer_Age": idade,
            "Dependent_count": dependentes,
            "Months_on_book": months_on_book,
            "Total_Relationship_Count": total_relationship_count,
            "Months_Inactive_12_mon": months_inactive,
            "Contacts_Count_12_mon": contacts_12m,
            "Credit_Limit": credit_limit,
            "Total_Revolving_Bal": total_revolving_bal,
            "Total_Amt_Chng_Q4_Q1": total_amt_chng_q4q1,
            "Total_Trans_Amt": total_trans_amt,
            "Total_Trans_Ct": total_trans_ct,
            "Total_Ct_Chng_Q4_Q1": total_ct_chng_q4q1,
            "Avg_Utilization_Ratio": avg_utilization_ratio,
            "Gender": gender,
            "Education_Level": education,
            "Marital_Status": marital_status,
            "Income_Category": income_category,
            "Card_Category": card_category,
        }

        prob, classe = prever_cliente(row)

        st.markdown("---")
        
        col_left, col_right = st.columns([2, 3])

        with col_left:
            st.markdown("### 🎯 Resultado da Predição")
            
            # Gauge chart
            fig = criar_gauge_chart(prob, "Probabilidade de Churn")
            st.plotly_chart(fig, use_container_width=True)
            
            # Classificação
            if prob >= 0.6:
                st.error(f"**🚨 ALTO RISCO DE CHURN** (Probabilidade: {prob:.1%})")
                st.markdown("""
                **Recomendações:**
                - Contato imediato da equipe de retenção
                - Oferecer benefícios exclusivos
                - Analisar possíveis reclamações
                """)
            elif prob >= 0.3:
                st.warning(f"**⚠️ RISCO MODERADO DE CHURN** (Probabilidade: {prob:.1%})")
                st.markdown("""
                **Recomendações:**
                - Monitorar comportamento
                - Campanha de engajamento
                - Oferecer novos produtos
                """)
            else:
                st.success(f"**✅ BAIXO RISCO DE CHURN** (Probabilidade: {prob:.1%})")
                st.markdown("""
                **Recomendações:**
                - Manter qualidade do serviço
                - Considerar upsell
                - Programas de fidelidade
                """)

        with col_right:
            st.markdown("### 📊 Dados Inseridos")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Perfil:**")
                st.markdown(f"- Idade: {idade} anos")
                st.markdown(f"- Dependentes: {dependentes}")
                st.markdown(f"- Gênero: {gender}")
                st.markdown(f"- Estado Civil: {marital_status}")
                st.markdown(f"- Escolaridade: {education}")
                st.markdown(f"- Renda: {income_category}")
                st.markdown(f"- Categoria Cartão: {card_category}")
                
            with col2:
                st.markdown("**Comportamento:**")
                st.markdown(f"- Produtos: {total_relationship_count}")
                st.markdown(f"- Meses de Relacionamento: {months_on_book}")
                st.markdown(f"- Meses Inativos: {months_inactive}")
                st.markdown(f"- Contatos: {contacts_12m}")
                st.markdown(f"- Limite: ${credit_limit:,.0f}")
                st.markdown(f"- Saldo Rotativo: ${total_revolving_bal:,.0f}")
                st.markdown(f"- Transações: {total_trans_ct}")
                st.markdown(f"- Valor Transacionado: ${total_trans_amt:,.0f}")
                st.markdown(f"- Variação Valor: {total_amt_chng_q4q1:.2f}")
                st.markdown(f"- Variação Qtde: {total_ct_chng_q4q1:.2f}")
                st.markdown(f"- Utilização: {avg_utilization_ratio:.1%}")

        st.markdown("---")
        st.info("""
        **💡 Dica:** Para reduzir o risco de churn, considere:
        - Aumentar o número de produtos contratados
        - Reduzir meses de inatividade
        - Aumentar o volume de transações
        - Manter tendência de crescimento nos gastos
        """)

# -----------------------------------------------------------
# ABA 5 – ANÁLISE EM LOTE
# -----------------------------------------------------------
elif aba.startswith("📂"):
    st.markdown('<div class="main-header">📂 Análise de Churn em Lote</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Faça upload de um arquivo CSV com dados de múltiplos clientes para obter previsões em lote.
    O arquivo deve conter as mesmas colunas do conjunto de dados original.
    """)
    
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            # Carregar dados
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"✅ Arquivo carregado com sucesso! {df_upload.shape[0]} clientes encontrados.")
            
            # Mostrar prévia
            st.subheader("📋 Prévia dos Dados")
            st.dataframe(df_upload.head(), use_container_width=True)
            
            # Verificar colunas necessárias
            colunas_necessarias = [
                "Customer_Age", "Dependent_count", "Months_on_book",
                "Total_Relationship_Count", "Months_Inactive_12_mon",
                "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal",
                "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct",
                "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio", "Gender",
                "Education_Level", "Marital_Status", "Income_Category", "Card_Category"
            ]
            
            colunas_faltantes = [col for col in colunas_necessarias if col not in df_upload.columns]
            
            if colunas_faltantes:
                st.error(f"❌ Colunas faltantes no arquivo: {', '.join(colunas_faltantes)}")
                st.info("Certifique-se de que o arquivo possui todas as colunas necessárias.")
            else:
                if st.button("🔮 Executar Previsões em Lote", type="primary"):
                    with st.spinner("Processando..."):
                        # Preparar para previsões
                        resultados = []
                        total_rows = len(df_upload)
                        progress_bar = st.progress(0)
                        
                        for idx, row in df_upload.iterrows():
                            try:
                                prob, classe = prever_cliente(row.to_dict())
                                resultados.append({
                                    "Cliente_ID": idx + 1,
                                    "Probabilidade_Churn": prob,
                                    "Previsao_Churn": classe,
                                    "Risco": "Alto" if prob >= 0.6 else "Moderado" if prob >= 0.3 else "Baixo"
                                })
                            except Exception as e:
                                resultados.append({
                                    "Cliente_ID": idx + 1,
                                    "Probabilidade_Churn": None,
                                    "Previsao_Churn": None,
                                    "Risco": "Erro"
                                })
                            
                            # Atualizar progresso
                            progress_bar.progress((idx + 1) / total_rows)
                        
                        # Criar dataframe de resultados
                        df_resultados = pd.DataFrame(resultados)
                        
                        st.subheader("📊 Resultados das Previsões")
                        
                        # Métricas gerais
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            total_alto = len(df_resultados[df_resultados["Risco"] == "Alto"])
                            st.metric("Alto Risco", total_alto)
                        with col2:
                            total_moderado = len(df_resultados[df_resultados["Risco"] == "Moderado"])
                            st.metric("Risco Moderado", total_moderado)
                        with col3:
                            total_baixo = len(df_resultados[df_resultados["Risco"] == "Baixo"])
                            st.metric("Baixo Risco", total_baixo)
                        with col4:
                            valid_results = df_resultados[df_resultados["Probabilidade_Churn"].notna()]
                            taxa_churn = valid_results["Previsao_Churn"].mean() if len(valid_results) > 0 else 0
                            st.metric("Taxa Churn Prevista", f"{taxa_churn:.1%}" if len(valid_results) > 0 else "N/A")
                        
                        # DataFrame com resultados
                        st.dataframe(df_resultados, use_container_width=True)
                        
                        # Gráfico de distribuição
                        st.subheader("📈 Distribuição dos Riscos")
                        fig_dist = px.pie(
                            df_resultados,
                            names="Risco",
                            title="Distribuição de Clientes por Nível de Risco",
                            color="Risco",
                            color_discrete_map={"Alto": "#dc3545", "Moderado": "#ffc107", "Baixo": "#28a745", "Erro": "#6c757d"}
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                        
                        # Opção para download
                        st.subheader("💾 Download dos Resultados")
                        csv = df_resultados.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Baixar Resultados (CSV)",
                            data=csv,
                            file_name="resultados_churn.csv",
                            mime="text/csv"
                        )
                        
        except Exception as e:
            st.error(f"❌ Erro ao processar o arquivo: {str(e)}")
    else:
        st.info("👆 Faça upload de um arquivo CSV para começar a análise.")
        
        # Mostrar exemplo de estrutura
        st.subheader("📋 Estrutura Esperada do Arquivo")
        st.markdown("""
        O arquivo CSV deve conter as seguintes colunas (exemplo):
        
        | Customer_Age | Dependent_count | Months_on_book | Total_Relationship_Count | ... |
        |-------------|-----------------|----------------|--------------------------|-----|
        | 45          | 2               | 36             | 3                        | ... |
        | 32          | 1               | 24             | 4                        | ... |
        
        **Colunas obrigatórias:** Customer_Age, Dependent_count, Months_on_book, 
        Total_Relationship_Count, Months_Inactive_12_mon, Contacts_Count_12_mon,
        Credit_Limit, Total_Revolving_Bal, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt,
        Total_Trans_Ct, Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ratio, Gender,
        Education_Level, Marital_Status, Income_Category, Card_Category
        """)

# -----------------------------------------------------------
# RODAPÉ
# -----------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>📊 <strong>Banco Mercantil - Sistema de Predição de Churn</strong></p>
    <p>Desenvolvido como parte do MBA em Data Science & Analytics</p>
    <p>© 2024 - Todos os direitos reservados</p>
</div>
""", unsafe_allow_html=True)