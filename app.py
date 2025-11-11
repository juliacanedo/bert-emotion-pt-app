# streamlit/app.py
import streamlit as st
import numpy as np
import torch

# Configuração
st.set_page_config(
    page_title="Classificador de Emoções BERT", 
    page_icon="🎭", 
    layout="wide"
)

# streamlit/app.py
import streamlit as st
import numpy as np
import torch

# Configuração
st.set_page_config(
    page_title="BERT Emotion Classifier", 
    page_icon="🎭", 
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
        background-color: #f8f9fa;
        color: #000000 !important;
    }
    .emotion-card strong {
        color: #000000 !important;
        font-size: 1.1rem;
    }
    .high-confidence {
        border-left-color: #2ecc71;
    }
    .medium-confidence {
        border-left-color: #f39c12;
    }
    .low-confidence {
        border-left-color: #e74c3c;
    }
    .probability-bar {
        height: 8px;
        background-color: #e0e0e0;
        border-radius: 4px;
        margin: 0.25rem 0;
    }
    .probability-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #e74c3c, #f39c12, #2ecc71);
    }
</style>
""", unsafe_allow_html=True)

# Cache do modelo
@st.cache_resource(show_spinner="🔄 Carregando modelo BERT...")
def load_predict_function():
    from predict_bert import predict, CLASSES, thresholds, cal_mode
    return predict, CLASSES, thresholds, cal_mode

def main():
    # Carrega tudo com cache
    predict_func, CLASSES, thresholds, cal_mode = load_predict_function()
    
    st.markdown('<h1 class="main-header">🎭 Classificador de Emoções BERT</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("ℹ️ Informações do Modelo")
        st.markdown(f"""
        **Configuração:**
        - **Modelo:** BERT Base com CB Loss
        - **Calibração:** {cal_mode} com Scut Adaptado por Classe
        - **Classes:** {len(CLASSES)} emoções
        - **Threshold médio:** {np.mean(thresholds):.3f}
        """)
        
        st.markdown("---")
        st.markdown("**📊 Estatísticas:**")
        st.metric("Total de Emoções", len(CLASSES))
        st.metric("Threshold Médio", f"{np.mean(thresholds):.3f}")
        
        st.markdown("---")
        st.markdown("**🎯 Emoções disponíveis:**")
        # Mostrar emoções em colunas na sidebar
        cols = st.columns(2)
        for i, emotion in enumerate(CLASSES):
            with cols[i % 2]:
                st.caption(f"• {emotion}")

    # Interface principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Digite seu texto para análise")
        
        # Text area simples
        text_input = st.text_area(
            "Digite o texto para análise:",
            height=150,
            placeholder="Exemplo: Estou muito feliz com os resultados incríveis deste projeto! 😊",
            help="O modelo analisará as emoções presentes no texto",
            key="main_text_input"
        )

        # Configurações
        with st.expander("⚙️ Configurações"):
            topk_fallback = st.slider("Top-K Fallback", 1, 10, 3, 
                                    help="Número máximo de emoções a mostrar se nenhuma passar do threshold")
            show_all = st.checkbox("Mostrar todas as probabilidades", False,
                                 help="Exibir probabilidades de todas as emoções")

        # Botão de análise
        analyze_btn = st.button("🔍 Analisar Emoções", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📈 Sobre a Análise")
        st.info("""
        **Como funciona:**
        - 🔍 **Análise** do texto com BERT
        - 🎯 **Thresholds** adaptativos por emoção  
        - 📊 **Calibração** Platt Scaling com Scut Adaptado por Classe
        - 🏷️ **Multilabel** - várias emoções
        """)
        
        # Mostrar contagem de caracteres se houver texto
        if text_input:
            st.metric("Texto inserido", f"{len(text_input)} caracteres")

    # Processamento
    if analyze_btn and text_input:
        if len(text_input.strip()) < 3:
            st.warning("⚠️ Digite pelo menos 3 caracteres")
        else:
            with st.spinner("🔮 Analisando emoções..."):
                try:
                    labels, all_probs = predict_func(text_input, topk_fallback=topk_fallback)
                    sorted_labels = sorted(labels, key=lambda x: x[1], reverse=True)
                    
                    # Resultados
                    st.subheader("🎯 Resultados da Análise")
                    
                    if not sorted_labels:
                        st.info("🤔 Nenhuma emoção identificada com confiança suficiente.")
                    else:
                        st.success(f"✅ **{len(sorted_labels)} emoção(ões) detectada(s):**")
                        
                        for emotion, prob in sorted_labels:
                            if prob >= 0.7:
                                css_class = "high-confidence"
                                color = "#2ecc71"
                                conf_text = "Alta confiança"
                            elif prob >= 0.4:
                                css_class = "medium-confidence"
                                color = "#f39c12" 
                                conf_text = "Média confiança"
                            else:
                                css_class = "low-confidence"
                                color = "#e74c3c"
                                conf_text = "Baixa confiança"
                            
                            st.markdown(f"""
                            <div class="emotion-card {css_class}">
                                <div style="display: flex; justify-content: between; align-items: center;">
                                    <strong>{emotion}</strong>
                                    <span style="margin-left: auto; font-weight: bold; color: {color} !important;">
                                        {prob:.3f}
                                    </span>
                                </div>
                                <div style="font-size: 0.8rem; color: {color}; margin-bottom: 0.5rem;">
                                    {conf_text}
                                </div>
                                <div class="probability-bar">
                                    <div class="probability-fill" style="width: {prob*100}%"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Todas as probabilidades
                    if show_all:
                        st.subheader("📊 Todas as Probabilidades")
                        for i, prob in enumerate(all_probs):
                            above_threshold = prob >= thresholds[i]
                            status = "✅" if above_threshold else "❌"
                            color = "#2ecc71" if above_threshold else "#e74c3c"
                            st.write(f"{status} **{CLASSES[i]}**: `{prob:.3f}` (threshold: `{thresholds[i]:.3f}`)")
                            
                except Exception as e:
                    st.error(f"❌ Erro na análise: {str(e)}")
    
    elif analyze_btn and not text_input:
        st.warning("⚠️ Por favor, digite algum texto para analisar")

if __name__ == "__main__":
    main()