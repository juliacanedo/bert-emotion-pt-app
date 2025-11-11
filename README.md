# 🎭 Classificador de Emoções BERT

<div align="center">

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)

**Uma aplicação web inteligente para classificação de emoções em texto usando BERT**

🚀 [Acesse o App](#) • 🤖 [Sobre o Modelo](#sobre-o-modelo) • 🛠️ [Tecnologias](#️-tecnologias)

</div>

---

## 🌟 Sobre o Projeto

Este projeto utiliza um modelo **BERT fine-tuned** para classificação **multirrótulo de emoções em textos** em Português.  
A aplicação identifica múltiplas emoções simultaneamente no mesmo texto, com **calibração avançada via Platt Scaling** para melhor precisão.

---

## ✨ Funcionalidades Principais

- 🧠 **Análise em tempo real** de emoções em texto  
- 🎯 **Detecção multilabel** (várias emoções por texto)  
- 📊 **Probabilidades calibradas** com Platt Scaling  
- 🎨 **Interface intuitiva** e visualmente atrativa  
- ⚡ **Processamento rápido** com modelo otimizado  

---

## 🚀 Como Usar

### 🌐 Versão Web (Recomendado)
Acesse o app: [https://bert-emotion-pt-app.streamlit.app/](#) e veja os resultados com probabilidades e níveis de confiança  

### 💻 Execução Local

```bash
# Clone o repositório
git clone https://github.com/juliacanedo/bert-emotion-pt-app.git

# Entre na pasta do projeto
cd bert-emotion-pt-app

# Instale as dependências
pip install -r requirements.txt

# Execute a aplicação
streamlit run app.py
```

---

## 🎯 Emoções Detectáveis

O modelo identifica **28 emoções** diferentes:

| Categoria | Emoções Principais |
|------------|-------------------|
| 😊 **Positivas** | admiration, approval, caring, curiosity, desire, excitement, gratitude, joy, love, optimism, pride, relief |
| 😠 **Negativas** | anger, annoyance, disapproval, disappointment, disgust, embarrassment, fear, grief, nervousness, remorse, sadness |
| 😐 **Neutras** | confusion, curiosity, realization, surprise, neutral |

---

## 🛠️ Tecnologias

### 🤖 Machine Learning
- **BERT Base** fine-tuned no dataset *GoEmotions* BR
- **PyTorch** para inferência
- **Transformers** da Hugging Face
- **Platt Scaling** para calibração de probabilidades
- **SCUT** para otimização de thresholds

### 💻 Desenvolvimento
- **Streamlit** para interface web
- **NumPy** e **Pandas** para processamento
- **Plotly** para visualizações (futuras)

### ☁️ Deploy
- **Streamlit Cloud** para hospedagem  
- **Git LFS** para versionamento de modelos grandes  
- **GitHub** para controle de versão  

---

## 📊 Sobre o Modelo

### 🎯 Arquitetura
- **Base Model:** `bert-base-uncased`  
- **Fine-tuning:** dataset *GoEmotions* (58k samples)  
- **Tarefa:** Classificação multilabel  
- **Classes:** 28 emoções  

### ⚡ Performance
- **Calibração:** Platt Scaling por classe  
- **Thresholds:** Adaptativos por emoção  
- **Otimização:** SCUT para F1-score balanceado  

### 🔧 Pipeline de Treinamento
1. Pré-processamento com tokenização BERT  
2. Fine-tuning com classificação multilabel  
3. Calibração com Platt Scaling  
4. Otimização de thresholds com SCUT  

---

## 🗂️ Estrutura do Projeto

```text
bert-emotion-pt-app/
├── app.py                       # Aplicação principal Streamlit
├── predict_bert.py              # Módulo de predição do modelo
├── requirements.txt             # Dependências do projeto
├── models/                      # Modelo BERT treinado
│   └── bert_base_cb_loss_final/
├── configs/                     # Configurações e classes
│   ├── classes.txt
│   ├── retune_config.json
│   └── thresholds.json
└── .streamlit/                  # Configurações do Streamlit
    └── config.toml
└── .devcontainer/               # Ambiente de desenvolvimento (VS Code / Codespaces)
    └── devcontainer.json
```

---

## 🚧 Desenvolvimento

### 📋 Pré-requisitos
- Python **3.8+**
- Git LFS (para baixar o modelo)

### 🔧 Instalação para Desenvolvimento

```bash
# Clonar com LFS
git lfs install
git clone https://github.com/juliacanedo/bert-emotion-pt-app.git
cd bert-emotion-pt-app

# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# ou
.venv\Scripts\activate      # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 🧪 Executar Localmente
```bash
streamlit run app.py
```

---

## 📈 Resultados e Métricas

O modelo foi avaliado com métricas robustas para classificação multilabel:

| Métrica | Valor |
|----------|-------|
| **F1-Score Macro** | *0,48* |
| **F1-Score Micro** | *0,55* |
| **mAP** | *0.4807* |
| **ECE** | *0.008356* |

---

## 🤝 Contribuindo

Contribuições são bem-vindas!  
Siga estes passos:

1. **Fork** o projeto  
2. Crie uma branch:  
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Faça commit das mudanças:  
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Envie para o repositório remoto:  
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Abra um **Pull Request**

---

## 📝 Licença

Este projeto está sob a licença **MIT**.  
Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 👩‍💻 Autora

**Julia Canedo**  
🔗 [GitHub](https://github.com/juliacanedo) • [LinkedIn](https://www.linkedin.com/in/juliacanedo)

---

## 🙏 Agradecimentos

- [Hugging Face](https://huggingface.co) pela biblioteca *Transformers* e o modelo *BERTimbau*
- [Google Research](https://github.com/google-research/google-research/tree/master/goemotions) pelo dataset *GoEmotions*
- [Antonio Menezes](https://huggingface.co/datasets/antoniomenezes/go_emotions_ptbr) pela tradução e disponibilização do dataset *GoEmotions-PTBR*  
- [Streamlit](https://streamlit.io) pela plataforma de deploy  

<div align="center">

⭐️ *Se você gostou, deixe uma estrela no repositório!*

</div>