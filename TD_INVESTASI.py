import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

# ────
# 1. KONFIGURASI API & HALAMAN
# ────

# Groq_API KEY
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(
    page_title="TEDIINVESTASI",
    page_icon="📓",
    layout="wide"
)

# CSS Styling
st.markdown(
    """
    <style>
    .chat-message { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .user-message { background-color: #f0f2f6; }
    .bot-message { background-color: #e8f0fe; }
    </style>
    """,
    unsafe_allow_html=True
)

# Judul Aplikasi
st.title("📓TEMAN DISKUSI KEWIRAUSAHAAN")
st.markdown(
    """
    ### Selamat Datang di Asisten Pengetahuan Tentang **Portofolio INVESTASI & TIPU-TIPU DUNIA INVESTASI**
    Chat Bot ini adalah TEMAN DISKUSI yang  akan membantu Anda memahami lebih dalam tentang aneka  investasi seperti Obligasi/Sukuk, Reksadana, Saham, termasuk SCAM dan tipuan investasi lainnya. Pergunakanlah chatbot ini secara bijak, Segala keputusan investasi baik didasarkanatas hasil diskusi dengan chat bot ini maupun TIDAK adalah tanggung jawab pribadi masing-masing. INGAT INVESTASI adalah aktivitas yang mengandung RISIKO!!!.
    """
)

# ────
# 2. STATE DAN INISIALISASI
# ────
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ────
# 3. PROMPT UNTUK MENJAMIN BAHASA INDONESIA
# ────
PROMPT_INDONESIA = """\
Anda adalah seorang Ahli Porto Folio Investasi dan Risiko  yang berpengalaman praktis  lebih dari 25 tahun. Gunakan informasi konteks berikut untuk menjawab berbagai pertanyaan pengguna dalam bahasa Indonesia yang baik dan terstruktur.
Selalu berikan jawaban terbaik yang dapat kamu berikan dalam bahasa Indonesia dengan tone informatif.

Konteks: {context}
Riwayat Chat: {chat_history}
Pertanyaan: {question}

Jawaban:
"""

INDO_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=PROMPT_INDONESIA
)

# ────
# 4. FUNGSI INISIALISASI RAG
# ────
@st.cache_resource
def initialize_rag():
    try:
        loader = DirectoryLoader("documents", glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1233, chunk_overlap=234)
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="LazarusNLP/all-indo-e5-small-v4",
            model_kwargs={'device': 'cpu'}
        )

        vectorstore = FAISS.from_documents(texts, embeddings)

        llm = ChatGroq(
            temperature=0.54,
            model_name="gemma2-9b-it",
            max_tokens=1024
        )

        memory = ConversationBufferWindowMemory(
            k=3,
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                'prompt': INDO_PROMPT_TEMPLATE,
                'output_key': 'answer'
            }
        )

        return chain

    except Exception as e:
        st.error(f"Error during initialization: {str(e)}")
        return None

# ────
# 5. INISIALISASI SISTEM
# ────
if st.session_state.chain is None:
    with st.spinner("Memuat sistem..."):
        st.session_state.chain = initialize_rag()

# ────
# 6. FUNGSI ANALISIS PROPOSAL
# ────
def analyze_proposal(file):
    loader = PyPDFLoader(file)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1233, chunk_overlap=234)
    texts = text_splitter.split_documents([document])

    # Create a prompt for analysis
    analysis_prompt = f"""\
    Mohon analisis dengan detail:
    1. ANALISIS RISIKO:
    - Risiko Finansial
    - Risiko Operasional
    - Risiko Pasar
    - Risiko Regulasi
    - Risiko lainnya yang teridentifikasi

    2. KESIMPULAN UMUM:
    - Potensi keuntungan
    - Kelayakan bisnis
    - Tingkat urgensi
    - Kekuatan proposal
    - Kelemahan proposal

    3. PERTANYAAN LANJUTAN:
    - Pertanyaan terkait model bisnis
    - Pertanyaan terkait keuangan
    - Pertanyaan terkait tim manajemen
    - Pertanyaan terkait strategi
    - Pertanyaan terkait mitigasi risiko

    Berikan analisis yang objektif dan terstruktur.

    Konteks: {texts}
    """

    result = st.session_state.chain({"question": analysis_prompt})
    return result.get('answer', 'Tidak ada jawaban yang dihasilkan.')

# ────
# 7. ANTARMUKA CHAT
# ────
if st.session_state.chain:
    # 7.1 Tampilkan riwayat chat
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # 7.2 Upload File
    uploaded_file = st.file_uploader("Upload Proposal Investasi (.pdf)", type="pdf")
    if st.button("Analisis Proposal Investasi"):
        if uploaded_file is not None:
            with st.spinner("Menganalisis proposal..."):
                analysis_result = analyze_proposal(uploaded_file)
                st.markdown("### Hasil Analisis Proposal Investasi")
                st.write(analysis_result)

    # 7.3 Chat Input
    prompt = st.chat_input("✍️tuliskan pertanyaan Anda tentang investasi disini")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # 7.4 Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Mencari jawaban..."):
                try:
                    result = st.session_state.chain({"question": prompt})
                    answer = result.get('answer', '')
                    st.write(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

# ────
# 8. FOOTER & DISCLAIMER
# ────
st.markdown(
    """
    ---
    **Disclaimer:**
    - Sistem ini menggunakan AI-LLM dan dapat menghasilkan jawaban yang tidak selalu akurat.
    - **INGAT KEPUTUSAN INVESTASI merupakan TANGGUNG JAWAB PRIBADI!!!.**
    - Ketik: LANJUTKAN JAWABANMU untuk kemungkinan mendapatkan jawaban yang lebih baik dan utuh.
    - Mohon verifikasi informasi penting dengan sumber terpercaya.
    """
)
