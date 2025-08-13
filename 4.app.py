import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA


# 1. إعداد مفاتيح API ومتغيرات البيئة
# تأكد من استبدال المفاتيح ببياناتك الصحيحة
os.environ["GOOGLE_API_KEY"] = "AIzaSyAWOKOFo1B79OcIjc30JIauQ_pHwsYV2pc"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# 2. تحديد مسارات الملفات
PDF_FILE_PATH = r"C:\Users\Laptop World\Downloads\Documents\C++-Programming-7th-edition-by-DS-Malik.pdf"
DB_PERSIST_DIRECTORY = r"C:\Users\Laptop World\Downloads\Documents\cpp_book_db"


# 3. إعداد وظائف التلخيص وQA
# @st.cache_resource: لتخزين قاعدة البيانات في الذاكرة لتجنب إعادة بنائها في كل مرة
@st.cache_resource
def setup_qa_system():
    """تجهيز نظام QA وتحميل قاعدة البيانات المتجهية."""
    # التحقق من وجود قاعدة البيانات
    if not os.path.exists(DB_PERSIST_DIRECTORY):
        st.info("قاعدة البيانات غير موجودة، يتم إنشاؤها الآن. قد يستغرق هذا بضع دقائق...")
        try:
            loader = PyPDFLoader(PDF_FILE_PATH)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
            chunks = text_splitter.split_documents(documents)
            
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            
            db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=DB_PERSIST_DIRECTORY,
                collection_name="cpp_book_chunks"
            )
            st.success("تم إنشاء قاعدة البيانات بنجاح!")
        except FileNotFoundError:
            st.error("خطأ: ملف PDF غير موجود. يرجى التحقق من المسار.")
            return None
    else:
        st.info("تم تحميل قاعدة البيانات المتجهية بنجاح.")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = Chroma(persist_directory=DB_PERSIST_DIRECTORY, embedding_function=embeddings)
    
    # إعداد نموذج LLM للردود (يمكنك اختيار نموذج آخر)
    # تذكر أن نماذج HuggingFace تحتاج إلى HuggingFace Hub API token
    llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2")

    # إعداد سلسلة QA (Retriever)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )
    return qa

def summarize_text(text_to_summarize):
    """توليد ملخص للنص باستخدام نموذج Hugging Face."""
    # استخدام نموذج مخصص للتلخيص
    # يمكنك تجربة "facebook/bart-large-cnn" أو "google/pegasus-cnn_dailymail"
    summarizer_llm = HuggingFaceHub(repo_id="t5-small")
    prompt = f"Summarize the following text:\n\n{text_to_summarize}\n\nSummary:"
    summary = summarizer_llm(prompt)
    return summary


# 4. بناء واجهة التطبيق باستخدام Streamlit
st.set_page_config(page_title="ملخص ومساعد C++", layout="wide")
st.title("👨‍💻 ملخص ومساعد C++ الذكي")

# استخدام قائمة جانبية للتنقل بين الوظائف
page = st.sidebar.selectbox("اختر وظيفة", ["توليد ملخص للنص", "الإجابة على أسئلة C++"])

if page == "توليد ملخص للنص":
    st.header("توليد ملخص للنص 📝")
    user_input = st.text_area("ألصق النص المراد تلخيصه هنا:", height=300)
    if st.button("تلخيص"):
        if user_input:
            with st.spinner("جاري التلخيص..."):
                summary = summarize_text(user_input)
                st.subheader("الملخص:")
                st.write(summary)
        else:
            st.warning("الرجاء إدخال نص لتلخيصه.")

elif page == "الإجابة على أسئلة C++":
    st.header("الإجابة على أسئلة من كتاب C++ 📚")
    qa_chain = setup_qa_system()

    if qa_chain:
        question = st.text_input("اطرح سؤالاً عن C++:", "ما هو الـ Class في C++؟")
        if st.button("ابحث عن الإجابة"):
            if question:
                with st.spinner("جاري البحث عن الإجابة..."):
                    response = qa_chain.invoke(question)
                    st.subheader("الإجابة:")
                    st.write(response["result"])
            else:
                st.warning("الرجاء إدخال سؤال.")
                streamlit run 4.app.py