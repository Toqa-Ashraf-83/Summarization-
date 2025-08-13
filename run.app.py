#!/usr/bin/env python3
"""
Startup script for Luxury Haven Hotel RAG Chatbot
Run this file to start the FastAPI server
"""

import uvicorn
from src.app import app

if __name__ == "__main__":
    print("🏨 Starting Luxury Haven Hotel RAG Chatbot...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API documentation will be available at: http://localhost:8000/docs")
    print("🔄 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )







# app.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 1. تحديد مفتاح API لـ Google
# تأكد من استبدال هذا النص بمفتاح API الصحيح الخاص بك
os.environ["GOOGLE_API_KEY"] = "AIzaSyCsDgd99-7cNXt4f7Exbs-IAbZefLgQJBs"

# 2. تحديد مسار الملف وقاعدة البيانات
PDF_FILE_PATH = r"C:\Users\Laptop World\Downloads\Documents\C++-Programming-7th-edition-by-DS-Malik.pdf"
DB_PERSIST_DIRECTORY = r"C:\Users\Laptop World\Downloads\Documents\cpp_book_db"

def create_or_load_db(file_path: str, persist_directory: str):
    """
    تقوم بإنشاء قاعدة بيانات Chroma المتجهية أو تحميلها إذا كانت موجودة.
    """
    # التحقق مما إذا كانت قاعدة البيانات موجودة بالفعل
    if os.path.exists(persist_directory):
        print("قاعدة البيانات موجودة، يتم تحميلها...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        print("قاعدة البيانات غير موجودة، يتم إنشاؤها...")
        # تحميل ملف PDF
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        except FileNotFoundError:
            print("خطأ: ملف PDF غير موجود في المسار المحدد.")
            return None

        # تقسيم المستندات إلى أجزاء
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=500,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # إعداد نموذج التضمين
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # إنشاء قاعدة البيانات من الأجزاء
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name="cpp_book_chunks"
        )
        print("تم إنشاء قاعدة البيانات بنجاح.")
    
    return db

def main():
    """
    الوظيفة الرئيسية للتطبيق.
    """
    # إنشاء أو تحميل قاعدة البيانات
    db = create_or_load_db(PDF_FILE_PATH, DB_PERSIST_DIRECTORY)
    
    if db:
        while True:
            # استقبال استعلام من المستخدم
            query = input("\nاطرح سؤالاً عن C++ (اكتب 'خروج' للإنهاء): ")
            if query.lower() == 'خروج':
                break
            
            # البحث عن تشابه
            print(f"جاري البحث عن: '{query}'...")
            results = db.similarity_search_with_score(query, k=3)
            
            # عرض النتائج
            print("\nنتائج البحث:")
            print("="*40)
            for res, score in results:
                print(f"درجة التشابه: {score:.3f}")
                print("محتوى الجزء:")
                print(res.page_content)
                print("-" * 20)
    else:
        print("لا يمكن تشغيل التطبيق بدون قاعدة بيانات.")

if __name__ == "__main__":
    main()