# =====================================================
# app.py
# =====================================================

import streamlit as st
import json
import os
import re
import requests
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import time
import logging

# FAISS
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Haystack (Fallback)
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder
)
from haystack.components.retrievers import InMemoryEmbeddingRetriever

# FLAN-T5 and LoRA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch

# Download NLTK data
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# =====================================================
# 1. CONFIGURATION
# =====================================================

LLAMA_SERVER_URL = "http://127.0.0.1:8080"
LLAMA_TIMEOUT = 600  # 10 minutes

FAISS_INDEX_PATH = "./vector_store/faiss_index.bin"
FAISS_MAPPING_PATH = "./vector_store/index_mapping.json"
JSONL_PATH = "./corpus/class8_science.jsonl"

# =====================================================
# 2. LLAMA SERVER API
# =====================================================

def check_llama_server() -> bool:
    try:
        response = requests.get(f"{LLAMA_SERVER_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def generate_llama_via_api(prompt: str, max_tokens: int = 150) -> Optional[str]:
    try:
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.3,
            "top_p": 0.95,
            "repeat_penalty": 1.1,
            "stop": ["\n\n", "Question:", "Context:", "Answer:"]
        }
        
        response = requests.post(
            f"{LLAMA_SERVER_URL}/completion",
            json=payload,
            timeout=LLAMA_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("content", "").strip()
        return None
    except Exception as e:
        print(f"❌ Llama API error: {e}")
        return None

# =====================================================
# 3. TEXT FORMATTING
# =====================================================

def clean_and_format_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    
    text = ' '.join(text.split())
    
    if text and not text[-1] in '.!?':
        text = text + '.'
    
    if text and len(text) > 0:
        text = text[0].upper() + text[1:]
    
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\s+([\.!?,;:])', r'\1', text)
    text = re.sub(r'([\.!?,;:])([^\s])', r'\1 \2', text)
    
    return text.strip()

def log_interaction(question: str, answer: str, model_used: str):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer[:500],
        "model_used": model_used
    }
    os.makedirs("./logs", exist_ok=True)
    with open("./logs/interactions.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# =====================================================
# 4. FAISS VECTOR STORE (Primary)
# =====================================================

class FAISSVectorStore:
    def __init__(self):
        self.embedder = None
        self.index = None
        self.mapping = []
        self.is_loaded = False
        
    def initialize_embedder(self):
        try:
            self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            return True
        except Exception as e:
            print(f"❌ Embedder error: {e}")
            return False
    
    def load_index(self) -> bool:
        try:
            if not os.path.exists(FAISS_INDEX_PATH):
                print(f"❌ FAISS index not found: {FAISS_INDEX_PATH}")
                return False
            
            if not os.path.exists(FAISS_MAPPING_PATH):
                print(f"❌ Mapping not found: {FAISS_MAPPING_PATH}")
                return False
            
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            print(f"✅ FAISS loaded: {self.index.ntotal} vectors")
            
            with open(FAISS_MAPPING_PATH, 'r', encoding='utf-8') as f:
                self.mapping = json.load(f)
            
            if self.mapping and 'content' not in self.mapping[0]:
                print("⚠️ Mapping format incorrect - fixing...")
                with open(FAISS_MAPPING_PATH, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                if isinstance(raw_data, dict):
                    self.mapping = [{"content": str(v), "chapter": i, "title": f"Chapter {i}"} 
                                   for i, v in enumerate(raw_data.values())]
                else:
                    self.mapping = [{"content": str(item), "chapter": i, "title": f"Chapter {i}"} 
                                   for i, item in enumerate(raw_data)]
            
            print(f"✅ Mapping loaded: {len(self.mapping)} documents")
            self.initialize_embedder()
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ FAISS load error: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.is_loaded:
            return []
        
        if self.embedder is None:
            self.initialize_embedder()
        
        try:
            query_embedding = self.embedder.encode([query], convert_to_numpy=True)
            distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.mapping):
                    doc = self.mapping[idx]
                    results.append({
                        "content": doc.get("content", ""),
                        "chapter": doc.get("chapter", 1),
                        "title": doc.get("title", "Unknown Chapter"),
                        "score": float(1 / (1 + distances[0][i]))
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []

# =====================================================
# 5. HAYSTACK PIPELINE (Fallback)
# =====================================================

@st.cache_resource
def load_haystack_pipeline():
    """Load Haystack pipeline as fallback"""
    try:
        documents = []
        
        if os.path.exists(JSONL_PATH):
            with open(JSONL_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    doc = Document(
                        content=data["content"],
                        meta={"chapter": data.get("chapter", 1), "title": data.get("title", "Unknown")}
                    )
                    documents.append(doc)
            print(f"✅ Loaded {len(documents)} documents from JSONL")
        
        if not documents and os.path.exists(FAISS_MAPPING_PATH):
            with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            for item in mapping:
                if isinstance(item, dict):
                    doc = Document(
                        content=item.get("content", ""),
                        meta={"chapter": item.get("chapter", 1), "title": item.get("title", "Unknown")}
                    )
                    documents.append(doc)
            print(f"✅ Loaded {len(documents)} documents from FAISS mapping")
        
        if not documents:
            print("❌ No documents found!")
            return None, None, []
        
        document_store = InMemoryDocumentStore()
        
        doc_embedder = SentenceTransformersDocumentEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        doc_embedder.warm_up()
        
        docs_with_embeds = doc_embedder.run(documents)
        document_store.write_documents(docs_with_embeds["documents"])
        
        text_embedder = SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        text_embedder.warm_up()
        
        retriever = InMemoryEmbeddingRetriever(document_store=document_store)
        
        return retriever, text_embedder, documents
        
    except Exception as e:
        print(f"❌ Haystack load error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, []

# =====================================================
# 6. FLAN-T5 LORA
# =====================================================

@st.cache_resource
def load_flant5_lora():
    try:
        adapter_path = "./lora_adapter/final_model"
        
        if not os.path.exists(adapter_path):
            adapter_path = "./lora_adapter/cpu_model"
            
        if not os.path.exists(adapter_path):
            st.warning(f"⚠️ FLAN-T5 LoRA not found")
            return None, None
        
        base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        return model, tokenizer
        
    except Exception as e:
        st.warning(f"⚠️ FLAN-T5 LoRA not available: {e}")
        return None, None

# =====================================================
# 7. SYLLABUS CHECK
# =====================================================

def is_out_of_syllabus(question: str) -> Tuple[bool, str]:
    valid_topics = [
        'crop', 'agriculture', 'farming', 'soil', 'irrigation', 'weeding', 'harvesting',
        'microorganism', 'bacteria', 'fungi', 'virus', 'fermentation', 'antibiotic',
        'coal', 'petroleum', 'fossil fuel', 'natural gas', 'conservation',
        'combustion', 'flame', 'ignition', 'fire', 'deforestation', 'greenhouse effect',
        'reproduction', 'fertilization', 'adolescence', 'puberty', 'force', 'pressure',
        'friction', 'static friction', 'sliding friction', 'rolling friction', 'sound', 
        'noise pollution', 'light', 'reflection', 'refraction', 'dispersion', 'mirror', 'lens',
        'electric current', 'conductor', 'insulator', 'lightning', 'earthquake', 'tsunami',
        'cell', 'nucleus', 'photosynthesis', 'ecosystem', 'biodiversity', 'constellation'
    ]
    
    q_lower = question.lower()
    
    if any(topic in q_lower for topic in valid_topics):
        return False, None
    
    return True, "I'm focused on Class 8 Science. Please ask questions from your NCERT textbook."

# =====================================================
# 8. GENERATE FLAN-T5 ANSWER
# =====================================================

def generate_flant5_answer(question: str, vector_store=None, retriever=None, 
                           text_embedder=None, model=None, tokenizer=None) -> Optional[str]:
    if model is None or tokenizer is None:
        return None
    
    try:
        context = ""
        
        if vector_store and vector_store.is_loaded:
            results = vector_store.search(question, top_k=1)
            if results:
                context = results[0]["content"][:400]
        
        if not context and retriever and text_embedder:
            query_embedding = text_embedder.run(text=question)
            retrieval_result = retriever.run(
                query_embedding=query_embedding['embedding'],
                top_k=1
            )
            if retrieval_result.get("documents"):
                context = retrieval_result["documents"][0].content[:400]
        
        if context:
            input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
        else:
            input_text = f"Question: {question}\nAnswer:"
        
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                temperature=0.2,
                do_sample=True,
                top_p=0.9,
                num_beams=2,
            )
        
        raw_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        nonsense_words = ['sandstone', 'sandbox', 'squat', 'repulsion']
        if any(word in raw_answer.lower() for word in nonsense_words) or len(raw_answer.split()) < 3:
            return None
        
        return clean_and_format_text(raw_answer)
        
    except Exception as e:
        print(f"FLAN-T5 generation error: {e}")
        return None

# =====================================================
# 9. MAIN ANSWER FUNCTION
# =====================================================

def answer_question(question: str, vector_store=None, retriever=None, 
                    text_embedder=None, flant5_model=None, flant5_tokenizer=None) -> Dict:
    
    out_of_syllabus, msg = is_out_of_syllabus(question)
    if out_of_syllabus:
        return {
            'answer': f"📖 {clean_and_format_text(msg)}\n\nPlease ask a question from your Class 8 Science NCERT textbook.",
            'model_used': "Syllabus Filter"
        }
    
    try:
        context = ""
        chapter = 1
        title = "Unknown"
        
        if vector_store and vector_store.is_loaded:
            results = vector_store.search(question, top_k=3)
            if results:
                context = results[0]["content"][:600]
                chapter = results[0]["chapter"]
                title = results[0]["title"]
                print("✅ Using FAISS retrieval")
        
        if not context and retriever and text_embedder:
            query_embedding = text_embedder.run(text=question)
            retrieval_result = retriever.run(
                query_embedding=query_embedding['embedding'],
                top_k=3
            )
            if retrieval_result.get("documents"):
                doc = retrieval_result["documents"][0]
                context = doc.content[:600]
                chapter = doc.meta.get("chapter", 1)
                title = doc.meta.get("title", "Unknown")
                print("✅ Using Haystack retrieval")
        
        if not context:
            return {'answer': "I couldn't find information about that. Please rephrase your question.", 'model_used': "No Results"}
        
        llama_available = check_llama_server()
        
        if llama_available:
            llama_prompt = f"""Answer based ONLY on this textbook excerpt:
{context}

Question: {question}
Answer (one sentence):"""
            
            with st.spinner("🦙 Llama 2 is thinking... (up to 10 minutes)"):
                start_time = time.time()
                llama_answer = generate_llama_via_api(llama_prompt, max_tokens=100)
                elapsed = time.time() - start_time
            
            if llama_answer and len(llama_answer) > 15:
                final_answer = clean_and_format_text(llama_answer)
                model_used = f"Llama 2 (⏱️ {elapsed:.1f}s)"
                
                citation = f"\n\n📚 Source: Chapter {chapter} - {title}"
                model_badge = f"\n🤖 Model: {model_used}"
                
                log_interaction(question, final_answer, model_used)
                
                return {
                    'answer': f"{final_answer}{citation}{model_badge}",
                    'model_used': model_used
                }
        
        with st.spinner("🔧 Generating answer with FLAN-T5..."):
            flant5_answer = generate_flant5_answer(
                question, vector_store, retriever, text_embedder,
                flant5_model, flant5_tokenizer
            )
        
        if flant5_answer:
            final_answer = flant5_answer
            model_used = "FLAN-T5 LoRA"
        else:
            final_answer = clean_and_format_text(context[:200] + "...")
            model_used = "Context Extract"
        
        citation = f"\n\n📚 Source: Chapter {chapter} - {title}"
        model_badge = f"\n🤖 Model: {model_used}"
        
        log_interaction(question, final_answer, model_used)
        
        return {
            'answer': f"{final_answer}{citation}{model_badge}",
            'model_used': model_used
        }
        
    except Exception as e:
        error_msg = clean_and_format_text(f"Error: {str(e)[:100]}. Please try again.")
        return {'answer': error_msg, 'model_used': "Error"}

# =====================================================
# 10. STREAMLIT UI - SINGLE CHAT INTERFACE
# =====================================================

st.set_page_config(
    page_title="Class 8 Science AI Tutor",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message-user {
        background-color: #E3F2FD;
        padding: 0.8rem 1rem;
        border-radius: 20px;
        margin: 0.5rem 0;
        max-width: 75%;
        float: right;
        clear: both;
    }
    .chat-message-assistant {
        background-color: #E8F5E9;
        padding: 0.8rem 1rem;
        border-radius: 20px;
        margin: 0.5rem 0;
        max-width: 75%;
        float: left;
        clear: both;
        border-left: 3px solid #2E7D32;
    }
    .stButton > button {
        background-color: #2E7D32;
        color: white;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #1B5E20;
        color: white;
    }
    .input-container {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 1rem 0;
        border-top: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# =====================================================
# LOAD RESOURCES
# =====================================================

with st.spinner("🔄 Loading AI Tutor..."):
    vector_store = FAISSVectorStore()
    vector_store_loaded = vector_store.load_index()
    
    retriever, text_embedder, haystack_docs = load_haystack_pipeline()
    
    flant5_model, flant5_tokenizer = load_flant5_lora()
    
    llama_server_running = check_llama_server()

# =====================================================
# STATUS BAR
# =====================================================

col1, col2, col3 = st.columns(3)
with col1:
    if vector_store_loaded:
        st.success(f"✅ FAISS ({vector_store.index.ntotal} vectors)")
    elif haystack_docs:
        st.success(f"✅ Haystack ({len(haystack_docs)} docs)")
    else:
        st.error("❌ No Vector Store")
with col2:
    if llama_server_running:
        st.success("✅ Llama 2 Server")
    else:
        st.info("ℹ️ Llama 2 Not Running")
with col3:
    if flant5_model:
        st.success("✅ FLAN-T5 Loaded")
    else:
        st.warning("⚠️ FLAN-T5 Not Available")

# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="main-header"><h1>📖 NCERT Class 8 Science AI Tutor</h1>'
    '<p>Ask any question from your textbook!</p></div>',
    unsafe_allow_html=True
)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("## 🎯 Model Priority")
    st.markdown("1️⃣ **Llama 2** (Primary, 10min)")
    st.markdown("2️⃣ **FLAN-T5** (Fallback)")
    st.divider()
    
    st.markdown("## 📊 Status")
    st.markdown(f"🦙 **Llama 2:** {'✅ Running' if llama_server_running else '❌ Not running'}")
    st.markdown(f"🔧 **FLAN-T5:** {'✅ Loaded' if flant5_model else '❌ Not available'}")
    if vector_store_loaded:
        st.markdown(f"💾 **FAISS:** ✅ ({vector_store.index.ntotal} vectors)")
    elif haystack_docs:
        st.markdown(f"💾 **Haystack:** ✅ ({len(haystack_docs)} docs)")
    else:
        st.markdown("💾 **Vector Store:** ❌")
    st.divider()
    
    st.markdown(f"⏰ **Timeout:** 600s (10 minutes)")
    st.divider()
    
    st.markdown(f"**💬 Interactions:** {len(st.session_state.chat_history)}")
    
    if st.button("📥 Download Chat", use_container_width=True):
        if st.session_state.chat_history:
            chat_data = []
            for chat in st.session_state.chat_history:
                chat_data.append({
                    'timestamp': chat['timestamp'],
                    'question': chat['question'],
                    'answer': chat['answer']
                })
            df = pd.DataFrame(chat_data)
            filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Click to Download",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# =====================================================
# CHAT DISPLAY - SINGLE CHAT AREA
# =====================================================

# Display chat history
chat_container = st.container()

with chat_container:
    for chat in st.session_state.chat_history:
        st.markdown(
            f'<div style="display: flex; justify-content: flex-end;">'
            f'<div class="chat-message-user"><b>👤 You:</b> {chat["question"]}</div></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="chat-message-assistant"><b>🤖 Tutor:</b> {chat["answer"]}</div>',
            unsafe_allow_html=True
        )

if not st.session_state.chat_history:
    st.info("💡 Ask any question about Class 8 Science!")
    st.markdown("""
    **Examples:**
    - "What is force?"
    - "What is photosynthesis?"
    - "What are microorganisms?"
    - "What is the greenhouse effect?"
    """)

st.markdown("---")

# =====================================================
# QUESTION INPUT
# =====================================================

with st.form(key="question_form", clear_on_submit=True):
    question = st.text_input(
        "Your Question",
        placeholder="Type your question here...",
        key="input",
        label_visibility="collapsed"
    )
    
    submitted = st.form_submit_button("📤 Ask", type="primary", use_container_width=True)
    
    if submitted and question:
        if not vector_store_loaded and not haystack_docs:
            st.error("❌ No vector store loaded. Please check your files.")
        else:
            with st.spinner("🤔 Thinking..."):
                try:
                    result = answer_question(
                        question,
                        vector_store=vector_store,
                        retriever=retriever,
                        text_embedder=text_embedder,
                        flant5_model=flant5_model,
                        flant5_tokenizer=flant5_tokenizer
                    )
                    
                    st.session_state.chat_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "question": question,
                        "answer": result['answer']
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #888;">'
    '📚 NCERT Class 8 Science • 🦙 Llama 2 (10min) • 🔧 FLAN-T5 • 💾 FAISS + Haystack'
    '</div>',
    unsafe_allow_html=True
)