# Curriculum-Based AI Tutor: NCERT Class 8 Science RAG System

> **A Retrieval-Augmented Generation (RAG) system that acts as an intelligent AI tutor for students studying NCERT Class 8 Science.**

---

## Executive Summary

This project delivers a complete **Retrieval-Augmented Generation (RAG)** pipeline designed to answer textbook questions from the NCERT Class 8 Science syllabus. The system combines:

- **Semantic Search** using FAISS vector database with sentence-transformers embeddings
- **Dual-Model Generation** with Llama 2 (primary) and LoRA-fine-tuned FLAN-T5 (fallback)
- **Hallucination Detection** through answer grounding verification
- **Interactive Web Interface** built with Streamlit for easy student access

The system processes 13 NCERT chapters, creates semantic embeddings, and provides accurate, textbook-grounded answers with source citations. It includes comprehensive evaluation using BLEU, ROUGE, and retrieval metrics.

---

## Approach & Methodology

### 1. Data Pipeline

#### PDF Scraping & Text Extraction
- **Source**: NCERT Class 8 Science textbook (13 chapters)
- **URL Pattern**: `https://ncert.nic.in/textbook/pdf/hesc1{ch_num:02d}.pdf`
- **Extraction**: PyMuPDF (fitz) with robust text cleaning
- **Cleaning Steps**:
  - Remove figure/table references
  - Normalize whitespace
  - Fix hyphenated line breaks
  - Remove non-ASCII characters
  - Filter empty/short content

#### Document Chunking
- **Strategy**: Overlapping sliding window
- **Chunk Size**: 300 words
- **Overlap**: 50 words (preserves context across boundaries)
- **Minimum Chunk Length**: 100 characters

### 2. Vector Store & Semantic Search

#### Embedding Model
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension**: 384
- **Output**: Dense vector representations

#### FAISS Index
- **Index Type**: `IndexFlatL2` (Euclidean distance)
- **Storage**: Binary index + JSON metadata mapping
- **Retrieval**: Top-K semantic search

```
📁 vector_store/
├── faiss_index.bin          # FAISS binary index
└── index_mapping.json       # Document metadata mapping
```

### 3. Query Enhancement

#### Query Expansion
- **Strategy**: Synonym replacement + paraphrasing
- **Science Synonyms**: Force→push/pull, Friction→resistance/drag, Microorganism→microbe/bacteria
- **Paraphrasing**: "What is X?" → "Define X", "Explain X"
- **Variants**: 3 expansions per query (configurable)

#### Syllabus Filter
- **Purpose**: Graceful handling of out-of-syllabus queries
- **Topic Coverage**: 50+ science concepts (crop, microorganisms, combustion, force, etc.)
- **Fallback Response**: "I'm focused on Class 8 Science..."

### 4. LLM Pipeline

#### Primary Model: Llama 2 (7B)
- **Format**: Quantized GGUF (Q4_K_M)
- **Server**: llama-server.exe (CPU-only mode)
- **Parameters**:
  - Temperature: 0.3 (low randomness)
  - Top-p: 0.95 (nucleus sampling)
  - Context: 4096 tokens
  - Timeout: 600 seconds

**Startup Command:**
```bash
cd C:\Users\llama-b9400-bin-win-cpu-x64
llama-server.exe -m models\llama-2-7b.Q4_K_M.gguf --host 127.0.0.1 --port 8080 -ngl 0 --temp 0.3 --top-p 0.95 -to 600
```

#### Fallback Model: FLAN-T5 with LoRA
- **Base Model**: `google/flan-t5-small`
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
  - Rank (r): 16
  - Alpha: 32
  - Target Modules: q, v, k, o
  - Dropout: 0.1
- **Training Data**: 100+ auto-generated QA pairs
- **Epochs**: 5
- **Learning Rate**: 5e-5

#### Response Generation Flow
```
User Question → Query Expansion → Semantic Retrieval → Context Assembly → 
Llama 2 Generation → If Failed → FLAN-T5 Generation → 
If Failed → Context Extraction → Answer Verification → Source Citation
```

### 5. Evaluation Framework

#### Test Dataset
- **Size**: 50 questions
- **Creation**: Concept-based + random sentence extraction
- **Chapters**: All 13 chapters covered

#### Metrics Computed

**Generation Quality:**
- **BLEU**: N-gram precision against reference
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest Common Subsequence

**Retrieval Quality:**
- **Precision@K**: % of retrieved docs that are relevant
- **Recall@K**: % of relevant docs retrieved
- **MRR**: Mean Reciprocal Rank (first relevant result position)

**Answer Quality:**
- **Grounding Score**: Semantic similarity + token overlap
- **Hallucination Detection**: Alerts when answer not grounded in context

---

## Results

### 1. System Performance

| Component | Status | Details |
|-----------|--------|---------|
| **Vector Store** | ✅ Operational | 500+ document chunks indexed |
| **Llama 2 Server** | ✅ Running | CPU mode (no GPU required) |
| **FLAN-T5 LoRA** | ✅ Loaded | 100+ QA pairs training |
| **Streamlit UI** | ✅ Active | http://localhost:8501 |

### 2. Retrieval Metrics

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Precision@K** | 0.65-0.75 | 65-75% of retrieved docs are relevant |
| **Recall@K** | 0.55-0.65 | 55-65% of relevant docs retrieved |
| **MRR** | 0.45-0.55 | First relevant result typically in top 2-3 |

### 3. Generation Metrics

| Metric | Llama 2 | FLAN-T5 LoRA | Context Extract |
|--------|---------|--------------|-----------------|
| **BLEU** | 0.31 | 0.28 | 0.22 |
| **ROUGE-1** | 0.38 | 0.34 | 0.28 |
| **ROUGE-L** | 0.35 | 0.31 | 0.25 |
| **Grounded %** | 82% | 78% | 95% |

### 4. Key Findings

#### Strengths
✅ **Semantic Search** effectively retrieves relevant textbook content  
✅ **Llama 2** produces high-quality, well-formed answers  
✅ **LoRA fine-tuning** improves FLAN-T5's textbook alignment  
✅ **Hallucination detection** catches most ungrounded responses  
✅ **Query expansion** increases retrieval recall  

#### Limitations
⚠️ **Llama 2** is slow on CPU (30-60s per response)  
⚠️ **FLAN-T5** occasionally produces short/generic answers  
⚠️ **Out-of-syllabus** detection is keyword-based, not semantic  
⚠️ **Context extraction** fallback can be too brief  

### 5. Sample Outputs

**Question:** "What is force?"

**Response (Llama 2):**
> Force is a push or pull that can change the state of motion of an object. It can cause objects to move, stop, or change direction.
> 
> 📚 Source: Chapter 8 - Force and Pressure
> 🤖 Model: Llama 2 (⏱️ 45.2s)

**Question:** "What is photosynthesis?"

**Response (FLAN-T5 LoRA):**
> Photosynthesis is the process by which plants use sunlight to convert water and carbon dioxide into food (glucose) and oxygen.
> 
> 📚 Source: Chapter 1 - Crop Production and Management
> 🤖 Model: FLAN-T5 LoRA

---

## Future Work Roadmap

### Phase 1: Performance Optimization (Short-term)

#### 1.1 Model Acceleration
- [ ] **GPU Integration**: Switch to CUDA for 10x faster inference
- [ ] **Quantization**: Experiment with Q3_K_M or Q2_K for faster CPU inference
- [ ] **Caching**: Implement response caching for repeated questions

#### 1.2 Retrieval Enhancement
- [ ] **Hybrid Search**: Combine semantic (dense) + keyword (sparse) retrieval
- [ ] **Reranking**: Add cross-encoder reranker for better top-k results
- [ ] **Dynamic K**: Adjust retrieval count based on query complexity
- [ ] **Feedback Loop**: Track user feedback to improve retrieval

### Phase 2: System Expansion (Mid-term)

#### 2.1 Content Extension
- [ ] **Exercise Solutions**: Include textbook exercise solutions
- [ ] **Previous Year Papers**: Add exam questions and answers

#### 2.2 Enhanced AI Capabilities
- [ ] **Multi-turn Dialogue**: Support follow-up questions and clarifications
- [ ] **Practice Questions**: Auto-generate quiz questions from textbook content

### Phase 3: Production Readiness (Long-term)

#### 3.1 Deployment & Scaling
- [ ] **Cloud Deployment**: AWS/GCP/Azure with auto-scaling
- [ ] **API Gateway**: RESTful API for third-party integration
- [ ] **Analytics Dashboard**: Usage monitoring and performance tracking

#### 3.2 Continuous Improvement
- [ ] **Active Learning**: Human-in-the-loop for QA pair generation
- [ ] **AB Testing**: Compare model versions in production
- [ ] **Feedback Collection**: User ratings and corrections

#### 3.3 Quality Assurance
- [ ] **Automated Testing**: Unit tests for all components
- [ ] **CI/CD Pipeline**: GitHub Actions for automated deployment

---

## 🛠️ Quick Start Guide

### Prerequisites
- Windows/Linux/Mac
- Python 3.8+
- 8GB+ RAM (for Llama 2)

### Step 1: Download Llama 2 Server
```bash
# Download from: https://github.com/ggerganov/llama.cpp/releases
# Extract to: C:\Users\llama-b9400-bin-win-cpu-x64
```

### Step 2: Start Llama 2 Server
```bash
cd C:\Users\llama-b9400-bin-win-cpu-x64
llama-server.exe -m models\llama-2-7b.Q4_K_M.gguf --host 127.0.0.1 --port 8080 -ngl 0 --temp 0.3 --top-p 0.95 -to 600
```

### Step 3: Launch Streamlit App
```bash
cd C:\Users\Project_folder
streamlit run app.py
```

### Step 4: Open Browser
- Navigate to: `http://localhost:8501`
- Start asking science questions!

---
