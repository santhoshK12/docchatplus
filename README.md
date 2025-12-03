# DocuChat++ — Hybrid RAG System with MiniLM & GPT
A Retrieval-Augmented Question Answering System for Long PDF Documents

## Project Overview
DocuChat++ is a hybrid Retrieval-Augmented Question Answering (RAG) system designed to answer queries from long PDF documents. It integrates semantic retrieval with both extractive and generative answering. The system supports efficient local inference using a fine-tuned MiniLM QA model and detailed reasoning using GPT in context-only mode. A Streamlit interface enables document upload, retrieval visualization, and answer comparison.

## Key Features
- PDF upload and ingestion  
- Text extraction and chunking  
- Dense embeddings using Sentence Transformers  
- FAISS vector store for semantic retrieval  
- MiniLM extractive QA model (fine-tuned on SQuAD)  
- GPT-based reasoning constrained to retrieved context  
- Answer modes: MiniLM, GPT, and comparison  
- Streamlit UI for transparent evaluation

## Tools & Technologies Used
- Python 3.10+  
- PyTorch  
- Hugging Face Transformers  
- Sentence Transformers  
- Hugging Face Datasets (SQuAD)  
- FAISS  
- PyPDF2 / PDFPlumber  
- Streamlit  
- OpenAI GPT  
- Google Colab (GPU for training)  
- Git / GitHub

# How the Project Was Executed (Full Workflow)

## 1. Dataset Preparation and Model Training
- Loaded and preprocessed the SQuAD dataset using Hugging Face Datasets.  
- Fine-tuned a MiniLM model using the Transformers Trainer API on Google Colab GPU.  
- Evaluated model performance and saved the best-performing checkpoint.  
- Exported the trained MiniLM model and integrated it into the application for local QA.

## 2. Retrieval Pipeline Development
- Extracted text from PDFs using PyPDF2/PDFPlumber.  
- Performed text cleaning and chunking with overlap to preserve context.  
- Generated dense vector embeddings for all chunks using Sentence Transformers.  
- Built a FAISS index for efficient semantic similarity search.  
- Implemented query embedding + top-k retrieval for evidence selection.

## 3. MiniLM Extractive QA Integration
- Integrated the fine-tuned MiniLM model for local span extraction.  
- Model predicts start/end positions to extract exact answer spans from retrieved chunks.  
- Provides fast, grounded, short factual answers.

## 4. GPT Generative Reasoning Integration
- Used OpenAI GPT models in context-only mode.  
- Passed only retrieved document chunks and user questions.  
- Generated multi-sentence grounded explanations suitable for complex queries.  
- Ensured no external knowledge was used, reducing hallucination risk.

## 5. Streamlit Web Application
- Developed a user-friendly interface for PDF upload, ingestion, and retrieval.  
- Implemented answer mode switching (MiniLM, GPT, Both).  
- Enabled optional debugging to display retrieved chunks.  
- Integrated error handling and environment variable-based API key management.

## 6. Evaluation and Testing
- Evaluated system performance using legal text from the Constitution of India (Articles 5–8).  
- Used three representative questions to test fact extraction and reasoning.  
- Observed:  
  - MiniLM produced short, concise answers.  
  - GPT produced detailed, multi-sentence explanations grounded in retrieved text.  
- Hybrid approach delivered the most reliable results.

## 7. Final Deliverables
- 20+ slide project presentation (PPTX)  
- IEEE-format final project report (PDF)  
- Complete GitHub repository with source code  
- README documentation  
- Live demonstration using Streamlit app

# How to Run the Project (Local Execution Guide)

## 1. Create Virtual Environment
python -m venv .venv
..venv\Scripts\activate

shell
Copy code

## 2. Install Dependencies
pip install -r requirements.txt

graphql
Copy code

## 3. Set OpenAI API Key
setx OPENAI_API_KEY "YOUR_KEY_HERE"

sql
Copy code
Open new CMD:
echo %OPENAI_API_KEY%

shell
Copy code

## 4. Run the Streamlit Application
streamlit run app.py

bash
Copy code

# Project Structure
docuchat_rag-main/
│── app.py
│── main.py
│── requirements.txt
│── models/
│ └── minilm_best/
│── ml/
│ ├── train_qa_models.py
│ └── utils.py
│── index/
│ ├── index.faiss
│ ├── meta.npy
│── PDF samples/

markdown
Copy code

# Conclusion
DocuChat++ successfully demonstrates a hybrid RAG approach combining semantic retrieval, extractive QA, and generative reasoning. The MiniLM model provides fast, factual extraction, while GPT handles complex reasoning grounded in retrieved evidence. The Streamlit interface offers transparency and ease of use for document-based QA tasks.

# Team Contributions
### **Santhosh Reddy Kistipati**
- Implemented entire Streamlit application.  
- Integrated MiniLM extractive QA model into the pipeline.  
- Integrated GPT reasoning module with context-only prompts.  
- Built FAISS vector index and embedding pipeline.  
- Conducted evaluation, testing, and presented the final demo.  
- Prepared full PPT and designed system architecture.

### **Ashimitha**
- Assisted with literature review and background study for the report.  
- Helped prepare sections of the IEEE paper including introduction, problem statement, and conclusion.  
- Contributed to proofreading and refining the presentation slides.

### **Sunil**
- Helped with dataset exploration and document preprocessing steps.  
- Supported extraction and cleaning of PDF text.  
- Assisted in evaluation by running multiple queries and analyzing model behavior.  
- Contributed to discussion and observations for the results section.

All members participated in discussions and decision-making throughout the development of the project.

