# 🚀 GenFactory
### Autonomous Synthetic Data Pipeline & Model Fine-Tuning Framework

GenFactory is an end-to-end, production-grade framework for **synthetic data generation and model distillation** using a **Teacher–Student architecture**.

The system leverages **Gemini (Teacher models)** to autonomously generate and evaluate high-quality, domain-specific datasets via a **self-healing LangGraph orchestration loop**, and then fine-tunes a lightweight **Phi-3 (Student model)** for efficient deployment in niche domains such as **Python automation**, **FinTech security**, and other specialized technical areas.

This project is designed with **industry-grade reliability, observability, and scalability** in mind.

---

## ✨ Key Features

### 🧠 Autonomous Orchestration
- LangGraph-powered stateful workflow
- Generator → Judge → Retry → Accept execution loop
- Automatic retry handling and failure recovery

### 🛡️ Self-Healing Data Quality
- Dedicated **Judge LLM (Gemini 1.5 Flash)**
- Multi-metric evaluation (relevance, correctness, clarity)
- Only samples scoring **≥ 8/10** are accepted
- Controlled retry logic for weak outputs

### 💾 Persistent Memory & Crash Safety
- **Redis Stack**–based checkpointing (RedisJSON + RediSearch)
- Safe resume for long-running generation jobs
- Durable state storage across batches

### 🧪 Hardware-Aware Fine-Tuning
- Apple Silicon (M3) optimized via **MPS + FP16**
- NVIDIA cluster training via **QLoRA + bf16**
- Separate pipelines for laptop and supercomputer training

### 📊 Industry-Level Observability
- Full tracing with **LangSmith**
- Inspect every LLM decision and graph transition
- Cost, retry, and quality visibility

---

## 🏗️ Technical Architecture

The system is structured into **four industrial layers**, each with a single responsibility:

### Level 1 — Orchestration Brain
**LangGraph + Redis Stack**
- Stateful control flow
- Conditional routing and retries
- Crash-safe checkpointing

### Level 2 — Data Engineering
**JSONL Sink + Hugging Face Hub**
- Deterministic formatting
- Incremental dataset construction
- Automated versioned publishing

### Level 3 — Domain Training
**SFT + LoRA / QLoRA**
- Phi-3 fine-tuning
- Hardware-specific optimization paths

### Level 4 — Evaluation Loop
**Accuracy Gain Benchmarking**
- Base vs fine-tuned model comparison
- Promotion gating
- Regression detection

---

## 📂 Project Structure

```text
synthetic_data_factory/
├── src/
│   ├── graph/                  # LangGraph orchestration
│   │   ├── state.py             # State schema (retry_count, score, etc.)
│   │   ├── nodes.py             # Gemini Generator & Judge logic
│   │   ├── edges.py             # Conditional routing rules
│   │   └── workflow.py          # Compiled LangGraph + Redis Saver
│   │
│   ├── data_eng/               # Data engineering layer
│   │   ├── formatter.py         # JSONL / ChatML standardization
│   │   └── hf_uploader.py       # Hugging Face dataset sync
│   │
│   ├── training/               # Fine-tuning pipelines
│   │   ├── train_laptop.py      # Apple Silicon (M3, MPS, FP16)
│   │   └── train_super.py       # NVIDIA A100 (CUDA, QLoRA, bf16)
│   │
│   └── main.py                 # Synthetic data factory entry point
│
├── .env                        # Secrets (Gemini, HF, LangSmith)
├── requirements.txt            # Dependency manifest
└── submit_job.sh               # SLURM job submission script
