GenFactory: Autonomous Synthetic Data Pipeline & Model Fine-Tuning
GenFactory is an end-to-end industrial pipeline designed to generate high-quality synthetic datasets using a Teacher-Student architecture. It leverages Gemini 2.0 (Teacher) to produce and evaluate data via a self-healing LangGraph orchestration loop, which is then used to fine-tune a specialized Phi-3 (Student) model for niche domains like Python automation and Fintech security.

🌟 Key Features
Autonomous Orchestration: Uses LangGraph to manage a generator-evaluator loop with built-in retry logic.

Self-Healing Data Quality: Implements a "Judge" LLM (Gemini 1.5 Flash) that only accepts data scoring ≥8/10.

Persistent Memory: State-aware checkpointing using Redis Stack to handle crashes and long-running batches.

Hardware-Aware Training: Optimized training paths for Apple Silicon (M3 MPS) and NVIDIA Clusters (SLURM/QLoRA).

Industry Observability: Full tracing of every LLM decision via LangSmith.

🏗️ Technical Architecture
The project is structured into four distinct industrial levels:

Level 1: Orchestration Brain (LangGraph + Redis)

Level 2: Data Engineering (JSONL Sink + Hugging Face Hub)

Level 3: Domain Training (SFT + LoRA/QLoRA)

Level 4: Evaluation Loop (Accuracy Gain Benchmarking)

📂 Project Structure
Plaintext
synthetic_data_factory/
├── src/
│   ├── graph/               # Orchestration logic
│   │   ├── state.py         # State schema (retry_count, score)
│   │   ├── nodes.py         # Gemini Generator & Evaluator logic
│   │   ├── edges.py         # Conditional routing (Retry vs. Format)
│   │   └── workflow.py      # Compiled LangGraph with Redis Saver
│   ├── data_eng/            # Data persistence layer
│   │   ├── formatter.py     # JSONL standardization
│   │   └── hf_uploader.py   # Automated Hugging Face syncing
│   ├── training/            # Hardware-specific fine-tuning
│   │   ├── train_laptop.py  # Optimized for M3 (MPS/float16)
│   │   └── train_super.py   # Optimized for A100 (CUDA/QLoRA/bf16)
│   └── main.py              # Entry point for generation
├── .env                     # Secrets (Gemini, HF, LangSmith)
├── requirements.txt         # Dependency manifest
└── submit_job.sh            # SLURM script for cluster deployment
🚀 Getting Started
1. Prerequisites

Python: 3.10+

Redis: Redis Stack (required for JSON checkpointing)

API Keys: Google AI Studio (Gemini), Hugging Face (Write Token), LangSmith.

2. Installation

Bash
git clone https://github.com/prashantshukla2410/GenFactory.git
cd GenFactory
pip install -r requirements.txt
3. Environment Setup

Create a .env file in the root:

Bash
GOOGLE_API_KEY="your_key"
HF_TOKEN="your_token"
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="your_key"
REDIS_URL="redis://localhost:6379"
🛠️ Usage
Step 1: Generate Synthetic Data

Run the factory to start the Gemini 2.0 Teacher-Evaluator loop:

Bash
python src/main.py
Step 2: Local Fine-Tuning (Laptop)

Train the Phi-3 student model on your MacBook M3:

Bash
python src/training/train_laptop.py
Step 3: Cluster Fine-Tuning (Supercomputer)

Submit the job to your college department's NVIDIA cluster:

Bash
sbatch submit_job.sh
📊 Evaluation & Metrics
The pipeline includes a "Promotion Gate" that compares the base model against the fine-tuned adapter.

Target Accuracy Gain: +15% over base model.

Quality Threshold: Only data with a score ≥8 is used for training.

📜 License & Acknowledgements
License: MIT


Author: Prashant Shukla