# Swiss Legal AI Assistant

A state-of-the-art legal assistance system combining:
- **Hybrid document retrieval** (vector + keyword search)
- **Fine-tuned Mistral-7B** for legal response generation
- **8k context understanding** of Swiss legal documents

## Features

### 🚀 Core Capabilities
- **Multilingual Legal Search** (German/French/English)
- **Precision Retrieval** with reciprocal rank fusion
- **Article Citation** in generated responses
- **Query Expansion** using legal thesaurus

### 📚 Supported Domains
- Data Protection (FADP/GDPR)
- Contract Law (OR/Swiss Code of Obligations)
- Federal Court Rulings
- Administrative Law

## System Architecture

```mermaid
graph LR
    A[User Query] --> B{Hybrid Retrieval}
    B --> C[Vector Search]
    B --> D[Keyword Search]
    C --> E[Reciprocal Rank Fusion]
    D --> E
    E --> F[Context Augmentation]
    F --> G[Mistral-7B Generation]
    G --> H[Response Formatting]
