# catalog-ai-assistant

> AI-powered sales assistant for **any store**: reads your catalog from Google Sheets (RAG) and uses a small fine-tuned model (QLoRA) to respond with consistent tone and formatting, without making up products.
>
> *English: a Google Sheets-powered sales chatbot that combines RAG (always-fresh prices and stock) with a LoRA fine-tuned small LLM (consistent tone, no hallucinated products).*

🚧 **Status: in development.** See the [roadmap](#roadmap) to know what is ready and what is still missing.

---

## The problem

A business has its catalog in a spreadsheet that changes every day: prices, stock, new products. Customers ask questions in many different ways:

- *"Do you have the IdeaPad 3?"*
- *"What computers do you have for less than 5000 Bs?"*
- *"What is the cheapest phone?"*

A generic chatbot responds with made-up or outdated information, and every response comes in a different format.

## The solution: RAG + fine-tuning, each for what it does best

| Component | What it is used for | Why |
|---|---|---|
| **RAG** | The *data* (price, stock, specs) | They change constantly; retraining every time is not viable |
| **Fine-tuning (QLoRA)** | The *behavior* (tone, format, saying "I can't find it") | It is stable, and a small model can learn it well |

The model **does not memorize products**: it learns to read the context provided by RAG and respond in a consistent format. If a price changes in the Sheet tomorrow, the assistant reflects it without retraining.

Example of the desired behavior:

\`\`\`
Customer:  Do you have the IdeaPad 3?
Assistant: Yes 😊 Lenovo IdeaPad 3 is available. It currently costs 4,500 Bs and
           we have 8 units available. Features: Ryzen 5, 8 GB RAM...

Customer:  Do you have the iPhone 15?
Assistant: I can't find that product in the current catalog.
\`\`\`

## Architecture

\`\`\`mermaid
flowchart LR
    A[Google Sheet<br/>catalog] --> B[Loading and cleaning]
    B --> C[Embeddings index<br/>stable text]
    Q[Customer question] --> D[Semantic search<br/>+ price filters]
    C --> D
    B -->|fresh price and stock| D
    D --> E[Context with<br/>retrieved products]
    E --> F[Small LLM<br/>with QLoRA fine-tuning]
    Q --> F
    F --> R[Response]
\`\`\`

## Design decisions

- **Only stable text is indexed** (name, category, brand, specs). Price and stock are read fresh from the catalog, so editing a price does not require re-indexing.
- **The same prompt format is used during training and production**, defined in a single module (`src/prompts.py`).
- **Anti-memorization:** each training example is generated using a version of the catalog with prices and stock randomly altered, so the model has to *read* the context.
- **Training responses are built from the context data**, and an automatic check verifies that no number in the response is absent from the context.
- **Discovery questions first:** customers rarely use the exact product name; most examples are of the type "what do you have for less than X", with synonyms and budgets written in different ways (`5000`, `5 thousand`, `5k`).
- **Question variations are reserved for testing:** there is a bank of phrases that the model never sees during training, in order to measure whether it generalizes or simply repeats templates.

## Project structure

\`\`\`
catalog-assistant/
├── data/
│   └── catalog_example.csv         # synthetic demonstration catalog
├── src/
│   ├── catalog.py                  # Sheet loading and cleaning, RAG documents
│   ├── prompts.py                  # shared prompt format (train + production)
│   └── dataset_builder.py          # fine-tuning dataset generator
├── notebooks/
│   ├── 01_catalog_rag.ipynb        # read the catalog and prepare it for RAG
│   └── 02_dataset_finetuning.ipynb # build and validate the dataset
├── requirements.txt
└── README.md
\`\`\`

## How to test it

\`\`\`bash
git clone https://github.com/AliciaV15/catalog-ai-assistant.git
cd catalog-assistant
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
\`\`\`

Open `notebooks/01_catalog_rag.ipynb`. To use **your own catalog**, share your Google Sheet as *"Anyone with the link can view"* and paste the link into the `SOURCE` variable. Required columns: `nombre`, `categoria`, `precio`, `stock`. Optional: `id`, `marca`, `specs`.

## Roadmap

- [x] Load and clean the catalog from Google Sheets or CSV
- [x] Shared prompt format
- [x] Dataset: discovery and product-specific questions
- [ ] Dataset: honesty cases (non-existent product, off-topic question, greeting) and train/val/test split
- [ ] Multi-store support: configuration per business and catalogs from different industries
- [ ] QLoRA fine-tuning (Google Colab)
- [ ] Retriever: embeddings + price and category filters
- [ ] Evaluation: base model vs. fine-tuned model, including an industry not seen during training
- [ ] Gradio app and deployment on Hugging Face Spaces
- [ ] Need-based requests ("something for gaming") through query understanding

## Evaluation

*Pending.* It will be measured on a test set containing products and phrases that the model has not seen:

| Metric | Base model + prompt | Fine-tuned |
|---|---|---|
| Correct response format | — | — |
| Exact price and stock | — | — |
| Responds "I can't find it" when appropriate | — | — |
| Meets budget filters | — | — |

## Known limitations

- Training questions are synthetic; they have not yet been tested with real customers.
- For now, there is only one example catalog (technology).
- The target language is Spanish.
