# Catalog AI Assistant

> AI-powered sales assistant that combines **RAG** with **QLoRA fine-tuning** to answer product questions using up-to-date catalog information.

🚧 **Status: In development**

## Overview

Businesses often manage their products in spreadsheets where **prices and stock change frequently**. This project combines RAG and fine-tuning to handle these two types of information separately:

| Component | Purpose |
|---|---|
| **RAG** | Retrieves current product information such as price, stock, and specifications |
| **QLoRA** | Teaches the model consistent response behavior, tone, and formatting |

The model does **not memorize products**. Instead, it learns to use the context retrieved from the catalog, allowing product information to stay up to date without retraining the model.

### Example

```text
Customer: Do you have the IdeaPad 3?

Assistant: Yes! The Lenovo IdeaPad 3 is available for
4,500 Bs, with 8 units in stock.

Customer: Do you have the iPhone 15?

Assistant: I couldn't find that product in the current catalog.