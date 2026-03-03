# Fine-Tuning-LLMs-with-LoRA-for-Schema-Aware-Structured-Data-Reasoning
Developed a schema-aware fine-tuning pipeline using LoRA to teach LLMs to reason over structured, tabular data. Trained models on schema-annotated SQL and natural language pairs, enabling recognition of column relationships, data types, and deterministic reasoning across tables.

Cryptic Tables:
<img width="2385" height="1112" alt="image" src="https://github.com/user-attachments/assets/74ad66b3-d1d9-4661-aab8-3dfcb9562f9c" />


<img width="2385" height="1227" alt="image" src="https://github.com/user-attachments/assets/d2d2be19-74be-4868-9377-89ef7f6cfacc" />

Demo:
<img width="1470" height="139" alt="image" src="https://github.com/user-attachments/assets/e5a3f0cc-efcc-4776-b1c5-50ef6e90ec94" />
This project provides a seamless natural language interface that allows non-technical stakeholders to query complex, cryptic enterprise databases using plain English. By integrating a LoRA-tuned model with an autonomous MCP agentic loop, the system successfully decodes non-semantic schemas, automatically mapping terms like "average amount" to AVG(Q_VAL) and "customer names" to S_LBL. This architecture enables the instant generation and execution of SQL queries, transforming raw, obscure ledger data into actionable business insights.

