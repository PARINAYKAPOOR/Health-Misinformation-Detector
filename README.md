# Health-Misinformation-Detector

Tags: NLP, LLMs, Web Dev
Data: Tweets / Reddit + WHO verified COVID/health info
Solution: Use fine-tuned language models to detect and flag misinformation in real-time. Create a browser extension or website that allows users to verify claims.
Impact: Health literacy, countering misinformation.

**Phase 1: Problem Framing & Data Collection (2 weeks)**

Goals:
-> Understand the scope of health misinformation online.

-> Gather misinformation vs. verified data samples.

Tasks:
1)- Literature review on misinformation detection (COVID, vaccines, supplements, etc.)

2)- Collect Twitter & Reddit health claims using APIs or open datasets (e.g., CoAID, ReCOVery, HealthStory).

3)- Collect ground truth data from WHO, CDC, NIH, and Snopes.

Deliverables:

1)- Problem brief with examples of misinformation


2- Dataset inventory & source attribution


3)- Bias/ethics consideration draft


**Phase 2: NLP Model Development:-**

Goals:
Build or fine-tune a model to classify claims as True, False, or Misleading.


Tasks:
1)- Preprocess and clean text data 


2)- Fine-tune a transformer model (e.g., BERT, RoBERTa, or DeBERTa) on misinformation datasets.


3)- Add explainability via attention weights 


Tools:
Sklearn, Pandas , Pretrained models: (bert-base-uncased, roberta-base, or twitter-roberta-base)

**Phase 3: Front-End Development & Integration (3 weeks)**

Tools:
Browser Extension: JavaScript, Manifest V3, Chrome Extension APIs
Web App: Streamlit or Flask + HTML/CSS
Backend: FastAPI + MongoDB for logs


