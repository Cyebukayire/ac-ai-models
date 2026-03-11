# AI Models of the Tunga Chatbot

In Rwanda, many farmers struggle to access timely, personalized agricultural information. Traditional channels like radio, TV, and online sources, offer limited reach and interactivity, while extension services and a national call center, staffed by only two agents for over two million farmers, face capacity constraints. To address these gaps, [C4IR Rwanda](https://c4ir.rw), with support from [GIZ](https://www.giz.de/) [Fair Forward](https://www.bmz-digital.global/en/overview-of-initiatives/fair-forward/) (funder) and [KiNLP](https://kinlp.com/) (technology partner), developed a 24/7 AI-enabled Interactive Voice Response (IVR) tool for the [Ministry of Agriculture and Animal Resources](https://www.minagri.gov.rw/). Accessible via a Kinyarwanda-speaking hotline, this tool will provide critical support such as pest and disease diagnosis, agro-climatic advisories, and updates on MINAGRI’s programs, including crop insurance and climate-resilient practices. By utilizing AI and IVR technology, this project will make agricultural advisories more accessible, timely, and responsive to farmers’ needs. For more information, please reach out to [C4IR](https://c4ir.rw/).

Implemented by: [C4IR Rwanda](https://c4ir.rw/) & [KiNLP](https://kinlp.com/); Supported by [GIZ](https://www.giz.de/); Financed by: [BMZ](https://www.bmz.de/en).

## Repository Overview

[You can find the technical documentation in this folder.](/DeepKIN-AgAI)

The **DeepKIN-AgAI** repository provides a specialized suite of models designed to bridge the gap between spoken Kinyarwanda and digital agricultural intelligence. By integrating morphological awareness with state-of-the-art neural architectures, this project enables high-accuracy Retrieval-Augmented Generation (RAG) and Interactive Voice Response (IVR) systems.

The repository contains three primary model categories:

## KinyaBERT (Language Understanding)

At the heart of the system is **KinyaBERT**, a morphology-aware language model. Unlike standard BERT models, KinyaBERT is specifically optimized for the complex grammatical structure of Kinyarwanda. It serves as the "brain" for understanding text and is available in two sizes:

* **Base:** A 107M parameter model for efficient processing.  
* **Large:** A 365M parameter model for higher accuracy and deeper linguistic nuance.

## KinyaColBERT (Retrieval)

This is a fine-tuned version of KinyaBERT using the **ColBERT** (Contextualized Late Interaction over BERT) architecture. It is designed for **Dense Retrieval**, allowing the chatbot to:

* Search through massive agricultural knowledge bases.  
* Match user queries to the most relevant technical documents with high precision.  
* Utilize lexically grounded embeddings to handle low-resource language constraints effectively.

## Flex-TTS (Speech Synthesis)

To support IVR (voice-based) interactions, the repository includes **Flex-TTS**, a multi-speaker Text-to-Speech engine. This model converts the retrieved text answers into natural-sounding Kinyarwanda speech, making agricultural advice accessible to users who prefer or require audio communication over text.

## Supporting Tools

In addition to the neural models, the repository leverages **MorphoKIN**, a critical tool for morphological disambiguation and parsing. This ensures that the models correctly interpret Kinyarwanda's prefix and suffix systems before the data is processed by the transformers.

Would you like me to help you draft a specific "Model Card" summary for any of these individual models to include in your documentation?

## License

This code and the models are released unter the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/)
