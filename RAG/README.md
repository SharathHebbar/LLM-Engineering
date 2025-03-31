# RAG: Retrieval Augmented Generation

## What

RAG or Retrieval Augmented Generation is an approach that combines LLM with external data sources.
It enhances the models responses by retrieving relevant information from a database or search system and using that data to generate more accurate and contextually relevant outputs.

## Core Concepts

1. External Knowledge Intergration: Combining language models with external data sources for more informed responses.
2. Dynamic Information Retrieval: Automatically fetching relevant data from databases or web sources during interaction.
3. Contextual Response Generation: Using retrieved data to craft more accurate and context aware answers.

## Principles

1. Grounding LLMs outputs to factual information: Ensuring that generated responses are based on reliable and up-to-date external data.
2. Enhancing model knowledge without retraining: Using retrieval mechanisms to improve model performance without the need for extensive retraining on new datasets.

## Advantages of RAG over Traditional LLM

Aspects | Traditional LLMs | RAG 
| :---: | :---: | :---: 
Knowledge Source | Internal parameters are the weights and biases in a model that are adjusted during training to define its behaviour and outputs | External parameters combines a models internal paramters with information retrieved from external sources to improve responses
Updatability | Requires retraining means a model needs to be re-trained to update or improve its knowledge. | Easy to update means a model can quickly incorporate new information without requiring re-training. 
Factual Accuracy | Can be inconsistent means the models outputs might vary or lack reliability. | Generally more accurate means the model consistently provides correct and reliable responses.
Black Box | Black box referes to a system or model where the internal workings are not visible or understandable, making it difficult to see how decision are made. | Provides sources means the model cites or references the external information it uses, offering transparency and traceability for its responses.


## Use Cases of RAG

1. Enhanced Content Creation
2. Customer Feedback Analysis
3. Market Intelligence
4. Personalized Recommendations
5. Dialogue Systems and Chatbots

## Role of RAG in Enhancing LLM Capabilities

### Improving factual consistency

1. Without RAG: An LLM might produce incorrect responses like "The capital of Australia is Sydney." relying only on its internal knowledge
2. With RAG: The LLM retrieves up-to-date information from trusted sources, generating accurate responses lik, "The capital of Australia is Canberra."

#### How RAG grounds responses in retrieved information

- RAG enhances LLMs by integrating external information retrieval for more accurate responses.
Steps
1. Query Generation: The model creates a query from the input
2. Information Retrieval: Relevant data is fetched from external sources.
3. Responses Generation: The LLM uses this data to generate a fact-based response.

### Reducing hallucinations

- Define Hallucination: In the context of LLM, a hallucination refers to the generation of incorrect or fabricated information that the model produces, which is not based on actual facts or reliable sources.

Techniques RAG uses to Mitigate False Information.
1. Contextual Validation: Cross-checks data from multiple sources to improve accuracy and consistency.
2. Dynamic Querying: Generates context-specific queries to retrieve precise, relevant information.
3. Information Retrieval: Fetches relevant data from external sources, reducing false information by grouding responses in factual content.
4. Post processing checks: Applies checks after response generation to correct inaccuracies before delivery.

### Enhancing domain-specific knowledge

Adapting General LLMs to specialized fields
1. Fine Tuning: Adapts the LLM to specialized knowledge by training it on domain-specifice datasets (Ex: Medical Texts for healthcare queries).
2. Prompt Engineering: Crafts detailed prompts to guide the LLM in generating accurate, field-specific responses (Ex: Legal or Technical Advice).
3. Domain-Specific Retrieval: Integrated specialized knowledge bases (Ex: Legal Databases) for precise, relevant information retrieval.

Case Studies of RAG in Various Industries

1. Healthcare: Retrieves the latest medical research to provide accurate treatment options and drug interaction advice, improving clinical decision-making.
2. Finance: Pulls data from market reports for informed investment advice, enhancing accuracy in financial forecasting.
3. Legal: Accesses current case law and precedents improving legal research and document drafting.
4. Customer Support: Uses a companys' knowledge base to deliver accurate answers, improving customer support interactions.


# Vector Databases and Embeddings

## What is Vector Databases

A vector database stores and queries vector embeddings, numerical representations of data like text, images, or audio. Generated by ML models, these embeddings capture semantic meaning and enable efficient similarity search.

## Key Concepts

1. Vector Embeddings: Numerical representation (multi-dimensional vectors) that encode semantic properties of objects. Generated by models like BERT, GPT or CNNs, they represent data in a compact, machine-readable form.
2. Similarity Search: Finds vectors close to the query using metrics like:
    - Cosine Similarity: Measures the angle between vectors.
    - Euclidean Distance: Measures straight-line distance.
    - Dot Product: Measures similariy based on vector multiplication.
3. Use Cases:
    - Recommedation Systems: Retrieve similar products, media or content.
    - Search Augmentation: Retrieve semantically relevant results even without keyword matches.
    - NLP: Embed queries and documents to find relevant text.


## Core Components of Vector Databases

1. Storage: Designed to store large numbers of high dimensional embeddings sometimes with thousands of dimensions.
2. Querying: Focused on similarity-based search, finding vectors closest to the query for relevant results.
3. Indexing: Uses techniques like ANN(Approximate Nearest Neighbor) algorithms (Ex: FAISS, HNSW) and partitioning to enable fast searches and reduce computational cost.
4. Scalability: Optimized to handle massive datasets, scaling to millions and billions of vectors.

## Roles in RAG Systems

### Efficient Storage for Document Embeddings

1. Storage: RAG systems generate embeddings for documents, which are stored in vector databases. These embeddings capture the semantic content of the documents in a format that can be efficiently managed, even with large volumes of data.
2. Scalability: Vector databases are designed to handle extensive datasets, making them ideal for RAG systems that require storing embeddings for entire knowledge bases or documnet collections.

### Fast Similarity Search for Relevant Information Retrieval

1. Similarity Search: When a query is received, the RAG system generates a query embedding. The vector database performs a fast similarity search, comparing this embedding against the stored document embeddings.
2. Nearest Neighbor Search: The database identifies the most semantically similar documents or passages to the query, enabling the retrieval of the most relevant information.
3. Real Time Performance: The speed and efficiency of vector databases ensure that relevant documents can be retrievd quickly, supporting real-time applications like chatbots, search engines and recommendation systems.

## Types of Vector Databases

1. FAISS - Facebook AI Similarity Search: An open source library developed by Facebook AI that supports efficient similarity search and clustering of dense vectors.
2. Pinecone: A fully managed vector database services designed for fast similarity search, including support for real-time use cases.
3. Weaviate: An open source vector search engine that combines vector embeddings with traditional search techniques.
4. Milvus: A cloud-native, open-source vector database optimized for large-scale embedding data and ML workloads.

## Use Cases Beyond RAG

1. Recommendation Systems: Vector embeddings help match user preferences with similar products, media, or content, enhancing personalization.
2. Image Search: Embeddings of images allow finding visually similar images based on content rather than keywords.
3. Anomaly Detection: Detects outliers by identifying vectors that deviate from the norm in financial transactions, manufacturing, or cybersecurity systems.

# Embeddings

## Definition
Embeddings are dense vector representations of data (Ex: Text, Image, Audio) that capture semantic meaning. Generated by machine learning models, they position similar data points closer in vector space. This enables tasks like similarity search and classification.

## How embedding works

1. Visualization: Word and sentence embeddings can be visualized in a lower dimensional space (Ex. 2D or 3D) using techniques like PCA or t-SNE, showing how related concepts cluster together based on meaning.
2. Capturing Semantic Meaning: Embeddings transform data into high-dimensional vectors, where similar data points(Ex. Words, sentences) are positioned closed together, reflectin their semantic similarity.

## Different Embedding Models

1. Word2Vec: Generated Embeddings by predicting words in a context or using words to predict their surrouding words. Produces fixed-size vectors based on co-occurence in a corpus.
2. GloVe: Creates embeddings by factorizing the word co-occurrence matrix of a corpus, capturing global statistical information about word pairs.
3. BERT Based Embeddings: Provides context-aware embeddings by considering the entire sentence or passage, allowing for nuanced understanding of word meanings in different contexts.

# Indexing and Similarity Search

Indexing structures embeddings for fast access, while similarity search uses metrics like cosine similarity to find the closest matches. These processes enable efficient and accurate data retrieval in AI applications.

## Techniques for Efficient Retrieval

### FAISS (Facebook AI Similarity Search)

1. IVF (Inverted File Index): Partitions vector space into clusters, searching only within relevant ones.
Advantages: Balances speed and accuracy, ideal for large datasets.

2. HNSW (Hierarchical Navigable Small World): Uses a multi-layered graph for efficient data search.
Advantages: High Accuracy and fast searches especially in high-dimensional spaces.

### ANNOY (Approximate Nearest Neighbors oh Yeah)

- Description: A tree based indexing method that uses random projections to build multiple trees, providing a fast and memory efficient way to find approximate nearest neighbors.
- Advantages: Optimized for scenarios where retrieval speed is crucial, even at the cost of some accuracy.

## Speed V/S Accuracy

| Techniques | Accuracy | Speed
| :---: | :---: | :---: 
| HNSW | Known for high accuracy, making it suitable for tasks where precision is critical. | Offers a fast search with high accuracy, though it can be more computationally intensive to build the index.
| IVF | Provides a balanced approach with decent accuracy and speed, suitable for a wide range of applications. | Balances speed with reasonable accuracy, making it a versatile choice for large-scale searches.
| ANNOY | Focuses more on speed, which can lead to less accurate results in some cases. | Prioritizes speed, making it ideal for real-time applications where rapid responses are needed, though it may sacrifice some accuracy.

## Similarity Metrics

1. Cosine Similarity
    - Description: Measures the angle between vectors, focusing on direction.
    - When to use: Text analysis and high-dimensional spaces where direction is key.

2. Euclidean Distance:
    - Description: Calculates the straight-line distance between points, considering both direction and magnitude.
    - When to use: Clustering and classification in low-dimensional spaces.

3. Dot product:
    - Description: Computes the product of vector magnitudes and the cosine of the angle.
    - When to use: Large-scale searches and recommendation systems with normalized vectors.

# Basic Building Blocks of RAG Pipeline

## Overview

1. Document Corpus: The collection of documents or knowledge base from which information will be retrieved.
2. Language Model: The generative AI model that produces responses.
3. Embedding Model: Used to convert text into vector representations.
4. Prompt Templates: Structures how retrieved information and the query are presented to the language model.
5. Vector Database: Stores and indexes the embedded documents for efficient similarity search.
6. Orchestrator: Coordinates the flow of information between components.
7. Retriever: Responsible for finding relevant documents based on the input query.

## Document Loaders and Text Splitting

### Types of Document Loaders

1. PDF Loaders
    - Functions: Extracts text and data from PDFs
    - Use Cases: Research papers, reports, legal documents.
    - Tools: PyMuPDF, PDFMiner, pdfplumber.

2. HTML Loaders
    - Functions: Parses and extracts content from HTML.
    - Use Cases: Blog posts, articles, web content.
    - Tools: BeautifulSoup, lxml, html.parser.

3. CSV Loaders
    - Function: Reads and processes CSV data.
    - Use Cases: Datasets, spreadsheets, tabular data.
    - Tools: Pandas, CSV Module.

4. JSON Loaders
    - Function: Parses and loads JSON data.
    - Use Cases: APIs, configuration files, data exchanges.
    - Tools: Python's JSON Module.

### Text Splitting Techniques

1. By Character Count: Divides text based on a fixed number of characters.
    - Use case: Useful for processing text in chunks of manageable size.
2. By Paragraph: Divides text at paragraph breaks.
    - Use Case: Maintains natural context for more coherent processing.
3. By Sentence: Splits text at sentence boundaries.
    - Use Case: Ideal for tasks needing coherent sentence-level context.
4. Importance of Maintaining Context: Ensures that splits do not disrupt the meaning or flow of the text, preserving coherence and relevance in analysis or generation tasks.

## Embedding Generation and Vector Storage

- Choosing Embedding Model

1. Speed Oriented Models
    - Examples: Word2Vec, FastText.
    - Advantages: Fast Training and inference; suited for real-time and large-scale data.
    - Trade-Offs: Lower accuracy and contextual understanding.

2. Accuracy-Oriented Models
    - Examples: BERT, GPT.
    - Advantages: High accuracy and contextual understanding.
    - Trade-Offs: Slower and costlier inference.

3. Balanced Models
    - Examples: DistilBERT, MiniLM.
    - Advantages: Good balance of speed and accuracy.
    - Trade-Offs: Not as fast as speed-oriented models or as accurate as full-scale transformers.


## Embedding Generation and Vector Storage

### Integrating with Vector Databases

- Indexing Strategies for Large Document Collections

1. Flat Indexing (Brute Force Search):
    - Description: Compares  query embeddings with all stored embeddings.
    - Advantages: Exact nearest neighbors, simple implementation.

2. Approximate Nearest Neighbor (ANN) Indexing:
    - Example: FAISS, ANNOY, HNSW.
    - Description: Uses Algorithms for faster, approximate searches
    - Advantages: Fast Search, scalable to millions of vectors


## Query Processing and Augmentation

### Query Understanding and Reformulation

- Description: Analyzes and refines user queries to better align with relevant data.
- Techniques: Includes parsing queries for intent, correcting typos, and enhancing clarity.

### Techniques for Expanding Queries

1. Synonym Expansion: Adds related term or synonyms to broaden search scope.
2. Contextual Expansion: Uses context to include relevant terms or concepts.
3. Feedback Incorporation: Adjusts queries based on user interactions and previous search results to improve retrieval accuracy.


# LLM Intergration for Generation

1. Prompt Engineering for effective use of retrieved context
    - Method: Craft prompts to incorporate retrieved information for accurate LLM responses.
    - Approach: Frame teh retrieved data to align with the LLMs language strengths.
    - Benefit: Enhances response quality by using relevant, up-to-data information, Ex., referencing specific product details for focused output.

2. Balancing Retrieved Information with LLM Knowledge
    - Method: Combine retrieved data with LLMs internal knowledge for contextually rich and accurate responses.
    - Approach: Balance retrieved data and LLM knowledge to ensure coherence and alignment.
    - Benefit: Avoids over-reliance on either source, producing accurate, contextually appropriate responses and reducing the risk of hallucination or irrelevance.


# Advanced RAG Techniques

## Introduction

- Advanced RAG Techniques are sophisticated strategies that enhance the performance and accuracy of RAG systems. These techniques refine the integration of information retrieval and generative models, enabling more accurate and contextually relevant responses. They involve leveraging complex methods to improve the systems capabilities and adaptability in diverse applications.

## Key Components

1. Cross-Modal Retrieval: Retrieves across different data types using unified embeddings for versatility.
2. Multi-Vector Retrieval: Uses multiple vectors for detailed query representation improving precision.
3. Contextual Reranking: Refines result rankings based on additional context, such as user interactions.
4. Knowledge Enhanced Retrieval: Integrates external knowledge bases for richer context and relevance.
5. Active Learning for Retrieval: Uses user feedback to continually improve retrieval relevance.
6. Hybrid Retrieval: Combines keyword and vector-based searches for balanced retrieval effectiveness.
7. Hierarchical Retrieval: Structures ther retrieval process in layers for improved efficiency and specificity.

## Text Splitting and Chunking Strategies

- Importance of Effective Text Splitting

1. Context Preservation: Proper splitting preserves the integrity of ideas, ensuring retrieved chunks remain coherent and meaningful.
2. Impact on Retrieval Accuracy: Effective text splitting ensures key details are retrieved without losing important context. Poor splitting can result in incomplete or irrelevant results.

- Balancing Chunk Size with Semantic Coherence

1. Chunk Size: Smaller Chunks boost retrieval speed but risk losing context; larger chunks maintain coherence but may include irrelevant info.
2. Semantic Coherence: Balance chunk size to keep meaning intact while ensuring relevance, enhancing retrieval and response quality.

## Different Splitting Techniques


| Techniques | Token-Based Splitting | Semantic Splitting | Paragraph Splitting | Sentence Splitting
| :---: | :---: | :---: | :---: | :---: 
| How | Splits text by token count, suitable for fitting within model limits. | Divides text by meaning using embeddings. | Splits at paragraph boundaries to keep related ideas intact. | Divides text by sentence markers. 
| Use Case | Long texts for LLMs. | Tasks needing deep context understanding. | Document retrieval or topic modeling. | Summarization or Translation.
| Advantage | Precise control over chunk size. | Produces meaningful, contextually relevant chunks. | Retains thematic coherence | Ensures logical flow and coherent analysis.
 
## Query Transformation and Expansion

| Techniques | Description | Advantage
| :---: | :---: | :---: 
| Synonym Expansion | Expands queries with synonyms or related words using tools like WordNet or Embeddings. | Increases chances of retrieving relevant results by capturing phrasing variations.
| Concept Expansion | Adds related terms or entities to broaden the query scope. | Enhances comprehensiveness by retrieving documents on related concepts.
| Query Reformulation | Rephrases queries for better alignment with target content. | Improves precision by matching query language with document phrasing.

## Hypothesis Generation: Hypothetical Document Embeddings - HyDE Technique

- Description: Generates synthetic document embeddings to explore potential retrieval scenarios for hypothetical or unseen documents.
- Purpose: Improves retrieval by considering conceptually relevant documents that aren't present in the database.

### Implementing HyDE with LangChain

- LangChain Overview
    - A framework for building applications that integrate language models for various tasks.

- Implementation Steps
    1. Define Scenarios: Identify where hypothetical embeddings are needed.
    2. Generate Embeddings: Use LangChain to create these embeddings based on context.
    3. Integrate: Incorporate embeddings into the retrieval system for better search results.
    - Advantages: HyDE with Langchain expands retrieval by exploring more possibilities.

## Re-ranking and Filtering Retrieved Documents

1. Reranking

    - Reordering search results based on additional criteria to improve relevance.
    - Contextual Reranking: Adjusts based on user context or query history.
    - Model Based Reranking: Uses ML Models to predict and reorder results by relevance.
    - Feedback Based Reranking: Adapts rankings using user feedback.

2. Filtering

    - Refining results by excluding irrelevant or low-quality documents.
    - Content-based Filtering: Filters by content relevance.
    - Metadata Filtering: Uses Attributes like date or author.
    - User Preference Filtering: Tailors results to individual preferences.

## Using Cross Encoders for Reranking

|  | Cross-Encoders | Bi-Encoders 
| :---: | :---: | :---: 
| How | Jointly encode query and document to assess relevance. | Encode query and document separately, comparing them with a similarity metric.
| Advantage 1 | Higher Accuracy by capturing interactions. | More efficient with precomputed embeddings.
| Advantage 2 | Better contextual understanding of relevance. | Scalable for large retrieval systems.

## Popular Models

|  | Sentence Transformers | Mono T5
| :---: | :---: | :---: 
| Description | Uses Bi-encoders for generating high-quality sentence embeddings, imporving similarity tasks. | Employs cross-encoder for text generation and relevance assessment.
| Use Cases | Fast semantic similarity and embedding tasks. | Detailed text pair ranking and fine-grained reranking.


## LLM Based Document Filtering

- Using LLMs for Document Relevance
    - Description: LLMs assess document relevance by understanding content in relation to the query, capturing nuanced meanings and context.
    - Use Case: Effective for precise relevance needs, such as legal or research document analysis.

- Implementing a relevance scoring system
    - Input Processing: Feed query and document pairs to the LLM.
    - Relevance Assessment: LLM assigns scores based on query-document match.
    - Threshold Filtering: Retain documents meeting the relevance threshold.
    - Continuous Learning: Refine scoring with user feedback and new data.

# Advanced RAG Architectures

## Overview

- Advanced RAG architectures enhance traditional RAG systems with sophisticated components and methods. They aim to boost accuracy, efficiency, and scalability, making them ideal for complex tasks and large-scale applications.

## Multi-Step Retrieval

- Coarse to Fine Retrieval Strategies

- Description: A hierarchical approach where an initial broad search (coarse) is followed by refined searches (fine) to handle large datasets efficiently.

- Steps
1. Coarse Retrieval: Quickly retrieves a broad set of documents using methods like keyword matching or ANN search.
2. Fine Retrieval: Refines this set using advanced techniques like vector search or cross-encoders for improved precision.

- Advantages:
1. Scalability: Efficiently manages large data volumnes by filtering out less relevant documents early.
2. Accuracy: Enhances relevance with detailed analysis in later stages.


### Implementing Multi-Step Retrieval with LangChain

- LangChain Overview
    - A framework for chaining language tasks, suited for multi-step retrieval.

- Implementation Steps
    1. Coarse Retrieval: Set up basic search; retrieve broad document set.
    2. Fine Retrieval: Refine with advanced models; integrate coarse and fine steps.
    3. Final Output: Optimize and deliver refined.

- Advantages
    1. Modularity: Easy to adjust steps.
    2. Efficiency: Balances speed and accuracy.
    3. Customization: Flexible integration of models/tools.


# Hybrid Search

- Overview: Hybrid Search combines dense and sparse retrieval methods to leverage the strengths of both approaches, enhancing the accuracy and relevance of search results.

## Combining Dense and Sparse Retrieval: 

- Sparse Retrieval (Ex. BM25)
    - Description: Uses term-frequency methods for exact keyword matches.
    - Strength: High precision with exact terms.

- Dense Retrieval (Ex. Embeddings)
    - Description: Uses Vector-based methods for semantic similarity.
    - Strength: Finds similar content based on meaning.

- BM 25 + Embedding Hybrid Approach
    - Approach
        1. Initial Retrieval: Use BM25 for broad keyword-based search.
        2. Dense Filtering: Refine with embeddings for semantic relevance.
        3. Final Ranking: Combine BM25 and embedding scores for a ranked list.


# Metrics for Evaluating RAG Performance.

1. Precision
    - Definition: Precision measures the proportion of relevant documents retrieved out of all the retrieved documents.
    - Formula: ![Precision Formula](https://latex.codecogs.com/png.latex?%5Ctext%7BPrecision%7D%20%3D%20%5Cfrac%7B%5Ctext%7BTrue%20Positives%7D%7D%7B%5Ctext%7BTrue%20Positives%7D%20+%20%5Ctext%7BFalse%20Positives%7D%7D)
