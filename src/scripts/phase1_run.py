# scripts/phase1_run.py
import sys
from pathlib import Path

# ensure project root is on sys.path so imports like `from fetch...` work
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # ../ from scripts/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- then normal imports below ---

import json
import logging
from pathlib import Path
from fetch.arxiv_fetcher import fetch_many_queries
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------
# Topic -> arXiv query mapping (100 topics from user)
# ---------------------------
# You can Add More topic as per your need
TOPIC_QUERIES: Dict[str, str] = {
    # === 🧱 Classical NLP Foundations ===
    "n_gram_statistical_language_modeling": '"n-gram language model" OR "statistical language modeling" AND cat:cs.CL',
    "maximum_entropy_models": '"maximum entropy" AND (NLP OR "language modeling") AND cat:cs.CL',
    "hmm_crf_sequence_labeling": '"hidden markov model" OR "conditional random field" OR "sequence labeling" AND cat:cs.CL',
    "neural_language_models_word_embeddings": '"word2vec" OR "glove" OR "neural language model" AND cat:cs.CL',
    "contextual_embeddings": '"contextual embeddings" OR "ELMo" OR "ULMFiT" AND cat:cs.CL',
    "seq2seq_rnn_mt": '"sequence to sequence" OR "neural machine translation" OR "encoder-decoder RNN" AND cat:cs.CL',

    # === ⚡ Transformer & Modern LMs ===
    "transformer_architecture": '"Attention Is All You Need" OR "Transformer architecture" AND cat:cs.CL',
    "masked_language_models_BERT": '"BERT" OR "masked language modeling" AND cat:cs.CL',
    "autoregressive_GPT_style": '"GPT" OR "autoregressive language model" AND cat:cs.CL',
    "encoder_decoder_models": '"T5" OR "text-to-text transformer" OR "BART" AND cat:cs.CL',
    "scaling_laws_efficiency": '"scaling laws" OR "compute optimal" OR "Chinchilla" OR "PaLM" AND cat:cs.CL',
    "mixture_of_experts_models": '"mixture of experts" OR "sparse transformer" OR "Switch Transformer" AND cat:cs.CL',
    "small_efficient_lms": '"small language model" OR "tiny LLM" OR "DistilBERT" OR "Mistral" AND cat:cs.CL',

    # === 🧩 Fine-tuning, Adapters, and PEFT ===
    "fine_tuning_adapters": '"fine-tuning" OR "adapter tuning" OR "prefix tuning" OR "LoRA" AND cat:cs.CL',
    "parameter_efficient_finetuning": '"PEFT" OR "low-rank adaptation" OR "adapter fusion" AND cat:cs.CL',
    "distillation_compression": '"knowledge distillation" OR "model compression" OR "quantization" AND cat:cs.CL',
    "instruction_tuning_RLHF": '"instruction tuning" OR "RLHF" OR "preference learning" AND cat:cs.CL',
    "dpo_alignment": '"Direct Preference Optimization" OR DPO OR "alignment fine-tuning" AND cat:cs.CL',
    "self_alignment_synthetic_data": '"self alignment" OR "synthetic preference data" OR "auto alignment" AND cat:cs.CL',

    # === 🧠 Prompting, Reasoning, and In-Context Learning ===
    "prompting_in_context_learning": '"prompt engineering" OR "in-context learning" OR "few-shot" AND cat:cs.CL',
    "prompt_engineering_optimization": '"prompt optimization" OR "prompt tuning" OR "prompt search" AND cat:cs.CL',
    "chain_of_thought_reasoning": '"chain of thought" OR "reasoning in language models" AND cat:cs.CL',
    "tree_of_thought_reasoning": '"tree of thought" OR "structured reasoning" OR "multi-step reasoning" AND cat:cs.CL',
    "self_consistency_reasoning": '"self consistency" OR "reasoning consistency" AND cat:cs.CL',
    "program_aided_reasoning": '"Program-aided language model" OR "PAL" OR "symbolic reasoning" AND cat:cs.CL',
    "tool_use_react": '"ReAct" OR "reason+act" OR "LLM tool use" AND cat:cs.CL',
    "toolformer_self_supervised_tool_use": '"Toolformer" OR "self-supervised tool learning" AND cat:cs.CL',

    # === 🔍 Retrieval, Memory, and RAG Variants ===
    "retrieval_augmented_generation": '"retrieval augmented generation" OR RAG AND cat:cs.CL',
    "dense_passage_retrieval": '"dense passage retrieval" OR DPR AND cat:cs.CL',
    "hybrid_retrieval_reranking": '"hybrid retrieval" OR "dense retrieval" OR "reranker" OR "ColBERT" AND cat:cs.CL',
    "hyde_hypothetical_document_embeddings": '"HyDE" OR "hypothetical document embeddings" AND cat:cs.CL',
    "realm_retrieval_pretraining": '"REALM" OR "retrieval augmented pretraining" AND cat:cs.CL',
    "fusion_in_decoder_FiD": '"Fusion-in-Decoder" OR "FiD" AND cat:cs.CL',
    "graphrag_longrag": '"GraphRAG" OR "LongRAG" OR "long context retrieval" AND cat:cs.CL',
    "retrieval_memory_models": '"memory networks" OR "neural turing machine" OR "episodic memory" AND cat:cs.CL',
    "knowledge_graph_rag": '"knowledge graph retrieval" OR "graph-based RAG" AND cat:cs.CL',
    "retrieval_evaluation_benchmarking": '"retrieval benchmark" OR "RAG evaluation" AND cat:cs.CL',

    # === 🧬 Continual, Editing, and Knowledge Models ===
    "continual_lifelong_learning": '"continual learning" OR "lifelong learning" AND (NLP OR language) AND cat:cs.CL',
    "model_editing_knowledge_update": '"model editing" OR "knowledge editing" OR "ROME" OR "MEMIT" AND cat:cs.CL',
    "factuality_hallucination": '"factual consistency" OR "hallucination" OR "faithfulness" AND cat:cs.CL',
    "robustness_adversarial_nlp": '"adversarial attacks" OR "robustness" OR "text perturbation" AND cat:cs.CL',
    "knowledge_injection_models": '"knowledge injection" OR "external memory" OR "fact integration" AND cat:cs.CL',

    # === ⚖️ Fairness, Interpretability, and Safety ===
    "fairness_bias": '"bias" OR "fairness" OR "debiasing" AND (NLP OR embeddings) AND cat:cs.CL',
    "interpretability_probing": '"interpretability" OR "probing" OR "representation analysis" AND cat:cs.CL',
    "explainable_nlp": '"explainable NLP" OR "XNLP" OR "explainable AI" AND cat:cs.CL',
    "language_safety_content_filtering": '"safety" OR "content moderation" OR "toxicity detection" AND cat:cs.CL',
    "responsible_ai_ethics": '"AI ethics" OR "responsible AI" OR "ethical NLP" AND cat:cs.CL',
    "hallucination_detection": '"hallucination detection" OR "fact verification" OR "faithfulness metrics" AND cat:cs.CL',

    # === 🌍 Multilingual, Domain, and Cross-Lingual NLP ===
    "multilingual_cross_lingual": '"multilingual" OR "cross-lingual" OR "translation models" AND cat:cs.CL',
    "machine_translation_neural": '"machine translation" OR "neural translation" OR "Transformer MT" AND cat:cs.CL',
    "domain_specific_nlp": '"biomedical NLP" OR "legal NLP" OR "financial NLP" AND cat:cs.CL',
    "low_resource_nlp": '"low-resource" OR "zero-shot cross-lingual" OR "few-shot translation" AND cat:cs.CL',
    "code_programming_language_models": '"code generation" OR "CodeBERT" OR "Codex" OR "CodeT5" AND cat:cs.CL',

    # === 🧩 Multimodal & Vision-Language ===
    "multimodal_nlp": '"multimodal transformer" OR "vision-language" OR "CLIP" AND cat:cs.CL',
    "vision_language_agents": '"vision language model" OR "multimodal agent" OR "VLM" AND cat:cs.CL',
    "video_language_models": '"video language model" OR "video captioning" OR "multimodal reasoning" AND cat:cs.CL',
    "audio_speech_nlp": '"speech recognition" OR "ASR" OR "whisper" OR "speech-language model" AND cat:cs.CL',

    # === 💬 Applied NLP Tasks ===
    "dialogue_conversational_agents": '"dialogue systems" OR "conversational agent" OR "chatbot" AND cat:cs.CL',
    "summarization": '"text summarization" OR "abstractive summarization" OR "extractive summarization" AND cat:cs.CL',
    "question_answering": '"question answering" OR "open-domain QA" OR "machine reading comprehension" AND cat:cs.CL',
    "coreference_discourse": '"coreference resolution" OR "discourse coherence" AND cat:cs.CL',
    "relation_information_extraction": '"relation extraction" OR "information extraction" OR "event extraction" AND cat:cs.CL',
    "semantic_parsing_text_to_sql": '"semantic parsing" OR "text-to-SQL" OR "structured prediction" AND cat:cs.CL',
    "sentiment_opinion_mining": '"sentiment analysis" OR "opinion mining" OR "aspect-based sentiment" AND cat:cs.CL',
    "ner_entity_linking": '"named entity recognition" OR "entity linking" AND cat:cs.CL',
    "pos_parsing_syntax": '"part-of-speech tagging" OR "dependency parsing" OR "constituency parsing" AND cat:cs.CL',
    "semantic_role_labeling": '"semantic role labeling" OR "PropBank" OR "FrameNet" AND cat:cs.CL',

    # === 🧰 NLP Tools, Frameworks, and Pipelines ===
    "nlp_tools_pipelines": '"spaCy" OR "Stanza" OR "AllenNLP" OR "tokenization" AND cat:cs.CL',
    "huggingface_ecosystem": '"Hugging Face" OR "Transformers library" AND cat:cs.CL',
    "sentence_transformers_embeddings": '"Sentence Transformers" OR "semantic search embeddings" AND cat:cs.CL',
    "langchain_llamaindex_frameworks": '"LangChain" OR "LlamaIndex" OR "retrieval framework" AND cat:cs.CL',
    "vector_databases_faiss_chroma": '"FAISS" OR "ChromaDB" OR "vector database" AND cat:cs.CL',

    # === 🧮 Evaluation, Metrics, and Benchmarks ===
    "evaluation_metrics_nlp": '"evaluation metrics" OR "BLEU" OR "ROUGE" OR "BERTScore" AND cat:cs.CL',
    "benchmarks_datasets_evaluation": '"NLP benchmark" OR "dataset" OR "evaluation suite" AND cat:cs.CL',
    "truthfulqa_factscore": '"TruthfulQA" OR "FactScore" OR "FActEval" AND cat:cs.CL',
    "holistic_evaluation_helm": '"HELM" OR "LLM evaluation" OR "holistic benchmark" AND cat:cs.CL',
    "mt_bench_alpacaeval": '"MT-Bench" OR "AlpacaEval" OR "Chatbot evaluation" AND cat:cs.CL',

    # === 🧩 Emerging Areas & Data-centric AI ===
    "data_quality_and_curation": '"data curation" OR "data quality" OR "dataset bias" AND cat:cs.CL',
    "synthetic_data_generation": '"synthetic data" OR "data augmentation" OR "self-generated data" AND cat:cs.CL',
    "active_learning_nlp": '"active learning" OR "data selection" OR "uncertainty sampling" AND cat:cs.CL',
    "self_training_and_bootstrapping": '"self-training" OR "pseudo labeling" OR "semi-supervised NLP" AND cat:cs.CL',

    # === 🔬 Cognitive & Causal NLP ===
    "causal_counterfactual_nlp": '"causal inference" OR "counterfactual" OR "causal NLP" AND cat:cs.CL',
    "human_cognitive_models": '"human cognition" OR "psycholinguistic modeling" OR "cognitive NLP" AND cat:cs.CL',
    "language_acquisition_and_learning": '"language acquisition" OR "emergent communication" AND cat:cs.CL',

    # === 🧠 Agents, Planning, and Multi-step Reasoning ===
    "autonomous_llm_agents": '"autonomous agent" OR "language agent" OR "LLM agent" OR "AutoGPT" AND cat:cs.CL',
    "planning_and_tool_use": '"planning" OR "tool use" OR "API calling" OR "retrieval tool use" AND cat:cs.CL',
    "multi_agent_systems": '"multi-agent" OR "collaborative agents" OR "agentic reasoning" AND cat:cs.CL',

    # === 🧩 Knowledge Graphs & Structured Reasoning ===
    "knowledge_graph_embeddings": '"knowledge graph embeddings" OR "TransE" OR "Graph Neural Network" AND cat:cs.CL',
    "symbolic_neuro_symbolic_nlp": '"neurosymbolic" OR "symbolic reasoning" OR "logic-based NLP" AND cat:cs.CL',
}


OUT_INDEX = Path("data/raw/topic_index.json")
PER_TOPIC = 3   # adjust per-topic fetch count; set to small number for initial runs

def main():
    logger.info("Starting Phase 1 run: fetching arXiv by topic")
    aggregated = fetch_many_queries(TOPIC_QUERIES, per_topic=PER_TOPIC)

    # Save an index mapping topic -> [paper_meta...]
    OUT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_INDEX, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)

    logger.info("Phase 1 complete. Index written to %s", OUT_INDEX)

if __name__ == "__main__":
    main()
