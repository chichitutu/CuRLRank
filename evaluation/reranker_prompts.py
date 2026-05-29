RELEVANCE_DEFINITIONS = {
    # BRIGHT benchmark
    "biology": "Given a query (biology post) and a set of documents (passages), the document is relevant to the query if the critical concepts or theories discussed in the document can provide helpful insights for domain experts to answer the query.",
    "earth_science": "Given a query (earth science post) and a set of documents (passages), the document is relevant to the query if the critical concepts or theories discussed in the document can provide helpful insights for domain experts to answer the query.",
    "economics": "Given a query (economics post) and a set of documents (passages), the document is relevant to the query if the critical concepts or theories discussed in the document can provide helpful insights for domain experts to answer the query.",
    "psychology": "Given a query (psychology post) and a set of documents (passages), the document is relevant to the query if the critical concepts or theories discussed in the document can provide helpful insights for domain experts to answer the query.",
    "robotics": "Given a query (robotics post) and a set of documents (passages), the document is relevant to the query if the critical concepts or theories discussed in the document can provide helpful insights for domain experts to answer the query.",
    "stackoverflow": "Given a query (Stack Overflow post) and a set of documents (passages), the document is relevant to the query if the critical concepts or theories discussed in the document can provide helpful insights for domain experts to answer the query.",
    "sustainable_living": "Given a query (sustainable living post) and a set of documents (passages), the document is relevant to the query if the critical concepts or theories discussed in the document can provide helpful insights for domain experts to answer the query.",
    "leetcode": "Given a query (LeetCode problem) and a set of documents (coding problem solutions), the document is relevant to the query if the underlying algorithms or data structures used in the document can provide helpful insights for solving the problem in the query.",
    "pony": "Given a query (Pony coding instruction) and a set of documents (Pony documentation passages), the document is relevant to the query if the Pony syntax described in the document is necessary for beginners with no prior knowledge of Pony to complete the coding instruction in the query.",
    "aops": "Given a query (math problem) and a set of documents (math problem solutions), the document is relevant to the query if the theorems used in the document can provide helpful insights for solving the problem in the query.",
    "theoremqa_questions": "Given a query (math problem) and a set of documents (math problem solutions), the document is relevant to the query if the theorems used in the document can provide helpful insights for solving the problem in the query.",
    "theoremqa_theorems": "Given a query (math problem) and a set of documents (math-related passages), the document is relevant to the query if the theorem described in the document can provide helpful insights for solving the problem in the query.",
    # BEIR benchmark
    "trec-covid": "Given a COVID-19 related query and a set of documents (passages), the document is relevant to the query if it answers the query.",
    "dbpedia-entity": "Given a query and a set of documents (passages) describing an entity from DBpedia, the document is relevant to the query if the entity described in the document matches the query.",
    "scifact": "Given a scientific claim query and a set of documents (passages), the document is relevant to the query if it provides evidence supporting or refuting the scientific claim.",
    "nfcorpus": "Given a question and a set of documents (passages), the document is relevant to the query if the document can best answer the question.",
    "signal": "Given a news event or topic query and a set of documents (headlines or summaries), the document is relevant to the query if it reports on, summarizes, or directly relates to the same event or topic.",
    "robust04": "Given an information-need query and a set of documents, the document is relevant to the query if it contains information that satisfies the intent or topic described in the query, even if phrased differently.",
    "news": "Given a query about a contemporary news topic or event and a set of documents (news article from The Washington Post), the document is relevant to the query if it discusses, explains, or provides factual coverage of that event or topic.",
    # R2MED benchmark
    "r2med_Biology": "Given a query (biology question) and a set of documents (passages), the document is relevant to the query if it provides scientific explanations, biological mechanisms, or essential evidence that can help answer the query.",
    "r2med_Bioinformatics": "Given a query (bioinformatics question) and a set of documents (passages), the document is relevant to the query if it provides methods, databases, tools, or evidence that can help answer the query.",
    "r2med_Medical-Sciences": "Given a query (medical question) and a set of documents (passages), the document is relevant to the query if it provides scientific explanations, biological or clinical mechanisms, or evidence that can help answer the query.",
    "r2med_MedXpertQA-Exam": "Given a query (clinical diagnostic question) and a set of documents (passages), the document is relevant to the query if it provides clinical guidelines, evidence, reasoning, or explanations that can help answer the query.",
    "r2med_MedQA-Diag": "Given a query (clinical diagnostic question) and a set of documents (passages), the document is relevant if it provides evidence, clinical reasoning, or explanations that support the identification or confirmation of the correct diagnosis for the patient scenario described in the query.",
    "r2med_PMC-Treatment": "Given a query (clinical case and treatment question) and a set of documents (passages), the document is relevant if it provides evidence, guidelines, or explanations that support appropriate treatment strategies, drug choices, or therapeutic rationale for the patient case described in the query.",
    "r2med_PMC-Clinical": "Given a query (clinical case and diagnostic question) and a set of documents (passages), a document is considered relevant if it provides clinical evidence, case descriptions, or diagnostic reasoning that supports identifying the correct diagnosis for the patient described in the query.",
    "r2med_IIYi-Clinical": "Given a query (clinical case) and a set of documents (passages), the document is relevant if it provides evidence, case descriptions, or diagnostic reasoning that supports identifying the correct diagnosis or understanding the patient’s clinical presentation.",
}

QUERY_MAX_LEN = 768
DOC_MAX_LEN = 768
WINDOW_SIZE = 10