import os
import re

from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from langdetect import detect
from deep_translator import GoogleTranslator
from groq import Groq


# 1. LOAD ENVIRONMENT VARIABLES

load_dotenv()


# 2. CONFIGURATION

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "heart_diseases"

EMBEDDING_MODEL = "BAAI/bge-m3"

# Retrieve 5 candidates from ChromaDB
TOP_K = 5

# IMPORTANT:
# Only the highest similarity chunk will be used
# for the final answer.
FINAL_TOP_K = 1

# Minimum similarity required
RELEVANCE_THRESHOLD = 70.0

# Minimum text length
MIN_CHUNK_LENGTH = 80

GROQ_MODEL = "qwen/qwen3.6-27b"
TEMPERATURE = 0.2


# 3. LOAD EMBEDDING MODEL

print("Loading embedding model...")
print(f"Model: {EMBEDDING_MODEL}")

_groq_key = os.getenv("GROQ_API_KEY")

if _groq_key:
    print(
        "Groq API key: "
        f"Loaded (starts with {_groq_key[:8]}...)"
    )
else:
    print("Groq API key: NOT FOUND")


try:

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(
        "Embedding model loaded successfully!"
    )

except Exception as e:

    print(
        "\n❌ Failed to load embedding model:"
    )

    print(e)

    raise SystemExit(1)

# 4. CONNECT TO CHROMADB

print("\nConnecting to ChromaDB...")


try:

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print(
        f"✅ Collection loaded: "
        f"{COLLECTION_NAME}"
    )

    print(
        f"📊 Number of documents: "
        f"{collection.count()}"
    )

except Exception as e:

    print(
        "\n❌ Failed to connect to ChromaDB:"
    )

    print(e)

    raise SystemExit(1)


# 5. DETECT DISTANCE METRIC

collection_metadata = (
    collection.metadata or {}
)

distance_metric = collection_metadata.get(
    "hnsw:space",
    "cosine"
)

print(
    f"📏 Distance metric: "
    f"{distance_metric}"
)


if distance_metric != "cosine":

    print(
        "⚠️ WARNING: Collection is not using "
        "'cosine' metric!"
    )


# 6. DISTANCE → SIMILARITY

def distance_to_similarity(
    distance,
    metric
):

    if metric == "cosine":

        similarity = 1 - (
            distance / 2
        )

    elif metric == "l2":

        similarity = 1 / (
            1 + distance
        )

    elif metric == "ip":

        similarity = distance

    else:

        similarity = 1 / (
            1 + distance
        )

    similarity = max(
        0.0,
        min(1.0, similarity)
    )

    return similarity * 100


# 7. LANGUAGE DETECTION

def detect_language(
    text: str
) -> str:

    """
    Detect Arabic or English.
    """

    try:

        lang = detect(text)

        if lang == "ar":

            return "ar"

        return "en"

    except Exception:

        return "en"


# 8. TRANSLATE ARABIC QUERY

def prepare_search_query(
    query,
    user_lang
):

    """
    Translate Arabic queries to English
    before vector search.
    """

    if user_lang != "ar":

        return query

    try:

        translated_query = (
            GoogleTranslator(
                source="auto",
                target="en"
            ).translate(query)
        )

        print(
            "\n[Info] Translated query "
            "for vector search:"
        )

        print(
            f"       {translated_query}"
        )

        return translated_query

    except Exception as e:

        print(
            "\n[Warning] Translation failed."
        )

        print(
            "          Searching using "
            "original query."
        )

        print(
            f"          Error: {e}"
        )

        return query


# 9. RETRIEVAL

def retrieve_chunks(
    search_query
):

    """
    Retrieve TOP_K chunks from ChromaDB.
    """

    try:

        query_embedding = model.encode(
            search_query,
            normalize_embeddings=True
        )

        results = collection.query(

            query_embeddings=[
                query_embedding.tolist()
            ],

            n_results=TOP_K,

            include=[
                "documents",
                "distances",
                "metadatas"
            ]
        )

    except Exception as e:

        print(
            "\n❌ Retrieval failed:"
        )

        print(e)

        return []

    retrieved_chunks = []

    documents = results.get(
        "documents",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    for i, document in enumerate(
        documents
    ):

        distance = distances[i]

        metadata = (
            metadatas[i]
            if i < len(metadatas)
            else {}
        )

        similarity = (
            distance_to_similarity(
                distance,
                distance_metric
            )
        )

        chunk = {

            "text": document,

            "metadata": metadata,

            "distance": distance,

            "similarity": similarity
        }

        retrieved_chunks.append(
            chunk
        )

    return retrieved_chunks


# 10. METADATA HELPERS

def get_document_name(
    metadata
):

    return (

        metadata.get(
            "document_name"
        )

        or metadata.get(
            "source"
        )

        or metadata.get(
            "document"
        )

        or metadata.get(
            "filename"
        )

        or "Unknown Document"
    )


def get_page(
    metadata
):

    return (

        metadata.get(
            "page_number"
        )

        or metadata.get(
            "page"
        )

        or "Unknown Page"
    )


def get_section(
    metadata
):

    return (

        metadata.get(
            "section_title"
        )

        or metadata.get(
            "section"
        )

        or metadata.get(
            "section_name"
        )

        or "Unknown Section"
    )


def get_chunk_id(
    metadata
):

    return (

        metadata.get(
            "chunk_id"
        )

        or "Unknown"
    )


# 11. CLEAN TEXT

def clean_chunk_text(
    text
):

    if not text:

        return ""

    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# 12. DETECT LOW-INFORMATION CHUNKS

def is_low_information_chunk(
    chunk
):

    """
    Detect chunks that contain little useful
    medical information.
    """

    text = clean_chunk_text(
        chunk.get(
            "text",
            ""
        )
    )

    # Too short
    if len(text) < MIN_CHUNK_LENGTH:

        return True

    lower_text = text.lower()

    # Table of contents indicators

    toc_patterns = [

        "table of contents",

        "contents",

        "potential for prevention",

        "page of",

        "................................"
    ]

    toc_score = sum(

        1

        for pattern in toc_patterns

        if pattern in lower_text
    )

    if toc_score >= 2:

        return True

    # Very short content

    words = text.split()

    if len(words) < 20:

        return True

    # Navigation-like content

    navigation_terms = [

        "chapter",

        "contents",

        "appendix",

        "index"
    ]

    navigation_count = sum(

        1

        for term in navigation_terms

        if term in lower_text
    )

    if (
        navigation_count >= 2
        and len(words) < 60
    ):

        return True

    return False


# 13. REMOVE DUPLICATES

def remove_duplicate_chunks(
    chunks
):

    """
    Remove exact duplicates.
    """

    unique_chunks = []

    seen_ids = set()

    seen_texts = set()

    for chunk in chunks:

        metadata = chunk.get(
            "metadata",
            {}
        )

        chunk_id = get_chunk_id(
            metadata
        )

        text = clean_chunk_text(
            chunk.get(
                "text",
                ""
            )
        )

        normalized_text = (
            text.lower()
        )

        if chunk_id in seen_ids:

            continue

        if normalized_text in seen_texts:

            continue

        seen_ids.add(
            chunk_id
        )

        seen_texts.add(
            normalized_text
        )

        unique_chunks.append(
            chunk
        )

    return unique_chunks


# 14. SELECT BEST CHUNK

def select_best_chunk(
    retrieved_chunks
):

    """
    Select ONLY the highest similarity chunk.

    IMPORTANT:
    The LLM will receive only this chunk.
    """

    if not retrieved_chunks:

        return []

    # Step 1: Remove low-information chunks

    quality_chunks = []

    for chunk in retrieved_chunks:

        if is_low_information_chunk(
            chunk
        ):

            continue

        quality_chunks.append(
            chunk
        )

    # Step 2: Remove duplicates

    quality_chunks = (
        remove_duplicate_chunks(
            quality_chunks
        )
    )

    # Step 3: Sort by similarity

    quality_chunks = sorted(

        quality_chunks,

        key=lambda x: x["similarity"],

        reverse=True
    )

    # Step 4: Select ONLY #1

    if not quality_chunks:

        return []

    best_chunk = quality_chunks[0]

    # Step 5: Apply similarity threshold

    if (
        best_chunk["similarity"]
        < RELEVANCE_THRESHOLD
    ):

        return []

    return [
        best_chunk
    ]


# 15. DISPLAY RETRIEVED CHUNKS

def display_retrieved_chunks(
    retrieved_chunks
):

    print(
        "\n" + "=" * 70
    )

    print(
        "RETRIEVED CHUNKS"
    )

    print(
        "=" * 70
    )

    for i, chunk in enumerate(
        retrieved_chunks
    ):

        metadata = chunk.get(
            "metadata",
            {}
        )

        print(
            f"\n--- Chunk {i + 1} ---"
        )

        similarity = chunk.get(
            "similarity",
            0
        )

        distance = chunk.get(
            "distance",
            0
        )

        status = (

            "✅ RELEVANT"

            if similarity >= RELEVANCE_THRESHOLD

            else "❌ NOT RELEVANT"
        )

        print(
            f"Similarity: "
            f"{similarity:.2f}%"
        )

        print(
            f"Distance:   "
            f"{distance:.4f}"
        )

        print(
            f"Status:     "
            f"{status}"
        )

        document_name = (
            get_document_name(
                metadata
            )
        )

        page = get_page(
            metadata
        )

        section = get_section(
            metadata
        )

        chunk_id = get_chunk_id(
            metadata
        )

        print(
            f"Source:     "
            f"{document_name}"
        )

        print(
            f"Page:       "
            f"{page}"
        )

        print(
            f"Section:    "
            f"{section}"
        )

        print(
            f"Chunk ID:   "
            f"{chunk_id}"
        )

        text = chunk.get(
            "text",
            ""
        )

        print(
            f"Text:\n{text}"
        )

    print(
        "\n" + "=" * 70
    )


# 16. BUILD GROUNDED CONTEXT

def build_context(
    relevant_chunks
):

    context_parts = []

    for i, chunk in enumerate(
        relevant_chunks,
        start=1
    ):

        metadata = chunk.get(
            "metadata",
            {}
        )

        document_name = (
            get_document_name(
                metadata
            )
        )

        section = get_section(
            metadata
        )

        page = get_page(
            metadata
        )

        chunk_id = get_chunk_id(
            metadata
        )

        similarity = chunk.get(
            "similarity",
            0
        )

        text = clean_chunk_text(
            chunk.get(
                "text",
                ""
            )
        )

        context_part = f"""
SOURCE {i}

Document: {document_name}
Page: {page}
Section: {section}
Chunk ID: {chunk_id}
Similarity: {similarity:.2f}%

Retrieved Text:
{text}
"""

        context_parts.append(
            context_part.strip()
        )

    return (
        "\n\n"
        "-------------------------"
        "\n\n"
    ).join(
        context_parts
    )


# 17. BUILD CITATIONS

def build_citations(
    relevant_chunks
):

    citations = []

    for i, chunk in enumerate(
        relevant_chunks,
        start=1
    ):

        metadata = chunk.get(
            "metadata",
            {}
        )

        document_name = (
            get_document_name(
                metadata
            )
        )

        page = get_page(
            metadata
        )

        section = get_section(
            metadata
        )

        chunk_id = get_chunk_id(
            metadata
        )

        citations.append({

            "number": i,

            "document": document_name,

            "page": page,

            "section": section,

            "chunk_id": chunk_id
        })

    return citations


# 18. FORMAT CITATIONS

def format_citations(
    citations
):

    if not citations:

        return ""

    lines = []

    for citation in citations:

        number = citation[
            "number"
        ]

        document = citation[
            "document"
        ]

        page = citation[
            "page"
        ]

        section = citation[
            "section"
        ]

        if (
            page != "Unknown Page"
            and
            section != "Unknown Section"
        ):

            line = (
                f"[{number}] "
                f"{document} — "
                f"Page {page} — "
                f"{section}"
            )

        elif page != "Unknown Page":

            line = (
                f"[{number}] "
                f"{document} — "
                f"Page {page}"
            )

        elif section != "Unknown Section":

            line = (
                f"[{number}] "
                f"{document} — "
                f"{section}"
            )

        else:

            line = (
                f"[{number}] "
                f"{document}"
            )

        lines.append(
            line
        )

    return "\n".join(
        lines
    )


# 19. SYSTEM PROMPT

def build_system_prompt(
    user_lang
):

    language = (

        "Arabic"

        if user_lang == "ar"

        else "English"
    )

    return f"""
You are a citation-bound clinical evidence assistant.

Your job is to answer the user's question ONLY
using the SINGLE retrieved evidence chunk provided
in the context.

=========================================================
STRICT GROUNDING RULES
=========================================================

1. Use ONLY information explicitly supported
   by the retrieved evidence.

2. Do NOT use your general medical knowledge.

3. Do NOT add facts that are not present
   in the retrieved evidence.

4. Do NOT guess missing information.

5. Do NOT invent:
   - diagnoses
   - treatments
   - dosages
   - thresholds
   - statistics
   - recommendations
   - causes
   - risk factors
   - clinical interpretations

6. ONLY use the provided evidence chunk.

7. Do NOT use information from your general knowledge
   to complete missing parts of the answer.

8. If the evidence does not fully answer the question,
   clearly say that the available evidence is limited.

9. Do not treat headings or navigation text as evidence.

10. Every factual claim in the Recommendation must
    be directly supported by the provided evidence.

11. Do not fabricate citations.

12. Do not reveal internal reasoning or chain of thought.

13. NEVER output:
    <think>
    </think>
    "Let's analyze"
    "Let's think"
    "Here's my reasoning"

=========================================================
LANGUAGE
=========================================================

Respond strictly in {language}.

=========================================================
OUTPUT FORMAT
=========================================================

Recommendation:

Give a direct answer based ONLY on the single
retrieved evidence chunk.

Excerpt:

Provide the exact relevant evidence from
the retrieved chunk.

Citation:

Use the citation corresponding to the retrieved chunk.

=========================================================
IMPORTANT
=========================================================

ONLY ONE retrieved chunk is available.

Do NOT introduce information from any other source.

The retrieved evidence is the only source of truth.
"""


# 20. GENERATE ANSWER

def generate_answer(
    query,
    relevant_chunks,
    user_lang
):

    if not relevant_chunks:

        if user_lang == "ar":

            return (
                "عذرًا، لم أجد معلومات كافية في "
                "المصادر المتاحة للإجابة عن هذا السؤال بثقة.\n\n"
                "يُرجى إعادة صياغة السؤال أو استشارة طبيب."
            )

        else:

            return (
                "Sorry, I couldn't find enough information "
                "in the available sources to answer this confidently.\n\n"
                "Please rephrase your question or consult a clinician."
            )

    # ONLY ONE CHUNK

    context = build_context(
        relevant_chunks
    )

    citations = build_citations(
        relevant_chunks
    )

    formatted_citations = (
        format_citations(
            citations
        )
    )

    system_prompt = (
        build_system_prompt(
            user_lang
        )
    )

    user_prompt = f"""
Retrieved Evidence:

{context}

Available Citation:

{formatted_citations}

User Question:

{query}

IMPORTANT:

You have ONLY ONE retrieved evidence chunk.

Use ONLY this chunk.

Do not use information from any other source.

Answer strictly according to the system instructions.
"""

    try:

        api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not api_key:

            raise ValueError(
                "GROQ_API_KEY environment variable "
                "is not set."
            )

        groq_client = Groq(
            api_key=api_key
        )

        response = (
            groq_client
            .chat
            .completions
            .create(

                model=GROQ_MODEL,

                messages=[

                    {
                        "role": "system",
                        "content": system_prompt
                    },

                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],

                temperature=TEMPERATURE,

                max_tokens=4000
            )
        )

        answer = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        # Remove thinking blocks

        answer = re.sub(

            r"<think>.*?</think>",

            "",

            answer,

            flags=(
                re.DOTALL
                |
                re.IGNORECASE
            )
        ).strip()

        # Remove anything before Recommendation

        if "Recommendation:" in answer:

            answer = answer[
                answer.find(
                    "Recommendation:"
                ):
            ]

        return answer

    except Exception as e:

        return (
            "[ERROR] Failed to generate response:\n"
            f"{e}"
        )


# 21. MAIN

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "🫀 CARDIOVASCULAR DISEASE GROUNDED RAG"
    )

    print(
        "=" * 70
    )

    query = input(

        "\nAsk your question about heart diseases "
        "/ اسأل سؤالك عن أمراض القلب:\n> "

    ).strip()

    if not query:

        print(
            "⚠️ Please enter a question."
        )

        return

    # LANGUAGE DETECTION

    user_lang = detect_language(
        query
    )

    print(
        "\n[Info] Detected Language: "
        f"{'Arabic 🇪🇬' if user_lang == 'ar' else 'English 🇬🇧'}"
    )

    # PREPARE SEARCH QUERY

    search_query = (
        prepare_search_query(
            query,
            user_lang
        )
    )

    # RETRIEVAL

    print(
        "\n[Info] Searching ChromaDB..."
    )

    retrieved_chunks = (
        retrieve_chunks(
            search_query
        )
    )

    if not retrieved_chunks:

        print(
            "\n❌ No chunks were retrieved."
        )

        return

    # DISPLAY ALL 5 RETRIEVED CHUNKS

    display_retrieved_chunks(
        retrieved_chunks
    )

    # SORT BY SIMILARITY

    retrieved_chunks = sorted(

        retrieved_chunks,

        key=lambda x: x["similarity"],

        reverse=True
    )

    # SHOW TOP 5 RANKING

    print(
        "\n[Info] Top retrieved chunks:"
    )

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        print(
            f"  {i}. "
            f"{chunk['similarity']:.2f}% | "
            f"{get_document_name(chunk.get('metadata', {}))} | "
            f"Page {get_page(chunk.get('metadata', {}))}"
        )

    # SELECT ONLY THE BEST CHUNK

    relevant_chunks = (
        select_best_chunk(
            retrieved_chunks
        )
    )

    # SHOW SELECTED CHUNK

    if relevant_chunks:

        best_chunk = (
            relevant_chunks[0]
        )

        metadata = (
            best_chunk.get(
                "metadata",
                {}
            )
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "🏆 SELECTED BEST CHUNK"
        )

        print(
            "=" * 70
        )

        print(
            f"Similarity: "
            f"{best_chunk['similarity']:.2f}%"
        )

        print(
            f"Source:     "
            f"{get_document_name(metadata)}"
        )

        print(
            f"Page:       "
            f"{get_page(metadata)}"
        )

        print(
            f"Section:    "
            f"{get_section(metadata)}"
        )

        print(
            f"Chunk ID:   "
            f"{get_chunk_id(metadata)}"
        )

        print(
            f"Text:\n"
            f"{best_chunk['text']}"
        )

        print(
            "=" * 70
        )

        print(
            "\n[Info] Selected top 1 chunk:"
        )

        print(
            f"  1. "
            f"{best_chunk['similarity']:.2f}%"
        )

        print(
            f"\n[Info] Final relevant chunks "
            f"(≥{RELEVANCE_THRESHOLD}%): "
            f"1 / {len(retrieved_chunks)}"
        )

    else:

        print(
            f"\n[Info] Final relevant chunks "
            f"(≥{RELEVANCE_THRESHOLD}%): "
            f"0 / {len(retrieved_chunks)}"
        )

    # REFUSAL IF BEST CHUNK IS NOT RELEVANT

    if not relevant_chunks:

        if user_lang == "ar":

            print(
                "\n📭 Response:\n"
                "عذرًا، لم أجد معلومات كافية في "
                "المصادر المتاحة للإجابة عن هذا السؤال بثقة.\n"
                "يُرجى إعادة صياغة السؤال أو استشارة طبيب."
            )

        else:

            print(
                "\n📭 Response:\n"
                "Sorry, I couldn't find enough information "
                "in the available sources to answer this confidently.\n"
                "Please rephrase your question or consult a clinician."
            )

        return

    # GENERATE GROUNDED ANSWER
    print(
        "\n[Info] Generating grounded answer from LLM..."
    )

    answer = generate_answer(

        query,

        relevant_chunks,

        user_lang
    )

    # DISPLAY FINAL ANSWER

    print(
        "\n" + "=" * 70
    )

    print(
        "📄 FINAL ANSWER"
    )

    print(
        "=" * 70
    )

    print(
        answer
    )

    print(
        "\n" + "=" * 70
    )


# 22. RUN

if __name__ == "__main__":

    main()