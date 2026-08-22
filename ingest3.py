import os
import re
import chromadb

from sentence_transformers import SentenceTransformer
from unstructured.partition.pdf import partition_pdf


# CONFIGURATION

DATA_FOLDER = "./data"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "heart_diseases"

# Chunk settings

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Embedding model

EMBEDDING_MODEL = "BAAI/bge-m3"

BATCH_SIZE = 5000


# TEXT CLEANING

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text.

    Goals:
    - Remove PDF garbage
    - Remove dot leaders
    - Remove page navigation
    - Fix whitespace
    - Preserve useful medical punctuation
    - Preserve readable sentences
    """

    if not text:
        return ""

    text = str(text)

    # 1. Normalize line endings

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")


    text = re.sub(
        r"\.{4,}",
        " ",
        text
    )

    # 3. Remove underscore leaders

    text = re.sub(
        r"_{4,}",
        " ",
        text
    )



    text = re.sub(
        r"-{5,}",
        " ",
        text
    )



    text = re.sub(
        r"([A-Za-z]{2,})-\s*\n\s*([A-Za-z]{2,})",
        r"\1\2",
        text
    )

    # 6. Convert remaining newlines to spaces

    text = re.sub(
        r"\n+",
        " ",
        text
    )

    # 7. Normalize tabs

    text = text.replace(
        "\t",
        " "
    )


    text = re.sub(
        r"\bPage\s+\d+\s+of\s+\d+\b",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # 9. Remove excessive punctuation garbage
  
    text = re.sub(
        r"[.]{3,}",
        " ",
        text
    )

    # 10. Keep useful characters only
  
    text = re.sub(
        r"[^\w\s\.\,\-\(\)\%\:\;\/\?\!\&\+\=]",
        " ",
        text
    )

    # 11. Normalize spaces
   

    text = re.sub(
        r"\s+",
        " ",
        text
    )


    text = re.sub(
        r"\s+([.,;:!?])",
        r"\1",
        text
    )

   
    # 13. Normalize spaces around brackets

    text = re.sub(
        r"\(\s+",
        "(",
        text
    )

    text = re.sub(
        r"\s+\)",
        ")",
        text
    )

    return text.strip()


# SAFE METADATA VALUE

def safe_metadata_value(
    value,
    default="Unknown"
):

    if value is None:
        return default

    value = str(value).strip()

    if not value:
        return default

    return value


# EXTRACT PAGE NUMBER

def get_page_number(element):

    try:

        metadata = element.metadata

        page_number = getattr(
            metadata,
            "page_number",
            None
        )

        if page_number is not None:

            return str(
                page_number
            )

    except Exception:
        pass

    return "Unknown"


# EXTRACT SECTION / TITLE


def get_section_title(element):

    try:

        metadata = element.metadata

        category = getattr(
            element,
            "category",
            None
        )

        if category:

            category = str(
                category
            ).strip()

            if category.lower() in [
                "title",
                "header",
                "heading"
            ]:

                text = str(
                    element
                ).strip()

                if text:

                    return clean_text(
                        text
                    )

        section = getattr(
            metadata,
            "section",
            None
        )

        if section:

            return clean_text(
                str(section)
            )

    except Exception:
        pass

    return "Unknown"


# REMOVE TOC / NAVIGATION ELEMENTS


def is_navigation_text(text):

    if not text:
        return True

    cleaned = clean_text(
        text
    )

    if not cleaned:
        return True

    lower = cleaned.lower()

    # Very obvious table-of-contents indicators
   
    toc_patterns = [

        "table of contents",

        "contents",

        "list of contents",

    ]

    for pattern in toc_patterns:

        if lower == pattern:

            return True

    # Page navigation


    if re.fullmatch(
        r"page\s+\d+",
        lower
    ):

        return True

    if re.fullmatch(
        r"\d+\s+of\s+\d+",
        lower
    ):

        return True

    # Dot-leader remnants
  

    if re.search(
        r"\.{3,}",
        text
    ):

        return True

    # Navigation-like text
   

    navigation_words = [

        "table of contents",

        "appendix",

        "index",

    ]

    navigation_count = sum(
        word in lower
        for word in navigation_words
    )

    if (
        navigation_count >= 1
        and len(cleaned.split()) < 30
    ):

        return True

    return False


# REMOVE EMPTY / BAD ELEMENTS

def is_valid_text(text):

    if not text:
        return False

    text = clean_text(
        text
    )

    if not text:
        return False

    # Too short
    if len(text) < 15:
        return False

    # Mostly punctuation
    alphanumeric_count = len(
        re.findall(
            r"[A-Za-z0-9]",
            text
        )
    )

    if alphanumeric_count < 10:
        return False

    return True



# SENTENCE SPLITTING

def split_into_sentences(text):

    """
    Split text into sentences while preserving
    medical abbreviations reasonably well.
    """

    if not text:
        return []

    # Protect common abbreviations

    protected = {
        "e.g.": "EGPLACEHOLDER",
        "i.e.": "IEPLACEHOLDER",
        "etc.": "ETCPLACEHOLDER",
        "vs.": "VSPLACEHOLDER",
        "Dr.": "DRPLACEHOLDER",
        "Mr.": "MRPLACEHOLDER",
        "Mrs.": "MRSPLACEHOLDER",
    }

    for old, new in protected.items():

        text = text.replace(
            old,
            new
        )

    # Split after sentence punctuation


    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    cleaned_sentences = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # Restore abbreviations

        for old, new in protected.items():

            sentence = sentence.replace(
                new,
                old
            )

        cleaned_sentences.append(
            sentence
        )

    return cleaned_sentences


# SMART CHUNKING


def create_smart_chunks(
    text,
    chunk_size=500,
    chunk_overlap=100
):

    """
    Create chunks using sentence boundaries.

    This is better than blindly cutting every 500
    characters because it tries to keep complete
    sentences together.
    """

    text = clean_text(
        text
    )

    if not text:
        return []

    sentences = split_into_sentences(
        text
    )

    if not sentences:
        return []

    chunks = []

    current_sentences = []
    current_length = 0

    for sentence in sentences:

        sentence_length = len(
            sentence
        )

        # Normal case

        if (
            current_length
            + sentence_length
            + 1
            <= chunk_size
        ):

            current_sentences.append(
                sentence
            )

            current_length += (
                sentence_length + 1
            )

            continue

        # Save current chunk

        if current_sentences:

            chunk = " ".join(
                current_sentences
            ).strip()

            if chunk:

                chunks.append(
                    chunk
                )

        
        # Create overlap using previous sentences
       

        overlap_sentences = []

        overlap_length = 0

        for previous_sentence in reversed(
            current_sentences
        ):

            if (
                overlap_length
                + len(previous_sentence)
                + 1
                <= chunk_overlap
            ):

                overlap_sentences.insert(
                    0,
                    previous_sentence
                )

                overlap_length += (
                    len(previous_sentence)
                    + 1
                )

            else:

                break

        
        # Start next chunk
       

        current_sentences = (
            overlap_sentences
            + [sentence]
        )

        current_length = sum(
            len(x) + 1
            for x in current_sentences
        )

    
    # Last chunk
    

    if current_sentences:

        chunk = " ".join(
            current_sentences
        ).strip()

        if chunk:

            chunks.append(
                chunk
            )

    return chunks



# DEDUPLICATE TEXT


def normalize_for_duplicate_check(text):

    text = clean_text(
        text
    )

    return text.lower()



# START


print("=" * 70)

print(
    "CARDIOVASCULAR DISEASE RAG — DOCUMENT INGESTION"
)

print("=" * 70)



# 1. LOAD EMBEDDING MODEL


print(
    "\n[1/7] Loading embedding model..."
)

print(
    f"      Model: {EMBEDDING_MODEL}"
)

try:

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

except Exception as e:

    print(
        "\n❌ Failed to load embedding model."
    )

    print(
        f"   Error: {e}"
    )

    raise

print(
    "      ✅ Model loaded successfully"
)



# 2. CONNECT TO CHROMADB

print(
    "\n[2/7] Connecting to ChromaDB..."
)

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)



# DELETE OLD COLLECTION


try:

    client.delete_collection(
        name=COLLECTION_NAME
    )

    print(
        f"      🗑️ Deleted old collection: "
        f"{COLLECTION_NAME}"
    )

except Exception:

    print(
        "      ℹ️ No existing collection found"
    )


# CREATE COLLECTION

collection = client.create_collection(

    name=COLLECTION_NAME,

    metadata={
        "hnsw:space": "cosine"
    }

)

print(
    f"      ✅ Collection created: "
    f"{COLLECTION_NAME}"
)

print(
    "      📏 Distance metric: cosine"
)

# 3. CHUNKING

print(
    "\n[3/7] Initializing smart chunking..."
)

print(
    f"      Target chunk size: "
    f"{CHUNK_SIZE} chars"
)

print(
    f"      Target overlap: "
    f"{CHUNK_OVERLAP} chars"
)

print(
    "      Strategy: sentence-aware"
)

print(
    "      ✅ Chunking ready"
)


# 4. CHECK DATA FOLDER


print(
    "\n[4/7] Checking data folder..."
)

if not os.path.exists(
    DATA_FOLDER
):

    raise FileNotFoundError(
        f"""
❌ Directory '{DATA_FOLDER}' does not exist.

Please create it and add PDF files.
"""
    )


pdf_files = [

    f

    for f in os.listdir(
        DATA_FOLDER
    )

    if f.lower().endswith(
        ".pdf"
    )

]


if not pdf_files:

    raise FileNotFoundError(
        f"""
❌ No PDF files found in '{DATA_FOLDER}'.

Please add cardiovascular guideline PDFs.
"""
    )


print(
    f"      ✅ Found "
    f"{len(pdf_files)} PDF file(s)"
)


for pdf in pdf_files:

    print(
        f"         - {pdf}"
    )




# 5. PROCESS PDF FILES



print(
    "\n[5/7] Processing PDFs..."
)

print(
    "=" * 70
)


all_chunks = []
all_metadata = []

global_seen_texts = set()



# PROCESS EACH PDF



for filename in pdf_files:

    pdf_path = os.path.join(
        DATA_FOLDER,
        filename
    )

    print(
        f"\n📄 Processing: {filename}"
    )

    try:

        # PARSE PDF
       

        elements = partition_pdf(

            filename=pdf_path,

            strategy="fast"

        )

        print(
            f"   📑 Extracted elements: "
            f"{len(elements)}"
        )


        # TRACK SECTION
        

        current_section = "Unknown"


        
        # GROUP TEXT BY PAGE
       

        page_texts = {}

        page_sections = {}


      
        # PROCESS ELEMENTS
        

        for element in elements:

            raw_text = str(
                element
            ).strip()

            if not raw_text:
                continue

            
            # Clean first
          

            cleaned_text = clean_text(
                raw_text
            )

            if not is_valid_text(
                cleaned_text
            ):

                continue

         
            # Ignore navigation / TOC garbage
           

            if is_navigation_text(
                raw_text
            ):

                continue

            # Metadata
            

            page_number = get_page_number(
                element
            )

            section_title = get_section_title(
                element
            )

            
            # Update current section

            if section_title != "Unknown":

                current_section = (
                    section_title
                )

            # Page key

            if page_number == "Unknown":

                page_key = "unknown"

            else:

                page_key = page_number

            # Initialize page

            if page_key not in page_texts:

                page_texts[page_key] = []

                page_sections[page_key] = (
                    current_section
                )

            # Store cleaned text

            page_texts[page_key].append(
                cleaned_text
            )


        # CREATE SMART CHUNKS PAGE BY PAGE

        file_chunk_count = 0

        file_skipped_count = 0


        for page_number, texts in page_texts.items():

            # Combine page elements

            page_text = " ".join(
                texts
            )

            page_text = clean_text(
                page_text
            )

            if not page_text:
                continue

            # Section

            section_title = page_sections.get(
                page_number,
                "Unknown"
            )

            # SMART CHUNKING

            chunks = create_smart_chunks(

                page_text,

                chunk_size=CHUNK_SIZE,

                chunk_overlap=CHUNK_OVERLAP

            )


            # STORE CHUNKS

            for local_chunk_number, chunk in enumerate(
                chunks,
                start=1
            ):

                chunk = clean_text(
                    chunk
                )
                # Quality checks

                if not chunk:
                    file_skipped_count += 1
                    continue

                if len(chunk) < 30:
                    file_skipped_count += 1
                    continue

                if is_navigation_text(
                    chunk
                ):

                    file_skipped_count += 1
                    continue

                # Duplicate check

                normalized_chunk = (
                    normalize_for_duplicate_check(
                        chunk
                    )
                )

                if (
                    normalized_chunk
                    in global_seen_texts
                ):

                    file_skipped_count += 1
                    continue

                global_seen_texts.add(
                    normalized_chunk
                )

                # Chunk ID


                chunk_id = (
                    f"{filename}"
                    f"-p{page_number}"
                    f"-c{local_chunk_number}"
                )

                # Metadata

                metadata = {

                    "document_name":
                        safe_metadata_value(
                            filename
                        ),

                    "source":
                        safe_metadata_value(
                            filename
                        ),

                    "page_number":
                        safe_metadata_value(
                            page_number
                        ),

                    "chunk_id":
                        chunk_id,

                    "section_title":
                        safe_metadata_value(
                            section_title
                        ),

                    "element_type":
                        "Text",

                    "chunk_size":
                        str(
                            len(chunk)
                        )

                }

                # Store

                all_chunks.append(
                    chunk
                )

                all_metadata.append(
                    metadata
                )

                file_chunk_count += 1


        print(
            f"   📦 Generated chunks: "
            f"{file_chunk_count}"
        )

        print(
            f"   🧹 Skipped bad/duplicate chunks: "
            f"{file_skipped_count}"
        )


    except Exception as e:

        print(
            f"   ❌ Error processing "
            f"{filename}:"
        )

        print(
            f"      {e}"
        )


# SUMMARY

print(
    "\n" + "=" * 70
)

print(
    f"📦 Total clean chunks generated: "
    f"{len(all_chunks)}"
)

print(
    "=" * 70
)


if not all_chunks:

    print(
        "\n❌ No usable chunks were generated."
    )

    print(
        "   Please check your PDF files."
    )

    raise SystemExit


# SHOW SAMPLE METADATA

print(
    "\n🔎 Metadata preview:"
)


for i in range(
    min(3, len(all_metadata))
):

    print(
        f"\n   Chunk {i + 1}:"
    )

    print(
        f"      ID: "
        f"{all_metadata[i]['chunk_id']}"
    )

    print(
        f"      Document: "
        f"{all_metadata[i]['document_name']}"
    )

    print(
        f"      Page: "
        f"{all_metadata[i]['page_number']}"
    )

    print(
        f"      Section: "
        f"{all_metadata[i]['section_title']}"
    )

    print(
        f"      Size: "
        f"{all_metadata[i]['chunk_size']}"
    )

    print(
        f"      Text: "
        f"{all_chunks[i][:250]}..."
    )


# DEBUG — CHECK TUV.PDF PAGE 26

print(
    "\n" + "=" * 70
)

print(
    "🔍 DEBUG: Checking TUV.pdf - Page 26"
)

print(
    "=" * 70
)


found_debug_chunks = False


for i, metadata in enumerate(
    all_metadata
):

    if (

        metadata["document_name"]
        == "TUV.pdf"

        and

        metadata["page_number"]
        == "26"

    ):

        found_debug_chunks = True

        print(
            "\n" + "-" * 70
        )

        print(
            f"Chunk ID: "
            f"{metadata['chunk_id']}"
        )

        print(
            f"Chunk Size: "
            f"{metadata['chunk_size']}"
        )

        print(
            "-" * 70
        )

        print(
            all_chunks[i]
        )


if not found_debug_chunks:

    print(
        "\n⚠️ No TUV.pdf Page 26 chunks found."
    )


print(
    "\n" + "=" * 70
)

print(
    "🔍 END DEBUG"
)

print(
    "=" * 70
)


# 6. GENERATE EMBEDDINGS


print(
    "\n[6/7] Creating embeddings..."
)

print(
    f"      Model: {EMBEDDING_MODEL}"
)

print(
    "      Normalizing: Yes"
)


embeddings = model.encode(

    all_chunks,

    show_progress_bar=True,

    normalize_embeddings=True,

    batch_size=32

).tolist()


print(
    f"      ✅ Generated "
    f"{len(embeddings)} embeddings"
)

# 7. STORE IN CHROMADB

print(
    "\n[7/7] Storing in ChromaDB..."
)


ids = [

    meta["chunk_id"]

    for meta in all_metadata

]


# BATCH INSERT


for i in range(

    0,

    len(all_chunks),

    BATCH_SIZE

):

    batch_end = min(

        i + BATCH_SIZE,

        len(all_chunks)

    )


    collection.upsert(

        ids=ids[
            i:batch_end
        ],

        documents=all_chunks[
            i:batch_end
        ],

        embeddings=embeddings[
            i:batch_end
        ],

        metadatas=all_metadata[
            i:batch_end
        ]

    )


    print(
        f"      Stored chunks "
        f"{i} → {batch_end}"
    )


print(
    "      ✅ All chunks stored successfully"
)

# FINAL SUMMARY

print(
    "\n" + "=" * 70
)

print(
    "✅ INGESTION COMPLETE"
)

print(
    "=" * 70
)


print(
    f"📂 PDF files processed:  "
    f"{len(pdf_files)}"
)

print(
    f"📦 Total clean chunks:   "
    f"{len(all_chunks)}"
)

print(
    f"🗄  Collection:           "
    f"{COLLECTION_NAME}"
)

print(
    f"🤖 Embedding model:      "
    f"{EMBEDDING_MODEL}"
)

print(
    "📐 Embedding dims:       1024"
)

print(
    "📏 Distance metric:      cosine"
)

print(
    f"✂️  Target chunk size:    "
    f"{CHUNK_SIZE} chars"
)

print(
    f"🔗 Chunk overlap:        "
    f"{CHUNK_OVERLAP} chars"
)

print(
    "🧹 Text cleaning:        Enabled"
)

print(
    "🧹 TOC filtering:        Enabled"
)

print(
    "🧹 Duplicate filtering:  Enabled"
)

print(
    "✂️  Sentence chunking:    Enabled"
)

print(
    "📚 Metadata:              "
    "source + page + section + chunk_id"
)


print(
    "\n" + "=" * 70
)

print(
    "🎯 Ready for retrieval!"
)

print(
    "   Run retrieval.py to test queries."
)

print(
    "=" * 70
)