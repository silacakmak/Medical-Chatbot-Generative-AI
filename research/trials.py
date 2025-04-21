import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Weaviate environment variables
os.environ["WEAVIATE_URL"] = "https://9vfyboq2rbqbng4hwnhuqg.c0.us-east1.gcp.weaviate.cloud"
os.environ["WEAVIATE_API_KEY"] = "5olyn8aWk6Z8IF45eQRc60dKLFQIk6Umdwq1"

print("Importing required modules...")

# Import modules with error handling
try:
    # Import text splitter
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        print("✓ Text splitter imported from langchain")
    except ImportError:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            print("✓ Text splitter imported from langchain_text_splitters")
        except ImportError:
            print("❌ Failed to import RecursiveCharacterTextSplitter")
            sys.exit(1)

    # Import embeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings

    print("✓ Embeddings imported")

    # Import weaviate
    import weaviate

    print("✓ Weaviate imported")

except ImportError as e:
    print(f"❌ Error importing modules: {str(e)}")
    print("Please install the required packages using:")
    print("pip install -r requirements.txt")
    sys.exit(1)


# Function to get the best available PDF loader
def get_pdf_loader():
    """Determine the best available PDF loader and return the loader class"""
    # Try importing different loaders in order of preference
    loaders = [
        ('PyPDFLoader', 'langchain_community.document_loaders.PyPDFLoader'),
        ('PyMuPDFLoader', 'langchain_community.document_loaders.PyMuPDFLoader'),
        ('PDFMinerLoader', 'langchain_community.document_loaders.PDFMinerLoader'),
        ('UnstructuredPDFLoader', 'langchain_community.document_loaders.UnstructuredPDFLoader')
    ]

    for loader_name, loader_path in loaders:
        try:
            loader_module, loader_class = loader_path.rsplit('.', 1)
            module = __import__(loader_module, fromlist=[loader_class])
            pdf_loader = getattr(module, loader_class)
            print(f"✓ Using {loader_name} for PDF loading")
            return pdf_loader
        except (ImportError, AttributeError) as e:
            print(f"✗ {loader_name} not available: {str(e)}")

    print(
        "❌ No PDF loaders are available. Please install at least one of: pypdf, pymupdf, pdfminer.six, or unstructured")
    sys.exit(1)


# Function to load PDF files from a directory
def load_pdf_files(data_path):
    print(f"Loading PDFs from: {data_path}")

    # Get the best available PDF loader
    PDFLoader = get_pdf_loader()

    from langchain_community.document_loaders import DirectoryLoader

    try:
        # Try batch loading first
        loader = DirectoryLoader(
            data_path,
            glob="*.pdf",
            loader_cls=PDFLoader
        )
        documents = loader.load()
        print(f"✓ Successfully loaded {len(documents)} document chunks from directory")
        return documents
    except Exception as e:
        print(f"Error with batch loading: {str(e)}")
        print("Falling back to loading files individually...")

        # Try loading each file individually
        documents = []
        for file in os.listdir(data_path):
            if file.endswith('.pdf'):
                file_path = os.path.join(data_path, file)
                try:
                    print(f"Attempting to load {file}...")
                    loader = PDFLoader(file_path)
                    file_docs = loader.load()
                    documents.extend(file_docs)
                    print(f"✓ Successfully loaded {file}, got {len(file_docs)} pages/chunks")
                except Exception as file_error:
                    print(f"❌ Error loading {file}: {str(file_error)}")

        if documents:
            print(f"✓ Successfully loaded {len(documents)} total document chunks")
        else:
            print("❌ No documents were successfully loaded")

        return documents


# Function to split documents into text chunks
def text_split(documents, chunk_size=500, chunk_overlap=20):
    if not documents:
        print("Warning: No documents to split")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks


# Function to initialize HuggingFace embeddings
def initialize_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    return embeddings


# Function to connect to Weaviate
def connect_to_weaviate():
    # Get credentials from environment variables
    weaviate_url = os.environ.get("WEAVIATE_URL")
    weaviate_api_key = os.environ.get("WEAVIATE_API_KEY")

    if not weaviate_url or not weaviate_api_key:
        raise ValueError("Weaviate URL or API key not found in environment variables")

    # Try different authentication methods for Weaviate
    try:
        # Method 1: Using auth_client_secret
        client = weaviate.Client(
            url=weaviate_url,
            auth_client_secret=weaviate.AuthApiKey(api_key=weaviate_api_key)
        )
        print("Connected to Weaviate using AuthApiKey")
        return client
    except (AttributeError, TypeError) as e:
        print(f"First connection method failed: {str(e)}")
        try:
            # Method 2: Using older API style
            client = weaviate.Client(
                url=weaviate_url,
                auth_client_secret=weaviate.auth.AuthApiKey(api_key=weaviate_api_key)
            )
            print("Connected to Weaviate using auth.AuthApiKey")
            return client
        except Exception as e:
            print(f"Second connection method failed: {str(e)}")
            try:
                # Method 3: Using direct API key
                client = weaviate.Client(
                    url=weaviate_url,
                    additional_headers={"Authorization": f"Bearer {weaviate_api_key}"}
                )
                print("Connected to Weaviate using additional_headers")
                return client
            except Exception as e:
                print(f"Third connection method failed: {str(e)}")
                raise ConnectionError("Failed to connect to Weaviate with all available methods")


# Main execution
if __name__ == "__main__":
    # Print Python environment info for debugging
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")

    # Set the path to your data directory - try both absolute and relative paths
    print("Current working directory:", os.getcwd())

    # Try different paths for data
    possible_paths = [
        # Original path
        r"D:\Private\Projeler\Python\Medical-Chatbot-Generative-AI\Data",
        # Current dir with relative path
        os.path.join(os.getcwd(), "Data"),
        # One level up
        os.path.join(os.getcwd(), "..", "Data"),
        # Script location based path
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data"),
        # Script location based path (direct)
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data"),
    ]

    # Find the first valid path
    data_path = None
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and os.path.isdir(abs_path):
            data_path = abs_path
            print(f"✓ Found data directory: {data_path}")
            break
        else:
            print(f"✗ Data directory not found at: {abs_path}")

    if not data_path:
        print("❌ Could not find the Data directory.")
        data_path = input("Please enter the full path to your Data directory: ")
        if not os.path.exists(data_path):
            print(f"❌ Path does not exist: {data_path}")
            sys.exit(1)

    try:
        # Load and process PDF documents
        extracted_data = load_pdf_files(data_path=data_path)

        if not extracted_data:
            print("❌ No documents were loaded. Exiting.")
            sys.exit(1)

        # Split documents into chunks
        text_chunks = text_split(extracted_data)
        print(f"Length of Text Chunks: {len(text_chunks)}")

        # Initialize embeddings
        embeddings = initialize_embeddings()

        # Test embedding
        query_result = embeddings.embed_query("hello world")
        print(f"Embedding length: {len(query_result)}")

        # Connect to Weaviate
        client = connect_to_weaviate()
        print(f"Weaviate connection status: {client.is_ready()}")

        # Create schema in Weaviate
        class_name = "MedicalDocument"

        # Check if class already exists
        if not client.schema.exists(class_name):
            print(f"Creating Weaviate schema for class {class_name}...")
            schema = {
                "classes": [
                    {
                        "class": class_name,
                        "description": "Medical document chunks for retrieval",
                        "vectorizer": "none",  # We'll bring our own vectors
                        "properties": [
                            {
                                "name": "content",
                                "dataType": ["text"],
                                "description": "The content of the document chunk"
                            },
                            {
                                "name": "source",
                                "dataType": ["string"],
                                "description": "The source document of the chunk"
                            },
                            {
                                "name": "page",
                                "dataType": ["int"],
                                "description": "The page number in the source document"
                            }
                        ]
                    }
                ]
            }
            client.schema.create(schema)
            print("✓ Schema created successfully")
        else:
            print(f"✓ Schema for class {class_name} already exists")

        # Import data into Weaviate
        print(f"Importing {len(text_chunks)} chunks into Weaviate...")
        with client.batch as batch:
            batch.batch_size = 50  # Process in smaller batches
            for i, chunk in enumerate(text_chunks):
                print(f"Processing chunk {i + 1}/{len(text_chunks)}...", end="\r")

                # Extract metadata
                source = chunk.metadata.get("source", "unknown")
                page = chunk.metadata.get("page", 0)

                # Create data object
                data_object = {
                    "content": chunk.page_content,
                    "source": source,
                    "page": page
                }

                # Get the embedding vector
                vector = embeddings.embed_query(chunk.page_content)

                # Add to batch
                batch.add_data_object(
                    data_object=data_object,
                    class_name=class_name,
                    vector=vector
                )

        print("\n✓ Data import complete!")

        # Test a query
        print("Testing a medical query...")
        query = "What are the symptoms of diabetes?"
        query_vector = embeddings.embed_query(query)

        result = client.query.get(
            class_name=class_name,
            properties=["content", "source", "page"]
        ).with_near_vector({
            "vector": query_vector,
            "certainty": 0.7
        }).with_limit(3).do()

        print("\nQuery Results:")
        if "data" in result and "Get" in result["data"]:
            objects = result["data"]["Get"][class_name]
            for i, obj in enumerate(objects):
                print(f"\nResult {i + 1}:")
                print(f"Source: {obj['source']} (Page {obj['page']})")
                print(f"Content: {obj['content'][:150]}...")
        else:
            print("No results found or unexpected result format")

    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        import traceback

        traceback.print_exc()