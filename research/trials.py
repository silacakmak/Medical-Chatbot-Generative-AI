import os
import sys
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth, AdditionalConfig, Timeout

# Load environment variables
load_dotenv()

# Get Weaviate credentials from environment
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

if not WEAVIATE_URL or not WEAVIATE_API_KEY:
    print("❌ Error: WEAVIATE_URL and WEAVIATE_API_KEY must be set in .env file")
    sys.exit(1)

print("Importing required modules...")

# Import modules with error handling
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    print("✓ Text splitter imported from langchain")
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        print("✓ Text splitter imported from langchain_text_splitters")
    except ImportError:
        print("❌ Failed to import RecursiveCharacterTextSplitter")
        print("Please install langchain or langchain_text_splitters")
        sys.exit(1)

# Updated to use langchain-huggingface
try:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        print("✓ Embeddings imported from langchain_huggingface")
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        print("✓ Embeddings imported from langchain_community (consider upgrading to langchain-huggingface)")
except ImportError:
    print("❌ Failed to import HuggingFaceEmbeddings")
    print("Please install langchain-huggingface or langchain-community")
    sys.exit(1)

try:
    import weaviate

    print("✓ Weaviate imported")
except ImportError:
    print("❌ Failed to import weaviate")
    print("Please install weaviate-client")
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

           # module = _import_(loader_module, fromlist=[loader_class])
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
    """
    Establishes a connection to Weaviate using environment variables.
    Returns a connected Weaviate client.
    Raises ValueError if credentials are missing or ConnectionError if connection fails.
    """
    try:
        # Connect to Weaviate Cloud using the new method
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
            skip_init_checks=True,
            additional_config=AdditionalConfig(
                timeout=Timeout(insert=300)  # Increase timeout for large imports
            )
        )

        if client.is_ready():
            print("✓ Connected to Weaviate Cloud successfully")
            return client
        else:
            raise ConnectionError("Weaviate client is not ready")

    except Exception as e:
        raise ConnectionError(f"Failed to connect to Weaviate: {str(e)}")


# Function to get user query
def get_user_query():
    """Get a search query from the user"""
    print("\n=== Enter Your Query ===")
    return input("Please enter your medical question: ")


# Main execution
if __name__ == "__main__":
    # Print Python environment info for debugging
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Weaviate client version: {weaviate.__version__}")
   # print(f"Weaviate client version: {weaviate._version_}")

    # Get the base directory (where the script is located)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    #base_dir = os.path.dirname(os.path.abspath(_file_))
    print(f"Base directory: {base_dir}")

    # Try to find the Data directory
    data_paths = [
        r"C:\Users\Sıla\Desktop\med1\Data",  # Specific user path
        os.path.join(base_dir, "Data"),  # Same level as script
        os.path.join(base_dir, "..", "Data"),  # One level up
        os.path.join(os.getcwd(), "Data"),  # Current working directory
    ]

    # Find the first valid path
    data_path = None
    for path in data_paths:
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
        print("\n=== Loading PDF Documents ===")
        extracted_data = load_pdf_files(data_path=data_path)

        if not extracted_data:
            print("❌ No documents were loaded. Please check if there are PDF files in the Data directory.")
            sys.exit(1)

        # Split documents into chunks
        print("\n=== Processing Documents ===")
        text_chunks = text_split(extracted_data)
        print(f"✓ Created {len(text_chunks)} text chunks")

        # Initialize embeddings
        print("\n=== Initializing Embeddings ===")
        embeddings = initialize_embeddings()

        # Test embedding
        test_query = "test query"
        query_result = embeddings.embed_query(test_query)
        print(f"✓ Embedding test successful (vector length: {len(query_result)})")

        # Connect to Weaviate
        print("\n=== Connecting to Weaviate ===")
        client = connect_to_weaviate()
        if not client.is_ready():
            raise ConnectionError("Weaviate client is not ready")
        print("✓ Connected to Weaviate successfully")

        # Create schema in Weaviate
        class_name = "medibot"

        # Check if collection already exists
        try:
            # Check if collection exists
            try:
                collection = client.collections.get(class_name)
                print(f"✓ Collection {class_name} already exists")
            except Exception:
                print(f"\n=== Creating Weaviate Schema ===")
                # Create the collection
                collection = client.collections.create(
                    name=class_name,
                    description="Medical document chunks for retrieval",
                    vectorizer_config=None,  # We'll bring our own vectors
                    properties=[
                        {
                            "name": "content",
                            "data_type": "text",
                            "description": "The content of the document chunk"
                        },
                        {
                            "name": "source",
                            "data_type": "text",
                            "description": "The source document of the chunk"
                        },
                        {
                            "name": "page",
                            "data_type": "number",
                            "description": "The page number in the source document"
                        }
                    ]
                )
                print("✓ Schema created successfully")

            # Import data into Weaviate
            print(f"\n=== Importing Data ({len(text_chunks)} chunks) ===")

            # Use the correct syntax for Weaviate 4.x
            # For fixed size batch (recommended for predictable memory usage)
            with collection.batch.fixed_size(batch_size=50) as batch:
                try:
                    # Process chunks in batches
                    for i, chunk in enumerate(text_chunks, 1):
                        print(f"Processing chunk {i}/{len(text_chunks)}...", end="\r")

                        # Get metadata with defaults
                        source = chunk.metadata.get("source", "unknown")
                        page = chunk.metadata.get("page", 0)

                        try:
                            # Get the embedding vector
                            vector = embeddings.embed_query(chunk.page_content)

                            # Add to batch with the correct syntax for Weaviate 4.x
                            batch.add_object(
                                properties={
                                    "content": chunk.page_content,
                                    "source": source,
                                    "page": page
                                },
                                vector=vector
                            )

                        except Exception as e:
                            print(f"\n❌ Error processing chunk {i}: {str(e)}")
                            continue

                    print("\n✓ All chunks processed!")

                except Exception as e:
                    print(f"\n❌ Error during batch processing: {str(e)}")

                # Batch context manager automatically calls flush() when exiting the with block

            # Check for failed objects
            failed_objects = collection.batch.failed_objects
            if failed_objects:
                print(f"\n⚠ Warning: {len(failed_objects)} objects failed to import")
                # Optionally log or handle failed objects
            else:
                print("\n✓ Data import complete!")

            # Test a query - using direct fetch_objects for Weaviate 4.8.1
            print("\n=== Testing Search Query ===")
            query = get_user_query()
            try:
                query_vector = embeddings.embed_query(query)

                # Use a simpler query approach for Weaviate 4.8.1
                result = collection.query.near_vector(
                    near_vector=query_vector,
                    limit=3
                )

                print("\nQuery Results:")
                if result and hasattr(result, 'objects') and result.objects:
                    for i, obj in enumerate(result.objects, 1):
                        print(f"\nResult {i}:")
                        print(
                            f"Source: {obj.properties.get('source', 'unknown')} (Page {obj.properties.get('page', 'unknown')})")
                        content = obj.properties.get('content', '')
                        print(f"Content: {content[:150]}..." if content else "Content: Not available")
                else:
                    print("No results found")

            except Exception as e:
                print(f"❌ Error during query test: {str(e)}")
                # Print detailed error for debugging
                import traceback

                traceback.print_exc()

        except Exception as e:
            print(f"\n❌ An error occurred during schema creation or data import: {str(e)}")
            # Make sure to close the client connection even if an error occurs
            if 'client' in locals():
                client.close()
            raise

    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("\nDetailed error information:")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Make sure to close the client connection
        if 'client' in locals():
            print("\nClosing Weaviate connection...")
            client.close()

    print("\n✓ All operations completed successfully!")