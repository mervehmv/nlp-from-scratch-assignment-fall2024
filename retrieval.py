import requests
from bs4 import BeautifulSoup
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

# List of HTML pages to process
html_urls = [
    "https://en.wikipedia.org/wiki/Pittsburgh",
    "https://en.wikipedia.org/wiki/History_of_Pittsburgh",
    "https://www.pittsburghpa.gov/Home",
    "https://www.britannica.com/place/Pittsburgh",
    "https://www.visitpittsburgh.com/",
    "https://www.pittsburghpa.gov/City-Government/Finances-Budget/Taxes/Tax-Forms",
    "https://www.cmu.edu/about/",
    "https://www.pghcitypaper.com/pittsburgh/EventSearch?v=d",
    "https://www.cmu.edu/engage/alumni/events/campus/index.html",
    "https://pittsburghopera.org/",
    "https://carnegiemuseums.org/",
    "https://www.heinzhistorycenter.org/",
    "https://www.thefrickpittsburgh.org/",
    "https://en.wikipedia.org/wiki/List_of_museums_in_Pittsburgh",
    "https://www.visitpittsburgh.com/events-festivals/food-festivals/", "https://www.picklesburgh.com/",
    "https://www.pghtacofest.com/",
    "https://pittsburghrestaurantweek.com/",
    "https://bananasplitfest.com/",
    "https://www.visitpittsburgh.com/things-to-do/pittsburgh-sports-teams/", "https://www.mlb.com/pirates",
    "https://www.steelers.com/",
    "https://www.nhl.com/penguins/"
]

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="html_docs")

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Function to extract text from an HTML page
def extract_text_from_html(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unnecessary elements
        for script in soup(["script", "style"]):
            script.extract()

        text = soup.get_text(separator="\n", strip=True)
        return text
    else:
        print(f"Failed to load {url}")
        return None

# Process and store multiple web pages
for url in html_urls:
    text = extract_text_from_html(url)
    if text:
        # Convert text into embeddings
        embedding = embedding_model.embed_query(text)
        collection.add(ids=[url], embeddings=[embedding], metadatas=[{"source": url}])
        print(f"Stored {url} in ChromaDB.")





import pandas as pd

answers = {}
df = pd.read_csv("/content/nlp-from-scratch-assignment-fall2024/data/test_questions.csv", header=None)
for row in df[0]:
    print(row)
    query = row
    query_embedding = embedding_model.embed_query(query)

    # Search for the most relevant chunk
    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    retrieved_texts = [doc["source"] for doc in results["metadatas"][0]]
    context = "\n\n".join(retrieved_texts)
    print("Retrieved Context:", context)


    from langchain.chat_models import ChatOpenAI
    llm = init_chat_model("mistralai/Mixtral-8x7B-Instruct-v0.1", model_provider="together")
    # Generate answer using retrieved context
    prompt = f"Use the following information to answer the query:\n{context}\n\nQuery: {query}"
    response = llm.invoke(prompt)

    print("AI Answer:", response.content)
    answers[row] = response.content
