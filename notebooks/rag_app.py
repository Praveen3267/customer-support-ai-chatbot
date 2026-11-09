import re
import streamlit as st
import chromadb
import ollama
from sentence_transformers import SentenceTransformer


# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Customer Support AI Chatbot",
    page_icon="🤖",
    layout="wide"
)


# -------------------------------
# Clean Retrieved Documents
# -------------------------------

def clean_placeholders(text):
    """Remove template placeholders and known generic template labels."""

    # Remove complete placeholders such as {{Order Number}}
    text = re.sub(r"\{\{.*?\}\}", "", text)

    # Remove unfinished placeholders
    text = re.sub(r"\{\{.*$", "", text)

    # Remove known generic template labels
    blocked_terms = [
        "Online Order Interaction",
        "Company Account"
    ]

    for term in blocked_terms:
        text = re.sub(
            re.escape(term),
            "",
            text,
            flags=re.IGNORECASE
        )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# -------------------------------
# Clean Generated Answers
# -------------------------------

def clean_generated_answer(answer):
    """Remove incomplete contact-information sentences and placeholders."""

    # Reject answers containing unresolved template placeholders
    if "{{" in answer or "}}" in answer:
        return (
            "Sorry, I could not generate a reliable answer. "
            "Please check the official customer support website "
            "or your order confirmation for verified information."
        )

    blocked_phrases = [
        "Customer Support Phone Number",
        "Website URL",
        "Customer Support Hours",
        "dedicated customer support team is available",
        "feel free to call us",
        "call us at",
        "live chat feature on our website",
        "website at",
        "available during"
    ]

    # Split at sentence endings and discard sentences with blocked phrases
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    clean_sentences = []

    for sentence in sentences:
        if not any(
            phrase.lower() in sentence.lower()
            for phrase in blocked_phrases
        ):
            clean_sentences.append(sentence)

    answer = " ".join(clean_sentences)

    # Clean leftover spaces and punctuation
    answer = re.sub(r"\s+", " ", answer)
    answer = re.sub(r"\s+([,.!?])", r"\1", answer)

    return answer.strip()


# -------------------------------
# Load Models and Database
# -------------------------------

@st.cache_resource
def load_resources():

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(
        path=r"C:\Users\PRAVEEN\chroma_db"
    )

    collection = client.get_collection(
        name="customer_support_knowledge"
    )

    return embedding_model, collection


embedding_model, collection = load_resources()


# -------------------------------
# Handle Greetings and Thanks
# -------------------------------

def handle_simple_messages(query):

    normalized_query = query.lower().strip()

    normalized_query = re.sub(
        r"[^\w\s]",
        "",
        normalized_query
    )

    greetings = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening"
    ]

    thanks = [
        "thank you",
        "thanks",
        "thank you so much",
        "thanks a lot"
    ]

    if normalized_query in greetings:
        return (
            "Hello! 👋 Welcome to Customer Support. "
            "How can I help you today?"
        )

    if normalized_query in thanks:
        return (
            "You're welcome! 😊 "
            "I'm happy to help. Let me know if you "
            "have any other questions."
        )

    return None


# -------------------------------
# Improve Search Query
# -------------------------------

def improve_search_query(query):

    query_lower = query.lower()

    # Order tracking
    if any(word in query_lower for word in [
        "package",
        "shipment",
        "delivery status",
        "where is my order",
        "track my order",
        "tracking"
    ]):
        return (
            "How can I track my order and check its "
            "shipment status and estimated delivery date?"
        )

    # Order cancellation
    elif any(word in query_lower for word in [
        "cancel",
        "cancellation"
    ]):
        return (
            "How can I cancel an existing order? "
            "Provide order cancellation instructions."
        )

    # Refund
    elif any(word in query_lower for word in [
        "refund",
        "money back",
        "reimbursement"
    ]):
        return (
            "How can I request a refund and get my money back?"
        )

    # Damaged product
    elif any(word in query_lower for word in [
        "damaged",
        "broken",
        "defective",
        "faulty"
    ]):
        return (
            "I received a damaged or defective product. "
            "How can I report the issue and request a return?"
        )

    return query


# -------------------------------
# Retrieve Relevant Documents
# -------------------------------

def retrieve_documents(query, top_k=5):

    improved_query = improve_search_query(query)

    query_embedding = embedding_model.encode(
        [improved_query]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved_documents = []

    documents = results["documents"][0] if results.get("documents") else []
    distances = results["distances"][0] if results.get("distances") else []

    for document, distance in zip(documents, distances):

        cleaned_document = clean_placeholders(document)

        if cleaned_document:
            retrieved_documents.append({
                "document": cleaned_document,
                "distance": distance
            })

    return retrieved_documents


# -------------------------------
# Generate RAG Response
# -------------------------------

def get_rag_response(query):

    try:

        # Handle greetings and thanks
        simple_response = handle_simple_messages(query)

        if simple_response:
            return simple_response

        # Handle order tracking questions
        tracking_patterns = [
            r"\bwhere is my package\b",
            r"\btrack my order\b",
            r"\btracking status\b",
            r"\bwhere is my order\b",
            r"\bpackage status\b"
        ]

        if any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in tracking_patterns
        ):
            return (
                "To check your package status, follow these steps:\n\n"
                "1. Sign in to your account.\n"
                "2. Open Order History or Track My Order.\n"
                "3. Enter your order number or tracking number.\n"
                "4. Check the latest available shipment status "
                "and delivery updates.\n\n"
                "I cannot access live shipment information directly."
            )

        # Handle damaged or defective product questions
        damaged_patterns = [
            r"\bdamaged\b",
            r"\bbroken\b",
            r"\bdefective\b",
            r"\bfaulty\b"
        ]

        if any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in damaged_patterns
        ):
            return (
                "I'm sorry your product arrived damaged. "
                "Please contact the official customer support team "
                "and provide your order details along with photos "
                "of the damage. They can review your case and confirm "
                "whether a replacement or refund is available. "
                "I don't have a verified replacement policy in "
                "my knowledge base."
            )

        # Handle unavailable support phone-number requests
        contact_patterns = [
            r"\bphone number\b",
            r"\bcontact number\b",
            r"\bhelpline\b",
            r"\btoll[- ]free number\b"
        ]

        if any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in contact_patterns
        ):
            return (
                "Sorry, I don't have a verified customer "
                "support phone number in my knowledge base. "
                "Please check your official order confirmation "
                "or account for verified contact details."
            )

        # Handle refund processing-time questions BEFORE general refund questions
        refund_time_patterns = [
            r"\brefund processing time\b",
            r"\bhow long.*refund\b",
            r"\bwhen.*refund\b",
            r"\brefund.*how long\b",
            r"\brefund.*timeframe\b",
            r"\btimeframe.*refund\b",
            r"\bhow much time.*refund\b",
            r"\brefund.*take\b"
        ]

        if any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in refund_time_patterns
        ):
            return (
                "The refund processing timeframe is not specified "
                "in my knowledge base. Please contact the relevant "
                "customer support team to confirm the expected "
                "processing time for your refund."
            )

        # Handle refund-status questions BEFORE general refund questions
        refund_status_patterns = [
            r"\brefund status\b",
            r"\bstatus of my refund\b",
            r"\bcheck my refund\b",
            r"\btrack my refund\b",
            r"\bwhere is my refund\b",
            r"\bupdate on my refund\b"
        ]

        if any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in refund_status_patterns
        ):
            return (
                "I cannot check your current refund status directly. "
                "According to my knowledge base, customer support "
                "can assist with refund status information after "
                "you have submitted your refund request. Please "
                "contact the relevant customer support team for "
                "an update."
            )

        # Handle general refund questions
        refund_patterns = [
            r"\brefund\b",
            r"\brefunded\b",
            r"\brefunds\b",
            r"\bmoney back\b",
            r"\breimbursement\b"
        ]

        if any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in refund_patterns
        ):
            return (
                "You may be eligible to request a refund in "
                "situations such as:\n\n"
                "1. The product or service is defective or does "
                "not match its description.\n"
                "2. You cancel an order within the permitted "
                "cancellation period.\n"
                "3. You notice unauthorized charges.\n"
                "4. An event is cancelled without being rescheduled.\n"
                "5. You are charged more than once for the same purchase.\n"
                "6. You do not receive the purchased goods or services.\n"
                "7. Your order is not delivered within the expected "
                "timeframe.\n\n"
                "Refund eligibility depends on the vendor's specific "
                "terms and conditions. Please review the terms for "
                "your purchase or contact the relevant customer "
                "support team to confirm whether your situation qualifies."
            )

        # Retrieve relevant documents for other questions
        retrieved_documents = retrieve_documents(
            query,
            top_k=5
        )

        if not retrieved_documents:
            return (
                "Sorry, I could not find relevant information "
                "in the customer support knowledge base."
            )

        # Prepare context
        context = "\n\n".join(
            item["document"]
            for item in retrieved_documents
        )

        # RAG prompt
        prompt = f"""
You are a helpful customer support AI assistant.

Answer the customer's question using ONLY the
information provided in the context below.

Important rules:

1. Give clear and simple answers.
2. Do not invent company policies.
3. Do not invent refund amounts or delivery dates.
4. Do not invent phone numbers, email addresses,
   website URLs, or support hours.
5. Do not display template placeholders.
6. If the context does not contain the answer,
   clearly say that the information is unavailable.
7. Give step-by-step instructions when appropriate.
8. Do not claim that an action has been completed.
9. Do not include information that is not supported
   by the retrieved context.
10. Do not suggest contacting support through a specific
    channel unless that channel is provided in the context.

Context:

{context}

Customer Question:

{query}

Answer:
"""

        # Generate answer using Ollama
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.1
            }
        )

        answer = response["message"]["content"]

        # Clean generated answer
        answer = clean_generated_answer(answer)

        if not answer:
            return (
                "Sorry, I could not generate a reliable answer "
                "from the available knowledge base."
            )

        return answer

    except Exception as e:

        st.error(
            f"Detailed error: {type(e).__name__}: {str(e)}"
        )

        return (
            "Sorry, something went wrong while "
            "processing your request."
        )


# -------------------------------
# Streamlit User Interface
# -------------------------------

# Welcome Screen

st.title("🤖 Customer Support AI Chatbot")

st.markdown(
    """
    ### Welcome! 👋

    I'm your AI Customer Support Assistant.

    I can help you with:
    - 📦 Order tracking
    - ❌ Order cancellation
    - 💰 Refund information
    - 📱 Damaged product issues
    - 🚚 Delivery information

    **How can I help you today?**
    """
)

st.divider()


# Suggested Questions

st.subheader("💡 Try asking")

col1, col2 = st.columns(2)

if col1.button("📦 Where is my package?", use_container_width=True):
    st.session_state.selected_question = "Where is my package?"

if col2.button("❌ How can I cancel my order?", use_container_width=True):
    st.session_state.selected_question = "How can I cancel my order?"

col3, col4 = st.columns(2)

if col3.button("💰 How can I get a refund?", use_container_width=True):
    st.session_state.selected_question = "How can I get my money back?"

if col4.button("📱 My product arrived damaged", use_container_width=True):
    st.session_state.selected_question = "My product arrived damaged."



# -------------------------------
# Sidebar
# -------------------------------

with st.sidebar:

    st.header("Project Information")

    st.success(
        f"Knowledge Base Records: {collection.count():,}"
    )

    st.write("**Embedding Model:**")
    st.write("all-MiniLM-L6-v2")

    st.write("**LLM Model:**")
    st.write("llama3.2:3b")

    st.write("**Vector Database:**")
    st.write("ChromaDB")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# -------------------------------
# Chat History
# -------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    if message["role"] == "user":

        with st.chat_message("user", avatar="👤"):
            st.markdown(message["content"])

    else:

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message["content"])


# -------------------------------
# User Input
# -------------------------------

user_query = st.chat_input(
    "Type your customer support question..."
)

if "selected_question" in st.session_state:
    user_query = st.session_state.pop("selected_question")


if user_query:

    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("🤖 AI is thinking... Please wait"):
            answer = get_rag_response(user_query)

        st.markdown(answer)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
