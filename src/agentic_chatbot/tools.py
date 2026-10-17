import os
import math
import requests
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

ACTIVE_THREAD_ID: str | None = None
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


def set_active_thread(thread_id: str | None):
    """Set the thread whose PDF knowledge base should be used."""
    global ACTIVE_THREAD_ID
    ACTIVE_THREAD_ID = thread_id


def get_active_thread() -> str | None:
    return ACTIVE_THREAD_ID


def store_pdf(thread_id: str | None, file_path: str) -> str:
    """Add a PDF to the FAISS store for the given thread."""
    if not thread_id:
        return "No active thread selected. Please choose a conversation before uploading a PDF."

    store_dir = os.path.join("faiss_store", thread_id)
    os.makedirs(store_dir, exist_ok=True)

    try:
        docs = PyPDFLoader(file_path).load()
    except Exception as exc:
        return f"Could not read PDF: {exc}"

    if not docs:
        return "The uploaded PDF is empty or unreadable."

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    if os.path.exists(os.path.join(store_dir, "index.faiss")):
        vector_store = FAISS.load_local(
            folder_path=store_dir,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        vector_store.add_documents(chunks)
    else:
        vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(store_dir)
    return f"Saved {len(chunks)} document chunks for thread '{thread_id}'."


def get_retriever(thread_id: str | None = None):
    target_thread = thread_id or get_active_thread()
    if not target_thread:
        raise ValueError("No active thread is selected.")

    db_path = os.path.join("faiss_store", target_thread)
    if not os.path.exists(db_path) or not os.path.exists(os.path.join(db_path, "index.faiss")):
        raise FileNotFoundError(f"No PDF store exists for thread '{target_thread}'.")

    vector_store = FAISS.load_local(
        folder_path=db_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )


@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from the PDF document.

    Use this tool when the user asks factual or conceptual questions
    that may be answered using the stored PDF documents.

    Args:
        query: The question or search query used to retrieve PDF content.
    """
    try:
        retriever = get_retriever()
        documents = retriever.invoke(query)
    except (ValueError, FileNotFoundError):
        return "No PDF knowledge base has been uploaded for this thread yet."

    if not documents:
        return "No relevant information was found in the PDF documents for this thread."

    formatted_documents = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")

        formatted_documents.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)

search_tool = (
    TavilySearch(
        max_results=5,
        topic="general",
        search_depth="advanced",
    )
    if os.getenv("TAVILY_API_KEY")
    else None
)

@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """

    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    api_key = os.getenv("ALPHAVANTAG_API_KEY")
    if not api_key:
        return {"error": "ALPHAVANTAG_API_KEY is missing. Add it to your environment."}

    url = (
        "https://www.alphavantage.co/query?function=GLOBAL_QUOTE"
        f"&symbol={symbol}&apikey={api_key}"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {"error": f"Could not fetch stock price: {exc}"}

@tool
def get_current_weather(location: str) -> str:
    """
    Get the current real-time weather for a given city or location.

    Args:
        location: City or location name, for example:
                  "Dhaka", "London, UK", or "New York, US".

    Returns:
        A formatted current weather report.
    """

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return (
            "Weather API key is missing. "
            "Set the OPENWEATHER_API_KEY environment variable."
        )

    try:
        # Step 1: Convert the location name into latitude and longitude
        geocoding_url = "https://api.openweathermap.org/geo/1.0/direct"

        geocoding_params = {
            "q": location,
            "limit": 1,
            "appid": api_key,
        }

        geo_response = requests.get(
            geocoding_url,
            params=geocoding_params,
            timeout=10,
        )
        geo_response.raise_for_status()

        locations: list[dict[str, Any]] = geo_response.json()

        if not locations:
            return f"Could not find the location: {location}"

        latitude = locations[0]["lat"]
        longitude = locations[0]["lon"]
        resolved_name = locations[0].get("name", location)
        country = locations[0].get("country", "")
        state = locations[0].get("state", "")

        # Step 2: Get current weather using latitude and longitude
        weather_url = "https://api.openweathermap.org/data/2.5/weather"

        weather_params = {
            "lat": latitude,
            "lon": longitude,
            "appid": api_key,
            "units": "metric",
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10,
        )
        weather_response.raise_for_status()

        weather_data = weather_response.json()

        temperature = weather_data["main"]["temp"]
        feels_like = weather_data["main"]["feels_like"]
        humidity = weather_data["main"]["humidity"]
        pressure = weather_data["main"]["pressure"]
        description = weather_data["weather"][0]["description"]
        wind_speed = weather_data.get("wind", {}).get("speed", "N/A")
        visibility_meters = weather_data.get("visibility")

        visibility_km = (
            round(visibility_meters / 1000, 1)
            if visibility_meters is not None
            else "N/A"
        )

        location_parts = [resolved_name]

        if state:
            location_parts.append(state)

        if country:
            location_parts.append(country)

        display_location = ", ".join(location_parts)

        return (
            f"Current weather in {display_location}:\n"
            f"- Condition: {description.title()}\n"
            f"- Temperature: {temperature}°C\n"
            f"- Feels like: {feels_like}°C\n"
            f"- Humidity: {humidity}%\n"
            f"- Pressure: {pressure} hPa\n"
            f"- Wind speed: {wind_speed} m/s\n"
            f"- Visibility: {visibility_km} km"
        )

    except requests.Timeout:
        return "The weather service request timed out. Please try again."

    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response else "unknown"

        if status_code == 401:
            return "The OpenWeather API key is invalid or inactive."

        return f"Weather API returned an HTTP error: {status_code}"

    except requests.RequestException as error:
        return f"Could not connect to the weather service: {error}"

    except (KeyError, TypeError, ValueError) as error:
        return f"Unexpected weather API response: {error}"