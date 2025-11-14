# The code, in LangChain, for processing Images, PDFs, and Audios
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document

from PIL import Image
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

import pymupdf

import speech_recognition as sr 
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence

import tempfile


# Function to process Images
def process_images(image):
    pytesseract.pytesseract.tesseract_cmd = ( r"C:\Program Files\Tesseract-OCR\tesseract.exe" )
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    img_color = Image.open(image)
    img_gray = img_color.convert('L')
    inputs = processor(images=img_color, return_tensors="pt") #type: ignore (pylance error; code works fine)

    text = pytesseract.image_to_string(img_color)
    outputs = model.generate(**inputs) #type: ignore (pylance error; code works fine)
    caption = processor.decode(outputs[0], skip_special_tokens=True) #type: ignore (pylance error; code works fine)

    return {
        "text" : text.replace("\x0c", "").strip(),
        "caption": caption
    }


# Function to process PDFs
def process_pdf(pdf):
    pdf_response = []

    doc = pymupdf.open('documents/sample.pdf')
    for page in doc:
        text = page.get_text()
        pdf_response.append(text)

    return pdf_response


# Function to process Audios
r = sr.Recognizer()

def transcribe_audio(path):
    with sr.AudioFile(path) as source:
        audio_listened = r.record(source)
        text = r.recognize_google(audio_listened) #type: ignore (pylance error; code works fine)
    return text

def process_audio(path):
    """Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks"""
    sound = AudioSegment.from_file(path)  
    chunks = split_on_silence(sound,
        min_silence_len = 500,
        silence_thresh = sound.dBFS-14,
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    audio_response = ""
    for i, audio_chunk in enumerate(chunks, start=1):
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        try:
            text = transcribe_audio(chunk_filename)
        except sr.UnknownValueError as e:
            print("Error:", str(e))
        else:
            text = f"{text.capitalize()}. "
            print(chunk_filename, ":", text)
            audio_response += text
    return audio_response


def handle_uploaded_file(uploaded_file):
    suffix = uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

def process_uploaded_file(uploaded_file):
    path = handle_uploaded_file(uploaded_file)
    mime = uploaded_file.type

    if mime.startswith("image/"):
        return process_images(path), uploaded_file.name
    elif mime.startswith("audio/"):
        return process_audio(path), uploaded_file.name
    elif mime == "application/pdf":
        return process_pdf(path), uploaded_file.name
    else:
        raise ValueError(f"Unsupported file type: {mime}")
    



# Declaring Vector Database
index = faiss.IndexFlatL2(len(HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2").embed_query("hello world")))

vector_store = FAISS(
    embedding_function=HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2"),
    index=index,
    docstore= InMemoryDocstore(),
    index_to_docstore_id={}
)

def add_to_vector_store(data_dict, source_name):
    combined_text = ""
    combined_caption = None

    if isinstance(data_dict, dict):
        combined_text = data_dict.get("text", "")
        combined_caption = data_dict.get("caption")

    elif isinstance(data_dict, list):
        all_texts = []
        all_captions = []
        for item in data_dict:
            if isinstance(item, dict):
                all_texts.append(item.get("text", ""))
                if item.get("caption"):
                    all_captions.append(item["caption"])
            elif isinstance(item, str):
                all_texts.append(item)
            else:
                raise TypeError(f"Unexpected item type in list: {type(item)}")
        combined_text = "\n".join(all_texts)
        if all_captions:
            combined_caption = "\n".join(all_captions)

    else:
        raise TypeError(f"Unexpected data_dict type: {type(data_dict)}")

    if combined_caption:
        combined_text += f"\nCaption: {combined_caption}"

    doc = Document(
        page_content=combined_text,
        metadata={"caption": combined_caption, "source": source_name}
    )

    vector_store.add_documents([doc])
    return doc


def retrieve_from_vector_store():
    return vector_store.as_retriever(search_kwargs={"k": 5})