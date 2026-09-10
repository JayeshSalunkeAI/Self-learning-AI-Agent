from io import BytesIO
from PyPDF2 import PdfReader


def extract_text_from_uploaded_file(file_name: str, file_bytes: bytes) -> str:
    file_name = file_name.lower()

    if file_name.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    if file_name.endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))

        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages_text.append(page_text)

        return "\n".join(pages_text)

    raise ValueError("Only PDF and TXT files are supported.")