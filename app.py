def load_documents():
    text = ""
    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            pdf = PdfReader(os.path.join("data", file))
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    return text
