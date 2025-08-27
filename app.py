import streamlit as st
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import PyPDF2
import docx
import tempfile

def main():
    
    # Utility Functions
    def extract_text(file):
        #for PDF files
        if file.name.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(file)
            return ''.join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        #For DOCX files
        elif file.name.endswith('.docx'):
            doc = docx.Document(file)
            return '\n'.join([para.text for para in doc.paragraphs])
        #For TXT files
        elif file.name.endswith('.txt'):
            return file.read().decode('utf-8')
        else:
            return ""

    @st.cache_resource
    #Translation
    def get_translation_pipeline(target_lang):
        model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"
        return pipeline("translation", model=model_name)

    @st.cache_resource
    #Summarizer
    def get_summarizer():
        return pipeline("summarization", model="facebook/bart-large-cnn")

    @st.cache_resource
    #QnA 
    def get_qa_model():
        return pipeline("question-answering", model="deepset/roberta-base-squad2")

    # Streamlit UI
    st.title("Multilingual Document Q&A & Summarizer (LangChain + Hugging Face + Streamlit)")
    tab1, tab2, tab3 = st.tabs(["Upload & Preview", "Q&A", "Summarize"])

    #File Upload Tab
    with tab1:
        st.header("Upload your document")
        uploaded_file = st.file_uploader("Choose a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])
        document_text = ""
        if uploaded_file:
            document_text = extract_text(uploaded_file)
            st.text_area("Extracted Document Text", document_text, height=300)
    #QnA Tab
    with tab2:
        st.header("Ask Questions (in English)")
        question = st.text_input("Enter your question about the document:")
        target_lang = st.text_input("Target language code (fr, de, es, zh or leave blank for English)", value='')

        if st.button("Get Answer") and document_text and question:
            qa_model = get_qa_model()
            answer = qa_model({'question': question, 'context': document_text})['answer']
            if target_lang:
                try:
                    translator = get_translation_pipeline(target_lang)
                    translated = translator(answer)[0]['translation_text']
                    st.success(f"Answer ({target_lang}): {translated}")
                except Exception as e:
                    st.warning(f"Translation error: {e}")
            else:
                st.success(f"Answer: {answer}")
                
    #Summarizer Tab
    with tab3:
        st.header("Summarize the Document (in English or other language)")
        target_lang_summary = st.text_input("Target language code for summary (fr, de, es, zh or leave blank for English)", value='')

        if st.button("Summarize") and document_text:
            summarizer = get_summarizer()
            summary = summarizer(document_text[:2000])[0]['summary_text']  # Limiting input for demo
            if target_lang_summary:
                try:
                    translator = get_translation_pipeline(target_lang_summary)
                    translated_summary = translator(summary)[0]['translation_text']
                    st.success(f"Summary ({target_lang_summary}): {translated_summary}")
                except Exception as e:
                    st.warning(f"Translation error: {e}")
            else:
                st.success(f"Summary: {summary}")

if __name__ == '__main__':
    main()