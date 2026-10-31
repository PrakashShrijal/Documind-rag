
import streamlit as st

from rag_pipeline import (
    extract_text_from_pdf,
    split_into_chunks,
    store_chunks,
    retrieve_chunks,
    generate_answer
)

st.set_page_config(
    page_title="DocuMind",
    page_icon="📚",
    layout="wide"
)

st.title("📚 DocuMind — AI Document Assistant")

st.write(
    "Upload a PDF to extract its text and store searchable chunks."
)

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:
    if st.button("Process PDF"):
        with st.spinner("Processing your PDF..."):
            pages = extract_text_from_pdf(uploaded_file)

            total_chunks = 0

            for page in pages:
                page_chunks = split_into_chunks(page["text"])

                stored_count = store_chunks(
                    page_chunks,
                    page["page"]
                )

                total_chunks += stored_count

                with st.expander(
                    f"Page {page['page']} — {len(page_chunks)} chunks"
                ):
                    for i, chunk in enumerate(page_chunks, start=1):
                        st.markdown(f"**Chunk {i}**")
                        st.write(chunk)

        st.success("PDF processed successfully!")
        st.success(
            f"Stored {total_chunks} chunks in ChromaDB!"
        )
else:
    st.info("Please upload a PDF to get started.")




st.divider()
st.subheader("Ask DocuMind")

question = st.text_input(
    "Ask a question about your uploaded PDF"
)

if st.button("Generate Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Finding information and generating an answer..."):
            try:
                result = generate_answer(question)

                st.markdown("### Answer")
                st.write(result["answer"])

                if result["sources"]:
                    source_text = ", ".join(
                        f"Page {page}"
                        for page in result["sources"]
                    )
                    st.caption(f"Retrieved sources: {source_text}")

            except Exception as e:
                st.error(
                    "Could not generate an answer. "
                    "Check your API configuration and terminal for details."
                )
                print(f"Gemini error: {e}")
