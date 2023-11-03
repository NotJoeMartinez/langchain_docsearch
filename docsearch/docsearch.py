import os
import click

import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader, UnstructuredPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma

DOCS_VERSION_NUMBER = '0.0.1'

@click.version_option(DOCS_VERSION_NUMBER, message='docs version: %(version)s')
@click.command()
@click.option('--persist', is_flag=True, help='Reuse the index if available.')
@click.argument('query', required=False)
def main(persist, query):
    """
    A CLI for conversational retrieval using langchain and OpenAI.

    QUERY is the initial question to start the conversation.
    """
    if persist and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:

        loader = DirectoryLoader("data/")

        if persist:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    chat_history = []
    while True:
        if not query:
            query = input("Prompt: ")
        if query in ['quit', 'q', 'exit']:
            click.echo("Exiting...")
            break
        result = chain({"question": query, "chat_history": chat_history})
        click.echo(result['answer'])

        chat_history.append((query, result['answer']))
        query = None

if __name__ == "__main__":
    main()
