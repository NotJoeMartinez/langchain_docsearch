import os
import click
from .config import get_or_create_config_path

# import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma

DOCS_VERSION_NUMBER = '0.0.2'

@click.group()
@click.version_option(DOCS_VERSION_NUMBER, message='docs version: %(version)s')
def cli():
    """
    A CLI for conversational retrieval using langchain and OpenAI.

    QUERY is the initial question to start the conversation.
    """
    pass


@click.command(help='The initial question to start the conversation.')
@click.argument('query', required=True )
def search(query):
    
    config_path = os.path.expanduser('~/.config/docs/persist')

    if not os.path.exists(config_path):
        print("No datasets found. use load command to load a dataset.")
        return
    
    vectorstore = Chroma(persist_directory=config_path, embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)

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


@click.command(help='Load a dataset from a directory')
@click.argument('data_dir', required=True)
def load(data_dir):
    persistant_path = os.path.join(get_or_create_config_path(), "persist")
    loader = DirectoryLoader(data_dir)
    index = VectorstoreIndexCreator(vectorstore_kwargs={
        "persist_directory": persistant_path 
        }).from_loaders([loader])
    
    # print(f"Loaded {len(index.vectorstore)} documents. to {persistant_path}")



    


cli.add_command(search)
cli.add_command(load)