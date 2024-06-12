import warnings
warnings.filterwarnings('ignore')

import streamlit as st

import os

import openai

import pandas as pd
from PIL import Image
import boto3
from io import StringIO
from rouge import Rouge
import tiktoken

from htbuilder import HtmlElement, div, br, hr, a, p, img, styles
from htbuilder.units import percent, px

from qa_llamaindex import indexgenerator

from llama_index.llms.groq import Groq
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import QueryBundle
from llama_index.core.schema import MetadataMode
from llama_index.core.postprocessor import LongContextReorder 
from llama_index.llms.openai import OpenAI
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from create_context import answer_question
def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: display;}
      footer {visibility: display;}
     .stApp { bottom: 105px; }
    </style>
    """
    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 50, 0, 50),
        width=percent(100),
        color="black",
        text_align="left",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(1.5)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)



openai.api_key =os.environ['SECRET_TOKEN']

# App title
st.set_page_config(page_title="🤗💬 I-Venture @ ISB AI-Chat Bot")
st.header("I-Venture @ ISB AI-Chat Bot")

# Hugging Face Credentials
with st.sidebar:
    st.title('🤗💬I-Venture @ ISB Chat Bot')
    st.success('Access to this Gen-AI Powered Chatbot is provided by  [Anupam](https://www.linkedin.com/in/anupamisb/)!!', icon='✅')
    hf_email = 'anupam_purwar2019@pgp.isb.edu'
    hf_pass = 'PASS'
    st.markdown('📖 This app is hosted by I-Venture @ ISB [website](https://i-venture.org/)!')

#nodes=index.docstore.docs.values()

indexPath_2000=r"Indices/dlabs-indices/updated_data_index"
documentsPath_2000=r"striped_new"
index_2000=indexgenerator(indexPath_2000,documentsPath_2000)
vector_retriever_2000 = VectorIndexRetriever(index=index_2000,similarity_top_k=3,embed_model=OpenAIEmbedding(model="text-embedding-ada-002"))
bm25_retriever_2000 = BM25Retriever.from_defaults(index=index_2000,similarity_top_k=3)
postprocessor = LongContextReorder()
class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        # combine the two lists of nodes
        all_nodes = bm25_nodes + vector_nodes
        all_nodes = postprocessor.postprocess_nodes(nodes=all_nodes,query_bundle=QueryBundle(query))
        return all_nodes
hybrid_retriever = HybridRetriever(vector_retriever_2000,bm25_retriever_2000)
# User-provided prompt
page_bg_img = '''
<style>
body {
background-image: url("https://csrbox.org/media/Hero-Image.png");
background-size: cover;
}
</style>
'''
memory=ChatMemoryBuffer.from_defaults(token_limit=3900) #3900 earlier
rouge = Rouge()

condense_prompt = (
  "Given the following conversation between a user and an AI assistant and a follow up question from user,"
  "rephrase the follow up question to be a standalone question.\n"
  "Chat History:\n"
  "{chat_history}"
  "\nFollow Up Input: {question}"
  "\nStandalone question:")

#Relaxed prompt with language identification, ans in form of bullet points or short paragraphs



context_prompt=(
        "You are a helpful and friendly chatbot who addresses queries in detail and bulleted points regarding I-Venture @ ISB."
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history above and context, to interact and help the user. Never give any kinds of links, email addresses or contact numbers in the answer."
        )

def get_response(prompt,message_history):
    llm_chat = Groq(model="llama3-70b-8192", api_key="API_KEY")
    chat_engine=CondensePlusContextChatEngine.from_defaults(llm=llm_chat,retriever=hybrid_retriever,chat_history=message_history,context_prompt=context_prompt,condense_prompt=condense_prompt,streaming=True)
    nodes = hybrid_retriever.retrieve(prompt.lower())
    response = chat_engine.chat(str(prompt.lower()))
    context_str = "\n\n".join([n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in response.source_nodes])
    validating_prompt = """You are an intelligent bot designed to assist users on an organization's website by answering their queries. You'll be given a user's question and an associated answer. Your task is to determine if the provided answer effectively resolves the query. If the answer is unsatisfactory, return 0.\n
                           Query: {question}  
                           Answer: {answer}
                           Your Feedback:"""
                        
    feedback = OpenAI(model="gpt-3.5-turbo").complete(validating_prompt.format(question=prompt,answer=response.response))
    if feedback.text==str(0): #feedback.text
        st.write("DISTANCE APPROACH")
        response , joined_text=answer_question(prompt.lower())
        scores = rouge.get_scores(response, joined_text)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
        #message_history.append(ChatMessage(role=MessageRole.ASSISTANT,content=str(response)),)
        response_list = [response, prompt , scores]  
        #df = pd.read_csv(f'logs/conversation_logs.csv')
        #new_row = {'Question': str(prompt), 'Answer': response,'Unigram_Recall' : scores[0]["rouge-1"]["r"],'Unigram_Precision' : scores[0]["rouge-1"]["p"],'Bigram_Recall' : scores[0]["rouge-2"]["r"],'Bigram_Precision' : scores[0]["rouge-2"]["r"]}
        #df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
        #df.to_csv(f'logs/conversation_logs.csv', index=False)
        #bucket = 'aiex' # already created on S3
        #csv_buffer = StringIO()
        #df.to_csv(csv_buffer)
        #s3_resource= boto3.resource('s3',aws_access_key_id=os.environ['ACCESS_ID'],aws_secret_access_key= os.environ['ACCESS_KEY'])
        #s3_resource.Object(bucket, 'conversation_log.csv').put(Body=csv_buffer.getvalue()) 
        #st.write(joined_text) 
        return response_list                                 
    else:
        context_str = "\n\n".join([n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in nodes])
        scores=rouge.get_scores(response.response,context_str)
        message = {"role": "assistant", "content": response.response}
        st.session_state.messages.append(message)
        message_history.append(ChatMessage(role=MessageRole.ASSISTANT,content=str(response.response)),)
        response_list = [response.response, prompt , scores]
        #df = pd.read_csv(f'logs/conversation_logs.csv')
        #new_row = {'Question': str(prompt), 'Answer': response.response,'Unigram_Recall' : scores[0]["rouge-1"]["r"],'Unigram_Precision' : scores[0]["rouge-1"]["p"],'Bigram_Recall' : scores[0]["rouge-2"]["r"],'Bigram_Precision' : scores[0]["rouge-2"]["r"] , "Context" : context_str}
        #df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
        #df.to_csv(f'logs/conversation_logs.csv', index=False)
        #bucket = 'aiex' # already created on S3
        #csv_buffer = StringIO()
        #df.to_csv(csv_buffer)
        #s3_resource= boto3.resource('s3',aws_access_key_id=os.environ['ACCESS_ID'],aws_secret_access_key=os.environ['ACCESS_KEY'])
        #s3_resource.Object(bucket, 'conversation_log.csv').put(Body=csv_buffer.getvalue())
        return response_list 



def get_web_answer(question):
    client = openai.OpenAI(api_key="API_KEY", base_url="https://api.perplexity.ai")
    prompt = (
        "You are a helpful and friendly chatbot who addresses queries in detail and bulleted points regarding I-Venture @ ISB."
        "You get relevant context from the internet."
        "Here's the question:\n"
        f"{question}"+ "I-Venture ISB"
        "Instruction: Respectfully decline to answer any questions that are not related to I-Venture @ ISB."
        )
    messages = [{"role":"user","content": prompt}]
    response = client.chat.completions.create(model="llama-3-sonar-large-32k-online",messages=messages)
    return response.choices[0].message.content

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [{"role": "assistant", "content": "Ask anything about I-Venture @ ISB!"}]
if "message_history" not in st.session_state.keys():
    st.session_state.message_history=[ChatMessage(role=MessageRole.ASSISTANT,content="Ask anything aboug I-Venture @ ISB."),]
if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.message_history.append(ChatMessage(role=MessageRole.USER,content=str(prompt)))




for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])
# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("RAG Answer..."):
                st.write("RAG Answer")
                response = get_response(prompt=prompt,message_history=st.session_state.message_history)
                st.write(response[0])
            with st.spinner("Web Answer.."):
                st.write("WEB Answer")
                web_answer = get_web_answer(prompt)
                st.write(web_answer)
