#!/usr/bin/env python
# coding: utf-8

# # Retrieval-Augmented Generation: Question Answering based on Custom Dataset with Open-sourced [LangChain](https://python.langchain.com/en/latest/index.html) Library

# ---
# 
# This notebook has been tested in us-east-1 with **Pytorch 2.0 Python 3.10** kernel
# 
# ---

# ## Installations 

# In[2]:


# get_ipython().run_line_magic('pip', 'install ../dependencies/botocore-1.30.1-py3-none-any.whl ../dependencies/boto3-1.27.1-py3-none-any.whl ../dependencies/awscli-1.28.1-py3-none-any.whl --force-reinstall')
# get_ipython().run_line_magic('pip', 'install langchain==0.0.190 --quiet')
# get_ipython().run_line_magic('pip', 'install pypdf==3.8.1 faiss-cpu==1.7.4 --quiet')


# ## Imports

# In[3]:

import os
from typing import Optional

# External Dependencies:
import boto3
from botocore.config import Config


def get_bedrock_client(
    assumed_role: Optional[str] = None,
    region: Optional[str] = None,
    runtime: Optional[bool] = True,
):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    assumed_role :
        Optional ARN of an AWS IAM role to assume for calling the Bedrock service. If not
        specified, the current active credentials will be used.
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-1").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    runtime :
        Optional choice of getting different client to perform operations with the Amazon Bedrock service.
    """
    if region is None:
        target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    else:
        target_region = region

    #print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        print(f"  Using role: {assumed_role}", end='')
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=str(assumed_role),
            RoleSessionName="langchain-llm-1"
        )
        print(" ... successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

    if runtime:
        service_name='bedrock-runtime'
    else:
        service_name='bedrock'

    bedrock_client = session.client(
        service_name=service_name,
        config=retry_config,
        **client_kwargs
    )

    #print("boto3 Bedrock client successfully created!")
    #print(bedrock_client._endpoint)
    return bedrock_client


import boto3
import json
import os
import sys
from langchain.embeddings import SagemakerEndpointEmbeddings
from typing import Any, Dict, List, Optional

import numpy as np
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.document_loaders import TextLoader

from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts.prompt import PromptTemplate

import streamlit as st


module_path = ".."
sys.path.append(os.path.abspath(module_path))
#from utils import bedrock, print_ww
from langchain.llms.bedrock import Bedrock

os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
#boto3_bedrock = bedrock.get_bedrock_client(os.environ.get('BEDROCK_ASSUME_ROLE', None))

boto3_bedrock = get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)
# ## Step 1. Deploy large language model (LLM) and embedding model in SageMaker JumpStart

# In[4]:


# We will be using the Titan Embeddings Model to generate our Embeddings.
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# - create the Anthropic Model
llm = Bedrock(model_id="anthropic.claude-v2", client=boto3_bedrock, model_kwargs={'temperature': 0,'max_tokens_to_sample':5000})
bedrock_embeddings = BedrockEmbeddings(client=boto3_bedrock)

from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# - create the Anthropic Model

# In[5]:


from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler


class SagemakerEndpointEmbeddingsJumpStart(SagemakerEndpointEmbeddings):
    def embed_documents(self, texts: List[str], chunk_size: int = 5) -> List[List[float]]:
        """Compute doc embeddings using a SageMaker Inference Endpoint.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size defines how many input texts will
                be grouped together as request. If None, will use the
                chunk size specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        results = []
        _chunk_size = len(texts) if chunk_size > len(texts) else chunk_size

        for i in range(0, len(texts), _chunk_size):
            response = self._embedding_func(texts[i : i + _chunk_size])
            results.extend(response)
        return results


class ContentHandler(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs={}) -> bytes:
        input_str = json.dumps({"text_inputs": prompt, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        embeddings = response_json["embedding"]
        return embeddings


content_handler = ContentHandler()
aws_region = boto3.Session().region_name

embeddings = SagemakerEndpointEmbeddingsJumpStart(
    endpoint_name= "jumpstart-dft-hf-textembedding-gpt-j-6b-fp16",
    region_name=aws_region,
    content_handler=content_handler,
) 
from langchain.vectorstores import Chroma, AtlasDB, FAISS


def pdf_tool(filename : str):
    #print(f"filename - {filename.name}")
    with open(filename.name, "wb") as f:
        f.write(filename.getbuffer())

    
    loader = PyPDFLoader(filename.name)

    documents = loader.load()

    
    return documents


# Create the Streamlit app
def main():
    st.set_page_config(page_title="👨‍💻 Document Analytics")
    st.title("👨‍💻 Document Analytics")

    st.write("Please upload your document file below.")

    data = st.file_uploader("Upload a Document" , type="pdf")

    question = st.text_area("Send a Question")
    
    

    if st.button("Submit Query", type="primary"):
    # Create an agent from the CSV file.
        documents = pdf_tool(data)
        docsearch = FAISS.from_documents(documents, embeddings)
        prompt_template = """Human:Use the below text to answer the question. If you don't know the answer, let us know but don't make up the answer yourself. Strictly limit the answer to less than 40 words. Strictly return only the output and nothing else.  \n\n Use the context - \n
        {context}\n Now Answer the following Question :- \n {question} 
        
         Assistant:"""

        # prompt_template = """\n\nHuman: Use the below text to answer the question. \n\n Use the context - \n
        # {context}\n Now Answer the following Question :- \n {question} \n\nAssistant:"""

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        chain = load_qa_chain(llm=llm, prompt=PROMPT)

        docs = docsearch.similarity_search(question, k=20)

        result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)[
            "output_text"
        ]
        st.text_area(result)

    if st.button("Summarize Document", type="primary"):
        documents = pdf_tool(data)
        docsearch = FAISS.from_documents(documents, embeddings)
        prompt_template = """Human:Use the below text to write a high quality summary of around 100 words..  \n\n Use the context - \n
        {context}\n Now Generate the summary:-
        
        Assistant:"""

        # prompt_template = """\n\nHuman: Use the below text to answer the question. \n\n Use the context - \n
        # {context}\n Now Answer the following Question :- \n {question} \n\nAssistant:"""

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context"])

        chain = load_qa_chain(llm=llm, prompt=PROMPT)


        result = chain({"input_documents": documents, "question": question}, return_only_outputs=True)[
            "output_text"
        ]
        st.text_area(result)
if __name__=="__main__":
    main()
    

