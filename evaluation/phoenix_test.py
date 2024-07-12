import phoenix as px
import json
import pandas as pd
import argparse

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter,DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever, BM25Retriever, EnsembleRetriever

from phoenix.evals import (
    HallucinationEvaluator,
    OpenAIModel,
    QAEvaluator,
    RelevanceEvaluator,
    run_evals,
)
from phoenix.trace import using_project
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    context_relevancy
)


from datasets import Dataset
from phoenix.session.evaluation import get_qa_with_reference, get_retrieved_documents
from phoenix.trace import DocumentEvaluations, SpanEvaluations
import time


from phoenix.trace.langchain import LangChainInstrumentor
from dotenv import load_dotenv
import os

load_dotenv('.env')
openai_api_key = os.getenv('OPENAI_API_KEY')
phnx_endpoint = os.getenv('PHOENIX_COLLECTOR_ENDPOINT')


# By default, the traces will be exported to the locally running Phoenix 
# server. If a different endpoint is desired, change the environment




#set up parser
parser = argparse.ArgumentParser(description='whether apply reranking and retrieval improvements')
parser.add_argument('--better_rag', type=str, default='no', choices=['no', 'yes'],
                    help='Whether apply reranking and retrieval improvements. Choices: yes, no. Default: no')
args = parser.parse_args()

if args.better_rag == 'yes':
        # use bge reranker
        reranker = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')
        redundant_filter = EmbeddingsRedundantFilter(embeddings=reranker)
        relevant_filter = EmbeddingsFilter(embeddings=reranker, k=3)

        # compressor
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[redundant_filter, relevant_filter]
        )

        phoenix_project_name = 'Refined RAG'
else:
        phoenix_project_name = 'Vanilla RAG'

os.environ['PHOENIX_PROJECT_NAME'] = phoenix_project_name
LangChainInstrumentor().instrument()

os.environ['OPENAI_API_KEY']=openai_api_key
eval_model = OpenAIModel(
            model="gpt-4o"

        )



#function to load the vectordatabase
def load_knowledgeBase():
        embeddings=OpenAIEmbeddings(api_key=openai_api_key)
        DB_FAISS_PATH = 'vectorstore/db_faiss'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True)
        return db
        
#function to load the OPENAI LLM
def load_llm():
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)
        return llm

#creating prompt template using langchain
def load_prompt():
        prompt = """

                Task:
                You are required to match the user’s question to the content available in a provided PDF document and generate an appropriate response.

                Instructions:

                        1.	Extract and analyze the content from the PDF.
                        2.	Compare the user’s question with the PDF content.
                        3.	If the answer to the user’s question is found in the PDF, generate a response that accurately reflects the information.
                        4.	If the PDF does not contain information relevant to the user’s question, respond with: “I do not have the information you are seeking.”

                Input:

                        Context: {context}
                        Question: {question}
                """
        prompt = ChatPromptTemplate.from_template(prompt)
        return prompt


def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


knowledgeBase=load_knowledgeBase()
llm=load_llm()
prompt=load_prompt()



queries = [
    "What are the primary types of learning in machine learning as mentioned in the document?",
    "Can you explain the concept of VC (Vapnik-Chervonenkis) dimension as described in the text?",
    "What role does the concept of 'Generalization' play in machine learning according to the lecture notes?",
    "Describe the ID3 algorithm for decision trees as outlined in the document.",
    "What is the PAC (Probably Approximately Correct) learning framework and how is it applied?",
    "How does the document explain the use of neural networks in machine learning?",
    "What are the main components of a machine learning system as detailed in the provided PDF?",
    "Discuss the example of the 'Enjoy Sport' learning task mentioned in the text.",
    "Explain the difference between supervised and unsupervised learning as described in the notes.",
    "What is the Candidate-Elimination algorithm and how does it operate based on the information in the document?",
    "Who is living on Mars?",
    ""
]

ground_answers = [
    "The primary types of learning mentioned are supervised learning, unsupervised learning, and reinforcement learning.",
    "The VC dimension is a measure of the capacity of a set of functions that can be learned by a classification algorithm. It represents the largest set of points that can be shattered (correctly classified) by the hypothesis set.",
    "Generalization describes the process of applying the knowledge learned from the training data to new, unseen situations. It involves discovering properties of the data that will be relevant to performing well on future tasks.",
    "The ID3 algorithm is a method used to create a decision tree by splitting the dataset based on the attribute that offers the highest information gain. It recursively partitions the data till it meets a stopping criterion.",
    "PAC learning is a framework that evaluates machine learning algorithms based on the concept that a learner should be able to learn a function from training examples such that, with high probability, the learned function will have low error on unseen examples.",
    "Neural networks are presented as a powerful model for learning complex patterns in data. They consist of layers of interconnected nodes that can learn to represent data features and perform tasks like classification and regression through training.",
    "The main components include data storage, abstraction, generalization, and evaluation. These components represent the stages through which data is transformed into actionable knowledge.",
    "The 'Enjoy Sport' example is a concept learning task where the goal is to determine whether or not one should play a sport based on various weather conditions like sky, air temperature, humidity, etc., using a series of hypotheses represented as a decision tree.",
    "Supervised learning involves training a model on a labeled dataset, where the correct answers (targets) are known, allowing the model to learn a mapping from inputs to outputs. Unsupervised learning involves training a model on data without labels to discover the underlying patterns or distributions.",
    "The Candidate-Elimination algorithm incrementally refines the hypothesis space by removing hypotheses that are inconsistent with any observed training example. It maintains a set of general hypotheses (G) and specific hypotheses (S) that are consistent with all observed examples.",
    "I do not have the information you are seeking."
]

# queries = [
#     "What are the primary types of learning in machine learning as mentioned in the document?",
#     "Can you explain the concept of VC (Vapnik-Chervonenkis) dimension as described in the text?",
#     "What role does the concept of 'Generalization' play in machine learning according to the lecture notes?",
#     ""
# ]

# ground_answers = [
#     "The primary types of learning mentioned are supervised learning, unsupervised learning, and reinforcement learning.",
#     "The VC dimension is a measure of the capacity of a set of functions that can be learned by a classification algorithm. It represents the largest set of points that can be shattered (correctly classified) by the hypothesis set.",
#     "Generalization describes the process of applying the knowledge learned from the training data to new, unseen situations. It involves discovering properties of the data that will be relevant to performing well on future tasks.",

# ]

with using_project(phoenix_project_name):
    for query in queries:

        if args.better_rag == 'no':

            #getting only the chunks that are similar to the query for llm to produce the output
            similar_embeddings=knowledgeBase.similarity_search(query)
            similar_embeddings=FAISS.from_documents(documents=similar_embeddings, embedding=OpenAIEmbeddings(api_key=openai_api_key))

            #creating the chain for integrating llm,prompt,stroutputparser
            retriever = similar_embeddings.as_retriever()
            rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
            
        elif args.better_rag == 'yes':
            #getting all chunks for key word searching
            all_docs = [doc for _id, doc in knowledgeBase.docstore._dict.items()]

            #getting only the chunks that are similar to the query for llm to produce the output
            similar_docs=knowledgeBase.similarity_search(query,k=10)
            similar_embeddings=FAISS.from_documents(documents=similar_docs, embedding=OpenAIEmbeddings(api_key=openai_api_key))
            
            #creating the chain for integrating llm,prompt,stroutputparser
            retriever = similar_embeddings.as_retriever(k=10)

            compression_retriever_faq = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever = retriever)
            
            # initialize the bm25 retriever and faiss retriever
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = 1

            # emsemble all searching models and filters
            ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, compression_retriever_faq], weights=[0.8, 0.2])


            rag_chain = (
                    {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
        response = rag_chain.invoke(query)

    queries_df = get_qa_with_reference(px.Client(phnx_endpoint))
    retrieved_documents_df = get_retrieved_documents(px.Client(phnx_endpoint))

    ragas_eval_dataset = queries_df.copy()
    print(ragas_eval_dataset)
    print(queries_df['input'].iloc[0])
    # print(queries_df['reference'].iloc[0])
    # print(retrieved_documents_df)

    dataset_dict = {
        "question": [x for x in ragas_eval_dataset['input']],
        "answer": [x for x in ragas_eval_dataset['output']],
        "contexts": [[x] for x in ragas_eval_dataset['reference']],
        "ground_truth": ground_answers,
    }

    ragas_eval_dataset_final = Dataset.from_dict(dataset_dict)

    # Log the traces to the project "ragas-evals" just to view
    # how Ragas works under the hood
    print('test1')

    evaluation_result = evaluate(
        dataset=ragas_eval_dataset_final,
        metrics=[faithfulness, answer_correctness, context_recall, context_precision, answer_relevancy],
    )

    print('test1.5')
    eval_scores_df = pd.DataFrame(evaluation_result.scores)
    # print("eval_scores_df:", eval_scores_df)
    # Assign span ids to your ragas evaluation scores (needed so Phoenix knows where to attach the spans).
    eval_data_df = pd.DataFrame(evaluation_result.dataset)
    print('test2')



    queries_list = [x for x in queries_df['input']]
    print("eval_data_df.question.to_list():", eval_data_df.question.to_list())
    print("reversed(queries_df.input.to_list()):", list(reversed(queries_list)))

    assert eval_data_df.question.to_list() == list(
        queries_list  # The spans are in reverse order.
    ), "Phoenix spans are in an unexpected order. Re-start the notebook and try again."
    eval_scores_df.index = pd.Index(
        list(queries_df.index.to_list()), name=queries_df.index.name
    )

    # Log the evaluations to Phoenix under the project "llama-index"
    # This will allow you to visualize the scores alongside the spans in the UI
    for eval_name in eval_scores_df.columns:
        evals_df = eval_scores_df[[eval_name]].rename(columns={eval_name: "score"})
        evals = SpanEvaluations(eval_name, evals_df)
        px.Client(phnx_endpoint).log_evaluations(evals)


    hallucination_evaluator = HallucinationEvaluator(eval_model)
    qa_correctness_evaluator = QAEvaluator(eval_model)
    relevance_evaluator = RelevanceEvaluator(eval_model)

    hallucination_eval_df, qa_correctness_eval_df = run_evals(
        dataframe=queries_df,
        evaluators=[hallucination_evaluator, qa_correctness_evaluator],
        provide_explanation=True,
    )
    relevance_eval_df = run_evals(
        dataframe=retrieved_documents_df,
        evaluators=[relevance_evaluator],
        provide_explanation=True,
    )[0]

    px.Client(phnx_endpoint).log_evaluations(
        SpanEvaluations(eval_name="Hallucination", dataframe=hallucination_eval_df),
        SpanEvaluations(eval_name="QA Correctness", dataframe=qa_correctness_eval_df),
        DocumentEvaluations(eval_name="Relevance", dataframe=relevance_eval_df),
    )
