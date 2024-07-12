#import Essential dependencies
import streamlit as sl
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
import pandas as pd
from streamlit_feedback import streamlit_feedback
from utils.helper_funcs import update_vectordb

from dotenv import load_dotenv
import os
import numpy as np
import phoenix as px
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

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
phnx_endpoint = os.getenv('PHOENIX_COLLECTOR_ENDPOINT')

from phoenix.trace.langchain import LangChainInstrumentor
LangChainInstrumentor().instrument()



eval_model = OpenAIModel(
            model="gpt-4o",
        )

# use bge reranker
reranker = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')
redundant_filter = EmbeddingsRedundantFilter(embeddings=reranker)
relevant_filter = EmbeddingsFilter(embeddings=reranker, k=4)

# compressor
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[redundant_filter, relevant_filter]
)

# load FAQs data 
faq_collect = pd.read_excel('faq_timestamp.xlsx')

#function to load the vectordatabase
def load_knowledgeBase(db_path = 'vectorstore/db_faiss'):
        embeddings=OpenAIEmbeddings(api_key=openai_api_key)
        DB_FAISS_PATH = db_path
        db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True)
        return db
        
#function to load the OPENAI LLM
def load_llm():
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)
        return llm

#creating prompt template using langchain
def load_prompt():
        # prompt = """ You need to answer the question in the sentence as same as in the pdf content. 
        # Given below is the context and question of the user.
        # context = {context}
        # question = {question}
        # if the answer is not in the pdf answer "i donot know what the hell you are asking about"
        #  """

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


if __name__=='__main__':
        sl.header("welcome to the 📝PDF bot")
        sl.write("🤖 You can chat by Entering your queries ")
        knowledgeBase=load_knowledgeBase()
        faqBase=load_knowledgeBase(db_path='vectorstore/faq_db_faiss')

        llm=load_llm()
        prompt=load_prompt()
        
        query=sl.text_input('Enter some text')
        gen_button = sl.button("Generate Answer")
        
        if(query):
                faqs = faqBase.similarity_search_with_score(query,k=5)
                faqs_index = [doc.metadata['index'] for doc,score in faqs if score<=0.1] # filter out search results with score threshold
                
                if len(faqs_index)>0:
                        sl.subheader('Find similar questions in FAQs')
                        faq_collect_filter = faq_collect.loc[faq_collect['index_col'].isin(faqs_index)]
                        for _,row in faq_collect_filter.iterrows():
                                with sl.expander(row['Original Question']):
                                        sl.write(row['final answer'] + '\n\n')
                                        sl.write("Source: " + row['Source'])
                                        sl.write('\n\n\n\n')
                        sl.divider()

                if gen_button:
                        with sl.spinner("Generating..."):
                                sl.subheader('Generated answer')
                                #getting all chunks for key word searching
                                all_docs = [doc for _id, doc in knowledgeBase.docstore._dict.items()]

                                #getting only the chunks that are similar to the query for llm to produce the output
                                similar_docs=knowledgeBase.similarity_search(query,k=25)
                                similar_embeddings=FAISS.from_documents(documents=similar_docs, embedding=OpenAIEmbeddings(api_key=openai_api_key))
                                
                                #creating the chain for integrating llm,prompt,stroutputparser
                                retriever = similar_embeddings.as_retriever(k=25)

                                compression_retriever_faq = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever = retriever)
                                
                                # initialize the bm25 retriever and faiss retriever
                                bm25_retriever = BM25Retriever.from_documents(all_docs)
                                bm25_retriever.k = 2

                                # emsemble all searching models and filters
                                ensemble_retriever = EnsembleRetriever(retrievers=[compression_retriever_faq,bm25_retriever], weights=[0.8, 0.2])


                                rag_chain = (
                                        {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
                                        | prompt
                                        | llm
                                        | StrOutputParser()
                                )
                                
                                response=rag_chain.invoke(query)
                                sl.write(response)
                                # Display retrieved documents
                                # retrieved_docs = retriever.similarity_search(query)
                                retrieved_docs = ensemble_retriever.get_relevant_documents(query)
                                # print(retrieved_docs)
                                sl.subheader("Retrieved Document Snippets:")
                                for doc in retrieved_docs:
                                        with sl.expander(doc.metadata['source'].split('/')[-1] + ', ' + 'page: ' + str(doc.metadata['page'])):
                                                sl.write(doc.page_content)

                                feedback_temp = streamlit_feedback(
                                feedback_type="thumbs",
                                optional_text_label="Please confirm your feedback. [Optional] Provide an explanation",
                                key=f'thumbs')


                




                                ## log query
                                queries_df = get_qa_with_reference(px.Client(endpoint=phnx_endpoint))
                                retrieved_documents_df = get_retrieved_documents(px.Client(endpoint=phnx_endpoint))
                                # print(px.Client().get_spans_dataframe())
                                spans_dataframe = px.Client(endpoint=phnx_endpoint).get_spans_dataframe()
                                # spans_dataframe = spans_dataframe.loc[(spans_dataframe['name']=='RunnableSequence')&(spans_dataframe['parent_id'].apply(lambda x:x is None))]
                                spans_dataframe.to_csv('logged_query.csv',index=False)


                                ### uncomment below to add evaluation to the log - cost much more tokens

                                # ragas_eval_dataset = queries_df.copy()
                                # print(ragas_eval_dataset)
                                # print(queries_df['input'].iloc[0])
                                # # print(queries_df['reference'].iloc[0])
                                # # print(retrieved_documents_df)

                                # dataset_dict = {
                                #         "question": [x for x in ragas_eval_dataset['input']],
                                #         "answer": [x for x in ragas_eval_dataset['output']],
                                #         "contexts": [[x] for x in ragas_eval_dataset['reference']]
                                # }

                                # ragas_eval_dataset_final = Dataset.from_dict(dataset_dict)

                                # # Log the traces to the project "ragas-evals" just to view
                                # # how Ragas works under the hood
                                # print('test1')

                                # evaluation_result = evaluate(
                                #         dataset=ragas_eval_dataset_final,
                                #         metrics=[faithfulness, answer_relevancy, context_relevancy],
                                # )

                                # print('test1.5')
                                # eval_scores_df = pd.DataFrame(evaluation_result.scores)
                                # print("eval_scores_df:", eval_scores_df)
                                # # Assign span ids to your ragas evaluation scores (needed so Phoenix knows where to attach the spans).
                                # eval_data_df = pd.DataFrame(evaluation_result.dataset)
                                # print('test2')



                                # queries_list = [x for x in queries_df['input']]
                                # print("eval_data_df.question.to_list():", eval_data_df.question.to_list())
                                # print("reversed(queries_df.input.to_list()):", list(reversed(queries_list)))

                                # assert eval_data_df.question.to_list() == list(
                                #         queries_list  # The spans are in reverse order.
                                # ), "Phoenix spans are in an unexpected order. Re-start the notebook and try again."
                                # eval_scores_df.index = pd.Index(
                                #         list(queries_df.index.to_list()), name=queries_df.index.name
                                # )

                                # # Log the evaluations to Phoenix under the project "llama-index"
                                # # This will allow you to visualize the scores alongside the spans in the UI
                                # for eval_name in eval_scores_df.columns:
                                #         evals_df = eval_scores_df[[eval_name]].rename(columns={eval_name: "score"})
                                #         evals = SpanEvaluations(eval_name, evals_df)
                                #         px.Client().log_evaluations(evals)


                                # hallucination_evaluator = HallucinationEvaluator(eval_model)
                                # qa_correctness_evaluator = QAEvaluator(eval_model)
                                # relevance_evaluator = RelevanceEvaluator(eval_model)

                                # hallucination_eval_df, qa_correctness_eval_df = run_evals(
                                #         dataframe=queries_df,
                                #         evaluators=[hallucination_evaluator, qa_correctness_evaluator],
                                #         provide_explanation=True,
                                # )
                                # relevance_eval_df = run_evals(
                                #         dataframe=retrieved_documents_df,
                                #         evaluators=[relevance_evaluator],
                                #         provide_explanation=True,
                                # )[0]

                                # px.Client().log_evaluations(
                                #         SpanEvaluations(eval_name="Hallucination", dataframe=hallucination_eval_df),
                                #         SpanEvaluations(eval_name="QA Correctness", dataframe=qa_correctness_eval_df),
                                #         DocumentEvaluations(eval_name="Relevance", dataframe=relevance_eval_df),
                                # )


        ## section to add new FAQs
        sl.write('\n')
        sl.divider()
        sl.write('\n')
        sl.subheader('FAQs Editor')
        with sl.expander('Open to add new Q&A pairs or modify existing Q&A pairs'):
                with sl.form(key='my_form_add_QA'):
                        add_qa_question = sl.text_area(':blue[Input question]','',key=f"add_qa_question")
                        add_qa_answer = sl.text_area(':blue[Input answer to the question if any]','',key=f"add_qa_answer")
                        add_qa_source = sl.text_area(':blue[Input sources to the answer if any]','',key=f"add_qa_source")

                        if (add_qa_question is not None) and (add_qa_question.strip()!=''):
                                update_vectordb()
                        sl.form_submit_button('Submit changes', type="primary")
                
        
        
        
        