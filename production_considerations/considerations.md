# Considerations:
 
## Deployment Strategy

**Question: Describe how you would deploy this chatbot in a production environment. What platforms or services would you use and why?**

As described in the repo, the main scripts for the chatbot would be containerized using Docker, along with the monitoring tools and external database. Then all the containers would be deployed to a cluster (can either be cloud or on-prem) and orchestrated using kubernetes yaml files. And then services would be created to expose the applications to users.

Cloud platforms such as Azure could be a better choice for the chatbot deployment in production considering their scalability and diverse integrated functions such as monitoring, model training and other analytics tools.

If there is strigent privacy and data security policy, I would also consider on-prem platforms. We can allocate resources from on-prem clusters and apply Openshift to deploy and orchestrate the app.

 
## Scalability
**Question: How would you ensure that the chatbot can scale to handle a large number of concurrent users?**

Deploying the chatbot on a cloud platform enables automatic scaling of resources based on user demand. This approach ensures that resources are scaled up or down according to the number of active users and their requests with minimum manual efforts.

Additionally, we can pre-allocate sufficient resources within the cluster. Each user interaction can trigger the launch of a separate pod—customized as small, basic, or resource-intensive based on specific needs. These pods operate independently to prevent interference. Inactive pods, such as those inactive for over 30 minutes, are automatically terminated to free up resources.
 
## Monitoring and Logging
**Question: What tools and metrics would you use to monitor the chatbot’s performance and ensure it is operating correctly in production?**

From deployment perspective, some cloud platforms such as AWS and Openshift have already integrated pod performance watchers to monotor specific metrics. But we can also leverage open source tools such as Prometheus and Grafana to customize our visualization.

Interesting metrics:
* CPU usage/requests/limits
* Memory usage/requests/limits
* User query response time
* Request error ratio
* Number of user query
* Number of active sessions

From RAG performance perspective, we can evaluate RAG performance using RAGAs framework and use tools such as Langsmith or Phoenix for monitoring. We can also log users' feedback directly.

Interesting metrics:
* Faithfulness
* Answer relevancy
* Context recall
* Context precision
* negative/positive feedback ratio

 
## Vector Database Improvements
**Question: The current implementation uses a basic vector database. What are the limitations of this approach, and what alternatives would you suggest for a production-ready system?**


Limitations:
A basic vector database struggles with multimodal data such as images, charts, or tables, and can be slow and resource-intensive when handling large datasets or extensive documents.

Alternatives:

*	Utilizing an unstructured data library, though this may increase costs due to the use of OCR models for PDF layout detection.
*	Implementing a distributed vector database like Pinecone or Weaviate, which supports large datasets and horizontal scaling.
*	For systems with GPU capabilities, employing FAISS-GPU can significantly accelerate vector similarity searches, enhancing overall performance.
 


## Reranking and Retrieval Improvements
**Question: How would you improve the retrieval and reranking components of the chatbot to ensure more accurate and relevant responses?**
 

Implemented in this repo:
- Hybrid search: combine key-word based search and embedding search
- Integrating reranking (https://huggingface.co/BAAI/bge-base-en-v1.5) and redundancy filter
- Prompt optimization


Other potential methods:
- Query expansion
- TF-IDF to distinguish similar but distant words such as 'Google earnings' and 'Apple earnings'
- Fine-tune embedding models
- Metadata filtering (for vectorDB with many different documents)
- Context window (parent documents)



## Performance Optimization
**Question: What techniques would you use to optimize the performance of the chatbot?**

In terms of improving chatbot output quality, can refer to the 2 questions above.
Other options could be fine-tune foundation model, agentic workflow, chain of thoughts, self criticizing etc.

To improve the app performance, we can:
* Parallel processing: leverage scalable system to parallel process multiple users’ requests, and free resources from inactive sessions
* Cache FAQs for repeated queries
* Refine vectorDB searching process: metadata filtering etc.
* Quantize LLMs for inference to save memory footprint and speed up response




## Testing
**Question: What types of testing would you perform to ensure the chatbot is reliable and performs well in production?**

To ensure reliability and optimal performance in production, I would implement the following testing methods:

* Scalable Evaluation: Utilize scalable evaluation methods such as RAGAs (Retrieval-Augmented Generation Assessment) on testing datasets. This approach helps in assessing the chatbot’s accuracy and relevance in handling various queries.
* User Acceptance Testing (UAT): Collect real user feedback during the UAT phase. This direct feedback from end-users helps identify practical issues and user experience improvements, ensuring the chatbot meets user expectations and requirements.