from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

def gate(query, db, key):
    template = """

        You are an expert summarizer. Given this {ctxt} whihc is extracted from a PDF document, your task is to generate a comprehensive and well-structured summary. 
        Use the information provided and find relevant information to generate a summary similar to the example provided to answer {query}.Use the following format for the output.
        
        Document Summary Structure:
        
        ##Title
            [Extract or infer the main title of the document. If not available, create a concise title based on the content.]
        
        ##Purpose / Objective
            Summarize the primary goal or intent of the document. Why was it written?
        
        ##Key Topics Covered
            List the major topics or themes discussed in the document in bullet points.
        
        ##Detailed Summary
            Provide a concise summary of each section or chapter in the document using appropriate subheadings. Maintain the logical flow of the original content.
        
        Example format:
        
            Section 1: [Section Title]
            Brief summary of this section.
        
            Section 2: [Section Title]
            Brief summary of this section.
        
        ##Important Findings / Insights
            Highlight any key takeaways, research findings, conclusions, or statistics.
        
        ##Conclusion
            Summarize the overall conclusion or final remarks from the document.
        
        ##Glossary (Optional)
            Define any technical terms or acronyms mentioned in the document.
        
        Instructions for Processing:
        
            Keep the language clear, neutral, and professional.
        
        Do not copy-paste large portions from the original text.
        
        Summarize in your own words.
        
        If the document is too long, prioritize the executive summary and most informative sections.
        
        Use bullet points or short paragraphs for readability.


    """
    llm = init_chat_model(model="command-r-plus", model_provider="cohere", cohere_api_key = key)
    prompt = PromptTemplate.from_template(template)
    chain_extract = prompt | llm
    retrieved_docs = db.similarity_search(query=query)
    ctxt = "\n\n".join([doc.page_content for doc in retrieved_docs])
    answer = chain_extract.invoke(input={"ctxt": ctxt, "query": query})
    return answer
