# Manufacturing Agent

## Goal
- A plant operator wants floor supervisors to get answers from the right
documentation. 
- The system should intelligently route questions to the appropriate source
(safety procedures, maintenance manuals, or quality control standards) and provide accurate answers. 

- This is a single agent that retrieves the required information along with source/citation from a local Weaviate Vector DB.

## Input
- A floor supervisor queries the agent about various scenarios such as safety procedures, maintenance manuals, or quality control standards. 
- If user's query does not fall under any of the options, prompt the user to only ask questions within these categories.
- Use NeMo Guardrails with GPT 5.6 Luna as the guardrail LLM, to detect toxicity, PII and jailbreaking attempts.

## Agent 
- LLM to be used - GPT5.6-Luna
- Tools avaliable - RAG as a tool
- RAG tool : The Agent must be able to determine which category that the user's query falls into. The categories are safety procedures, maintenance manuals, and quality control standards. Frame the tool description accordingly. The parameters of the tool are the query and the category decided by the LLM.

## Output
- Answer provided by the LLM along with document source(name of document, page number etc)

## Schema of the Vector DB
- category (procedures, maintenance manuals, and quality control standards)
- page number
- embedding
- document name
- section
- sub-section

## Vector DB
- create the documents for the various categories. Break down the information into sections and subsections.
- Use section aware chunking approach

## Evals
- Use Ragas as a the evals framework.
- Focus on 3 metrics : response relevancy, faithfulness and context precision.
- Use gemini-3.5-flash-lite as judge.

## Frontend
- A chat interface using Streamlit.
- LLM responses will contain both text as well as citations. Display both.

## Tracing and observability
- Enable tracing and observability using Phoenix, locally installed on this computer
## Implementation
- Agent Framework : Use LangGraph. There will be a guardian agent for the user's question and agent's answer.
- Frontend Framework : Use Streamlit with FastAPI server.
- Use 4 subagents to parallely perform the task: agent development, frontend, evals and RAG(ingestion of documents and retriever).
- Provide structured response using PyDantic
- Memory : checkpointing . This memory is stored in a Postgres database and is called Checkpoint_DB 

## Development Plan
