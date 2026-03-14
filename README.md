# EdCatalyst AI

**Turning ML research into real educational impact.**

EdCatalyst AI is an agentic AI research assistant that analyzes AI/ML research related to education access, especially for rural and low-resource communities. It retrieves relevant papers, evaluates key barriers, identifies grounded research gaps, proposes evidence-backed research ideas, and generates an action plan for teachers and schools.

## Problem

A lot of AI and machine learning research in education focuses on performance, but much less attention is given to whether those ideas can actually work in rural, low-bandwidth, multilingual, and under-resourced environments.

Researchers, educators, and institutions often struggle to answer questions like:

- What are the most important gaps in AI for education access?
- Which papers are most relevant for real-world deployment?
- What practical next steps can teachers and schools take?
- How can research be translated into meaningful educational impact?

## Solution

EdCatalyst AI helps bridge that gap.

Given a topic like:

> AI for rural education low bandwidth

the system:

1. creates a research plan
2. retrieves relevant papers from arXiv
3. filters and scores the most relevant papers
4. uses Amazon Bedrock with Nova reasoning to generate:
   - selected papers
   - grounded access gaps
   - grounded research ideas
   - teacher and school action plans
   - impact score and reasoning

## Key Features

- **Agentic workflow**: retrieval → filtering → reasoning → action planning
- **Grounded output**: gaps and ideas are tied to paper evidence
- **Social impact focus**: designed for education access in under-resourced settings
- **Teacher + school action plan**: not just research insights, but practical recommendations
- **Impact scoring**: estimates how meaningful and actionable a direction is

## Example Output

EdCatalyst AI returns structured output including:

- `selected_papers`
- `access_gaps`
- `research_ideas`
- `action_plan`
- `impact_score`
- `impact_reasons`

## Tech Stack

- **Python**
- **FastAPI**
- **arXiv API**
- **Amazon Bedrock**
- **Amazon Nova (via inference profile)**
- **Pydantic**
- **Uvicorn**

## How It Works

### 1. Planning
The system expands the user topic into refined search queries and screening criteria.

### 2. Retrieval
It fetches papers from arXiv using multiple targeted queries.

### 3. Relevance Scoring
It scores papers using education-access keywords and constraints like rural, offline, low-bandwidth, and underserved.

### 4. Nova Reasoning
It sends the best papers to Amazon Nova through Bedrock and asks for structured JSON output containing:
- grounded gaps
- grounded ideas
- selected papers
- action plan
- impact score

### 5. Final Response
The system returns a clean structured analysis for researchers, educators, and institutions.

## Project Structure

```bash
edcatalyst-ai/
├── main.py
├── requirements.txt
├── README.md
├── venv/
