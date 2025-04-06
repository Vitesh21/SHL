from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation System",
    description="API for recommending SHL assessments based on job descriptions and queries",
    version="1.0.0"
)

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define request model
class QueryRequest(BaseModel):
    text: str
    max_results: Optional[int] = 10
    max_duration: Optional[int] = None

# Define response model
class Assessment(BaseModel):
    name: str
    url: str
    remote_testing: bool
    adaptive_support: bool
    duration: str
    test_type: str

class RecommendationResponse(BaseModel):
    recommendations: List[Assessment]

# Function to scrape SHL catalog
def scrape_shl_catalog():
    base_url = "https://www.shl.com"
    catalog_url = f"{base_url}/solutions/products/product-catalog/"
    max_retries = 3
    timeout = 10
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            session = requests.Session()
            response = session.get(catalog_url, timeout=timeout, headers=headers, allow_redirects=True)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            assessments = []
            # Find all assessment cards/sections using multiple selector patterns
            assessment_sections = []
            selector_patterns = [
                {'tags': ['div', 'article'], 'classes': ['product-card', 'assessment-item']},
                {'tags': ['div'], 'classes': ['product', 'assessment', 'catalog-item']},
                {'tags': ['section', 'div'], 'classes': ['product-listing', 'assessment-listing']}
            ]
            
            for pattern in selector_patterns:
                sections = soup.find_all(pattern['tags'], class_=lambda x: x and any(c in x for c in pattern['classes']))
                if sections:
                    assessment_sections.extend(sections)
                    break
            
            if not assessment_sections:
                # Try a more general approach to find potential assessment sections
                sections = soup.find_all(['div', 'article', 'section'], class_=lambda x: x and any(keyword in x.lower() for keyword in ['product', 'assessment', 'catalog']))
                if sections:
                    assessment_sections.extend(sections)
                else:
                    raise ValueError("No assessment sections found on the page. The page structure might have changed.")
            
            for section in assessment_sections:
                # Try multiple selectors for name
                name_elem = section.find(['h3', 'h2', '.assessment-title'])
                name = name_elem.text.strip() if name_elem else ''
                
                # Try multiple selectors for URL
                url_elem = section.find('a')
                url = ''
                if url_elem and 'href' in url_elem.attrs:
                    href = url_elem['href']
                    if href.startswith('/'):
                        url = base_url + href
                    elif href.startswith('http'):
                        url = href
                    else:
                        url = base_url + '/' + href
                
                # Try multiple selectors for description
                desc_elem = section.find(['p', '.assessment-description'])
                description = desc_elem.text.strip() if desc_elem else ''
                
                # More robust duration extraction
                duration_match = re.search(r'\d+\s*(?:minutes?|mins?)', description, re.IGNORECASE)
                duration = duration_match.group(0) if duration_match else 'Not specified'
                
                # More comprehensive test type detection
                test_type = 'General'
                if any(keyword in description.lower() or keyword in name.lower() 
                       for keyword in ['cognitive', 'ability', 'aptitude']):
                    test_type = 'Cognitive'
                elif any(keyword in description.lower() or keyword in name.lower() 
                         for keyword in ['personality', 'behavior', 'style']):
                    test_type = 'Personality'
                elif any(keyword in description.lower() or keyword in name.lower() 
                         for keyword in ['skill', 'proficiency', 'knowledge']):
                    test_type = 'Skills'
                
                if name and url:  # Only add if we have at least a name and URL
                    assessments.append(Assessment(
                        name=name,
                        url=url,
                        remote_testing=True,
                        adaptive_support=False,
                        duration=duration,
                        test_type=test_type
                    ))
            
            if not assessments:
                raise ValueError("No valid assessments could be extracted from the page.")
                
            return assessments
            
        except requests.Timeout:
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=503,
                    detail="Failed to fetch assessments: Connection timeout. Please try again later."
                )
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to fetch assessments: Network error - {str(e)}"
                )
        except ValueError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse assessments: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error while fetching assessments: {str(e)}"
            )

# Function to process query and get embeddings
def process_query(query: str):
    return model.encode([query])[0]

# Function to get assessment embeddings
def get_assessment_embeddings(assessments):
    # Create description texts for each assessment
    texts = [f"{a.name} {a.test_type} assessment. Duration: {a.duration}" for a in assessments]
    # Generate embeddings
    return model.encode(texts)

@app.get("/")
async def root():
    return {"message": "Welcome to SHL Assessment Recommendation System"}

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_assessments(request: QueryRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Query text cannot be empty")

        # Get assessments from catalog
        assessments = scrape_shl_catalog()
        
        if not assessments:
            raise HTTPException(status_code=500, detail="Failed to fetch assessments from catalog. Please try again later.")
        
        # Process query
        try:
            query_embedding = process_query(request.text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
        
        # Get assessment embeddings
        try:
            assessment_embeddings = get_assessment_embeddings(assessments)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating assessment embeddings: {str(e)}")
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], assessment_embeddings)[0]
        
        # Create list of (similarity, assessment) tuples and sort by similarity
        ranked_assessments = list(zip(similarities, assessments))
        ranked_assessments.sort(reverse=True)
        
        # Filter by duration if specified
        filtered_assessments = []
        for sim, assessment in ranked_assessments:
            # Only consider assessments with similarity above threshold
            if sim < 0.1:  # Adjust this threshold as needed
                continue
                
            if request.max_duration:
                # Extract numeric duration value
                duration_match = re.search(r'\d+', assessment.duration)
                if duration_match:
                    duration_value = int(duration_match.group())
                    if duration_value <= request.max_duration:
                        filtered_assessments.append(assessment)
                else:
                    # Include assessments with unspecified duration
                    filtered_assessments.append(assessment)
            else:
                filtered_assessments.append(assessment)
        
        if not filtered_assessments:
            if request.max_duration:
                raise HTTPException(
                    status_code=404,
                    detail=f"No assessments found matching the duration criteria of {request.max_duration} minutes"
                )
            raise HTTPException(
                status_code=404,
                detail="No relevant assessments found for your query. Please try with different keywords."
            )
        
        # Return top recommendations
        return RecommendationResponse(
            recommendations=filtered_assessments[:request.max_results]
        )
    
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)