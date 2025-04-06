# app.py - Main application file

from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)

# Initialize the embedding model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Load or scrape SHL product catalog data
def load_assessment_data():
    # In production, this might be replaced with a database call
    # For now, we'll use a simple scraping function
    try:
        # Try to load cached data first
        df = pd.read_csv('shl_assessments.csv')
        print("Loaded assessment data from cache")
    except:
        print("Scraping fresh assessment data...")
        df = scrape_shl_catalog()
        # Cache the data
        df.to_csv('shl_assessments.csv', index=False)
    
    # Create embeddings for each assessment
    descriptions = df['description'].fillna('').tolist()
    embeddings = model.encode(descriptions)
    
    return df, embeddings

def scrape_shl_catalog():
    """Scrape the SHL product catalog"""
    url = "https://www.shl.com/solutions/products/product-catalog/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    assessments = []
    
    # This is a placeholder - actual scraping would need to be adapted to the site structure
    assessment_cards = soup.find_all('div', class_='product-card')
    
    for card in assessment_cards:
        name = card.find('h3').text.strip() if card.find('h3') else "Unknown"
        url = card.find('a')['href'] if card.find('a') else "#"
        description = card.find('p', class_='description').text.strip() if card.find('p', class_='description') else ""
        
        # Extract other attributes - this would need customization based on the actual site structure
        remote_testing = "Yes" if "remote" in description.lower() else "No"
        adaptive_irt = "Yes" if "adaptive" in description.lower() or "irt" in description.lower() else "No"
        
        # Extract duration using regex
        duration_match = re.search(r'(\d+)\s*minutes', description, re.IGNORECASE)
        duration = duration_match.group(1) if duration_match else "Unknown"
        
        # Extract test type
        test_type = "Unknown"
        if "cognitive" in description.lower():
            test_type = "Cognitive"
        elif "personality" in description.lower():
            test_type = "Personality"
        elif "skills" in description.lower():
            test_type = "Skills"
        
        assessments.append({
            'name': name,
            'url': url,
            'description': description,
            'remote_testing': remote_testing,
            'adaptive_irt': adaptive_irt,
            'duration': duration,
            'test_type': test_type
        })
    
    return pd.DataFrame(assessments)

# Mock data for development (in production, use the scraper)
def create_mock_data():
    """Create mock data for development purposes"""
    assessments = [
        {
            'name': 'Java Coding Assessment',
            'url': 'https://www.shl.com/solutions/products/coding-assessment/',
            'description': 'Evaluates Java programming skills with practical coding challenges. Duration: 40 minutes. Suitable for all development roles.',
            'remote_testing': 'Yes',
            'adaptive_irt': 'Yes',
            'duration': '40',
            'test_type': 'Skills'
        },
        {
            'name': 'Python Programming Test',
            'url': 'https://www.shl.com/solutions/products/python-test/',
            'description': 'Comprehensive assessment of Python skills including data structures and algorithms. Duration: 45 minutes.',
            'remote_testing': 'Yes',
            'adaptive_irt': 'No',
            'duration': '45',
            'test_type': 'Skills'
        },
        {
            'name': 'SQL Database Skills',
            'url': 'https://www.shl.com/solutions/products/sql-assessment/',
            'description': 'Tests SQL query writing and database knowledge. Duration: 30 minutes.',
            'remote_testing': 'Yes',
            'adaptive_irt': 'No',
            'duration': '30',
            'test_type': 'Skills'
        },
        {
            'name': 'JavaScript Proficiency',
            'url': 'https://www.shl.com/solutions/products/javascript-test/',
            'description': 'Evaluates JavaScript programming skills including DOM manipulation and async programming. Duration: 35 minutes.',
            'remote_testing': 'Yes',
            'adaptive_irt': 'No',
            'duration': '35',
            'test_type': 'Skills'
        },
        {
            'name': 'Full Stack Developer Assessment',
            'url': 'https://www.shl.com/solutions/products/fullstack-assessment/',
            'description': 'Comprehensive test covering front-end, back-end, and database skills. Includes JavaScript, Python, SQL components. Duration: 60 minutes.',
            'remote_testing': 'Yes',
            'adaptive_irt': 'Yes',
            'duration': '60',
            'test_type': 'Skills'
        },
        {
            'name': 'Cognitive Ability Test',
            'url': 'https://www.shl.com/solutions/products/cognitive-test/',
            'description': 'Measures reasoning abilities, problem-solving, and learning aptitude. Duration: 25 minutes.',
            'remote_testing': 'Yes',
            'adaptive_irt': 'Yes',
            'duration': '25',
            'test_type': 'Cognitive'
        },
        {
            'name': 'Personality Profile',
            'url': 'https://www.shl.com/solutions/products/personality-assessment/',
            'description': 'Evaluates work style, team fit, and behavioral tendencies. Duration: 20 minutes.',
            'remote_testing': 'Yes',
            'adaptive_irt': 'No',
            'duration': '20',
            'test_type': 'Personality'
        },
        {
            'name': 'Communication Skills Assessment',
            'url': 'https://www.shl.com/solutions/products/communication-assessment/',
            'description': 'Evaluates written and verbal communication abilities. Duration: 30 minutes. Good for client-facing roles.',
            'remote_testing': 'Yes',
            'adaptive_irt': 'No',
            'duration': '30',
            'test_type': 'Skills'
        },
        {
            'name': 'Data Analyst Assessment',
            'url': 'https://www.shl.com/solutions/products/data-analyst-test/',
            'description': 'Evaluates data manipulation, visualization, and statistical analysis skills. Includes SQL and Python components. Duration: 45 minutes.',
            'remote_testing': 'Yes',
            'adaptive_irt': 'Yes',
            'duration': '45',
            'test_type': 'Skills'
        },
        {
            'name': 'Leadership Potential Assessment',
            'url': 'https://www.shl.com/solutions/products/leadership-assessment/',
            'description': 'Identifies leadership qualities, decision-making styles, and management potential. Duration: 35 minutes.',
            'remote_testing': 'Yes',
            'adaptive_irt': 'Yes',
            'duration': '35',
            'test_type': 'Personality'
        },
        {
            'name': 'Business Collaboration Assessment',
            'url': 'https://www.shl.com/solutions/products/business-collaboration/',
            'description': 'Measures ability to work effectively with business teams, communicate technical concepts, and understand business requirements. Duration: 30 minutes.',
            'remote_testing': 'Yes',
            'adaptive_irt': 'No',
            'duration': '30',
            'test_type': 'Skills'
        },
        {
            'name': 'Programming Logic Test',
            'url': 'https://www.shl.com/solutions/products/programming-logic/',
            'description': 'Language-agnostic test of programming logic and algorithm design. Duration: 25 minutes.',
            'remote_testing': 'Yes',
            'adaptive_irt': 'Yes',
            'duration': '25',
            'test_type': 'Cognitive'
        }
    ]
    df = pd.DataFrame(assessments)
    return df

# Load data
try:
    # Attempt to scrape or load real data
    df_assessments, assessment_embeddings = load_assessment_data()
except:
    # Fall back to mock data for development
    print("Using mock data")
    df_assessments = create_mock_data()
    descriptions = df_assessments['description'].tolist()
    assessment_embeddings = model.encode(descriptions)

# Process query using LLM to extract key requirements
def process_query_with_llm(query):
    """Use Gemini to extract key requirements from the query"""
    prompt = f"""
    Extract key information from this job requisition query:
    
    Query: {query}
    
    Please identify and structure the following information:
    1. Technical skills required (e.g., programming languages, tools)
    2. Soft skills required (e.g., communication, leadership)
    3. Time constraints for assessment (maximum duration in minutes)
    4. Type of assessment needed (e.g., cognitive, personality, skills)
    5. Any other specific requirements
    
    Output the information in a structured format.
    """
    
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(prompt)
    return response.text

# Find relevant assessments based on query
def find_relevant_assessments(query, max_results=10):
    """Find relevant assessments based on the query"""
    # Get query embedding
    query_embedding = model.encode([query])[0]
    
    # Calculate similarity scores
    similarity_scores = cosine_similarity([query_embedding], assessment_embeddings)[0]
    
    # Get structured requirements from LLM
    structured_req = process_query_with_llm(query)
    
    # Extract duration constraint if any
    duration_constraint = None
    duration_match = re.search(r'(\d+)\s*min', query, re.IGNORECASE)
    if duration_match:
        duration_constraint = int(duration_match.group(1))
    
    # Extract test types if mentioned
    test_types = []
    if "cognitive" in query.lower():
        test_types.append("Cognitive")
    if "personality" in query.lower():
        test_types.append("Personality")
    if any(skill in query.lower() for skill in ["java", "python", "sql", "javascript", "coding"]):
        test_types.append("Skills")
    
    # Create a DataFrame with scores
    result_df = df_assessments.copy()
    result_df['similarity_score'] = similarity_scores
    
    # Apply filters based on requirements
    if duration_constraint:
        # Convert duration to numeric, with non-numeric values set to a high value
        result_df['duration_num'] = pd.to_numeric(result_df['duration'], errors='coerce').fillna(999)
        result_df = result_df[result_df['duration_num'] <= duration_constraint]
    
    if test_types:
        result_df = result_df[result_df['test_type'].isin(test_types)]
    
    # Sort by similarity score
    result_df = result_df.sort_values('similarity_score', ascending=False)
    
    # Get top results
    top_results = result_df.head(max_results)
    
    # Format results
    recommendations = []
    for _, row in top_results.iterrows():
        recommendations.append({
            'name': row['name'],
            'url': row['url'],
            'remote_testing': row['remote_testing'],
            'adaptive_irt': row['adaptive_irt'],
            'duration': f"{row['duration']} minutes",
            'test_type': row['test_type']
        })
    
    return recommendations

# Process job description URL
def process_job_description_url(url):
    """Extract content from a job description URL"""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract job description text - this would need customization based on URL structure
        job_description = soup.find('div', class_='job-description').text.strip()
        return job_description
    except Exception as e:
        return f"Error processing URL: {str(e)}"

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/recommend', methods=['GET'])
def recommend():
    query = request.args.get('query', '')
    url = request.args.get('url', '')
    
    if url:
        job_description = process_job_description_url(url)
        recommendations = find_relevant_assessments(job_description)
    else:
        recommendations = find_relevant_assessments(query)
    
    return jsonify({'recommendations': recommendations})

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """Endpoint to evaluate model performance using benchmark queries"""
    test_queries = request.json.get('queries', [])
    ground_truth = request.json.get('ground_truth', [])
    
    recall_scores = []
    ap_scores = []
    
    for i, query in enumerate(test_queries):
        recommendations = find_relevant_assessments(query)
        rec_names = [rec['name'] for rec in recommendations[:3]]
        
        # Calculate Recall@3
        relevant_count = sum(1 for name in rec_names if name in ground_truth[i])
        recall = relevant_count / len(ground_truth[i]) if ground_truth[i] else 0
        recall_scores.append(recall)
        
        # Calculate AP@3
        ap = 0
        rel_count = 0
        for j, name in enumerate(rec_names):
            if name in ground_truth[i]:
                rel_count += 1
                precision = rel_count / (j + 1)
                ap += precision
        
        if rel_count > 0:
            ap /= min(3, len(ground_truth[i]))
        ap_scores.append(ap)
    
    mean_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    mean_ap = sum(ap_scores) / len(ap_scores) if ap_scores else 0
    
    return jsonify({
        'mean_recall_at_3': mean_recall,
        'map_at_3': mean_ap,
        'individual_scores': [
            {'query': test_queries[i], 'recall': recall_scores[i], 'ap': ap_scores[i]}
            for i in range(len(test_queries))
        ]
    })

if __name__ == '__main__':
    app.run(debug=True)
