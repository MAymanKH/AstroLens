from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any
import uvicorn
import httpx
from bs4 import BeautifulSoup
import re
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from difflib import SequenceMatcher
import hashlib
from datetime import datetime, timedelta

load_dotenv()

# Initialize Google Generative AI
def initialize_ai():
    """Initialize the Google Generative AI model."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Warning: GOOGLE_API_KEY environment variable not set. AI features will be disabled.")
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        return model
    except Exception as e:
        print(f"Error initializing AI model: {e}")
        return None

# Initialize AI model on startup
ai_model = initialize_ai()

# In-memory cache for storing summarization results
class SummarizationCache:
    def __init__(self, ttl_hours: int = 24):
        """
        Initialize cache with TTL (Time To Live) in hours.
        
        Args:
            ttl_hours: How long to keep cached entries (default: 24 hours)
        """
        self.cache: Dict[str, Dict] = {}
        self.ttl_hours = ttl_hours
    
    def _generate_cache_key(self, url: str) -> str:
        """Generate a unique cache key for a URL."""
        # Normalize URL by removing common variations
        normalized_url = url.lower().strip()
        # Remove trailing slashes and common URL parameters that don't affect content
        normalized_url = re.sub(r'[/?#&]$', '', normalized_url)
        normalized_url = re.sub(r'[?&]utm_[^&]*', '', normalized_url)  # Remove UTM parameters
        
        # Create hash of the normalized URL for consistent key generation
        return hashlib.md5(normalized_url.encode('utf-8')).hexdigest()
    
    def _is_expired(self, timestamp: datetime) -> bool:
        """Check if a cache entry has expired."""
        expiry_time = timestamp + timedelta(hours=self.ttl_hours)
        return datetime.now() > expiry_time
    
    def get(self, url: str) -> Dict[str, Any] | None:
        """
        Retrieve cached response for a URL.
        
        Args:
            url: The URL to look up
            
        Returns:
            Cached response data or None if not found/expired
        """
        cache_key = self._generate_cache_key(url)
        
        if cache_key not in self.cache:
            print(f"DEBUG: Cache miss for URL: {url}")
            return None
        
        entry = self.cache[cache_key]
        
        # Check if entry has expired
        if self._is_expired(entry['timestamp']):
            print(f"DEBUG: Cache entry expired for URL: {url}")
            del self.cache[cache_key]
            return None
        
        print(f"DEBUG: Cache hit for URL: {url}")
        return entry['data']
    
    def set(self, url: str, data: Dict[str, Any]) -> None:
        """
        Store response data in cache.
        
        Args:
            url: The URL to cache
            data: The response data to store
        """
        cache_key = self._generate_cache_key(url)
        
        self.cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now(),
            'url': url  # Store original URL for debugging
        }
        
        print(f"DEBUG: Cached response for URL: {url}")
    
    def clear_expired(self) -> int:
        """
        Remove all expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        expired_keys = []
        
        for key, entry in self.cache.items():
            if self._is_expired(entry['timestamp']):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            print(f"DEBUG: Cleared {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.cache)
        expired_count = 0
        
        for entry in self.cache.values():
            if self._is_expired(entry['timestamp']):
                expired_count += 1
        
        return {
            'total_entries': total_entries,
            'expired_entries': expired_count,
            'active_entries': total_entries - expired_count,
            'ttl_hours': self.ttl_hours
        }

# Initialize global cache instance
summarization_cache = SummarizationCache(ttl_hours=24)

app = FastAPI(
    title="Paper Summarizer API",
    description="API for summarizing research papers from URLs with AI-powered summarization",
    version="1.0.0"
)

# Add CORS middleware to allow requests from Flutter app
app.add_middleware(
    CORSMiddleware,
    # allow_origins=origins,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class SummarizeRequest(BaseModel):
    url: HttpUrl

# Response model
class SummarizeResponse(BaseModel):
    title: str
    link: str
    abstract: str
    introduction: str
    materials_methods: str
    results: str
    discussion: str
    simplified_ai_version: str

def extract_text_content(soup):
    """Extract clean text content from BeautifulSoup object."""
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()
    
    # Get text and clean it up
    text = soup.get_text()
    # Break into lines and remove leading/trailing space
    lines = (line.strip() for line in text.splitlines())
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Drop blank lines
    text = ' '.join(chunk for chunk in chunks if chunk)
    return text

def extract_keywords_from_text(text, max_keywords=10):
    """Extract simple keywords from text (basic implementation)."""
    # This is a very basic keyword extraction
    # In production, you'd use NLP libraries like spaCy or NLTK
    
    # Common stop words to filter out
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
    }
    
    # Extract words and filter
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    word_freq = {}
    for word in words:
        if word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get most frequent words as keywords
    keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in keywords[:max_keywords]]

async def generate_ai_summary_and_keywords(text_content):
    """Use AI to generate summary and keywords for space biology content."""
    if not ai_model or not text_content.strip():
        return None, None
    
    try:
        # Create prompt for AI
        prompt = f"""
        Summarize this space biology paper in simple language for non-experts. Also list 5-8 keywords.

        Content:
        {text_content[:3000]}  # Limit content to avoid token limits

        Please format your response as:
        SUMMARY: [Your summary here - make it accessible to non-experts, focusing on the main findings and implications]

        KEYWORDS: [keyword1, keyword2, keyword3, keyword4, keyword5]

        Guidelines:
        - Keep the summary under 300 words
        - Use simple, clear language that a general audience can understand
        - Focus on what was studied, what was found, and why it matters
        - Keywords should be relevant scientific terms related to space biology
        """
        
        # Generate content using AI
        response = ai_model.generate_content(prompt)
        
        # Check if response is empty or invalid
        if not response or not hasattr(response, 'text') or not response.text:
            print("Warning: AI model returned empty response")
            return None, None
            
        ai_text = response.text.strip()
        if not ai_text:
            print("Warning: AI model returned empty text")
            return None, None
        
        # Parse the response
        summary = ""
        keywords = []
        
        # Extract summary
        summary_match = re.search(r'SUMMARY:\s*(.*?)(?=KEYWORDS:|$)', ai_text, re.DOTALL | re.IGNORECASE)
        if summary_match:
            summary = summary_match.group(1).strip()
        
        # Extract keywords
        keywords_match = re.search(r'KEYWORDS:\s*(.*?)$', ai_text, re.DOTALL | re.IGNORECASE)
        if keywords_match:
            keywords_text = keywords_match.group(1).strip()
            # Parse keywords - handle different formats
            keywords_text = keywords_text.replace('[', '').replace(']', '')
            keywords = [kw.strip() for kw in re.split(r'[,\n]', keywords_text) if kw.strip()]
            # Clean up keywords and limit to 8
            keywords = [kw for kw in keywords if len(kw) > 2][:8]
        
        # Validate results
        if not summary or not keywords:
            print(f"Warning: Failed to parse AI response properly. Summary: {bool(summary)}, Keywords: {len(keywords)}")
            return None, None
        
        return summary, keywords
        
    except Exception as e:
        print(f"Error generating AI summary: {e}")
        return None, None

async def generate_simplified_summary(abstract, introduction, materials_methods, results, discussion):
    """Generate simplified summary using AI."""
    if not ai_model:
        return "AI summarization not available."
    
    try:
        # Combine all sections
        combined_text = f"Abstract: {abstract}"
        # combined_text = f"Abstract: {abstract}\n\nIntroduction: {introduction}\n\nMaterials and Methods: {materials_methods}\n\nResults: {results}\n\nDiscussion: {discussion}"
        
        # Truncate if too long
        if len(combined_text) > 3000:
            combined_text = combined_text[:3000]
        
        prompt = f"""
        Please create a simplified summary of this scientific paper for non-experts. 
        Explain the research in simple terms that anyone can understand.
        
        Paper content:
        {combined_text}
        
        Write a clear, engaging summary that explains:
        1. What the researchers studied
        2. How they did it (methods)
        3. What they found
        4. Why it matters
        
        Use simple language and avoid technical jargon. Keep it under 300 words. Start directly with the subject.
        """
        
        # Generate content using AI
        response = ai_model.generate_content(prompt)
        
        if response and hasattr(response, 'text') and response.text:
            return response.text.strip()
        else:
            return "AI could not generate summary."
            
    except Exception as e:
        print(f"Error generating simplified summary: {e}")
        return f"Error generating simplified summary: {str(e)}"

async def parse_pmc_xml_content(xml_content, original_url):
    """Parse PMC XML content from E-utilities API."""
    try:
        print("DEBUG: Parsing PMC XML content...")
        
        # Parse XML with BeautifulSoup
        soup = BeautifulSoup(xml_content, 'xml')
        
        # Extract title
        title = "Untitled PMC Article"
        title_group = soup.find('title-group')
        if title_group:
            article_title = title_group.find('article-title')
            if article_title:
                title = article_title.get_text().strip()
        
        print(f"DEBUG: Extracted title: {title}")
        
        # Extract abstract
        abstract = ""
        abstract_element = soup.find('abstract')
        if abstract_element:
            # Remove any nested tags and get clean text
            for tag in abstract_element(['title', 'label']):
                tag.decompose()
            abstract = abstract_element.get_text().strip()
        
        print(f"DEBUG: Extracted abstract length: {len(abstract)} chars")
        
        # Extract specific sections
        introduction = ""
        materials_methods = ""
        results = ""
        discussion = ""
        
        # Look for sections in the body
        body = soup.find('body')
        if body:
            # Find sections by title
            sections = body.find_all('sec')
            
            for section in sections:
                # Get section title
                section_title = ""
                title_element = section.find('title')
                if title_element:
                    section_title = title_element.get_text().strip().lower()
                
                # Remove title from section content
                section_copy = section.__copy__()
                if section_copy.find('title'):
                    section_copy.find('title').decompose()
                
                # Remove references, figures, tables
                for unwanted in section_copy(['ref-list', 'fig', 'table-wrap', 'fn-group', 'ref']):
                    unwanted.decompose()
                
                section_text = section_copy.get_text().strip()
                section_text = ' '.join(section_text.split())  # Normalize whitespace
                
                # Categorize sections
                if any(keyword in section_title for keyword in ['introduction', 'background', 'intro']):
                    introduction = section_text
                    print(f"DEBUG: Found introduction section: {len(introduction)} chars")
                elif any(keyword in section_title for keyword in ['material', 'method', 'procedure', 'experimental']):
                    materials_methods = section_text
                    print(f"DEBUG: Found materials/methods section: {len(materials_methods)} chars")
                elif any(keyword in section_title for keyword in ['result', 'finding', 'outcome']):
                    results = section_text
                    print(f"DEBUG: Found results section: {len(results)} chars")
                elif any(keyword in section_title for keyword in ['discussion', 'conclusion', 'implication']):
                    discussion = section_text
                    print(f"DEBUG: Found discussion section: {len(discussion)} chars")
        
        # If sections not found by title, try to extract from paragraphs
        if not introduction or not materials_methods or not results or not discussion:
            print("DEBUG: Attempting to extract sections from paragraph content...")
            all_paragraphs = body.find_all('p') if body else []
            
            for para in all_paragraphs:
                para_text = para.get_text().strip()
                
                # Simple heuristics to identify sections
                if not introduction and len(para_text) > 100:
                    introduction = para_text
                elif not materials_methods and any(word in para_text.lower()[:50] for word in ['material', 'method', 'procedure', 'protocol']):
                    materials_methods = para_text
                elif not results and any(word in para_text.lower()[:50] for word in ['result', 'finding', 'observed', 'measured']):
                    results = para_text
                elif not discussion and any(word in para_text.lower()[:50] for word in ['discussion', 'conclusion', 'suggest', 'implication']):
                    discussion = para_text
        
        # Generate simplified summary using AI
        simplified_summary = await generate_simplified_summary(abstract, introduction, materials_methods, results, discussion)
        
        response_data = SummarizeResponse(
            title=title,
            link=original_url,
            abstract=abstract,
            introduction=introduction,
            materials_methods=materials_methods,
            results=results,
            discussion=discussion,
            simplified_ai_version=simplified_summary
        )
        
        print(f"DEBUG: PMC XML parsing completed successfully")
        return response_data
        
    except Exception as e:
        print(f"DEBUG: Error parsing PMC XML: {e}")
        raise HTTPException(status_code=500, detail=f"Error parsing PMC content: {str(e)}")

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_paper(request: SummarizeRequest):
    """
    Summarize a research paper from the provided URL.
    
    Args:
        request: JSON object containing the URL of the paper to summarize
        
    Returns:
        JSON response with paper title, link, summary, keywords, and abstract
    """
    try:
        print(f"DEBUG: Attempting to fetch URL: {request.url}")
        
        # Try multiple strategies for PMC access
        url_str = str(request.url)
        
        # Initialize response variable
        response = None
        
        # PMC E-utilities API strategy
        if 'pmc.ncbi.nlm.nih.gov' in url_str:
            print("DEBUG: Detected PMC URL, using NCBI E-utilities API...")
            
            # Extract PMC ID from URL
            pmc_id_match = re.search(r'/PMC(\d+)', url_str)
            if pmc_id_match:
                pmc_id = f"PMC{pmc_id_match.group(1)}"
                print(f"DEBUG: Extracted PMC ID: {pmc_id}")
                
                # Get NCBI API key from environment
                ncbi_api_key = os.getenv('NCBI_API_KEY')
                if ncbi_api_key:
                    print(f"DEBUG: Using NCBI API key: {ncbi_api_key[:8]}...")
                    
                    # Build E-utilities URL
                    eutils_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmc_id}&rettype=full&retmode=xml&api_key={ncbi_api_key}"
                    
                    print(f"DEBUG: E-utilities URL: {eutils_url}")
                    
                    try:
                        headers = {
                            'User-Agent': 'AstroLens-PaperSummarizer/1.0 (contact@example.com)',
                            'Accept': 'application/xml, text/xml, */*',
                        }
                        
                        async with httpx.AsyncClient(timeout=60.0) as client:
                            pmc_response = await client.get(eutils_url, headers=headers)
                            
                            print(f"DEBUG: E-utilities response status: {pmc_response.status_code}")
                            print(f"DEBUG: E-utilities content length: {len(pmc_response.content)} bytes")
                            print(f"DEBUG: E-utilities content preview: {pmc_response.text[:500]}")
                            
                            if pmc_response.status_code == 200 and len(pmc_response.content) > 100:
                                print("DEBUG: SUCCESS! Got PMC content via E-utilities API")
                                # Parse XML content instead of HTML
                                return await parse_pmc_xml_content(pmc_response.content, str(request.url))
                            else:
                                print("DEBUG: E-utilities failed, falling back to generic approach")
                                
                    except Exception as e:
                        print(f"DEBUG: E-utilities failed with error: {e}")
                else:
                    print("DEBUG: No NCBI_API_KEY found in environment, skipping E-utilities")
            else:
                print("DEBUG: Could not extract PMC ID from URL")
        
        # Generic approach for non-PMC URLs or if PMC strategies failed
        if response is None:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            print(f"DEBUG: Using generic approach for: {url_str}")
            
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(url_str, headers=headers)
                
                print(f"DEBUG: Response status: {response.status_code}")
                print(f"DEBUG: Content length: {len(response.content)} bytes")
        
        print(f"DEBUG: Final response preview (first 500 chars): {response.text[:500]}")
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title = "Untitled Document"
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
        
        # Look for meta title as backup
        if not title or title == "Untitled Document":
            meta_title = soup.find('meta', property='og:title')
            if meta_title:
                title = meta_title.get('content', '').strip()
        
        # Extract abstract - try multiple approaches
        abstract = ""
        
        # Method 1: Look for meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            abstract = meta_desc.get('content', '').strip()
        
        # Method 2: Look for OpenGraph description
        if not abstract:
            og_desc = soup.find('meta', property='og:description')
            if og_desc:
                abstract = og_desc.get('content', '').strip()
        
        # Method 3: Look for abstract section in body
        if not abstract:
            abstract_section = soup.find(['div', 'section', 'p'], class_=re.compile(r'abstract', re.I))
            if abstract_section:
                abstract = extract_text_content(abstract_section)[:500]
        
        # Method 4: Look for abstract by ID
        if not abstract:
            abstract_by_id = soup.find(id=re.compile(r'abstract', re.I))
            if abstract_by_id:
                abstract = extract_text_content(abstract_by_id)[:500]
        
        # Method 5: Look for first paragraph that might be abstract
        if not abstract:
            paragraphs = soup.find_all('p')
            for p in paragraphs[:5]:  # Check first 5 paragraphs
                text = p.get_text().strip()
                if len(text) > 100:  # Substantial paragraph
                    abstract = text[:500]
                    break
        
        # Fallback: Use first 500 characters of body text
        if not abstract:
            body_text = extract_text_content(soup)
            abstract = body_text[:500] if body_text else "No content available"
        
        # Extract specific sections from HTML content
        introduction = ""
        materials_methods = ""
        results = ""
        discussion = ""
        
        # Try to find sections by headers or content patterns
        all_text = extract_text_content(soup)
        paragraphs = soup.find_all(['p', 'div', 'section'])
        
        for element in paragraphs:
            text = element.get_text().strip()
            if len(text) < 50:  # Skip short elements
                continue
                
            # Check for section indicators
            text_lower = text.lower()
            
            # Look for introduction section
            if not introduction and any(indicator in text_lower[:100] for indicator in ['introduction', 'background', 'overview']):
                introduction = text[:1000]  # Limit length
                print(f"DEBUG: Found introduction section in HTML: {len(introduction)} chars")
            
            # Look for materials/methods section
            elif not materials_methods and any(indicator in text_lower[:100] for indicator in ['materials', 'methods', 'methodology', 'procedure', 'experimental']):
                materials_methods = text[:1000]
                print(f"DEBUG: Found materials/methods section in HTML: {len(materials_methods)} chars")
            
            # Look for results section
            elif not results and any(indicator in text_lower[:100] for indicator in ['results', 'findings', 'outcomes', 'data show']):
                results = text[:1000]
                print(f"DEBUG: Found results section in HTML: {len(results)} chars")
            
            # Look for discussion section
            elif not discussion and any(indicator in text_lower[:100] for indicator in ['discussion', 'conclusion', 'implications', 'suggest']):
                discussion = text[:1000]
                print(f"DEBUG: Found discussion section in HTML: {len(discussion)} chars")
        
        # If sections still not found, extract from general content
        if not introduction or not materials_methods or not results or not discussion:
            print("DEBUG: Using fallback content extraction for missing sections...")
            paragraphs_text = [p.get_text().strip() for p in soup.find_all('p') if len(p.get_text().strip()) > 100]
            
            if not introduction and len(paragraphs_text) > 0:
                introduction = paragraphs_text[0][:1000]
            if not materials_methods and len(paragraphs_text) > 1:
                materials_methods = paragraphs_text[1][:1000]
            if not results and len(paragraphs_text) > 2:
                results = paragraphs_text[2][:1000]
            if not discussion and len(paragraphs_text) > 3:
                discussion = paragraphs_text[3][:1000]
        
        # Generate simplified summary using AI
        simplified_summary = await generate_simplified_summary(abstract, introduction, materials_methods, results, discussion)
        
        response_data = SummarizeResponse(
            title=title,
            link=str(request.url),
            abstract=abstract,
            introduction=introduction,
            materials_methods=materials_methods,
            results=results,
            discussion=discussion,
            simplified_ai_version=simplified_summary
        )
        
        return response_data
        
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=400, detail=f"HTTP error fetching URL: {e.response.status_code} - {str(e)}")
    except httpx.TimeoutException as e:
        raise HTTPException(status_code=400, detail=f"Timeout fetching URL: {str(e)}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Network error fetching URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "Paper Summarizer API",
        "version": "1.0.0",
        "endpoints": {
            "/summarize": "POST - Summarize a research paper from URL",
            "/summarize-get": "GET - Summarize a research paper from URL (browser-friendly, with caching)",
            "/chat": "GET - Chat endpoint that finds relevant papers based on user query",
            "/cache/stats": "GET - Get cache statistics",
            "/cache/clear": "POST - Clear expired cache entries"
        },
        "cache_info": {
            "enabled": True,
            "ttl_hours": summarization_cache.ttl_hours,
            "description": "Responses are cached for 24 hours to improve performance"
        }
    }

@app.get("/summarize-get")
async def summarize_paper_get(url: str):
    """
    Browser-friendly GET endpoint for summarizing papers with caching.
    
    Usage: http://localhost:8000/summarize-get?url=https://example.com/paper
    """
    try:
        # Validate URL format
        from pydantic import HttpUrl, ValidationError
        try:
            validated_url = HttpUrl(url)
        except ValidationError:
            raise HTTPException(status_code=422, detail="Invalid URL format")
        
        # Check cache first
        cached_response = summarization_cache.get(str(validated_url))
        if cached_response:
            print(f"DEBUG: Returning cached response for URL: {url}")
            return cached_response
        
        # Cache miss - process the request
        print(f"DEBUG: Cache miss - processing URL: {url}")
        
        # Create request object and use the existing logic
        request = SummarizeRequest(url=validated_url)
        response_data = await summarize_paper(request)
        
        # Convert Pydantic model to dict for caching
        response_dict = response_data.dict() if hasattr(response_data, 'dict') else response_data
        
        # Store in cache
        summarization_cache.set(str(validated_url), response_dict)
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/favicon.ico")
async def favicon():
    """Favicon endpoint to prevent 404 errors."""
    return {"message": "No favicon"}

@app.options("/summarize-get")
async def summarize_options():
    """Handle preflight OPTIONS request for CORS."""
    return {}

def load_papers() -> List[Dict[str, Any]]:
    """Load papers from local JSON file.

    Supports two formats:
    - Legacy: a top-level list of paper objects.
    - Categorized: a top-level dict where each key is a category and the value is a list of paper objects.

    The function flattens either structure into a single List[Dict[str, Any]] so the API
    can work with papers regardless of categories. If a paper item does not include a
    'category' field, the loader will set it from the surrounding category (helpful for
    downstream debugging), but the rest of the API will ignore categories.
    """
    try:
        papers_path = os.path.join(os.path.dirname(__file__), "../assets/papers.json")
        with open(papers_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        papers: List[Dict[str, Any]] = []

        # Legacy format: top-level list of paper objects
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    papers.append(item)

        # Categorized format: top-level dict mapping category -> list of papers
        elif isinstance(data, dict):
            for category, items in data.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if isinstance(item, dict):
                        # Preserve category on the item if not present; API will not depend on it.
                        if "category" not in item:
                            try:
                                item["category"] = str(category)
                            except Exception:
                                pass
                        papers.append(item)
        else:
            print("Warning: Unexpected papers.json structure — expected list or dict")

        return papers
    except FileNotFoundError:
        print("Warning: papers.json file not found")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing papers.json: {e}")
        return []
    except Exception as e:
        print(f"Error loading papers: {e}")
        return []

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two strings using SequenceMatcher."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def find_most_relevant_paper(query: str, papers: List[Dict[str, Any]]) -> tuple[Dict[str, Any], float]:
    """Find the most relevant paper based on comprehensive string similarity matching."""
    if not papers:
        return None, 0.0
    
    query_lower = query.lower().strip()
    best_paper = None
    best_score = 0.0
    
    for paper in papers:
        # Get paper content
        title = paper.get('title', '').lower()
        summary = paper.get('summary', '').lower()
        keywords = [kw.lower() for kw in paper.get('keywords', [])]
        
        # Combine all text content for comprehensive matching
        combined_text = f"{title} {summary} {' '.join(keywords)}"
        
        # Calculate primary similarity using SequenceMatcher
        primary_score = SequenceMatcher(None, query_lower, combined_text).ratio()
        
        # Calculate individual field similarities
        title_similarity = SequenceMatcher(None, query_lower, title).ratio()
        summary_similarity = SequenceMatcher(None, query_lower, summary).ratio()
        
        # Calculate keyword similarities
        keyword_similarities = [
            SequenceMatcher(None, query_lower, keyword).ratio() 
            for keyword in keywords
        ]
        max_keyword_similarity = max(keyword_similarities) if keyword_similarities else 0.0
        
        # Calculate word-level matches for bonus scoring
        query_words = set(query_lower.split())
        paper_words = set(combined_text.split())
        word_overlap = len(query_words.intersection(paper_words)) / len(query_words) if query_words else 0.0
        
        # Weighted final score combining different similarity measures
        final_score = (
            primary_score * 0.4 +           # Overall similarity: 40%
            title_similarity * 0.25 +       # Title similarity: 25%
            summary_similarity * 0.2 +      # Summary similarity: 20%
            max_keyword_similarity * 0.1 +  # Best keyword match: 10%
            word_overlap * 0.05            # Word overlap bonus: 5%
        )
        
        if final_score > best_score:
            best_score = final_score
            best_paper = paper
    
    return best_paper, best_score

async def generate_conversational_response(user_query: str, paper_summary: str, paper_title: str) -> str:
    """Generate conversational AI response based on user query and paper content."""
    if not ai_model:
        return f"Based on your question about '{user_query}', I found this relevant research: {paper_summary}"
    
    try:
        prompt = f"""
        User asked: "{user_query}"
        
        Based on this research paper: "{paper_title}"
        Summary: {paper_summary}
        
        Please provide a conversational response that:
        1. Directly addresses the user's question
        2. Uses the paper's findings to answer their query
        3. Explains the research in simple, conversational language
        4. Is engaging and helpful
        5. Keeps the response under 200 words
        
        Answer in simple conversational language as if you're having a friendly chat about science.
        """
        
        # Generate content using AI
        response = ai_model.generate_content(prompt)
        
        if response and hasattr(response, 'text') and response.text:
            return response.text.strip()
        else:
            # Fallback response
            return f"Based on your question about '{user_query}', I found this relevant research: {paper_summary}"
            
    except Exception as e:
        print(f"Error generating conversational response: {e}")
        # Fallback response
        return f"Based on your question about '{user_query}', I found this relevant research: {paper_summary}"

@app.get("/chat")
async def chat_endpoint(message: str):
    """
    Chat endpoint that finds the most relevant paper and generates AI-powered conversational responses.
    
    Args:
        message: User query string
        
    Returns:
        JSON response with AI-generated conversational response and link to most relevant paper
    """
    try:
        if not message or not message.strip():
            raise HTTPException(status_code=422, detail="Message parameter is required and cannot be empty")
        
        # Load papers from JSON file
        papers = load_papers()
        
        if not papers:
            return {
                "response": "I'm sorry, but I couldn't load the research papers database. Please try again later.",
                "link": None
            }
        
        # Find the most relevant paper using enhanced similarity matching
        relevant_paper, similarity_score = find_most_relevant_paper(message, papers)
        
        # Check if we found a good match (minimum threshold)
        if not relevant_paper or similarity_score < 0.1:
            return {
                "response": "Hmm, I couldn't find any papers that closely match your query. Could you try asking about topics like bone loss in space, stem cell research in microgravity, or how space affects mice? I have research papers on these space biology topics!",
                "link": None
            }
        
        # Extract paper details
        paper_title = relevant_paper.get('title', 'Unknown Title')
        paper_summary = relevant_paper.get('summary', 'No summary available')
        paper_link = relevant_paper.get('link', '')
        
        # Generate AI-powered conversational response
        ai_response = await generate_conversational_response(message, paper_summary, paper_title)
        
        return {
            "response": ai_response,
            "link": paper_link
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/cache/stats")
async def get_cache_stats():
    """
    Get cache statistics.
    
    Returns:
        JSON response with cache statistics including total entries, 
        expired entries, active entries, and TTL configuration
    """
    try:
        # Clean up expired entries first
        summarization_cache.clear_expired()
        
        # Get current stats
        stats = summarization_cache.get_cache_stats()
        
        return {
            "cache_stats": stats,
            "message": f"Cache contains {stats['active_entries']} active entries out of {stats['total_entries']} total entries"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving cache stats: {str(e)}")

@app.post("/cache/clear")
async def clear_expired_cache():
    """
    Manually clear expired cache entries.
    
    Returns:
        JSON response with number of entries cleared
    """
    try:
        cleared_count = summarization_cache.clear_expired()
        
        return {
            "message": f"Successfully cleared {cleared_count} expired cache entries",
            "entries_cleared": cleared_count,
            "cache_stats": summarization_cache.get_cache_stats()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "paper-summarizer-api"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
