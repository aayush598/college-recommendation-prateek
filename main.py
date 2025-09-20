import os
import logging
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict
import uuid
import sqlite3
from dataclasses import dataclass, asdict

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import uvicorn

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema import OutputParserException
import openai
import re

from dotenv import load_dotenv
import os

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Academic Chatbot API",
    description="Academic chatbot with college recommendation capabilities",
    version="1.0.0"
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str

class CollegeRecommendation(BaseModel):
    """College recommendation model"""
    id: str
    name: str
    location: str
    fees: Optional[int] = None
    match_score: int
    type: str
    admission: str
    highlights: List[str]
    description: str
    established: Optional[int] = None
    ranking: Optional[int] = None
    courses: List[str]
    facilities: List[str]
    source: str

class ChatResponse(BaseModel):
    response: str
    is_academic: bool
    is_recommendation: bool
    timestamp: str
    recommendations: Optional[List[CollegeRecommendation]] = []

class UserPreferences(BaseModel):
    """User preferences extracted from conversation"""
    location: Optional[str] = Field(None, description="Preferred city or state for college")
    state: Optional[str] = Field(None, description="Preferred state for college")
    course_type: Optional[str] = Field(None, description="Type of course like Engineering, Medicine, Arts, Commerce, etc.")
    college_type: Optional[str] = Field(None, description="Government, Private, or Deemed university")
    level: Optional[str] = Field(None, description="UG (Undergraduate) or PG (Postgraduate)")
    budget_range: Optional[str] = Field(None, description="Budget preference like low, medium, high")
    specific_course: Optional[str] = Field(None, description="Specific course like BTech, MBA, MBBS, etc.")

@dataclass
class College:
    college_id: str
    name: str
    type: str
    affiliation: str
    location: str
    website: str
    contact: str
    email: str
    courses: str
    scholarship: str
    admission_process: str

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT,
                message_type TEXT,
                content TEXT,
                is_academic BOOLEAN DEFAULT TRUE,
                is_recommendation BOOLEAN DEFAULT FALSE,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS preferences (
                chat_id TEXT PRIMARY KEY,
                preferences TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_message(self, chat_id: str, message_type: str, content: str, is_academic: bool = True, is_recommendation: bool = False):
        """Save a message"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO messages (chat_id, message_type, content, is_academic, is_recommendation) VALUES (?, ?, ?, ?, ?)',
            (chat_id, message_type, content, is_academic, is_recommendation)
        )
        conn.commit()
        conn.close()
    
    def get_chat_messages(self, chat_id: str) -> List[Dict]:
        """Get messages for a chat"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT message_type, content, timestamp, is_academic, is_recommendation
            FROM messages 
            WHERE chat_id = ? 
            ORDER BY timestamp
        ''', (chat_id,))
        messages = cursor.fetchall()
        conn.close()
        
        return [
            {
                'type': msg[0],
                'content': msg[1],
                'timestamp': msg[2],
                'is_academic': msg[3],
                'is_recommendation': msg[4]
            }
            for msg in messages
        ]
    
    def save_preferences(self, chat_id: str, preferences: dict):
        """Save user preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO preferences (chat_id, preferences) VALUES (?, ?)',
            (chat_id, json.dumps(preferences))
        )
        conn.commit()
        conn.close()
    
    def get_preferences(self, chat_id: str) -> dict:
        """Get user preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT preferences FROM preferences WHERE chat_id = ?',
            (chat_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        return {}

class CollegeDataManager:
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.colleges = self.load_college_data()
    
    def load_college_data(self) -> List[College]:
        """Load college data from Excel file"""
        try:
            if not os.path.exists(self.excel_path):
                logger.warning(f"Excel file not found at {self.excel_path}")
                return []
            
            df = pd.read_excel(self.excel_path)
            colleges = []
            
            for _, row in df.iterrows():
                college = College(
                    college_id=str(row.get('College ID', '')),
                    name=str(row.get('College', '')),
                    type=str(row.get('Type', '')),
                    affiliation=str(row.get('Affiliation', '')),
                    location=str(row.get('Location', '')),
                    website=str(row.get('Website', '')),
                    contact=str(row.get('Contact', '')),
                    email=str(row.get('E-mail', '')),
                    courses=str(row.get('Courses (ID, Category, Duration, Eligibility, Language, Accreditation, Fees)', '')),
                    scholarship=str(row.get('Scholarship', '')),
                    admission_process=str(row.get('Admission Process', ''))
                )
                colleges.append(college)
            
            logger.info(f"Loaded {len(colleges)} colleges from Excel file")
            return colleges
        except Exception as e:
            logger.error(f"Error loading Excel data: {e}")
            return []
    
    def filter_colleges_by_preferences(self, preferences: UserPreferences) -> List[Dict]:
        """Filter colleges based on user preferences"""
        matching_colleges = []
        
        for college in self.colleges:
            match_score = 0
            match_reasons = []
            missing_criteria = []
            
            # Location filtering
            location_match = False
            if preferences.location:
                location_terms = [preferences.location.lower()]
                if preferences.state:
                    location_terms.append(preferences.state.lower())
                
                college_location = college.location.lower()
                for term in location_terms:
                    if term in college_location:
                        location_match = True
                        match_score += 30
                        match_reasons.append(f"Located in {preferences.location}")
                        break
                
                if not location_match:
                    missing_criteria.append(f"Not in preferred location: {preferences.location}")
                    continue
            
            # College type filtering
            if preferences.college_type:
                if preferences.college_type.lower() in college.type.lower():
                    match_score += 25
                    match_reasons.append(f"Matches college type: {preferences.college_type}")
                else:
                    missing_criteria.append(f"Not a {preferences.college_type} college")
                    continue
            
            # Course type filtering
            course_match = False
            if preferences.course_type or preferences.specific_course:
                college_courses = college.courses.lower()
                
                if preferences.specific_course:
                    course_terms = [preferences.specific_course.lower()]
                    if preferences.course_type:
                        course_terms.append(preferences.course_type.lower())
                else:
                    course_terms = [preferences.course_type.lower()]
                
                for term in course_terms:
                    if term in college_courses:
                        course_match = True
                        match_score += 25
                        match_reasons.append(f"Offers {term} courses")
                        break
                
                if not course_match:
                    missing_criteria.append(f"Doesn't offer preferred course type")
                    continue
            
            # Level filtering
            if preferences.level:
                if preferences.level.lower() in college.courses.lower():
                    match_score += 10
                    match_reasons.append(f"Offers {preferences.level} programs")
            
            if match_score > 0:
                matching_colleges.append({
                    'college': college,
                    'score': match_score,
                    'reasons': match_reasons,
                    'missing': missing_criteria
                })
        
        matching_colleges.sort(key=lambda x: x['score'], reverse=True)
        return matching_colleges[:5]

class IntegratedAcademicChatbot:
    def __init__(self, openai_api_key: str, excel_path: str, db_path: str, model_name: str = "gpt-3.5-turbo"):
        """Initialize the integrated chatbot"""
        
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        
        # Initialize LLM for both pipelines
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.7,
            max_tokens=500
        )
        
        self.academic_llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model_name,
            temperature=0.3,
            max_tokens=500
        )
        
        # Initialize managers
        self.db_manager = DatabaseManager(db_path)
        self.college_data_manager = CollegeDataManager(excel_path)
        
        # Memory for context awareness per chat
        self.chat_memories = defaultdict(lambda: ConversationBufferWindowMemory(
            k=10,
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        ))
        
        # Setup prompts and chains
        self._setup_prompts()
        self._create_chains()
    
    def _setup_prompts(self):
        """Setup prompt templates"""
        
        # Academic filter prompt
        self.filter_prompt = ChatPromptTemplate.from_template(
            """You are an academic query classifier. Determine if the following query is related to academics, education, research, or learning.
            
            Academic topics include: 
            - Subject-specific questions (mathematics, science, literature, history, philosophy, etc.)
            - Research methods and study techniques
            - Academic writing and educational concepts
            - Learning strategies and preparation advice
            - Career preparation through education
            - Study guidance and academic planning
            - Educational pathways and skill development
            
            Non-academic topics include: personal relationships, entertainment, shopping, cooking, sports (unless academic analysis), general chat, etc.
            
            Query: {query}
            
            Respond with only 'YES' if it's academic, or 'NO' if it's not academic."""
        )
        
        # College recommendation detector - Enhanced with better discrimination
        self.recommendation_detector_prompt = PromptTemplate(
            template="""
            Determine if the user is SPECIFICALLY asking for college recommendations, suggestions, or wants a LIST of colleges.
            
            User message: "{message}"
            
            Return "YES" ONLY if the user is explicitly asking for:
            - Specific college names or suggestions ("recommend colleges", "which college should I choose", "suggest some colleges")
            - A list of colleges ("show me colleges", "colleges for engineering", "good colleges in Delhi")
            - Help choosing between colleges or finding colleges
            
            Return "NO" if the user is asking for:
            - General advice on how to prepare for college admission
            - What subjects to study or focus on
            - How to get into college (preparation strategies)
            - Study tips or academic guidance
            - Career advice or what to study for jobs
            - General educational guidance
            
            Examples of YES (college recommendations):
            - "recommend some engineering colleges"
            - "which colleges are good for MBA"
            - "suggest colleges in Mumbai"
            - "show me top medical colleges"
            
            Examples of NO (general academic advice):
            - "how can i get into college, how should i prepare for it"
            - "what should i study to get into a good college"
            - "how to prepare for entrance exams"
            - "what subjects should I focus on"
            - "what should i study to get a good job"
            
            Answer with only YES or NO.
            """,
            input_variables=["message"]
        )
        
        # Preference extraction prompt
        self.preference_parser = PydanticOutputParser(pydantic_object=UserPreferences)
        self.preference_prompt = PromptTemplate(
            template="""
            Extract user preferences for college search from the following conversation history.
            Look for mentions of:
            - Location/City/State (like "Indore", "MP", "Delhi", "Bangalore", etc.)
            - Course types (like "Engineering", "Medical", "Commerce", "Arts", "Management")
            - Specific courses (like "BTech", "MBA", "MBBS", "BCom")
            - College types (like "Government", "Private", "Deemed")
            - Level (like "UG", "PG", "Undergraduate", "Postgraduate")
            - Budget preferences

            Conversation History:
            {conversation_history}

            Current Message:
            {current_message}

            {format_instructions}

            Extract preferences as JSON. If no clear preference is mentioned, use null for that field.
            """,
            input_variables=["conversation_history", "current_message"],
            partial_variables={"format_instructions": self.preference_parser.get_format_instructions()}
        )
        
        # Main academic chatbot prompt - Enhanced for comprehensive guidance
        self.academic_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert academic assistant designed to help with educational and research-related queries. Your expertise spans across:

- All academic subjects (STEM, humanities, social sciences)
- Research methodologies and techniques
- Academic writing and citation
- Study strategies and learning techniques
- Educational concepts and theories
- Critical thinking and analysis
- College preparation and admission guidance
- Career preparation through education
- Academic planning and pathway advice

Guidelines:
1. Provide accurate, well-structured, and educational responses
2. Use examples and explanations appropriate for learning
3. Encourage critical thinking and deeper understanding
4. Give practical, actionable advice for academic success
5. Help with study strategies, preparation methods, and educational planning
6. Provide guidance on what to study, how to prepare for exams, and academic skill development
7. Offer career-oriented educational advice when relevant
8. Be supportive and encouraging in your educational approach

Important: You handle ALL academic queries including:
- How to prepare for college admissions
- What subjects to study for specific goals
- Study techniques and academic strategies
- Educational pathways and career preparation
- Academic skill development

Only redirect to college recommendations when users specifically ask for a list of colleges or college suggestions."""),
            
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
    
    def _create_chains(self):
        """Create processing chains"""
        
        # Academic filter chain
        self.filter_chain = self.filter_prompt | self.llm | StrOutputParser()
        
        # Recommendation detector chain
        self.recommendation_detector_chain = LLMChain(llm=self.llm, prompt=self.recommendation_detector_prompt)
        
        # Preference extraction chain
        self.preference_chain = LLMChain(llm=self.academic_llm, prompt=self.preference_prompt)
        
        # Academic conversation chain
        def format_chat_history(inputs):
            return inputs
        
        self.academic_chain = (
            RunnablePassthrough.assign(
                chat_history=lambda x: self.chat_memories[x.get("chat_id", "default")].chat_memory.messages
            )
            | self.academic_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def is_academic_query(self, query: str) -> bool:
        """Check if query is academic"""
        try:
            result = self.filter_chain.invoke({"query": query})
            return result.strip().upper() == "YES"
        except Exception as e:
            logger.error(f"Error in academic filtering: {e}")
            return True
    
    def is_asking_for_recommendations(self, user_input: str) -> bool:
        """Detect recommendation requests with enhanced discrimination"""
        try:
            result = self.recommendation_detector_chain.run(message=user_input)
            is_recommendation = "YES" in result.upper()
            
            # Additional keyword-based validation for edge cases
            user_input_lower = user_input.lower()
            
            # Strong indicators for recommendations (explicit requests)
            strong_recommendation_indicators = [
                'recommend college', 'suggest college', 'which college should',
                'best college for', 'good college for', 'colleges in',
                'show me college', 'list of college', 'college suggestion',
                'college option', 'help me find college', 'looking for college'
            ]
            
            # Strong indicators for general advice (NOT recommendations)
            advice_indicators = [
                'how to get into college', 'how can i get into college',
                'how should i prepare', 'what should i study',
                'how to prepare for', 'what subjects should',
                'how can i prepare', 'preparation for college',
                'study tips', 'how to study'
            ]
            
            # Check for strong advice indicators first (override LLM if present)
            if any(indicator in user_input_lower for indicator in advice_indicators):
                logger.info(f"Advice indicator detected, overriding to general academic: {user_input}")
                return False
            
            # Check for strong recommendation indicators
            if any(indicator in user_input_lower for indicator in strong_recommendation_indicators):
                logger.info(f"Strong recommendation indicator detected: {user_input}")
                return True
            
            logger.info(f"Recommendation detection result: {is_recommendation} for query: {user_input}")
            return is_recommendation
            
        except Exception as e:
            logger.error(f"Error in recommendation detection: {e}")
            
            # Fallback with stricter keyword matching
            user_input_lower = user_input.lower()
            
            # Only trigger recommendations for very explicit requests
            explicit_recommendation_keywords = [
                'recommend college', 'suggest college', 'which college',
                'best college', 'good college', 'colleges in', 'college for',
                'show me college', 'list college'
            ]
            
            return any(keyword in user_input_lower for keyword in explicit_recommendation_keywords)
    
    def extract_preferences_with_llm(self, chat_id: str, current_message: str) -> UserPreferences:
        """Extract user preferences using LLM"""
        try:
            messages = self.db_manager.get_chat_messages(chat_id)
            conversation_history = "\n".join([
                f"{msg['type'].title()}: {msg['content']}" for msg in messages[-10:]
            ])
            
            result = self.preference_chain.run(
                conversation_history=conversation_history,
                current_message=current_message
            )
            
            try:
                preferences = self.preference_parser.parse(result)
                pref_dict = preferences.dict()
                self.db_manager.save_preferences(chat_id, pref_dict)
                return preferences
            except OutputParserException as e:
                logger.error(f"Parser error: {e}")
                fixing_parser = OutputFixingParser.from_llm(parser=self.preference_parser, llm=self.llm)
                preferences = fixing_parser.parse(result)
                return preferences
                
        except Exception as e:
            logger.error(f"Error extracting preferences: {e}")
            prev_prefs = self.db_manager.get_preferences(chat_id)
            if prev_prefs:
                return UserPreferences(**prev_prefs)
            return UserPreferences()
    
    def get_openai_college_recommendations(self, preferences: UserPreferences, location: str = None) -> List[Dict]:
        """Get college recommendations from OpenAI"""
        try:
            pref_parts = []
            if location:
                pref_parts.append(f"Location: {location}")
            if preferences.course_type:
                pref_parts.append(f"Course type: {preferences.course_type}")
            if preferences.specific_course:
                pref_parts.append(f"Specific course: {preferences.specific_course}")
            if preferences.college_type:
                pref_parts.append(f"College type: {preferences.college_type}")
            if preferences.level:
                pref_parts.append(f"Level: {preferences.level}")
            
            preference_text = ", ".join(pref_parts) if pref_parts else "General preferences"
            
            prompt = f"""
            Recommend 5 good colleges/universities in India based on these preferences: {preference_text}
            
            Focus on well-known, reputable institutions. If location is specified, prioritize colleges in that area.
            
            Provide response as a JSON array with this exact structure:
            [
                {{
                    "id": "unique_identifier",
                    "name": "College Name",
                    "location": "City, State",
                    "fees": 300000,
                    "type": "Government/Private/Deemed",
                    "admission": "Entrance exam name or Own Entrance",
                    "highlights": ["Notable feature 1", "Notable feature 2"],
                    "description": "Brief description of the college",
                    "established": 2005,
                    "ranking": 25,
                    "courses": ["Course1", "Course2", "Course3"],
                    "facilities": ["Facility1", "Facility2", "Facility3"],
                    "website": "Official website URL if known"
                }}
            ]
            
            Return only the JSON array, no additional text.
            """
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            
            result = response.choices[0].message.content.strip()
            
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                json_match = re.search(r'\[.*\]', result, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return []
                
        except Exception as e:
            logger.error(f"Error getting OpenAI recommendations: {e}")
            return []
    
    def convert_database_college_to_json(self, college: College, match_score: int, match_reasons: List[str]) -> Dict:
        """Convert database college to JSON format"""
        try:
            # Extract courses from the courses string
            courses_list = []
            if college.courses:
                # Simple extraction - you might want to improve this based on your data format
                courses_text = college.courses.lower()
                if 'btech' in courses_text or 'b.tech' in courses_text:
                    courses_list.append('B.Tech')
                if 'mtech' in courses_text or 'm.tech' in courses_text:
                    courses_list.append('M.Tech')
                if 'mba' in courses_text:
                    courses_list.append('MBA')
                if 'mbbs' in courses_text:
                    courses_list.append('MBBS')
                if 'bca' in courses_text:
                    courses_list.append('BCA')
                if 'mca' in courses_text:
                    courses_list.append('MCA')
                if not courses_list:
                    courses_list = ['Various Programs']
            
            # Extract fees if available in courses string
            fees = None
            if college.courses:
                import re
                fee_match = re.search(r'(\d+(?:,\d{3})*(?:\.\d{2})?)', college.courses)
                if fee_match:
                    try:
                        fees = int(fee_match.group(1).replace(',', ''))
                    except:
                        fees = None
            
            # Generate highlights from match reasons and college data
            highlights = match_reasons[:2] if match_reasons else []
            if college.scholarship and college.scholarship.lower() != 'nan' and 'scholarship' not in str(highlights).lower():
                highlights.append("Scholarship Available")
            if len(highlights) < 2:
                if college.type.lower() == 'government':
                    highlights.append("Government Institution")
                elif college.type.lower() == 'private':
                    highlights.append("Private Institution")
            
            # Basic facilities (you might want to extract this from your data)
            facilities = ["Library", "Campus", "Labs"]
            if college.website and college.website.lower() != 'nan':
                facilities.append("Online Portal")
            
            # Generate description
            description = f"{college.name} is a {college.type.lower()} institution located in {college.location}."
            if college.affiliation and college.affiliation.lower() != 'nan':
                description += f" Affiliated with {college.affiliation}."
            
            # Combine website into source
            source_url = college.website if college.website and college.website.lower() != 'nan' else "database"
            
            return {
                "id": college.college_id,
                "name": college.name,
                "location": college.location,
                "fees": fees,
                "match_score": match_score,
                "type": college.type,
                "admission": college.admission_process if college.admission_process and college.admission_process.lower() != 'nan' else "Check Official Website",
                "highlights": highlights[:3],  # Limit to 3 highlights
                "description": description,
                "established": None,  # Not available in your current data
                "ranking": None,  # Not available in your current data
                "courses": courses_list,
                "facilities": facilities,
                "source": source_url
            }
            
        except Exception as e:
            logger.error(f"Error converting database college to JSON: {e}")
            return None
    
    def convert_openai_college_to_json(self, college_data: Dict, match_score: int = 75) -> Dict:
        """Convert OpenAI college recommendation to standardized JSON format"""
        try:
            # Combine website into source
            website = college_data.get('website', 'openai_knowledge')
            source = website if website and website != 'N/A' else 'openai_knowledge'
            
            return {
                "id": college_data.get('id', str(uuid.uuid4())),
                "name": college_data.get('name', ''),
                "location": college_data.get('location', ''),
                "fees": college_data.get('fees'),
                "match_score": match_score,
                "type": college_data.get('type', ''),
                "admission": college_data.get('admission', ''),
                "highlights": college_data.get('highlights', []),
                "description": college_data.get('description', ''),
                "established": college_data.get('established'),
                "ranking": college_data.get('ranking'),
                "courses": college_data.get('courses', []),
                "facilities": college_data.get('facilities', []),
                "source": source
            }
            
        except Exception as e:
            logger.error(f"Error converting OpenAI college to JSON: {e}")
            return None
    
    def format_college_recommendations(self, filtered_colleges: List[Dict], openai_colleges: List[Dict], preferences: UserPreferences) -> tuple:
        """Format college recommendations and return both JSON and text response"""
        recommendations = []
        
        # Process database colleges
        for item in filtered_colleges:
            college = item['college']
            json_rec = self.convert_database_college_to_json(
                college, 
                item['score'], 
                item['reasons']
            )
            if json_rec:
                recommendations.append(json_rec)
        
        # Process OpenAI colleges if needed
        if len(recommendations) < 3 and openai_colleges:
            needed = min(5 - len(recommendations), len(openai_colleges))
            for college in openai_colleges[:needed]:
                json_rec = self.convert_openai_college_to_json(college)
                if json_rec:
                    recommendations.append(json_rec)
        
        # Create text response
        if recommendations:
            text_response = f"Based on your preferences, I found {len(recommendations)} colleges that match your criteria:"
            for i, rec in enumerate(recommendations, 1):
                text_response += f"\n\n{i}. {rec['name']} ({rec['type']})"
                text_response += f"\n   Location: {rec['location']}"
                text_response += f"\n   Match Score: {rec['match_score']}%"
                if rec['fees']:
                    text_response += f"\n   Fees: ₹{rec['fees']:,}"
                text_response += f"\n   Courses: {', '.join(rec['courses'])}"
        else:
            text_response = "I couldn't find specific colleges matching your preferences. Please provide more details about your requirements."
            recommendations = []  # Ensure it's always an empty array, never None
            
        return recommendations, text_response
    
    def get_response(self, message: str, chat_id: str) -> Dict[str, Any]:
        """Main processing function that routes between academic and recommendation pipelines"""
        
        timestamp = datetime.now().isoformat()
        
        # Check if query is academic
        is_academic = self.is_academic_query(message)
        
        if not is_academic:
            response = """I'm an academic assistant focused on educational and research-related topics. 
I'd be happy to help you with:
- Subject-specific questions (math, science, literature, etc.)
- Research and study techniques
- Academic writing and analysis
- Learning strategies
- Educational concepts
- College recommendations

Could you please ask me something related to academics or learning?"""
            
            self.db_manager.save_message(chat_id, 'human', message, is_academic, False)
            self.db_manager.save_message(chat_id, 'ai', response, is_academic, False)
            
            return {
                "response": response,
                "is_academic": False,
                "is_recommendation": False,
                "timestamp": timestamp,
                "recommendations": []
            }
        
        # Save user message
        self.db_manager.save_message(chat_id, 'human', message, is_academic, False)
        
        try:
            # Check if asking for college recommendations
            if self.is_asking_for_recommendations(message):
                logger.info("Recommendation request detected - using college recommendation pipeline")
                
                # Extract preferences
                preferences = self.extract_preferences_with_llm(chat_id, message)
                
                # Filter colleges from database
                filtered_colleges = self.college_data_manager.filter_colleges_by_preferences(preferences)
                
                # Get OpenAI recommendations if needed
                openai_colleges = []
                if len(filtered_colleges) < 3:
                    openai_colleges = self.get_openai_college_recommendations(preferences, preferences.location)
                
                # Format recommendations
                recommendations, text_response = self.format_college_recommendations(
                    filtered_colleges, openai_colleges, preferences
                )
                
                is_recommendation = True
                final_response = text_response
            
            else:
                logger.info("Regular academic query - using academic pipeline")
                
                # Use academic conversation chain
                chat_memory = self.chat_memories[chat_id]
                
                # Load previous messages into memory if not already loaded
                if len(chat_memory.chat_memory.messages) == 0:
                    previous_messages = self.db_manager.get_chat_messages(chat_id)
                    for msg in previous_messages[-10:]:
                        if msg['type'] == 'human':
                            chat_memory.chat_memory.add_user_message(msg['content'])
                        elif msg['type'] == 'ai':
                            chat_memory.chat_memory.add_ai_message(msg['content'])
                
                # Get response from academic chain
                final_response = self.academic_chain.invoke({
                    "input": message,
                    "chat_id": chat_id
                })
                
                # Save to memory
                chat_memory.save_context(
                    {"input": message},
                    {"output": final_response}
                )
                
                is_recommendation = False
                recommendations = []
            
            # Save AI response
            self.db_manager.save_message(chat_id, 'ai', final_response, True, is_recommendation)
            
            return {
                "response": final_response,
                "is_academic": True,
                "is_recommendation": is_recommendation,
                "timestamp": timestamp,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            error_response = "I apologize, but I encountered an error while processing your request. Please try again."
            self.db_manager.save_message(chat_id, 'ai', error_response, True, False)
            
            return {
                "response": error_response,
                "is_academic": True,
                "is_recommendation": False,
                "timestamp": timestamp,
                "recommendations": []
            }

# Initialize environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PATH = os.getenv("DB_PATH", "chatbot.db")
EXCEL_PATH = os.getenv("EXCEL_PATH", "colleges.xlsx")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables")
    # You should set this properly in production
    OPENAI_API_KEY = ""

# Initialize the integrated chatbot
try:
    chatbot = IntegratedAcademicChatbot(OPENAI_API_KEY, EXCEL_PATH, DB_PATH)
    logger.info("Integrated chatbot initialized successfully")
except Exception as e:
    logger.error(f"Error initializing chatbot: {e}")
    raise

# FastAPI Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Academic Chatbot API is running!",
        "features": ["Academic Q&A", "College Recommendations"],
        "version": "1.0.0"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, chat_id: str = Query(..., description="Chat ID managed by backend")):
    """Single chat endpoint for both academic and recommendation queries"""
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if not chat_id.strip():
        raise HTTPException(status_code=400, detail="Chat ID cannot be empty")
    
    try:
        result = chatbot.get_response(
            message=request.message,
            chat_id=chat_id
        )
        return ChatResponse(**result)
    
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Test college data
        college_count = len(chatbot.college_data_manager.colleges)
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "Academic Chatbot API",
            "version": "1.0.0",
            "database": "connected",
            "college_data": f"{college_count} colleges loaded",
            "features": {
                "academic_qa": "active",
                "college_recommendations": "active",
                "chat_memory": "active",
                "preference_extraction": "active"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "status_code": 404}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "status_code": 500}

if __name__ == "__main__":
    # Validate required environment variables
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your-openai-api-key-here":
        logger.error("Please set OPENAI_API_KEY environment variable!")
        exit(1)
    
    # Create database if it doesn't exist
    try:
        chatbot.db_manager.init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        exit(1)
    
    # Check if Excel file exists
    if not os.path.exists(EXCEL_PATH):
        logger.warning(f"Excel file not found at {EXCEL_PATH}")
        logger.warning("College recommendations will use OpenAI knowledge only")
    
    logger.info("Starting Academic Chatbot API...")
    logger.info("Features: Academic Q&A + College Recommendations")
    logger.info("Access the API at: http://localhost:8000")
    logger.info("API Documentation: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )