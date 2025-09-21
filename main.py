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
    type: str
    courses_offered: str
    website: str
    admission_process: str
    approximate_fees: str
    notable_features: str
    source: str

class ChatResponse(BaseModel):
    response: str
    is_academic: bool
    is_recommendation: bool
    timestamp: str
    recommendations: Optional[List[CollegeRecommendation]] = []
    chat_title: Optional[str] = None

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
    
    def is_greeting_or_casual_question(self, query: str) -> bool:
        """Check if query is a greeting or casual conversational question"""
        greeting_patterns = [
            # Basic greetings
            'hi', 'hello', 'hey', 'hii', 'hiii', 'hiiii', 'helo', 'hellooo',
            # Time-based greetings
            'good morning', 'good afternoon', 'good evening', 'good night',
            # Casual greetings
            'greetings', 'howdy', 'what\'s up', 'whats up', 'sup', 'yo',
            # Conversational questions
            'how are you', 'how do you do', 'how are you doing', 'how\'s it going', 
            'hows it going', 'how is your day', 'how\'s your day', 'hows your day',
            'how was your day', 'how\'s your day going', 'hows your day going',
            'what are you doing', 'what\'s happening', 'whats happening',
            'how are things', 'how are you feeling', 'are you okay', 'are you ok',
            'nice to meet you', 'pleased to meet you', 'good to see you',
            # Status questions
            'how are you today', 'how have you been', 'how\'s everything',
            'hows everything', 'what\'s new', 'whats new', 'how\'s life',
            'hows life', 'all good', 'you good', 'you ok'
        ]
        
        query_lower = query.lower().strip()
        
        # Direct matches
        if query_lower in greeting_patterns:
            return True
            
        # Check for patterns that start the message
        for pattern in greeting_patterns:
            if query_lower.startswith(pattern + ' ') or query_lower.startswith(pattern + '?'):
                return True
        
        # Short informal greetings (1-3 words)
        words = query_lower.split()
        if len(words) <= 3:
            greeting_words = ['hi', 'hey', 'hello', 'sup', 'yo', 'morning', 'evening', 'afternoon']
            if any(word in greeting_words for word in words):
                return True
                
        return False
    
    def generate_conversational_response(self, query: str, chat_id: str) -> str:
        """Generate friendly, conversational responses to greetings and casual questions"""
        import random
        
        query_lower = query.lower().strip()
        previous_messages = self.db_manager.get_chat_messages(chat_id)
        is_returning_user = len(previous_messages) > 0
        
        # Handle "how are you" type questions
        if any(phrase in query_lower for phrase in ['how are you', 'how do you do', 'how are you doing', 'how\'s it going', 'hows it going']):
            responses = [
                "I'm doing great, thank you for asking! I'm energized and ready to help you with any academic questions or college guidance you need. How can I assist you today?",
                "I'm fantastic! I love helping students with their academic journey. Whether you need study tips, subject explanations, or college recommendations, I'm here for you. What's on your mind?",
                "I'm wonderful, thanks! I'm having a good day helping students learn and grow. I'm excited to help you too - what academic topic interests you today?",
                "I'm doing excellent! Every day is a great day when I get to help with education and learning. How about you? What brings you here today?",
                "I'm thriving! I really enjoy our academic conversations and helping students succeed. What would you like to explore or learn about today?"
            ]
        
        # Handle "how's your day" type questions
        elif any(phrase in query_lower for phrase in ['how is your day', 'how\'s your day', 'hows your day', 'how was your day']):
            responses = [
                "My day has been wonderful! I've been helping students with everything from calculus problems to college applications. It's been really fulfilling! How's your day going?",
                "It's been an amazing day! I've had great conversations about science, literature, study strategies, and college planning. I love what I do! What about your day?",
                "My day has been fantastic! I've been busy helping students tackle challenging subjects and find the right colleges for their goals. How has your day been?",
                "It's been a great day filled with interesting academic discussions! From physics questions to MBA program recommendations, every conversation teaches me something new. How are you doing today?"
            ]
        
        # Handle "what are you doing" type questions
        elif any(phrase in query_lower for phrase in ['what are you doing', 'what\'s happening', 'whats happening', 'what\'s up', 'whats up', 'sup']):
            responses = [
                "Just here ready to help with academic questions! I love discussing everything from complex math problems to college admissions strategies. What's up with you?",
                "I'm here waiting to dive into some great academic discussions! Whether you need help with homework, study techniques, or college planning, I'm all ears. What's going on?",
                "Nothing much, just excited to help students learn and grow! I'm ready to tackle any subject or help with college recommendations. What brings you here today?",
                "I'm here doing what I love most - helping with education! From science concepts to college guidance, I'm ready for any academic challenge. What's happening with you?"
            ]
        
        # Handle basic greetings
        else:
            if is_returning_user:
                responses = [
                    "Hey there! Welcome back! It's great to see you again. I'm excited to continue our academic journey together - what shall we explore today?",
                    "Hello! So good to have you back! I'm ready to help with any new questions, whether they're about studies, research, or college planning. What's on your agenda?",
                    "Hi! Welcome back! I've been looking forward to our next conversation. Ready to tackle some interesting academic topics together?",
                    "Hey! Great to see you again! I'm here and eager to help with whatever academic challenge you're facing today. What can we work on together?",
                    "Hello again! I'm so glad you're back! Whether you need help with subjects, study strategies, or college advice, I'm ready to dive in. What's your focus today?"
                ]
            else:
                responses = [
                    "Hello! It's wonderful to meet you! I'm your academic companion, and I'm genuinely excited to help you with your educational journey. What academic topic can we explore together today?",
                    "Hi there! Welcome! I'm thrilled you're here! I specialize in making learning engaging and helping with everything from subject questions to college guidance. What interests you most?",
                    "Hey! So nice to meet you! I'm passionate about education and love helping students succeed. Whether you need study help, subject explanations, or college advice, I'm here for you. What shall we start with?",
                    "Hello and welcome! I'm really happy you're here! I'm your academic assistant who genuinely cares about your success. From homework help to college planning, I'm ready to support you. What's your biggest academic interest right now?",
                    "Hi! It's great to have you here! I'm your friendly academic guide, ready to make learning fun and help you achieve your educational goals. What would you like to discover together today?"
                ]
        
        # Add time-based greeting occasionally
        import datetime
        current_hour = datetime.datetime.now().hour
        
        if random.choice([True, False]):  # 50% chance
            if 5 <= current_hour < 12:
                time_greeting = "Good morning! "
            elif 12 <= current_hour < 17:
                time_greeting = "Good afternoon! "
            elif 17 <= current_hour < 21:
                time_greeting = "Good evening! "
            else:
                time_greeting = ""
            
            if time_greeting:
                selected_response = time_greeting + random.choice(responses)
            else:
                selected_response = random.choice(responses)
        else:
            selected_response = random.choice(responses)
        
        return selected_response
    
    def generate_chat_title(self, chat_id: str) -> str:
        """Generate a meaningful title based on the main conversation topics"""
        try:
            messages = self.db_manager.get_chat_messages(chat_id)
            
            # Filter out greetings and get substantial messages
            substantial_messages = []
            for msg in messages:
                if msg['type'] == 'human' and len(msg['content'].split()) > 3:
                    # Skip obvious greetings
                    content_lower = msg['content'].lower()
                    greeting_indicators = ['hi', 'hello', 'hey', 'how are you', 'good morning', 'good evening']
                    if not any(indicator in content_lower for indicator in greeting_indicators):
                        substantial_messages.append(msg['content'])
            
            if not substantial_messages:
                return "New Conversation"
            
            # Use OpenAI to generate a concise title based on the conversation
            conversation_text = " | ".join(substantial_messages[:5])  # Use first 5 substantial messages
            
            prompt = f"""
            Generate a short, descriptive title (3-6 words) for this academic conversation based on the main topics discussed:
            
            Conversation snippets: {conversation_text}
            
            The title should reflect the main academic subject or topic. Examples:
            - "Mathematics Problem Solving"
            - "College Engineering Programs"
            - "Biology Study Strategies"
            - "MBA Application Guidance"
            - "Chemistry Homework Help"
            - "Computer Science Career"
            
            Return only the title, nothing else.
            """
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=50
            )
            
            title = response.choices[0].message.content.strip().replace('"', '')
            
            # Fallback titles based on keywords if OpenAI fails
            if not title or len(title) < 3:
                conversation_lower = conversation_text.lower()
                if any(word in conversation_lower for word in ['college', 'university', 'admission', 'recommend']):
                    return "College Guidance"
                elif any(word in conversation_lower for word in ['math', 'calculus', 'algebra', 'geometry']):
                    return "Mathematics Help"
                elif any(word in conversation_lower for word in ['science', 'physics', 'chemistry', 'biology']):
                    return "Science Discussion"
                elif any(word in conversation_lower for word in ['study', 'exam', 'preparation', 'tips']):
                    return "Study Strategies"
                elif any(word in conversation_lower for word in ['career', 'job', 'profession']):
                    return "Career Planning"
                else:
                    return "Academic Discussion"
            
            return title
            
        except Exception as e:
            logger.error(f"Error generating chat title: {e}")
            return "Academic Chat"
    
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
                    "type": "Government/Private/Deemed",
                    "courses_offered": "Main courses or programs offered",
                    "website": "Official website URL if known",
                    "admission_process": "Brief admission process description",
                    "approximate_fees": "Fee range with currency",
                    "notable_features": "Key highlights or notable features",
                    "source": "openai_knowledge"
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
            courses_offered = "Various Programs"
            if college.courses:
                courses_text = college.courses.lower()
                course_list = []
                if 'btech' in courses_text or 'b.tech' in courses_text:
                    course_list.append('B.Tech')
                if 'mtech' in courses_text or 'm.tech' in courses_text:
                    course_list.append('M.Tech')
                if 'mba' in courses_text:
                    course_list.append('MBA')
                if 'mbbs' in courses_text:
                    course_list.append('MBBS')
                if 'bca' in courses_text:
                    course_list.append('BCA')
                if 'mca' in courses_text:
                    course_list.append('MCA')
                if course_list:
                    courses_offered = ", ".join(course_list)
            
            # Extract fees if available in courses string and format as string
            approximate_fees = "Fee information not available"
            if college.courses:
                import re
                fee_match = re.search(r'(\d+(?:,\d{3})*(?:\.\d{2})?)', college.courses)
                if fee_match:
                    try:
                        fee_amount = int(fee_match.group(1).replace(',', ''))
                        approximate_fees = f"INR {fee_amount:,} per year"
                    except:
                        approximate_fees = "Fee information not available"
            
            # Generate notable features from match reasons and college data
            notable_features_list = []
            if match_reasons:
                notable_features_list.extend(match_reasons[:2])
            if college.scholarship and college.scholarship.lower() != 'nan' and 'scholarship' not in str(notable_features_list).lower():
                notable_features_list.append("Scholarship Available")
            if college.type.lower() == 'government':
                notable_features_list.append("Government Institution")
            elif college.type.lower() == 'private':
                notable_features_list.append("Private Institution")
            
            # Join notable features into a single string
            notable_features = ". ".join(notable_features_list[:3]) if notable_features_list else "Quality education institution"
            
            # Format admission process
            admission_process = college.admission_process if college.admission_process and college.admission_process.lower() != 'nan' else "Check official website for admission details"
            
            # Format website
            website = college.website if college.website and college.website.lower() != 'nan' else "Website information not available"
            
            return {
                "id": college.college_id,
                "name": college.name,
                "location": college.location,
                "type": college.type,
                "courses_offered": courses_offered,
                "website": website,
                "admission_process": admission_process,
                "approximate_fees": approximate_fees,
                "notable_features": notable_features,
                "source": website if website != "Website information not available" else "database"
            }
            
        except Exception as e:
            logger.error(f"Error converting database college to JSON: {e}")
            return None
    
    def convert_openai_college_to_json(self, college_data: Dict, match_score: int = 75) -> Dict:
        """Convert OpenAI college recommendation to standardized JSON format"""
        try:
            return {
                "id": college_data.get('id', str(uuid.uuid4())),
                "name": college_data.get('name', ''),
                "location": college_data.get('location', ''),
                "type": college_data.get('type', ''),
                "courses_offered": college_data.get('courses_offered', ''),
                "website": college_data.get('website', ''),
                "admission_process": college_data.get('admission_process', ''),
                "approximate_fees": college_data.get('approximate_fees', ''),
                "notable_features": college_data.get('notable_features', ''),
                "source": college_data.get('source', 'openai_knowledge')
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
                text_response += f"\n   Courses: {rec['courses_offered']}"
                text_response += f"\n   Fees: {rec['approximate_fees']}"
                text_response += f"\n   Features: {rec['notable_features']}"
        else:
            text_response = "I couldn't find specific colleges matching your preferences. Please provide more details about your requirements."
            recommendations = []  # Ensure it's always an empty array, never None
            
        return recommendations, text_response
    
    def get_response(self, message: str, chat_id: str) -> Dict[str, Any]:
        """Main processing function that routes between academic and recommendation pipelines"""
        
        timestamp = datetime.now().isoformat()
        
        # Check if it's a greeting or casual conversational question first
        if self.is_greeting_or_casual_question(message):
            response = self.generate_conversational_response(message, chat_id)
            
            self.db_manager.save_message(chat_id, 'human', message, True, False)
            self.db_manager.save_message(chat_id, 'ai', response, True, False)
            
            # Generate title after a few meaningful exchanges
            chat_title = self.generate_chat_title(chat_id)
            
            return {
                "response": response,
                "is_academic": True,  # Treat greetings as academic-friendly
                "is_recommendation": False,
                "timestamp": timestamp,
                "recommendations": [],
                "chat_title": chat_title
            }
        
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
            
            chat_title = self.generate_chat_title(chat_id)
            
            return {
                "response": response,
                "is_academic": False,
                "is_recommendation": False,
                "timestamp": timestamp,
                "recommendations": [],
                "chat_title": chat_title
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
            
            # Generate title based on the conversation content
            chat_title = self.generate_chat_title(chat_id)
            
            return {
                "response": final_response,
                "is_academic": True,
                "is_recommendation": is_recommendation,
                "timestamp": timestamp,
                "recommendations": recommendations,
                "chat_title": chat_title
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            error_response = "I apologize, but I encountered an error while processing your request. Please try again."
            self.db_manager.save_message(chat_id, 'ai', error_response, True, False)
            
            chat_title = self.generate_chat_title(chat_id)
            
            return {
                "response": error_response,
                "is_academic": True,
                "is_recommendation": False,
                "timestamp": timestamp,
                "recommendations": [],
                "chat_title": chat_title
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