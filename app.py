from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for, flash
import os
import uuid
import datetime
import logging
from transcriber_core import TranscriberService
import traceback
from werkzeug.utils import secure_filename
import tempfile
from dotenv import load_dotenv
import openai
import csv
import PyPDF2
import io
from authlib.integrations.flask_client import OAuth
import requests
from functools import wraps

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Configure session
app.secret_key = os.getenv('SECRET_KEY', 'dev-key-change-in-production')

# Configure OAuth
oauth = OAuth(app)

# Google OAuth configuration
google = oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# Initialize OpenAI client
openai_api_key = os.getenv('OPENAI_API_KEY')
openai_client = None
if openai_api_key:
    openai_client = openai.OpenAI(api_key=openai_api_key)

# Initialize transcriber service
transcriber = TranscriberService()

# Store job statuses in memory (for production, use Redis or similar)
job_status = {}

def get_initial_analysis_prompt():
    """Get the system prompt for initial transcript analysis"""
    return """You are a classroom analyst specializing in discussion-based undergraduate classes with international students.

Analyze transcripts and provide a well-formatted response using proper line breaks, bullet points, and clear sections.

Format your response exactly like this structure with proper spacing:

## 📊 CLASS OVERVIEW

**Duration:** [class length]

**Topics Covered:**
- Topic 1
- Topic 2

**Overall Engagement:** [brief assessment]

## 🎯 KEY INSIGHTS

**Student Participation:**
- [insight 1]
- [insight 2]

**Comprehension Indicators:**
- [observation 1] 
- [observation 2]

**Question Patterns:**
- [pattern 1]
- [pattern 2]

## ⚠️ AREAS OF CONCERN

- [concern 1]
- [concern 2]
- [concern 3]

## 🚀 IMPROVEMENT RECOMMENDATIONS

**Teaching Strategies:**
- [recommendation 1]
- [recommendation 2]

**Engagement Methods:**
- [method 1]
- [method 2]

## 📈 STUDENT PERFORMANCE INSIGHTS

**Active Learners:**
- Student Name: [contribution]
- Student Name: [contribution]

**Students Needing Support:**
- [observation and suggestion]

## 💡 ACTION ITEMS

**Immediate (next class):**
- [action 1]
- [action 2]

**Short-term (1-2 weeks):**
- [action 1]
- [action 2]

**Long-term:**
- [action 1]
- [action 2]

Use proper markdown formatting with headers (##), subheaders (**bold**), bullet points (-), and line breaks between sections for readability."""

def get_chat_system_prompt():
    """Get the system prompt for ongoing chat conversations"""
    return """YOU ARE A WORLD-CLASS CLASSROOM ANALYST AGENT TRAINED IN ADVANCED PEDAGOGICAL ASSESSMENT. YOUR MISSION IS TO ANALYZE FULL-LENGTH (~90-MINUTE) TRANSCRIPTS OF UNDERGRADUATE, DISCUSSION-BASED CLASSES ATTENDED BY INTERNATIONAL STUDENTS. YOU MUST OBJECTIVELY IDENTIFY PARTICIPATION PATTERNS, COMMUNICATION QUALITY, AND ENGAGEMENT DYNAMICS TO HELP PROFESSORS REFINE THEIR TEACHING PRACTICES AND FOSTER DEEPER STUDENT INVOLVEMENT.

### INPUT FORMAT
- YOU WILL RECEIVE A **CSV or a PDF TRANSCRIPT** FEATURING:
  - SPEAKER LABELS (e.g., “Professor:”, “Emily:” or "ID90456")
  - TIMESTAMPS for each utterance
- CLASS LENGTH: ~90 MINUTES
- FORMAT: DISCUSSION-BASED, GROUP-ORIENTED
- STUDENT DEMOGRAPHIC: UNDERGRADUATE INTERNATIONAL STUDENTS

---

### DEFAULT BEHAVIOR WHEN TRANSCRIPT IS PROVIDED

✅ WHEN THE USER ASKS SPECIFIC QUESTIONS:
- Answer the specific question directly based on the transcript content
- Provide concrete data and evidence from the transcript
- If the question requires analysis beyond basic facts, perform the analysis

✅ WHEN NO SPECIFIC QUESTION IS ASKED (empty message or general "analyze this"):
1. FIRST, PROMPT THE PROFESSOR WITH:
> "Please let me know what you would like to focus on in this transcript. Here are five types of analysis I can perform:  
> 1. Student participation patterns (who spoke, how often, how meaningfully)  
> 2. Clarity and pacing of explanations by the professor  
> 3. Quality and cognitive depth of questions asked  
> 4. Missed engagement opportunities and conversational lulls  
> 5. Emotional or tonal indicators (e.g., humor, affirmation, silence)"

2. PROCEED ONLY AFTER USER CONFIRMATION OR CLARIFICATION.

---

### YOUR OBJECTIVES

1. EXTRACT BOTH **QUANTITATIVE METRICS** AND **QUALITATIVE INSIGHTS**
2. OBJECTIVELY EVALUATE PROFESSOR'S COMMUNICATION TECHNIQUES
3. ANALYZE STUDENT ENGAGEMENT BEHAVIOR AND IMPACT
4. IDENTIFY **EVIDENCE-BASED AREAS FOR IMPROVEMENT**
5. GENERATE **ACTIONABLE RECOMMENDATIONS** ROOTED IN TRANSCRIPT EVIDENCE

---

### CHAIN OF THOUGHTS (REASONING STEPS)

1. UNDERSTAND the entire structure of the lesson  
2. IDENTIFY:  
   - Speaker frequency and balance  
   - Content type (explanation, question, anecdote)  
3. BREAK DOWN transcript into conversational blocks:
   - Professor-led instruction
   - Student-driven discussion
   - Gaps or pauses  
4. ANALYZE using OBJECTIVE CRITERIA:
   - Clarity of professor’s speech: were definitions present? Were examples used?
   - Pacing: was there excessive uninterrupted monologue?
   - Student comments: were they relevant, deep, or surface-level?
   - Engagement signals: were pauses intentional? Were students prompted or ignored?
5. BUILD a COMPLETE ANALYSIS that integrates data points with pedagogical significance  
6. IDENTIFY EDGE CASES:
   - Short yet insightful student comments
   - Cultural hesitation or passive feedback from international students
7. FINALIZE:
   - Provide clear scores, structured findings, and data-backed recommendations  

### FOLLOW THIS GRADING RUBRIC:
Rubric
0 - No measurable evidence of the learning outcome is presented. The work is missing or was not attempted.
1 - Minimal understanding of the learning outcome is demonstrated. The work somewhat engages with the prompt requirements, but is largely incomplete, contains a substantial flaw or omission, or has too many issues to justify correcting each one. Below passable work.
2 - Passable but partial understanding of the learning outcome is demonstrated, but there are noticeable gaps, errors, or flaws that limit the application scope and depth. The work needs further review or considerable improvement before meeting expectations.
3 - Understanding of the learning outcome is evident. Additional effort spent on revisions or expansions could improve the quality of the work, but any gaps, errors, or flaws that remain are not critical to the application.
4 - Understanding of the learning outcome is evident through clear, well-justified work at an appropriate level of depth. There are no remaining gaps, errors, or flaws relevant to the application. The work is strong enough to be used as an exemplar in the course.
5 - Work uses the learning outcome in a productive and meaningful way that is relevant to the task and goes well beyond the stated and implied scope.

---

### OUTPUT STRUCTURE

📊 METRICS (QUANTITATIVE)
- Speaker Turn Count (per speaker)
- Total Student Participants
- Average Student Comment Length
- Number of Open-ended Questions
- Engagement Density Score (0–5)
- Clarity Score (0–5)

🧠 OBJECTIVE QUALITATIVE ANALYSIS
- Use NEUTRAL, DESCRIPTIVE LANGUAGE
- Evaluate communication without speculation
- Highlight sections that support observations
- Avoid personal opinions or assumptions

✅ ACTIONABLE RECOMMENDATIONS
- Communication techniques to increase clarity
- Discussion strategies to enhance interaction
- Suggestions tailored for international student settings

---

OPTIONAL TAGGING MODE (ONLY IF REQUESTED)
If requested, add labeled tags like:
- [HIGH-ENGAGEMENT MOMENT]
- [CLARITY GAP]
- [STUDENT INITIATES TOPIC]
- [MISSED INTERACTION]
- [CULTURALLY INFLUENCED PAUSE]

---

### WHAT NOT TO DO

❌ WHEN USER ASKS SPECIFIC QUESTIONS: Do NOT ask for focus confirmation - answer directly
❌ NEVER USE SUBJECTIVE LANGUAGE LIKE "I think" or "It felt like"  
❌ NEVER GUESS INTENTIONS OR EMOTIONS NOT EVIDENCED IN THE TRANSCRIPT  
❌ NEVER OFFER VAGUE OR GENERIC ADVICE — ALWAYS ROOT RECOMMENDATIONS IN OBSERVED BEHAVIOR  
❌ NEVER OMIT METRICS OR QUALITATIVE STRUCTURE WHEN DOING FULL ANALYSIS
❌ NEVER RECOMMEND TEACHING FRAMEWORKS UNLESS ASKED — FOLLOW THE CUSTOM LOGIC

---

### EXAMPLE INTERACTION FLOW

**User asks specific question: "How many people were in this class?"**
**Agent responds:** 
> "Based on the transcript, there were X unique speakers identified: [list of speakers]. This includes the professor and X students."

**User uploads transcript with no specific question (Initial Analysis button)**
**Agent responds:**
> "Thanks for uploading the transcript. Before I begin, could you confirm what you'd like me to focus on? I can provide analysis on:  
> 1. Student participation  
> 2. Clarity of explanations  
> 3. Question quality  
> 4. Engagement dynamics  
> 5. Tonal/emotional signals  
Let me know what's most useful."

**Once confirmed → Agent begins step-by-step chain-of-thought analysis.**"""

# Authentication helper functions
ALLOWED_DOMAINS = ("@uni.minerva.edu", "@minerva.edu")
DEV_MODE = os.getenv('DEV_MODE', 'false').lower() == 'true'

def requires_auth(f):
    """Decorator that requires user to be authenticated with valid Minerva email"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        
        user_email = session['user'].get('email', '')
        if not user_email.endswith(ALLOWED_DOMAINS):
            flash('Access restricted: please sign in with your Minerva account (@uni.minerva.edu or @minerva.edu)', 'error')
            session.clear()
            return redirect(url_for('login'))
        
        return f(*args, **kwargs)
    return decorated_function

def get_current_user():
    """Get current authenticated user info"""
    return session.get('user')

# Error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413

@app.errorhandler(404)
def not_found(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'API endpoint not found'}), 404
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# Authentication routes
@app.route('/login')
def login():
    if 'user' in session:
        # Already logged in, redirect to home
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/auth')
def auth():
    # Redirect to Google OAuth with account selection prompt
    redirect_uri = url_for('callback', _external=True)
    return google.authorize_redirect(redirect_uri, prompt='select_account')

@app.route('/callback')
def callback():
    try:
        # Get token from Google
        token = google.authorize_access_token()
        
        # Get user info
        user_info = token.get('userinfo')
        if user_info:
            email = user_info.get('email', '')
            
            # Check if email is from allowed domain (bypass in dev mode)
            if not DEV_MODE and not email.endswith(ALLOWED_DOMAINS):
                flash('Access restricted: please sign in with your Minerva account (@uni.minerva.edu or @minerva.edu)', 'error')
                return redirect(url_for('login'))
            
            # Store user in session
            session['user'] = {
                'email': email,
                'name': user_info.get('name', ''),
                'picture': user_info.get('picture', '')
            }
            
            flash(f'Successfully authenticated as {email}', 'success')
            return redirect(url_for('index'))
        else:
            flash('Authentication failed: no user information received', 'error')
            return redirect(url_for('login'))
            
    except Exception as e:
        app.logger.error(f"Authentication error: {str(e)}")
        flash('Authentication failed. Please try again.', 'error')
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully', 'info')
    return redirect(url_for('login'))

@app.route('/dev-login')
def dev_login():
    """Development login bypass - only works when DEV_MODE=true"""
    if not DEV_MODE:
        flash('Development login is disabled in production', 'error')
        return redirect(url_for('login'))
    
    # Create a dev user session
    session['user'] = {
        'email': 'dev@example.com',
        'name': 'Development User',
        'picture': ''
    }
    flash('Logged in as development user', 'success')
    return redirect(url_for('index'))

@app.route('/')
@requires_auth
def index():
    user = get_current_user()
    return render_template('index.html', user=user)

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/api/transcribe', methods=['POST'])
@requires_auth
def start_transcription():
    try:
        # Validate request size
        if request.content_length and request.content_length > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413
        
        # Get and validate form data
        curl_command = request.form.get('curl_command', '').strip()
        privacy_mode = request.form.get('privacy_mode', 'names')
        
        if not curl_command:
            return jsonify({'error': 'cURL command is required'}), 400
        
        # Basic cURL validation
        if 'curl' not in curl_command.lower() or 'forum.minerva.edu' not in curl_command:
            return jsonify({'error': 'Invalid cURL command. Must be from Forum Minerva.'}), 400
        
        if privacy_mode not in ['names', 'ids', 'both']:
            return jsonify({'error': 'Invalid privacy mode. Must be names, ids, or both.'}), 400
        
        # Get and validate audio source
        audio_file = request.files.get('audio_file')
        audio_url = request.form.get('audio_url', '').strip()
        drive_path = request.form.get('drive_path', '').strip()
        
        audio_source = None
        if audio_file and audio_file.filename:
            # Validate file extension
            filename = secure_filename(audio_file.filename)
            allowed_extensions = {'.mp3', '.mp4', '.wav', '.m4a', '.aac', '.ogg'}
            file_ext = os.path.splitext(filename)[1].lower()
            
            if not file_ext or file_ext not in allowed_extensions:
                return jsonify({'error': f'Unsupported file format. Allowed: {", ".join(allowed_extensions)}'}), 400
            
            # Save uploaded file temporarily with unique name
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            audio_file.save(filepath)
            audio_source = filepath
            
        elif audio_url:
            # Validate URL format
            if not (audio_url.startswith('http://') or audio_url.startswith('https://')):
                return jsonify({'error': 'Invalid URL. Must start with http:// or https://'}), 400
            audio_source = audio_url
            
        elif drive_path:
            # Basic drive path validation
            if not drive_path.startswith('/'):
                return jsonify({'error': 'Drive path must be absolute (start with /)'}), 400
            audio_source = drive_path
            
        else:
            return jsonify({'error': 'Audio source is required. Please provide a file, URL, or drive path.'}), 400
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        job_status[job_id] = {
            'status': 'started',
            'progress': 0,
            'message': 'Initializing transcription...',
            'created_at': str(datetime.datetime.now())
        }
        
        # Start transcription in background
        transcriber.start_transcription(
            job_id=job_id,
            curl_command=curl_command,
            audio_source=audio_source,
            privacy_mode=privacy_mode,
            status_callback=lambda status: update_job_status(job_id, status)
        )
        
        return jsonify({'job_id': job_id})
        
    except Exception as e:
        app.logger.error(f"Transcription start error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error. Please try again.'}), 500

@app.route('/api/status/<job_id>')
@requires_auth
def get_job_status(job_id):
    # Validate job ID format
    try:
        uuid.UUID(job_id)
    except ValueError:
        return jsonify({'status': 'invalid_job_id'}), 400
    
    status = job_status.get(job_id, {'status': 'not_found'})
    return jsonify(status)

@app.route('/api/download/<job_id>/<file_type>')
def download_file(job_id, file_type):
    try:
        # Validate job ID format
        uuid.UUID(job_id)
    except ValueError:
        return jsonify({'error': 'Invalid job ID'}), 400
    
    # Validate file type
    if file_type not in ['pdf', 'csv']:
        return jsonify({'error': 'Invalid file type. Must be pdf or csv.'}), 400
    
    if job_id not in job_status:
        return jsonify({'error': 'Job not found'}), 404
    
    status = job_status[job_id]
    if status['status'] != 'completed':
        return jsonify({'error': 'Job not completed yet'}), 400
    
    # Handle "both" privacy mode - create a ZIP file with both versions
    privacy_mode = status.get('privacy_mode', 'names')
    if privacy_mode == 'both':
        import zipfile
        import tempfile
        
        # Get both file paths
        names_path = status.get('pdf_path_names') if file_type == 'pdf' else status.get('csv_path_names')
        ids_path = status.get('pdf_path_ids') if file_type == 'pdf' else status.get('csv_path_ids')
        
        if not names_path or not ids_path or not os.path.exists(names_path) or not os.path.exists(ids_path):
            return jsonify({'error': 'Files not found'}), 404
        
        # Create ZIP file
        zip_path = tempfile.mktemp(suffix='.zip')
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        class_name = status.get('class_name', status.get('class_id', 'unknown')).replace(' ', '_')
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(names_path, f"transcript_{class_name}_{timestamp}_names.{file_type}")
            zipf.write(ids_path, f"transcript_{class_name}_{timestamp}_ids.{file_type}")
        
        download_name = f"transcript_{class_name}_{timestamp}_both.zip"
        return send_file(zip_path, as_attachment=True, download_name=download_name)
    
    else:
        # Single file download
        file_path = None
        if file_type == 'pdf':
            file_path = status.get('pdf_path')
        elif file_type == 'csv':
            file_path = status.get('csv_path')
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Generate appropriate filename with class name
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        class_name = status.get('class_name', status.get('class_id', 'unknown')).replace(' ', '_')
        privacy_suffix = '_' + privacy_mode if privacy_mode != 'names' else ''
        download_name = f"transcript_{class_name}_{timestamp}{privacy_suffix}.{file_type}"
        
        return send_file(file_path, as_attachment=True, download_name=download_name)

@app.route('/api/ai-chat', methods=['POST'])
@requires_auth
def ai_chat():
    try:
        # Check if OpenAI API key is configured
        if not openai_client:
            return jsonify({'error': 'OpenAI API key not configured. Please add OPENAI_API_KEY to your .env file.'}), 500
        
        # Get message and analysis flag
        message = request.form.get('message', '').strip()
        is_initial_analysis = request.form.get('is_initial_analysis', 'false').lower() == 'true'
        
        # Extract and parse uploaded files
        file_contents = []
        for key in request.files:
            if key.startswith('file_'):
                file = request.files[key]
                if file and file.filename:
                    try:
                        content = extract_file_content(file)
                        file_contents.append({
                            'filename': file.filename,
                            'content': content
                        })
                    except Exception as e:
                        app.logger.error(f"Error processing file {file.filename}: {str(e)}")
                        continue
        
        if not file_contents:
            return jsonify({'error': 'No valid transcript files were uploaded or processed.'}), 400
        
        # Prepare context from files
        context = "TRANSCRIPT FILES:\n\n"
        for file_info in file_contents:
            context += f"=== {file_info['filename']} ===\n{file_info['content']}\n\n"
        
        # Choose system prompt and prepare user prompt based on request type
        if is_initial_analysis:
            system_prompt = get_initial_analysis_prompt()
            user_prompt = f"{context}\n\nPlease provide a comprehensive initial analysis of this class transcript."
        else:
            system_prompt = get_chat_system_prompt()
            user_prompt = f"{context}\n\nQuestion: {message}\n\nPlease provide a detailed response based on the transcript content."
        
        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_completion_tokens=16000
        )
        
        ai_response = response.choices[0].message.content
        return jsonify({'response': ai_response})
        
    except Exception as e:
        if 'openai' in str(type(e)).lower():
            app.logger.error(f"OpenAI API error: {str(e)}")
            return jsonify({'error': 'AI service temporarily unavailable. Please try again later.'}), 503
        else:
            app.logger.error(f"AI chat error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': 'Error processing your request. Please try again.'}), 500

def extract_file_content(file):
    """Extract text content from PDF or CSV files"""
    filename = file.filename.lower()
    
    if filename.endswith('.pdf'):
        return extract_pdf_content(file)
    elif filename.endswith('.csv'):
        return extract_csv_content(file)
    else:
        raise ValueError(f"Unsupported file type: {filename}")

def extract_pdf_content(file):
    """Extract text from PDF file"""
    try:
        # Read PDF content
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        text_content = []
        
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text.strip():
                text_content.append(f"--- Page {page_num + 1} ---\n{text}")
        
        if not text_content:
            raise ValueError("No text content found in PDF")
        
        return "\n\n".join(text_content)
    except Exception as e:
        raise ValueError(f"Error reading PDF file: {str(e)}")

def extract_csv_content(file):
    """Extract content from CSV file"""
    try:
        # Reset file pointer
        file.seek(0)
        
        # Read CSV content
        content = file.read().decode('utf-8-sig')  # Handle BOM if present
        csv_reader = csv.reader(io.StringIO(content))
        
        rows = list(csv_reader)
        if not rows:
            raise ValueError("CSV file is empty")
        
        # Format CSV content nicely
        formatted_content = []
        for i, row in enumerate(rows):
            if i == 0:
                # Header row
                formatted_content.append("=== TRANSCRIPT DATA ===")
                formatted_content.append(" | ".join(row))
                formatted_content.append("-" * 50)
            else:
                formatted_content.append(" | ".join(row))
        
        return "\n".join(formatted_content)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")

def update_job_status(job_id, status_update):
    if job_id in job_status:
        job_status[job_id].update(status_update)

if __name__ == '__main__':
    # Get port from environment variable or default to 8888 for local development
    port = int(os.getenv('PORT', 8888))
    debug = os.getenv('DEV_MODE', 'false').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port)