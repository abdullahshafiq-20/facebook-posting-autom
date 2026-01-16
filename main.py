import yt_dlp
from google import genai
from google.oauth2.service_account import Credentials
import gspread
import requests
import os
import json
import time
from pathlib import Path
from datetime import datetime
import logging
import sys
from typing import Optional, List, Dict
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import secrets
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging with UTF-8 encoding to handle emojis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('facebook_automation.log', encoding='utf-8'),
        logging.StreamHandler(stream=sys.stdout)
    ]
)

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

logger = logging.getLogger(__name__)

# FastAPI app setup
app = FastAPI(title="Facebook Reel Automation API")
security = HTTPBasic()

# Load configuration from environment variables
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "admin123")
PAGE_ID = os.getenv("FACEBOOK_PAGE_ID")
ACCESS_TOKEN = os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SPREADSHEET_NAME = os.getenv("SPREADSHEET_NAME")
# Validate required environment variables
def validate_env_vars():
    """Validate that all required environment variables are set"""
    required_vars = {
        'FACEBOOK_PAGE_ID': PAGE_ID,
        'FACEBOOK_PAGE_ACCESS_TOKEN': ACCESS_TOKEN,
        'GEMINI_API_KEY': GEMINI_API_KEY,
        'GOOGLE_SERVICE_ACCOUNT': os.getenv('GOOGLE_SERVICE_ACCOUNT'),
        'SPREADSHEET_NAME': SPREADSHEET_NAME
    }
    
    missing = [name for name, value in required_vars.items() if not value]
    
    if missing:
        error_msg = f"Missing required environment variables: {', '.join(missing)}"
        logger.error(error_msg)
        logger.error("Please set these in your .env file or environment")
        raise ValueError(error_msg)
    
    logger.info("✓ All required environment variables are set")

# Validate on module load
try:
    validate_env_vars()
except ValueError as e:
    logger.warning(f"Environment validation failed: {e}")
    logger.warning("Some endpoints may not work until environment variables are set")

# Google Sheets setup
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]


class ProcessResult(BaseModel):
    status: str
    message: str
    row_number: Optional[int] = None
    instagram_link: Optional[str] = None
    post_url: Optional[str] = None
    error_reason: Optional[str] = None
    started_at: str
    completed_at: str
    duration_seconds: float
    pending_count: int = 0


class AddVideoRequest(BaseModel):
    instagram_url: str


class AddVideoResponse(BaseModel):
    success: bool
    message: str
    row_number: int
    instagram_url: str
    status: str


class FacebookPoster:
    def __init__(self):
        
        # Validate environment variables are loaded
        if not all([PAGE_ID, ACCESS_TOKEN, GEMINI_API_KEY]):
            raise ValueError("Required environment variables are not set. Check your .env file.")
        
        # Initialize Gemini with new API
        logger.info("Initializing Gemini AI client...")
        self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        self.gemini_model = 'gemini-2.5-flash'  # Using latest model
        logger.info(f"Gemini model: {self.gemini_model}")
        
        # Create temp directory for downloads
        self.temp_dir = Path('temp_downloads')
        self.temp_dir.mkdir(exist_ok=True)
        
        # Configure yt-dlp options
        self.ydl_opts = {
            'format': 'best',
            'outtmpl': str(self.temp_dir / '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'writeinfojson': False,
        }
        
        # Initialize Google Sheets
        self.sheet = self.init_google_sheets()
    
    def download_instagram_video(self, url):
        """Download Instagram video and extract caption using yt-dlp"""
        try:
            logger.info(f"[STEP 1] Starting video download from: {url}")
            
            # Use yt-dlp to get video info and download
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                logger.info("Extracting video info...")
                # Extract info first
                info = ydl.extract_info(url, download=False)
                
                # Get caption/description
                original_caption = info.get('description', '')
                if not original_caption:
                    # Try alternative fields
                    original_caption = info.get('title', '')
                
                logger.info(f"Caption extracted (length: {len(original_caption)} chars)")
                
                # Check if it's a video
                if info.get('_type') == 'playlist':
                    logger.error("URL is a playlist, not a single video")
                    return None, None, "URL is a playlist, not a single video"
                
                logger.info("Downloading video file...")
                # Download the video
                ydl.download([url])
                
                # Find the downloaded video file
                video_id = info.get('id', '')
                video_ext = info.get('ext', 'mp4')
                video_path = self.temp_dir / f"{video_id}.{video_ext}"
                
                if not video_path.exists():
                    # Try to find any video file in temp directory
                    video_files = list(self.temp_dir.glob('*.mp4')) + list(self.temp_dir.glob('*.webm'))
                    if video_files:
                        video_path = video_files[-1]  # Get the most recent one
                    else:
                        logger.error("Video file not found after download")
                        return None, None, "Video file not found after download"
                
                file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                logger.info(f"✓ Video downloaded successfully: {video_path.name} ({file_size_mb:.2f} MB)")
                return str(video_path), original_caption, None
            
        except Exception as e:
            error_msg = f"Error downloading video with yt-dlp: {str(e)}"
            logger.error(error_msg)
            return None, None, error_msg
    
    def generate_caption_with_gemini(self, original_caption):
        """Generate new caption and hashtags using Gemini AI"""
        try:
            logger.info("[STEP 2] Generating new caption with Gemini AI...")
            
            prompt = f"""Original caption: {original_caption}

Create a short, fresh Facebook caption inspired by the original caption.
- You MAY slightly vary or rephrase the original caption if needed
- Understand the context and add light humor or sarcasm when appropriate
- Use a funny or expressive emoji if it fits (💀 preferred, optional)
- Keep it concise, natural, and engaging for a Facebook audience
- Add 5-10 relevant hashtags at the end
- ALWAYS include this line at the end:
(All credits belong to their respective owners.)

Return ONLY the new caption with hashtags. No explanations."""
            
            logger.info(f"Sending request to Gemini model: {self.gemini_model}")
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt
            )
            
            new_caption = response.text.strip()
            logger.info(f"✓ Caption generated successfully (length: {len(new_caption)} chars)")
            logger.info(f"Preview: {new_caption[:80]}...")
            
            return new_caption, None
            
        except Exception as e:
            error_msg = f"Error generating caption with Gemini: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    def upload_video_to_facebook(self, video_path, caption):
        """Upload video to Facebook Page as Reel using proper resumable upload"""
        try:
            logger.info("[STEP 3] Uploading video to Facebook Page as Reel...")
            
            file_size = os.path.getsize(video_path)
            file_size_mb = file_size / (1024 * 1024)
            logger.info(f"File size: {file_size_mb:.2f} MB")
            logger.info(f"Facebook Page ID: {PAGE_ID}")
            
            # Step 1: Initialize resumable upload session
            init_url = f"https://graph.facebook.com/v19.0/{PAGE_ID}/video_reels"
            
            init_data = {
                'upload_phase': 'start',
                'access_token': ACCESS_TOKEN,
            }
            
            logger.info("Step 1: Initializing upload session...")
            init_response = requests.post(init_url, data=init_data)
            init_result = init_response.json()
            
            if 'error' in init_result:
                error_msg = f"Facebook API Error (init): {init_result['error']['message']}"
                logger.error(error_msg)
                logger.error(f"Init response: {init_result}")
                return None, error_msg
            
            video_id = init_result.get('video_id')
            upload_url = init_result.get('upload_url')
            
            if not video_id or not upload_url:
                error_msg = f"Missing video_id or upload_url in init response: {init_result}"
                logger.error(error_msg)
                return None, error_msg
            
            logger.info(f"✓ Session initialized")
            logger.info(f"  Video ID: {video_id}")
            logger.info(f"  Upload URL: {upload_url}")
            
            # Step 2: Upload video file to the provided upload_url
            logger.info("Step 2: Uploading video file...")
            
            with open(video_path, 'rb') as video_file:
                video_data = video_file.read()
                
                # Upload to Facebook's upload URL
                upload_headers = {
                    'Authorization': f'OAuth {ACCESS_TOKEN}',
                    'offset': '0',
                    'file_size': str(file_size)
                }
                
                upload_response = requests.post(upload_url, data=video_data, headers=upload_headers)
                
                if upload_response.status_code not in [200, 201]:
                    error_msg = f"Video upload failed: {upload_response.status_code} - {upload_response.text}"
                    logger.error(error_msg)
                    return None, error_msg
                
                upload_result = upload_response.json() if upload_response.text else {}
                logger.info(f"Upload response: {upload_result}")
            
            logger.info("✓ Video file uploaded successfully")
            
            # Step 3: Finalize the upload with video_id and description
            logger.info("Step 3: Finalizing upload with description...")
            
            finalize_data = {
                'upload_phase': 'finish',
                'video_id': video_id,  # CRITICAL: Include video_id here
                'access_token': ACCESS_TOKEN,
                'description': caption,
                'video_state': 'PUBLISHED'  # Publish immediately
            }
            
            finalize_response = requests.post(init_url, data=finalize_data)
            finalize_result = finalize_response.json()
            
            if 'error' in finalize_result:
                error_msg = f"Facebook API Error (finalize): {finalize_result['error']['message']}"
                logger.error(error_msg)
                logger.error(f"Finalize response: {finalize_result}")
                return None, error_msg
            
            # Get final status
            success = finalize_result.get('success', False)
            
            logger.info(f"✓ Upload finalized")
            logger.info(f"  Video ID: {video_id}")
            logger.info(f"  Success: {success}")
            logger.info(f"  Finalize response: {finalize_result}")
            
            # Construct post URL
            post_url = f"https://facebook.com/{PAGE_ID}/videos/{video_id}"
            logger.info(f"✓✓✓ Reel posted successfully to Facebook!")
            logger.info(f"Post URL: {post_url}")
            
            return post_url, None
            
        except Exception as e:
            error_msg = f"Error uploading to Facebook: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, error_msg
    
    def cleanup_temp_files(self, video_path):
        """Clean up temporary downloaded files"""
        try:
            if video_path and os.path.exists(video_path):
                # Remove the video file
                os.remove(video_path)
                logger.info(f"[CLEANUP] Removed temporary file: {Path(video_path).name}")
        except Exception as e:
            logger.warning(f"[CLEANUP] Error cleaning up temp files: {str(e)}")
    
    def add_video_to_sheet(self, instagram_url):
        """Add a new video URL to Google Sheet with status='pending'"""
        try:
            logger.info(f"Adding video to Google Sheet: {instagram_url}")
            
            # Get current headers
            headers = self.sheet.row_values(1)
            
            # Ensure required columns exist
            required_columns = ['instagram_link', 'status', 'posted_url', 'error_reason', 'processed_date']
            for col in required_columns:
                if col not in headers:
                    self.sheet.update_cell(1, len(headers) + 1, col)
                    headers.append(col)
            
            # Get all current values to find next empty row
            all_values = self.sheet.get_all_values()
            next_row = len(all_values) + 1
            
            # Get column indices
            col_instagram_link = headers.index('instagram_link') + 1
            col_status = headers.index('status') + 1
            
            # Add the new row
            self.sheet.update_cell(next_row, col_instagram_link, instagram_url)
            self.sheet.update_cell(next_row, col_status, 'pending')
            
            logger.info(f"✓ Video added to row {next_row}")
            
            return {
                'success': True,
                'row_number': next_row,
                'message': f'Video added successfully at row {next_row}'
            }
            
        except Exception as e:
            logger.error(f"Error adding video to sheet: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'row_number': -1,
                'message': f'Error: {str(e)}'
            }
    
    def init_google_sheets(self):
        """Initialize Google Sheets connection"""
        try:
            logger.info("Initializing Google Sheets connection...")
            
            # Load service account from environment variable
            service_account_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT")
            if not service_account_json:
                raise ValueError("GOOGLE_SERVICE_ACCOUNT environment variable not set")
            
            # Clean up the JSON string (remove any extra quotes or whitespace)
            service_account_json = service_account_json.strip()
            
            # Try to parse JSON
            try:
                service_account_info = json.loads(service_account_json)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse GOOGLE_SERVICE_ACCOUNT JSON: {e}")
                logger.error(f"First 200 chars: {service_account_json[:200]}")
                raise ValueError(f"Invalid JSON in GOOGLE_SERVICE_ACCOUNT: {e}")
            
            # Fix private key formatting
            service_account_info["private_key"] = service_account_info["private_key"].replace("\\n", "\n")
            
            # Create credentials
            creds = Credentials.from_service_account_info(
                service_account_info,
                scopes=SCOPES
            )
            
            # Authorize gspread client
            client = gspread.authorize(creds)
            
            # Open the spreadsheet
            spreadsheet_name = SPREADSHEET_NAME
            sheet = client.open(spreadsheet_name).sheet1
            
            logger.info(f"✓ Connected to Google Sheet: {spreadsheet_name}")
            return sheet
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Sheets: {str(e)}")
            raise
    
    def process_next_pending_video(self):
        """Process only the first pending video from the Google Sheet"""
        result = {
            'success': False,
            'row_number': None,
            'instagram_link': None,
            'status': 'no_pending',
            'message': 'No pending videos found',
            'post_url': None,
            'error_reason': None,
            'pending_count': 0
        }
        
        try:
            logger.info("Reading data from Google Sheet...")
            
            # Get all records from sheet
            records = self.sheet.get_all_records()
            
            if not records:
                logger.warning("No data found in Google Sheet")
                return result
            
            logger.info(f"Loaded Google Sheet with {len(records)} rows")
            
            # Ensure required columns exist in first row (headers)
            headers = self.sheet.row_values(1)
            required_columns = ['instagram_link', 'status']
            
            if not all(col in headers for col in required_columns):
                logger.error(f"Sheet must contain columns: {required_columns}")
                result['message'] = f"Missing required columns: {required_columns}"
                return result
            
            # Add missing column headers if needed
            if 'posted_url' not in headers:
                self.sheet.update_cell(1, len(headers) + 1, 'posted_url')
                headers.append('posted_url')
            if 'error_reason' not in headers:
                self.sheet.update_cell(1, len(headers) + 1, 'error_reason')
                headers.append('error_reason')
            if 'processed_date' not in headers:
                self.sheet.update_cell(1, len(headers) + 1, 'processed_date')
                headers.append('processed_date')
            
            # Get column indices
            col_instagram_link = headers.index('instagram_link') + 1
            col_status = headers.index('status') + 1
            col_posted_url = headers.index('posted_url') + 1
            col_error_reason = headers.index('error_reason') + 1
            col_processed_date = headers.index('processed_date') + 1
            
            # Find the first pending video (top to bottom)
            pending_video_idx = None
            pending_video_link = None
            pending_count = 0
            
            for idx, record in enumerate(records, start=2):  # Start from row 2
                status = record.get('status', '').strip().lower()
                instagram_link = record.get('instagram_link', '').strip()
                
                # Count pending videos
                if status in ('', 'pending') and instagram_link:
                    pending_count += 1
                    # Only process the first one
                    if pending_video_idx is None:
                        pending_video_idx = idx
                        pending_video_link = instagram_link
            
            result['pending_count'] = pending_count
            
            # If no pending video found
            if pending_video_idx is None:
                logger.info("No pending videos found to process")
                result['message'] = f"No pending videos found. Total videos checked: {len(records)}"
                return result
            
            # Process the first pending video
            idx = pending_video_idx
            instagram_link = pending_video_link
            
            result['row_number'] = idx
            result['instagram_link'] = instagram_link
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing row {idx} (First pending video)")
            logger.info(f"Remaining pending videos: {pending_count}")
            logger.info(f"{'='*60}")
            logger.info(f"Instagram URL: {instagram_link}")
                
            # Step 1: Download Instagram video
            video_path, original_caption, error = self.download_instagram_video(instagram_link)
            if error:
                logger.error(f"✗ Download failed: {error}")
                self.sheet.update_cell(idx, col_status, 'error')
                self.sheet.update_cell(idx, col_error_reason, str(error))
                self.sheet.update_cell(idx, col_processed_date, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                
                result['success'] = False
                result['status'] = 'error'
                result['message'] = f"Download failed: {error}"
                result['error_reason'] = str(error)
                return result
                
            # Step 2: Generate new caption with Gemini
            new_caption, error = self.generate_caption_with_gemini(original_caption)
            if error:
                logger.error(f"✗ Caption generation failed: {error}")
                self.sheet.update_cell(idx, col_status, 'error')
                self.sheet.update_cell(idx, col_error_reason, str(error))
                self.sheet.update_cell(idx, col_processed_date, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                self.cleanup_temp_files(video_path)
                
                result['success'] = False
                result['status'] = 'error'
                result['message'] = f"Caption generation failed: {error}"
                result['error_reason'] = str(error)
                return result
                
            # Step 3: Upload to Facebook
            post_url, error = self.upload_video_to_facebook(video_path, new_caption)
            if error:
                logger.error(f"✗ Facebook upload failed: {error}")
                self.sheet.update_cell(idx, col_status, 'error')
                self.sheet.update_cell(idx, col_error_reason, str(error))
                self.sheet.update_cell(idx, col_processed_date, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                self.cleanup_temp_files(video_path)
                
                result['success'] = False
                result['status'] = 'error'
                result['message'] = f"Facebook upload failed: {error}"
                result['error_reason'] = str(error)
                return result
            
            # Success! Update Google Sheet
            self.sheet.update_cell(idx, col_status, 'posted')
            self.sheet.update_cell(idx, col_posted_url, str(post_url))
            self.sheet.update_cell(idx, col_error_reason, '')
            self.sheet.update_cell(idx, col_processed_date, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            # Clean up temp files
            self.cleanup_temp_files(video_path)
            
            result['success'] = True
            result['status'] = 'posted'
            result['message'] = 'Successfully posted to Facebook'
            result['post_url'] = post_url
            
            logger.info(f"✓✓✓ SUCCESS! Row {idx} posted to Facebook!")
            logger.info(f"Remaining pending videos: {pending_count - 1}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            result['success'] = False
            result['status'] = 'error'
            result['message'] = f"Fatal error: {str(e)}"
            result['error_reason'] = str(e)
            return result


# FastAPI Authentication
def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify username and password"""
    # Ensure credentials are strings and not None
    username = credentials.username if credentials.username else ""
    password = credentials.password if credentials.password else ""
    expected_username = API_USERNAME if API_USERNAME else "admin"
    expected_password = API_PASSWORD if API_PASSWORD else "admin123"
    
    logger.info(f"Authentication attempt - Username: {username}")
    logger.info(f"Expected username: {expected_username}")
    
    correct_username = secrets.compare_digest(username.encode('utf-8'), expected_username.encode('utf-8'))
    correct_password = secrets.compare_digest(password.encode('utf-8'), expected_password.encode('utf-8'))
    
    if not (correct_username and correct_password):
        logger.warning(f"Authentication failed for user: {username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    logger.info(f"Authentication successful for user: {username}")
    return credentials.username


# Helper function to process next pending video
def process_next_video():
    """Process the next pending video and return results"""
    start_time = time.time()
    start_datetime = datetime.now()
    
    try:
        logger.info("=" * 60)
        logger.info("Facebook Reel Automation - Processing Next Video")
        logger.info("=" * 60)
        
        # Initialize poster
        poster = FacebookPoster()
        
        # Process next pending video
        logger.info("Looking for next pending video...\n")
        result = poster.process_next_pending_video()
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info("\n" + "=" * 60)
        logger.info("Processing completed!")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info("=" * 60)
        
        result['started_at'] = start_datetime.isoformat()
        result['completed_at'] = datetime.now().isoformat()
        result['duration_seconds'] = duration
        
        return result
        
    except Exception as e:
        logger.error(f"Error in processing: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return {
            'success': False,
            'status': 'error',
            'message': str(e),
            'error_reason': str(e),
            'started_at': start_datetime.isoformat(),
            'completed_at': datetime.now().isoformat(),
            'duration_seconds': time.time() - start_time,
            'pending_count': 0
        }


# FastAPI Endpoints
@app.get("/")
def root():
    """API root endpoint"""
    # Check environment variables status
    env_status = {
        'facebook_page_id': 'Set' if PAGE_ID else 'Missing',
        'facebook_token': 'Set' if ACCESS_TOKEN else 'Missing',
        'gemini_api_key': 'Set' if GEMINI_API_KEY else 'Missing',
        'google_service_account': 'Set' if os.getenv('GOOGLE_SERVICE_ACCOUNT') else 'Missing'
    }
    
    return {
        "message": "Facebook Reel Automation API",
        "version": "2.0",
        "endpoints": {
            "/process": "POST - Process ONE pending video (requires authentication)",
            "/pending": "GET - Get count of pending videos (requires authentication)",
            "/health": "GET - Health check",
            "/auth-test": "GET - Test authentication (requires authentication)"
        },
        "environment_variables": env_status
    }


@app.get("/auth-test")
def test_auth(username: str = Depends(verify_credentials)):
    """Test authentication endpoint"""
    return {
        "status": "authenticated",
        "username": username,
        "message": "Authentication successful!"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/process", response_model=ProcessResult)
def process_one_video(username: str = Depends(verify_credentials)):
    """
    Process ONE pending video from Google Sheet
    
    Processes the first video with status='pending' from top to bottom.
    After processing, the status will be updated to either 'posted' or 'error'.
    
    Requires HTTP Basic Authentication:
    - Username: Set via API_USERNAME env var (default: admin)
    - Password: Set via API_PASSWORD env var (default: admin123)
    
    Returns:
    - status: 'posted', 'error', or 'no_pending'
    - row_number: Which row was processed
    - instagram_link: The video URL that was processed
    - post_url: Facebook post URL (if successful)
    - error_reason: Error message (if failed)
    - pending_count: Number of pending videos remaining
    - duration_seconds: How long the processing took
    """
    try:
        logger.info(f"Processing initiated by user: {username}")
        
        # Process next pending video
        result = process_next_video()
        
        return ProcessResult(
            status=result['status'],
            message=result['message'],
            row_number=result.get('row_number'),
            instagram_link=result.get('instagram_link'),
            post_url=result.get('post_url'),
            error_reason=result.get('error_reason'),
            started_at=result['started_at'],
            completed_at=result['completed_at'],
            duration_seconds=result['duration_seconds'],
            pending_count=result.get('pending_count', 0)
        )
        
    except Exception as e:
        logger.error(f"Error in processing: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}"
        )


@app.get("/pending", response_model=Dict)
def get_pending_count(username: str = Depends(verify_credentials)):
    """
    Get count of pending videos without processing any
    
    Requires HTTP Basic Authentication.
    
    Returns:
    - pending_count: Number of videos with status='pending'
    - total_rows: Total number of rows in sheet
    """
    try:
        logger.info(f"Pending count requested by user: {username}")
        
        poster = FacebookPoster()
        records = poster.sheet.get_all_records()
        
        pending_count = 0
        for record in records:
            status = record.get('status', '').strip().lower()
            instagram_link = record.get('instagram_link', '').strip()
            if status == 'pending' and instagram_link:
                pending_count += 1
        
        return {
            'pending_count': pending_count,
            'total_rows': len(records),
            'message': f"{pending_count} videos pending out of {len(records)} total"
        }
        
    except Exception as e:
        logger.error(f"Error getting pending count: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pending count: {str(e)}"
        )


@app.post("/add-video", response_model=AddVideoResponse)
def add_video(
    request: AddVideoRequest,
    username: str = Depends(verify_credentials)
):
    """
    Add a new Instagram reel URL to Google Sheet
    
    This endpoint adds the video with status='pending' so it can be
    processed later by calling the /process endpoint.
    
    Requires HTTP Basic Authentication.
    
    Request Body:
    - instagram_url: Full Instagram reel/post URL
    
    Returns:
    - success: True/False
    - message: Success or error message
    - row_number: Row number where video was added
    - instagram_url: The URL that was added
    - status: Will be 'pending'
    
    Example:
    ```
    POST /add-video
    {
        "instagram_url": "https://www.instagram.com/reel/ABC123/"
    }
    ```
    """
    try:
        logger.info(f"Add video requested by user: {username}")
        
        # Validate URL format
        instagram_url = request.instagram_url.strip()
        
        if not instagram_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Instagram URL cannot be empty"
            )
        
        # Basic validation - check if it's an Instagram URL
        if 'instagram.com' not in instagram_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="URL must be an Instagram link"
            )
        
        # Initialize poster and add to sheet
        poster = FacebookPoster()
        result = poster.add_video_to_sheet(instagram_url)
        
        if not result['success']:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result['message']
            )
        
        return AddVideoResponse(
            success=True,
            message=result['message'],
            row_number=result['row_number'],
            instagram_url=instagram_url,
            status='pending'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding video: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add video: {str(e)}"
        )


def main():
    """Main function for direct execution"""
    print("=" * 60)
    print("Facebook Video Posting Automation")
    print("=" * 60)
    
    logger.info("Starting Facebook Video Posting Automation")
    
    # Initialize poster
    try:
        poster = FacebookPoster()
    except Exception as e:
        logger.error(f"Failed to initialize: {str(e)}")
        return
    
    # Process next pending video
    logger.info("Processing next pending video...\n")
    result = poster.process_next_pending_video()
    
    logger.info(f"\nResult:")
    logger.info(f"  Status: {result['status']}")
    logger.info(f"  Message: {result['message']}")
    if result.get('row_number'):
        logger.info(f"  Row: {result['row_number']}")
    logger.info(f"  Pending videos remaining: {result.get('pending_count', 0)}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Done! Check facebook_automation.log for full details.")
    logger.info("=" * 60)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)