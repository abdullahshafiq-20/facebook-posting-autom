# Facebook Video Posting Automation

This tool automates downloading Instagram videos, generating new captions with AI, and posting them to Facebook.

## Features

- ✅ Downloads videos from Instagram (posts, reels, and more) using yt-dlp
- ✅ Extracts original captions/descriptions automatically
- ✅ Generates new captions + hashtags using Google Gemini AI
- ✅ Posts videos to Facebook Page as **Reels**
- ✅ Updates CSV with posting status
- ✅ Error handling and logging
- ✅ Automatic cleanup of temporary files
- ✅ Works with many video platforms (Instagram, YouTube, TikTok, etc.)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Credentials

Edit `config.json` with your credentials:

```json
{
  "facebook": {
    "page_id": "YOUR_FACEBOOK_PAGE_ID",
    "page_access_token": "YOUR_PAGE_ACCESS_TOKEN"
  },
  "gemini": {
    "api_key": "YOUR_GEMINI_API_KEY"
  }
}
```

#### Getting Facebook Credentials:

1. Go to [Facebook Developers](https://developers.facebook.com/)
2. Create an app or use existing one
3. Add "Facebook Login" product
4. Get your Page ID from your Facebook Page settings
5. Generate a Page Access Token:
   - Go to Graph API Explorer
   - Select your app
   - Select your page
   - Add permissions: `pages_manage_posts`, `pages_read_engagement`, `pages_manage_engagement`
   - Generate token
   - **Important**: For Reels, ensure your Page has access to the Reels feature

#### Getting Gemini API Key:

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to config.json

### 3. Set up Google Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project or select existing one
3. Enable Google Sheets API and Google Drive API
4. Create a Service Account:
   - Go to IAM & Admin → Service Accounts
   - Create new service account
   - Download the JSON key file
5. Copy the entire JSON content to environment variable:
   ```bash
   export GOOGLE_SERVICE_ACCOUNT='{"type":"service_account",...}'
   ```
6. Share your Google Sheet with the service account email

### 4. Prepare Google Sheet

Create a Google Sheet named "Mindrots" (or configure name in config.json) with these columns:

| Column Name | Description | Required |
|-------------|-------------|----------|
| instagram_link | Full Instagram video URL | Yes |
| status | Initial value: "pending" | Yes |
| posted_url | (Auto-filled) Facebook post URL | No |
| error_reason | (Auto-filled) Error message if failed | No |
| processed_date | (Auto-filled) Processing timestamp | No |

**Example Google Sheet:**
```
| instagram_link                          | status  |
|-----------------------------------------|---------|
| https://www.instagram.com/p/ABC123/     | pending |
| https://www.instagram.com/reel/XYZ789/  | pending |
```

The script will automatically add these columns:
- `posted_url` - Facebook post URL after success
- `error_reason` - Error message if failed
- `processed_date` - Processing timestamp

## Usage

### Option 1: Direct Execution

Run the script directly:

```bash
python main.py
```

### Option 2: FastAPI Server (Recommended)

Start the API server:

```bash
python run_api.py
# or
python main.py api
```

The server will start on `http://localhost:8000`

**API Endpoints:**

1. **Start Processing** (Protected)
   ```bash
   curl -X POST http://localhost:8000/start \
     -u admin:admin123
   ```
   
   Or use Python:
   ```python
   import requests
   from requests.auth import HTTPBasicAuth
   
   response = requests.post(
       'http://localhost:8000/start',
       auth=HTTPBasicAuth('admin', 'admin123')
   )
   print(response.json())
   ```

2. **Health Check**
   ```bash
   curl http://localhost:8000/health
   ```

3. **API Documentation**
   Visit: `http://localhost:8000/docs`

The workflow:
1. Read each row from the Google Sheet
2. Download the Instagram video
3. Generate a new caption with Gemini AI
4. Post to Facebook
5. Update the Excel status column
6. Log all actions to `facebook_automation.log`

## Sheet Status Values

- `pending` - Not yet processed
- `posted` - Successfully posted to Facebook
- `error` - Failed (see error_reason column)

## Logging

All operations are logged to:
- Console output
- `facebook_automation.log` file

## Error Handling

If any step fails:
- The error is logged
- Google Sheet status is set to "error"  
- The error_reason column is populated
- Processing continues with the next row

## Rate Limiting

The script includes a 5-second delay between posts to avoid rate limiting.

## Troubleshooting

### "Invalid Instagram URL"
- Ensure URLs are complete Instagram post/reel URLs
- Format: `https://www.instagram.com/p/SHORTCODE/` or `https://www.instagram.com/reel/SHORTCODE/`

### "Facebook API Error"
- Check your page access token is valid
- Ensure you have the required permissions
- Verify your page ID is correct

### "Gemini API Error"
- Check your API key is valid
- Ensure you have API quota remaining

### Video download fails
- Some Instagram accounts may be private
- The video URL might be expired or invalid
- For private accounts, yt-dlp cannot access the content
- Make sure the URL is a direct link to a video post/reel

## Notes

- Videos are temporarily downloaded to `temp_downloads/` and automatically cleaned up
- The script updates Google Sheet after each row, so you can safely stop and resume
- Already posted videos (status = "posted") are automatically skipped
- yt-dlp automatically extracts captions/descriptions from Instagram posts
- yt-dlp also works with other platforms (YouTube, TikTok, etc.) - just use their URLs!
- **Videos are posted as Facebook Reels** (short-form vertical videos)
- Reel requirements: Videos should ideally be vertical (9:16 aspect ratio) and under 90 seconds
- Multiple people can collaborate on the same Google Sheet
- Real-time updates visible to all collaborators
- API runs processing in the background, so you get an immediate response

## License

MIT License
