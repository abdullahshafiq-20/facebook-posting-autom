"""
Example script to process ALL pending videos by calling the API repeatedly
"""
import requests
from requests.auth import HTTPBasicAuth
import time

# Configuration
API_URL = "http://localhost:8000"
USERNAME = "admin"
PASSWORD = "admin123"

def get_pending_count():
    """Get the number of pending videos"""
    response = requests.get(
        f"{API_URL}/pending",
        auth=HTTPBasicAuth(USERNAME, PASSWORD)
    )
    if response.status_code == 200:
        return response.json()['pending_count']
    return 0

def process_one_video():
    """Process one pending video"""
    response = requests.post(
        f"{API_URL}/process",
        auth=HTTPBasicAuth(USERNAME, PASSWORD)
    )
    return response.json()

def main():
    print("=" * 60)
    print("Processing All Pending Videos")
    print("=" * 60)
    
    # Get initial count
    pending = get_pending_count()
    print(f"\nFound {pending} pending videos to process")
    
    if pending == 0:
        print("No videos to process!")
        return
    
    processed_count = 0
    posted_count = 0
    error_count = 0
    
    # Process all pending videos one by one
    while True:
        print(f"\n{'-' * 60}")
        print(f"Processing video {processed_count + 1}...")
        
        try:
            result = process_one_video()
            
            processed_count += 1
            
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")
            
            if result.get('row_number'):
                print(f"Row: {result['row_number']}")
                print(f"Link: {result.get('instagram_link', 'N/A')}")
            
            if result['status'] == 'posted':
                posted_count += 1
                print(f"✓ Posted URL: {result.get('post_url', 'N/A')}")
            elif result['status'] == 'error':
                error_count += 1
                print(f"✗ Error: {result.get('error_reason', 'Unknown error')}")
            elif result['status'] == 'no_pending':
                print("No more pending videos!")
                break
            
            remaining = result.get('pending_count', 0)
            print(f"Remaining: {remaining} videos")
            print(f"Duration: {result.get('duration_seconds', 0):.2f}s")
            
            if remaining == 0:
                print("\n✓ All videos processed!")
                break
            
            # Small delay between API calls
            time.sleep(2)
            
        except Exception as e:
            print(f"Error calling API: {e}")
            error_count += 1
            break
    
    # Summary
    print(f"\n{'=' * 60}")
    print("PROCESSING SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total processed: {processed_count}")
    print(f"Successfully posted: {posted_count}")
    print(f"Errors: {error_count}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
