import json
from yt_dlp import YoutubeDL

VIDEO_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"

print(f"1. Pinging YouTube for expanded metadata: {VIDEO_URL}...")
ydl_opts = {
    'quiet': True,
    'no_warnings': True,
}

try:
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(VIDEO_URL, download=False)
        
    print("2. Building the comprehensive dictionary...")
    metadata = {
        "video_id": info.get("id"),
        "title": info.get("title"),
        "channel": info.get("uploader"),
        "channel_url": info.get("uploader_url"),
        "subscriber_count": info.get("channel_follower_count"), # Subscribers!
        "duration_seconds": info.get("duration"),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),         # Total Likes
        "comment_count": info.get("comment_count"),   # Total Comments
        "upload_date": info.get("upload_date"),
        "tags": info.get("tags", []),                 # The hidden Search Tags!
        "categories": info.get("categories", []),     # YouTube Category
        "thumbnail_url": info.get("thumbnail"),       # High-Res Cover Image
        "is_live": info.get("is_live", False),        # Is it a live stream?
        "language": info.get("language"),             # Primary video language
        "description": info.get("description")
    }
    
    print("\n" + "="*50)
    print("📊 EXPANDED METADATA JSON OUTPUT")
    print("="*50)
    print(json.dumps(metadata, indent=4, ensure_ascii=False))
    print("="*50 + "\n")
    
    print("✅ SUCCESS! Expanded metadata extraction is complete.")

except Exception as e:
    print(f"\n❌ FAILED during metadata extraction: {e}")