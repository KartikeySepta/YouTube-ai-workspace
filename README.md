# YouTube Transcript Scraper

YouTube Transcript Scraper is a Python CLI for downloading YouTube audio, extracting video metadata, and generating timestamped transcripts. It supports local transcription with Faster Whisper and optional cloud transcription with Google Gemini, then exports clean structured JSON for search, research, content analysis, datasets, and automation workflows.

## SEO Keywords

YouTube transcript scraper, YouTube transcription tool, YouTube audio downloader, YouTube metadata extractor, Python YouTube scraper, yt-dlp transcript CLI, Faster Whisper YouTube transcription, Gemini transcription CLI, video transcript dataset, YouTube transcript JSON exporter.

## Features

- Extracts YouTube audio with `yt-dlp`
- Converts audio to MP3 through FFmpeg
- Captures rich video metadata including title, channel, duration, views, likes, comments, tags, categories, thumbnail, language, upload date, and description
- Generates timestamped transcripts locally with `faster-whisper`
- Supports GPU acceleration with CUDA and falls back to CPU when needed
- Optional Google Gemini cloud transcription mode
- Prints structured JSON or appends results to an output JSON file
- Designed for research, SEO analysis, video indexing, content repurposing, and dataset creation

## Project Structure

```text
youtube-scrap/
├── youtube.py          # Main CLI pipeline
├── test_audio.py       # Audio extraction test script
├── test_gpu.py         # GPU / Whisper environment test script
├── test_meta.py        # Metadata extraction test script
├── test_pipeline.py    # End-to-end pipeline test script
├── .gitignore          # Git ignore rules for envs, cache, audio, and output data
└── README.md           # Project documentation
```

## Requirements

- Python 3.10+
- FFmpeg installed and available on your PATH
- A working internet connection for downloading YouTube media
- Optional NVIDIA GPU with CUDA for faster local transcription
- Optional Google AI Studio API key for Gemini cloud transcription

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/YOUR_USERNAME/youtube-scrap.git
cd youtube-scrap
python3 -m venv venv
source venv/bin/activate
```

Install the Python packages:

```bash
pip install -r requirements.txt
```

Install FFmpeg if you do not already have it:

```bash
sudo apt install ffmpeg
```

On macOS with Homebrew:

```bash
brew install ffmpeg
```

## Usage

Run local transcription with the default `small` Whisper model:

```bash
python youtube.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Save or append the result to a JSON file:

```bash
python youtube.py "https://www.youtube.com/watch?v=VIDEO_ID" --output final_data.json
```

Use a smaller local model:

```bash
python youtube.py "https://www.youtube.com/watch?v=VIDEO_ID" --engine local --model base
```

Use Gemini cloud transcription:

```bash
python youtube.py "https://www.youtube.com/watch?v=VIDEO_ID" --engine cloud --output final_data.json
```

Before using cloud mode, replace `GEMINI_API_KEY` in `youtube.py` with your Google AI Studio API key.

## JSON Output

The CLI exports a payload like this:

```json
{
    "metadata": {
        "video_id": "VIDEO_ID",
        "title": "Video title",
        "channel": "Channel name",
        "channel_url": "https://www.youtube.com/@channel",
        "subscriber_count": 100000,
        "duration_seconds": 600,
        "view_count": 250000,
        "like_count": 12000,
        "comment_count": 500,
        "upload_date": "20260702",
        "tags": ["youtube", "transcript"],
        "categories": ["Education"],
        "thumbnail_url": "https://...",
        "is_live": false,
        "language": "en",
        "description": "Video description"
    },
    "transcript": "[00:00] Transcript text starts here..."
}
```

When `--output` points to an existing JSON file, the script loads the current data, converts it to a list if needed, appends the new result, and writes the updated list back to disk.

## Common Use Cases

- Build searchable YouTube transcript datasets
- Extract video metadata for SEO research
- Generate transcripts for content repurposing
- Analyze podcasts, lectures, tutorials, interviews, and long-form videos
- Create JSON records for RAG, search indexes, dashboards, or data pipelines

## GitHub Topics

Suggested repository topics:

```text
youtube-scraper, youtube-transcript, transcription, speech-to-text, yt-dlp, faster-whisper, whisper, gemini-api, metadata-extraction, python-cli, video-analysis, seo-tools
```

## Contributing

Contributions are welcome. Good first improvements include adding a `requirements.txt`, moving secrets to environment variables, improving error handling, adding automated tests, supporting more Whisper model sizes, and improving output formats.

To contribute:

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature/your-feature-name
```

3. Make your changes
4. Run a basic syntax check

```bash
python3 -m py_compile youtube.py
```

5. Commit your work

```bash
git add .
git commit -m "Add your change summary"
```

6. Open a pull request with a clear description of what changed and why

## Development Notes

- Do not commit API keys, local virtual environments, generated MP3 files, or large transcript datasets
- Keep generated JSON output out of Git unless it is small sample data
- Prefer focused pull requests that solve one problem at a time
- Test both local and cloud transcription paths when touching shared pipeline logic

## Responsible Use

Use this tool only for videos you are allowed to process. Respect YouTube's Terms of Service, creator rights, copyright rules, and privacy expectations.

## License

No license has been added yet. Add a license before publishing if you want others to use, modify, or distribute the project.
