from pytube import YouTube
import urllib.request
import urllib.error

def download_video(url, output_path=None):
    try:
        # Add headers to mimic a web browser request
        opener = urllib.request.build_opener()
        opener.addheaders = [
            ('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        ]
        urllib.request.install_opener(opener)
        
        # Create YouTube object with additional parameters
        yt = YouTube(
            url,
            use_oauth=False,
            allow_oauth_cache=True
        )
        
        # Get the highest resolution stream
        video = yt.streams.get_highest_resolution()
        
        # Print video details
        print(f"Title: {yt.title}")
        print(f"Length: {yt.length} seconds")
        print(f"Views: {yt.views}")
        
        # Download the video
        print("Downloading...")
        video.download(output_path)
        print("Download completed!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Example usage
    video_url = "https://youtube.com/watch?v=5b-LESXSyDQ"
    download_path = "./"
    
    if not download_path:
        download_path = None
        
    download_video(video_url, download_path)
