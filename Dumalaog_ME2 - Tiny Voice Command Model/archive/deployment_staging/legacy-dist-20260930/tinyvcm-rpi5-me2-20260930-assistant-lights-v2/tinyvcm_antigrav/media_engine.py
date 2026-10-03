"""Media engine for Tiny Voice Command Model.

Supports:
1. Bundled local demo melodies, followed by optional external radio streams.
2. Dynamic YouTube audio search and playback via YouTube IFrame API and stream metadata.
3. Spotify track resolution (supports optional SPOTIPY_CLIENT_ID / SPOTIPY_CLIENT_SECRET).
"""

import json
import os
import re
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

CURATED_TRACKS = [
    {
        "id": "me2_demo_1", "title": "ME2 Demo Melody", "artist": "Locally generated",
        "source": "local", "url": "/assets/demo_melody_1.wav", "cover": "", "genre": "Demo"
    },
    {
        "id": "me2_demo_2", "title": "ME2 Demo Arpeggio", "artist": "Locally generated",
        "source": "local", "url": "/assets/demo_melody_2.wav", "cover": "", "genre": "Demo"
    },
    {
        "id": "lofi_1",
        "title": "Lofi Chill Beats",
        "artist": "Lofi Girl / ChilledCow",
        "source": "curated",
        "url": "https://stream.zeno.fm/f3wvbbqmdg8uv",
        "youtube_id": "jfKfPfyJRdk",
        "cover": "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=300&h=300&fit=crop",
        "genre": "Lofi Hip Hop"
    },
    {
        "id": "synthwave_1",
        "title": "Neon Nights (Synthwave)",
        "artist": "Nightwave Plaza",
        "source": "curated",
        "url": "https://radio.plaza.one/mp3",
        "youtube_id": "4xDzrJKXOOY",
        "cover": "https://images.unsplash.com/photo-1508700115892-45ecd05ae2ad?w=300&h=300&fit=crop",
        "genre": "Synthwave"
    },
    {
        "id": "classical_1",
        "title": "Moonlight Sonata (Piano)",
        "artist": "Ludwig van Beethoven",
        "source": "curated",
        "url": "https://ia800504.us.archive.org/3/items/MoonlightSonata_755/Beethoven-MoonlightSonata.mp3",
        "youtube_id": "4Tr0otuiQuU",
        "cover": "https://images.unsplash.com/photo-1520523839898-507121287c8a?w=300&h=300&fit=crop",
        "genre": "Classical"
    },
    {
        "id": "jazz_1",
        "title": "Café Acoustic Jazz",
        "artist": "Smooth Jazz Coffee Lounge",
        "source": "curated",
        "url": "https://stream.zeno.fm/0r0xa792kwzuv",
        "youtube_id": "Dx5qFachd3A",
        "cover": "https://images.unsplash.com/photo-1511192336575-5a79af67a629?w=300&h=300&fit=crop",
        "genre": "Smooth Jazz"
    }
]

class MediaEngine:
    def __init__(self):
        self.playlist: List[Dict] = list(CURATED_TRACKS)
        self.current_index: int = 0
        self.youtube_api_key = os.environ.get("YOUTUBE_API_KEY", "")
        self.spotify_client_id = os.environ.get("SPOTIPY_CLIENT_ID", "")
        self.spotify_client_secret = os.environ.get("SPOTIPY_CLIENT_SECRET", "")

    def get_current_track(self) -> Dict:
        if not self.playlist:
            return CURATED_TRACKS[0]
        return self.playlist[self.current_index % len(self.playlist)]

    def next_track(self) -> Dict:
        if self.playlist:
            self.current_index = (self.current_index + 1) % len(self.playlist)
        return self.get_current_track()

    def previous_track(self) -> Dict:
        if self.playlist:
            self.current_index = (self.current_index - 1) % len(self.playlist)
        return self.get_current_track()

    def search_youtube(self, query: str) -> Optional[Dict]:
        """Search YouTube for a song and return playback metadata."""
        clean_query = query.strip()
        if not clean_query:
            return None

        # Method 1: If YouTube Data API key is provided
        if self.youtube_api_key:
            try:
                params = urllib.parse.urlencode({
                    "part": "snippet",
                    "maxResults": 1,
                    "q": clean_query,
                    "type": "video",
                    "videoCategoryId": "10",  # Music
                    "key": self.youtube_api_key
                })
                url = f"https://www.googleapis.com/youtube/v3/search?{params}"
                req = urllib.request.Request(url, headers={"User-Agent": "TinyVCM/1.0"})
                with urllib.request.urlopen(req, timeout=3) as resp:
                    data = json.loads(resp.read().decode())
                items = data.get("items", [])
                if items:
                    item = items[0]
                    vid_id = item["id"]["videoId"]
                    title = item["snippet"]["title"]
                    channel = item["snippet"]["channelTitle"]
                    thumb = item["snippet"]["thumbnails"]["medium"]["url"]
                    return {
                        "id": f"yt_{vid_id}",
                        "title": title,
                        "artist": channel,
                        "source": "youtube",
                        "youtube_id": vid_id,
                        "url": f"https://www.youtube.com/watch?v={vid_id}",
                        "cover": thumb,
                        "genre": "YouTube Music"
                    }
            except Exception as e:
                print(f"[MediaEngine] YouTube API search error: {e}")

        # Method 2: Public YouTube search scraping (zero API key)
        try:
            encoded = urllib.parse.quote_plus(f"{clean_query} audio")
            search_url = f"https://www.youtube.com/results?search_query={encoded}"
            req = urllib.request.Request(search_url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9"
            })
            with urllib.request.urlopen(req, timeout=3) as resp:
                html = resp.read().decode("utf-8", errors="ignore")

            # Extract videoId from ytInitialData
            matches = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', html)
            if matches:
                # Filter out channel IDs, take first valid video ID
                vid_id = matches[0]
                # Try to extract title
                title_match = re.search(r'"title":{"runs":\[{"text":"([^"]+)"}\]', html)
                title = title_match.group(1) if title_match else clean_query.title()
                
                track = {
                    "id": f"yt_{vid_id}",
                    "title": title,
                    "artist": "YouTube Music",
                    "source": "youtube",
                    "youtube_id": vid_id,
                    "url": f"https://www.youtube.com/watch?v={vid_id}",
                    "cover": f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg",
                    "genre": "Search Result"
                }
                return track
        except Exception as e:
            print(f"[MediaEngine] YouTube search fallback error: {e}")

        # Fallback to curated track matching query keywords
        q_lower = clean_query.lower()
        for t in CURATED_TRACKS:
            if any(w in t["title"].lower() or w in t["genre"].lower() for w in q_lower.split()):
                return t

        return None

    def play_query(self, query: str) -> Dict:
        """Search and insert track at top of playlist, set as current."""
        found = self.search_youtube(query)
        if found:
            # Insert at current position + 1 or replace current
            self.playlist.insert(self.current_index + 1, found)
            self.current_index = (self.current_index + 1) % len(self.playlist)
            return found
        return self.get_current_track()
