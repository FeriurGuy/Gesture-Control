from gtts import gTTS # type: ignore
import os
import threading
import queue
import pygame
import time
import hashlib

class Speaker:
    def __init__(self):
        self.q = queue.Queue()
        self.running = True
        self.cache_dir = "audio_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        pygame.mixer.init()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        print("[SYSTEM] Audio Engine Initialized.")

    def _worker(self):
        while self.running:
            text = self.q.get()
            if text is None: break
            self._play_audio(text)
            self.q.task_done()

    def say(self, text: str):
        if text and text.strip():
            self.q.put(text)

    def _get_filename(self, text):
        hash_name = hashlib.md5(text.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_name}.mp3")

    def _play_audio(self, text):
        print(f"[AUDIO] Playing: {text}")
        filename = self._get_filename(text)
        
        if not os.path.exists(filename):
            try:
                tts = gTTS(text=text, lang="id")
                tts.save(filename)
            except Exception as e:
                print(f"[ERROR] TTS Generation failed: {e}")
                return

        try:
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() and self.running:
                time.sleep(0.1)
        except Exception as e:
            print(f"[ERROR] Playback failed: {e}")

    def stop(self):
        self.running = False
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        pygame.mixer.quit()
        self.q.put(None)
        self.thread.join(timeout=2)