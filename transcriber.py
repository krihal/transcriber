import json
import logging
import os
import queue
import threading
import time

from pathlib import Path

import whisper
import zmq

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("whisper_transcriber")

# Configuration
WATCH_DIRECTORY = "transcoded"
OUTPUT_DIRECTORY = "transcribed"
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".mp4", ".m4a", ".flac", ".ogg"}
MAX_WORKER_THREADS = 2
file_queue = queue.Queue()

# Create directories if they don't exist
os.makedirs(WATCH_DIRECTORY, exist_ok=True)
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


def transcribe_audio(audio_path, model):
    """Transcribe audio file using Whisper"""
    try:
        logger.info(f"Starting transcription of {audio_path}")

        # Load Whisper model
        logger.info(f"Loading Whisper model: {model}")
        model = whisper.load_model(model)

        logger.info("Whisper model loaded successfully")

        # Get transcription
        result = model.transcribe(str(audio_path))

        # Create output file path
        input_path = Path(audio_path)
        output_filename = f"{input_path.stem}_transcription.txt"
        output_path = Path(OUTPUT_DIRECTORY) / output_filename

        # Write transcription to file
        with open(output_path, "w", encoding="utf-8") as f:
            for line in result["segments"]:
                f.write(line["text"] + "\n")

        logger.info(f"Transcription completed and saved to {output_path}")

    except Exception as e:
        logger.error(f"Error transcribing {audio_path}: {str(e)}")


def worker():
    """Worker thread that processes files from the queue"""
    files = {}

    while True:
        # Get a file path from the queue
        file_data = file_queue.get()

        if file_data is None:  # None is our signal to stop
            break

        if file_data["file_path"] not in files:
            files[file_data["file_path"]] = {"ready": False, "model": None}

        files[file_data["file_path"]]["ready"] = True

        if "model" in file_data:
            files[file_data["file_path"]]["model"] = file_data["model"]

        # Process the file
        if files[file_data["file_path"]]["ready"] and files[file_data["file_path"]]["model"]:
            transcribe_audio(file_data["file_path"], file_data["model"])

        file_queue.task_done()


class AudioFileHandler(FileSystemEventHandler):
    """Handles file system events"""

    def __init__(self):
        self.processed_files = set()

    def on_created(self, event):
       # Skip directories and non-audio files
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if file_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            return

        # Avoid processing the same file multiple times (watchdog can trigger multiple events)
        if str(file_path) in self.processed_files:
            return

        logger.info(f"New audio file detected: {file_path}")
        self.processed_files.add(str(file_path))

        # Add file to processing queue
        file_queue.put({"file_path": str(file_path), "ready": True})


def main():
    # Start worker threads
    threads = []
    for _ in range(MAX_WORKER_THREADS):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        threads.append(t)

    # Start the file system observer
    event_handler = AudioFileHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIRECTORY, recursive=False)
    observer.start()

    context = zmq.Context()
    mq_socket = context.socket(zmq.SUB)  # Subscriber socket
    mq_socket.connect("tcp://localhost:5555")
    mq_socket.subscribe("")

    logger.info(f"Watching directory: {WATCH_DIRECTORY}")
    logger.info(f"Transcriptions will be saved to: {OUTPUT_DIRECTORY}")

    try:
        while True:
            print("Listening for messages...")
            filedata = mq_socket.recv_string()

            logger.info(f"Received message: {filedata}")

            file_queue.put(json.loads(filedata))

            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping observer...")
        observer.stop()

        # Stop worker threads
        logger.info("Stopping worker threads...")
        for _ in range(len(threads)):
            file_queue.put(None)  # Signal each thread to stop

        # Wait for all threads to complete
        for t in threads:
            t.join()

    observer.join()
    logger.info("Whisper transcription service stopped")


if __name__ == "__main__":
    main()
