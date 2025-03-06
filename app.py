import json
import os
import shutil

from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import Request
from fastapi import UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import zmq

context = zmq.Context()
mq_socket = context.socket(zmq.PUB)  # Publisher socket
try:
    mq_socket.bind("tcp://*:5555")
except zmq.error.ZMQError:
    mq_socket.connect("tcp://localhost:5555")

app = FastAPI(title="SUNET Transcriber")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

TRANSCRIBE_DIR = Path("transcribed")
TRANSCRIBE_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), model: str = Form("whisper-large-v3")):
    try:
        print(f"Selected model: {model}")
        file_path = UPLOAD_DIR / file.filename

        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Send the file path to the transcriber
        mq_socket.send_string(json.dumps(
            {"file_path": file_path, "model": model}, default=str))

        print("File sent to transcriber")

        # Return success message (HTMX will insert this into #messages)
        return HTMLResponse(
            f"""
        <div id="messages" class="alert alert-success" role="alert">
            File uploaded successfully!
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
        <script>
            document.body.dispatchEvent(new Event('fileUploaded'));
        </script>
        """
        )

    except Exception as e:
        return HTMLResponse(
            f"""
        <div id="messages" class="alert alert-danger" role="alert">
            Upload failed: {str(e)}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
        """
        )


@ app.get("/api/files", response_class=HTMLResponse)
async def get_files():
    html = """
    <table class="table table-striped table-hover">
        <thead>
            <tr>
                <th>Transcription</th>
                <th>Size</th>
                <th>Date</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody>
    """

    files = 0
    for file in os.listdir(TRANSCRIBE_DIR):
        files += 1
        file_path = os.path.join(TRANSCRIBE_DIR, file)

        creation_time = os.path.getctime(file_path)
        upload_time = datetime.fromtimestamp(creation_time).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        file_size = os.path.getsize(file_path)

        html += f"""
        <tr>
            <td>{file}</td>
            <td>{file_size}</td>
            <td>{upload_time}</td>
            <td>
                <a href="/api/files/{file}" class="btn btn-sm btn-primary">
                    Download
                </a>
                <button
                    hx-delete="/api/files/{file}"
                    hx-confirm="Are you sure you want to delete this file?"
                    hx-target="#files-table"
                    class="btn btn-sm btn-danger">
                    Delete
                </button>
            </td>
        </tr>
        """

    html += """
        </tbody>
    </table>
    """

    if files == 0:
        html = """
        <div class="alert alert-info">
            No files transribed yet.
        </div>
        """

    return HTMLResponse(html)


@ app.get("/api/files/{file_id}")
async def download_file(file_id: str):
    for file in os.listdir(TRANSCRIBE_DIR):
        filename = os.path.join(TRANSCRIBE_DIR, file)
        if filename == str(TRANSCRIBE_DIR) + "/" + file_id:
            return FileResponse(
                path=os.path.join(TRANSCRIBE_DIR, file),
                filename=filename,
                media_type="application/octet-stream",
            )
    return {"error": "File not found"}


@ app.delete("/api/files/{file_id}", response_class=HTMLResponse)
async def delete_file(file_id: str):
    for file in os.listdir(TRANSCRIBE_DIR):
        filename = os.path.join(TRANSCRIBE_DIR, file)
        if filename == str(TRANSCRIBE_DIR) + "/" + file_id:
            # Delete the actual file
            file_path = Path(os.path.join(TRANSCRIBE_DIR, file))
            if file_path.exists():
                os.remove(file_path)

            # Return updated file list
            return await get_files()

    return HTMLResponse(
        """
    <div class="alert alert-danger">
        File not found!
    </div>
    """
    )


# Demonstration code to run the app (for development)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
