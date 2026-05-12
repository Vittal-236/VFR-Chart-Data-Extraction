from fastapi import FastAPI, UploadFile, Form
from fastapi.staticfiles import StaticFiles
import subprocess, json, tempfile, os

app = FastAPI()
#app.mount("/", StaticFiles(directory="ui", html=True), name="ui")

@app.post("/extract")
async def extract(file: UploadFile, ansp: str = Form("GR")):
    with tempfile.NamedTemporaryFile(suffix=file.filename, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    subprocess.run(["python", "main.py", "--input", tmp_path, "--ansp", ansp])
    output = f"data/output/{os.path.basename(tmp_path).split('.')[0]}_extracted.json"
    return json.load(open(output))