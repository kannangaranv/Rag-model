# FastAPI Project Setup

## 1. Install & Link FFmpeg (Windows)

1. **Download FFmpeg**
   - Go to: [gyan.dev FFmpeg builds](https://www.gyan.dev/ffmpeg/builds/)
   - Download the **Release full build** ZIP.
   - Extract the ZIP to: `C:\ffmpeg`  
     (Expect `C:\ffmpeg\bin\ffmpeg.exe` afterward.)

2. **Add FFmpeg to the System PATH**
   - Press **Win + S** → search **Environment Variables** → open **Edit the system environment variables**.
   - Click **Environment Variables…**.
   - Under **System variables**, select **Path** → **Edit** → **New**.
   - Add: `C:\ffmpeg\bin` → confirm with **OK** on all dialogs.

3. **Verify the installation**
   ```cmd
   ffmpeg -version
   
## 2 Add the `.env` file

1. Rename `.env.template` to `.env`.
2. Open `.env` and fill in values for each environment variable.

## 3. Create Virtual Environment (Command Prompt)
```cmd
python -m venv .venv
```
## 4. Activate Virtual Environment

```bash
.venv\Scripts\activate.bat
```

## 5. Install packages 
```bash
pip install -r requirements.txt
```

## 6. Run the project
```bash
uvicorn app.main:app --reload --port 8000
```







