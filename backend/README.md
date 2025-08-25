# FastAPI Project Setup

## 1 Install & Link FFmpeg (Windows)

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

## 2. Create Databases and tables

Create a database with name ```KnowledgeBase```.
```bash
CREATE DATABASE [KnowledgeBase];
GO
```
Run following sql codes on sql server to create the table.
```bash
USE [KnowledgeBase];
GO

IF OBJECT_ID('dbo.Documents', 'U') IS NULL
BEGIN
  CREATE TABLE dbo.Documents (
      Id            UNIQUEIDENTIFIER NOT NULL DEFAULT NEWID() PRIMARY KEY,
      FileName      NVARCHAR(255)    NOT NULL,
      ContentType   NVARCHAR(100)    NOT NULL,
      FileSizeBytes INT              NOT NULL,
      Content       VARBINARY(MAX)   NOT NULL,
      MdText        NVARCHAR(MAX)    NULL,        -- optional: store extracted markdown too
      UploadedAt    DATETIME2 (7)    NOT NULL DEFAULT SYSUTCDATETIME()
  );
END
GO
```
```bash
USE [KnowledgeBase];
GO

IF OBJECT_ID('dbo.Videos', 'U') IS NULL
BEGIN
  CREATE TABLE dbo.Videos (
      Id            UNIQUEIDENTIFIER NOT NULL DEFAULT NEWID() PRIMARY KEY,
      FileName      NVARCHAR(255)    NOT NULL,
      ContentType   NVARCHAR(100)    NOT NULL,
      FileSizeBytes INT              NOT NULL,
      Content       VARBINARY(MAX)   NOT NULL,
      Transcript    NVARCHAR(MAX)    NULL,        -- optional: store extracted markdown too
      UploadedAt    DATETIME2 (7)    NOT NULL DEFAULT SYSUTCDATETIME()
  );
END
GO
```

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







