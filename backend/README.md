# FastAPI Project Setup

## 1. Create Database and table

Create a database with name ```KnowledgeBase```.
Run following sql code on sql server to create the table.
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

## 2. Create Virtual Environment (Command Prompt)
```cmd
python -m venv .venv
```
## 3. Activate Virtual Environment

```bash
.venv\Scripts\activate.bat
```

## 4. Install packages 
```bash
pip install -r requirements.txt
```

## 5. Run the project
```bash
uvicorn app.main:app --reload --port 8000
```







