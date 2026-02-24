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
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

## Auth (JWT)

- Login: `POST /api/auth/login` with JSON `{ "username": "alice", "password": "secret" }` → returns `{ "access_token": "...", "token_type": "bearer" }`.
- OAuth2 token: `POST /api/auth/token` with form fields `username`, `password`.
- Register: `POST /api/auth/register` with JSON `{ "username", "password" }`.
- Current user: `GET /api/auth/me` with header `Authorization: Bearer <token>`.
- Users support role-based registration (`role`) with backward-compatible `level` payload support. JWT includes `role` and `lvl` claims.

All existing `/api/*` endpoints (documents/videos/query) now require a valid Bearer token.

### Environment

Set JWT values in `backend/.env` or use defaults:

- `JWT_SECRET_KEY` (required in production)
- `JWT_ALGORITHM=HS256`
- `ACCESS_TOKEN_EXPIRE_MINUTES=60`

Notes
- Register with `role` (recommended) or legacy `level`.
```

## User Role Matrix Upload Vector DB

User-role matrix uploads (`POST /api/upload-user-roles`) can use a dedicated vector DB separate from the main manual/paper stores.

Set in `backend/.env`:

- `USER_ROLE_VECTOR_DB=faiss|milvus|pinecone`
- `USER_ROLE_MILVUS_COLLECTION=bp_user_role_collection`
- `USER_ROLE_PINECONE_INDEX_NAME=bp-user-role-index`

If unset, it defaults to `faiss` for user-role uploads only.

## Milvus Setup

# download docker-compose.yml
```bash
Invoke-WebRequest https://github.com/milvus-io/milvus/releases/download/v2.6.2/milvus-standalone-docker-compose.yml -OutFile docker-compose.yml
```

# start Milvus
```bash
docker compose up -d
```

## Milvus Auth (No Anonymous Access)

1. Milvus auth is enabled in `backend/milvus/configs/milvus.yaml`:
   - `common.security.authorizationEnabled: true`
   - `common.security.enablePublicPrivilege: false`
2. The config is mounted by `backend/docker-compose.yml` into the Milvus container.
3. Set app credentials in `backend/.env`:
   - `MILVUS_URI=http://localhost:19530`
   - `MILVUS_TOKEN=root:Milvus` (or your custom root password)
   - `MILVUS_REQUIRE_AUTH=true`
4. Recreate Milvus after config changes:

```bash
docker compose up -d --force-recreate
```

## RAG Reranking

Set these in `backend/.env`:

- `RERANK_ENABLED=true`
- `RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`
- `MANUAL_RERANK_CANDIDATES=20`
- `PAPER_RERANK_CANDIDATES=20`
- `MANUAL_RERANK_MAX_DOC_CHARS=2500`
- `PAPER_RERANK_MAX_DOC_CHARS=2500`

How it works:
- Retriever first fetches top candidates from vector DB using manual/paper candidate settings.
- Cross-encoder reranks those chunks against the query.
- Final top chunks are passed to the LLM answer step.

## Paper Chunking and Retrieval

Tune in `backend/.env`:

- `PAPER_CHUNK_SIZE=100`
- `PAPER_CHUNK_OVERLAP=20`
- `PAPER_RETRIEVAL_K=10`

Notes:
- Chunk values are currently word-based in code (not true tokenizer token counts).
- Increase `PAPER_RETRIEVAL_K` if paper answers are missing context.







