import hashlib
import tempfile
from dataclasses import asdict
from pathlib import Path, PurePath

import psycopg
from fastapi import APIRouter, BackgroundTasks, HTTPException, Response, UploadFile

from app.api.deps import CurrentUser, Store
from app.core.config import MAX_UPLOAD_BYTES
from app.db.store import Document
from app.schemas.documents import DocumentOut
from app.services.chunking import chunk_pages
from app.services.ingestion import process_document
from app.services.loaders import SUPPORTED_EXTENSIONS, LoaderError, load

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_FILENAME_CHARS = 255


def _out(doc: Document | None) -> DocumentOut:
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    return DocumentOut(**asdict(doc))


def _clean_filename(raw: str | None) -> str:
    # The client's filename is only stored as a label, never used as a path.
    name = PurePath(raw or "").name
    return "".join(c for c in name if c.isprintable()).strip()[:MAX_FILENAME_CHARS]


@router.post("", status_code=202, response_model=DocumentOut)
def upload_document(
    file: UploadFile, background: BackgroundTasks, user_id: CurrentUser, store: Store
) -> DocumentOut:
    filename = _clean_filename(file.filename)
    ext = PurePath(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(415, f"unsupported file type {ext or '(none)'}; allowed: {allowed}")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if not data:
        raise HTTPException(400, "empty file")
    # Check the content, not just the extension.
    if ext == ".pdf" and not data.startswith(b"%PDF-"):
        raise HTTPException(415, "file is not a PDF")

    sha256 = hashlib.sha256(data).hexdigest()
    if (existing := store.find_document(user_id, sha256)) is not None:
        raise HTTPException(409, {"message": "duplicate file", "document_id": existing})

    # Parsing and chunking are fast and reject bad files before we answer.
    # Only embedding (slow and rate-limited) runs in the background.
    with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            pages = load(tmp.name)
        except LoaderError as e:
            # Show the user's filename, not the random temp name.
            raise HTTPException(422, str(e).replace(Path(tmp.name).name, filename)) from e
    chunks = chunk_pages(pages)
    if not chunks:
        raise HTTPException(422, "no text found (a scanned PDF needs OCR, which is not supported)")

    try:
        doc_id = store.create_document(user_id, filename, sha256, len(pages))
    except psycopg.errors.UniqueViolation as e:
        # The same file uploaded twice at the same moment.
        raise HTTPException(409, {"message": "duplicate file"}) from e

    background.add_task(process_document, doc_id, user_id, chunks)
    return _out(store.get_document(user_id, doc_id))


@router.get("", response_model=list[DocumentOut])
def list_documents(user_id: CurrentUser, store: Store) -> list[DocumentOut]:
    return [DocumentOut(**asdict(d)) for d in store.list_documents(user_id)]


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: int, user_id: CurrentUser, store: Store) -> DocumentOut:
    return _out(store.get_document(user_id, document_id))


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: int, user_id: CurrentUser, store: Store) -> Response:
    # 404 for other users' documents too, so ids of others' files are not revealed.
    if not store.delete_document(user_id, document_id):
        raise HTTPException(404, "document not found")
    return Response(status_code=204)
