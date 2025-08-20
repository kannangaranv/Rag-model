import { Component, OnInit } from '@angular/core';
import { HttpEventType } from '@angular/common/http';
import { OpenAiApiService, DocumentMeta, DocumentListResponse } from '../services/open-ai-api.service';

@Component({
  selector: 'app-upload',
  templateUrl: './upload.component.html',
  styleUrls: ['./upload.component.css'],
})
export class UploadComponent implements OnInit {
  file: File | null = null;
  uploading = false;
  progress = 0;
  message = '';
  error = '';
  dragOver = false;

  docs: DocumentMeta[] = [];
  docsLoading = false;
  page = 1;
  pageSize = 10;
  total = 0;
  q = ''; 

  constructor(private api: OpenAiApiService) {}

  ngOnInit(): void {
    this.loadDocuments(1);
  }

  onFileSelected(evt: Event) {
    const input = evt.target as HTMLInputElement;
    const picked = input.files?.[0] || null;
    if (picked) this.setFile(picked);
    if (input) input.value = ''; 
  }

  onDragOver(e: DragEvent) {
    e.preventDefault();
    this.dragOver = true;
  }

  onDragLeave(_: DragEvent) {
    this.dragOver = false;
  }

  onDrop(e: DragEvent) {
    e.preventDefault();
    this.dragOver = false;
    const dropped = e.dataTransfer?.files?.[0] || null;
    if (dropped) this.setFile(dropped);
  }

  setFile(f: File) {
    this.message = '';
    this.error = '';
    if (f.type !== 'application/pdf') {
      this.error = 'Only PDF files are allowed.';
      this.file = null;
      return;
    }
    const MAX_MB = 50;
    if (f.size > MAX_MB * 1024 * 1024) {
      this.error = `File is too large. Max ${MAX_MB} MB.`;
      this.file = null;
      return;
    }
    this.file = f;
  }

  clearFile() {
    if (this.uploading) return;
    this.file = null;
    this.message = '';
    this.error = '';
    this.progress = 0;
  }

  upload() {
    if (!this.file || this.uploading) return;
    this.uploading = true;
    this.progress = 0;
    this.message = '';
    this.error = '';

    this.api.uploadDocumentWithProgress(this.file).subscribe({
      next: (event) => {
        if (event.type === HttpEventType.UploadProgress && event.total) {
          this.progress = Math.round((event.loaded / event.total) * 100);
        } else if (event.type === HttpEventType.Response) {
          this.uploading = false;
          this.message = event.body?.message ?? 'Uploaded.';
          this.file = null;
          this.loadDocuments(1);
        }
      },
      error: () => {
        this.uploading = false;
        this.error = 'Upload failed. Please try again.';
      },
    });
  }

  loadDocuments(page: number = this.page) {
    this.docsLoading = true;
    this.api.getDocuments(page, this.pageSize, this.q).subscribe({
      next: (res: DocumentListResponse) => {
        this.docs = res.items || [];
        this.total = res.total || 0;
        this.page = res.page || page;
        this.pageSize = res.page_size || this.pageSize;
        this.docsLoading = false;
      },
      error: () => {
        this.docs = [];
        this.total = 0;
        this.docsLoading = false;
      }
    });
  }

  onSearchEnter() {
    this.loadDocuments(1);
  }

  prevPage() {
    if (this.page > 1) this.loadDocuments(this.page - 1);
  }
  nextPage() {
    const maxPage = Math.max(1, Math.ceil(this.total / this.pageSize));
    if (this.page < maxPage) this.loadDocuments(this.page + 1);
  }

  view(doc: DocumentMeta) {
    window.open(this.api.docViewUrl(doc.id), '_blank');
  }

  download(doc: DocumentMeta) {
    window.open(this.api.docDownloadUrl(doc.id), '_blank');
  }

  isInVectorStore(doc: DocumentMeta) {
    if (typeof doc.in_vector_store === 'boolean') return doc.in_vector_store;
    if (typeof doc.has_md_text === 'boolean') return doc.has_md_text;
    return false; 
  }

  formatSize(bytes: number) {
    if (bytes < 1024) return `${bytes} B`;
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(1)} KB`;
    return `${(kb / 1024).toFixed(1)} MB`;
  }

    get totalPages(): number {
      if (!this.total || !this.pageSize) return 1;
      return Math.max(1, Math.ceil(this.total / this.pageSize));
    }

}
