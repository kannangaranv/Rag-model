import { Component } from '@angular/core';
import { HttpEventType } from '@angular/common/http';
import { OpenAiApiService } from '../services/open-ai-api.service';

@Component({
  selector: 'app-upload',
  templateUrl: './upload.component.html',
  styleUrls: ['./upload.component.css'],
})
export class UploadComponent {
  file: File | null = null;
  uploading = false;
  progress = 0;
  message = '';
  error = '';
  dragOver = false;

  constructor(private api: OpenAiApiService) {}

  onFileSelected(evt: Event) {
    const input = evt.target as HTMLInputElement;
    const picked = input.files?.[0] || null;
    if (picked) this.setFile(picked);
    if (input) input.value = ''; // allow re-selecting same file
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
    // validations
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
        }
      },
      error: () => {
        this.uploading = false;
        this.error = 'Upload failed. Please try again.';
      },
    });
  }

  formatSize(bytes: number) {
    if (bytes < 1024) return `${bytes} B`;
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(1)} KB`;
    return `${(kb / 1024).toFixed(1)} MB`;
    }
}
