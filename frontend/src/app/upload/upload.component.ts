import { Component, OnInit } from '@angular/core';
import { HttpEventType } from '@angular/common/http';
import {
  OpenAiApiService,
  DocumentMeta,
  DocumentListResponse,
  VideoMeta,
  VideoListResponse
} from '../services/open-ai-api.service';

@Component({
  selector: 'app-upload',
  templateUrl: './upload.component.html',
  styleUrls: ['./upload.component.css'],
})
export class UploadComponent implements OnInit {
  pdfFile: File | null = null;
  pdfUploading = false;
  pdfProgress = 0;
  pdfMessage = '';
  pdfError = '';
  dragOverPdf = false;

  docs: DocumentMeta[] = [];
  docsLoading = false;
  pageDocs = 1;
  pageSizeDocs = 10;
  totalDocs = 0;
  qDocs = '';

  videoFile: File | null = null;
  videoUploading = false;
  videoProgress = 0;
  videoMessage = '';
  videoError = '';
  dragOverVideo = false;

  videos: VideoMeta[] = [];
  videosLoading = false;
  pageVideos = 1;
  pageSizeVideos = 10;
  totalVideos = 0;
  qVideos = '';

  constructor(private api: OpenAiApiService) {}

  ngOnInit(): void {
    this.loadDocuments(1);
    this.loadVideos(1);
  }

  onPdfFileSelected(evt: Event) {
    const input = evt.target as HTMLInputElement;
    const picked = input.files?.[0] || null;
    if (picked) this.setPdfFile(picked);
    if (input) input.value = '';
  }

  onPdfDragOver(e: DragEvent) {
    e.preventDefault();
    this.dragOverPdf = true;
  }
  onPdfDragLeave(_: DragEvent) {
    this.dragOverPdf = false;
  }
  onPdfDrop(e: DragEvent) {
    e.preventDefault();
    this.dragOverPdf = false;
    const dropped = e.dataTransfer?.files?.[0] || null;
    if (dropped) this.setPdfFile(dropped);
  }

  setPdfFile(f: File) {
    this.pdfMessage = '';
    this.pdfError = '';
    if (f.type !== 'application/pdf') {
      this.pdfError = 'Only PDF files are allowed.';
      this.pdfFile = null;
      return;
    }
    const MAX_MB = 50;
    if (f.size > MAX_MB * 1024 * 1024) {
      this.pdfError = `File is too large. Max ${MAX_MB} MB.`;
      this.pdfFile = null;
      return;
    }
    this.pdfFile = f;
  }

  clearPdfFile() {
    if (this.pdfUploading) return;
    this.pdfFile = null;
    this.pdfMessage = '';
    this.pdfError = '';
    this.pdfProgress = 0;
  }

  uploadPdf() {
    if (!this.pdfFile || this.pdfUploading) return;
    this.pdfUploading = true;
    this.pdfProgress = 0;
    this.pdfMessage = '';
    this.pdfError = '';

    this.api.uploadDocumentWithProgress(this.pdfFile).subscribe({
      next: (event) => {
        if (event.type === HttpEventType.UploadProgress && event.total) {
          this.pdfProgress = Math.round((event.loaded / event.total) * 100);
        } else if (event.type === HttpEventType.Response) {
          this.pdfUploading = false;
          this.pdfMessage = event.body?.message ?? 'Uploaded.';
          this.pdfFile = null;
          this.loadDocuments(1);
        }
      },
      error: () => {
        this.pdfUploading = false;
        this.pdfError = 'Upload failed. Please try again.';
      },
    });
  }

  loadDocuments(page: number = this.pageDocs) {
    this.docsLoading = true;
    this.api.getDocuments(page, this.pageSizeDocs, this.qDocs).subscribe({
      next: (res: DocumentListResponse) => {
        this.docs = res.items || [];
        this.totalDocs = res.total || 0;
        this.pageDocs = res.page || page;
        this.pageSizeDocs = res.page_size || this.pageSizeDocs;
        this.docsLoading = false;
      },
      error: () => {
        this.docs = [];
        this.totalDocs = 0;
        this.docsLoading = false;
      }
    });
  }

  onPdfSearchEnter() {
    this.loadDocuments(1);
  }

  prevDocPage() {
    if (this.pageDocs > 1) this.loadDocuments(this.pageDocs - 1);
  }
  nextDocPage() {
    const maxPage = Math.max(1, Math.ceil(this.totalDocs / this.pageSizeDocs));
    if (this.pageDocs < maxPage) this.loadDocuments(this.pageDocs + 1);
  }

  viewDoc(doc: DocumentMeta) {
    window.open(this.api.docViewUrl(doc.id), '_blank');
  }
  downloadDoc(doc: DocumentMeta) {
    window.open(this.api.docDownloadUrl(doc.id), '_blank');
  }

  deleteDoc(doc: DocumentMeta) {
    if (this.pdfUploading) return;
    if (!confirm(`Delete document "${doc.file_name}"? This action cannot be undone.`)) return;
    this.pdfUploading = true;
    this.api.docDeleteUrl(doc.id).subscribe({
      next: () => {
        this.pdfUploading = false;
        this.loadDocuments(this.pageDocs);
      },
      error: () => {
        this.pdfUploading = false;
        alert('Failed to delete document. Please try again.');
      }
    });
  }

  onVideoFileSelected(evt: Event) {
    const input = evt.target as HTMLInputElement;
    const picked = input.files?.[0] || null;
    if (picked) this.setVideoFile(picked);
    if (input) input.value = '';
  }

  onVideoDragOver(e: DragEvent) {
    e.preventDefault();
    this.dragOverVideo = true;
  }
  onVideoDragLeave(_: DragEvent) {
    this.dragOverVideo = false;
  }
  onVideoDrop(e: DragEvent) {
    e.preventDefault();
    this.dragOverVideo = false;
    const dropped = e.dataTransfer?.files?.[0] || null;
    if (dropped) this.setVideoFile(dropped);
  }

  setVideoFile(f: File) {
    this.videoMessage = '';
    this.videoError = '';

    const allowed = [
      'video/mp4',
      'video/quicktime',
      'video/x-matroska',
      'video/webm',
      'video/mpeg'
    ];
    if (!allowed.includes(f.type)) {
      this.videoError = 'Only video files (.mp4, .mov, .mkv, .webm, .mpeg) are allowed.';
      this.videoFile = null;
      return;
    }
    const MAX_MB = 1024 * 2;
    if (f.size > MAX_MB * 1024 * 1024) {
      this.videoError = `File is too large. Max ${MAX_MB} MB.`;
      this.videoFile = null;
      return;
    }
    this.videoFile = f;
  }

  clearVideoFile() {
    if (this.videoUploading) return;
    this.videoFile = null;
    this.videoMessage = '';
    this.videoError = '';
    this.videoProgress = 0;
  }

  uploadVideo() {
    if (!this.videoFile || this.videoUploading) return;
    this.videoUploading = true;
    this.videoProgress = 0;
    this.videoMessage = '';
    this.videoError = '';

    this.api.uploadVideoWithProgress(this.videoFile).subscribe({
      next: (event) => {
        if (event.type === HttpEventType.UploadProgress && event.total) {
          this.videoProgress = Math.round((event.loaded / event.total) * 100);
        } else if (event.type === HttpEventType.Response) {
          this.videoUploading = false;
          this.videoMessage = event.body?.message ?? 'Uploaded.';
          this.videoFile = null;
          this.loadVideos(1);
        }
      },
      error: () => {
        this.videoUploading = false;
        this.videoError = 'Upload failed. Please try again.';
      },
    });
  }

  loadVideos(page: number = this.pageVideos) {
    this.videosLoading = true;
    this.api.getVideos(page, this.pageSizeVideos, this.qVideos).subscribe({
      next: (res: VideoListResponse) => {
        this.videos = res.items || [];
        this.totalVideos = res.total || 0;
        this.pageVideos = res.page || page;
        this.pageSizeVideos = res.page_size || this.pageSizeVideos;
        this.videosLoading = false;
      },
      error: () => {
        this.videos = [];
        this.totalVideos = 0;
        this.videosLoading = false;
      }
    });
  }

  onVideoSearchEnter() {
    this.loadVideos(1);
  }

  prevVideoPage() {
    if (this.pageVideos > 1) this.loadVideos(this.pageVideos - 1);
  }
  nextVideoPage() {
    const maxPage = Math.max(1, Math.ceil(this.totalVideos / this.pageSizeVideos));
    if (this.pageVideos < maxPage) this.loadVideos(this.pageVideos + 1);
  }

  viewVideo(v: VideoMeta) {
    window.open(this.api.videoViewUrl(v.id), '_blank');
  }
  downloadVideo(v: VideoMeta) {
    window.open(this.api.videoDownloadUrl(v.id), '_blank');
  }

  deleteVideo(v: VideoMeta) {
    if (!confirm(`Delete video "${v.file_name}"? This action cannot be undone.`)) return;
    this.api.videoDeleteUrl(v.id).subscribe({
      next: () => {
        this.loadVideos(this.pageVideos);
      },
      error: () => {
        alert('Failed to delete video. Please try again.');
      }
    });
  }

  formatSize(bytes: number) {
    if (bytes < 1024) return `${bytes} B`;
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(1)} KB`;
    return `${(kb / 1024).toFixed(1)} MB`;
  }

  get totalDocPages(): number {
    if (!this.totalDocs || !this.pageSizeDocs) return 1;
    return Math.max(1, Math.ceil(this.totalDocs / this.pageSizeDocs));
  }

  get totalVideoPages(): number {
    if (!this.totalVideos || !this.pageSizeVideos) return 1;
    return Math.max(1, Math.ceil(this.totalVideos / this.pageSizeVideos));
  }
}
