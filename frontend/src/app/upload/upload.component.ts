import { Component, OnInit } from '@angular/core';
import { HttpEventType } from '@angular/common/http';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import {
  OpenAiApiService,
  DocumentMeta,
  DocumentListResponse,
  VideoMeta,
  VideoListResponse,
  PaperMeta,
  PaperListResponse
} from '../services/open-ai-api.service';

@Component({
  selector: 'app-upload',
  templateUrl: './upload.component.html',
  styleUrls: ['./upload.component.css'],
})
export class UploadComponent implements OnInit {
  // Selected roles for tagging uploads
  docLevel: number = 2;
  videoLevel: number = 2;
  levelOptions = [
    { value: 2, label: 'Board Admin (2)' },
    { value: 3, label: 'Sys Admin (3)' },
    { value: 4, label: 'Organizer (4)' },
    { value: 5, label: 'Actionee (5)' },
    { value: 6, label: 'Invittee (6)' },
  ];
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

  paperLevel: number = 2;
  paperFile: File | null = null;
  paperUploading = false;
  paperProgress = 0;
  paperMessage = '';
  paperError = '';
  dragOverPaper = false;

  papers: PaperMeta[] = [];
  papersLoading = false;
  pagePapers = 1;
  pageSizePapers = 10;
  totalPapers = 0;
  qPapers = '';
  paperPreviewName = '';
  paperPreviewId = '';
  paperPreviewUrl: SafeResourceUrl | null = null;
  paperQuestion = '';
  paperAsking = false;
  paperAnswerHtml = '';
  paperAnswerError = '';

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

  constructor(private api: OpenAiApiService, private sanitizer: DomSanitizer) {}

  ngOnInit(): void {
    this.loadDocuments(1);
    this.loadPapers(1);
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

    this.api.uploadDocumentWithProgress(this.pdfFile, this.docLevel).subscribe({
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

  onPaperFileSelected(evt: Event) {
    const input = evt.target as HTMLInputElement;
    const picked = input.files?.[0] || null;
    if (picked) this.setPaperFile(picked);
    if (input) input.value = '';
  }

  onPaperDragOver(e: DragEvent) {
    e.preventDefault();
    this.dragOverPaper = true;
  }
  onPaperDragLeave(_: DragEvent) {
    this.dragOverPaper = false;
  }
  onPaperDrop(e: DragEvent) {
    e.preventDefault();
    this.dragOverPaper = false;
    const dropped = e.dataTransfer?.files?.[0] || null;
    if (dropped) this.setPaperFile(dropped);
  }

  setPaperFile(f: File) {
    this.paperMessage = '';
    this.paperError = '';
    if (f.type !== 'application/pdf') {
      this.paperError = 'Only PDF files are allowed.';
      this.paperFile = null;
      return;
    }
    const MAX_MB = 50;
    if (f.size > MAX_MB * 1024 * 1024) {
      this.paperError = `File is too large. Max ${MAX_MB} MB.`;
      this.paperFile = null;
      return;
    }
    this.paperFile = f;
  }

  clearPaperFile() {
    if (this.paperUploading) return;
    this.paperFile = null;
    this.paperMessage = '';
    this.paperError = '';
    this.paperProgress = 0;
  }

  uploadPaper() {
    if (!this.paperFile || this.paperUploading) return;
    this.paperUploading = true;
    this.paperProgress = 0;
    this.paperMessage = '';
    this.paperError = '';

    this.api.uploadPaperWithProgress(this.paperFile, this.paperLevel).subscribe({
      next: (event) => {
        if (event.type === HttpEventType.UploadProgress && event.total) {
          this.paperProgress = Math.round((event.loaded / event.total) * 100);
        } else if (event.type === HttpEventType.Response) {
          this.paperUploading = false;
          this.paperMessage = event.body?.message ?? 'Uploaded.';
          this.paperFile = null;
          this.loadPapers(1);
        }
      },
      error: () => {
        this.paperUploading = false;
        this.paperError = 'Upload failed. Please try again.';
      },
    });
  }

  loadPapers(page: number = this.pagePapers) {
    this.papersLoading = true;
    this.api.getPapers(page, this.pageSizePapers, this.qPapers).subscribe({
      next: (res: PaperListResponse) => {
        this.papers = res.items || [];
        this.totalPapers = res.total || 0;
        this.pagePapers = res.page || page;
        this.pageSizePapers = res.page_size || this.pageSizePapers;
        this.papersLoading = false;
      },
      error: () => {
        this.papers = [];
        this.totalPapers = 0;
        this.papersLoading = false;
      }
    });
  }

  onPaperSearchEnter() {
    this.loadPapers(1);
  }

  prevPaperPage() {
    if (this.pagePapers > 1) this.loadPapers(this.pagePapers - 1);
  }
  nextPaperPage() {
    const maxPage = Math.max(1, Math.ceil(this.totalPapers / this.pageSizePapers));
    if (this.pagePapers < maxPage) this.loadPapers(this.pagePapers + 1);
  }

  viewPaper(p: PaperMeta) {
    this.paperPreviewId = p.id;
    this.paperPreviewName = p.file_name;
    this.paperPreviewUrl = this.sanitizer.bypassSecurityTrustResourceUrl(`${this.api.paperViewUrl(p.id)}#toolbar=1`);
    this.paperQuestion = '';
    this.paperAnswerHtml = '';
    this.paperAnswerError = '';
  }
  downloadPaper(p: PaperMeta) {
    window.open(this.api.paperDownloadUrl(p.id), '_blank');
  }

  deletePaper(p: PaperMeta) {
    if (!confirm(`Delete paper "${p.file_name}"? This action cannot be undone.`)) return;
    this.api.paperDeleteUrl(p.id).subscribe({
      next: () => {
        this.loadPapers(this.pagePapers);
      },
      error: () => {
        alert('Failed to delete paper. Please try again.');
      }
    });
  }

  closePaperPreview() {
    this.paperPreviewId = '';
    this.paperPreviewName = '';
    this.paperPreviewUrl = null;
    this.paperQuestion = '';
    this.paperAnswerHtml = '';
    this.paperAnswerError = '';
  }

  askAboutSelectedPaper() {
    const text = (this.paperQuestion || '').trim();
    if (!this.paperPreviewId || !text || this.paperAsking) return;

    this.paperAsking = true;
    this.paperAnswerError = '';
    this.api.sendPaperMessage(this.paperPreviewId, text).subscribe({
      next: (res) => {
        this.paperAnswerHtml = res?.response || '<p>No answer received.</p>';
        this.paperAsking = false;
      },
      error: () => {
        this.paperAnswerError = 'Failed to get answer. Please try again.';
        this.paperAsking = false;
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

    this.api.uploadVideoWithProgress(this.videoFile, this.videoLevel).subscribe({
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

  get totalPaperPages(): number {
    if (!this.totalPapers || !this.pageSizePapers) return 1;
    return Math.max(1, Math.ceil(this.totalPapers / this.pageSizePapers));
  }

  roleName(level?: number): string {
    switch (level) {
      case 1: return 'Admin';
      case 2: return 'Board Admin';
      case 3: return 'Sys Admin';
      case 4: return 'Organizer';
      case 5: return 'Actionee';
      case 6: return 'Invittee';
      default: return level ? `Level ${level}` : '';
    }
  }
}
