import { Component, OnDestroy, OnInit } from '@angular/core';
import { OpenAiApiService, PaperListResponse, PaperMeta } from '../services/open-ai-api.service';
import { AuthService } from '../auth/auth.service';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { PaperContextService } from '../services/paper-context.service';

@Component({
  selector: 'app-papers',
  templateUrl: './papers.component.html',
  styleUrls: ['./papers.component.css'],
})
export class PapersComponent implements OnInit, OnDestroy {
  papers: PaperMeta[] = [];
  loading = false;
  page = 1;
  pageSize = 10;
  total = 0;
  q = '';
  canUpload = false;
  previewPaper: PaperMeta | null = null;
  previewUrl: SafeResourceUrl | null = null;
  question = '';
  asking = false;
  answerHtml = '';
  answerError = '';

  constructor(
    private api: OpenAiApiService,
    private auth: AuthService,
    private sanitizer: DomSanitizer,
    private paperContext: PaperContextService
  ) {}

  ngOnInit(): void {
    this.canUpload = this.auth.canUpload();
    this.load(1);
  }

  load(page: number = this.page) {
    this.loading = true;
    this.api.getPapers(page, this.pageSize, this.q).subscribe({
      next: (res: PaperListResponse) => {
        this.papers = res.items || [];
        this.total = res.total || 0;
        this.page = res.page || page;
        this.pageSize = res.page_size || this.pageSize;
        this.loading = false;
      },
      error: () => {
        this.papers = [];
        this.total = 0;
        this.loading = false;
      }
    });
  }

  onSearchEnter() {
    this.load(1);
  }

  viewPaper(p: PaperMeta) {
    this.previewPaper = p;
    this.previewUrl = this.sanitizer.bypassSecurityTrustResourceUrl(`${this.api.paperViewUrl(p.id)}#toolbar=1`);
    this.paperContext.setActivePaper({ id: p.id, name: p.file_name });
    this.question = '';
    this.answerHtml = '';
    this.answerError = '';
  }

  closePreview() {
    this.previewPaper = null;
    this.previewUrl = null;
    this.paperContext.clearActivePaper();
    this.question = '';
    this.answerHtml = '';
    this.answerError = '';
  }

  ngOnDestroy(): void {
    this.paperContext.clearActivePaper();
  }

  askAboutCurrentPaper() {
    const text = (this.question || '').trim();
    if (!this.previewPaper || !text || this.asking) return;

    this.asking = true;
    this.answerError = '';
    this.api.sendPaperMessage(this.previewPaper.id, text).subscribe({
      next: (res) => {
        this.answerHtml = res?.response || '<p>No answer received.</p>';
        this.asking = false;
      },
      error: () => {
        this.answerError = 'Failed to get answer. Please try again.';
        this.asking = false;
      },
    });
  }

  downloadPaper(p: PaperMeta) {
    window.open(this.api.paperDownloadUrl(p.id), '_blank');
  }

  deletePaper(p: PaperMeta) {
    if (!this.canUpload) return;
    if (!confirm(`Delete paper "${p.file_name}"? This action cannot be undone.`)) return;
    this.api.paperDeleteUrl(p.id).subscribe({
      next: () => this.load(this.page),
      error: () => alert('Failed to delete paper. Please try again.')
    });
  }

  prevPage() {
    if (this.page > 1) this.load(this.page - 1);
  }

  nextPage() {
    if (this.page < this.totalPages) this.load(this.page + 1);
  }

  get totalPages(): number {
    if (!this.total || !this.pageSize) return 1;
    return Math.max(1, Math.ceil(this.total / this.pageSize));
  }

  formatSize(bytes: number) {
    if (bytes < 1024) return `${bytes} B`;
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(1)} KB`;
    return `${(kb / 1024).toFixed(1)} MB`;
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
