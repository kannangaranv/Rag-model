import {  HttpClient, HttpEvent } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { environment } from 'src/environments/enviornment';
import { HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface DocumentMeta {
  id: string;
  file_name: string;
  content_type: string;
  file_size_bytes: number;
  uploaded_at: string;          
  has_md_text?: boolean;       
  in_vector_store?: boolean;   
  level?: number;
}

export interface DocumentListResponse {
  items: DocumentMeta[];
  total: number;
  page: number;
  page_size: number;
}

export interface VideoMeta {
  id: string;
  file_name: string;
  file_size_bytes: number;
  uploaded_at: string;
  status?: 'pending' | 'processing' | 'processed' | 'failed';
  // optional fields if backend provides:
  duration_seconds?: number;
  thumbnail_url?: string;
  level?: number;
}

export interface VideoListResponse {
  items: VideoMeta[];
  page: number;
  page_size: number;
  total: number;
}

export interface PaperMeta {
  id: string;
  file_name: string;
  content_type: string;
  file_size_bytes: number;
  uploaded_at: string;
  has_md_text?: boolean;
  level?: number;
}

export interface PaperListResponse {
  items: PaperMeta[];
  page: number;
  page_size: number;
  total: number;
}

@Injectable({
  providedIn: 'root'
})
export class OpenAiApiService {

  private apiUrl = environment.apiUrl; 

  constructor(private http: HttpClient) { }

  public sendMessage(message: string, level: number) {
    return this.http.post<any>(`${this.apiUrl}/query/${level}`, { query: message });
  }

  public sendPaperMessage(paperId: string, message: string) {
    return this.http.post<any>(`${this.apiUrl}/papers/${paperId}/query`, { query: message });
  }

  uploadDocument(file: File, level: number) {
      const form = new FormData();
      form.append('file', file);            
      return this.http.post<{ message: string }>(
        `${this.apiUrl}/upload-documents/${level}`,
        form
      );
    }

  uploadDocumentWithProgress(file: File, level: number) {
    const form = new FormData();
    form.append('file', file);
    return this.http.post<{ message: string }>(
      `${this.apiUrl}/upload-documents/${level}`,
      form,
      { observe: 'events', reportProgress: true }
    );
  }

  getDocuments(page = 1, pageSize = 10, q?: string) {
      let params = new HttpParams()
        .set('page', String(page))
        .set('page_size', String(pageSize));
      if (q) params = params.set('q', q);  
      return this.http.get<DocumentListResponse>(`${this.apiUrl}/documents`, { params });
    }
  docViewUrl(id: string) {
      return `${this.apiUrl}/documents/${id}/view`;
    }

  docDownloadUrl(id: string) {
      return `${this.apiUrl}/documents/${id}/download`;
    }

  docDeleteUrl(id: string) {
      return this.http.delete(`${this.apiUrl}/documents/${id}`);
    }

  uploadVideoWithProgress(file: File, level: number): Observable<HttpEvent<any>> {
    const form = new FormData();
    form.append('file', file);
    return this.http.post<any>(`${this.apiUrl}/upload-videos/${level}`, form, {
      reportProgress: true,
      observe: 'events',
    });
  }

  getVideos(page = 1, pageSize = 10, q = ''): Observable<VideoListResponse> {
    const params = { page, page_size: pageSize, q };
    return this.http.get<VideoListResponse>(`${this.apiUrl}/videos`, { params: params as any });
  }

  videoViewUrl(id: string) {
    return `${this.apiUrl}/videos/${id}/view`;
  }

  videoDownloadUrl(id: string) {
    return `${this.apiUrl}/videos/${id}/download`;
  }

  videoDeleteUrl(id: string) {
    return this.http.delete(`${this.apiUrl}/videos/${id}`);
  } 

  uploadPaperWithProgress(file: File, level: number): Observable<HttpEvent<any>> {
    const form = new FormData();
    form.append('file', file);
    return this.http.post<any>(`${this.apiUrl}/upload-papers/${level}`, form, {
      reportProgress: true,
      observe: 'events',
    });
  }

  getPapers(page = 1, pageSize = 10, q = ''): Observable<PaperListResponse> {
    const params = { page, page_size: pageSize, q };
    return this.http.get<PaperListResponse>(`${this.apiUrl}/papers`, { params: params as any });
  }

  paperViewUrl(id: string) {
    return `${this.apiUrl}/papers/${id}/view`;
  }

  paperDownloadUrl(id: string) {
    return `${this.apiUrl}/papers/${id}/download`;
  }

  paperDeleteUrl(id: string) {
    return this.http.delete(`${this.apiUrl}/papers/${id}`);
  }

}
