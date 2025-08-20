import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { environment } from 'src/environments/enviornment';
import { HttpParams } from '@angular/common/http';

export interface DocumentMeta {
  id: string;
  file_name: string;
  content_type: string;
  file_size_bytes: number;
  uploaded_at: string;          
  has_md_text?: boolean;       
  in_vector_store?: boolean;   
}

export interface DocumentListResponse {
  items: DocumentMeta[];
  total: number;
  page: number;
  page_size: number;
}

@Injectable({
  providedIn: 'root'
})
export class OpenAiApiService {

  private apiUrl = environment.apiUrl; 

  constructor(private http: HttpClient) { }

  public sendMessage(message: string) {
    return this.http.post<any>(`${this.apiUrl}/query`, { query: message });
  }

  uploadDocument(file: File) {
      const form = new FormData();
      form.append('file', file);            
      return this.http.post<{ message: string }>(
        `${this.apiUrl}/upload-documents`,
        form
      );
    }

  uploadDocumentWithProgress(file: File) {
    const form = new FormData();
    form.append('file', file);
    return this.http.post<{ message: string }>(
      `${this.apiUrl}/upload-documents`,
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

}