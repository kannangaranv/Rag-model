import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { environment } from 'src/environments/enviornment';

@Injectable({
  providedIn: 'root'
})
export class OpenAiApiService {

  private apiUrl = environment.apiUrl; // Update with your Node.js server URL

  constructor(private http: HttpClient) { }

  public sendMessage(message: string) {
    return this.http.post<any>(`${this.apiUrl}/query`, { query: message });
  }

uploadDocument(file: File) {
    const form = new FormData();
    form.append('file', file); // key MUST be "file" to match backend
    return this.http.post<{ message: string }>(
      `${this.apiUrl}/upload-documents`,
      form
    );
  }
  // openai-api.service.ts
uploadDocumentWithProgress(file: File) {
  const form = new FormData();
  form.append('file', file);
  return this.http.post<{ message: string }>(
    `${this.apiUrl}/upload-documents`,
    form,
    { observe: 'events', reportProgress: true }
  );
}
}