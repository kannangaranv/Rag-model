import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from 'src/environments/enviornment';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  ts: number;
}

@Injectable({ providedIn: 'root' })
export class ChatWidgetService {
  private http = inject(HttpClient);
  private apiUrl = environment.apiUrl;

  // Matches your signature
  public sendMessage(message: string): Observable<any> {
    return this.http.post<any>(`${this.apiUrl}/query`, { query: message });
  }
}
