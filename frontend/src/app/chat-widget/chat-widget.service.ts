import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import { AuthService } from '../auth/auth.service';
import { PaperContextService } from '../services/paper-context.service';
import { OpenAiApiService } from '../services/open-ai-api.service';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  ts: number;
}

@Injectable({ providedIn: 'root' })
export class ChatWidgetService {
  private auth = inject(AuthService);
  private paperContext = inject(PaperContextService);
  private openAiApi = inject(OpenAiApiService);

  // Matches your signature
  public sendMessage(message: string): Observable<any> {
    const activePaper = this.paperContext.getActivePaper();
    const level = this.auth.getStoredUser()?.level ?? 6;
    return this.openAiApi.sendMessage(message, level, activePaper?.id ?? null);
  }
}
