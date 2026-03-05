import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
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
  private paperContext = inject(PaperContextService);
  private openAiApi = inject(OpenAiApiService);

  // Matches your signature
  public sendMessage(message: string): Observable<any> {
    const activePaper = this.paperContext.getActivePaper();
    return this.openAiApi.sendMessage(message, activePaper?.id ?? null);
  }
}
