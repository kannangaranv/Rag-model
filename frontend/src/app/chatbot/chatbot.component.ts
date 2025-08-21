import { Component, ElementRef, ViewChild } from '@angular/core';
import { OpenAiApiService } from '../services/open-ai-api.service';

type Role = 'user' | 'assistant';
interface ChatMessage {
  role: Role;
  content: string;
  isHtml?: boolean;
}

@Component({
  selector: 'app-chatbot',
  templateUrl: './chatbot.component.html',
  styleUrls: ['./chatbot.component.css']
})
export class ChatbotComponent {
  @ViewChild('msgPane') msgPane!: ElementRef<HTMLDivElement>;

  userMessage = '';
  isLoading = false;
  chatMessages: ChatMessage[] = [];

  constructor(private openAiApiService: OpenAiApiService) {}

  private extractHtml(payload: string): string {
    const m = payload?.match(/^```html\s*([\s\S]*?)\s*```$/i);
    return m ? m[1] : payload ?? '';
  }

  private scrollToBottom() {
    queueMicrotask(() => {
      const el = this.msgPane?.nativeElement;
      if (el) el.scrollTop = el.scrollHeight;
    });
  }

  onEnter(e: Event) {
  const ke = e as KeyboardEvent;
  ke.preventDefault();
  if (ke.key === 'Enter' && !ke.shiftKey) {
    this.sendMessage();
  }
}


  sendMessage() {
    const text = this.userMessage.trim();
    if (!text || this.isLoading) return;

    this.chatMessages.push({ role: 'user', content: text });
    this.scrollToBottom();

    this.isLoading = true;

    this.openAiApiService.sendMessage(text).subscribe({
      next: (res) => {
        const raw = res?.response ?? '';
        const html = this.extractHtml(raw);
        this.chatMessages.push({ role: 'assistant', content: html, isHtml: true });
        this.userMessage = '';
        this.isLoading = false;
        this.scrollToBottom();
      },
      error: () => {
        this.chatMessages.push({
          role: 'assistant',
          content: 'Sorry—something went wrong. Please try again.'
        });
        this.isLoading = false;
        this.scrollToBottom();
      }
    });
  }

  
}
