import { Component, ElementRef, ViewChild, AfterViewInit } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { ChatWidgetService, ChatMessage } from './chat-widget.service';

@Component({
  selector: 'app-chat-widget',
  templateUrl: './chat-widget.component.html',
  styleUrls: ['./chat-widget.component.css']
})
export class ChatWidgetComponent implements AfterViewInit {
  open = false;
  input = '';
  loading = false;
  messages: ChatMessage[] = [];

  @ViewChild('messagesRef') messagesRef?: ElementRef<HTMLDivElement>;
  @ViewChild('inputRef') inputRef?: ElementRef<HTMLTextAreaElement>;

  constructor(private api: ChatWidgetService) {
    this.append('assistant', 'Hi! I’m your assistant. How can I help today?');
  }

  ngAfterViewInit(): void {}

  toggle(): void {
    this.open = !this.open;
    if (this.open) setTimeout(() => this.inputRef?.nativeElement?.focus(), 0);
  }

  restart(): void {
    this.messages = [];
    this.append('assistant', 'Chat reset. Ask me anything about our site.');
    this.input = '';
    setTimeout(() => this.scrollToBottom(), 0);
  }

  onChipFill(text: string): void {
    this.input = text;
    setTimeout(() => this.inputRef?.nativeElement?.focus(), 0);
  }

  async onSend(): Promise<void> {
    const text = (this.input || '').trim();
    if (!text || this.loading) return;

    this.append('user', text);
    this.input = '';
    this.loading = true;

    try {
      const res = await firstValueFrom(this.api.sendMessage(text));
      this.append('assistant', res.response);
    } catch (err) {
      console.error(err);
      this.append('assistant', '⚠️ Sorry, something went wrong. Please try again.');
    } finally {
      this.loading = false;
    }
  }

  private append(role: ChatMessage['role'], text: string): void {
    const id = (window.crypto && 'randomUUID' in window.crypto)
      ? window.crypto.randomUUID()
      : Math.random().toString(36).slice(2);
    const msg: ChatMessage = { id, role, text, ts: Date.now() };
    this.messages = [...this.messages, msg];
    setTimeout(() => this.scrollToBottom(), 0);
  }

  private scrollToBottom(): void {
    const el = this.messagesRef?.nativeElement;
    if (el) el.scrollTop = el.scrollHeight;
  }
}
