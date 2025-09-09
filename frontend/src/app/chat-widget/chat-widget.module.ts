import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatWidgetComponent } from './chat-widget.component';

@NgModule({
  declarations: [ChatWidgetComponent],
  imports: [CommonModule, FormsModule],
  exports: [ChatWidgetComponent]
})
export class ChatWidgetModule {}
