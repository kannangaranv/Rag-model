import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';

import { AppComponent } from './app.component';
import { ChatbotComponent } from './chatbot/chatbot.component';
import { FormsModule } from '@angular/forms';
import { UploadComponent } from './upload/upload.component';
import { LayoutComponent } from './layout/layout.component';
import { RouterModule, Routes } from '@angular/router';

const routes: Routes = [
  {
    path: '',
    component: LayoutComponent,
    children: [
      { path: 'chat', component: ChatbotComponent, title: 'AI Assistant' },
      { path: 'knowledge-base', component: UploadComponent, title: 'Knowledge Base' },
      { path: '', redirectTo: 'chat', pathMatch: 'full' },
    ],
  },
  { path: '**', redirectTo: 'chat' },
];

@NgModule({
  declarations: [
    AppComponent,
    ChatbotComponent,
    UploadComponent,
    LayoutComponent
  ],
  imports: [
    RouterModule.forRoot(routes),
    BrowserModule,
    HttpClientModule,
    FormsModule
  ],
  exports: [RouterModule],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
