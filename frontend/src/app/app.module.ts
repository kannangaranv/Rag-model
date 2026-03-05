import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule, HTTP_INTERCEPTORS } from '@angular/common/http';

import { AppComponent } from './app.component';
import { ChatbotComponent } from './chatbot/chatbot.component';
import { FormsModule } from '@angular/forms';
import { UploadComponent } from './upload/upload.component';
import { LayoutComponent } from './layout/layout.component';
import { RouterModule, Routes } from '@angular/router';
import { ChatWidgetModule } from './chat-widget/chat-widget.module';
import { AuthGuard } from './auth/auth.guard';
import { AdminGuard } from './auth/admin.guard';
import { NotAdminGuard } from './auth/not-admin.guard';
import { AuthInterceptor } from './auth/auth.interceptor';
import { LoginComponent } from './login/login.component';
import { UserCreateComponent } from './users/user-create.component';
import { UserListComponent } from './users/user-list.component';
import { UploadGuard } from './auth/upload.guard';
import { PapersComponent } from './papers/papers.component';
const routes: Routes = [
  {
    path: '',
    component: LayoutComponent,
    canActivate: [AuthGuard],
    children: [
      { path: 'chat', component: ChatbotComponent, title: 'AI Assistant', canActivate: [NotAdminGuard] },
      { path: 'knowledge-base', component: UploadComponent, title: 'Knowledge Base', canActivate: [UploadGuard] },
      { path: 'papers', component: PapersComponent, title: 'Papers' },
      { path: 'users', component: UserListComponent, title: 'Users', canActivate: [AdminGuard] },
      { path: 'users/create', component: UserCreateComponent, title: 'Create User', canActivate: [AdminGuard] },
      { path: '', redirectTo: 'chat', pathMatch: 'full' },
    ],
  },
  { path: 'login', component: LoginComponent, title: 'Login' },
  { path: '**', redirectTo: 'chat' },
];

@NgModule({
  declarations: [
    AppComponent,
    ChatbotComponent,
    UploadComponent,
    LayoutComponent,
    LoginComponent,
    UserCreateComponent,
    UserListComponent,
    PapersComponent
  ],
  imports: [
    RouterModule.forRoot(routes),
    BrowserModule,
    HttpClientModule,
    FormsModule,
    ChatWidgetModule
  ],
  exports: [RouterModule],
  providers: [
    AuthGuard,
    AdminGuard,
    NotAdminGuard,
    UploadGuard,
    { provide: HTTP_INTERCEPTORS, useClass: AuthInterceptor, multi: true },
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
