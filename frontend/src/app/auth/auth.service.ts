import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
import { BehaviorSubject, Observable, tap, switchMap } from 'rxjs';
import { environment } from 'src/environments/enviornment';

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface AuthUser {
  id: number;
  username: string;
  level: number;
  can_upload?: boolean;
}

@Injectable({ providedIn: 'root' })
export class AuthService {
  private readonly tokenKey = 'auth_token';
  private readonly userKey = 'auth_user';
  private api = environment.apiUrl;

  private user$ = new BehaviorSubject<AuthUser | null>(this.getStoredUser());

  constructor(private http: HttpClient, private router: Router) {}

  fetchMe(): Observable<AuthUser> {
    return this.http.get<AuthUser>(`${this.api}/auth/me`).pipe(
      tap((u) => this.setUser(u))
    );
  }

  login(username: string, password: string): Observable<AuthUser> {
    return this.http
      .post<TokenResponse>(`${this.api}/auth/login`, { username, password })
      .pipe(
        tap((res) => this.setToken(res.access_token)),
        switchMap(() => this.fetchMe())
      );
  }

  logout(): void {
    localStorage.removeItem(this.tokenKey);
    localStorage.removeItem(this.userKey);
    this.user$.next(null);
    this.router.navigate(['/login']);
  }

  setToken(token: string) {
    localStorage.setItem(this.tokenKey, token);
  }

  getToken(): string | null {
    return localStorage.getItem(this.tokenKey);
  }

  setUser(user: AuthUser) {
    localStorage.setItem(this.userKey, JSON.stringify(user));
    this.user$.next(user);
  }

  getStoredUser(): AuthUser | null {
    const raw = localStorage.getItem(this.userKey);
    if (!raw) return null;
    try {
      return JSON.parse(raw) as AuthUser;
    } catch {
      return null;
    }
  }

  currentUser$(): Observable<AuthUser | null> {
    return this.user$.asObservable();
  }

  isAuthenticated(): boolean {
    const token = this.getToken();
    if (!token) return false;
    const payload = this.decodeJwt(token);
    if (!payload?.exp) return true; 
    const now = Math.floor(Date.now() / 1000);
    return payload.exp > now;
  }

  isAdmin(): boolean {
    const u = this.getStoredUser();
    return !!u && u.level === 1;
  }

  canUpload(): boolean {
    const u = this.getStoredUser();
    return !!u && !!u.can_upload;
  }

  private decodeJwt(token: string): any | null {
    try {
      const parts = token.split('.');
      if (parts.length !== 3) return null;
      const payload = atob(parts[1].replace(/-/g, '+').replace(/_/g, '/'));
      return JSON.parse(payload);
    } catch {
      return null;
    }
  }
}
