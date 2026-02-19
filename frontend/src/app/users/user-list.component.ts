import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from 'src/environments/enviornment';
import { AuthService } from '../auth/auth.service';

type UserMeta = { id: number; username: string; role?: string; level?: number; created_at?: string };

@Component({
  selector: 'app-user-list',
  templateUrl: './user-list.component.html',
  styleUrls: ['./user-list.component.css']
})
export class UserListComponent implements OnInit {
  loading = false;
  deletingUserId: number | null = null;
  error: string | null = null;
  users: UserMeta[] = [];
  meId: number | null = null;
  private api = environment.apiUrl;

  constructor(private http: HttpClient, private auth: AuthService) {}

  ngOnInit(): void {
    this.meId = this.auth.getStoredUser()?.id ?? null;
    this.fetch();
  }

  roleName(u: UserMeta): string {
    if (u.role) return u.role;
    return u.level ? `Level ${u.level}` : '';
  }

  fetch() {
    this.loading = true; this.error = null;
    this.http.get<{ items: UserMeta[] }>(`${this.api}/auth/users`).subscribe({
      next: (res) => { this.users = res.items; this.loading = false; },
      error: (err) => { this.error = err?.error?.detail || 'Failed to load users'; this.loading = false; }
    });
  }

  canDelete(u: UserMeta): boolean {
    return this.meId !== null && u.id !== this.meId;
  }

  deleteUser(u: UserMeta) {
    if (!this.canDelete(u) || this.deletingUserId !== null) return;
    if (!confirm(`Delete user "${u.username}"? This action cannot be undone.`)) return;

    this.deletingUserId = u.id;
    this.error = null;
    this.http.delete(`${this.api}/auth/users/${u.id}`).subscribe({
      next: () => {
        this.deletingUserId = null;
        this.fetch();
      },
      error: (err) => {
        this.deletingUserId = null;
        this.error = err?.error?.detail || 'Failed to delete user';
      }
    });
  }
}
