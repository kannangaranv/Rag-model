import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from 'src/environments/enviornment';

type UserMeta = { id: number; username: string; level: number; created_at?: string };

@Component({
  selector: 'app-user-list',
  templateUrl: './user-list.component.html',
  styleUrls: ['./user-list.component.css']
})
export class UserListComponent implements OnInit {
  loading = false;
  error: string | null = null;
  users: UserMeta[] = [];
  private api = environment.apiUrl;

  constructor(private http: HttpClient) {}

  ngOnInit(): void {
    this.fetch();
  }

  roleName(level: number): string {
    switch (level) {
      case 1: return 'Admin';
      case 2: return 'Board Admin';
      case 3: return 'Sys Admin';
      case 4: return 'Organizer';
      case 5: return 'Actionee';
      case 6: return 'Invittee';
      default: return `Unknown (${level})`;
    }
  }

  fetch() {
    this.loading = true; this.error = null;
    this.http.get<{ items: UserMeta[] }>(`${this.api}/auth/users`).subscribe({
      next: (res) => { this.users = res.items; this.loading = false; },
      error: (err) => { this.error = err?.error?.detail || 'Failed to load users'; this.loading = false; }
    });
  }
}

