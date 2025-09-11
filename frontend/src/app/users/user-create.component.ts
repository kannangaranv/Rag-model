import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from 'src/environments/enviornment';

@Component({
  selector: 'app-user-create',
  templateUrl: './user-create.component.html',
  styleUrls: ['./user-create.component.css']
})
export class UserCreateComponent {
  username = '';
  password = '';
  level = 6;
  saving = false;
  message: string | null = null;
  error: string | null = null;
  private api = environment.apiUrl;

  constructor(private http: HttpClient) {}

  submit() {
    this.message = this.error = null;
    if (!this.username || !this.password) {
      this.error = 'Username and password required';
      return;
    }
    this.saving = true;
    this.http.post(`${this.api}/auth/register`, {
      username: this.username,
      password: this.password,
      level: Number(this.level)
    }).subscribe({
      next: () => {
        this.saving = false;
        this.username = '';
        this.password = '';
        this.level = 6;
        this.message = 'User created successfully';
      },
      error: (err) => {
        this.saving = false;
        this.error = err?.error?.detail || 'Failed to create user';
      }
    })
  }
}
