import { Component } from '@angular/core';
import { AuthService, AuthUser } from '../auth/auth.service';

@Component({
  selector: 'app-layout',
  templateUrl: './layout.component.html',
  styleUrls: ['./layout.component.css'],
})
export class LayoutComponent {
  collapsed = false;
  me: AuthUser | null = this.auth.getStoredUser();

  constructor(private auth: AuthService) {
    this.auth.currentUser$().subscribe(u => this.me = u);
  }

  logout() {
    this.auth.logout();
  }
}
