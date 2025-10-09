import { Injectable } from '@angular/core';
import { CanActivate, Router, UrlTree } from '@angular/router';
import { AuthService } from './auth.service';

@Injectable({ providedIn: 'root' })
export class NotAdminGuard implements CanActivate {
  constructor(private auth: AuthService, private router: Router) {}

  canActivate(): boolean | UrlTree {
    if (!this.auth.isAuthenticated()) return this.router.parseUrl('/login');
    if (!this.auth.isAdmin()) return true;
    // admins go to knowledge base
    return this.router.parseUrl('/knowledge-base');
  }
}

