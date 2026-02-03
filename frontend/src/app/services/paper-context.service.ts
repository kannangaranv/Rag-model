import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';

export interface ActivePaperContext {
  id: string;
  name: string;
}

@Injectable({ providedIn: 'root' })
export class PaperContextService {
  private activePaperSubject = new BehaviorSubject<ActivePaperContext | null>(null);

  setActivePaper(paper: ActivePaperContext) {
    this.activePaperSubject.next(paper);
  }

  clearActivePaper() {
    this.activePaperSubject.next(null);
  }

  getActivePaper(): ActivePaperContext | null {
    return this.activePaperSubject.value;
  }

  activePaper$(): Observable<ActivePaperContext | null> {
    return this.activePaperSubject.asObservable();
  }
}
