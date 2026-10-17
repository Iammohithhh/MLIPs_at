// Camera synchronisation for side-by-side scenes (inside-mace; not in the Atlas). Each <Stage> given the
// same CameraSync publishes its orbit camera after user interaction and copies the others', so the
// baseline and perturbed panes always show the same view of the same coordinates.
export interface CameraPose {
  position: [number, number, number];
  target: [number, number, number];
}

export class CameraSync {
  private listeners = new Map<number, (p: CameraPose) => void>();
  private next = 0;
  subscribe(fn: (p: CameraPose) => void): { id: number; off: () => void } {
    const id = this.next++;
    this.listeners.set(id, fn);
    return { id, off: () => this.listeners.delete(id) };
  }
  publish(from: number, pose: CameraPose): void {
    for (const [id, fn] of this.listeners) if (id !== from) fn(pose);
  }
}
