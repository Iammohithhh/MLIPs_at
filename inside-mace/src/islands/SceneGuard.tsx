// Static fallback when WebGL is unavailable or the 3D scene fails at runtime. The measured values,
// plots and channel inspector do not depend on WebGL and stay usable.
import { Component, useEffect, useState, type ReactNode } from 'react';

export function hasWebGL(): boolean {
  try {
    const c = document.createElement('canvas');
    return !!(c.getContext('webgl2') ?? c.getContext('webgl'));
  } catch {
    return false;
  }
}

function Fallback({ reason }: { reason: string }) {
  return (
    <div className="scene-fallback" role="note">
      <p>
        <strong>3D view unavailable.</strong> {reason}
      </p>
      <p>The displacement control, measured energies and forces, plots and channel inspector below still show the evaluated frames.</p>
    </div>
  );
}

class Boundary extends Component<{ children: ReactNode }, { error: string | null }> {
  override state = { error: null as string | null };
  static getDerivedStateFromError(e: unknown) {
    return { error: String(e) };
  }
  override render() {
    return this.state.error ? <Fallback reason={`The scene failed to render (${this.state.error}).`} /> : this.props.children;
  }
}

export function SceneGuard({ children }: { children: ReactNode }) {
  const [ok, setOk] = useState<boolean | null>(null);
  useEffect(() => setOk(hasWebGL()), []);
  if (ok === null) return <div className="scene-fallback" role="status">Preparing the 3D view…</div>;
  if (!ok) return <Fallback reason="This browser does not provide WebGL, which the atomic scene needs." />;
  return <Boundary>{children}</Boundary>;
}
