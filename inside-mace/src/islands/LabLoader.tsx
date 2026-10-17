// Lazy entry for the instrument: the page's first paint needs no Three.js and no experiment data; the
// island code, renderer and the 650 KB manifest arrive as a separate chunk (brief §15: lazy-load data).
import { lazy, Suspense } from 'react';
import './InterventionLab.css';

const InterventionLab = lazy(() => import('./InterventionLab'));

export default function LabLoader() {
  return (
    <Suspense
      fallback={
        <div className="lab lab--loading" role="status">
          Loading the precomputed evaluations…
        </div>
      }
    >
      <InterventionLab />
    </Suspense>
  );
}
