// Top-level destinations (vision §11). `phase` is the build phase (vision §VI) in
// which the destination's signature interface is built; `href` is set only once a
// destination has real content, so navigation never leads to a thin page.

import type { Tone } from './labels';

export interface Destination {
  id: string;
  title: string;
  /** Short label for the header navigation. */
  navTitle: string;
  question: string;
  signature: string;
  phase: 1 | 2 | 3 | 4 | 5 | 6;
  tone: Tone;
  href?: string;
  /** Shown in the primary header navigation. */
  inNav: boolean;
}

export const DESTINATIONS: Destination[] = [
  {
    id: 'physics-lab',
    title: 'Physics Lab',
    navTitle: 'Physics Lab',
    question: 'What physics are we approximating?',
    signature: '3D experiments with linked plots and linked-symbol equations',
    phase: 2,
    tone: 'phys',
    href: '/physics-lab/',
    inNav: true,
  },
  {
    id: 'observatory',
    title: 'Observatory',
    navTitle: 'Observatory',
    question: 'What do real models actually predict here?',
    signature: 'Several real checkpoints on the same structure, with their disagreement made visible',
    phase: 4,
    tone: 'phys',
    inNav: true,
  },
  {
    id: 'models',
    title: 'Models',
    navTitle: 'Models',
    question: 'How does a model family work, and what is each checkpoint?',
    signature: 'Monographs, Architecture X-ray, checkpoint cards, side-by-side comparison',
    phase: 3,
    tone: 'arch',
    inNav: true,
  },
  {
    id: 'data',
    title: 'Data',
    navTitle: 'Data',
    question: 'What did models learn from?',
    signature: 'Dataset atlas, element × level-of-theory coverage, lineage',
    phase: 3,
    tone: 'arch',
    inNav: true,
  },
  {
    id: 'benchmarks',
    title: 'Benchmarks',
    navTitle: 'Benchmarks',
    question: 'What has been measured, and how?',
    signature: 'Protocol-aware result explorer — never a universal ranking',
    phase: 3,
    tone: 'lit',
    inNav: true,
  },
  {
    id: 'failures',
    title: 'Failure Atlas',
    navTitle: 'Failures',
    question: 'Where and why do models fail?',
    signature: 'Evidence-graded case studies with diagnostics',
    phase: 5,
    tone: 'fail',
    inNav: true,
  },
  {
    id: 'explore',
    title: 'Explore',
    navTitle: 'Explore',
    question: 'How is the field connected?',
    signature: '2D atlas map, timeline, ⌘K search',
    phase: 3,
    tone: 'neutral',
    inNav: false,
  },
  {
    id: 'applications',
    title: 'Applications',
    navTitle: 'Applications',
    question: 'What matters for my system?',
    signature: 'Scroll-driven 3D research journeys, starting with electrochemical interfaces',
    phase: 5,
    tone: 'phys',
    inNav: false,
  },
  {
    id: 'papers',
    title: 'Papers',
    navTitle: 'Papers',
    question: 'Where did ideas come from?',
    signature: 'Typed citation graph, chronology, paper records',
    phase: 3,
    tone: 'lit',
    inNav: false,
  },
  {
    id: 'learn',
    title: 'Learn',
    navTitle: 'Learn',
    question: 'Where should I begin?',
    signature: 'Pathways at three depths: intuition, technical, research',
    phase: 2,
    tone: 'neutral',
    inNav: false,
  },
  {
    id: 'method',
    title: 'Method',
    navTitle: 'Method',
    question: 'How is the Atlas verified?',
    signature: 'Labels, editorial rules, compute tiers, design system',
    phase: 1,
    tone: 'neutral',
    href: '/method/',
    inNav: true,
  },
];
