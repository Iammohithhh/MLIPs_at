// One shared state for every linked view (brief §6). URL sync lives in the island.
import { create } from 'zustand';
import type { ExperimentState } from './urlState';

interface Store {
  s: ExperimentState | null;
  set: (patch: Partial<ExperimentState>) => void;
  init: (s: ExperimentState) => void;
}

export const useExperiment = create<Store>((set) => ({
  s: null,
  init: (s) => set({ s }),
  set: (patch) => set((st) => (st.s ? { s: { ...st.s, ...patch } } : st)),
}));
