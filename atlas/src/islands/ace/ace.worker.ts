/// <reference lib="webworker" />
// Runs the ACE fit lab off the main thread: reference data, ACE features with derivatives and
// every (ν, l_max) fit (src/lib/models/ace-fit.ts, runLab), about two to three seconds of work.

import { runLab, type LabOptions, type LabResult } from '~/lib/models/ace-fit';

export type AceWorkerRequest = { id: number; options: LabOptions };
export type AceWorkerResponse = { id: number; type: 'progress'; message: string } | { id: number; type: 'done'; result: LabResult } | { id: number; type: 'error'; message: string };

self.onmessage = (e: MessageEvent<AceWorkerRequest>) => {
  const { id, options } = e.data;
  const post = (m: AceWorkerResponse) => (self as unknown as DedicatedWorkerGlobalScope).postMessage(m);
  try {
    const result = runLab(options, (message) => post({ id, type: 'progress', message }));
    post({ id, type: 'done', result });
  } catch (err) {
    post({ id, type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
