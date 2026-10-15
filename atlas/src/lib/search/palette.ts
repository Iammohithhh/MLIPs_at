// ⌘K search palette. Loaded on first use only. Two sources:
//  - Pagefind: full-text index of every page, built after `astro build`;
//  - Orama: typo-tolerant search over every Atlas record (/search/records.json).
import { create, insert, search as oramaSearch, type AnyOrama } from '@orama/orama';
import './palette.css';

interface RecordDoc {
  id: string;
  collection: string;
  kind: string;
  title: string;
  text: string;
  status: string;
  url: string;
}

interface Hit {
  url: string;
  title: string;
  meta: string;
  excerpt?: string;
  group: 'Pages' | 'Records' | 'Go to';
}

interface PagefindResult {
  url: string;
  excerpt: string;
  meta: { title?: string };
}
interface Pagefind {
  init?: () => Promise<void>;
  search: (q: string) => Promise<{ results: Array<{ data: () => Promise<PagefindResult> }> }>;
}

const QUICK: Hit[] = [
  { url: '/physics-lab/', title: 'Physics Lab', meta: 'destination', group: 'Go to' },
  { url: '/models/', title: 'Models', meta: 'destination', group: 'Go to' },
  { url: '/data/', title: 'Data', meta: 'destination', group: 'Go to' },
  { url: '/benchmarks/', title: 'Benchmarks', meta: 'destination', group: 'Go to' },
  { url: '/explore/', title: 'Explore the Atlas map', meta: 'destination', group: 'Go to' },
  { url: '/failures/', title: 'Failure Atlas', meta: 'destination', group: 'Go to' },
  { url: '/models/mace/', title: 'MACE', meta: 'model family', group: 'Go to' },
  { url: '/papers/', title: 'Papers', meta: 'destination', group: 'Go to' },
  { url: '/method/', title: 'How the Atlas is verified', meta: 'method', group: 'Go to' },
  { url: '/method/atomic-viewer/', title: 'Atomic viewer', meta: 'method', group: 'Go to' },
  { url: '/method/content-model/', title: 'Content model', meta: 'method', group: 'Go to' },
];

let db: AnyOrama | null = null;
let pagefind: Pagefind | null | undefined;
let root: HTMLElement | null = null;
let returnFocus: HTMLElement | null = null;

async function records(): Promise<AnyOrama> {
  if (db) return db;
  const docs = (await (await fetch('/search/records.json')).json()) as RecordDoc[];
  const o = create({
    schema: { id: 'string', collection: 'string', kind: 'string', title: 'string', text: 'string', status: 'string', url: 'string' },
  });
  // Ids repeat across collections (the Allegro family and the allegro code), so key by both.
  for (const d of docs) await insert(o, { ...d, id: `${d.collection}:${d.id}` });
  db = o;
  return o;
}

async function pages(): Promise<Pagefind | null> {
  if (pagefind !== undefined) return pagefind;
  try {
    const url = '/pagefind/pagefind.js';
    pagefind = (await import(/* @vite-ignore */ url)) as Pagefind;
    await pagefind.init?.();
  } catch {
    pagefind = null; // e.g. `astro dev`, where the index has not been built
  }
  return pagefind;
}

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);

function build(): HTMLElement {
  const el = document.createElement('div');
  el.className = 'sp-backdrop';
  el.hidden = true;
  el.innerHTML = `
    <div class="sp" role="dialog" aria-modal="true" aria-label="Search the Atlas">
      <div class="sp-head">
        <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true"><circle cx="8.5" cy="8.5" r="5.5" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="m13 13 4 4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
        <input class="sp-input" type="search" role="combobox" aria-expanded="true" aria-controls="sp-list" aria-autocomplete="list" autocomplete="off" spellcheck="false" placeholder="Search pages, papers, claims, models…" />
        <kbd>Esc</kbd>
      </div>
      <div class="sp-list" id="sp-list" role="listbox" aria-label="Results"></div>
      <p class="sp-foot"><span>↑ ↓ move · ↵ open</span><span class="sp-status" aria-live="polite"></span></p>
    </div>`;
  document.body.appendChild(el);
  const input = el.querySelector<HTMLInputElement>('.sp-input')!;
  const list = el.querySelector<HTMLElement>('.sp-list')!;
  const status = el.querySelector<HTMLElement>('.sp-status')!;
  let active = 0;
  let timer = 0;
  let seq = 0;

  const options = () => Array.from(list.querySelectorAll<HTMLAnchorElement>('[role=option]'));
  const highlight = (k: number) => {
    const opts = options();
    if (!opts.length) return input.removeAttribute('aria-activedescendant');
    active = (k + opts.length) % opts.length;
    opts.forEach((o, i) => o.setAttribute('aria-selected', String(i === active)));
    input.setAttribute('aria-activedescendant', opts[active]!.id);
    opts[active]!.scrollIntoView({ block: 'nearest' });
  };

  const render = (hits: Hit[], note: string) => {
    const groups = ['Go to', 'Pages', 'Records'] as const;
    let n = 0;
    list.innerHTML = groups
      .map((g) => {
        const gh = hits.filter((h) => h.group === g);
        if (!gh.length) return '';
        return `<div class="sp-group" role="presentation"><p class="sp-group-title" role="presentation">${g}</p>${gh
          .map(
            (h) => `<a role="option" id="sp-opt-${n++}" class="sp-opt" href="${esc(h.url)}" aria-selected="false">
              <span class="sp-title">${esc(h.title)}</span><span class="sp-meta">${esc(h.meta)}</span>
              ${h.excerpt ? `<span class="sp-excerpt">${h.excerpt}</span>` : ''}</a>`,
          )
          .join('')}</div>`;
      })
      .join('');
    status.textContent = note;
    highlight(0);
  };

  const run = async (q: string) => {
    const mine = ++seq;
    if (!q.trim()) return render(QUICK, '');
    const [o, pf] = await Promise.all([records(), pages()]);
    const r = await oramaSearch(o, { term: q, properties: ['title', 'text', 'kind'], boost: { title: 2 }, tolerance: 1, limit: 8 });
    const recHits: Hit[] = r.hits.map((h) => {
      const d = h.document as unknown as RecordDoc;
      return { url: d.url, title: d.title, meta: `${d.kind.replace(/s$/, '')} · ${d.status}`, group: 'Records' };
    });
    let pageHits: Hit[] = [];
    if (pf) {
      const res = await pf.search(q);
      const data = await Promise.all(res.results.slice(0, 6).map((x) => x.data()));
      pageHits = data.map((d) => ({ url: d.url, title: d.meta.title ?? d.url, meta: 'page', excerpt: d.excerpt, group: 'Pages' }));
    }
    if (mine !== seq) return;
    const total = recHits.length + pageHits.length;
    render([...pageHits, ...recHits], `${total} result${total === 1 ? '' : 's'}${pf ? '' : ' (page index not built)'}`);
  };

  input.addEventListener('input', () => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => run(input.value), 110);
  });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      highlight(active + 1);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      highlight(active - 1);
    } else if (e.key === 'Enter') {
      const opt = options()[active];
      if (opt) {
        e.preventDefault();
        close();
        window.location.href = opt.href;
      }
    }
  });
  el.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      close();
    }
    if (e.key === 'Tab') {
      e.preventDefault();
      input.focus();
    }
  });
  el.addEventListener('mousedown', (e) => {
    if (e.target === el) close();
  });
  list.addEventListener('click', () => close());
  render(QUICK, '');
  return el;
}

export function close(): void {
  if (!root || root.hidden) return;
  root.hidden = true;
  document.documentElement.classList.remove('sp-open');
  returnFocus?.focus();
}

export function open(): void {
  returnFocus = document.activeElement as HTMLElement | null;
  root ??= build();
  root.hidden = false;
  document.documentElement.classList.add('sp-open');
  const input = root.querySelector<HTMLInputElement>('.sp-input')!;
  input.focus();
  input.select();
  void records();
  void pages();
}
