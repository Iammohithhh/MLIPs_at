import { useId, useMemo, useRef, useState, type PointerEvent, type WheelEvent } from 'react';
import './AtlasMap.css';
import type { MapEdge, MapNode } from '~/lib/atlas-graph';

// The whole Atlas as a map: records as nodes, references as edges, laid out at build
// time. Pan by dragging, zoom with the wheel or buttons, select a node to see its
// neighbourhood and open its page.

export default function AtlasMap({
  nodes,
  edges,
  kinds,
}: {
  nodes: MapNode[];
  edges: MapEdge[];
  kinds: Array<{ id: string; label: string; tone: string; count: number }>;
}) {
  const ids = useId();
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [query, setQuery] = useState('');
  const [selId, setSelId] = useState<string>('model-families/mace');
  const [hoverId, setHoverId] = useState<string | null>(null);

  const byId = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);
  const nbrs = useMemo(() => {
    const m = new Map<string, Set<string>>();
    for (const e of edges) {
      if (!m.has(e.source)) m.set(e.source, new Set());
      if (!m.has(e.target)) m.set(e.target, new Set());
      m.get(e.source)!.add(e.target);
      m.get(e.target)!.add(e.source);
    }
    return m;
  }, [edges]);

  const focus = hoverId ?? selId;
  const focusSet = useMemo(() => new Set([focus, ...(nbrs.get(focus) ?? [])]), [focus, nbrs]);
  const q = query.trim().toLowerCase();
  const matches = useMemo(() => (q ? new Set(nodes.filter((n) => n.title.toLowerCase().includes(q)).map((n) => n.id)) : null), [q, nodes]);
  const visible = (n: MapNode) => !hidden.has(n.collection);

  // View box: pan and zoom.
  const xs = nodes.map((n) => n.x);
  const ys = nodes.map((n) => n.y);
  const base = { x: Math.min(...xs) - 60, y: Math.min(...ys) - 40, w: Math.max(...xs) - Math.min(...xs) + 120, h: Math.max(...ys) - Math.min(...ys) + 80 };
  const [view, setView] = useState(base);
  const svgRef = useRef<SVGSVGElement>(null);
  const drag = useRef<{ x: number; y: number; vx: number; vy: number; moved: boolean } | null>(null);

  const zoom = (f: number, cx = view.x + view.w / 2, cy = view.y + view.h / 2) =>
    setView((v) => {
      const w = Math.min(base.w * 1.5, Math.max(base.w / 8, v.w * f));
      const h = (w / v.w) * v.h;
      return { x: cx - ((cx - v.x) * w) / v.w, y: cy - ((cy - v.y) * h) / v.h, w, h };
    });
  const toWorld = (e: { clientX: number; clientY: number }) => {
    const r = svgRef.current!.getBoundingClientRect();
    return { x: view.x + ((e.clientX - r.left) / r.width) * view.w, y: view.y + ((e.clientY - r.top) / r.height) * view.h };
  };
  const onWheel = (e: WheelEvent<SVGSVGElement>) => {
    e.preventDefault();
    const p = toWorld(e);
    zoom(e.deltaY > 0 ? 1.15 : 1 / 1.15, p.x, p.y);
  };
  const onDown = (e: PointerEvent<SVGSVGElement>) => {
    drag.current = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y, moved: false };
    (e.target as Element).setPointerCapture?.(e.pointerId);
  };
  const onMove = (e: PointerEvent<SVGSVGElement>) => {
    const d = drag.current;
    if (!d) return;
    const r = svgRef.current!.getBoundingClientRect();
    const dx = ((e.clientX - d.x) / r.width) * view.w;
    const dy = ((e.clientY - d.y) / r.height) * view.h;
    if (Math.hypot(e.clientX - d.x, e.clientY - d.y) > 3) d.moved = true;
    setView((v) => ({ ...v, x: d.vx - dx, y: d.vy - dy }));
  };
  const onUp = () => {
    drag.current = null;
  };

  const sel = byId.get(selId) ?? nodes[0]!;
  const selNbrs = [...(nbrs.get(sel.id) ?? [])].map((k) => byId.get(k)!).sort((a, b) => a.collection.localeCompare(b.collection) || a.title.localeCompare(b.title));
  const rOf = (n: MapNode) => 3.5 + Math.sqrt(n.degree) * 1.9;
  const kindLabel = (c: string) => kinds.find((k) => k.id === c)?.label ?? c;
  const toneOf = (c: string) => kinds.find((k) => k.id === c)?.tone ?? 'neutral';

  return (
    <div className="am">
      <div className="am-bar">
        <label htmlFor={`${ids}-q`} className="am-search">
          <span className="sr-only">Find a record</span>
          <input id={`${ids}-q`} type="search" placeholder="Find a record…" value={query} onChange={(e) => setQuery(e.target.value)} />
          {matches && <span className="am-count">{matches.size}</span>}
        </label>
        <div className="am-kinds" role="group" aria-label="Record kinds">
          {kinds.map((k) => (
            <button
              key={k.id}
              type="button"
              aria-pressed={!hidden.has(k.id)}
              className={`am-kind tone-${k.tone}`}
              onClick={() =>
                setHidden((s) => {
                  const n = new Set(s);
                  if (n.has(k.id)) n.delete(k.id);
                  else n.add(k.id);
                  return n;
                })
              }
            >
              <span className="am-dot" /> {k.label} <small>{k.count}</small>
            </button>
          ))}
        </div>
      </div>

      <div className="am-body">
        <div className="am-stage">
          <svg
            ref={svgRef}
            viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
            onWheel={onWheel}
            onPointerDown={onDown}
            onPointerMove={onMove}
            onPointerUp={onUp}
            onPointerLeave={onUp}
            role="application"
            aria-label="Map of every Atlas record and the references between them. Use the list beside it to navigate by keyboard."
          >
            <g className="am-edges">
              {edges.map((e) => {
                const a = byId.get(e.source)!;
                const b = byId.get(e.target)!;
                if (!visible(a) || !visible(b)) return null;
                const on = focusSet.has(a.id) && focusSet.has(b.id) && (a.id === focus || b.id === focus);
                return <line key={`${e.source}|${e.target}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} className={on ? 'on' : undefined} />;
              })}
            </g>
            <g>
              {nodes.filter(visible).map((n) => {
                const dim = (matches && !matches.has(n.id)) || (!matches && !focusSet.has(n.id));
                const isFocus = n.id === focus;
                return (
                  <g
                    key={n.id}
                    className={`am-node tone-${toneOf(n.collection)}${dim ? ' dim' : ''}${isFocus ? ' focus' : ''}${n.id === selId ? ' sel' : ''}`}
                    transform={`translate(${n.x} ${n.y})`}
                    onPointerEnter={() => setHoverId(n.id)}
                    onPointerLeave={() => setHoverId(null)}
                    onClick={() => {
                      if (!drag.current?.moved) setSelId(n.id);
                    }}
                  >
                    <circle r={rOf(n)} />
                    {(n.degree >= 6 || isFocus || (matches && matches.has(n.id))) && (
                      <text x={rOf(n) + 3} y={3.5}>
                        {n.title.length > 38 ? `${n.title.slice(0, 36)}…` : n.title}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          </svg>
          <div className="am-zoom">
            <button type="button" aria-label="Zoom in" onClick={() => zoom(1 / 1.3)}>+</button>
            <button type="button" aria-label="Zoom out" onClick={() => zoom(1.3)}>−</button>
            <button type="button" onClick={() => setView(base)}>Reset</button>
          </div>
        </div>

        <aside className="am-panel" aria-live="polite">
          <p className={`am-kindtag tone-${toneOf(sel.collection)}`}>{kindLabel(sel.collection)}</p>
          <p className="am-title">{sel.title}</p>
          <p className="am-meta">
            {sel.degree} link{sel.degree === 1 ? '' : 's'} · record {sel.status}
          </p>
          <a className="am-open" href={sel.url}>
            Open →
          </a>
          <p className="am-sub">Linked records</p>
          <ul className="am-list">
            {selNbrs.map((n) => (
              <li key={n.id}>
                <button type="button" onClick={() => setSelId(n.id)} className={`tone-${toneOf(n.collection)}`}>
                  <span className="am-dot" />
                  <span className="am-li-t">{n.title}</span>
                  <small>{kindLabel(n.collection)}</small>
                </button>
              </li>
            ))}
          </ul>
        </aside>
      </div>
    </div>
  );
}
