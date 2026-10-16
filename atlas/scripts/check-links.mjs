// Check every internal link and #anchor in the built site (dist/). Fails on
// links to missing pages or missing ids. External links are not fetched.
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const dist = fileURLToPath(new URL('../dist/', import.meta.url));
const files = [];
(function walk(d) {
  for (const f of readdirSync(d)) {
    const p = join(d, f);
    if (statSync(p).isDirectory()) {
      if (f !== 'pagefind' && f !== '_astro') walk(p);
    } else if (f.endsWith('.html')) files.push(p);
  }
})(dist);

const ids = new Map();
const pageOf = (file) => '/' + relative(dist, file).split(sep).join('/').replace(/index\.html$/, '').replace(/\.html$/, '');
for (const f of files) {
  const html = readFileSync(f, 'utf8');
  ids.set(pageOf(f), new Set([...html.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1])));
}

function resolve(path) {
  const clean = path.split('?')[0];
  if (ids.has(clean)) return clean;
  if (ids.has(clean + '/')) return clean + '/';
  return null;
}

const problems = [];
let checked = 0;
for (const f of files) {
  const from = pageOf(f);
  const html = readFileSync(f, 'utf8');
  for (const m of html.matchAll(/\shref="([^"]+)"/g)) {
    const href = m[1].replace(/&amp;/g, '&');
    if (/^(https?:|mailto:|data:)/.test(href)) continue;
    checked++;
    const [path, hash] = href.split('#');
    let target = from;
    if (path) {
      if (!path.startsWith('/')) continue;
      if (/\.(svg|png|json|css|js|xml|txt|ico|webmanifest)$/.test(path)) {
        if (!existsSync(join(dist, path))) problems.push(`${from}: missing file ${path}`);
        continue;
      }
      const r = resolve(path);
      if (!r) {
        problems.push(`${from}: link to missing page ${path}`);
        continue;
      }
      target = r;
    }
    if (hash && !ids.get(target)?.has(decodeURIComponent(hash))) problems.push(`${from}: missing anchor ${target}#${hash}`);
  }
}
const unique = [...new Set(problems)];
console.log(unique.length ? unique.join('\n') : `links ok: ${checked} internal links on ${files.length} pages`);
process.exit(unique.length ? 1 : 0);
