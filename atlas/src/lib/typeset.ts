// Tiny inline typesetter for plain-text labels and captions: escapes HTML, then
// renders single-letter symbols with short subscripts (r_c, d_G, q_i) as an
// italic letter with a subscript. Use KaTeX (<Math>) for anything richer.

export function escapeHtml(text: string): string {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

export function typesetInline(text: string): string {
  return escapeHtml(text).replace(/\b([A-Za-z])_([A-Za-z0-9]{1,3})\b/g, '<i>$1</i><sub>$2</sub>');
}
