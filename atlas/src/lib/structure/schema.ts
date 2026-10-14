// Runtime validation of Atlas-format structure JSON (pipeline output, uploads).
import { z } from 'zod';
import { element } from './elements';

const vec3 = z.tuple([z.number().finite(), z.number().finite(), z.number().finite()]);

export const structureSchema = z
  .object({
    format: z.literal('atlas-structure').optional(),
    version: z.literal(1).optional(),
    id: z.string().regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/),
    title: z.string().min(1),
    symbols: z.array(z.string()).min(1),
    positions: z.array(vec3),
    cell: z.tuple([vec3, vec3, vec3]).nullable(),
    pbc: z.tuple([z.boolean(), z.boolean(), z.boolean()]),
    provenance: z
      .object({ source: z.string(), generator: z.string().optional(), created: z.string().optional(), notes: z.string().optional() })
      .optional(),
  })
  .superRefine((s, ctx) => {
    if (s.positions.length !== s.symbols.length) {
      ctx.addIssue({ code: 'custom', message: 'one position per symbol', path: ['positions'] });
    }
    s.symbols.forEach((sym, i) => {
      try {
        element(sym);
      } catch {
        ctx.addIssue({ code: 'custom', message: `unknown element "${sym}"`, path: ['symbols', i] });
      }
    });
    if (s.pbc.some(Boolean) && !s.cell) {
      ctx.addIssue({ code: 'custom', message: 'periodic structures need a cell', path: ['cell'] });
    }
  });
