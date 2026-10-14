import { useCallback, useEffect, useState } from 'react';
import { getActiveSymbol, onSymbolChange, setActiveSymbol } from './linked-symbols';

/**
 * React binding for linked symbols. Returns the active key and a setter, so an
 * island can both react to equation hovers and highlight equations itself
 * (e.g. when the user hovers the cutoff line in a plot or a shell in 3D).
 */
export function useLinkedSymbol(source: string): [string | null, (sym: string | null) => void] {
  const [active, setActive] = useState<string | null>(null);

  useEffect(() => {
    setActive(getActiveSymbol());
    return onSymbolChange(({ sym }) => setActive(sym));
  }, []);

  const set = useCallback((sym: string | null) => setActiveSymbol(sym, source), [source]);
  return [active, set];
}
