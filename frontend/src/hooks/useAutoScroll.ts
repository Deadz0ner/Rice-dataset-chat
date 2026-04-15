import { useEffect, useRef } from "react";

export function useAutoScroll<T>(dependency: T) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!ref.current) {
      return;
    }
    ref.current.scrollTop = ref.current.scrollHeight;
  }, [dependency]);

  return ref;
}
