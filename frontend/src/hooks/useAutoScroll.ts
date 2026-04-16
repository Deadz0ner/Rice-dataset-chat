import { useEffect, useRef } from "react";

export function useAutoScroll(...dependencies: unknown[]) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    // Use a short timeout to ensure the DOM has fully rendered
    // before scrolling. requestAnimationFrame alone isn't enough
    // because React may not have committed layout yet.
    const id = setTimeout(() => {
      el.scrollIntoView({ block: "end", behavior: "smooth" });
    }, 50);

    return () => clearTimeout(id);
  }, dependencies);

  return ref;
}
