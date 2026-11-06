import { NextResponse } from "next/server";
import { readFile } from "fs/promises";
import path from "path";

/**
 * Injects a navigation script that:
 * 1. Wires all [data-path] sidebar links to Next.js routes
 * 2. Marks the current active page correctly
 */
export function withNav(html: string, currentPath: string): string {
  const routes: Record<string, string> = {
    "overview":              "/",
    "athletes":              "/athletes",
    "injury-prediction":     "/prediction",
    "try-your-data":         "/try-your-data",
    "performance-analytics": "/analytics",
    "model-insights":        "/insights",
    "executive-pitch":       "/pitch",
  };

  const navScript = `
<script>
(function(){
  var routes = ${JSON.stringify(routes)};
  document.querySelectorAll('[data-path]').forEach(function(el) {
    var p = el.getAttribute('data-path');
    if (routes[p]) {
      el.href = routes[p];
      el.removeAttribute('aria-current');
    }
    if (p === '${currentPath}') {
      el.setAttribute('aria-current', 'page');
    }
  });
  // Also wire any inline href="#" links with data-path on overview
  document.querySelectorAll('a[href="#"][data-path]').forEach(function(el){
    var p = el.getAttribute('data-path');
    if(routes[p]) el.href = routes[p];
  });
  var nav = document.querySelector('nav');
  if (nav && !nav.querySelector('[data-path="try-your-data"]')) {
    var link = document.createElement('a');
    link.href = routes['try-your-data'];
    link.setAttribute('data-path', 'try-your-data');
    link.className = 'flex items-center gap-space-md px-space-md py-space-sm rounded-lg text-on-surface-variant hover:bg-surface-container hover:text-on-surface transition-colors';
    link.innerHTML = '<span class="material-symbols-outlined text-[20px]">edit_note</span><span class="font-body-md text-body-md">Try your data</span>';
    nav.appendChild(link);
  }
})();
</script>`;

  return html.replace("</body>", navScript + "\n</body>");
}

export async function readDesign(slug: string): Promise<string> {
  const p = path.join(process.cwd(), "public", "designs", slug, "code.html");
  return readFile(p, "utf-8");
}

export function htmlResponse(html: string): NextResponse {
  return new NextResponse(html, {
    headers: { "Content-Type": "text/html; charset=utf-8" },
  });
}
