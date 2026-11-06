import { readDesign, withNav, htmlResponse } from "@/lib/serveDesign";

export async function GET() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
  const html = (await readDesign("try_your_data"))
    .replace("__API_BASE__", JSON.stringify(apiBase));
  return htmlResponse(withNav(html, "try-your-data"));
}
