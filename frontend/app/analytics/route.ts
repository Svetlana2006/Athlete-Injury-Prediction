import { readDesign, withNav, htmlResponse } from "@/lib/serveDesign";
export async function GET() {
  return htmlResponse(withNav(await readDesign("performance_analytics"), "performance-analytics"));
}
