import { writeFileSync } from "node:fs";
import { renderReport } from "./dist/report.js";

let out = "report.html";
for (let i = 0; i < process.argv.length - 1; i++) {
  const a = process.argv[i + 1];
  if (a === "--out" || a === "-o") out = process.argv[i + 2];
}
const html = renderReport();
writeFileSync(out, html);
console.log(`report written to ${out} (${Buffer.byteLength(html)} bytes)`);
