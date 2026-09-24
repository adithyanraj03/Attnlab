// Bundle the compiled modules into one self-contained index.html.
// No bundler: tsc emits ES modules, we strip import/export and inline.

import { readFileSync, writeFileSync } from "node:fs";

const ORDER = ["prng", "mat", "attn", "svg", "model", "app"];
let js = "";
for (const name of ORDER) {
  let src = readFileSync(`dist/${name}.js`, "utf8");
  src = src.replace(/^\s*import[\s\S]*?from\s*["'][^"']+["'];/g, "");
  src = src.replace(/^export\s+/gm, "");
  js += `// ---- ${name} ----\n${src}\n\n`;
}
const tail = `if (typeof document !== "undefined") {
  const __run = () => Attnlab.init();
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", __run);
  else __run();
}
`;
const tpl = readFileSync("public/index.html", "utf8");
const marker = "/*__ATTRLAB_JS__*/";
if (!tpl.includes(marker)) throw new Error("index template missing " + marker);
const html = tpl.replace(marker, () => js + tail);
writeFileSync("index.html", html);
console.log(`index.html written (${Buffer.byteLength(html)} bytes, self-contained)`);
