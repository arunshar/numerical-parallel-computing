#!/usr/bin/env python3
"""Check that every page of a site fits a phone screen.

Usage:
  python3 check_mobile.py SITE [--width 375] [--include-static] [--pages a.html b.html]
                          [--inject-css CSS] [--chrome PATH] [--no-sandbox] [--chrome-arg ARG ...]
                          [--jobs 4] [--timeout 600] [--max-pages 500] [--strict] [--json out.json]

SITE is either a folder that holds the built HTML (for example docs/_build/html) or the address of a
published site (for example https://arunshar.com/numerical-parallel-computing/).

How it works. The script serves SITE on 127.0.0.1 and opens a headless Chrome on a small test page. For a
folder it serves the files. For an address it forwards each request to the live site, so the iframes and
the test page share one origin. The test page loads each site page in an iframe that is WIDTH pixels wide,
waits for the load event and for MathJax, and measures the page. A live address needs the network. When no
pages are named, the script finds them by reading the links on the site, starting from the address.

A page FAILS when:
  1. it is wider than the screen (documentElement.scrollWidth is above WIDTH). The report names the widest
     elements that stick out and have no scrolling ancestor. A code block or table inside a box with
     overflow-x: auto or scroll is allowed to be wide.
  2. it has no <meta name="viewport"> tag, or the tag lacks width=device-width.
  3. it does not answer with a 2xx status, or it does not finish loading in 20 seconds.
A page gets a WARNING (and fails only with --strict) when:
  4. the viewport tag blocks zooming (user-scalable=no or maximum-scale=1).
  5. text is smaller than 12 px, or a button or input is smaller than 24 by 24 px (WCAG 2.2, 2.5.8).
  6. content sticks out of a box that clips it with overflow: hidden, so a reader cannot scroll to it.

The last line also prints how many MathJax containers were typeset. Zero on a site with equations means
MathJax did not load (no network), and the equations were not tested.

Exit code 0: every page passes. Exit code 1: a page failed (or warned, with --strict). Exit code 2: Chrome
is missing, Chrome exited early, the live site could not be read, or the run timed out.

A single-file packet or reader that shows one section at a time hides the others, and a hidden section is
never measured. Pass --inject-css to show them all. For the house packet template use
  --inject-css ".page{display:block !important}"
and for the offline study reader use
  --inject-css "main>section.level1[hidden]{display:block !important}"

Chrome runs with its sandbox by default. GitHub-hosted ubuntu-24.04 and ubuntu-26.04 runners need no extra flag
(checked on 28 Oct 2026). Inside a Docker container that runs as root, Chrome refuses to start its sandbox, and
--no-sandbox is the usual fix (not tested here).

It needs Python 3.9 or newer and a Chrome or Chromium binary. It installs nothing. MathJax loads from the
network, as it does for readers. Run it before a site is published and after any CSS change.
"""
import argparse
import html.parser
import http.server
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome",
]
SKIP_DIRS = {"_sources", "_modules", "_images", "_downloads", ".doctrees"}
SKIP_FILES = {"genindex.html", "search.html", "py-modindex.html", "404.html"}
UA = "Mozilla/5.0 (compatible; check_mobile.py)"
WRAPPER_PATH = "/__mobile_check__.html"
RESULT_PATH = "/__result__"

WRAPPER = r"""<!doctype html><meta charset="utf-8"><title>mobile check</title><body><pre id="out">running</pre>
<script>
const W = __WIDTH__, PAGES = __PAGES__, JOBS = __JOBS__, EXTRA_CSS = __CSS__, LOAD_MS = 20000;
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function measure(path) {
  const res = { page: path };
  try {
    const r = await fetch('/' + path, { cache: 'no-store' });
    res.status = r.status;
    await r.arrayBuffer();
  } catch (e) { res.status = 0; res.error = 'fetch failed: ' + e; return res; }
  if (res.status < 200 || res.status >= 300) { res.error = 'HTTP ' + res.status; return res; }
  const f = document.createElement('iframe');
  f.style.cssText = 'position:absolute;left:0;top:0;border:0;visibility:hidden;width:' + W + 'px;height:900px';
  f.src = '/' + path;
  document.body.appendChild(f);
  let loaded = false;
  await new Promise(r => { f.onload = () => { loaded = true; r(); }; setTimeout(r, LOAD_MS); });
  if (!loaded) { res.error = 'load timeout'; f.remove(); return res; }
  const win = f.contentWindow, doc = f.contentDocument;
  if (EXTRA_CSS && doc) { const st = doc.createElement('style'); st.textContent = EXTRA_CSS; doc.head.appendChild(st); }
  try { if (win.MathJax && win.MathJax.startup && win.MathJax.startup.promise) await Promise.race([win.MathJax.startup.promise, sleep(15000)]); } catch (e) {}
  await sleep(600);
  try {
    const de = doc.documentElement;
    res.scrollW = de.scrollWidth;
    res.over = de.scrollWidth > W + 1;
    const vps = [...doc.querySelectorAll('meta[name=viewport]')].map(m => m.content || '');
    res.no_viewport = vps.length === 0;
    res.bad_viewport = vps.length > 0 && !vps.some(c => /width\s*=\s*device-width/i.test(c));
    res.blocks_zoom = vps.some(c => /user-scalable\s*=\s*(no|0)|maximum-scale\s*=\s*1(\.0+)?\s*(,|;|$)/i.test(c));
    res.math = doc.querySelectorAll('mjx-container').length;
    const visible = e => e.getClientRects().length > 0;
    // An element in an off-canvas menu is not on the screen until the reader opens it, so it is skipped here.
    const onScreen = e => { const r = e.getBoundingClientRect(); return r.right > 0 && r.left < W; };
    const scrolls = o => o === 'auto' || o === 'scroll';
    const guard = e => {
      for (let p = e.parentElement; p && p !== doc.body; p = p.parentElement) {
        const o = win.getComputedStyle(p).overflowX;
        if (o !== 'visible' && p.getBoundingClientRect().right <= W + 0.5) return o;
      }
      return null;
    };
    const wide = [...doc.querySelectorAll('body *')]
      .map(e => ({ e, r: e.getBoundingClientRect() }))
      .filter(x => x.r.width > 0 && x.r.right > W + 1);
    const label = x => ({ tag: x.e.tagName.toLowerCase(), cls: String(x.e.className).slice(0, 40), right: Math.round(x.r.right),
      text: (x.e.getAttribute('data-latex') || x.e.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 60) });
    res.offenders = wide.filter(x => guard(x.e) === null).sort((a, b) => b.r.right - a.r.right).slice(0, 3).map(label);
    const cut = wide.filter(x => { const g = guard(x.e); return (g === 'hidden' || g === 'clip') && x.r.width > 2 && x.r.height > 2; });
    res.cut_off = cut.length;
    res.cut_sample = cut.slice(0, 2).map(label);
    const small = [];
    const walker = doc.createTreeWalker(doc.body, NodeFilter.SHOW_TEXT);
    let n;
    while ((n = walker.nextNode())) {
      const t = n.textContent.trim();
      if (t.length < 12) continue;
      const p = n.parentElement;
      if (!p || p.closest('pre,code,script,style,svg,mjx-container,noscript,sup,sub,[hidden]') || !visible(p) || !onScreen(p)) continue;
      const fs = parseFloat(win.getComputedStyle(p).fontSize);
      if (fs < 12) small.push({ fs: Math.round(fs * 10) / 10, text: t.slice(0, 40) });
    }
    res.small_text = small.length;
    res.small_text_sample = small.slice(0, 2);
    const taps = [...doc.querySelectorAll('button,input:not([type=hidden]),select,textarea,[role=button],.run-btn')]
      .filter(e => visible(e) && onScreen(e)).map(e => ({ e, r: e.getBoundingClientRect() })).filter(x => x.r.width < 24 || x.r.height < 24);
    res.small_taps = taps.length;
    res.small_taps_sample = taps.slice(0, 2).map(x => ({ tag: x.e.tagName.toLowerCase(), cls: String(x.e.className).slice(0, 30),
      w: Math.round(x.r.width), h: Math.round(x.r.height), text: (x.e.textContent || x.e.value || '').trim().slice(0, 30) }));
  } catch (e) { res.error = String(e); }
  f.remove();
  return res;
}
(async () => {
  const results = [], pending = PAGES.slice();
  await Promise.all(Array.from({ length: JOBS }, async () => { while (pending.length) results.push(await measure(pending.shift())); }));
  await fetch('__RESULT__', { method: 'POST', body: JSON.stringify(results) });
  document.getElementById('out').textContent = 'done';
})();
</script>"""


class _Links(html.parser.HTMLParser):
    """Collect the href of every <a> tag."""

    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v:
                    self.links.append(v)


def find_chrome(explicit):
    if explicit:
        return shutil.which(explicit)  # None when the path or command is missing or not executable
    for c in CHROME_CANDIDATES:
        p = c if os.path.isabs(c) else shutil.which(c)
        if p and os.path.exists(p):
            return p
    return None


def is_url(s):
    return s.startswith("http://") or s.startswith("https://")


def skip_page(rel_parts, name, include_static):
    """True for pages that are not part of the reader-facing site."""
    if set(rel_parts) & SKIP_DIRS or name in SKIP_FILES:
        return True
    return "_static" in rel_parts and not include_static


def list_pages(site, include_static):
    out = []
    for p in sorted(Path(site).rglob("*.html")):
        rel = p.relative_to(site)
        if skip_page(rel.parts[:-1], rel.name, include_static):
            continue
        out.append(rel.as_posix())
    return out


def fetch(url, timeout=30):
    """Return (status, final_url, body bytes, content type). HTTP errors return their status and body."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.geturl(), r.read(), r.headers.get("Content-Type", "application/octet-stream")
    except urllib.error.HTTPError as e:
        with e:  # close the error response when done
            return e.code, url, e.read() or b"", e.headers.get("Content-Type", "text/plain") if e.headers else "text/plain"


def normalize_base(url):
    """Return the site root with a trailing slash and no query or fragment."""
    u = urllib.parse.urlsplit(url)
    path = u.path if u.path.endswith("/") else (u.path.rsplit("/", 1)[0] + "/" if "." in u.path.rsplit("/", 1)[-1] else u.path + "/")
    return urllib.parse.urlunsplit((u.scheme, u.netloc, path, "", ""))


def map_remote_path(base, request_path):
    """Map the path that Chrome requests from 127.0.0.1 to the address on the live site."""
    b = urllib.parse.urlsplit(base)
    r = urllib.parse.urlsplit(request_path)
    path = r.path or "/"
    origin = f"{b.scheme}://{b.netloc}"
    if b.path != "/" and path.startswith(b.path):
        target = origin + path
    else:
        target = base + path.lstrip("/")
    return target + ("?" + r.query if r.query else "")


def discover_remote(base, include_static, max_pages, fetcher=fetch):
    """Find the pages of a live site by following links from its first page.

    Returns (canonical_base, pages) where pages are paths relative to canonical_base."""
    status, final, body, _ = fetcher(base)
    if status != 200:
        raise RuntimeError(f"{base} answered HTTP {status}")
    canon = normalize_base(final)
    # The address may have redirected to one file, for example home.html. That file is then the first page.
    first = urllib.parse.urlsplit(final).path[len(urllib.parse.urlsplit(canon).path):] or "index.html"
    seen, order, todo = set(), [], [first]
    pages_html = {first: body}
    while todo and len(order) < max_pages:
        rel = todo.pop(0)
        if rel in seen:
            continue
        seen.add(rel)
        if rel not in pages_html:
            st, fin, bd, ctype = fetcher(canon + rel)
            if st != 200 or "html" not in ctype.lower():
                continue
            pages_html[rel] = bd
        order.append(rel)
        parser = _Links()
        parser.feed(pages_html[rel].decode("utf-8", "replace"))
        for href in parser.links:
            try:
                u = urllib.parse.urljoin(canon + rel, href)
                s = urllib.parse.urlsplit(u)
            except ValueError:  # a malformed address such as http://[host is not a page of this site
                continue
            u = urllib.parse.urlunsplit((s.scheme, s.netloc, s.path, "", ""))
            if not u.startswith(canon) or s.scheme not in ("http", "https"):
                continue
            nxt = u[len(canon):]
            if nxt == "" or nxt.endswith("/"):
                nxt += "index.html"
            if not nxt.endswith(".html") or nxt in seen:
                continue
            parts = nxt.split("/")
            if skip_page(parts[:-1], parts[-1], include_static):
                continue
            todo.append(nxt)
    return canon, sorted(set(order))


def make_handler(kind, root, wrapper, got):
    """Build the request handler. kind is 'dir' (root is a folder) or 'live' (root is the site address)."""
    base_cls = http.server.SimpleHTTPRequestHandler if kind == "dir" else http.server.BaseHTTPRequestHandler

    class H(base_cls):
        def __init__(self, *args, **kw):
            if kind == "dir":
                super().__init__(*args, directory=str(root), **kw)
            else:
                super().__init__(*args, **kw)

        def log_message(self, *args):
            pass

        def _send(self, status, body, ctype):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split("?")[0] == WRAPPER_PATH:
                self._send(200, wrapper.encode(), "text/html; charset=utf-8")
            elif kind == "dir":
                super().do_GET()
            else:
                try:
                    st, _, body, ctype = fetch(map_remote_path(root, self.path))
                except Exception as e:  # network failure
                    st, body, ctype = 502, str(e).encode(), "text/plain"
                self._send(st, body, ctype)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            got.put(self.rfile.read(n).decode())
            self.send_response(204)
            self.end_headers()

    return H


def _js(value):
    """JSON for the inline script. A < becomes \\u003c, so a </script> or <!-- in the text cannot end the script."""
    return json.dumps(value).replace("<", "\\u003c")


def build_wrapper(width, pages, jobs, css):
    values = {"WIDTH": str(width), "PAGES": _js(pages), "JOBS": str(jobs), "CSS": _js(css), "RESULT": RESULT_PATH}
    # One pass, so text that looks like a placeholder (a page named __CSS__.html) is never substituted again.
    return re.sub(r"__(WIDTH|PAGES|JOBS|CSS|RESULT)__", lambda m: values[m.group(1)], WRAPPER)


def _default_chrome_args():
    """Flags every headless run needs. On macOS, avoid the login keychain so Chrome does not hang on
    'Keychain Not Found' / SecItem dialogs when many temp profiles start at once (mutation tests, CI)."""
    args = ["--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
            "--password-store=basic", "--use-mock-keychain"]
    return args


def _stop_chrome(proc):
    """Terminate Chrome and its helpers. start_new_session puts them in their own process group."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def run_chrome(chrome, port, width, extra_args, timeout, got):
    """Open the test page in headless Chrome. Return (raw result or None, tail of Chrome's log)."""
    prof = tempfile.mkdtemp(prefix="mobilecheck-")
    log = os.path.join(prof, "chrome.log")
    cmd = [chrome] + _default_chrome_args() + [
        f"--user-data-dir={prof}/profile", f"--window-size={width},900",
    ] + list(extra_args) + [f"http://127.0.0.1:{port}{WRAPPER_PATH}"]
    with open(log, "w") as lf:
        # start_new_session=True so helpers die with the group; without it, orphaned Chromes keep
        # hammering the macOS keychain after the parent exits.
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, start_new_session=True)
        raw, t0 = None, time.time()
        try:
            while time.time() - t0 < timeout:
                try:
                    raw = got.get(timeout=1)
                    break
                except queue.Empty:
                    if proc.poll() is not None and got.empty():
                        break
        finally:
            _stop_chrome(proc)
    try:
        tail = "\n".join(Path(log).read_text(errors="replace").splitlines()[-12:])
    except OSError:
        tail = ""
    shutil.rmtree(prof, ignore_errors=True)
    return raw, tail


def summarize(results, strict):
    """Return (failures, warnings), each a list of (page, [reasons])."""
    fails, warns = [], []
    for r in sorted(results, key=lambda r: r["page"]):
        f, w = [], []
        if r.get("error"):
            f.append(r["error"])
        if r.get("over"):
            f.append(f"wider than the screen ({r['scrollW']} px)")
        if r.get("no_viewport"):
            f.append("no viewport meta tag")
        if r.get("bad_viewport"):
            f.append("viewport tag lacks width=device-width")
        if r.get("blocks_zoom"):
            w.append("viewport blocks zooming")
        if r.get("small_text"):
            w.append(f"{r['small_text']} text run(s) under 12 px")
        if r.get("small_taps"):
            w.append(f"{r['small_taps']} button or input under 24 px")
        if r.get("cut_off"):
            w.append(f"{r['cut_off']} element(s) cut off by overflow: hidden")
        if strict:
            f, w = f + w, []  # one entry per page, so the count of failed pages is right
        if f:
            fails.append((r["page"], f))
        if w:
            warns.append((r["page"], w))
    return fails, warns


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("site", help="a folder of built HTML, or the address of a published site")
    ap.add_argument("--width", type=int, default=375)
    ap.add_argument("--include-static", action="store_true", help="also test HTML files under _static (for example an offline reader)")
    ap.add_argument("--pages", nargs="*", help="test only these pages, relative to SITE")
    ap.add_argument("--inject-css", default="", help="CSS added to every page before it is measured")
    ap.add_argument("--chrome")
    ap.add_argument("--no-sandbox", action="store_true", help="start Chrome with --no-sandbox and --disable-dev-shm-usage (CI runners)")
    ap.add_argument("--chrome-arg", action="append", default=[], help="extra Chrome argument, repeatable")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max-pages", type=int, default=500, help="stop link discovery on a live site after this many pages")
    ap.add_argument("--strict", action="store_true", help="treat warnings as failures")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.jobs < 1:  # with no worker the run would measure nothing and still print PASS
        ap.error("--jobs must be 1 or more")

    chrome = find_chrome(a.chrome)
    if not chrome:
        print("FAIL: no Chrome or Chromium found. Pass --chrome PATH.")
        return 2

    if is_url(a.site):
        kind = "live"
        base = a.site if a.site.endswith("/") else a.site + "/"
        try:
            if a.pages:
                st, final, _, _ = fetch(base)
                if st != 200:
                    raise RuntimeError(f"{base} answered HTTP {st}")
                root, pages = normalize_base(final), a.pages
            else:
                root, pages = discover_remote(base, a.include_static, a.max_pages)
        except Exception as e:
            print(f"FAIL: could not read the live site: {e}")
            return 2
    else:
        kind = "dir"
        root = Path(a.site).resolve()
        if not root.is_dir():
            print(f"FAIL: {root} is not a folder")
            return 2
        missing = [p for p in (a.pages or []) if not (root / p).is_file()]
        if missing:
            print("FAIL: page not found under the site folder: " + ", ".join(missing))
            return 2
        pages = a.pages or list_pages(root, a.include_static)
    if not pages:
        print("FAIL: no HTML pages found")
        return 2

    got = queue.Queue()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), make_handler(kind, root, build_wrapper(a.width, pages, a.jobs, a.inject_css), got))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    extra = list(a.chrome_arg) + (["--no-sandbox", "--disable-dev-shm-usage"] if a.no_sandbox else [])
    try:
        raw, tail = run_chrome(chrome, srv.server_address[1], a.width, extra, a.timeout, got)
    finally:
        srv.shutdown()
        srv.server_close()
    if raw is None:
        print(f"FAIL: no result within {a.timeout} s. Chrome exited early or the run is too slow. Try --jobs 2, a larger --timeout, or --no-sandbox when Chrome runs as root inside a container.")
        if tail:
            print("Last lines of Chrome's log:\n" + tail)
        return 2

    results = json.loads(raw)
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=1))
    fails, warns = summarize(results, a.strict)
    where = "live site " + root if kind == "live" else "folder"
    print(f"checked {len(results)} page(s) at {a.width} px, {where}, with {chrome}")
    by_page = {r["page"]: r for r in results}
    for page, why in fails:
        print(f"  FAIL {page}: " + ", ".join(why))
        for o in by_page[page].get("offenders", []):
            print(f"       {o['tag']}.{o['cls']} reaches {o['right']} px: {o['text']!r}")
    for page, why in warns[:15]:
        print(f"  WARN {page}: " + ", ".join(why))
    if len(warns) > 15:
        print(f"  WARN ... and {len(warns) - 15} more page(s) with warnings (use --json for the full list)")
    total_math = sum(r.get("math", 0) for r in results)
    print(f"math containers typeset: {total_math}" + ("  (zero means MathJax did not load, so equations were not tested)" if total_math == 0 else ""))
    if fails:
        print(f"MOBILE CHECK: FAIL ({len(fails)} of {len(results)} pages)")
        return 1
    print("MOBILE CHECK: PASS" + (f" ({len(warns)} page(s) with warnings)" if warns else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
