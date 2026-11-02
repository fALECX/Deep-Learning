/* Study Desk: renders the Lernen markdown files as one interactive site.
   Content comes from content.js (built by build_site.py) and exam_bank.js. */
(function () {
  "use strict";

  const C = window.CONTENT || {};
  const EX = window.EXAM || { A: [], B: [], C: [] };
  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  // ---------- storage (per browser; falls back to memory) ----------
  const mem = {};
  const store = {
    get(k, d) {
      try {
        const v = localStorage.getItem("sd:" + k);
        if (v !== null) return JSON.parse(v);
      } catch (e) { /* blocked storage */ }
      return k in mem ? mem[k] : d;
    },
    set(k, v) {
      mem[k] = v;
      try { localStorage.setItem("sd:" + k, JSON.stringify(v)); } catch (e) { /* ignore */ }
    },
    del(k) {
      delete mem[k];
      try { localStorage.removeItem("sd:" + k); } catch (e) { /* ignore */ }
    }
  };

  // ---------- course structure in topic order ----------
  const CHAPTERS = [
    { id: "01", file: "01_neural_network_basics.md", weeks: [2, 3], nick: "Basics" },
    { id: "07", file: "07_evaluation_and_workflow.md", weeks: [1, 6], nick: "Evaluation" },
    { id: "10", file: "10_cnns_and_vision.md", weeks: [4, 5], nick: "CNNs" },
    { id: "02", file: "02_text_to_numbers.md", weeks: [7], nick: "Text → numbers" },
    { id: "03", file: "03_sequences_and_rnns.md", weeks: [7], nick: "RNNs" },
    { id: "04", file: "04_attention_and_transformers.md", weeks: [8], nick: "Transformers" },
    { id: "06", file: "06_task_types_multilabel_tagging.md", weeks: [8], nick: "Task framing" },
    { id: "05", file: "05_pretraining_and_adaptation.md", weeks: [9], nick: "Pretraining, LoRA" },
    { id: "11", file: "11_generative_models.md", weeks: [10], nick: "Generative" },
    { id: "08", file: "08_interpretability.md", weeks: [11], nick: "Interpretability" },
    { id: "09", file: "09_bias_shift_deployment.md", weeks: [9, 12], nick: "Bias and shift" }
  ];
  const CH = {};
  CHAPTERS.forEach((c, i) => { c.step = i + 1; CH[c.id] = c; });

  const BLOCKS = [
    { key: "A", title: "How a network learns", flow: ["neuron", "activation", "output + loss", "learning loop"], ch: [["01", 1, 4]] },
    { key: "B", title: "Train it well", flow: ["optimiser", "stable training", "overfitting"], ch: [["01", 5, 99]] },
    { key: "C", title: "Measure it honestly", flow: ["splits", "metrics", "workflow and learning curves"], ch: [["07"]] },
    { key: "D", title: "Images", flow: ["CNN blocks", "architectures, detection, transfer"], ch: [["10"]] },
    { key: "E", title: "Text and sequences", flow: ["text → numbers", "RNN", "attention", "which head"], ch: [["02"], ["03"], ["04"], ["06"]] },
    { key: "F", title: "Big pretrained models", flow: ["self-supervised", "adaptation", "LoRA"], ch: [["05"]] },
    { key: "G", title: "Generate", flow: ["AE", "VAE", "GAN", "diffusion"], ch: [["11"]] },
    { key: "H", title: "Trust it", flow: ["interpretability", "bias and shift"], ch: [["08"], ["09"]] }
  ];
  CHAPTERS.forEach((c) => { c.blocks = BLOCKS.filter((b) => b.ch.some((x) => x[0] === c.id)).map((b) => b.key); });

  // ---------- markdown + math ----------
  function slugify(s) {
    return String(s)
      .replace(/MATHX\d+X/g, "")
      .replace(/\$[^$]*\$/g, "")
      .replace(/<[^>]+>/g, "")
      .replace(/[*_`~]/g, "")
      .toLowerCase().trim()
      .replace(/[^\p{L}\p{N}\s-]/gu, "")
      .replace(/\s+/g, "-");
  }

  const R = { prefix: "", seen: {} };
  marked.use({
    gfm: true,
    renderer: {
      heading(text, level, raw) {
        let id = R.prefix + slugify(raw);
        if (R.seen[id]) { const n = R.seen[id]++; id += "-" + n; } else R.seen[id] = 1;
        return `<h${level} id="${id}">${text}</h${level}>\n`;
      }
    }
  });

  function protect(src) {
    const math = [];
    const parts = src.split(/(```[\s\S]*?```|`[^`\n]+`)/g);
    for (let i = 0; i < parts.length; i += 2) {
      parts[i] = parts[i]
        .replace(/\$\$([\s\S]+?)\$\$/g, (m, t) => { math.push({ d: true, t: t.replace(/^[ \t]*>[ \t]?/gm, "") }); return "MATHX" + (math.length - 1) + "X"; })
        .replace(/\$([^\s$](?:[^$\n]*?[^\s$])?)\$/g, (m, t) => { math.push({ d: false, t }); return "MATHX" + (math.length - 1) + "X"; });
    }
    return { src: parts.join(""), math };
  }

  function restore(html, math) {
    html = html.replace(/<p>\s*MATHX(\d+)X\s*<\/p>/g, (m, i) => {
      const x = math[+i];
      return x && x.d ? `<div class="math-d">\\[${esc(x.t)}\\]</div>` : m;
    });
    return html.replace(/MATHX(\d+)X/g, (m, i) => {
      const x = math[+i];
      if (!x) return m;
      return x.d ? `<span class="math-d">\\[${esc(x.t)}\\]</span>` : `<span class="math-i">\\(${esc(x.t)}\\)</span>`;
    });
  }

  function md(src, opts) {
    opts = opts || {};
    R.prefix = opts.prefix || "";
    if (!opts.keepSeen) R.seen = {};
    const p = protect(src || "");
    const html = opts.inline ? marked.parseInline(p.src) : marked.parse(p.src);
    return restore(html, p.math);
  }

  // Inline text with $math$ for exam items (plain text, newlines kept).
  function plainWithMath(s) {
    const p = protect(String(s));
    return restore(esc(p.src).replace(/\n/g, "<br>"), p.math);
  }

  const mathQueue = [];
  let lastTypeset = Promise.resolve();
  function typeset(el) {
    if (!el) return;
    if (window.MathJax && MathJax.typesetPromise && window.__mathOk) {
      lastTypeset = MathJax.typesetPromise([el]).catch(() => {});
    } else mathQueue.push(el);
  }
  window.__mathReady = function () {
    window.__mathOk = true;
    const els = mathQueue.splice(0);
    els.filter((e) => e.isConnected).forEach((e) => typeset(e));
  };

  function routeForMdLink(href) {
    let m = href.match(/^(\d{2})_[\w-]+\.md(?:#(.*))?$/);
    if (m && CH[m[1]]) return { route: "ch-" + m[1], anchor: m[2] || "" };
    m = href.match(/^(QUESTIONS_AND_SUMMARY|CHEAT_SHEET_BY_TOPIC|CHEAT_SHEET_DRAFT|deep-learning-videos|README)\.md(?:#(.*))?$/);
    if (m) return { route: { QUESTIONS_AND_SUMMARY: "practice", CHEAT_SHEET_BY_TOPIC: "cheatsheet", CHEAT_SHEET_DRAFT: "cheatsheet", "deep-learning-videos": "videos", README: "home" }[m[1]], anchor: "" };
    return null;
  }

  function post(el) {
    $$("blockquote", el).forEach((bq) => {
      const p = bq.firstElementChild;
      if (!p || p.tagName !== "P") return;
      const m = p.innerHTML.match(/^\s*\[!(\w+)\]\s*/);
      if (!m) return;
      p.innerHTML = p.innerHTML.slice(m[0].length);
      bq.classList.add("callout", "callout-" + m[1].toLowerCase());
      if (/^\s*<strong>From the course/.test(p.innerHTML)) bq.classList.add("course-task");
      if (!p.textContent.trim() && !p.querySelector("img")) p.remove();
    });
    $$("a[href]", el).forEach((a) => {
      const h = a.getAttribute("href");
      const r = routeForMdLink(h);
      if (r) {
        a.setAttribute("href", "#" + r.route);
        a.dataset.route = r.route;
        if (r.anchor) a.dataset.anchor = r.anchor;
      } else if (/^https?:/i.test(h)) {
        a.target = "_blank";
        a.rel = "noopener";
      }
    });
    $$("table", el).forEach((t) => {
      if (t.parentElement.classList.contains("table-wrap")) return;
      const w = document.createElement("div");
      w.className = "table-wrap";
      t.replaceWith(w);
      w.appendChild(t);
    });
    $$("img", el).forEach((img) => {
      img.classList.add("zoomable");
    });
  }

  // ---------- parsing the sources ----------
  function splitSections(src) {
    const lines = src.split("\n");
    const pre = [];
    const secs = [];
    let cur = null;
    let fence = false;
    for (const line of lines) {
      if (/^```/.test(line)) fence = !fence;
      if (!fence && /^## /.test(line)) {
        const title = line.slice(3).trim();
        cur = { title, slug: slugify(title), lines: [line] };
        secs.push(cur);
        continue;
      }
      (cur ? cur.lines : pre).push(line);
    }
    secs.forEach((s) => {
      while (s.lines.length && /^(\s*|---)$/.test(s.lines[s.lines.length - 1])) s.lines.pop();
      s.md = s.lines.join("\n");
      const n = s.title.match(/^(\d+)(?:\.\d+)?\.?\s/);
      s.num = n ? +n[1] : null;
    });
    return { pre: pre.join("\n"), secs };
  }

  const parsedCache = {};
  function parsed(ch) {
    if (parsedCache[ch.id]) return parsedCache[ch.id];
    const src = C[ch.file] || "";
    const { pre, secs } = splitSections(src);
    const h1 = (pre.match(/^# (.+)$/m) || [])[1] || ch.file;
    const full = h1.replace(/^\d{2}\s*·\s*/, "");
    const short = full.split(":")[0].trim();
    const watch = secs.find((s) => /^watch and read/i.test(s.title));
    const sheet = secs.find((s) => /^cheat sheet/i.test(s.title));
    const content = secs.filter((s) => s !== watch && s !== sheet);
    const p = { full, short, pre: pre.replace(/^# .+$/m, "").trim(), watch, sheet, content };
    parsedCache[ch.id] = p;
    return p;
  }

  // Cheat sheet by topic: "# A · …" blocks and "## N · …" sections.
  let SHEET = null;
  function sheetData() {
    if (SHEET) return SHEET;
    const src = C["CHEAT_SHEET_BY_TOPIC.md"] || "";
    const { secs } = splitSections(src);
    const list = secs.map((s) => {
      const n = s.title.match(/^(\d+)\s*·\s*(.+)$/);
      const sub = (s.md.match(/<sub>([\s\S]*?)<\/sub>/) || [])[1] || "";
      const chs = [];
      const lm = sub.match(/Lernen\s+([^·]+)/);
      if (lm) (lm[1].match(/\b(0\d|1[01])\b/g) || []).forEach((x) => { if (!chs.includes(x)) chs.push(x); });
      return { title: s.title, num: n ? +n[1] : null, name: n ? n[2] : s.title, slug: s.slug, md: s.md.replace(/\n---\s*$/, ""), chs };
    });
    SHEET = list;
    return list;
  }
  function sheetFor(chId) { return sheetData().filter((s) => s.chs.includes(chId)); }

  // Videos by week from deep-learning-videos.md
  let WEEKS = null;
  function weekVideos() {
    if (WEEKS) return WEEKS;
    WEEKS = {};
    splitSections(C["deep-learning-videos.md"] || "").secs.forEach((s) => {
      const m = s.title.match(/^Week (\d+):\s*(.+)$/);
      if (!m) return;
      const body = s.lines.slice(1).filter((l) => !/^Chapter:/.test(l)).join("\n").trim();
      WEEKS[+m[1]] = { title: m[2], md: body };
    });
    return WEEKS;
  }

  // Self-test questions from QUESTIONS_AND_SUMMARY.md
  let QA = null;
  function qaData() {
    if (QA) return QA;
    QA = {};
    const src = C["QUESTIONS_AND_SUMMARY.md"] || "";
    src.split(/^## /m).slice(1).forEach((part) => {
      const m = part.match(/^(\d{2}) · (.+)\n/);
      if (!m) return;
      const id = m[1];
      const body = part.slice(m[0].length);
      const sm = body.match(/\*\*Summary\*\*\s*\n([\s\S]*?)\n\*\*Questions\*\*/);
      const qs = [];
      const re = /^(\d+)\.\s+([\s\S]*?)\n<details>\s*<summary>Answer<\/summary>([\s\S]*?)<\/details>/gm;
      let x;
      while ((x = re.exec(body))) qs.push({ key: id + "-" + x[1], n: +x[1], ch: id, q: x[2].trim(), a: x[3].trim() });
      QA[id] = { summary: sm ? sm[1].trim() : "", qs };
    });
    return QA;
  }

  // Course tasks: a NOTE callout followed by a <details> answer, taken from the markdown.
  const tasksCache = {};
  function courseTasks(chId) {
    if (tasksCache[chId]) return tasksCache[chId];
    const lines = (C[CH[chId].file] || "").split("\n");
    const out = [];
    for (let i = 0; i < lines.length; i++) {
      if (!/^<details>/.test(lines[i])) continue;
      let j = i - 1;
      while (j >= 0 && !lines[j].trim()) j--;
      let e = i;
      while (e < lines.length && !/<\/details>/.test(lines[e])) e++;
      if (j >= 0 && /^>/.test(lines[j])) {
        let k = j;
        while (k > 0 && /^>/.test(lines[k - 1])) k--;
        const quote = lines.slice(k, j + 1).join("\n");
        if (/From the course/.test(quote)) out.push({ ch: chId, quote, det: lines.slice(i, e + 1).join("\n"), key: chId + "-t" + out.length });
      }
      i = e;
    }
    tasksCache[chId] = out;
    return out;
  }

  // ---------- progress ----------
  const doneKey = (chId, slug) => "done:" + chId + ":" + slug;
  function chapterProgress(chId) {
    const secs = parsed(CH[chId]).content;
    const done = secs.filter((s) => store.get(doneKey(chId, s.slug))).length;
    return { done, total: secs.length };
  }
  function qaStats(chId) {
    const ids = chId ? [chId] : CHAPTERS.map((c) => c.id);
    let total = 0, right = 0, wrong = 0;
    ids.forEach((id) => (qaData()[id] || { qs: [] }).qs.forEach((q) => {
      total++;
      const s = store.get("qa:" + q.key);
      if (s === "right") right++;
      else if (s === "wrong") wrong++;
    }));
    return { total, right, wrong };
  }

  // ---------- shell: nav, routing ----------
  const main = $("#main");
  const sidenav = $("#sidenav");
  let current = { route: "home" };

  function chLabel(id) { return id + " · " + parsed(CH[id]).short; }
  function weeksLabel(w) {
    if (w.length === 1) return "Week " + w[0];
    const sorted = w.slice().sort((a, b) => a - b);
    const consecutive = sorted.every((x, i) => i === 0 || x === sorted[i - 1] + 1);
    return "Weeks " + (consecutive ? sorted[0] + "–" + sorted[sorted.length - 1] : sorted.join(", "));
  }

  function renderNav() {
    const r = current.route;
    let h = `<div class="nav-group">
      <a href="#home" data-route="home" class="nav-item ${r === "home" ? "on" : ""}"><span class="nav-ico">⌂</span>Overview</a>
      <a href="#practice" data-route="practice" class="nav-item ${r === "practice" ? "on" : ""}"><span class="nav-ico">✎</span>Practice</a>
      <a href="#exam" data-route="exam" class="nav-item ${r === "exam" ? "on" : ""}"><span class="nav-ico">§</span>Exam tasks</a>
      <a href="#cheatsheet" data-route="cheatsheet" class="nav-item ${r === "cheatsheet" ? "on" : ""}"><span class="nav-ico">≡</span>Cheat sheet</a>
      <a href="#videos" data-route="videos" class="nav-item ${r === "videos" ? "on" : ""}"><span class="nav-ico">▶</span>Videos by week</a>
    </div>
    <div class="nav-label">Chapters</div>`;
    BLOCKS.forEach((b) => {
      // chapter 01 covers blocks A and B, so B gets no row of its own
      if (b.key === "B") return;
      const chs = b.ch.map((x) => x[0]);
      const letters = b.key === "A" ? "A·B" : b.key;
      const title = b.key === "A" ? "How a network learns · Train it well" : b.title;
      h += `<div class="nav-block"><span class="blk">${letters}</span><span class="blk-title">${esc(title)}</span></div>`;
      chs.forEach((id) => {
        const on = r === "ch-" + id;
        const pr = chapterProgress(id);
        h += `<a href="#ch-${id}" data-route="ch-${id}" class="nav-item nav-ch ${on ? "on" : ""}">
          <span class="nav-num">${id}</span><span class="nav-t">${esc(parsed(CH[id]).short)}</span>
          <span class="nav-pr" title="${pr.done} of ${pr.total} sections done">${pr.done ? Math.round((100 * pr.done) / pr.total) + "%" : ""}</span></a>`;
        if (on) {
          h += `<div class="nav-toc">`;
          parsed(CH[id]).content.forEach((s) => {
            const d = store.get(doneKey(id, s.slug));
            h += `<a href="#ch-${id}" data-route="ch-${id}" data-anchor="${s.slug}" class="${d ? "done" : ""}">${esc(s.title.replace(/\$[^$]*\$/g, "…"))}</a>`;
          });
          h += `<a href="#ch-${id}" data-route="ch-${id}" data-anchor="all-tasks" class="toc-tasks">Tasks</a></div>`;
        }
      });
    });
    sidenav.innerHTML = h;
  }

  const VALID = /^(home|practice|exam|cheatsheet|videos|ch-\d{2})$/;
  function go(route, anchor, opts) {
    opts = opts || {};
    if (!VALID.test(route)) route = "home";
    if (route.startsWith("ch-") && !CH[route.slice(3)]) route = "home";
    const same = current.route === route;
    current = { route, anchor };
    if (!opts.fromHash) {
      try { if (location.hash !== "#" + route) history.replaceState(null, "", "#" + route); } catch (e) { /* sandboxed */ }
    }
    store.set("last", route);
    closeNarrow();
    if (!same || opts.force || !main.firstElementChild || main.querySelector(".loading")) {
      main.innerHTML = "";
      if (route === "home") renderHome();
      else if (route === "practice") renderPractice();
      else if (route === "exam") renderExam();
      else if (route === "cheatsheet") renderCheatPage();
      else if (route === "videos") renderVideos();
      else renderChapter(route.slice(3));
      renderNav();
      main.scrollTop = 0;
    } else renderNav();
    updateSheetContext();
    if (anchor) scrollToAnchor(anchor);
  }

  function scrollToAnchor(anchor) {
    if (anchor === "watch-and-read") anchor = "prep";
    requestAnimationFrame(() => {
      const t = document.getElementById(anchor) || document.getElementById(slugify(anchor));
      if (!t) return;
      t.scrollIntoView({ block: "start" });
      t.classList.add("flash");
      setTimeout(() => t.classList.remove("flash"), 1400);
      // images and formulas above the target change the layout after the jump: jump again once they settle
      const at = main.scrollTop;
      const imgs = $$("img", main).filter((i) => !i.complete).map((i) => new Promise((r) => { i.onload = i.onerror = r; }));
      const wait = Promise.race([Promise.all(imgs.concat([lastTypeset])), new Promise((r) => setTimeout(r, 3000))]);
      wait.then(() => { if (Math.abs(main.scrollTop - at) < 4 && t.isConnected) t.scrollIntoView({ block: "start" }); });
    });
  }

  // ---------- home ----------
  function renderHome() {
    const totalSecs = CHAPTERS.reduce((a, c) => a + chapterProgress(c.id).total, 0);
    const doneSecs = CHAPTERS.reduce((a, c) => a + chapterProgress(c.id).done, 0);
    const qs = qaStats();
    const exA = EX.A.filter((q) => store.get("exA:" + q.id)).length;
    const last = store.get("last", "");
    const lastCh = /^ch-\d\d$/.test(last) && CH[last.slice(3)] ? last : "ch-01";

    let h = `<div class="page home">
      <section class="home-top">
        <div class="home-actions">
          <a class="btn primary" href="#${lastCh}" data-route="${lastCh}">${esc(chLabel(lastCh.slice(3)))}</a>
          <a class="btn" href="#practice" data-route="practice">Practice</a>
          <a class="btn" href="#exam" data-route="exam">Exam tasks</a>
        </div>
        <dl class="stats">
          <div><dt>Sections</dt><dd>${doneSecs}<small> / ${totalSecs}</small></dd><div class="bar"><i style="width:${totalSecs ? (100 * doneSecs) / totalSecs : 0}%"></i></div></div>
          <div><dt>Self-test</dt><dd>${qs.right}<small> / ${qs.total}</small></dd><div class="bar"><i style="width:${qs.total ? (100 * qs.right) / qs.total : 0}%"></i></div></div>
          <div><dt>Part A</dt><dd>${exA}<small> / ${EX.A.length}</small></dd><div class="bar"><i style="width:${EX.A.length ? (100 * exA) / EX.A.length : 0}%"></i></div></div>
        </dl>
      </section>

      <section class="map">
        <ol class="blocks">`;
    BLOCKS.forEach((b) => {
      const sheetSecs = sheetData().filter((s) => s.num && b.ch.some((x) => s.chs[0] === x[0]) && inRangeForBlock(b, s));
      h += `<li class="block">
        <div class="block-letter">${b.key}</div>
        <div class="block-body">
          <h3>${esc(b.title)}</h3>
          <p class="flow">${b.flow.map(esc).join(' <span class="arr">→</span> ')}</p>
          <div class="block-chs">`;
      b.ch.forEach(([id, lo, hi]) => {
        const c = CH[id];
        const p = parsed(c);
        const secs = p.content.filter((s) => lo == null || (s.num != null && s.num >= lo && s.num <= hi));
        const pr = chapterProgress(id);
        const q = qaStats(id);
        h += `<div class="ch-card">
          <a class="ch-card-head" href="#ch-${id}" data-route="ch-${id}">
            <span class="ch-num">${id}</span>
            <span class="ch-name">${esc(p.short)}<small>${weeksLabel(c.weeks)}</small></span>
            <span class="ch-pr">${pr.done}/${pr.total}</span>
          </a>
          <ul class="sec-list">${secs.map((s) => `<li class="${store.get(doneKey(id, s.slug)) ? "done" : ""}"><a href="#ch-${id}" data-route="ch-${id}" data-anchor="${s.slug}">${md(s.title.replace(/^(\d+(?:\.\d+)?)\.?\s/, '<b>$1</b> '), { inline: true })}</a></li>`).join("")}</ul>
          <div class="ch-card-foot">
            <a href="#ch-${id}" data-route="ch-${id}" data-anchor="prep">Videos · Cheat sheet</a>
            <a href="#ch-${id}" data-route="ch-${id}" data-anchor="all-tasks">${q.total + courseTasks(id).length} tasks</a>
          </div>
        </div>`;
      });
      h += `</div>`;
      if (sheetSecs.length) h += `<p class="block-sheet">${sheetSecs.map((s) => `<button class="linkish" data-sheet="${s.slug}">§${s.num} ${esc(s.name.replace(/★/g, "").trim())}</button>`).join(" · ")}</p>`;
      h += `</div></li>`;
    });
    h += `</ol></section>
    </div>`;
    main.innerHTML = h;
    typeset(main);
  }
  function inRangeForBlock(b, s) {
    // chapter 01 is split over blocks A (sheet §1–4) and B (§5–7)
    if (b.key === "A") return s.num >= 1 && s.num <= 4;
    if (b.key === "B") return s.num >= 5 && s.num <= 7;
    return true;
  }

  // ---------- chapter ----------
  function videosHtml(c) {
    const p = parsed(c);
    const W = weekVideos();
    let h = `<div class="vid-weeks">`;
    c.weeks.forEach((w) => {
      if (!W[w]) return;
      h += `<div class="vid-week"><h4>Week ${w} · ${esc(W[w].title)}</h4>${md(W[w].md)}</div>`;
    });
    h += `</div>`;
    if (p.watch) {
      let body = p.watch.lines.slice(1).join("\n");
      body = body.replace(/\*\*Official course videos\*\*[\s\S]*?(?=\*\*More videos and readings:?\*\*|$)/, "").replace(/\*\*More videos and readings:?\*\*/, "").trim();
      if (body) h += `<details class="more-vids"><summary>More videos and readings</summary>${md(body)}</details>`;
    }
    return h;
  }

  function renderChapter(id) {
    const c = CH[id];
    const p = parsed(c);
    const idx = CHAPTERS.indexOf(c);
    const prev = CHAPTERS[idx - 1];
    const next = CHAPTERS[idx + 1];
    const topic = sheetFor(id);
    const qa = qaData()[id] || { summary: "", qs: [] };

    const bodyMd = [p.pre].concat(p.content.map((s) => s.md)).join("\n\n");
    R.seen = {};

    let h = `<article class="page chapter">
      <header class="ch-head">
        <p class="eyebrow">${c.blocks.map((k) => `<span class="blk">${k}</span>`).join("")} ${weeksLabel(c.weeks)}</p>
        <h1>${md(p.full, { inline: true })}</h1>
        <div class="ch-jump">
          <a href="#ch-${id}" data-route="ch-${id}" data-anchor="prep">Videos · Cheat sheet</a>
          <a href="#ch-${id}" data-route="ch-${id}" data-anchor="all-tasks">Tasks (${courseTasks(id).length + qa.qs.length + examFor(id).count})</a>
          <span class="legend" title="Colour code used in every formula"><i style="--c:var(--k-input)"></i>input <i style="--c:var(--k-weight)"></i>weights <i style="--c:var(--k-output)"></i>output <i style="--c:var(--k-label)"></i>label <i style="--c:var(--k-setting)"></i>setting</span>
        </div>
      </header>

      <section class="prep" id="prep">
        <div class="prep-videos">
          <h2 class="prep-h">▶ Videos</h2>
          ${videosHtml(c)}
        </div>
        <div class="prep-sheet">
          <div class="tabs" role="tablist">
            ${p.sheet ? `<button role="tab" class="tab on" data-tab="own">Chapter cheat sheet</button>` : ""}
            ${topic.length ? `<button role="tab" class="tab ${p.sheet ? "" : "on"}" data-tab="topic">Topic sheet ${topic.map((s) => "§" + s.num).join(" ")}</button>` : ""}
            ${qa.summary ? `<button role="tab" class="tab" data-tab="sum">Summary</button>` : ""}
          </div>
          <div class="tab-panels md">
            ${p.sheet ? `<div class="tab-panel" data-panel="own">${md(p.sheet.lines.slice(1).join("\n"), { prefix: "own-" })}</div>` : ""}
            ${topic.length ? `<div class="tab-panel" data-panel="topic" ${p.sheet ? "hidden" : ""}>${topic.map((s) => md(s.md.replace(/^## /, "### "), { prefix: "tp-", keepSeen: true })).join("")}</div>` : ""}
            ${qa.summary ? `<div class="tab-panel" data-panel="sum" hidden>${md(qa.summary)}</div>` : ""}
          </div>
          <button class="expand" data-act="expand">Show all</button>
        </div>
      </section>

      <div class="md ch-body" id="chBody">${md(bodyMd)}</div>

      <section class="all-tasks" id="all-tasks">
        <h2>Tasks</h2>
        <div id="taskCourse"></div>
        <div id="taskQA"></div>
        <div id="taskExam"></div>
      </section>

      <nav class="pager">
        ${prev ? `<a href="#ch-${prev.id}" data-route="ch-${prev.id}"><small>← Previous</small>${esc(chLabel(prev.id))}</a>` : "<span></span>"}
        ${next ? `<a class="next" href="#ch-${next.id}" data-route="ch-${next.id}"><small>Next →</small>${esc(chLabel(next.id))}</a>` : `<a class="next" href="#exam" data-route="exam"><small>Next →</small>Exam tasks</a>`}
      </nav>
    </article>`;
    main.innerHTML = h;
    const art = $(".chapter", main);
    post(art);

    // section "done" toggles
    $$("h2", $("#chBody")).forEach((h2) => {
      if (h2.closest(".callout")) return;
      const s = p.content.find((x) => x.slug === h2.id.replace(/-\d+$/, "") || x.slug === h2.id);
      if (!s) return;
      const key = doneKey(id, s.slug);
      const b = document.createElement("button");
      b.className = "done-btn" + (store.get(key) ? " on" : "");
      b.dataset.done = key;
      b.textContent = store.get(key) ? "✓ Done" : "Mark done";
      h2.appendChild(b);
    });

    // course tasks: number the inline ones and repeat them below
    const inline = $$("details", $("#chBody")).filter((d) => {
      const prevEl = d.previousElementSibling;
      return prevEl && prevEl.classList.contains("course-task");
    });
    let ct = "";
    if (inline.length) {
      ct = `<h3 class="tg-h">From the course <span class="count">${inline.length}</span></h3><div class="task-list">`;
      inline.forEach((d, i) => {
        const q = d.previousElementSibling;
        q.id = "task-" + (i + 1);
        const tag = document.createElement("span");
        tag.className = "task-no";
        tag.textContent = "Task " + (i + 1);
        q.prepend(tag);
        const sec = sectionOf(q);
        ct += `<div class="task-card"><div class="task-meta"><span>Task ${i + 1}</span>${sec ? `<a href="#ch-${id}" data-route="ch-${id}" data-anchor="task-${i + 1}">in ${esc(sec)} ↑</a>` : ""}</div>
          <div class="task-q">${q.innerHTML.replace(/<span class="task-no">[\s\S]*?<\/span>/, "")}</div>${d.outerHTML}</div>`;
      });
      ct += `</div>`;
    }
    $("#taskCourse").innerHTML = ct;
    $("#taskQA").innerHTML = qa.qs.length ? `<h3 class="tg-h">Self-test <span class="count">${qa.qs.length}</span></h3><div class="qa-list">${qa.qs.map(qaCard).join("")}</div>` : "";
    const ex = examFor(id);
    $("#taskExam").innerHTML = ex.count ? `<h3 class="tg-h">Exam tasks <span class="count">${ex.count}</span></h3>
      ${ex.A.length ? `<h4 class="part-h">Part A · Multiple choice</h4>${ex.A.map((q) => cardA(q)).join("")}` : ""}
      ${ex.B.length ? `<h4 class="part-h">Part B · Short answer</h4>${ex.B.map((q) => cardB(q)).join("")}` : ""}
      ${ex.C.length ? `<h4 class="part-h">Part C · Case study</h4>${ex.C.map((q) => cardC(q)).join("")}` : ""}` : "";
    post($("#all-tasks"));
    typeset(art);
  }

  function sectionOf(el) {
    let n = el;
    while (n && n.parentElement && n.parentElement.id !== "chBody") n = n.parentElement;
    while (n) {
      if (n.tagName === "H2") return n.childNodes[0] ? n.childNodes[0].textContent.trim() : n.textContent;
      n = n.previousElementSibling;
    }
    return "";
  }

  function examFor(id) {
    const A = EX.A.filter((q) => q.ch.includes(id));
    const B = EX.B.filter((q) => q.ch.includes(id));
    const Cc = EX.C.filter((q) => q.ch.includes(id));
    return { A, B, C: Cc, count: A.length + B.length + Cc.length };
  }

  // ---------- self-test cards ----------
  function stripTags(s) { return String(s).replace(/<[^>]+>/g, "").replace(/\*\*/g, ""); }
  function qaTruth(q) {
    if (!/^true or false/i.test(q.q)) return null;
    const a = stripTags(q.a).trim();
    if (/^true\b/i.test(a)) return true;
    if (/^false\b/i.test(a)) return false;
    return null;
  }
  function qaCard(q, showCh) {
    const st = store.get("qa:" + q.key);
    const truth = qaTruth(q);
    const qText = truth === null ? q.q : q.q.replace(/^true or false[:,]?\s*/i, "").replace(/^./, (m) => m.toUpperCase());
    return `<div class="qa-card ${st || ""}" data-qa="${q.key}">
      <div class="qa-meta">${showCh ? `<a class="chip" href="#ch-${q.ch}" data-route="ch-${q.ch}">${esc(chLabel(q.ch))}</a>` : ""}<span class="qa-n">Q${q.n}</span>${truth !== null ? `<span class="qa-type">True or false</span>` : ""}<span class="pill">${st === "right" ? "Got it" : st === "wrong" ? "Not yet" : ""}</span></div>
      <div class="qa-q">${md(qText)}</div>
      <div class="qa-actions">${truth !== null
        ? `<button class="btn sm" data-act="tf" data-v="true">True</button><button class="btn sm" data-act="tf" data-v="false">False</button>`
        : `<button class="btn sm" data-act="reveal">Show answer</button>`}</div>
      <div class="qa-a" hidden>
        <div class="verdict"></div>
        <div class="md">${md(q.a)}</div>
        <div class="self"><button class="btn sm good" data-act="mark" data-v="right">Got it</button><button class="btn sm bad" data-act="mark" data-v="wrong">Not yet</button></div>
      </div>
    </div>`;
  }

  // ---------- exam cards ----------
  const ROMAN = ["i", "ii", "iii", "iv", "v", "vi"];
  const srcTag = (q) => q.src === "practice" ? `<span class="tag practice">Practice exam</span>` : `<span class="tag extra">Extra</span>`;
  const chChips = (q) => q.ch.map((c) => `<a class="chip" href="#ch-${c}" data-route="ch-${c}">${c}</a>`).join("");

  function cardA(q, mock) {
    const saved = mock ? (mockState.answers[q.id] || []) : (store.get("exAsel:" + q.id) || []);
    return `<div class="ex-card exA" data-a="${q.id}" ${mock ? 'data-mock="1"' : ""}>
      <div class="ex-meta">${srcTag(q)}<b>${q.id}</b><span>${q.correct.length} mark${q.correct.length > 1 ? "s" : ""}</span>${chChips(q)}<span class="ex-score"></span></div>
      <p class="ex-q">${plainWithMath(q.q)}</p>
      <ol class="opts">${q.options.map((o, i) => `<li><label><input type="checkbox" id="opt-${mock ? "m-" : ""}${q.id}-${i}" data-i="${i}" ${saved.includes(i) ? "checked" : ""}><span class="rn">(${ROMAN[i]})</span><span>${plainWithMath(o)}</span></label></li>`).join("")}</ol>
      ${mock ? "" : `<div class="ex-actions"><button class="btn sm primary" data-act="checkA">Check</button><button class="btn sm" data-act="resetA">Reset</button></div>`}
      <div class="ex-sol" hidden><b>Solution:</b> ${q.correct.map((i) => "(" + ROMAN[i] + ")").join(", ")}${q.why ? `<p>${plainWithMath(q.why)}</p>` : ""}</div>
    </div>`;
  }

  function scoreA(card) {
    const q = EX.A.find((x) => x.id === card.dataset.a);
    const sel = $$("input[type=checkbox]", card).filter((i) => i.checked).map((i) => +i.dataset.i);
    let s = 0;
    $$("li", card).forEach((li, i) => {
      const isC = q.correct.includes(i);
      const picked = sel.includes(i);
      li.className = picked && isC ? "ok" : picked && !isC ? "no" : !picked && isC ? "missed" : "";
      if (picked) s += isC ? 1 : -1;
    });
    s = Math.max(0, s);
    const max = q.correct.length;
    $(".ex-score", card).innerHTML = `<b class="${s === max ? "full" : s ? "part" : "zero"}">${s} / ${max}</b>`;
    $(".ex-sol", card).hidden = false;
    return { s, max, sel };
  }

  function draftArea(key, mock) {
    const v = mock ? (mockState.answers[key] || "") : store.get("draft:" + key, "");
    return `<textarea class="answer" id="ans-${mock ? "m-" : ""}${key.replace(/[^\w-]/g, "_")}" data-draft="${key}" ${mock ? 'data-mock="1"' : ""} rows="4" placeholder="Answer">${esc(v)}</textarea>`;
  }
  function selfMark(key) {
    const v = store.get("exS:" + key);
    return `<div class="self">
      <button class="btn sm good ${v === 2 ? "on" : ""}" data-act="markB" data-k="${key}" data-v="2">Full</button>
      <button class="btn sm mid ${v === 1 ? "on" : ""}" data-act="markB" data-k="${key}" data-v="1">Partly</button>
      <button class="btn sm bad ${v === 0 ? "on" : ""}" data-act="markB" data-k="${key}" data-v="0">Missed</button></div>`;
  }

  function cardB(q, mock) {
    return `<div class="ex-card exB" data-b="${q.id}">
      <div class="ex-meta">${srcTag(q)}<b>${q.id}</b>${chChips(q)}</div>
      <p class="ex-q">${plainWithMath(q.q)}</p>
      ${draftArea(q.id, mock)}
      ${mock ? "" : `<div class="ex-actions"><button class="btn sm primary" data-act="revealB">Show sample solution</button></div>`}
      <div class="ex-sol" hidden><b>Sample solution</b><p>${plainWithMath(q.a)}</p>${selfMark(q.id)}</div>
    </div>`;
  }

  function cardC(q, mock) {
    return `<div class="ex-card exC" data-c="${q.id}">
      <div class="ex-meta">${srcTag(q)}<b>${q.id}</b>${chChips(q)}</div>
      <h4 class="case-title">Problem ${esc(q.id)} · ${esc(q.title)}</h4>
      <p class="scenario">${plainWithMath(q.scenario)}</p>
      ${q.parts.map((pt, i) => {
        const key = q.id + "-" + i;
        return `<div class="case-part">
          <h5>${esc(pt.label)}</h5>
          <p class="ex-q">${plainWithMath(pt.q)}</p>
          ${draftArea(key, mock)}
          ${mock ? "" : `<div class="ex-actions"><button class="btn sm primary" data-act="revealB">Show sample solution</button></div>`}
          <div class="ex-sol" hidden><b>Sample solution</b><p>${plainWithMath(pt.a)}</p>${selfMark(key)}</div>
        </div>`;
      }).join("")}
    </div>`;
  }

  // ---------- practice page ----------
  const pState = { ch: "all", status: "all", shuffle: false, course: true };
  function renderPractice() {
    const QAD = qaData();
    const order = CHAPTERS.map((c) => c.id);
    const st = qaStats();
    let h = `<div class="page practice">
      <header class="pg-head">
        <h1>Practice</h1>
        <div class="score-row"><span><b>${st.right}</b> got it</span><span><b>${st.wrong}</b> not yet</span><span><b>${st.total - st.right - st.wrong}</b> new</span><button class="linkish" data-act="resetQA">Reset</button></div>
      </header>
      <div class="filters">
        <div class="chips" role="group" aria-label="Chapter">
          <button class="fchip ${pState.ch === "all" ? "on" : ""}" data-pch="all">All chapters</button>
          ${order.map((id) => `<button class="fchip ${pState.ch === id ? "on" : ""}" data-pch="${id}" title="${esc(chLabel(id))}">${id} ${esc(CH[id].nick)}</button>`).join("")}
        </div>
        <div class="filter-row">
          <label for="pStatus">Show</label>
          <select id="pStatus">
            <option value="all" ${pState.status === "all" ? "selected" : ""}>All</option>
            <option value="new" ${pState.status === "new" ? "selected" : ""}>New</option>
            <option value="wrong" ${pState.status === "wrong" ? "selected" : ""}>Not yet</option>
            <option value="right" ${pState.status === "right" ? "selected" : ""}>Got it</option>
          </select>
          <label class="check"><input type="checkbox" id="pShuffle" ${pState.shuffle ? "checked" : ""}> Shuffle</label>
          <label class="check"><input type="checkbox" id="pCourse" ${pState.course ? "checked" : ""}> Course tasks</label>
        </div>
      </div>
      <div id="pList"></div>
    </div>`;
    main.innerHTML = h;
    fillPractice();
  }

  function fillPractice() {
    const QAD = qaData();
    const ids = pState.ch === "all" ? CHAPTERS.map((c) => c.id) : [pState.ch];
    let items = [];
    ids.forEach((id) => {
      (QAD[id] || { qs: [] }).qs.forEach((q) => items.push({ t: "qa", q, ch: id }));
      if (pState.course) courseTasks(id).forEach((t) => items.push({ t: "task", q: t, ch: id }));
    });
    items = items.filter((it) => {
      if (pState.status === "all") return true;
      const s = store.get((it.t === "qa" ? "qa:" : "task:") + (it.q.key));
      return pState.status === "new" ? !s : s === pState.status;
    });
    if (pState.shuffle) {
      for (let i = items.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [items[i], items[j]] = [items[j], items[i]]; }
    }
    let h = `<p class="muted count-line">${items.length} item${items.length === 1 ? "" : "s"}</p>`;
    if (!items.length) h += `<p class="empty">None</p>`;
    let lastCh = null;
    items.forEach((it) => {
      if (!pState.shuffle && it.ch !== lastCh) {
        lastCh = it.ch;
        const sum = (QAD[it.ch] || {}).summary;
        h += `<div class="p-ch"><h2><a href="#ch-${it.ch}" data-route="ch-${it.ch}">${esc(chLabel(it.ch))}</a></h2>${sum ? `<details class="p-sum"><summary>Summary</summary><div class="md">${md(sum)}</div></details>` : ""}</div>`;
      }
      if (it.t === "qa") h += qaCard(it.q, pState.shuffle);
      else h += taskCard(it.q, pState.shuffle);
    });
    const list = $("#pList");
    list.innerHTML = h;
    post(list);
    typeset(list);
  }

  function taskCard(t, showCh) {
    const st = store.get("task:" + t.key);
    const det = t.det.replace(/<summary>[\s\S]*?<\/summary>/, "<summary><b>Show the answer</b></summary>");
    return `<div class="qa-card task ${st || ""}" data-task="${t.key}">
      <div class="qa-meta">${showCh ? `<a class="chip" href="#ch-${t.ch}" data-route="ch-${t.ch}">${esc(chLabel(t.ch))}</a>` : ""}<span class="qa-type">Course task</span><span class="pill">${st === "right" ? "Got it" : st === "wrong" ? "Not yet" : ""}</span></div>
      <div class="md">${md(t.quote)}${md(det)}</div>
      <div class="self"><button class="btn sm good" data-act="markT" data-v="right">Got it</button><button class="btn sm bad" data-act="markT" data-v="wrong">Not yet</button></div>
    </div>`;
  }

  // ---------- exam page ----------
  const eState = { tab: "A", ch: "all", src: "all" };
  const mockState = { running: false, submitted: false, items: null, end: 0, minutes: 60, nA: 8, nB: 4, nC: 1, answers: {}, timer: null };

  function examFilter(list) {
    return list.filter((q) => (eState.ch === "all" || q.ch.includes(eState.ch)) && (eState.src === "all" || q.src === eState.src));
  }

  function renderExam() {
    const tabs = [["A", "Part A · Multiple choice", EX.A.length], ["B", "Part B · Short answer", EX.B.length], ["C", "Part C · Case studies", EX.C.length], ["M", "Mock exam", null]];
    let h = `<div class="page exam">
      <header class="pg-head">
        <h1>Exam tasks</h1>
      </header>
      <div class="tabs big" role="tablist">${tabs.map(([k, l, n]) => `<button role="tab" class="tab ${eState.tab === k ? "on" : ""}" data-etab="${k}">${l}${n != null ? ` <span class="count">${n}</span>` : ""}</button>`).join("")}</div>`;
    if (eState.tab !== "M") {
      h += `<div class="filter-row">
        <label for="eCh">Chapter</label>
        <select id="eCh"><option value="all">All chapters</option>${CHAPTERS.map((c) => `<option value="${c.id}" ${eState.ch === c.id ? "selected" : ""}>${c.id} ${esc(c.nick)}</option>`).join("")}</select>
        <label for="eSrc">Source</label>
        <select id="eSrc">
          <option value="all" ${eState.src === "all" ? "selected" : ""}>All</option>
          <option value="practice" ${eState.src === "practice" ? "selected" : ""}>Practice exam</option>
          <option value="extra" ${eState.src === "extra" ? "selected" : ""}>Extra</option>
        </select>
        ${eState.tab === "A" ? `<span class="a-total" id="aTotal"></span>` : ""}
      </div><div id="eList"></div>`;
    } else h += `<div id="mock"></div>`;
    h += `</div>`;
    main.innerHTML = h;
    if (eState.tab === "M") renderMock();
    else fillExam();
  }

  function fillExam() {
    const list = $("#eList");
    let h = "";
    if (eState.tab === "A") {
      const items = examFilter(EX.A);
      h = items.map((q) => cardA(q)).join("") || `<p class="empty">None</p>`;
    } else if (eState.tab === "B") {
      const items = examFilter(EX.B);
      h = items.map((q) => cardB(q)).join("") || `<p class="empty">None</p>`;
    } else {
      const items = examFilter(EX.C);
      h = items.map((q) => cardC(q)).join("") || `<p class="empty">None</p>`;
    }
    list.innerHTML = h;
    if (eState.tab === "A") {
      $$(".exA", list).forEach((card) => { if (store.get("exA:" + card.dataset.a)) scoreA(card); });
      updateATotal();
    }
    typeset(list);
  }

  function updateATotal() {
    const el = $("#aTotal");
    if (!el) return;
    let s = 0, m = 0, n = 0;
    examFilter(EX.A).forEach((q) => { const r = store.get("exA:" + q.id); if (r) { s += r.s; m += r.max; n++; } });
    el.textContent = n ? `${s} / ${m}` : "";
  }

  function pick(list, n) {
    const a = list.slice();
    for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
    return a.slice(0, Math.min(n, a.length));
  }

  function renderMock() {
    const box = $("#mock");
    const M = mockState;
    if (!M.running && !M.submitted) {
      box.innerHTML = `<div class="mock-setup">
        <div class="mock-form">
          <label for="mA">Part A</label><input id="mA" type="number" min="0" max="${EX.A.length}" value="${M.nA}">
          <label for="mB">Part B</label><input id="mB" type="number" min="0" max="${EX.B.length}" value="${M.nB}">
          <label for="mC">Part C</label><input id="mC" type="number" min="0" max="${EX.C.length}" value="${M.nC}">
          <label for="mT">Minutes</label><input id="mT" type="number" min="5" max="180" value="${M.minutes}">
        </div>
        <label class="check"><input type="checkbox" id="mPractice"> Practice exam only</label>
        <div><button class="btn primary" data-act="mockStart">Start</button></div>
      </div>`;
      return;
    }
    let h = `<div class="mock-bar"><span id="mockTimer" class="timer"></span>`;
    h += M.submitted ? `<span id="mockScore"></span><button class="btn sm" data-act="mockNew">New</button>` : `<button class="btn sm primary" data-act="mockSubmit">Submit</button>`;
    h += `</div>`;
    const A = M.items.A, B = M.items.B, Cc = M.items.C;
    if (A.length) h += `<h3 class="part-h">Part A · Multiple choice</h3>` + A.map((q) => cardA(q, true)).join("");
    if (B.length) h += `<h3 class="part-h">Part B · Short answer</h3>` + B.map((q) => cardB(q, true)).join("");
    if (Cc.length) h += `<h3 class="part-h">Part C · Problem solving</h3>` + Cc.map((q) => cardC(q, true)).join("");
    box.innerHTML = h;
    if (M.submitted) showMockResults();
    tickMock();
    typeset(box);
  }

  function tickMock() {
    const M = mockState;
    const el = $("#mockTimer");
    if (!el) return;
    if (M.submitted) { el.textContent = "Submitted"; return; }
    const left = Math.max(0, Math.round((M.end - Date.now()) / 1000));
    const mm = Math.floor(left / 60), ss = left % 60;
    el.textContent = `${mm}:${String(ss).padStart(2, "0")}`;
    el.classList.toggle("low", left < 300);
    if (left === 0) submitMock();
  }

  function submitMock() {
    const M = mockState;
    if (!M.running) return;
    M.running = false;
    M.submitted = true;
    clearInterval(M.timer);
    if (current.route === "exam" && eState.tab === "M") renderMock();
  }

  function showMockResults() {
    let s = 0, m = 0;
    $$("#mock .exA").forEach((card) => { const r = scoreA(card); s += r.s; m += r.max; });
    $$("#mock .ex-sol").forEach((x) => (x.hidden = false));
    $$("#mock input, #mock textarea").forEach((x) => (x.readOnly = true));
    $$("#mock input[type=checkbox]").forEach((x) => (x.disabled = true));
    const el = $("#mockScore");
    if (el) el.innerHTML = `Part A: <b>${s} / ${m}</b>`;
  }

  // ---------- cheat sheet page and videos page ----------
  function renderCheatPage() {
    main.innerHTML = `<div class="page cheat-page"><div class="md" id="cheatFull">${md(C["CHEAT_SHEET_BY_TOPIC.md"] || "", { prefix: "cp-" })}</div></div>`;
    post(main);
    typeset(main);
  }
  function renderVideos() {
    main.innerHTML = `<div class="page videos-page"><div class="md">${md(C["deep-learning-videos.md"] || "")}</div></div>`;
    post(main);
  }

  // ---------- right-hand cheat sheet ----------
  const sheet = $("#sheet");
  const sheetBody = $("#sheetBody");
  let sheetBuilt = false;
  let hits = [], hitIdx = -1;

  function buildSheet() {
    if (sheetBuilt) return;
    sheetBuilt = true;
    sheetBody.innerHTML = md(C["CHEAT_SHEET_BY_TOPIC.md"] || "", { prefix: "cs-" });
    post(sheetBody);
    const nav = $("#sheetNav");
    let opts = `<option value="">Sections</option>`;
    let open = false;
    $$("h1, h2", sheetBody).forEach((hd) => {
      if (hd.tagName === "H1") {
        if (open) opts += `</optgroup>`;
        opts += `<optgroup label="${esc(hd.textContent)}">`;
        open = true;
      } else opts += `<option value="${hd.id}">${esc(hd.textContent)}</option>`;
    });
    if (open) opts += `</optgroup>`;
    nav.innerHTML = opts;
    typeset(sheetBody);
  }

  const wide = () => window.matchMedia("(min-width: 1280px)").matches;
  function setSheet(open) {
    if (open) buildSheet();
    document.body.classList.toggle("sheet-open", open);
    if (wide()) store.set("sheetOpen", open);
    $("#scrim").hidden = !(open && !wide()) && !document.body.classList.contains("nav-open");
  }
  function sheetIsOpen() { return document.body.classList.contains("sheet-open"); }

  function sheetGo(slug) {
    setSheet(true);
    const t = document.getElementById(slug.startsWith("cs-") ? slug : "cs-" + slug);
    if (t) {
      sheetBody.scrollTop = t.offsetTop - 8;
      t.classList.add("flash");
      setTimeout(() => t.classList.remove("flash"), 1400);
    }
  }

  function updateSheetContext() {
    let ctx = $("#sheetCtx");
    if (!ctx) {
      ctx = document.createElement("div");
      ctx.id = "sheetCtx";
      ctx.className = "sheet-ctx";
      $(".sheet-head").appendChild(ctx);
    }
    const m = current.route.match(/^ch-(\d\d)$/);
    const list = m ? sheetFor(m[1]) : [];
    ctx.innerHTML = list.length ? `${list.map((s) => `<button class="fchip sm" data-sheet="${s.slug}">§${s.num} ${esc(s.name.replace(/★/g, "").trim())}</button>`).join("")}` : "";
  }

  function clearHits() {
    $$("mark.hit", sheetBody).forEach((mk) => mk.replaceWith(document.createTextNode(mk.textContent)));
    sheetBody.normalize();
    hits = [];
    hitIdx = -1;
  }
  function sheetSearch(q) {
    buildSheet();
    clearHits();
    const cnt = $("#sheetCount");
    q = q.trim();
    if (q.length < 2) { cnt.textContent = ""; return; }
    const re = new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi");
    const walker = document.createTreeWalker(sheetBody, NodeFilter.SHOW_TEXT, {
      acceptNode(n) {
        if (!n.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
        if (n.parentElement.closest("mjx-container, script, style")) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach((n) => {
      const text = n.nodeValue;
      re.lastIndex = 0;
      if (!re.test(text)) return;
      re.lastIndex = 0;
      const frag = document.createDocumentFragment();
      let last = 0, m;
      while ((m = re.exec(text))) {
        if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)));
        const mk = document.createElement("mark");
        mk.className = "hit";
        mk.textContent = m[0];
        frag.appendChild(mk);
        last = m.index + m[0].length;
      }
      if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
      n.replaceWith(frag);
    });
    hits = $$("mark.hit", sheetBody);
    // open collapsed answers that contain hits
    hits.forEach((h) => { const d = h.closest("details"); if (d) d.open = true; });
    cnt.textContent = hits.length ? `1 / ${hits.length}` : "0 found";
    if (hits.length) focusHit(0);
  }
  function focusHit(i) {
    if (!hits.length) return;
    if (hitIdx >= 0 && hits[hitIdx]) hits[hitIdx].classList.remove("now");
    hitIdx = (i + hits.length) % hits.length;
    const h = hits[hitIdx];
    h.classList.add("now");
    const top = h.getBoundingClientRect().top - sheetBody.getBoundingClientRect().top + sheetBody.scrollTop;
    sheetBody.scrollTop = top - sheetBody.clientHeight / 3;
    $("#sheetCount").textContent = `${hitIdx + 1} / ${hits.length}`;
  }

  // ---------- global search ----------
  let INDEX = null;
  function plain(s) {
    return String(s)
      .replace(/```[\s\S]*?```/g, (m) => m.replace(/```\w*/g, " "))
      .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
      .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
      .replace(/<[^>]+>/g, " ")
      .replace(/\\textcolor\{[^}]*\}/g, "")
      .replace(/\\(text|mathbf|mathrm|operatorname)\b/g, "")
      .replace(/[*_`#>|$\\{}]/g, " ")
      .replace(/\[!\w+\]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }
  function buildIndex() {
    if (INDEX) return INDEX;
    INDEX = [];
    const addFrom = (src, route, label, prefix) => {
      const lines = src.split("\n");
      let cur = null, fence = false;
      const seen = {};
      lines.forEach((l) => {
        if (/^```/.test(l)) fence = !fence;
        const m = !fence && l.match(/^(#{1,3})\s+(.+)$/);
        if (m) {
          let id = prefix + slugify(m[2]);
          if (seen[id]) { const n = seen[id]++; id += "-" + n; } else seen[id] = 1;
          cur = { route, anchor: id, title: plain(m[2]), label, text: "" };
          INDEX.push(cur);
        } else if (cur) cur.text += " " + l;
      });
    };
    CHAPTERS.forEach((c) => {
      const p = parsed(c);
      const body = [p.pre].concat(p.content.map((s) => s.md)).join("\n\n");
      addFrom(body, "ch-" + c.id, chLabel(c.id), "");
    });
    addFrom(C["CHEAT_SHEET_BY_TOPIC.md"] || "", "cheatsheet", "Cheat sheet", "cp-");
    INDEX.forEach((e) => { e.text = plain(e.text); e.lt = e.title.toLowerCase(); e.lx = e.text.toLowerCase(); });
    return INDEX;
  }
  function searchAll(q) {
    const terms = q.toLowerCase().split(/\s+/).filter((t) => t.length > 1);
    if (!terms.length) return [];
    const res = [];
    buildIndex().forEach((e) => {
      let score = 0;
      for (const t of terms) {
        const inT = e.lt.includes(t);
        const inX = e.lx.includes(t);
        if (!inT && !inX) return;
        score += (inT ? 10 : 0) + (inX ? Math.min(5, e.lx.split(t).length - 1) : 0);
      }
      res.push({ e, score });
    });
    res.sort((a, b) => b.score - a.score);
    return res.slice(0, 14).map((r) => {
      const e = r.e;
      const i = e.lx.indexOf(terms[0]);
      let snip = i >= 0 ? e.text.slice(Math.max(0, i - 50), i + 110) : e.text.slice(0, 140);
      snip = esc(snip);
      terms.forEach((t) => { snip = snip.replace(new RegExp(t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi"), (m) => `<mark>${m}</mark>`); });
      return { e, snip };
    });
  }
  let gIdx = -1;
  function showResults(q) {
    const box = $("#globalResults");
    const r = searchAll(q);
    gIdx = -1;
    if (!q.trim()) { box.hidden = true; return; }
    box.innerHTML = r.length ? r.map((x, i) => `<a href="#${x.e.route}" class="res" data-route="${x.e.route}" data-anchor="${x.e.anchor}" data-i="${i}"><small>${esc(x.e.label)}</small><b>${esc(x.e.title)}</b><span>…${x.snip}…</span></a>`).join("") : `<div class="res none">No results</div>`;
    box.hidden = false;
  }

  // ---------- events ----------
  function closeNarrow() {
    document.body.classList.remove("nav-open");
    if (!wide() && sheetIsOpen()) setSheet(false);
    $("#scrim").hidden = true;
  }

  document.addEventListener("click", (e) => {
    const t = e.target;
    const zoom = t.closest("img.zoomable");
    if (zoom && !t.closest("a")) {
      const ov = document.createElement("div");
      ov.className = "zoom";
      ov.innerHTML = `<img src="${zoom.getAttribute("src")}" alt="${esc(zoom.alt || "")}"><p>${esc(zoom.alt || "")}</p>`;
      ov.addEventListener("click", () => ov.remove());
      document.body.appendChild(ov);
      return;
    }
    const sh = t.closest("[data-sheet]");
    if (sh) { e.preventDefault(); sheetGo(sh.dataset.sheet); return; }
    const pch = t.closest("[data-pch]");
    if (pch && current.route === "practice" && !pch.dataset.route) {
      pState.ch = pch.dataset.pch;
      $$(".fchip[data-pch]").forEach((b) => b.classList.toggle("on", b.dataset.pch === pState.ch));
      fillPractice();
      return;
    }
    const link = t.closest("[data-route]");
    if (link) {
      e.preventDefault();
      if (link.dataset.pch) pState.ch = link.dataset.pch;
      $("#globalResults").hidden = true;
      go(link.dataset.route, link.dataset.anchor, link.dataset.pch ? { force: true } : {});
      return;
    }
    const tab = t.closest(".prep .tab");
    if (tab) {
      const box = tab.closest(".prep-sheet");
      $$(".tab", box).forEach((b) => b.classList.toggle("on", b === tab));
      $$(".tab-panel", box).forEach((p) => (p.hidden = p.dataset.panel !== tab.dataset.tab));
      return;
    }
    const etab = t.closest("[data-etab]");
    if (etab) { eState.tab = etab.dataset.etab; renderExam(); return; }
    const done = t.closest("[data-done]");
    if (done) {
      const k = done.dataset.done;
      const v = !store.get(k);
      if (v) store.set(k, true); else store.del(k);
      done.classList.toggle("on", v);
      done.textContent = v ? "✓ Done" : "Mark done";
      renderNav();
      return;
    }
    const act = t.closest("[data-act]");
    if (!act) {
      if (!t.closest(".search-wrap")) $("#globalResults").hidden = true;
      return;
    }
    const a = act.dataset.act;
    const qa = act.closest(".qa-card");
    if (a === "expand") {
      const box = act.closest(".prep-sheet");
      const on = box.classList.toggle("expanded");
      act.textContent = on ? "Show less" : "Show all";
    } else if (a === "tf" && qa) {
      const q = findQA(qa.dataset.qa);
      const truth = qaTruth(q);
      const ok = (act.dataset.v === "true") === truth;
      $$("[data-act=tf]", qa).forEach((b) => { b.disabled = true; b.classList.toggle("picked", b === act); });
      const v = $(".verdict", qa);
      v.className = "verdict " + (ok ? "good" : "bad");
      v.textContent = ok ? "Correct" : "Wrong";
      $(".qa-a", qa).hidden = false;
      $(".self", qa).hidden = true;
      setQA(qa, ok ? "right" : "wrong");
    } else if (a === "reveal" && qa) {
      $(".qa-a", qa).hidden = false;
      act.hidden = true;
    } else if (a === "mark" && qa) {
      setQA(qa, act.dataset.v);
    } else if (a === "markT" && qa) {
      store.set("task:" + qa.dataset.task, act.dataset.v);
      qa.classList.remove("right", "wrong");
      qa.classList.add(act.dataset.v);
      $(".pill", qa).textContent = act.dataset.v === "right" ? "Got it" : "Not yet";
    } else if (a === "resetQA") {
      Object.values(qaData()).forEach((x) => x.qs.forEach((q) => store.del("qa:" + q.key)));
      CHAPTERS.forEach((c) => courseTasks(c.id).forEach((tk) => store.del("task:" + tk.key)));
      renderPractice();
    } else if (a === "checkA") {
      const card = act.closest(".exA");
      const r = scoreA(card);
      store.set("exA:" + card.dataset.a, { s: r.s, max: r.max });
      store.set("exAsel:" + card.dataset.a, r.sel);
      updateATotal();
    } else if (a === "resetA") {
      const card = act.closest(".exA");
      store.del("exA:" + card.dataset.a);
      store.del("exAsel:" + card.dataset.a);
      $$("input", card).forEach((i) => (i.checked = false));
      $$("li", card).forEach((li) => (li.className = ""));
      $(".ex-score", card).innerHTML = "";
      $(".ex-sol", card).hidden = true;
      updateATotal();
    } else if (a === "revealB") {
      const sol = act.closest(".ex-actions").nextElementSibling;
      sol.hidden = false;
      act.hidden = true;
    } else if (a === "markB") {
      store.set("exS:" + act.dataset.k, +act.dataset.v);
      $$("[data-act=markB]", act.parentElement).forEach((b) => b.classList.toggle("on", b === act));
    } else if (a === "mockStart") {
      const M = mockState;
      const onlyPractice = $("#mPractice").checked;
      M.nA = +$("#mA").value || 0; M.nB = +$("#mB").value || 0; M.nC = +$("#mC").value || 0;
      M.minutes = Math.max(5, +$("#mT").value || 60);
      M.items = onlyPractice
        ? { A: EX.A.filter((q) => q.src === "practice"), B: EX.B.filter((q) => q.src === "practice"), C: EX.C.filter((q) => q.src === "practice") }
        : { A: pick(EX.A, M.nA), B: pick(EX.B, M.nB), C: pick(EX.C, M.nC) };
      M.answers = {};
      M.running = true; M.submitted = false;
      M.end = Date.now() + M.minutes * 60000;
      clearInterval(M.timer);
      M.timer = setInterval(tickMock, 1000);
      renderMock();
      main.scrollTop = 0;
    } else if (a === "mockSubmit") {
      submitMock();
      main.scrollTop = 0;
    } else if (a === "mockNew") {
      mockState.submitted = false; mockState.running = false;
      renderMock();
    }
  });

  function findQA(key) {
    const id = key.split("-")[0];
    return (qaData()[id] || { qs: [] }).qs.find((q) => q.key === key);
  }
  function setQA(card, v) {
    store.set("qa:" + card.dataset.qa, v);
    card.classList.remove("right", "wrong");
    card.classList.add(v);
    $(".pill", card).textContent = v === "right" ? "Got it" : "Not yet";
    $$("[data-act=mark]", card).forEach((b) => b.classList.toggle("on", b.dataset.v === v));
  }

  document.addEventListener("change", (e) => {
    const t = e.target;
    if (t.id === "pStatus") { pState.status = t.value; fillPractice(); }
    else if (t.id === "pShuffle") { pState.shuffle = t.checked; fillPractice(); }
    else if (t.id === "pCourse") { pState.course = t.checked; fillPractice(); }
    else if (t.id === "eCh") { eState.ch = t.value; fillExam(); }
    else if (t.id === "eSrc") { eState.src = t.value; fillExam(); }
    else if (t.id === "sheetNav" && t.value) { sheetGo(t.value); t.value = ""; }
    else if (t.type === "checkbox" && t.closest(".exA[data-mock]")) {
      const card = t.closest(".exA");
      mockState.answers[card.dataset.a] = $$("input[type=checkbox]", card).filter((i) => i.checked).map((i) => +i.dataset.i);
    }
  });

  document.addEventListener("input", (e) => {
    const t = e.target;
    if (t.dataset && t.dataset.draft) {
      if (t.dataset.mock) mockState.answers[t.dataset.draft] = t.value;
      else store.set("draft:" + t.dataset.draft, t.value);
    } else if (t.id === "globalSearch") showResults(t.value);
  });

  let sTimer = null;
  $("#sheetSearch").addEventListener("input", (e) => {
    clearTimeout(sTimer);
    const v = e.target.value;
    sTimer = setTimeout(() => sheetSearch(v), 180);
  });
  $("#sheetSearch").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); focusHit(hitIdx + (e.shiftKey ? -1 : 1)); }
    if (e.key === "Escape") { e.target.value = ""; clearHits(); $("#sheetCount").textContent = ""; }
  });
  $("#sheetPrev").addEventListener("click", () => focusHit(hitIdx - 1));
  $("#sheetNext").addEventListener("click", () => focusHit(hitIdx + 1));
  $("#sheetToggle").addEventListener("click", () => setSheet(!sheetIsOpen()));
  $("#sheetClose").addEventListener("click", () => setSheet(false));
  $("#navToggle").addEventListener("click", () => {
    const open = document.body.classList.toggle("nav-open");
    $("#scrim").hidden = !open;
  });
  $("#scrim").addEventListener("click", () => {
    document.body.classList.remove("nav-open");
    if (!wide()) setSheet(false);
    $("#scrim").hidden = true;
  });

  const gs = $("#globalSearch");
  gs.addEventListener("keydown", (e) => {
    const box = $("#globalResults");
    const items = $$(".res[data-route]", box);
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (!items.length) return;
      gIdx = (gIdx + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      items.forEach((x, i) => x.classList.toggle("now", i === gIdx));
      items[gIdx].scrollIntoView({ block: "nearest" });
    } else if (e.key === "Enter") {
      const it = items[gIdx >= 0 ? gIdx : 0];
      if (it) { box.hidden = true; go(it.dataset.route, it.dataset.anchor); gs.blur(); }
    } else if (e.key === "Escape") { box.hidden = true; gs.blur(); }
  });
  gs.addEventListener("focus", () => { if (gs.value.trim()) showResults(gs.value); });

  document.addEventListener("keydown", (e) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement && document.activeElement.tagName);
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); gs.focus(); gs.select(); return; }
    if (e.key === "Escape") { const z = $(".zoom"); if (z) z.remove(); }
    if (typing) return;
    if (e.key === "/") { e.preventDefault(); setSheet(true); $("#sheetSearch").focus(); }
    else if (e.key.toLowerCase() === "c" && !e.ctrlKey && !e.metaKey && !e.altKey) setSheet(!sheetIsOpen());
  });

  window.addEventListener("hashchange", () => {
    const h = location.hash.slice(1);
    if (VALID.test(h) && h !== current.route) go(h, null, { fromHash: true });
  });

  // ---------- start ----------
  const startHash = (location.hash || "").slice(1);
  if (wide() && store.get("sheetOpen", true)) setSheet(true);
  go(VALID.test(startHash) ? startHash : "home", null, { fromHash: true, force: true });
})();
