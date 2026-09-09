// exercise the reader against saved data with DOM/Plotly doubles; no browser automation.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

class Element {
  constructor(tag = "div") {
    this.tagName = tag; this.children = []; this.value = ""; this.textContent = "";
    this.dataset = {}; this.hidden = false; this.attributes = {};
    this.classList = {add() {}, remove() {}, toggle() {}};
  }
  append(...children) {
    this.children.push(...children);
    if (this.tagName === "select" && this.children.length && this.value === "") {
      this.value = String(this.children[0].value);
    }
  }
  replaceChildren(...children) { this.children = []; this.value = ""; this.append(...children); }
  setAttribute(key, value) { this.attributes[key] = value; }
  removeAttribute(key) { delete this.attributes[key]; }
  scrollIntoView() { this.scrolled = true; }
  removeAllListeners() {}
  on(event, handler) { this[event] = handler; }
  querySelector(selector) { assert.equal(selector, "tbody"); return this.body ||= new Element("tbody"); }
}

async function main() {
  const reportPath = process.argv[2];
  assert.ok(reportPath, "pass a generated report.html");
  const root = path.resolve(__dirname, "../experiments/dabp_node_dynamics");
  const html = fs.readFileSync(reportPath, "utf8");
  const payload = html.match(/<script id="experiment-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
  const elements = {};
  for (const match of html.matchAll(/<([a-z]+)[^>]*\bid="([^"]+)"[^>]*>/g)) elements[match[2]] = new Element(match[1]);
  for (const match of html.matchAll(/<select id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) {
    const first = match[2].match(/<option value="([^"]+)"/);
    if (first) elements[match[1]].value = first[1];
  }
  elements["experiment-data"].textContent = payload;
  const tabs = ["graph", "evolution", "changes", "structure", "balance", "architecture"].map(name => {
    const button = new Element("button"); button.dataset.tab = name; return button;
  });
  const sorts = ["node", "initial", "final", "movement", "jump"].map(name => {
    const button = new Element("button"); button.dataset.sort = name; return button;
  });
  const splitSorts = ["node", "departure"].map(name => {
    const button = new Element("button"); button.dataset.splitSort = name; return button;
  });
  const sections = tabs.map(t => {
    const section = elements[`view-${t.dataset.tab}`]; section.id = `view-${t.dataset.tab}`; return section;
  });
  const document = {
    body: new Element("body"),
    getElementById: id => elements[id] || null,
    createElement: tag => new Element(tag),
    querySelectorAll: selector => ({"nav button": tabs, "[data-sort]": sorts, "[data-split-sort]": splitSorts, "main>section": sections})[selector],
  };
  const plots = {};
  let plotCount = 0;
  const Plotly = {
    react(id, traces, layout) {
      plotCount++;
      if (id !== "network") {
        assert.equal(layout.xaxis.type, "linear"); assert.equal(layout.yaxis.type, "linear");
      }
      traces.forEach(trace => {
        assert.equal(trace.x.length, trace.y.length, `${id}: axis lengths differ`);
        assert.ok(trace.y.every(v => v === null || Number.isFinite(v)), `${id}: non-finite data`);
      });
      plots[id] = {traces, layout}; return Promise.resolve();
    },
    purge(id) { delete plots[id]; },
    Plots: {resize() { return Promise.resolve(); }},
  };
  const context = vm.createContext({document, window:{scrollTo() {}}, Plotly, console, Blob, Response, DecompressionStream, atob, setTimeout});
  vm.runInContext(fs.readFileSync(path.join(root, "reader.js"), "utf8"), context);
  for (let i = 0; i < 1000 && !document.body.dataset.ready; i++) {
    await new Promise(resolve => setTimeout(resolve, 10));
    assert.notEqual(elements.loading.textContent, "Load failed", elements.error.textContent);
  }
  assert.equal(document.body.dataset.ready, "true");
  const data = JSON.parse(payload), count = data.graph.nodes.length;
  assert.equal(elements.node.children.length, count);
  function checkRanking(scores, descending = true) {
    const expected = scores.map((score,node) => node).sort((a,b) => (scores[a]-scores[b])*(descending?-1:1)||a-b);
    assert.deepEqual(elements.node.children.map(e => Number(e.value)), expected);
    expected.forEach((node,rank) => assert.equal(elements.node.children[rank].textContent, `${rank+1}. ${data.graph.nodes[node].id}`));
    return expected[0];
  }
  const bothRuns = (quantity,metric) => data.graph.nodes.map((_,node) => Math.max(...["symmetric","asymmetric"].map(v => data.runs[v].summary[quantity][node][metric])));
  assert.equal(Number(elements.node.value), checkRanking(data.graph.nodes.map(n => n.betweenness)));
  for (const quantity of ["damping","coefficient"]) {
    elements["evolution-order"].value = quantity;
    vm.runInContext('tab("evolution",true)', context);
    assert.equal(Number(elements.node.value), checkRanking(bothRuns(quantity,"jump")));
  }
  vm.runInContext('tab("changes",true)', context);
  checkRanking(data.runs.asymmetric.summary.damping.map(row => row.jump));
  vm.runInContext('tab("structure",true)', context);
  checkRanking(bothRuns("damping","jump"));
  vm.runInContext('tab("balance",true)', context);
  checkRanking(data.runs.asymmetric.split_summary.map(row => row?.departure ?? -Infinity));
  assert.equal(elements["split-table"].body.children[0].children[0].children[0].textContent, data.graph.nodes[Number(elements.node.value)].id);
  for (let node = 0; node < count; node++) {
    vm.runInContext(`chooseNode(${node}); tab("evolution"); tab("balance");`, context);
    for (const variant of ["symmetric", "asymmetric"]) {
      const damping = plots[`${variant}-damping`].traces;
      assert.equal(damping.length, 2);
      assert.equal(damping[0].x.length, data.runs[variant].meta.outcome.iterations);
      const starts = elements["starting-weights"].body.children;
      const offset = variant === "symmetric" ? 1 : 3;
      for (const [row,plot] of [[0,`${variant}-damping`],[2,`${variant}-attention`],[3,`${variant}-coefficient`]]) {
        if (!plots[plot]) continue;
        [0,1].forEach(half => assert.equal(Number(starts[row].children[offset+half].textContent), plots[plot].traces[half].y[0]));
      }
      const share = plots[`${variant}-balance-share`];
      if (share) {
        assert.ok(share.traces[0].y.every(x => x > 0 && x < 100));
        if (variant === "symmetric") assert.ok(share.traces[0].y.every(x => Math.abs(x - 50) < 1e-10));
      }
    }
  }
  for (const variant of ["symmetric", "asymmetric"]) {
    elements["table-run"].value = variant;
    for (const quantity of ["damping", "coefficient"]) {
      elements["table-quantity"].value = quantity;
      vm.runInContext('tab("changes")', context);
      assert.equal(elements["change-table"].body.children.length, count);
      sorts.forEach(button => button.onclick());
      elements["change-table"].body.children[0].onclick();
    }
  }
  for (const metric of ["degree", "betweenness", "closeness", "clustering"]) {
    elements["structure-x"].value = metric;
    for (const quantity of ["damping", "coefficient"]) {
      elements["structure-quantity"].value = quantity;
      for (const behavior of ["final", "movement", "jump"]) {
        elements["structure-y"].value = behavior;
        vm.runInContext('tab("structure")', context);
        assert.equal(plots["symmetric-structure"].traces[0].x.length, count);
        assert.deepEqual(plots["symmetric-structure"].layout.yaxis.range, plots["asymmetric-structure"].layout.yaxis.range);
      }
    }
  }
  for (const variant of ["symmetric", "asymmetric"]) {
    elements["split-run"].value = variant;
    vm.runInContext('tab("balance")', context);
    assert.equal(elements["split-table"].body.children.length, count);
    for (let node = 0; node < count; node++) {
      const record = data.runs[variant].split_summary[node];
      if (!record) continue;
      const pair = data.runs[variant].pairs[record.pair_index];
      vm.runInContext(`inspectNodeSplit(${node})`, context);
      assert.equal(String(elements.node.value), String(node));
      assert.equal(elements["balance-factor"].value, pair.factor);
      assert.equal(String(elements["balance-destination"].value), String(pair.target));
      assert.equal(elements["split-display"].value, "change");
      assert.equal(elements["balance-detail"].scrolled, true);
      const trace = plots[`${variant}-balance-share`].traces[0];
      assert.ok(Math.abs(trace.customdata[0][0] - record.initial) < 1e-10);
      assert.ok(Math.abs(trace.customdata.at(-1)[0] - record.final) < 1e-10);
      assert.ok(Math.abs(Math.max(...trace.y.map(Math.abs)) - record.departure) < 1e-10);
      assert.equal(trace.y[0], 0);
      assert.deepEqual(plots["symmetric-balance-share"].layout.yaxis.range, plots["asymmetric-balance-share"].layout.yaxis.range);
    }
    splitSorts[1].onclick();
    if (elements["split-table"].body.children.length > 1) {
      const values = elements["split-table"].body.children.map(row => Number(row.children[3].textContent));
      const ascending = values.every((value, i) => i === 0 || value >= values[i-1]);
      const descending = values.every((value, i) => i === 0 || value <= values[i-1]);
      assert.ok(ascending || descending);
    }
    elements["split-back"].onclick();
    assert.equal(elements["split-overview"].scrolled, true);
  }
  elements["split-display"].value = "share";
  elements["axis-range"].value = "full";
  vm.runInContext('tab("balance")', context);
  assert.deepEqual(Array.from(plots["symmetric-balance-share"].layout.yaxis.range), [0, 100]);
  for (const split of ["0.95","0.5"]) {
    elements["architecture-split"].value = split;
    for (const target of ["x1-A","x1-B","x2-A","x2-B"]) {
      elements["architecture-target"].value = target;
      vm.runInContext('tab("architecture")', context);
      const [variable,factor] = target.split("-"), source = factor === "A" ? "B" : "A";
      assert.ok(elements["architecture-canvas"].innerHTML.includes(`${variable} to factor ${factor} uses the incoming message from factor ${source}`));
      assert.ok(elements["architecture-formula"].textContent.includes(`+ β × incoming ${source}`));
      assert.equal(elements["report-toolbar"].hidden, true);
      assert.equal(elements["node-order"].hidden, true);
      assert.ok(!elements["architecture-cost"].textContent.includes("00000000000"));
    }
  }
  vm.runInContext('tab("graph")', context);
  assert.equal(elements["report-toolbar"].hidden, false);
  console.log(`Reader checks passed: ${count} variables, both variants, six views, ranking, starting weights, exact split drilldowns, architecture variants, ${plotCount} plot updates. DOM/Plotly doubles; visual browser QA not included.`);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
