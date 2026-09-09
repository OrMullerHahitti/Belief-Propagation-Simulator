"use strict";
const experimentData = document.getElementById("experiment-data");
const D = JSON.parse(experimentData.textContent);
experimentData.textContent = "";
const variants = ["symmetric", "asymmetric"];
const titles = {symmetric:"Symmetric · 0.5 / 0.5", asymmetric:"Asymmetric · 0.95 / 0.05"};
const colors = ["#2563eb", "#77869b"];
const nodes = D.graph.nodes;
const meta = D.runs.symmetric.meta;
const $ = id => document.getElementById(id);
const num = value => Number(value).toLocaleString("en-US", {maximumFractionDigits:6, minimumFractionDigits:0});
const state = {tab:"graph",node:0,factor:null,destination:null,ready:false,sort:"jump",descending:true,splitSort:"departure",splitDescending:true};
const decoded = {};
const plotConfig = {responsive:true,displaylogo:false,scrollZoom:false,toImageButtonOptions:{format:"svg"},modeBarButtonsToRemove:["lasso2d","select2d"]};
const labels = {degree:"Degree",betweenness:"Betweenness",closeness:"Closeness",clustering:"Clustering",initial:"Initial mean",final:"Final mean",movement:"Total movement",jump:"Largest jump"};

function ranking() {
  const acrossRuns=(quantity,measure)=>nodes.map((_,node)=>Math.max(...variants.map(v=>D.runs[v].summary[quantity][node][measure])));
  if(["graph","architecture"].includes(state.tab)) return {scores:nodes.map(n=>n.betweenness),label:"betweenness",descending:true};
  if(state.tab==="evolution") {
    const quantity=$("evolution-order").value;
    return {scores:acrossRuns(quantity,"jump"),label:`largest ${quantity==="damping"?"damping":"message-coefficient"} jump across either run`,descending:true};
  }
  if(state.tab==="changes") {
    const variant=$("table-run").value,quantity=$("table-quantity").value;
    return {scores:nodes.map((_,node)=>state.sort==="node"?node:D.runs[variant].summary[quantity][node][state.sort]),label:`${state.sort==="node"?"variable number":labels[state.sort].toLowerCase()} · ${quantity} · ${variant}`,descending:state.descending};
  }
  if(state.tab==="structure") {
    const quantity=$("structure-quantity").value,measure=$("structure-y").value;
    return {scores:acrossRuns(quantity,measure),label:`${labels[measure].toLowerCase()} · ${quantity} · highest value across either run`,descending:true};
  }
  const variant=$("split-run").value;
  return {scores:D.runs[variant].split_summary.map((record,node)=>record?(state.splitSort==="node"?node:record.departure):null),label:`${state.splitSort==="node"?"variable number":"largest calculated-split departure"} · ${variant}`,descending:state.splitDescending};
}
function rankedNodes(spec=ranking()) {
  return nodes.map((_,node)=>node).sort((a,b)=>{
    const left=spec.scores[a],right=spec.scores[b];
    if(left===null || right===null) return Number(left===null)-Number(right===null)||a-b;
    return (left-right)*(spec.descending?-1:1)||a-b;
  });
}
function orderVariables(selectFirst=false) {
  const spec=ranking(),order=rankedNodes(spec);
  if(selectFirst) {state.node=order[0];state.factor=null;state.destination=null;}
  fill($("node"),order.map((node,rank)=>[node,`${rank+1}. ${nodes[node].id}`]),state.node);
  $("node-order").textContent=`Variables ordered by ${spec.label} (${spec.descending?"highest first":"lowest first"}). Equal values use variable number.`;
  return order;
}
function option(select, value, label) {
  const element = document.createElement("option"); element.value = value; element.textContent = label; select.append(element);
}
function fill(select, entries, chosen) {
  select.replaceChildren(); entries.forEach(([value,label])=>option(select,value,label));
  if(entries.some(([value])=>String(value)===String(chosen))) select.value = chosen;
  return select.value;
}
function factorLabel(name) {
  const f = D.graph.factors.find(f=>f.id===name);
  return f ? `${name} · ${f.variables.join(" ↔ ")}` : name;
}
function targetLabel(index) {
  const fn = meta.trg_fn_idxes[index];
  return `${meta.trg_var_names[index]} → ${meta.fn_factor_names[fn]} · half ${meta.fn_half[fn]}`;
}
function selections() {
  const factors = D.graph.factors.filter(f=>f.variables.length===2 && f.variables.includes(nodes[state.node].id));
  state.factor = fill($("factor"),factors.map(f=>[f.id,factorLabel(f.id)]),state.factor);
  fill($("balance-factor"),factors.map(f=>[f.id,factorLabel(f.id)]),state.factor);
  const destinations = D.runs.symmetric.pairs.filter(pair=>pair.factor===state.factor && D.runs.symmetric.target_owners[pair.target]===state.node);
  state.destination = fill($("destination"),destinations.map(p=>[p.target,targetLabel(p.target)]),state.destination);
  fill($("balance-destination"),destinations.map(p=>[p.target,targetLabel(p.target)]),state.destination);
  $("destination").disabled = !destinations.length;
  $("balance-destination").disabled = !destinations.length;
  $("evolution-context").textContent = `${nodes[state.node].id} · ${factorLabel(state.factor)}. Every recorded iteration; no temporal averaging.`;
}
function layout(title,xTitle="Iteration",yTitle="",ranges={},subtitle="") {
  const titleText=subtitle?`${title}<br><span style="font-size:10px;color:#8797aa">${subtitle}</span>`:title;
  return {title:{text:titleText,font:{size:13,color:"#172334"},x:.045,xanchor:"left",y:.95},
    font:{family:'-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',size:10,color:"#75849a"},
    margin:{l:76,r:24,t:subtitle?75:60,b:72},paper_bgcolor:"#fff",plot_bgcolor:"#fff",
    xaxis:{title:{text:xTitle,font:{size:11}},type:"linear",gridcolor:"#edf0f3",zeroline:false,...(ranges.x?{range:ranges.x}:{})},
    yaxis:{title:{text:yTitle,font:{size:11}},type:"linear",gridcolor:"#edf0f3",zeroline:false,tickformat:".6~f",...(ranges.y?{range:ranges.y}:{})},
    legend:{orientation:"h",x:0,y:-0.3,font:{size:10}},hovermode:"closest",showlegend:true};
}
async function chart(id,traces,figure) {
  const element=$(id);element.classList.remove("empty");
  await Plotly.react(id,traces,figure,plotConfig);
  if(element.offsetParent!==null) {
    try { await Plotly.Plots.resize(element); }
    catch(error) { if(element.offsetParent!==null) throw error; }
  }
}
function empty(id,message="Both halves do not feed a common destination at this variable.") {
  Plotly.purge(id); $(id).replaceChildren(); $(id).textContent=message; $(id).classList.add("empty");
}
function trajectory(variant,quantity,row) {
  const run=decoded[variant],n=run.iterations;
  if(quantity!=="coefficient") return Array.from(run[quantity].subarray(row*n,(row+1)*n));
  const r=D.runs[variant],target=r.meta.src_trg_idxes[row],count=r.source_count[target];
  const result=new Array(n);
  for(let t=0;t<n;t++) result[t]=run.attention[row*n+t]*count*run.new_weight[target*n+t];
  return result;
}
function extent(series,quantity) {
  let low=Infinity,high=-Infinity;
  series.forEach(s=>s.forEach(v=>{low=Math.min(low,v);high=Math.max(high,v);}));
  if(!Number.isFinite(low)) return [0,1];
  if($("axis-range").value==="full") return [0,quantity==="coefficient" ? Math.max(high*1.06,1) : quantity==="balance"?100:1];
  const pad=Math.max((high-low)*.08,quantity==="balance"?.02:.0005);
  return [low-pad,high+pad];
}
function lineTrace(values,name,index) {
  return {x:values.map((_,i)=>i+1),y:values,type:"scatter",mode:"lines",name,
    line:{color:colors[index],width:1.8,dash:index?"dash":"solid"},
    hovertemplate:"Iteration %{x}<br>%{y:.8f}<extra>"+name+"</extra>"};
}
function incidenceRows() {
  return [0,1].map(half=>meta.trg_fn_idxes.findIndex((fn,k)=>D.runs.symmetric.target_owners[k]===state.node && meta.fn_factor_names[fn]===state.factor && meta.fn_half[fn]===half));
}
function selectedPair() { return D.runs.symmetric.pairs.find(p=>p.factor===state.factor && p.target===Number(state.destination) && D.runs.symmetric.target_owners[p.target]===state.node); }
function pairedCharts(suffix,quantity,rows,title,scale=1) {
  if(!rows){variants.forEach(v=>empty(`${v}-${suffix}`));return;}
  const series=Object.fromEntries(variants.map(v=>[v,rows.map(row=>trajectory(v,quantity,row).map(x=>x*scale))]));
  const y=extent(variants.flatMap(v=>series[v]),quantity);
  const x=[1,Math.max(...variants.map(v=>decoded[v].iterations))];
  variants.forEach(v=>{
    const figure=layout(title,"Iteration",title,{x,y},titles[v]);
    const start=$(`${v}-${suffix}-start`);
    if(start) start.textContent=`${v==="symmetric"?"Symmetric":"Asymmetric"} · iteration 1: half 0 = ${series[v][0][0]}, half 1 = ${series[v][1][0]}`;
    chart(`${v}-${suffix}`,series[v].map((s,i)=>lineTrace(s,`Half ${i}`,i)),figure);
  });
}
function evolution() {
  if(!state.ready) return;
  orderVariables();selections();
  pairedCharts("damping","damping",incidenceRows(),"Old-message damping");
  const pair=selectedPair();
  const starts=$("starting-weights").querySelector("tbody");starts.replaceChildren();
  [["Old-message damping (λ)","damping",incidenceRows()],["New-message weight (β)","new_weight",incidenceRows()],["Attention share (α)","attention",pair?.rows],["Applied edge coefficient (c)","coefficient",pair?.rows]].forEach(([label,quantity,rows])=>{
    const tr=document.createElement("tr"),name=document.createElement("td");name.textContent=label;tr.append(name);
    variants.forEach(v=>[0,1].forEach(half=>{
      const td=document.createElement("td");td.textContent=rows?String(trajectory(v,quantity,rows[half])[0]):"No common destination";tr.append(td);
    }));starts.append(tr);
  });
  pairedCharts("attention","attention",pair?.rows,"Attention share");
  pairedCharts("coefficient","coefficient",pair?.rows,"Applied message coefficient");
}
function splitText(value) { return `${value.toFixed(6)} / ${(100-value).toFixed(6)}`; }
function inspectNodeSplit(node) {
  const run=D.runs[$("split-run").value],record=run.split_summary[node];
  if(!record) return;
  const pair=run.pairs[record.pair_index];
  state.node=node;state.factor=pair.factor;state.destination=pair.target;
  $("node").value=node;$("split-display").value="change";
  balance();$("balance-detail").scrollIntoView({block:"start"});
}
function splitOverview() {
  orderVariables();
  const variant=$("split-run").value,run=D.runs[variant];
  const rows=run.split_summary.map((record,node)=>({record,node}));
  rows.sort((a,b)=>{
    if(!a.record || !b.record) return Number(!a.record)-Number(!b.record);
    const difference=state.splitSort==="node"?a.node-b.node:a.record.departure-b.record.departure;
    return difference*(state.splitDescending?-1:1)||a.node-b.node;
  });
  document.querySelectorAll("[data-split-sort]").forEach(button=>{
    const key=button.dataset.splitSort;
    button.textContent=(key==="node"?"Variable":"Largest departure (pp)")+(key===state.splitSort?(state.splitDescending?" ↓":" ↑"):"");
  });
  const body=$("split-table").querySelector("tbody");body.replaceChildren();
  rows.forEach(({record,node})=>{
    const tr=document.createElement("tr"),name=document.createElement("td");
    tr.classList.toggle("selected",node===state.node);
    if(record) {
      const button=document.createElement("button");button.textContent=nodes[node].id;name.append(button);
      const pair=run.pairs[record.pair_index];
      tr.title=`${factorLabel(pair.factor)} · ${targetLabel(pair.target)} · largest departure at iteration ${record.peak_iteration}; ${record.pair_count} pairs checked`;
      tr.onclick=()=>inspectNodeSplit(node);
    } else name.textContent=nodes[node].id;
    tr.append(name);
    if(record) {
      [splitText(record.initial),splitText(record.final),record.departure.toFixed(6)].forEach(value=>{
        const cell=document.createElement("td");cell.textContent=value;tr.append(cell);
      });
    } else {
      const cell=document.createElement("td");cell.colSpan=3;cell.className="unavailable";
      cell.textContent="No destination receives both halves.";tr.append(cell);
    }
    body.append(tr);
  });
  const eligible=run.split_summary.filter(Boolean),peak=Math.max(0,...eligible.map(row=>row.departure));
  $("split-overview-context").textContent=`${titles[variant]} · iterations 1–${run.meta.outcome.iterations} · ${eligible.length} of ${nodes.length} variables have eligible pairs · ${run.pairs.length.toLocaleString("en-US")} pairs checked. Largest departure anywhere: ${peak.toFixed(6)} pp.`;
}
function balance() {
  if(!state.ready) return;
  selections();splitOverview();const pair=selectedPair(),change=$("split-display").value==="change";
  $("range-control").hidden=change;
  $("balance-context").textContent=`${nodes[state.node].id} · ${factorLabel(state.factor)} · ${state.destination!==""?targetLabel(Number(state.destination)):"no eligible outgoing destination"}`;
  $("split-chart-note").textContent=change?"Change in the first-half share since iteration 1, in percentage points. Focused linear scale, shared across both runs; the second half changes by the opposite amount.":"Calculated first-half percentage at each iteration. The second half is 100% minus this value. Hover to see both percentages.";
  pairedCharts("balance-coefficient","coefficient",pair?.rows,"Applied message coefficient");
  if(!pair){variants.forEach(v=>{empty(`${v}-balance-share`);$(`${v}-split-readout`).textContent="No eligible pair.";});return;}
  const shares={};
  variants.forEach(v=>{
    const [a,b]=pair.rows.map(r=>trajectory(v,"coefficient",r)),p=D.runs[v].meta.split_ratio;
    shares[v]=a.map((value,i)=>100*(p*value/(p*value+(1-p)*b[i])));
    $(`${v}-split-readout`).textContent=`${v==="symmetric"?"Symmetric":"Asymmetric"} · initial ${splitText(shares[v][0])}% → final ${splitText(shares[v].at(-1))}%`;
  });
  const plotted=Object.fromEntries(variants.map(v=>[v,change?shares[v].map(value=>value-shares[v][0]):shares[v]]));
  const peak=Math.max(...variants.flatMap(v=>plotted[v].map(Math.abs)));
  const span=Math.max(peak*1.08,0.000001);
  const y=change?[-span,span]:extent(variants.map(v=>shares[v]),"balance");
  const x=[1,Math.max(...variants.map(v=>decoded[v].iterations))];
  variants.forEach(v=>{
    const figure=layout(change?"Change in first-half share":"First-half share","Iteration",change?"Change from start (pp)":"Split-adjusted share (%)",{x,y},titles[v]);
    figure.shapes=[{type:"line",xref:"paper",x0:0,x1:1,y0:change?0:50,y1:change?0:50,line:{color:"#7b838b",dash:"dot",width:1}}];
    figure.showlegend=false;
    const trace=lineTrace(plotted[v],change?"Change from start":"Calculated split",0);
    trace.customdata=shares[v].map(value=>[value,100-value]);
    trace.hovertemplate="Iteration %{x}<br>Split: %{customdata[0]:.8f}% / %{customdata[1]:.8f}%"+(change?"<br>Change: %{y:.8f} pp":"")+"<extra></extra>";
    chart(`${v}-balance-share`,[trace],figure);
  });
}
function graph() {
  orderVariables();
  const selected=nodes[state.node],adjacent=new Set(selected.neighbors);
  $("node-title").textContent=selected.id;
  $("properties").replaceChildren();
  ["degree","betweenness","closeness","clustering"].forEach(key=>{
    const row=document.createElement("div"),dt=document.createElement("dt"),dd=document.createElement("dd");
    dt.textContent=labels[key];dd.textContent=num(selected[key]);row.append(dt,dd);$("properties").append(row);
  });
  const graphOrder=rankedNodes({scores:nodes.map(n=>n.betweenness),descending:true});
  $("neighbors").replaceChildren();graphOrder.map(i=>nodes[i].id).filter(name=>adjacent.has(name)).forEach(name=>{
    const button=document.createElement("button");button.textContent=name;button.onclick=()=>chooseNode(nodes.findIndex(n=>n.id===name));$("neighbors").append(button);
  });
  const byName=Object.fromEntries(nodes.map(n=>[n.id,n]));
  const regular={x:[],y:[]},highlight={x:[],y:[]};
  D.graph.factors.filter(f=>f.variables.length===2).forEach(f=>{
    const [a,b]=f.variables.map(n=>byName[n]); const group=f.variables.includes(selected.id)?highlight:regular;
    group.x.push(a.position[0],b.position[0],null);group.y.push(a.position[1],b.position[1],null);
  });
  const traces=[regular,highlight].map((g,i)=>({...g,type:"scatter",mode:"lines",line:{color:i?"#98b8f3":"#e3e9f1",width:i?1.8:1},hoverinfo:"skip",showlegend:false}));
  traces.push({x:nodes.map(n=>n.position[0]),y:nodes.map(n=>n.position[1]),type:"scatter",mode:"markers+text",text:nodes.map(n=>n.id),textposition:"middle center",textfont:{size:nodes.length>30?10:11,color:nodes.map((n,i)=>i===state.node?"#fff":"#52657e")},customdata:nodes.map((n,i)=>i),
    marker:{size:nodes.map((n,i)=>i===state.node?38:nodes.length>30?25:29),color:nodes.map((n,i)=>i===state.node?"#2563eb":adjacent.has(n.id)?"#d9e7ff":"#edf2f8"),line:{color:"white",width:2}},
    hovertemplate:"%{text}<extra></extra>",showlegend:false});
  const bounds=axis=>{const values=nodes.map(n=>n.position[axis]),lo=Math.min(...values),hi=Math.max(...values),pad=(hi-lo)*.12;return [lo-pad,hi+pad];};
  const figure=layout("");figure.margin={l:24,r:24,t:24,b:15};figure.xaxis={visible:false,range:bounds(0),fixedrange:false};figure.yaxis={visible:false,range:bounds(1),scaleanchor:"x",fixedrange:false};
  chart("network",traces,figure).then(()=>{
    $("network").removeAllListeners("plotly_click");
    $("network").on("plotly_click",event=>{const i=event.points[0].customdata;if(Number.isInteger(i))chooseNode(i);});
  });
  ["damping","coefficient"].forEach(quantity=>{
    const traces=variants.map((v,i)=>{
      const values=D.runs[v].overview[`max_${quantity}_change`];
      const trace=lineTrace(values,titles[v],i);trace.x=values.map((_,i)=>i+2);return trace;
    });
    chart(`global-${quantity}`,traces,layout(quantity==="damping"?"Largest damping change":"Largest message-coefficient change","Iteration","Absolute change"));
  });
}
function changes() {
  orderVariables();
  const variant=$("table-run").value,quantity=$("table-quantity").value,run=D.runs[variant];
  const rows=run.summary[quantity].map((row,node)=>({...row,node}));
  rows.sort((a,b)=>(a[state.sort]-b[state.sort])*(state.descending?-1:1));
  document.querySelectorAll("[data-sort]").forEach(button=>{
    const key=button.dataset.sort;
    button.textContent=(key==="node"?"Variable":labels[key])+(key===state.sort?(state.descending?" ↓":" ↑"):"");
  });
  const body=$("change-table").querySelector("tbody");body.replaceChildren();
  rows.forEach(row=>{
    const tr=document.createElement("tr");tr.classList.toggle("selected",row.node===state.node);
    ["node","initial","final","movement","jump"].forEach(key=>{
      const td=document.createElement("td");
      if(key==="node"){const button=document.createElement("button");button.textContent=nodes[row.node].id;td.append(button);}else {td.textContent=["initial","final"].includes(key)?String(row[key]):num(row[key]);td.title=Number(row[key]).toPrecision(16);}
      tr.append(td);
    });
    tr.onclick=()=>{$("evolution-order").value=quantity;chooseNode(row.node);tab("evolution");};body.append(tr);
  });
  $("table-window").textContent=`${titles[variant]} · iterations 1–${run.meta.outcome.iterations} · ${quantity==="damping"?"each outgoing split-half message":"each incoming-source / outgoing-destination coefficient"}. Click a column heading to sort; click a row to inspect.`;
}
function structure() {
  orderVariables();
  const xKey=$("structure-x").value,yKey=$("structure-y").value,quantity=$("structure-quantity").value;
  const allY=variants.map(v=>D.runs[v].summary[quantity].map(r=>r[yKey]));
  const ys=extent(allY,yKey==="final"?quantity:"coefficient");
  const xs=nodes.map(n=>n[xKey]),lo=Math.min(...xs),hi=Math.max(...xs),pad=Math.max((hi-lo)*.1,.01);
  variants.forEach(v=>{
    const values=D.runs[v].summary[quantity];
    const trace={x:xs,y:values.map(r=>r[yKey]),type:"scatter",mode:"markers",customdata:nodes.map((n,i)=>i),text:nodes.map(n=>n.id),
      marker:{size:nodes.map((n,i)=>i===state.node?13:9),color:colors[0],opacity:.85,line:{color:"white",width:1}},showlegend:false,
      hovertemplate:"%{text}<br>"+labels[xKey]+": %{x:.6f}<br>"+labels[yKey]+": %{y:.8f}<extra></extra>"};
    const figure=layout(`${labels[xKey]} & ${labels[yKey].toLowerCase()}`,labels[xKey],`${labels[yKey]} · ${quantity}`,{x:[lo-pad,hi+pad],y:ys},`${titles[v]} · ${D.runs[v].meta.outcome.iterations} iterations`);
    chart(`${v}-structure`,[trace],figure).then(()=>{
      const el=$(`${v}-structure`);el.removeAllListeners("plotly_click");
      el.on("plotly_click",event=>{$("evolution-order").value=quantity;chooseNode(event.points[0].customdata);tab("evolution");});
    });
  });
}
function architecture() {
  const p=Number($("architecture-split").value),[variable,target]=$("architecture-target").value.split("-");
  const source=target==="A"?"B":"A",symbol=variable==="x1"?"x₁":"x₂",left=variable==="x1";
  const x=left?148:732,tip=left?345:535,sourceY=source==="A"?104:326,targetY=target==="A"?104:326;
  const nearSource=source==="A"?188:242,nearTarget=target==="A"?188:242;
  $("architecture-canvas").innerHTML=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 430" role="img" aria-label="Two variables x1 and x2 linked through split factors A and B. Highlighted update ${variable} to factor ${target} uses the incoming message from factor ${source}.">
    <defs><marker id="flow-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#2563eb"/></marker><marker id="excluded-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#94a3b8"/></marker></defs>
    <g fill="none" stroke="#dce4ed" stroke-width="2"><path d="M150 190 L350 110 M150 240 L350 320 M730 190 L530 110 M730 240 L530 320"/></g>
    <path d="M${tip} ${sourceY} L${x} ${nearSource}" fill="none" stroke="#2563eb" stroke-width="4" marker-end="url(#flow-arrow)"/>
    <path d="M${x} ${nearTarget} L${tip} ${targetY}" fill="none" stroke="#2563eb" stroke-width="4" marker-end="url(#flow-arrow)"/>
    <path d="M${tip} ${targetY+(target==="A"?20:-20)} L${x} ${nearTarget+(target==="A"?20:-20)}" fill="none" stroke="#94a3b8" stroke-width="2" stroke-dasharray="6 5" marker-end="url(#excluded-arrow)"/>
    <g font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" text-anchor="middle">
      <circle cx="110" cy="215" r="43" fill="${left?"#2563eb":"#edf2f8"}" stroke="${left?"#2563eb":"#cbd5e1"}"/><text x="110" y="224" font-size="28" fill="${left?"#fff":"#172334"}">x₁</text>
      <circle cx="770" cy="215" r="43" fill="${left?"#edf2f8":"#2563eb"}" stroke="${left?"#cbd5e1":"#2563eb"}"/><text x="770" y="224" font-size="28" fill="${left?"#172334":"#fff"}">x₂</text>
      <rect x="350" y="64" width="180" height="80" rx="8" fill="#f6f9fe" stroke="#b8c9df"/><text x="440" y="95" font-size="19" fill="#172334">Factor A · half 0</text><text x="440" y="124" font-size="17" fill="#526276">${p} × C / 50</text>
      <rect x="350" y="286" width="180" height="80" rx="8" fill="#f6f9fe" stroke="#b8c9df"/><text x="440" y="317" font-size="19" fill="#172334">Factor B · half 1</text><text x="440" y="346" font-size="17" fill="#526276">${(1-p).toFixed(2)} × C / 50</text>
      <text x="440" y="205" font-size="16" fill="#2563eb">${source} → ${symbol} → ${target}</text><text x="440" y="230" font-size="13" fill="#65758b">one incoming message feeds this update</text>
      <text x="440" y="403" font-size="13" fill="#65758b">Original cost C = the cost table for the connection between x₁ and x₂</text>
    </g></svg>`;
  $("architecture-cost").textContent=`After the fixed internal scaling by 1/50: A receives ${num(100*p)}% of C and B receives ${num(100*(1-p))}% of C. The split stays fixed during the run.`;
  $("architecture-update").textContent=`For ${symbol} → ${target}, use the incoming ${source} → ${symbol} message. Exclude ${target} → ${symbol}. Keep the previous ${symbol} → ${target} message through damping.`;
  $("architecture-formula").textContent=`new ${symbol} → ${target} = λ × previous ${symbol} → ${target} + β × incoming ${source} → ${symbol}`;
  $("architecture-explanation").textContent="λ controls memory of the previous outgoing message. β controls the new incoming message. Here attention α = 1, so the applied edge coefficient is β.";
}
function render() { ({graph,evolution,changes,structure,balance,architecture})[state.tab](); }
function chooseNode(index) {state.node=Number(index);$("node").value=state.node;selections();render();}
function tab(name,selectFirst=false) {
  if(!state.ready && ["evolution","balance"].includes(name)) return;
  if(state.tab!==name) window.scrollTo({top:0,behavior:"instant"});
  state.tab=name;
  document.querySelectorAll("nav button").forEach(b=>{if(b.dataset.tab===name)b.setAttribute("aria-current","page");else b.removeAttribute("aria-current");});
  document.querySelectorAll("main>section").forEach(s=>s.hidden=s.id!==`view-${name}`);
  $("range-control").hidden=["changes","graph"].includes(name);
  $("report-toolbar").hidden=name==="architecture";$("node-order").hidden=name==="architecture";
  orderVariables(selectFirst);
  render();
}
async function decode(packed) {
  const binary=atob(packed.data),bytes=new Uint8Array(binary.length);
  for(let i=0;i<binary.length;i++) bytes[i]=binary.charCodeAt(i);
  const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
  const buffer=await new Response(stream).arrayBuffer();
  if(buffer.byteLength!==packed.shape[0]*packed.shape[1]*8) throw new Error("Trajectory size mismatch");
  return new Float64Array(buffer);
}
async function init() {
  const s=D.manifest.settings,edges=D.graph.factors.filter(f=>f.variables.length===2).length;
  $("scope").textContent=`${nodes.length} variables · ${edges} connections · requested density ${s.density} / realized ${num(D.graph.realized_density)} · graph seed ${s.graph_seed} · model seed ${s.model_seed}`;
  variants.forEach(v=>{
    const o=D.runs[v].meta.outcome,div=document.createElement("div");div.className="outcome";
    const strong=document.createElement("strong"),detail=document.createElement("span");
    strong.textContent=`${titles[v]} · ${o.iterations} iterations`;
    detail.textContent=o.stop_reason==="iteration_limit"?`Iteration limit · assignments ${o.assignment_stable?"stable":"not stable"} · weights ${o.weights_stable?"stable":"not stable"}`:"Assignments and weights stable";
    div.append(strong,detail);$("outcomes").append(div);
  });
  document.body.classList.toggle("large-graph",nodes.length>30);
  $("node").onchange=()=>chooseNode($("node").value);
  document.querySelectorAll("nav button").forEach(b=>b.onclick=()=>tab(b.dataset.tab,true));
  document.querySelectorAll("nav button").forEach(b=>{b.disabled=["evolution","balance"].includes(b.dataset.tab);});
  $("inspect").disabled=true;
  ["factor","balance-factor"].forEach(id=>$(id).onchange=()=>{state.factor=$(id).value;state.destination=null;selections();render();});
  ["destination","balance-destination"].forEach(id=>$(id).onchange=()=>{state.destination=$(id).value;selections();render();});
  ["table-run","table-quantity"].forEach(id=>$(id).onchange=changes);
  $("evolution-order").onchange=()=>{orderVariables(true);evolution();};
  ["architecture-split","architecture-target"].forEach(id=>$(id).onchange=architecture);
  $("split-run").onchange=splitOverview;
  $("split-display").onchange=balance;
  $("split-back").onclick=()=>$("split-overview").scrollIntoView({block:"start"});
  document.querySelectorAll("[data-split-sort]").forEach(button=>button.onclick=()=>{
    const key=button.dataset.splitSort;
    state.splitDescending=state.splitSort===key?!state.splitDescending:key!=="node";
    state.splitSort=key;splitOverview();
  });
  ["structure-x","structure-y","structure-quantity"].forEach(id=>$(id).onchange=structure);
  $("axis-range").onchange=render;$("inspect").onclick=()=>tab("evolution");
  document.querySelectorAll("[data-sort]").forEach(button=>button.onclick=()=>{state.descending=state.sort===button.dataset.sort?!state.descending:true;state.sort=button.dataset.sort;changes();});
  $("methods").innerHTML=`<p><strong>Stopping:</strong> ${s.stable_window} consecutive recorded iterations with an unchanged assignment vector and a full-window range strictly below ${s.tolerance} in every applied damping and incoming message coefficient. Maximum ${s.max_iterations} iterations. The window spans optimizer updates every ${s.update_interval} iterations. This is observed operational stability, not a mathematical convergence proof. Message state does not restart inside the run.</p><p><strong>Structural graph:</strong> undirected, unweighted variable graph before splitting. Degree is the neighbor count. Betweenness is normalized shortest-path betweenness; closeness is inverse mean shortest-path distance; clustering is the fraction of connected neighbor pairs. Exact, not sampled, calculations.</p><p><strong>Applied coefficients:</strong> damping = mean old-message weight across heads. Source coefficient = number of eligible incoming messages × mean attention share × mean new-message weight. DABP averages attention before damping. The full message update also includes the previous outgoing message and minimum subtraction.</p><p><strong>Comparison:</strong> same saved tables, graph, initialization seed, and initial network parameters; CPU float64. Curves use every recorded iteration, and each run stops independently. Initial means refer to iteration 1, not to uninitialized weights. Node changes use each run’s full duration; total movement therefore depends on duration. No causal claims are made from one graph.</p><p><strong>Split diagnostic:</strong> comparisons require both source halves to feed the same target. The attention score is bounded by DABP’s sigmoid, so the attention ratio between two sources cannot exceed about 2.718. Starting from 95/5, complete coefficient compensation is therefore impossible in this implementation. This bound does not describe the relative sizes of the full nonlinear messages. Symmetric halves may remain equal by construction.</p><p><strong>Data:</strong> <a href="graph.json">graph.json</a> · <a href="run.json">run.json</a> · <a href="symmetric.npz">symmetric.npz</a> · <a href="asymmetric.npz">asymmetric.npz</a>. The NPZ files retain all heads, assignments, costs, and variable/factor/destination provenance. The report contains losslessly compressed float64 displayed trajectories.</p><p><strong>Graph SHA-256:</strong> <code>${D.manifest.graph_sha256}</code></p>`;
  tab("graph",true);selections();
  for(const v of variants) {
    decoded[v]={iterations:D.runs[v].meta.outcome.iterations};
    for(const name of ["damping","new_weight","attention"]) {
      $("loading").textContent=`Preparing ${v} ${name.replace("_"," ")}…`;
      decoded[v][name]=await decode(D.runs[v].arrays[name]);
      delete D.runs[v].arrays[name].data;
    }
  }
  state.ready=true;document.body.dataset.ready="true";
  document.querySelectorAll("nav button").forEach(b=>{b.disabled=false;});
  $("inspect").disabled=false;
  $("loading").textContent="All iterations ready · offline";
  render();
}
init().catch(error=>{$("error").hidden=false;$("error").textContent=`Could not load the report: ${error.message}. Open this file in a current Chrome, Edge, Firefox or Safari browser.`;$("loading").textContent="Load failed";console.error(error);});
