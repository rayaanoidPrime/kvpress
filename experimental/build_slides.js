const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const BASE = "experimental_outputs/slides";

function imgB64(filename) {
  const data = fs.readFileSync(path.join(BASE, filename));
  const ext = path.extname(filename).slice(1).toLowerCase();
  return `image/${ext==="png"?"png":"jpeg"};base64,${data.toString("base64")}`;
}

const NV="1B2A4A",BL="2563EB",TL="0D9488",AM="D97706",PU="7C3AED",RD="DC2626",GY="6B7280",LG="F1F5F9",WH="FFFFFF",DT="1E293B",MT="475569";

let pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title = "KV Cache Compression - Slides 8-17";

function chip(s,x,y,w,label,bg=NV){
  s.addShape(s.slideLayout.shapes.RECTANGLE?pres.shapes.RECTANGLE:pres.shapes.RECTANGLE,{x,y,w,h:0.26,fill:{color:bg},line:{color:bg}});
  s.addText(label,{x,y,w,h:0.26,fontSize:7,bold:true,color:WH,fontFace:"Calibri",align:"center",valign:"middle",charSpacing:1.5,margin:0});
}

function topStrip(s,title,section="KV CACHE STRUCTURE"){
  s.addShape(pres.shapes.RECTANGLE,{x:0,y:0,w:10,h:1.0,fill:{color:NV},line:{color:NV}});
  s.addText(section,{x:0.4,y:0.1,w:2.2,h:0.22,fontSize:7,bold:true,color:"93C5FD",fontFace:"Calibri",align:"left",valign:"middle",charSpacing:1.5,margin:0});
  s.addText(title,{x:0.4,y:0.35,w:9.2,h:0.52,fontSize:19,bold:true,color:WH,fontFace:"Calibri",align:"left",margin:0});
}

function insightBar(s,text,y=5.1){
  s.addShape(pres.shapes.RECTANGLE,{x:0.35,y,w:9.3,h:0.38,fill:{color:LG},line:{color:"E2E8F0"}});
  s.addText(text,{x:0.55,y,w:9.0,h:0.38,fontFace:"Calibri",valign:"middle",margin:0,fontSize:10});
}

// ============ SLIDE 8 - EVICTION ============
{
  let s=pres.addSlide();s.background={color:WH};
  topStrip(s,"Eviction Patterns - Which Tokens Survive Compression?");
  [{lbl:"TinyLlama",col:BL,x:0.4},{lbl:"Qwen2.5-0.5B",col:TL,x:5.3}].forEach(m=>{
    s.addShape(pres.shapes.RECTANGLE,{x:m.x,y:1.08,w:2.0,h:0.24,fill:{color:m.col},line:{color:m.col}});
    s.addText(m.lbl,{x:m.x,y:1.08,w:2.0,h:0.24,fontSize:10,bold:true,color:WH,fontFace:"Calibri",align:"center",valign:"middle",margin:0});
  });
  s.addText([{text:"High ",options:{color:BL,bold:true}},{text:"| ",options:{color:GY}},{text:"Medium ",options:{color:AM,bold:true}},{text:"| ",options:{color:GY}},{text:"Low retention",options:{color:RD,bold:true}}],{x:7.5,y:1.08,w:2.2,h:0.24,fontFace:"Calibri",valign:"middle",margin:0,fontSize:10});
  s.addImage({data:imgB64("slide08/eviction_pattern_tinyllama.png"),x:0.5,y:1.4,w:9.0,h:3.5,sizing:{type:"contain",w:9.0,h:3.5}});
  insightBar(s,[{text:"Blue=kept, ",options:{color:BL,bold:true}},{text:"Red=evicted. KNorm retains high-magnitude tokens. SnapKV retains contextually attended tokens. Structural tokens survive.",options:{color:DT}}]);
}

// ============ SLIDE 9A - AUTOCORRELATION ============
{
  let s=pres.addSlide();s.background={color:WH};
  s.addShape(pres.shapes.RECTANGLE,{x:0,y:0,w:10,h:1.0,fill:{color:NV}});
  s.addText("KV CACHE STRUCTURE",{x:0.4,y:0.1,w:2.2,h:0.22,fontSize:7,bold:true,color:"93C5FD",fontFace:"Calibri",align:"left",valign:"middle",charSpacing:1.5,margin:0});
  s.addText("Temporal Autocorrelation - How Far Does Token Similarity Extend?",{x:0.4,y:0.35,w:9.2,h:0.52,fontSize:19,bold:true,color:WH,fontFace:"Calibri",align:"left",margin:0});
  s.addImage({data:imgB64("slide09a/autocorrelation.png"),x:0.5,y:1.15,w:9.0,h:3.8,sizing:{type:"contain",w:9.0,h:3.8}});
  [{col:BL,t:"Prose: ",b:"Cosine similarity >0.6 even at lag 20 - adjacent tokens remain meaningfully similar."},{col:TL,t:"Code: ",b:"Similarity drops faster with oscillations - syntactic boundaries create sharp transitions."}].forEach((o,i)=>{
    s.addShape(pres.shapes.RECTANGLE,{x:0.4+i*4.7,y:5.1,w:4.45,h:0.42,fill:{color:LG},line:{color:"E2E8F0"}});
    s.addShape(pres.shapes.RECTANGLE,{x:0.4+i*4.7,y:5.1,w:0.06,h:0.42,fill:{color:o.col},line:{color:o.col}});
    s.addText([{text:o.t+" ",options:{bold:true,color:o.col,fontSize:11}},{text:o.b,options:{color:DT,fontSize:10}}],{x:0.52+i*4.7,y:5.1,w:4.2,h:0.42,fontFace:"Calibri",valign:"middle",margin:0});
  });
}

// ============ SLIDE 9B - SCREE + RANK ============
{
  let s=pres.addSlide();s.background={color:WH};
  topStrip(s,"Intrinsic Rank Per Layer - How Compressible is the Head Dimension?");
  s.addText("Cumulative Variance by Component (Scree)",{x:0.5,y:1.08,w:9.0,h:0.26,fontSize:11,bold:true,color:NV,fontFace:"Calibri",align:"center",margin:0});
  s.addImage({data:imgB64("slide09b/scree_plot.png"),x:0.5,y:1.36,w:9.0,h:2.2,sizing:{type:"contain",w:9.0,h:2.2}});
  s.addText("Effective Rank (90% Variance Threshold) per Layer",{x:0.5,y:3.65,w:9.0,h:0.26,fontSize:11,bold:true,color:NV,fontFace:"Calibri",align:"center",margin:0});
  s.addImage({data:imgB64("slide09b/effective_rank_bar.png"),x:0.5,y:3.93,w:9.0,h:1.5,sizing:{type:"contain",w:9.0,h:1.5}});
}

// ============ SLIDE 10 - TAXONOMY ============
{
  let s=pres.addSlide();s.background={color:WH};
  s.addShape(pres.shapes.RECTANGLE,{x:0,y:0,w:10,h:1.0,fill:{color:NV}});
  s.addText("COMPRESSION LANDSCAPE",{x:0.4,y:0.1,w:2.2,h:0.22,fontSize:7,bold:true,color:"93C5FD",fontFace:"Calibri",align:"left",valign:"middle",charSpacing:1.5,margin:0});
  s.addText("Three-Axis Taxonomy of KV Cache Compression",{x:0.4,y:0.35,w:9.2,h:0.52,fontSize:22,bold:true,color:WH,fontFace:"Calibri",align:"left",margin:0});
  s.addImage({data:imgB64("slide10/taxonomy_diagram.png"),x:0.5,y:1.1,w:9.0,h:4.5,sizing:{type:"contain",w:9.0,h:4.5}});
}

// ============ SLIDE 11 - PIPELINE ============
{
  let s=pres.addSlide();s.background={color:WH};
  s.addShape(pres.shapes.RECTANGLE,{x:0,y:0,w:10,h:1.0,fill:{color:NV}});
  s.addText("EDGE PIPELINE",{x:0.4,y:0.1,w:1.8,h:0.22,fontSize:7,bold:true,color:"93C5FD",fontFace:"Calibri",align:"left",valign:"middle",charSpacing:1.5,margin:0});
  s.addText("KV Cache Compression Lifecycle",{x:0.4,y:0.35,w:9.2,h:0.52,fontSize:22,bold:true,color:WH,fontFace:"Calibri",align:"left",margin:0});
  s.addImage({data:imgB64("slide11/pipeline_diagram.png"),x:0.5,y:1.08,w:9.0,h:3.3,sizing:{type:"contain",w:9.0,h:3.3}});
  s.addShape(pres.shapes.RECTANGLE,{x:0.4,y:4.55,w:9.2,h:0.48,fill:{color:"FEF3C7"},line:{color:AM,width:1.5}});
  s.addShape(pres.shapes.RECTANGLE,{x:0.4,y:4.55,w:0.06,h:0.48,fill:{color:AM}});
  s.addText([{text:"Key insight: ",options:{bold:true,color:AM,fontSize:11}},{text:"Decompression runs once per generated token per layer. Decode speed is the critical path.",options:{color:DT,fontSize:11}}],{x:0.55,y:4.55,w:8.9,h:0.48,fontFace:"Calibri",valign:"middle",margin:0});
}

// ============ SLIDE 12 - DEPLOYMENT TABLE ============
{
  let s=pres.addSlide();s.background={color:WH};
  s.addShape(pres.shapes.RECTANGLE,{x:0,y:0,w:10,h:1.0,fill:{color:NV}});
  s.addText("COMPRESSION LANDSCAPE",{x:0.4,y:0.1,w:2.0,h:0.22,fontSize:7,bold:true,color:"93C5FD",fontFace:"Calibri",align:"left",valign:"middle",charSpacing:1.5,margin:0});
  s.addText("Prescribed Deployment Options by Task",{x:0.4,y:0.35,w:9.2,h:0.52,fontSize:22,bold:true,color:WH,fontFace:"Calibri",align:"left",margin:0});
  s.addText("Which compression axis fits your workload? Based on structural properties each method preserves.",{x:0.4,y:1.08,w:9.2,h:0.32,fontSize:11,color:MT,fontFace:"Calibri",align:"left",margin:0});
  const hf={color:NV};
  const rows=[
    [{text:"Task",options:{bold:true,color:WH,fill:hf,align:"center",fontSize:11}},{text:"Memory Concern",options:{bold:true,color:WH,fill:hf,align:"center",fontSize:11}},{text:"Suggested Axis",options:{bold:true,color:WH,fill:hf,align:"center",fontSize:11}},{text:"Reasoning",options:{bold:true,color:WH,fill:hf,align:"center",fontSize:11}}],
    [{text:"Long Document QA",options:{bold:true,color:BL,fontSize:11}},{text:"Retrieval of specific facts",options:{fontSize:10}},{text:"Eviction (careful)",options:{color:RD,bold:true,fontSize:11}},{text:"Pruned tokens cannot be retrieved",options:{fontSize:10}}],
    [{text:"Summarisation",options:{bold:true,color:BL,fontSize:11}},{text:"Global semantic coherence",options:{fontSize:10}},{text:"Precision reduction",options:{color:BL,bold:true,fontSize:11}},{text:"No single token is critical",options:{fontSize:10}}],
    [{text:"Code Completion",options:{bold:true,color:BL,fontSize:11}},{text:"Local syntactic structure",options:{fontSize:10}},{text:"Precision reduction",options:{color:BL,bold:true,fontSize:11}},{text:"Code has low temporal smoothness",options:{fontSize:10}}],
    [{text:"Multi-turn Chat",options:{bold:true,color:BL,fontSize:11}},{text:"Growing cache across turns",options:{fontSize:10}},{text:"Eviction + Precision",options:{color:TL,bold:true,fontSize:11}},{text:"Cache must stay bounded",options:{fontSize:10}}],
    [{text:"Structured Data",options:{bold:true,color:BL,fontSize:11}},{text:"Exact value preservation",options:{fontSize:10}},{text:"None / High-precision",options:{color:AM,bold:true,fontSize:11}},{text:"Numbers/fields are fragile",options:{fontSize:10}}],
  ];
  s.addTable(rows,{x:0.4,y:1.5,w:9.2,colW:[2.0,2.3,2.3,2.6],border:{pt:0.5,color:"E2E8F0"},fill:{color:"F8FAFC"},fontFace:"Calibri",rowH:0.56,valign:"middle"});
}

// ============ SLIDE 13 - CODEC SCATTERS ============
{
  let s=pres.addSlide();s.background={color:WH};
  s.addShape(pres.shapes.RECTANGLE,{x:0,y:0,w:10,h:1.0,fill:{color:NV}});
  s.addText("CODEC BENCHMARKS",{x:0.4,y:0.1,w:1.8,h:0.22,fontSize:7,bold:true,color:"93C5FD",fontFace:"Calibri",align:"left",valign:"middle",charSpacing:1.5,margin:0});
  s.addText("Codec Speed and Fidelity - No Model Required",{x:0.4,y:0.35,w:9.2,h:0.52,fontSize:20,bold:true,color:WH,fontFace:"Calibri",align:"left",margin:0});
  s.addText("Decode Latency vs Compression Ratio",{x:0.35,y:1.06,w:5.6,h:0.26,fontSize:11,bold:true,color:NV,fontFace:"Calibri",align:"center",margin:0});
  s.addImage({data:imgB64("slide13/codec_latency_scatter.png"),x:0.35,y:1.34,w:5.6,h:3.3,sizing:{type:"contain",w:5.6,h:3.3}});
  s.addText("Attn Logit Error vs MSE",{x:6.1,y:1.06,w:3.7,h:0.26,fontSize:11,bold:true,color:NV,fontFace:"Calibri",align:"center",margin:0});
  s.addImage({data:imgB64("slide13/codec_attn_logit_vs_mse.png"),x:6.1,y:1.34,w:3.7,h:3.3,sizing:{type:"contain",w:3.7,h:3.3}});
  insightBar(s,[{text:"delta_int4 ",options:{bold:true,color:RD}},{text:"is an outlier - high MSE AND high attention logit error. Most codecs cluster near origin with acceptable fidelity.",options:{color:DT}}]);
}

// ============ SLIDE 14 - MSE BAR ============
{
  let s=pres.addSlide();s.background={color:WH};
  s.addShape(pres.shapes.RECTANGLE,{x:0,y:0,w:10,h:1.0,fill:{color:NV}});
  s.addText("CODEC BENCHMARKS",{x:0.4,y:0.1,w:1.8,h:0.22,fontSize:7,bold:true,color:"93C5FD",fontFace:"Calibri",align:"left",valign:"middle",charSpacing:1.5,margin:0});
  s.addText("Reconstruction Quality - MSE and Attention Logit Error per Codec",{x:0.4,y:0.35,w:9.2,h:0.52,fontSize:18,bold:true,color:WH,fontFace:"Calibri",align:"left",margin:0});
  s.addImage({data:imgB64("slide14/codec_mse_bar.png"),x:0.45,y:1.0,w:9.1,h:3.8,sizing:{type:"contain",w:9.1,h:3.8}});
  [{col:BL,t:"INT8/FP16: near-zero error"},{col:TL,t:"KIVI families: low error at high compression"},{col:RD,t:"delta_int4: 10x higher MSE than all other int4"}].forEach((o,i)=>{
    s.addShape(pres.shapes.RECTANGLE,{x:0.4+i*3.1,y:5.05,w:2.9,h:0.42,fill:{color:LG},line:{color:"E2E8F0"}});
    s.addShape(pres.shapes.RECTANGLE,{x:0.4+i*3.1,y:5.05,w:0.06,h:0.42,fill:{color:o.col}});
    s.addText(o.t,{x:0.52+i*3.1,y:5.05,w:2.72,h:0.42,fontSize:9,color:DT,fontFace:"Calibri",valign:"middle",margin:0});
  });
}

// ============ SLIDE 15 - PPL vs RATIO ============
{
  let s=pres.addSlide();s.background={color:WH};
  topStrip(s,"Perplexity vs Compression Ratio - Precision Reduction Codecs","END-TO-END QUALITY");
  const plots=[{f:"slide15/ppl_vs_ratio_tinyllama_prose.png",l:"TinyLlama . Prose",c:BL},{f:"slide15/ppl_vs_ratio_qwen_prose.png",l:"Qwen . Prose",c:TL},{f:"slide15/ppl_vs_ratio_tinyllama_code.png",l:"TinyLlama . Code",c:BL},{f:"slide15/ppl_vs_ratio_qwen_code.png",l:"Qwen . Code",c:TL}];
  plots.forEach((p,i)=>{
    const cx=[0.35,5.3][i%2],cy=[1.12,3.38][Math.floor(i/2)];
    s.addText(p.l,{x:cx,y:cy,w:4.6,h:0.24,fontSize:10,bold:true,color:p.c,fontFace:"Calibri",align:"center",margin:0});
    s.addImage({data:imgB64(p.f),x:cx,y:cy+0.26,w:4.6,h:2.0,sizing:{type:"contain",w:4.6,h:2.0}});
  });
  insightBar(s,[{text:"KIVI maintains near-baseline PPL at 80% compression. ",options:{color:AM,bold:true}},{text:"delta_int4 explodes - do not use for quality-sensitive tasks.",options:{color:DT}}]);
}

// ============ SLIDE 16 - PPL vs BITS ============
{
  let s=pres.addSlide();s.background={color:WH};
  topStrip(s,"Perplexity vs Quantization Bits - How Low Can We Go?","END-TO-END QUALITY");
  s.addImage({data:imgB64("slide16/ppl_vs_bits.png"),x:0.3,y:1.05,w:9.4,h:4.2,sizing:{type:"contain",w:9.4,h:4.2}});
  insightBar(s,[{text:"KIVI (amber) ",options:{bold:true,color:AM}},{text:"stays flat from 8-bit down to 2-bit. ",options:{color:DT}},{text:"Delta (blue) ",options:{bold:true,color:BL}},{text:"works at 8-bit but explodes at 4-bit - quantization error compounds in cumulative sum. ",options:{color:DT}},{text:"Quantization (teal) ",options:{bold:true,color:TL}},{text:"degrades gracefully.",options:{color:DT}}]);
}

// ============ SLIDE 17 - CROSSOVER ============
{
  let s=pres.addSlide();s.background={color:WH};
  topStrip(s,"Crossover Comparison - Precision vs Eviction vs Dimension","END-TO-END QUALITY");
  s.addImage({data:imgB64("slide17/crossover_comparison.png"),x:0.3,y:1.05,w:9.4,h:4.2,sizing:{type:"contain",w:9.4,h:4.2}});
  insightBar(s,[{text:"Precision (blue) dominates at both 2x and 4x targets. ",options:{bold:true,color:BL}},{text:"Eviction (teal) is 10-50x worse at equal memory budget. ",options:{bold:true,color:TL}},{text:"Dimension reduction (purple) is moderate. Solid bars=2x, hatched=4x.",options:{color:DT}}]);
}

pres.writeFile({fileName:"experimental_outputs/slides/slides_8_17.pptx"})
  .then(()=>console.log("DONE - slides_8_17.pptx"))
  .catch(e=>{console.error(e);process.exit(1)});
