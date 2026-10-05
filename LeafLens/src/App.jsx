import React, { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import {
  Leaf, Upload, Cpu, RotateCcw, AlertTriangle,
  FlaskConical, Brain, Database, Target, Microscope, Layers, GitBranch, BarChart3, X, Menu
} from 'lucide-react';
import diseaseData from './disease_info.json';

const fadeUp = (delay = 0) => ({
  hidden: { opacity: 0, y: 20 },
  show:   { opacity: 1, y: 0, transition: { duration: 0.55, ease: [0.22,1,0.36,1], delay } },
});

/* ─── Splash ─────────────────────────────── */
function Splash({ onDone }) {
  return (
    <motion.div className="splash"
      exit={{ opacity: 0 }} transition={{ duration: 0.55 }}
    >
      <motion.div className="splash-mark"
        initial={{ scale: 0.4, opacity: 0 }}
        animate={{ scale: 1,   opacity: 1 }}
        transition={{ type:'spring', stiffness:220, damping:16, delay:0.1 }}
      >
        <Leaf size={34} color="#fff" strokeWidth={2}/>
      </motion.div>

      <motion.div className="splash-wordmark"
        initial={{ opacity:0, y:12 }} animate={{ opacity:1, y:0 }}
        transition={{ delay:0.3, duration:0.4 }}
      >
        Leaf<span>Lens</span>
      </motion.div>

      <motion.p className="splash-sub"
        initial={{ opacity:0 }} animate={{ opacity:1 }} transition={{ delay:0.45 }}
      >
        AI Plant Doctor
      </motion.p>

      <div className="splash-track">
        <motion.div className="splash-fill"
          initial={{ width:'0%' }} animate={{ width:'100%' }}
          transition={{ delay:0.4, duration:1.8, ease:'easeInOut' }}
          onAnimationComplete={onDone}
        />
      </div>
    </motion.div>
  );
}

/* ─── Scan Overlay ───────────────────────── */
function ScanOverlay() {
  return (
    <div className="scan-ov">
      <motion.div className="scan-beam"
        initial={{ top:'0%' }} animate={{ top:'100%' }}
        transition={{ duration:1.8, ease:'linear', repeat:Infinity }}
      />
      <div className="scan-corners">
        <span className="tl"/><span className="tr"/>
        <span className="bl"/><span className="br"/>
      </div>
      <motion.p className="scan-lbl"
        animate={{ opacity:[1,0.3,1] }} transition={{ duration:1.2, repeat:Infinity }}
      >
        Scanning Leaf…
      </motion.p>
    </div>
  );
}

/* ─── App ────────────────────────────────── */
export default function App() {
  const [ready,    setReady]    = useState(false);
  const [preview,  setPreview]  = useState(null);
  const [file,     setFile]     = useState(null);   // actual File object for API
  const [scanning, setScanning] = useState(false);
  const [result,   setResult]   = useState(null);
  const [error,    setError]    = useState(null);
  const [menuOpen, setMenuOpen] = useState(false);

  // lock scroll when menu open
  useEffect(() => {
    document.body.style.overflow = menuOpen ? 'hidden' : '';
    return () => { document.body.style.overflow = ''; };
  }, [menuOpen]);

  const onDrop = useCallback((files) => {
    if (!files[0]) return;
    setFile(files[0]);
    setPreview(URL.createObjectURL(files[0]));
    setResult(null);
    setError(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept:{'image/*':[]}, multiple:false, disabled:scanning,
  });

  const runDiagnosis = async () => {
    if (!file) return;
    setScanning(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setResult({
        name:      data.disease,
        isOk:      data.is_healthy,
        conf:      data.confidence,
        causes:    data.causes,
        solution:  data.solution,
      });
      setTimeout(()=>document.getElementById('res-anchor')?.scrollIntoView({behavior:'smooth'}),60);
    } catch (err) {
      setError('Could not reach the AI server. Make sure the backend is running on port 8000.');
    } finally {
      setScanning(false);
    }
  };

  const reset = () => { setFile(null); setPreview(null); setResult(null); setError(null); };

  return (
    <>
      <AnimatePresence>
        {!ready && <Splash key="splash" onDone={()=>setReady(true)}/>}
      </AnimatePresence>

      {ready && (
        <>
          {/* ── Navbar ── */}
          <motion.header className="navbar"
            initial={{y:-60,opacity:0}} animate={{y:0,opacity:1}}
            transition={{duration:0.45,delay:0.05}}
          >
            <div className="nav-inner">
              <a className="nav-logo" href="#" onClick={()=>setMenuOpen(false)}>
                <div className="nav-logo-mark"><Leaf size={15} color="#fff" strokeWidth={2.5}/></div>
                LeafLens
              </a>
              <nav className="nav-links">
                <a className="nav-link" href="#diagnose">Diagnose</a>
                <a className="nav-link" href="#how">How It Works</a>
                <a className="nav-link" href="#training">Training</a>
                <a className="nav-link" href="#technology">Technology</a>
                <span className="nav-pill">99.06% Accuracy</span>
              </nav>
              {/* Hamburger */}
              <button className="hamburger" onClick={()=>setMenuOpen(o=>!o)} aria-label="Menu">
                {menuOpen ? <X size={22} color="var(--ink)"/> : <Menu size={22} color="var(--ink)"/>}
              </button>
            </div>
          </motion.header>

          {/* ── Mobile Menu ── */}
          <AnimatePresence>
            {menuOpen && (
              <motion.div className="mobile-menu"
                initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}}
                transition={{duration:0.25}}
              >
                {['#diagnose','#how','#training','#technology'].map((href,i) => (
                  <motion.a key={href} href={href} className="mobile-link"
                    onClick={()=>setMenuOpen(false)}
                    initial={{opacity:0,y:12}} animate={{opacity:1,y:0}}
                    transition={{delay:i*0.06}}
                  >
                    {['Diagnose','How It Works','Training','Technology'][i]}
                  </motion.a>
                ))}
                <span className="mobile-pill">99.06% Accuracy</span>
              </motion.div>
            )}
          </AnimatePresence>

          {/* ── Hero ── */}
          <section className="hero" id="diagnose">
            <div className="hero-inner">
              <motion.div variants={fadeUp(0.1)} initial="hidden" animate="show" className="hero-label">
                <span className="hero-dot"/>
                MobileNetV2 · Transfer Learning · PlantVillage
              </motion.div>

              <motion.h1 className="hero-h1" variants={fadeUp(0.2)} initial="hidden" animate="show">
                Diagnose Tomato<br/>Leaf Diseases <em>Instantly</em>
              </motion.h1>

              <motion.p className="hero-desc" variants={fadeUp(0.3)} initial="hidden" animate="show">
                Upload a photo of your tomato plant's leaf. Our deep learning model identifies the disease in seconds and delivers a precise treatment plan.
              </motion.p>

              {/* Stats */}
              <motion.div className="stat-row" variants={fadeUp(0.35)} initial="hidden" animate="show">
                {[
                  ['99.06%','Val. Accuracy'],
                  ['18,160','Training Images'],
                  ['10','Disease Classes'],
                  ['0.17%','Overfit Gap'],
                ].map(([n,l]) => (
                  <div className="st" key={l}><div className="st-n">{n}</div><div className="st-l">{l}</div></div>
                ))}
              </motion.div>

              {/* Upload */}
              <motion.div className="upload-card" variants={fadeUp(0.42)} initial="hidden" animate="show">
                <div {...getRootProps()} className={`dropzone ${isDragActive?'drag':''}`}>
                  <input {...getInputProps()}/>
                  {!preview && (
                    <div className="dz-body">
                      <motion.div className="dz-icon"
                        animate={{y:[0,-7,0]}} transition={{duration:2.8,repeat:Infinity,ease:'easeInOut'}}
                      >
                        <Upload size={24} strokeWidth={1.5}/>
                      </motion.div>
                      <p className="dz-title">{isDragActive?'Drop to scan':'Drag & drop or click to upload'}</p>
                      <p className="dz-sub">JPG or PNG · Max 10 MB</p>
                    </div>
                  )}
                  {preview && (
                    <div className="preview-wrap">
                      <img src={preview} className="preview-img" alt="Leaf"/>
                      {scanning && <ScanOverlay/>}
                    </div>
                  )}
                </div>

                <AnimatePresence>
                  {preview && !result && (
                    <motion.button className="btn-run" onClick={runDiagnosis} disabled={scanning}
                      initial={{opacity:0,y:6}} animate={{opacity:1,y:0}} exit={{opacity:0}}
                    >
                      <Cpu size={16}/>
                      {scanning ? 'Analyzing…' : 'Run Diagnosis'}
                    </motion.button>
                  )}
                </AnimatePresence>
                {error && (
                  <p style={{marginTop:10,color:'var(--danger)',fontSize:'.82rem',textAlign:'center'}}>
                    ⚠ {error}
                  </p>
                )}
              </motion.div>
            </div>
          </section>

          {/* ── Result ── */}
          <span id="res-anchor"/>
          <AnimatePresence>
            {result && (
              <motion.section className="result-section"
                initial={{opacity:0,y:28}} animate={{opacity:1,y:0}} exit={{opacity:0}}
                transition={{duration:0.5,type:'spring',stiffness:150}}
              >
                <div className="result-inner">
                  <div className="result-card">
                    
                    <div className="report-header">
                      <div className="report-badge">AI Analysis Report</div>
                      <div className="report-date">{new Date().toLocaleDateString('en-US', { day: 'numeric', month: 'short', year: 'numeric' })}</div>
                    </div>

                    <div className="report-main">
                      <div className="report-left">
                        <img src={preview} className="report-thumb" alt="Uploaded leaf" />
                      </div>
                      
                      <div className="report-content">
                        <div className="rc-top">
                          <h4 className="rc-label">Detected Condition</h4>
                          <motion.h2 className={`disease-name ${result.isOk?'good':'bad'}`}
                            initial={{opacity:0,x:-12}} animate={{opacity:1,x:0}} transition={{delay:0.1}}
                          >
                            {result.name}
                          </motion.h2>
                        </div>

                        <div className="confidence-meter">
                          <div className="cm-head">
                            <span>Confidence Score</span>
                            <span>{result.conf}%</span>
                          </div>
                          <div className="cm-track">
                            <motion.div className={`cm-fill ${result.isOk?'good':'bad'}`}
                              initial={{width:0}} animate={{width:`${result.conf}%`}} transition={{delay:0.2, duration:1, ease:'easeOut'}}
                            />
                          </div>
                        </div>

                        <div className="info-grid">
                          <motion.div className="info-box"
                            initial={{opacity:0,y:8}} animate={{opacity:1,y:0}} transition={{delay:0.18}}
                          >
                            <div className="info-head danger"><AlertTriangle size={14}/> Root Cause</div>
                            <p>{result.causes}</p>
                          </motion.div>
                          <motion.div className="info-box"
                            initial={{opacity:0,y:8}} animate={{opacity:1,y:0}} transition={{delay:0.26}}
                          >
                            <div className="info-head good"><FlaskConical size={14}/> Recommended Treatment</div>
                            <p>{result.solution}</p>
                          </motion.div>
                        </div>
                      </div>
                    </div>

                    <div className="report-footer">
                      <p className="report-disclaimer">
                        <strong>Note:</strong> This model is specifically trained to analyze tomato leaves. Uploading non-leaf images may result in inaccurate predictions.
                      </p>
                      <button className="btn-reset" onClick={reset}>
                        <RotateCcw size={14}/> Scan Another Leaf
                      </button>
                    </div>

                  </div>
                </div>
              </motion.section>
            )}
          </AnimatePresence>

          {/* ── How It Works ── */}
          <section className="section-block" id="how">
            <div className="section-inner">
              <motion.div initial={{opacity:0,y:18}} whileInView={{opacity:1,y:0}} viewport={{once:true}} transition={{duration:0.5}}>
                <p className="section-kicker">How It Works</p>
                <h2 className="section-title">Three Steps to a Diagnosis</h2>
                <p className="section-body">From a single photo to a complete treatment plan — powered by a fine-tuned convolutional neural network.</p>
              </motion.div>
              <div className="how-grid">
                {[
                  {
                    n:'01', title:'Upload a Leaf Photo',
                    body:'Take a clear, well-lit photo of your tomato plant leaf. Any standard smartphone camera works. The model accepts JPG and PNG formats up to 10 MB.',
                    tag:'Drag & Drop Supported',
                  },
                  {
                    n:'02', title:'CNN Inference',
                    body:'The image is resized to 224×224 and normalized using ImageNet mean/std. It passes through the MobileNetV2 backbone, then a custom head: Dropout(0.3) → Dense(128, ReLU) → Softmax(10). The class with highest probability is returned.',
                    tag:'MobileNetV2 · Softmax(10)',
                  },
                  {
                    n:'03', title:'Diagnosis Report',
                    body:'LeafLens returns the predicted disease class with its confidence score, the responsible pathogen or pest, and a specific evidence-based treatment plan from a curated knowledge base.',
                    tag:'9 Diseases + Healthy Class',
                  },
                ].map(({n,title,body,tag},i) => (
                  <motion.div key={n} className="how-card"
                    initial={{opacity:0,y:16}} whileInView={{opacity:1,y:0}} viewport={{once:true}}
                    transition={{delay:i*0.1, duration:0.48}}
                  >
                    <div className="how-num">{n}</div>
                    <h3>{title}</h3>
                    <p>{body}</p>
                    <span className="how-tag">{tag}</span>
                  </motion.div>
                ))}
              </div>
            </div>
          </section>

          {/* ── Training Process ── */}
          <section className="section-block alt" id="training">
            <div className="section-inner">
              <motion.div initial={{opacity:0,y:18}} whileInView={{opacity:1,y:0}} viewport={{once:true}} transition={{duration:0.5}}>
                <p className="section-kicker">Training Process</p>
                <h2 className="section-title">How the Model Was Built</h2>
                <p className="section-body">A transparent look at every decision — from dataset preparation to final fine-tuning.</p>
              </motion.div>

              <div className="timeline">
                {[
                  {
                    icon: <Database size={17}/>,
                    title: 'Dataset — PlantVillage',
                    body:  '18,160 tomato leaf images across 10 classes (9 diseases + healthy). Split 80/20 into training (14,528) and validation (3,632) sets using a fixed seed of 42 for full reproducibility.',
                    meta:  'Seed 42 · Reproducible Split',
                  },
                  {
                    icon: <Layers size={17}/>,
                    title: 'Data Augmentation',
                    body:  'Training images were augmented on-the-fly: random horizontal flip, ±30° rotation, ColorJitter (brightness ±0.2, contrast ±0.2). Normalization: mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225] (ImageNet statistics).',
                    meta:  'On-the-fly · No disk storage overhead',
                  },
                  {
                    icon: <Brain size={17}/>,
                    title: 'Phase 1 — Head Training (10 Epochs)',
                    body:  'MobileNetV2 backbone frozen. Only the custom classification head trained: Dropout(0.3) → Linear(1280→128, ReLU) → Linear(128→10). Optimizer: Adam (lr = 1e-3). Best validation accuracy after Phase 1: 93.94%.',
                    meta:  'lr = 1e-3 · Val Acc 93.94%',
                  },
                  {
                    icon: <GitBranch size={17}/>,
                    title: 'Phase 2 — Full Fine-Tuning (10 Epochs)',
                    body:  'Entire model unfrozen. All layers updated with Adam (lr = 1e-5) to avoid catastrophic forgetting. ReduceLROnPlateau scheduler (factor 0.5, patience 2). Best model saved by validation accuracy at each epoch.',
                    meta:  'lr = 1e-5 · Val Acc 99.06%',
                  },
                  {
                    icon: <BarChart3 size={17}/>,
                    title: 'Final Results',
                    body:  'Training accuracy: 98.89%. Validation accuracy: 99.06%. Validation loss: 0.0261. Overfitting gap: 0.17% — confirming strong generalization. Even Mosaic Virus (only 373 images) achieved 100% validation accuracy.',
                    meta:  'Val Loss 0.0261 · Overfit Gap 0.17%',
                  },
                ].map(({icon,title,body,meta},i) => (
                  <motion.div key={title} className="tl-item"
                    initial={{opacity:0,x:-14}} whileInView={{opacity:1,x:0}} viewport={{once:true}}
                    transition={{delay:i*0.09, duration:0.46}}
                  >
                    <div className="tl-left">
                      <div className="tl-dot">{icon}</div>
                      <div className="tl-line"/>
                    </div>
                    <div className="tl-body">
                      <h3>{title}</h3>
                      <p>{body}</p>
                      <span className="tl-meta">{meta}</span>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </section>

          {/* ── Technology ── */}
          <section className="section-block" id="technology">
            <div className="section-inner">
              <motion.div initial={{opacity:0,y:18}} whileInView={{opacity:1,y:0}} viewport={{once:true}} transition={{duration:0.5}}>
                <p className="section-kicker">Technology</p>
                <h2 className="section-title">What's Under the Hood</h2>
                <p className="section-body">Every design decision is grounded in peer-reviewed research and real benchmark results.</p>
              </motion.div>

              <div className="specs-grid">
                {[
                  {
                    icon:<Brain size={19}/>, cls:'si-f',
                    title:'MobileNetV2 Architecture',
                    body:'Google\'s efficient CNN using inverted residual blocks and linear bottlenecks. Pre-trained on ImageNet (1.2M images, 1000 classes). Custom head: Dropout(0.3) → Dense(128, ReLU) → Softmax(10).',
                    chip:'Transfer Learning · 2-Phase',
                  },
                  {
                    icon:<Database size={19}/>, cls:'si-t',
                    title:'PlantVillage Dataset',
                    body:'18,160 real leaf images. Significant class imbalance: Mosaic Virus (373 images) to Yellow Leaf Curl (5,357 images) — a 14× gap. Handled effectively through transfer learning.',
                    chip:'10 Classes · 80/20 Split',
                  },
                  {
                    icon:<Target size={19}/>, cls:'si-i',
                    title:'99.06% Validation Accuracy',
                    body:'Achieved after 20 total epochs (10 head + 10 fine-tune). Training accuracy: 98.89%. Validation loss: 0.0261. Overfitting gap of only 0.17% confirms the model generalizes well.',
                    chip:'Val Loss 0.0261 · Gap 0.17%',
                  },
                  {
                    icon:<Microscope size={19}/>, cls:'si-d',
                    title:'Class Imbalance Result',
                    body:'Despite a 14× image count gap between classes, per-class validation accuracies ranged from 96.28% (Early Blight) to 100% (Mosaic Virus & Healthy). Average per-class accuracy: 98.86%.',
                    chip:'Mosaic Virus: 100.00%',
                  },
                ].map(({icon,cls,title,body,chip}) => (
                  <motion.div key={title} className="spec-card"
                    initial={{opacity:0,y:16}} whileInView={{opacity:1,y:0}} viewport={{once:true}}
                    transition={{duration:0.44}} whileHover={{scale:1.01}}
                  >
                    <div className={`spec-icon ${cls}`}>{icon}</div>
                    <h3>{title}</h3>
                    <p>{body}</p>
                    <span className="spec-chip">{chip}</span>
                  </motion.div>
                ))}
              </div>
            </div>
          </section>

          {/* ── Footer ── */}
          <footer>
            <div className="footer-inner">
              <div className="footer-brand">
                <div className="footer-logo">
                  <div className="footer-logo-mark"><Leaf size={14} color="#fff" strokeWidth={2.5}/></div>
                  LeafLens
                </div>
                <p>Crafted by <strong>Abhijeet Kangane</strong></p>
              </div>
              <div className="footer-stats">
                {[
                  ['99.06%','Validation Accuracy'],
                  ['18,160','Training Images'],
                  ['10','Disease Classes'],
                  ['0.17%','Overfit Gap'],
                ].map(([n,l]) => (
                  <div className="fs" key={l}><div className="fs-n">{n}</div><div className="fs-l">{l}</div></div>
                ))}
              </div>
            </div>
          </footer>
        </>
      )}
    </>
  );
}
