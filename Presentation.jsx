import React, { useState, useMemo, useEffect, useCallback } from "react";
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, LineChart, Line, ComposedChart, Area, AreaChart,
  ReferenceDot, ReferenceLine,
} from "recharts";
import {
  ChevronLeft, ChevronRight, AlertTriangle, ShieldCheck, Database, Layers,
  Target, TrendingUp, Lock, Scale, Sparkles, Code2, CheckCircle2, XCircle,
  Activity, Zap, Search, Ban, Copy, Shuffle, GitBranch, ArrowRight, ArrowDown,
  Gauge, BookOpen, Eye,
} from "lucide-react";

/* ============================================================
   1. DONNÉES FACTICES & UTILITAIRES
   ============================================================ */

// Générateur pseudo-aléatoire déterministe (résultats reproductibles)
function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const gauss = (rng) => {
  const u = Math.max(rng(), 1e-9), v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};
const r2 = (x) => Math.round(x * 100) / 100;

// Vrai algorithme SMOTE (k plus proches voisins + interpolation)
function buildSmoteData() {
  const rng = mulberry32(42);
  const majority = Array.from({ length: 100 }, () => ({
    x: r2(3.6 + gauss(rng) * 1.15),
    y: r2(3.6 + gauss(rng) * 1.15),
  }));
  const minority = Array.from({ length: 15 }, () => ({
    x: r2(6.2 + gauss(rng) * 0.75),
    y: r2(6.0 + gauss(rng) * 0.75),
  }));
  const k = 5;
  const synthetic = [];
  const nNeeded = majority.length - minority.length;
  for (let i = 0; i < nNeeded; i++) {
    const p = minority[i % minority.length];
    const neighbors = minority
      .filter((q) => q !== p)
      .map((q) => ({ q, d: Math.hypot(q.x - p.x, q.y - p.y) }))
      .sort((a, b) => a.d - b.d)
      .slice(0, k);
    const nn = neighbors[Math.floor(rng() * neighbors.length)].q;
    const lambda = rng();
    synthetic.push({
      x: r2(p.x + lambda * (nn.x - p.x)),
      y: r2(p.y + lambda * (nn.y - p.y)),
    });
  }
  return { majority, minority, synthetic };
}

// Courbes ROC : TPR = FPR^a  =>  AUC = 1 / (1 + a)
const A_GOOD = 1 / 9; // AUC = 0.90
const A_MID = 1 / 3;  // AUC = 0.75
const rocData = Array.from({ length: 151 }, (_, i) => {
  const f = Math.pow(i / 150, 2.2);
  return {
    fpr: +f.toFixed(4),
    model: +Math.pow(f, A_GOOD).toFixed(4),
    mid: +Math.pow(f, A_MID).toFixed(4),
    random: +f.toFixed(4),
  };
});

// Courbes Precision-Recall : même classifieur (AUC-ROC = 0.9), prévalences différentes
const precisionAt = (tpr, fpr, p) => {
  const tp = tpr * p, fp = fpr * (1 - p);
  return tp + fp === 0 ? 1 : tp / (tp + fp);
};
const prData = [
  { recall: 0, p50: 1, p10: 1, p1: 1 },
  ...Array.from({ length: 100 }, (_, i) => {
    const r = (i + 1) / 100;
    const f = Math.pow(r, 1 / A_GOOD); // inverse de TPR = FPR^a
    return {
      recall: r,
      p50: +precisionAt(r, f, 0.5).toFixed(4),
      p10: +precisionAt(r, f, 0.1).toFixed(4),
      p1: +precisionAt(r, f, 0.01).toFixed(4),
    };
  }),
];
const aucPR = (key) => {
  let s = 0;
  for (let i = 1; i < prData.length; i++)
    s += ((prData[i][key] + prData[i - 1][key]) / 2) * (prData[i].recall - prData[i - 1].recall);
  return s;
};

// Point de fonctionnement utilisé comme exemple (FPR = 5 %)
const OP = { fpr: 0.05, tpr: +Math.pow(0.05, A_GOOD).toFixed(2) }; // TPR ≈ 0.72

// Fonction de répartition normale (pour l'AUC de la slide 9)
const erf = (x) => {
  const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
};
const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
const pdf = (x, m) => Math.exp(-0.5 * (x - m) ** 2) / Math.sqrt(2 * Math.PI);

/* ============================================================
   2. PETITS COMPOSANTS UI
   ============================================================ */

const Card = ({ children, className = "" }) => (
  <div className={`rounded-2xl border border-slate-200 bg-white/90 p-5 shadow-sm backdrop-blur ${className}`}>
    {children}
  </div>
);

const Pill = ({ children, color = "indigo" }) => {
  const map = {
    indigo: "bg-indigo-50 text-indigo-700 border-indigo-200",
    rose: "bg-rose-50 text-rose-700 border-rose-200",
    emerald: "bg-emerald-50 text-emerald-700 border-emerald-200",
    amber: "bg-amber-50 text-amber-700 border-amber-200",
    slate: "bg-slate-100 text-slate-700 border-slate-200",
  };
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-3 py-1 text-xs font-semibold ${map[color]}`}>
      {children}
    </span>
  );
};

const Formula = ({ children }) => (
  <div className="rounded-xl bg-slate-900 px-4 py-3 font-mono text-sm text-indigo-100 shadow-inner">{children}</div>
);

const Stat = ({ label, value, tone = "slate", sub }) => {
  const tones = {
    slate: "text-slate-800", rose: "text-rose-600", emerald: "text-emerald-600",
    amber: "text-amber-600", indigo: "text-indigo-600",
  };
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4">
      <div className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`mt-1 text-3xl font-extrabold ${tones[tone]}`}>{value}</div>
      {sub && <div className="mt-1 text-xs text-slate-500">{sub}</div>}
    </div>
  );
};

const CodeBlock = ({ code }) => (
  <div className="overflow-hidden rounded-2xl bg-slate-900 shadow-lg">
    <div className="flex items-center gap-2 border-b border-slate-700 px-4 py-2">
      <span className="h-3 w-3 rounded-full bg-rose-400" />
      <span className="h-3 w-3 rounded-full bg-amber-400" />
      <span className="h-3 w-3 rounded-full bg-emerald-400" />
      <span className="ml-2 text-xs text-slate-400">pipeline.py</span>
    </div>
    <pre className="overflow-x-auto p-4 text-[12.5px] leading-relaxed">
      {code.split("\n").map((l, i) => (
        <div key={i} className={l.trim().startsWith("#") ? "text-emerald-400" : l.includes("imblearn") ? "text-amber-300" : "text-slate-100"}>
          {l || "\u00A0"}
        </div>
      ))}
    </pre>
  </div>
);

const chartTick = { fontSize: 12, fill: "#64748b" };

/* ============================================================
   3. SLIDES
   ============================================================ */

// ---------- 1. Titre ----------
function SlideTitle() {
  return (
    <div className="flex h-full flex-col items-center justify-center text-center">
      <div className="mb-6 flex items-center gap-3">
        <Pill color="indigo"><Sparkles size={14} /> Machine Learning</Pill>
        <Pill color="slate">Données déséquilibrées</Pill>
      </div>
      <h1 className="max-w-4xl text-5xl font-black leading-tight tracking-tight text-slate-900 md:text-6xl">
        SMOTE, ROC & AUC :
        <span className="block bg-gradient-to-r from-indigo-600 via-violet-600 to-rose-500 bg-clip-text text-transparent">
          Maîtriser les données déséquilibrées
        </span>
      </h1>
      <p className="mt-6 max-w-2xl text-lg text-slate-600">
        Gérer le déséquilibre de classes sans sur-apprentissage, et évaluer un modèle au-delà de la simple « accuracy ».
      </p>
      <div className="mt-10 grid w-full max-w-3xl grid-cols-1 gap-4 md:grid-cols-2">
        <Card className="text-left">
          <Scale className="mb-2 text-indigo-600" />
          <div className="font-bold text-slate-800">Objectif 1</div>
          <div className="text-sm text-slate-600">Rééquilibrer les classes <b>sans overfitting</b> ni fuite de données.</div>
        </Card>
        <Card className="text-left">
          <Gauge className="mb-2 text-rose-500" />
          <div className="font-bold text-slate-800">Objectif 2</div>
          <div className="text-sm text-slate-600">Évaluer avec les bonnes métriques : <b>ROC, AUC, Precision-Recall</b>.</div>
        </Card>
      </div>
      <p className="mt-8 text-sm text-slate-400">Utilisez les boutons ou les flèches ← → du clavier</p>
    </div>
  );
}

// ---------- 2. Limites de l'accuracy ----------
function SlideAccuracy() {
  const fraudIdx = 37;
  return (
    <div className="grid h-full grid-cols-1 gap-6 lg:grid-cols-2">
      <Card>
        <div className="mb-3 flex items-center justify-between">
          <div className="font-bold text-slate-800">100 transactions bancaires</div>
          <Pill color="rose">1 fraude</Pill>
        </div>
        <div className="grid grid-cols-10 gap-1.5">
          {Array.from({ length: 100 }, (_, i) => (
            <div
              key={i}
              className={`aspect-square rounded-md ${i === fraudIdx ? "bg-rose-500 ring-4 ring-rose-200" : "bg-slate-200"}`}
            />
          ))}
        </div>
        <div className="mt-4 flex gap-4 text-sm text-slate-600">
          <span className="flex items-center gap-2"><span className="h-3 w-3 rounded bg-slate-300" /> Légitime (99 %)</span>
          <span className="flex items-center gap-2"><span className="h-3 w-3 rounded bg-rose-500" /> Fraude (1 %)</span>
        </div>
      </Card>

      <div className="flex flex-col gap-4">
        <Card className="border-amber-200 bg-amber-50/80">
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 shrink-0 text-amber-600" />
            <div>
              <div className="font-bold text-amber-900">Le paradoxe de l'accuracy</div>
              <p className="mt-1 text-sm text-amber-900/80">
                Un modèle « idiot » qui répond <b>toujours « légitime »</b> obtient une excellente accuracy… tout en ne détectant
                <b> aucune</b> fraude.
              </p>
            </div>
          </div>
        </Card>
        <div className="grid grid-cols-2 gap-4">
          <Stat label="Accuracy" value="99 %" tone="emerald" sub="(TP + TN) / total — trompeuse" />
          <Stat label="Recall (fraudes)" value="0 %" tone="rose" sub="0 fraude détectée sur 1" />
        </div>
        <Formula>Accuracy = (TP + TN) / (TP + TN + FP + FN)</Formula>
        <Card>
          <div className="text-sm text-slate-700">
            <b>Conséquence :</b> l'accuracy est dominée par la classe majoritaire. En détection de fraude, de maladie rare ou de
            panne, c'est <span className="font-semibold text-rose-600">la classe minoritaire qui coûte cher</span>.
          </div>
        </Card>
      </div>
    </div>
  );
}

// ---------- 3. Intro SMOTE ----------
function SlideSmoteIntro() {
  const rows = [
    { icon: <Ban className="text-slate-500" />, name: "Under-sampling", desc: "Supprimer des exemples majoritaires", con: "Perte d'information", tone: "slate" },
    { icon: <Copy className="text-amber-500" />, name: "Random over-sampling", desc: "Dupliquer des exemples minoritaires", con: "Overfitting : le modèle mémorise", tone: "amber" },
    { icon: <Sparkles className="text-emerald-500" />, name: "SMOTE", desc: "Créer de nouveaux points synthétiques", con: "Généralise mieux, mais attention aux limites", tone: "emerald" },
  ];
  const border = { slate: "border-slate-200", amber: "border-amber-200", emerald: "border-emerald-300 ring-2 ring-emerald-100" };
  return (
    <div className="grid h-full grid-cols-1 gap-6 lg:grid-cols-5">
      <div className="flex flex-col gap-4 lg:col-span-3">
        {rows.map((r) => (
          <div key={r.name} className={`flex items-center gap-4 rounded-2xl border bg-white p-5 shadow-sm ${border[r.tone]}`}>
            <div className="rounded-xl bg-slate-50 p-3">{r.icon}</div>
            <div className="flex-1">
              <div className="font-bold text-slate-800">{r.name}</div>
              <div className="text-sm text-slate-600">{r.desc}</div>
            </div>
            <Pill color={r.tone}>{r.con}</Pill>
          </div>
        ))}
        <Card className="bg-indigo-50/70">
          <div className="font-bold text-indigo-900">SMOTE = Synthetic Minority Over-sampling Technique</div>
          <p className="mt-1 text-sm text-indigo-900/80">
            Proposée par Chawla et al. (2002). Au lieu de recopier des exemples, SMOTE <b>fabrique de nouveaux exemples plausibles</b> le
            long des segments qui relient des points minoritaires voisins.
          </p>
        </Card>
      </div>
      <Card className="lg:col-span-2">
        <div className="mb-3 font-bold text-slate-800">À garder en tête</div>
        <ul className="space-y-3 text-sm text-slate-700">
          <li className="flex gap-2"><CheckCircle2 size={18} className="mt-0.5 shrink-0 text-emerald-500" /> Élargit la région de décision de la classe minoritaire.</li>
          <li className="flex gap-2"><CheckCircle2 size={18} className="mt-0.5 shrink-0 text-emerald-500" /> Fonctionne sur des variables numériques continues.</li>
          <li className="flex gap-2"><XCircle size={18} className="mt-0.5 shrink-0 text-rose-500" /> Peut amplifier le bruit et les points aberrants.</li>
          <li className="flex gap-2"><XCircle size={18} className="mt-0.5 shrink-0 text-rose-500" /> Variables catégorielles : utiliser <code className="rounded bg-slate-100 px-1">SMOTENC</code>.</li>
          <li className="flex gap-2"><XCircle size={18} className="mt-0.5 shrink-0 text-rose-500" /> Efficacité réduite en très haute dimension.</li>
        </ul>
      </Card>
    </div>
  );
}

// ---------- 4. Mécanisme ----------
function SlideMechanism() {
  const [lambda, setLambda] = useState(0.4);
  const A = { x: 70, y: 150 };
  const neighbors = [
    { x: 330, y: 55, chosen: true },
    { x: 215, y: 175 },
    { x: 140, y: 45 },
  ];
  const B = neighbors[0];
  const S = { x: A.x + lambda * (B.x - A.x), y: A.y + lambda * (B.y - A.y) };
  const steps = [
    "Choisir un point minoritaire xᵢ",
    "Trouver ses k plus proches voisins minoritaires",
    "Tirer au hasard un voisin x_nn",
    "Créer x_new = xᵢ + λ · (x_nn − xᵢ), avec λ ∈ [0, 1]",
  ];
  return (
    <div className="grid h-full grid-cols-1 gap-6 lg:grid-cols-2">
      <div className="flex flex-col gap-4">
        <Card>
          <div className="mb-3 flex items-center gap-2 font-bold text-slate-800"><Shuffle size={18} className="text-indigo-600" /> L'algorithme en 4 étapes</div>
          <ol className="space-y-2">
            {steps.map((s, i) => (
              <li key={i} className="flex items-center gap-3 text-sm text-slate-700">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-indigo-600 text-xs font-bold text-white">{i + 1}</span>
                {s}
              </li>
            ))}
          </ol>
        </Card>
        <Formula>x_new = xᵢ + {lambda.toFixed(2)} × (x_nn − xᵢ)</Formula>
        <div className="grid grid-cols-2 gap-3">
          <Card className="border-amber-200">
            <Copy className="mb-1 text-amber-500" size={20} />
            <div className="text-sm font-bold text-slate-800">Duplication</div>
            <p className="text-xs text-slate-600">Points identiques → frontière « mémorisée » → overfitting.</p>
          </Card>
          <Card className="border-emerald-200">
            <Sparkles className="mb-1 text-emerald-500" size={20} />
            <div className="text-sm font-bold text-slate-800">Interpolation</div>
            <p className="text-xs text-slate-600">Points nouveaux et variés → zone minoritaire plus générale.</p>
          </Card>
        </div>
      </div>

      <Card>
        <div className="mb-2 flex items-center justify-between">
          <div className="font-bold text-slate-800">Interpolation interactive (k = 3)</div>
          <Pill color="emerald">λ = {lambda.toFixed(2)}</Pill>
        </div>
        <svg viewBox="0 0 400 220" className="w-full rounded-xl bg-slate-50">
          {neighbors.map((n, i) => (
            <line key={i} x1={A.x} y1={A.y} x2={n.x} y2={n.y} stroke={n.chosen ? "#6366f1" : "#cbd5e1"} strokeWidth={n.chosen ? 2.5 : 1.5} strokeDasharray={n.chosen ? "0" : "5 4"} />
          ))}
          {neighbors.map((n, i) => (
            <circle key={i} cx={n.x} cy={n.y} r="9" fill="#f43f5e" opacity={n.chosen ? 1 : 0.55} />
          ))}
          <circle cx={A.x} cy={A.y} r="11" fill="#f43f5e" stroke="#fff" strokeWidth="3" />
          <circle cx={S.x} cy={S.y} r="9" fill="#10b981" stroke="#fff" strokeWidth="3" />
          <text x={A.x - 8} y={A.y + 28} fontSize="12" fill="#475569">xᵢ</text>
          <text x={B.x - 20} y={B.y - 14} fontSize="12" fill="#475569">x_nn</text>
          <text x={S.x + 12} y={S.y - 10} fontSize="12" fontWeight="700" fill="#059669">x_new</text>
        </svg>
        <input type="range" min="0" max="1" step="0.01" value={lambda} onChange={(e) => setLambda(+e.target.value)} className="mt-4 w-full accent-emerald-600" />
        <div className="flex justify-between text-xs text-slate-500"><span>λ = 0 (= xᵢ)</span><span>λ = 1 (= x_nn)</span></div>
        <div className="mt-3 flex gap-4 text-xs text-slate-600">
          <span className="flex items-center gap-1"><span className="h-3 w-3 rounded-full bg-rose-500" /> minoritaire</span>
          <span className="flex items-center gap-1"><span className="h-3 w-3 rounded-full bg-emerald-500" /> synthétique</span>
        </div>
      </Card>
    </div>
  );
}

// ---------- 5. GRAPHIQUE 1 : Scatter ----------
function SlideScatter() {
  const data = useMemo(buildSmoteData, []);
  const [after, setAfter] = useState(false);
  return (
    <div className="grid h-full grid-cols-1 gap-6 lg:grid-cols-3">
      <Card className="lg:col-span-2">
        <div className="mb-3 flex items-center justify-between">
          <div className="inline-flex rounded-xl bg-slate-100 p-1">
            {[false, true].map((v) => (
              <button
                key={String(v)}
                onClick={() => setAfter(v)}
                className={`rounded-lg px-4 py-1.5 text-sm font-semibold transition ${after === v ? "bg-white text-indigo-700 shadow" : "text-slate-500 hover:text-slate-700"}`}
              >
                {v ? "Après SMOTE" : "Avant SMOTE"}
              </button>
            ))}
          </div>
          <Pill color={after ? "emerald" : "rose"}>{after ? "Classes équilibrées" : "Ratio 100 : 15"}</Pill>
        </div>
        <div className="h-[380px]">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis type="number" dataKey="x" domain={[0, 9]} tick={chartTick} name="Feature 1" />
              <YAxis type="number" dataKey="y" domain={[0, 9]} tick={chartTick} name="Feature 2" />
              <ZAxis range={[55, 55]} />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} />
              <Legend />
              <Scatter name="Majoritaire (100)" data={data.majority} fill="#64748b" fillOpacity={0.65} />
              <Scatter name="Minoritaire (15)" data={data.minority} fill="#f43f5e" />
              {after && <Scatter name="Synthétiques (85)" data={data.synthetic} fill="#10b981" fillOpacity={0.75} shape="diamond" />}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </Card>
      <div className="flex flex-col gap-4">
        <Stat label="Classe majoritaire" value="100" tone="slate" />
        <Stat label="Classe minoritaire" value={after ? "100" : "15"} tone={after ? "emerald" : "rose"} sub={after ? "15 réels + 85 synthétiques" : "15 réels"} />
        <Card className="text-sm text-slate-700">
          <Eye size={18} className="mb-1 text-indigo-600" />
          Les points verts se trouvent <b>entre</b> des points rouges existants : ils remplissent la zone minoritaire au lieu de s'empiler.
        </Card>
      </div>
    </div>
  );
}

// ---------- 6. Règle d'or ----------
function FlowBox({ children, tone = "slate" }) {
  const t = {
    slate: "bg-slate-100 text-slate-700 border-slate-200",
    rose: "bg-rose-50 text-rose-800 border-rose-300",
    emerald: "bg-emerald-50 text-emerald-800 border-emerald-300",
    indigo: "bg-indigo-50 text-indigo-800 border-indigo-200",
  };
  return <div className={`rounded-lg border px-3 py-2 text-center text-sm font-medium ${t[tone]}`}>{children}</div>;
}
function SlideGoldenRule() {
  return (
    <div className="flex h-full flex-col gap-5">
      <div className="flex items-center gap-4 rounded-2xl bg-gradient-to-r from-amber-400 to-rose-500 p-5 text-white shadow-lg">
        <Lock size={34} className="shrink-0" />
        <div>
          <div className="text-xs font-bold uppercase tracking-widest opacity-90">La règle d'or</div>
          <div className="text-xl font-extrabold md:text-2xl">SMOTE s'applique UNIQUEMENT sur le Train set. Jamais sur le Test / Validation.</div>
        </div>
      </div>
      <div className="grid flex-1 grid-cols-1 gap-5 lg:grid-cols-2">
        <Card className="border-rose-200">
          <div className="mb-3 flex items-center gap-2 font-bold text-rose-700"><XCircle size={20} /> Incorrect : fuite de données</div>
          <div className="flex flex-col items-stretch gap-2">
            <FlowBox>Dataset complet</FlowBox>
            <div className="flex justify-center text-rose-400"><ArrowDown size={18} /></div>
            <FlowBox tone="rose">SMOTE sur tout le dataset</FlowBox>
            <div className="flex justify-center text-rose-400"><ArrowDown size={18} /></div>
            <FlowBox>train_test_split</FlowBox>
            <div className="flex justify-center text-rose-400"><ArrowDown size={18} /></div>
            <FlowBox tone="rose">Test contient des points synthétiques dérivés du Train → score gonflé</FlowBox>
          </div>
        </Card>
        <Card className="border-emerald-200">
          <div className="mb-3 flex items-center gap-2 font-bold text-emerald-700"><CheckCircle2 size={20} /> Correct</div>
          <div className="flex flex-col items-stretch gap-2">
            <FlowBox>Dataset complet</FlowBox>
            <div className="flex justify-center text-emerald-400"><ArrowDown size={18} /></div>
            <FlowBox>train_test_split (stratifié)</FlowBox>
            <div className="flex justify-center text-emerald-400"><ArrowDown size={18} /></div>
            <div className="grid grid-cols-2 gap-2">
              <FlowBox tone="emerald">Train → SMOTE → fit</FlowBox>
              <FlowBox tone="indigo">Test intact (distribution réelle)</FlowBox>
            </div>
            <div className="flex justify-center text-emerald-400"><ArrowDown size={18} /></div>
            <FlowBox tone="emerald">Évaluation honnête sur données réelles</FlowBox>
          </div>
        </Card>
      </div>
      <Card className="bg-slate-50 text-sm text-slate-700">
        <b>Pourquoi ?</b> Un point synthétique est une combinaison de voisins réels : si ces voisins (ou le point lui-même) se retrouvent dans le Test,
        le modèle a déjà « vu » l'information. Le jeu de test doit refléter la <b>vraie distribution déséquilibrée</b> de production. Même règle pour
        chaque fold de la validation croisée.
      </Card>
    </div>
  );
}

// ---------- 7. Fondements ----------
function SlideFundamentals() {
  const TP = 80, FN = 20, FP = 45, TN = 855;
  const tpr = TP / (TP + FN), fpr = FP / (FP + TN);
  const Cell = ({ title, val, desc, cls }) => (
    <div className={`rounded-xl border p-4 ${cls}`}>
      <div className="text-xs font-bold uppercase tracking-wide opacity-80">{title}</div>
      <div className="text-3xl font-black">{val}</div>
      <div className="text-xs opacity-80">{desc}</div>
    </div>
  );
  return (
    <div className="grid h-full grid-cols-1 gap-6 lg:grid-cols-2">
      <Card>
        <div className="mb-3 font-bold text-slate-800">Matrice de confusion (exemple)</div>
        <div className="mb-2 grid grid-cols-[auto,1fr,1fr] items-center gap-2 text-center text-xs font-semibold text-slate-500">
          <div />
          <div>Prédit : Positif</div>
          <div>Prédit : Négatif</div>
        </div>
        <div className="grid grid-cols-[auto,1fr,1fr] items-stretch gap-2">
          <div className="flex items-center text-xs font-semibold text-slate-500 [writing-mode:vertical-rl] rotate-180">Réel : Positif</div>
          <Cell title="Vrais Positifs (TP)" val={TP} desc="Fraudes détectées" cls="border-emerald-300 bg-emerald-50 text-emerald-800" />
          <Cell title="Faux Négatifs (FN)" val={FN} desc="Fraudes manquées" cls="border-rose-300 bg-rose-50 text-rose-800" />
          <div className="flex items-center text-xs font-semibold text-slate-500 [writing-mode:vertical-rl] rotate-180">Réel : Négatif</div>
          <Cell title="Faux Positifs (FP)" val={FP} desc="Fausses alertes" cls="border-amber-300 bg-amber-50 text-amber-800" />
          <Cell title="Vrais Négatifs (TN)" val={TN} desc="Légitimes OK" cls="border-slate-200 bg-slate-50 text-slate-700" />
        </div>
      </Card>
      <div className="flex flex-col gap-4">
        <Card>
          <div className="flex items-center gap-2 font-bold text-slate-800"><Target size={18} className="text-emerald-600" /> TPR — Taux de vrais positifs</div>
          <p className="text-xs text-slate-500">Aussi appelé Recall ou Sensibilité : part des positifs réellement détectés.</p>
          <div className="mt-2"><Formula>TPR = TP / (TP + FN) = {TP}/{TP + FN} = {(tpr * 100).toFixed(0)} %</Formula></div>
        </Card>
        <Card>
          <div className="flex items-center gap-2 font-bold text-slate-800"><AlertTriangle size={18} className="text-amber-600" /> FPR — Taux de faux positifs</div>
          <p className="text-xs text-slate-500">Part des négatifs à tort classés positifs (= 1 − spécificité).</p>
          <div className="mt-2"><Formula>FPR = FP / (FP + TN) = {FP}/{FP + TN} = {(fpr * 100).toFixed(0)} %</Formula></div>
        </Card>
        <Card className="bg-indigo-50/70 text-sm text-indigo-900">
          TPR et FPR sont calculés <b>chacun au sein d'une seule classe réelle</b> : ils ne dépendent donc pas de la proportion de positifs.
          Bonne propriété… mais source d'un piège (slide 11).
        </Card>
      </div>
    </div>
  );
}

// ---------- 8. ROC : définition ----------
function SlideROCDef() {
  const rows = [
    { t: "Très strict (0.9)", fpr: 0.005 },
    { t: "Strict (0.7)", fpr: 0.02 },
    { t: "Équilibré (0.5)", fpr: 0.1 },
    { t: "Laxiste (0.3)", fpr: 0.35 },
    { t: "Très laxiste (0.1)", fpr: 0.75 },
  ];
  const path = rocData.map((d, i) => `${i ? "L" : "M"}${20 + d.fpr * 200},${220 - d.model * 200}`).join(" ");
  return (
    <div className="grid h-full grid-cols-1 gap-6 lg:grid-cols-2">
      <div className="flex flex-col gap-4">
        <Card>
          <div className="mb-2 flex items-center gap-2 font-bold text-slate-800"><BookOpen size={18} className="text-indigo-600" /> Définition</div>
          <p className="text-sm text-slate-700">
            La courbe <b>ROC</b> (Receiver Operating Characteristic) trace le <b className="text-emerald-600">TPR</b> en fonction du{" "}
            <b className="text-amber-600">FPR</b> pour <b>tous les seuils de classification</b> possibles. Chaque point = un seuil.
          </p>
        </Card>
        <Card className="p-0">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-left text-xs uppercase text-slate-500">
                <th className="px-4 py-3">Seuil</th><th className="px-4 py-3">FPR</th><th className="px-4 py-3">TPR</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.t} className="border-b border-slate-100 last:border-0">
                  <td className="px-4 py-2 text-slate-700">{r.t}</td>
                  <td className="px-4 py-2 font-mono text-amber-600">{r.fpr.toFixed(3)}</td>
                  <td className="px-4 py-2 font-mono text-emerald-600">{Math.pow(r.fpr, A_GOOD).toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
        <p className="text-sm text-slate-600">Abaisser le seuil → on détecte plus de positifs (TPR ↑) mais on déclenche plus de fausses alertes (FPR ↑).</p>
      </div>
      <Card>
        <div className="mb-2 font-bold text-slate-800">Lecture de la courbe</div>
        <svg viewBox="0 0 260 260" className="mx-auto w-full max-w-sm">
          <rect x="20" y="20" width="200" height="200" fill="#f8fafc" stroke="#cbd5e1" />
          <line x1="20" y1="220" x2="220" y2="20" stroke="#94a3b8" strokeDasharray="5 4" />
          <path d={path} fill="none" stroke="#6366f1" strokeWidth="3" />
          <circle cx="20" cy="220" r="5" fill="#64748b" />
          <circle cx="20" cy="20" r="6" fill="#10b981" />
          <circle cx="220" cy="20" r="5" fill="#64748b" />
          <text x="26" y="16" fontSize="10" fill="#059669" fontWeight="700">(0,1) modèle parfait</text>
          <text x="26" y="236" fontSize="10" fill="#475569">(0,0) tout négatif</text>
          <text x="150" y="236" fontSize="10" fill="#475569">FPR →</text>
          <text x="125" y="125" fontSize="10" fill="#64748b" transform="rotate(-45 125 125)">aléatoire</text>
          <text x="2" y="130" fontSize="10" fill="#475569" transform="rotate(-90 8 130)">TPR →</text>
          <text x="150" y="38" fontSize="10" fill="#475569">(1,1) tout positif</text>
        </svg>
        <p className="mt-2 text-center text-xs text-slate-500">Plus la courbe épouse le coin supérieur gauche, meilleur est le modèle.</p>
      </Card>
    </div>
  );
}

// ---------- 9. AUC ----------
function SlideAUC() {
  const [d, setD] = useState(1.8);
  const auc = Phi(d / Math.SQRT2);
  const dist = useMemo(
    () => Array.from({ length: 81 }, (_, i) => {
      const x = -4 + i * 0.15;
      return { x: +x.toFixed(2), neg: +pdf(x, 0).toFixed(4), pos: +pdf(x, d).toFixed(4) };
    }),
    [d]
  );
  const verdict = auc < 0.6 ? "Quasi aléatoire" : auc < 0.7 ? "Faible" : auc < 0.8 ? "Acceptable" : auc < 0.9 ? "Bon" : auc < 0.97 ? "Excellent" : "Quasi parfait";
  const scale = [
    { v: "0.5", l: "Aléatoire", c: "bg-slate-100 text-slate-700" },
    { v: "0.7 – 0.8", l: "Acceptable", c: "bg-amber-50 text-amber-800" },
    { v: "0.8 – 0.9", l: "Bon", c: "bg-indigo-50 text-indigo-800" },
    { v: "> 0.9", l: "Excellent", c: "bg-emerald-50 text-emerald-800" },
  ];
  return (
    <div className="grid h-full grid-cols-1 gap-6 lg:grid-cols-3">
      <Card className="lg:col-span-2">
        <div className="mb-1 font-bold text-slate-800">Distribution des scores prédits</div>
        <div className="h-[260px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={dist} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="x" type="number" domain={[-4, 8]} tick={chartTick} />
              <YAxis tick={chartTick} />
              <Area type="monotone" dataKey="neg" name="Négatifs" stroke="#64748b" fill="#64748b" fillOpacity={0.25} />
              <Area type="monotone" dataKey="pos" name="Positifs" stroke="#f43f5e" fill="#f43f5e" fillOpacity={0.25} />
              <Legend />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <label className="mt-2 block text-sm font-medium text-slate-600">Séparation entre les classes : {d.toFixed(1)}</label>
        <input type="range" min="0" max="4" step="0.1" value={d} onChange={(e) => setD(+e.target.value)} className="w-full accent-indigo-600" />
      </Card>
      <div className="flex flex-col gap-4">
        <div className="rounded-2xl bg-gradient-to-br from-indigo-600 to-violet-600 p-5 text-white shadow-lg">
          <div className="text-xs uppercase tracking-widest opacity-80">AUC-ROC</div>
          <div className="text-5xl font-black">{auc.toFixed(3)}</div>
          <div className="mt-1 text-sm font-semibold opacity-90">{verdict}</div>
        </div>
        <Card className="text-sm text-slate-700">
          <b>Interprétation :</b> l'AUC est la probabilité qu'un positif tiré au hasard reçoive un <b>score plus élevé</b> qu'un négatif tiré au hasard. Elle mesure la
          capacité de <b>séparation</b>, indépendamment du seuil.
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-3 lg:col-span-3 lg:grid-cols-4">
        {scale.map((s) => (
          <div key={s.v} className={`rounded-xl p-3 text-center ${s.c}`}>
            <div className="text-lg font-black">{s.v}</div>
            <div className="text-xs font-medium">{s.l}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------- 10. GRAPHIQUE 2 : ROC ----------
function SlideROCChart() {
  const [f, setF] = useState(0.1);
  const t = Math.pow(f, A_GOOD);
  return (
    <div className="grid h-full grid-cols-1 gap-6 lg:grid-cols-3">
      <Card className="lg:col-span-2">
        <div className="h-[420px]">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={rocData} margin={{ top: 10, right: 20, bottom: 25, left: 0 }}>
              <defs>
                <linearGradient id="aucFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#6366f1" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="#6366f1" stopOpacity={0.03} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis type="number" dataKey="fpr" domain={[0, 1]} tick={chartTick} label={{ value: "FPR (taux de faux positifs)", position: "insideBottom", offset: -12, fontSize: 12, fill: "#475569" }} />
              <YAxis type="number" domain={[0, 1]} tick={chartTick} label={{ value: "TPR (rappel)", angle: -90, position: "insideLeft", fontSize: 12, fill: "#475569" }} />
              <Tooltip formatter={(v) => v.toFixed(3)} labelFormatter={(l) => `FPR = ${Number(l).toFixed(3)}`} />
              <Legend verticalAlign="top" />
              <Area type="monotone" dataKey="model" name="Bon modèle (AUC = 0.90)" stroke="#6366f1" strokeWidth={3} fill="url(#aucFill)" dot={false} />
              <Line type="monotone" dataKey="mid" name="Modèle moyen (AUC = 0.75)" stroke="#f59e0b" strokeWidth={2} dot={false} />
              <Line type="linear" dataKey="random" name="Aléatoire (AUC = 0.50)" stroke="#94a3b8" strokeDasharray="6 5" strokeWidth={2} dot={false} />
              <ReferenceDot x={f} y={t} r={7} fill="#f43f5e" stroke="#fff" strokeWidth={2} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </Card>
      <div className="flex flex-col gap-4">
        <Card>
          <div className="mb-2 font-bold text-slate-800">Explorer un seuil</div>
          <input type="range" min="0.005" max="1" step="0.005" value={f} onChange={(e) => setF(+e.target.value)} className="w-full accent-rose-500" />
          <div className="mt-3 grid grid-cols-2 gap-3">
            <Stat label="FPR" value={f.toFixed(2)} tone="amber" />
            <Stat label="TPR" value={t.toFixed(2)} tone="emerald" />
          </div>
          <p className="mt-3 text-xs text-slate-500">Le point rouge se déplace le long de la courbe quand on change le seuil.</p>
        </Card>
        <Card className="text-sm text-slate-700">
          <TrendingUp size={18} className="mb-1 text-indigo-600" />
          Avec un FPR de seulement <b>5 %</b>, le bon modèle détecte déjà <b>≈ {Math.round(OP.tpr * 100)} %</b> des positifs.
        </Card>
      </div>
    </div>
  );
}

// ---------- 11. Piège du déséquilibre extrême ----------
function SlidePitfall() {
  const [exp, setExp] = useState(-3);
  const p = Math.pow(10, exp);
  const N = 100000;
  const pos = N * p, TP = OP.tpr * pos, FP = OP.fpr * (N - pos);
  const precision = TP / (TP + FP);
  const share = (precision * 100);
  const fmt = (n) => Math.round(n).toLocaleString("fr-FR");
  return (
    <div className="grid h-full grid-cols-1 gap-6 lg:grid-cols-2">
      <div className="flex flex-col gap-4">
        <Card className="border-amber-200 bg-amber-50/80">
          <div className="flex gap-3">
            <AlertTriangle className="shrink-0 text-amber-600" />
            <p className="text-sm text-amber-900">
              L'AUC-ROC reste <b>très flatteuse</b> quand les positifs sont rarissimes : le FPR divise les faux positifs par un
              <b> énorme</b> nombre de négatifs, donc même des milliers de fausses alertes donnent un « petit » FPR.
            </p>
          </div>
        </Card>
        <div className="grid grid-cols-3 gap-3">
          <Stat label="AUC-ROC" value="0.90" tone="emerald" sub="Semble excellent" />
          <Stat label="TPR" value={`${Math.round(OP.tpr * 100)} %`} tone="emerald" />
          <Stat label="FPR" value={`${OP.fpr * 100} %`} tone="amber" sub="« seulement »" />
        </div>
        <Formula>Precision = TP / (TP + FP)  — dépend de la prévalence</Formula>
        <Card className="text-sm text-slate-700">
          Le <b>FPR</b> n'est pas affecté par le nombre de positifs, mais la <b>précision</b> l'est fortement : c'est elle qui reflète la réalité
          opérationnelle (combien d'alertes sont de vraies fraudes ?).
        </Card>
      </div>

      <Card>
        <div className="mb-1 font-bold text-slate-800">Simulation sur {fmt(N)} transactions</div>
        <label className="text-sm text-slate-600">Prévalence des fraudes : <b className="text-rose-600">{(p * 100).toFixed(p < 0.01 ? 2 : 1)} %</b></label>
        <input type="range" min="-4" max="-0.3" step="0.05" value={exp} onChange={(e) => setExp(+e.target.value)} className="mt-1 w-full accent-rose-500" />
        <div className="mt-1 flex justify-between text-xs text-slate-400"><span>0.01 %</span><span>50 %</span></div>

        <div className="mt-4 text-xs font-semibold uppercase text-slate-500">Alertes déclenchées</div>
        <div className="mt-1 flex h-9 w-full overflow-hidden rounded-lg bg-slate-100">
          <div className="flex items-center justify-center bg-emerald-500 text-xs font-bold text-white transition-all" style={{ width: `${Math.max(share, 0.5)}%` }}>
            {share > 12 ? `${share.toFixed(0)} %` : ""}
          </div>
          <div className="flex flex-1 items-center justify-center bg-amber-400 text-xs font-bold text-amber-950">{share < 88 ? "fausses alertes" : ""}</div>
        </div>
        <div className="mt-3 grid grid-cols-3 gap-3">
          <Stat label="Vraies fraudes (TP)" value={fmt(TP)} tone="emerald" />
          <Stat label="Fausses alertes (FP)" value={fmt(FP)} tone="amber" />
          <Stat label="Précision" value={`${share.toFixed(1)} %`} tone={share < 20 ? "rose" : "indigo"} />
        </div>
        <p className="mt-3 text-xs text-slate-500">Même ROC, même AUC (0.90) : seule la prévalence change, et la précision s'effondre.</p>
      </Card>
    </div>
  );
}

// ---------- 12. Alternative PR ----------
function SlidePRDef() {
  return (
    <div className="grid h-full grid-cols-1 gap-6 lg:grid-cols-2">
      <div className="flex flex-col gap-4">
        <Card>
          <div className="mb-2 flex items-center gap-2 font-bold text-slate-800"><Search size={18} className="text-indigo-600" /> Courbe Precision-Recall</div>
          <p className="text-sm text-slate-700">
            Trace la <b className="text-indigo-600">précision</b> en fonction du <b className="text-emerald-600">rappel</b> pour tous les seuils.
            Elle ignore totalement les vrais négatifs : elle se concentre sur la classe positive.
          </p>
        </Card>
        <Formula>
          Precision = TP / (TP + FP)<br />
          Recall    = TP / (TP + FN)
        </Formula>
        <Card>
          <div className="font-bold text-slate-800">AUC-PR / Average Precision</div>
          <p className="mt-1 text-sm text-slate-700">
            Aire sous la courbe PR. Sa <b>référence aléatoire n'est pas 0.5</b> mais la prévalence des positifs (ex : 0.01 pour 1 % de fraudes).
          </p>
        </Card>
      </div>
      <div className="grid grid-cols-1 gap-4">
        <Card className="border-slate-200">
          <div className="font-bold text-slate-800">ROC</div>
          <ul className="mt-2 space-y-1 text-sm text-slate-600">
            <li>• Utilise TN → sensible au grand nombre de négatifs</li>
            <li>• Baseline fixe : 0.5</li>
            <li>• Bonne vue d'ensemble sur classes équilibrées</li>
          </ul>
        </Card>
        <Card className="border-indigo-300 ring-2 ring-indigo-100">
          <div className="font-bold text-indigo-800">Precision-Recall</div>
          <ul className="mt-2 space-y-1 text-sm text-slate-700">
            <li>• N'utilise pas les TN → insensible à leur masse</li>
            <li>• Baseline = prévalence des positifs</li>
            <li>• <b>Recommandée</b> quand les positifs sont rares et que l'on s'intéresse à eux</li>
          </ul>
        </Card>
        <Card className="bg-emerald-50/70 text-sm text-emerald-900">
          <ShieldCheck size={18} className="mb-1 text-emerald-600" />
          Scikit-learn : <code className="rounded bg-white px-1">average_precision_score</code> et <code className="rounded bg-white px-1">precision_recall_curve</code>.
        </Card>
      </div>
    </div>
  );
}

// ---------- 13. GRAPHIQUE 3 : PR ----------
function SlidePRChart() {
  const scenarios = [
    { key: "p50", label: "Prévalence 50 %", p: 0.5, color: "#10b981" },
    { key: "p10", label: "Prévalence 10 %", p: 0.1, color: "#6366f1" },
    { key: "p1", label: "Prévalence 1 %", p: 0.01, color: "#f43f5e" },
  ];
  return (
    <div className="grid h-full grid-cols-1 gap-6 lg:grid-cols-3">
      <Card className="lg:col-span-2">
        <div className="mb-1 text-sm text-slate-600">Même classifieur (AUC-ROC = 0.90), trois niveaux de rareté des positifs.</div>
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={prData} margin={{ top: 10, right: 20, bottom: 25, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis type="number" dataKey="recall" domain={[0, 1]} tick={chartTick} label={{ value: "Recall", position: "insideBottom", offset: -12, fontSize: 12, fill: "#475569" }} />
              <YAxis type="number" domain={[0, 1]} tick={chartTick} label={{ value: "Precision", angle: -90, position: "insideLeft", fontSize: 12, fill: "#475569" }} />
              <Tooltip formatter={(v) => v.toFixed(3)} labelFormatter={(l) => `Recall = ${Number(l).toFixed(2)}`} />
              <Legend verticalAlign="top" />
              {scenarios.map((s) => (
                <Line key={s.key} type="monotone" dataKey={s.key} name={s.label} stroke={s.color} strokeWidth={3} dot={false} />
              ))}
              <ReferenceLine y={0.01} stroke="#f43f5e" strokeDasharray="5 5" label={{ value: "baseline 1 %", fontSize: 11, fill: "#f43f5e", position: "insideBottomRight" }} />
              <ReferenceLine x={OP.tpr} stroke="#94a3b8" strokeDasharray="3 3" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>
      <div className="flex flex-col gap-3">
        <Card className="p-4">
          <div className="mb-2 text-sm font-bold text-slate-800">À Recall = {Math.round(OP.tpr * 100)} % (FPR = {OP.fpr * 100} %)</div>
          <div className="space-y-2">
            {scenarios.map((s) => (
              <div key={s.key} className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2 text-sm">
                <span className="flex items-center gap-2 text-slate-600"><span className="h-3 w-3 rounded-full" style={{ background: s.color }} />{s.label}</span>
                <span className="font-bold text-slate-800">{(precisionAt(OP.tpr, OP.fpr, s.p) * 100).toFixed(0)} %</span>
              </div>
            ))}
          </div>
        </Card>
        <Card className="p-4">
          <div className="mb-2 text-sm font-bold text-slate-800">AUC-PR approximée</div>
          <div className="grid grid-cols-3 gap-2 text-center">
            {scenarios.map((s) => (
              <div key={s.key} className="rounded-lg bg-slate-50 p-2">
                <div className="text-lg font-black" style={{ color: s.color }}>{aucPR(s.key).toFixed(2)}</div>
                <div className="text-[10px] text-slate-500">{s.p * 100} %</div>
              </div>
            ))}
          </div>
        </Card>
        <Card className="bg-rose-50/70 p-4 text-sm text-rose-900">
          <b>La ROC est identique</b> dans les 3 cas ; la courbe PR révèle la chute de précision à mesure que les positifs se raréfient.
        </Card>
      </div>
    </div>
  );
}

// ---------- 14. Conclusion ----------
const CODE = `from imblearn.pipeline import Pipeline          # ⚠ imblearn, PAS sklearn.pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# 1) Split stratifié : le Test garde la distribution réelle
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 2) SMOTE DANS le pipeline : appliqué au train de chaque fold uniquement
pipe = Pipeline([
    ("smote", SMOTE(k_neighbors=5, random_state=42)),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42)),
])

# 3) Validation croisée sans leakage, avec une métrique adaptée
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="average_precision").mean())

# 4) Évaluation finale sur le Test intact
pipe.fit(X_tr, y_tr)
proba = pipe.predict_proba(X_te)[:, 1]
print("AUC-ROC:", roc_auc_score(y_te, proba), "| AUC-PR:", average_precision_score(y_te, proba))`;

function SlideConclusion() {
  const tips = [
    { i: <Lock size={18} />, t: "SMOTE uniquement sur le Train (et dans chaque fold)" },
    { i: <GitBranch size={18} />, t: "imblearn.pipeline.Pipeline pour éviter le leakage" },
    { i: <Scale size={18} />, t: "Split & CV stratifiés ; Test à distribution réelle" },
    { i: <Activity size={18} />, t: "ROC/AUC pour la séparation globale" },
    { i: <Zap size={18} />, t: "PR/AUC-PR quand les positifs sont très rares" },
    { i: <Gauge size={18} />, t: "Choisir le seuil selon le coût métier (FP vs FN)" },
  ];
  return (
    <div className="grid h-full grid-cols-1 gap-6 lg:grid-cols-5">
      <div className="flex flex-col gap-2.5 lg:col-span-2">
        {tips.map((x, i) => (
          <div key={i} className="flex items-center gap-3 rounded-xl border border-slate-200 bg-white/90 px-4 py-3 shadow-sm">
            <span className="rounded-lg bg-indigo-50 p-2 text-indigo-600">{x.i}</span>
            <span className="text-sm font-medium text-slate-700">{x.t}</span>
          </div>
        ))}
        <div className="rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 p-4 text-sm font-medium text-white shadow-lg">
          Rééquilibrer proprement + mesurer avec les bonnes métriques = un modèle réellement fiable en production.
        </div>
      </div>
      <div className="lg:col-span-3">
        <div className="mb-2 flex items-center gap-2 text-sm font-bold text-slate-800"><Code2 size={18} className="text-indigo-600" /> Code Scikit-learn / imbalanced-learn suggéré</div>
        <CodeBlock code={CODE} />
      </div>
    </div>
  );
}

/* ============================================================
   4. COMPOSANT PRINCIPAL
   ============================================================ */

const SLIDES = [
  { kicker: null, title: null, C: SlideTitle },
  { kicker: "Le problème", title: "Les limites de l'Accuracy face au déséquilibre", C: SlideAccuracy },
  { kicker: "La solution", title: "SMOTE : Synthetic Minority Over-sampling Technique", C: SlideSmoteIntro },
  { kicker: "Le mécanisme", title: "Interpolation par k plus proches voisins vs duplication", C: SlideMechanism },
  { kicker: "Graphique 1 · ScatterChart", title: "Visualisation avant / après SMOTE", C: SlideScatter },
  { kicker: "Règle d'or", title: "SMOTE uniquement sur le Train set : évitez le Data Leakage", C: SlideGoldenRule },
  { kicker: "Changer de métrique", title: "Les fondements : TP, FP, TPR et FPR", C: SlideFundamentals },
  { kicker: "Courbe ROC", title: "TPR en fonction du FPR, pour tous les seuils", C: SlideROCDef },
  { kicker: "AUC", title: "Aire sous la courbe : la capacité de séparation", C: SlideAUC },
  { kicker: "Graphique 2 · LineChart", title: "Courbe ROC : modèle vs hasard", C: SlideROCChart },
  { kicker: "Le piège", title: "Déséquilibre extrême : l'AUC-ROC trop optimiste", C: SlidePitfall },
  { kicker: "L'alternative", title: "La courbe Precision-Recall et l'AUC-PR", C: SlidePRDef },
  { kicker: "Graphique 3 · LineChart", title: "Precision-Recall : la chute que la ROC ne montre pas", C: SlidePRChart },
  { kicker: "Conclusion", title: "Bonnes pratiques & code de référence", C: SlideConclusion },
];

export default function Presentation() {
  const [idx, setIdx] = useState(0);
  const total = SLIDES.length;
  const go = useCallback((n) => setIdx((i) => Math.min(total - 1, Math.max(0, typeof n === "function" ? n(i) : n))), [total]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "ArrowRight") go((i) => i + 1);
      if (e.key === "ArrowLeft") go((i) => i - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go]);

  const { kicker, title, C } = SLIDES[idx];

  return (
    <div
      className="relative flex min-h-screen flex-col text-slate-900"
      style={{
        backgroundColor: "#f8fafc",
        backgroundImage:
          "linear-gradient(to right, rgba(148,163,184,0.15) 1px, transparent 1px), linear-gradient(to bottom, rgba(148,163,184,0.15) 1px, transparent 1px)",
        backgroundSize: "36px 36px",
      }}
    >
      <style>{`@keyframes slideIn { from { opacity: 0; transform: translateY(12px);} to { opacity: 1; transform: translateY(0);} }
      .slide-in { animation: slideIn .45s ease both; }`}</style>

      {/* Barre de progression */}
      <div className="h-1.5 w-full bg-slate-200">
        <div className="h-full bg-gradient-to-r from-indigo-500 to-rose-500 transition-all duration-500" style={{ width: `${((idx + 1) / total) * 100}%` }} />
      </div>

      {/* En-tête */}
      <header className="mx-auto flex w-full max-w-6xl items-start justify-between gap-4 px-6 pt-6">
        <div className="min-h-[64px]">
          {kicker && (
            <>
              <div className="text-xs font-bold uppercase tracking-widest text-indigo-600">{kicker}</div>
              <h2 className="mt-1 text-2xl font-extrabold tracking-tight text-slate-900 md:text-3xl">{title}</h2>
            </>
          )}
        </div>
        <div className="shrink-0 rounded-full border border-slate-200 bg-white px-3 py-1 text-sm font-semibold text-slate-500 shadow-sm">
          {idx + 1} / {total}
        </div>
      </header>

      {/* Contenu */}
      <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-4">
        <div key={idx} className="slide-in h-full">
          <C />
        </div>
      </main>

      {/* Navigation */}
      <footer className="mx-auto flex w-full max-w-6xl items-center justify-between gap-4 px-6 pb-6">
        <button
          onClick={() => go(idx - 1)}
          disabled={idx === 0}
          className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-5 py-2.5 text-sm font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
        >
          <ChevronLeft size={18} /> Précédent
        </button>

        <div className="hidden flex-wrap items-center justify-center gap-1.5 md:flex">
          {SLIDES.map((_, i) => (
            <button
              key={i}
              onClick={() => go(i)}
              aria-label={`Aller à la slide ${i + 1}`}
              className={`h-2.5 rounded-full transition-all ${i === idx ? "w-8 bg-indigo-600" : "w-2.5 bg-slate-300 hover:bg-slate-400"}`}
            />
          ))}
        </div>

        <button
          onClick={() => go(idx + 1)}
          disabled={idx === total - 1}
          className="inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow-md transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Suivant <ChevronRight size={18} />
        </button>
      </footer>
    </div>
  );
}