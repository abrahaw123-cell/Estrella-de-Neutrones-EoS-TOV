import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, AreaChart, Area, ScatterChart, Scatter } from "recharts";

// ─── Constants ────────────────────────────────────────────────────────────────
const HBAR = 1.0545718e-34;
const M_N  = 1.6749274e-27;
const H    = 6.62607015e-34;
const E_CH = 1.60217663e-19;
const G    = 6.6743e-11;
const C    = 2.998e8;
const MSUN = 1.989e30;
const K_FERMI = Math.pow(3 * Math.PI ** 2, 2 / 3) * (HBAR ** 2 / (5 * M_N));

// ─── EoS ──────────────────────────────────────────────────────────────────────
function unifiedEoS(rho, rho_prev, rho_next, dr, p) {
  const P_deg  = K_FERMI * Math.pow(rho, 5 / 3);
  const P_conf = p.V0 * (Math.pow(rho / p.rho_c, 2) - 1);
  let P_info = 0;
  if (rho > 0 && rho_prev > 0 && rho_next > 0) {
    const lap = (Math.log(rho_next) - 2 * Math.log(rho) + Math.log(rho_prev)) / dr ** 2;
    P_info = p.alpha * p.n * (H / (2 * E_CH)) * lap;
  }
  const total = Math.max(1e-10, P_deg + P_conf + P_info);
  const eps   = rho * C * C; // simple energy density
  return { total, eps, P_deg, P_conf: Math.abs(P_conf), P_info: Math.abs(P_info) };
}

// Build EoS table: array of {rho, P, eps} sorted low→high density
function buildEoSTable(params) {
  const { rho_min, rho_max, rho_c, V0, alpha, n, dr, n_points } = params;
  const dens = Array.from({ length: n_points }, (_, i) =>
    rho_min + (rho_max - rho_min) * (i / (n_points - 1))
  );
  const p = { rho_c, V0, alpha, n };
  return dens.slice(1, -1).map((rho, i) => {
    const res = unifiedEoS(rho, dens[i], dens[i + 2], dr, p);
    return { rho, P: res.total, eps: res.eps, P_deg: res.P_deg, P_conf: res.P_conf, P_info: res.P_info };
  });
}

// Interpolate pressure for a given density from EoS table
function interpP(table, rho) {
  if (rho <= table[0].rho) return table[0].P;
  if (rho >= table[table.length - 1].rho) return table[table.length - 1].P;
  for (let i = 0; i < table.length - 1; i++) {
    if (rho >= table[i].rho && rho <= table[i + 1].rho) {
      const t = (rho - table[i].rho) / (table[i + 1].rho - table[i].rho);
      return table[i].P + t * (table[i + 1].P - table[i].P);
    }
  }
  return table[table.length - 1].P;
}

// Interpolate density for a given pressure (inverse)
function interpRho(table, P) {
  if (P <= table[0].P) return table[0].rho;
  if (P >= table[table.length - 1].P) return table[table.length - 1].rho;
  for (let i = 0; i < table.length - 1; i++) {
    if (P >= table[i].P && P <= table[i + 1].P) {
      const t = (P - table[i].P) / (table[i + 1].P - table[i].P);
      return table[i].rho + t * (table[i + 1].rho - table[i].rho);
    }
  }
  return table[table.length - 1].rho;
}

// ─── TOV Solver ───────────────────────────────────────────────────────────────
// dP/dr = -G(eps + P)(m + 4πr³P/c²) / [r²c²(1 - 2Gm/rc²)]
// dm/dr = 4πr²eps/c²
function solveTOV(rho_c_tov, eosTable, dr_tov = 100) {
  const P_c = interpP(eosTable, rho_c_tov);
  let r = dr_tov;
  let m = (4 / 3) * Math.PI * r ** 3 * rho_c_tov; // small initial mass
  let P = P_c;

  const profile = [{ r: 0, m: 0, P: P_c, rho: rho_c_tov }];
  const P_surface = 1e26; // ~surface pressure threshold
  const r_max = 25e3; // 25 km max

  while (P > P_surface && r < r_max) {
    const eps = interpRho(eosTable, P) * C * C;
    const rho = interpRho(eosTable, P);

    const compactness = 2 * G * m / (r * C * C);
    if (compactness >= 0.99) break; // black hole

    const dP_dr = -(G / (C * C)) * (eps + P) * (m + 4 * Math.PI * r ** 3 * P / (C * C)) /
      (r ** 2 * (1 - compactness));
    const dm_dr = 4 * Math.PI * r ** 2 * eps / (C * C);

    P += dP_dr * dr_tov;
    m += dm_dr * dr_tov;
    r += dr_tov;

    if (P < 0) P = 0;
    profile.push({ r, m, P, rho });
  }

  const R_star = r / 1e3; // km
  const M_star = m / MSUN;
  return { R: R_star, M: M_star, profile };
}

// Build mass-radius curve sweeping central densities
function buildMRCurve(eosTable, rho_min, rho_max) {
  const nPoints = 30;
  const results = [];
  for (let i = 0; i < nPoints; i++) {
    const rho_c = rho_min + (rho_max - rho_min) * (i / (nPoints - 1));
    try {
      const { R, M } = solveTOV(rho_c, eosTable, 150);
      if (R > 0 && M > 0 && R < 30 && M < 5) {
        results.push({ R: parseFloat(R.toFixed(2)), M: parseFloat(M.toFixed(3)), rho_c: rho_c / 1e17 });
      }
    } catch (e) {}
  }
  return results;
}

// ─── EoS chart data ───────────────────────────────────────────────────────────
function buildChartData(eosTable) {
  return eosTable.map(d => ({
    rho: parseFloat((d.rho / 1e17).toFixed(3)),
    logP: parseFloat(Math.log10(d.P).toFixed(3)),
    logFermi: parseFloat(Math.log10(Math.max(1e-10, d.P_deg)).toFixed(3)),
    logConf:  parseFloat(Math.log10(Math.max(1e-10, d.P_conf)).toFixed(3)),
    logInfo:  parseFloat(Math.log10(Math.max(1e-10, d.P_info)).toFixed(3)),
  }));
}

// ─── Neutron Star Canvas ──────────────────────────────────────────────────────
function StarCrossSection({ params }) {
  const canvasRef = useRef(null);
  const animRef   = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, HC = canvas.height;
    const cx = W / 2, cy = HC / 2;
    const R  = Math.min(W, HC) * 0.40;

    const layers = [
      { frac: 0.18, c0: "#ff6b35", c1: "#ff3300", label: "Núcleo\nquark?" },
      { frac: 0.42, c0: "#7c3aed", c1: "#4c1d95", label: "Neutrones\nsuperfluidos" },
      { frac: 0.72, c0: "#1d4ed8", c1: "#1e3a8a", label: "Corteza\ninterna" },
      { frac: 0.90, c0: "#0f766e", c1: "#134e4a", label: "Corteza\nexterna" },
      { frac: 1.00, c0: "#155e75", c1: "#0c4a6e", label: "Atmósfera" },
    ];

    function draw(t) {
      ctx.clearRect(0, 0, W, HC);

      layers.slice().reverse().forEach(({ frac, c0, c1 }) => {
        const g = ctx.createRadialGradient(cx - R * 0.15, cy - R * 0.15, 0, cx, cy, R * frac);
        g.addColorStop(0, c0 + "ee");
        g.addColorStop(0.6, c1 + "cc");
        g.addColorStop(1, c1 + "88");
        ctx.beginPath();
        ctx.arc(cx, cy, R * frac, 0, Math.PI * 2);
        ctx.fillStyle = g;
        ctx.fill();
      });

      for (let i = 0; i < 4; i++) {
        const phase = (t * 0.0008 + i * 0.25) % 1;
        ctx.beginPath();
        ctx.arc(cx, cy, R * phase, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(100,220,255,${(1 - phase) * 0.3})`;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      const og = ctx.createRadialGradient(cx, cy, R * 0.95, cx, cy, R * 1.2);
      og.addColorStop(0, "rgba(100,200,255,0.22)");
      og.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = og;
      ctx.beginPath();
      ctx.arc(cx, cy, R * 1.2, 0, Math.PI * 2);
      ctx.fill();

      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "rgba(255,255,255,0.12)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx, cy - R * 1.05);
      ctx.lineTo(cx, cy + R * 1.05);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();

      layers.forEach(({ frac, label }, i) => {
        const midFrac = i === 0 ? frac / 2 : (layers[i - 1].frac + frac) / 2;
        const lx = cx + R * midFrac + 6;
        ctx.beginPath();
        ctx.moveTo(cx + R * midFrac, cy);
        ctx.lineTo(lx, cy);
        ctx.strokeStyle = "rgba(255,255,255,0.18)";
        ctx.lineWidth = 0.7;
        ctx.stroke();
        label.split("\n").forEach((line, li) => {
          ctx.fillStyle = "rgba(200,230,255,0.65)";
          ctx.font = "9px 'IBM Plex Mono', monospace";
          ctx.fillText(line, lx + 3, cy - 7 + li * 11);
        });
      });

      ctx.strokeStyle = "rgba(100,220,255,0.35)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx - R, cy);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(100,220,255,0.55)";
      ctx.font = "9px 'IBM Plex Mono', monospace";
      ctx.fillText("R ~ 10 km", cx - R * 0.72, cy - 6);

      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,200,100,0.9)";
      ctx.fill();
    }

    function animate(ts) {
      draw(ts);
      animRef.current = requestAnimationFrame(animate);
    }
    animRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animRef.current);
  }, [params]);

  return <canvas ref={canvasRef} width={310} height={310} style={{ borderRadius: 12, display: "block" }} />;
}

// ─── Slider ───────────────────────────────────────────────────────────────────
function Slider({ label, value, min, max, step, onChange, fmt, color = "#64dcff" }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ color: "#5a7a96", fontSize: 10, fontFamily: "monospace" }}>{label}</span>
        <span style={{ color, fontSize: 10, fontFamily: "monospace", background: `${color}15`, padding: "1px 6px", borderRadius: 4 }}>{fmt(value)}</span>
      </div>
      <div style={{ position: "relative", height: 18, display: "flex", alignItems: "center" }}>
        <div style={{ position: "absolute", left: 0, right: 0, height: 3, background: "rgba(255,255,255,0.06)", borderRadius: 2 }} />
        <div style={{ position: "absolute", left: 0, width: `${pct}%`, height: 3, background: `linear-gradient(90deg,${color}77,${color})`, borderRadius: 2 }} />
        <input type="range" min={min} max={max} step={step} value={value}
          onChange={e => onChange(parseFloat(e.target.value))}
          style={{ position: "absolute", inset: 0, opacity: 0, cursor: "pointer", width: "100%" }} />
        <div style={{ position: "absolute", left: `calc(${pct}% - 7px)`, width: 14, height: 14, borderRadius: "50%", background: color, boxShadow: `0 0 8px ${color}99`, pointerEvents: "none" }} />
      </div>
    </div>
  );
}

// ─── Custom tooltip ───────────────────────────────────────────────────────────
function ChartTip({ active, payload, label, unit = "×10¹⁷ kg/m³" }) {
  if (!active || !payload?.length) return null;
  const colors = { logP: "#e2e8f0", logFermi: "#fb923c", logConf: "#c084fc", logInfo: "#4ade80", M: "#64dcff", R: "#fb923c" };
  const names  = { logP: "P total", logFermi: "P Fermi", logConf: "|P conf|", logInfo: "|P info|", M: "M (M☉)", R: "R (km)" };
  return (
    <div style={{ background: "rgba(4,8,20,0.97)", border: "1px solid rgba(100,220,255,0.2)", borderRadius: 8, padding: "9px 12px", fontFamily: "monospace", fontSize: 10 }}>
      <div style={{ color: "#64dcff", marginBottom: 5 }}>{Number(label).toFixed(2)} {unit}</div>
      {payload.map(d => (
        <div key={d.dataKey} style={{ color: colors[d.dataKey] || "#aaa", marginBottom: 2 }}>
          {names[d.dataKey] || d.dataKey}: {typeof d.value === "number" ? d.value.toFixed(3) : d.value}
        </div>
      ))}
    </div>
  );
}

// ─── Main ─────────────────────────────────────────────────────────────────────
export default function App() {
  const [params, setParams] = useState({
    rho_c: 2.8, V0exp: 33, alpha: 0.10, n: 1,
    rho_min: 0.5, rho_max: 8.0, dr: 10, n_points: 120,
  });
  const [tab, setTab] = useState("eos");
  const [vis, setVis] = useState({ logP: true, logFermi: true, logConf: false, logInfo: false });
  const [computing, setComputing] = useState(false);
  const [results, setResults] = useState(null);

  const fullParams = useMemo(() => ({
    ...params,
    rho_c:   params.rho_c   * 1e17,
    V0:      Math.pow(10, params.V0exp),
    rho_min: params.rho_min * 1e17,
    rho_max: params.rho_max * 1e17,
  }), [params]);

  const compute = useCallback(() => {
    setComputing(true);
    setTimeout(() => {
      const eosTable = buildEoSTable(fullParams);
      const chartData = buildChartData(eosTable);

      // TOV: solve for selected central density
      const tov = solveTOV(fullParams.rho_c, eosTable, 120);

      // Mass-radius curve
      const mrCurve = buildMRCurve(eosTable, fullParams.rho_min * 1.2, fullParams.rho_max * 0.95);

      // Radial profile
      const radialProfile = tov.profile.filter((_, i) => i % 5 === 0).map(d => ({
        r: parseFloat((d.r / 1e3).toFixed(2)),
        logP: parseFloat(Math.log10(Math.max(1e-10, d.P)).toFixed(3)),
        rho: parseFloat((d.rho / 1e17).toFixed(3)),
      }));

      setResults({ eosTable, chartData, tov, mrCurve, radialProfile });
      setComputing(false);
    }, 60);
  }, [fullParams]);

  useEffect(() => { compute(); }, []);

  const set = k => v => setParams(p => ({ ...p, [k]: v }));

  const LINE_CFG = [
    { key: "logP",     label: "P Total",  color: "#e2e8f0", w: 2.5 },
    { key: "logFermi", label: "P Fermi",  color: "#fb923c", w: 1.8 },
    { key: "logConf",  label: "|P conf|", color: "#c084fc", w: 1.8 },
    { key: "logInfo",  label: "|P info|", color: "#4ade80", w: 1.8 },
  ];

  const tov = results?.tov;

  return (
    <div style={{ minHeight: "100vh", background: "#03080f", color: "#c8dff0", fontFamily: "'IBM Plex Mono', monospace", overflow: "hidden" }}>
      {/* starfield */}
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none", background: "radial-gradient(ellipse at 70% 20%, rgba(30,60,120,0.22) 0%, transparent 60%), radial-gradient(ellipse at 20% 80%, rgba(80,20,140,0.12) 0%, transparent 50%)" }} />
      {[...Array(35)].map((_, i) => (
        <div key={i} style={{ position: "fixed", left: `${(i * 37 + 13) % 100}%`, top: `${(i * 61 + 7) % 100}%`, width: i % 5 === 0 ? 2 : 1, height: i % 5 === 0 ? 2 : 1, borderRadius: "50%", background: "white", opacity: 0.08 + (i % 4) * 0.06, pointerEvents: "none" }} />
      ))}

      {/* header */}
      <div style={{ borderBottom: "1px solid rgba(100,220,255,0.07)", padding: "16px 24px", display: "flex", alignItems: "center", justifyContent: "space-between", background: "rgba(3,8,15,0.75)", backdropFilter: "blur(12px)", position: "sticky", top: 0, zIndex: 10 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
            <div style={{ width: 26, height: 26, borderRadius: "50%", background: "radial-gradient(circle at 35% 35%, #fb923c, #7c3aed 60%, #1d4ed8)", boxShadow: "0 0 14px rgba(251,146,60,0.35)", flexShrink: 0 }} />
            <span style={{ fontSize: 14, fontWeight: 700, fontFamily: "'Space Mono', monospace", background: "linear-gradient(135deg,#fff 40%,#64dcff)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
              Estrella de Neutrones · EoS + TOV
            </span>
          </div>
          <div style={{ color: "#1e3a55", fontSize: 9, letterSpacing: "0.2em", marginTop: 2 }}>
            ECUACIÓN DE ESTADO UNIFICADA · INTEGRACIÓN TOLMAN–OPPENHEIMER–VOLKOFF
          </div>
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          {[
            { id: "eos",  label: "EoS" },
            { id: "tov",  label: "TOV" },
            { id: "mr",   label: "M–R" },
            { id: "star", label: "Estrella" },
            { id: "info", label: "Física" },
          ].map(({ id, label }) => (
            <button key={id} onClick={() => setTab(id)} style={{
              padding: "5px 13px", borderRadius: 20, fontSize: 10, letterSpacing: "0.08em",
              border: tab === id ? "1px solid rgba(100,220,255,0.45)" : "1px solid rgba(100,220,255,0.08)",
              background: tab === id ? "rgba(100,220,255,0.1)" : "transparent",
              color: tab === id ? "#64dcff" : "#2a4a6a", cursor: "pointer",
              fontFamily: "'IBM Plex Mono', monospace", transition: "all 0.15s",
            }}>{label}</button>
          ))}
        </div>
      </div>

      <div style={{ display: "flex", height: "calc(100vh - 65px)" }}>
        {/* sidebar */}
        <div style={{ width: 252, flexShrink: 0, borderRight: "1px solid rgba(100,220,255,0.06)", padding: "18px 16px", overflowY: "auto", background: "rgba(3,8,20,0.5)", backdropFilter: "blur(8px)" }}>
          <div style={{ fontSize: 9, color: "#1e3a55", letterSpacing: "0.25em", marginBottom: 16 }}>PARÁMETROS</div>

          <div style={{ fontSize: 9, color: "#fb923c", letterSpacing: "0.15em", marginBottom: 8, opacity: 0.8 }}>CONFINAMIENTO</div>
          <Slider label="ρ_c densidad crítica" value={params.rho_c} min={0.5} max={8} step={0.1}
            onChange={set("rho_c")} fmt={v => `${v.toFixed(1)}×10¹⁷`} color="#fb923c" />
          <Slider label="V₀ = 10^x Pa" value={params.V0exp} min={30} max={36} step={0.25}
            onChange={set("V0exp")} fmt={v => `10^${v.toFixed(2)}`} color="#c084fc" />

          <div style={{ fontSize: 9, color: "#4ade80", letterSpacing: "0.15em", margin: "14px 0 8px", opacity: 0.8 }}>CUANTIZACIÓN TOPOLÓGICA</div>
          <Slider label="α acoplamiento" value={params.alpha} min={0} max={1} step={0.01}
            onChange={set("alpha")} fmt={v => v.toFixed(2)} color="#4ade80" />
          <Slider label="n cuanta n·h/2e" value={params.n} min={1} max={6} step={1}
            onChange={set("n")} fmt={v => `n=${v}`} color="#4ade80" />

          <div style={{ fontSize: 9, color: "#64dcff", letterSpacing: "0.15em", margin: "14px 0 8px", opacity: 0.8 }}>PERFIL DE DENSIDAD</div>
          <Slider label="ρ_min superficie" value={params.rho_min} min={0.1} max={3} step={0.1}
            onChange={set("rho_min")} fmt={v => `${v.toFixed(1)}×10¹⁷`} color="#64dcff" />
          <Slider label="ρ_max centro" value={params.rho_max} min={2} max={12} step={0.1}
            onChange={set("rho_max")} fmt={v => `${v.toFixed(1)}×10¹⁷`} color="#64dcff" />
          <Slider label="Δr paso espacial m" value={params.dr} min={1} max={100} step={1}
            onChange={set("dr")} fmt={v => `${v} m`} color="#64dcff" />

          <button onClick={compute} disabled={computing} style={{
            width: "100%", marginTop: 14, padding: "9px",
            border: "1px solid rgba(100,220,255,0.28)", borderRadius: 8,
            cursor: computing ? "not-allowed" : "pointer",
            background: computing ? "transparent" : "rgba(100,220,255,0.07)",
            color: computing ? "#1a3a55" : "#64dcff",
            fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, letterSpacing: "0.12em", transition: "all 0.2s",
          }}>
            {computing ? "···" : "▶ SIMULAR"}
          </button>

          {/* TOV results */}
          {tov && (
            <div style={{ marginTop: 16, padding: "13px", background: "rgba(100,220,255,0.025)", border: "1px solid rgba(100,220,255,0.07)", borderRadius: 8 }}>
              <div style={{ fontSize: 9, color: "#1e3a55", letterSpacing: "0.2em", marginBottom: 10 }}>RESULTADO TOV</div>
              {[
                ["Radio R★",  `${tov.R.toFixed(2)} km`, "#64dcff"],
                ["Masa M★",   `${tov.M.toFixed(3)} M☉`, "#fb923c"],
                ["Compacidad", `${(tov.M * MSUN * 2 * G / (tov.R * 1e3 * C * C)).toFixed(3)}`, "#c084fc"],
                ["Capas TOV",  `${tov.profile.length}`, "#4ade80"],
              ].map(([k, v, c]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                  <span style={{ color: "#2a4a6a", fontSize: 10 }}>{k}</span>
                  <span style={{ color: c, fontSize: 10 }}>{v}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* main */}
        <div style={{ flex: 1, overflowY: "auto", padding: "18px 22px" }}>

          {/* ── EoS tab ── */}
          {tab === "eos" && results && (
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <div style={{ display: "flex", gap: 7, flexWrap: "wrap" }}>
                {LINE_CFG.map(({ key, label, color }) => (
                  <button key={key} onClick={() => setVis(v => ({ ...v, [key]: !v[key] }))} style={{
                    padding: "4px 12px", borderRadius: 20, fontSize: 9, letterSpacing: "0.07em",
                    border: `1px solid ${vis[key] ? color : "rgba(255,255,255,0.05)"}`,
                    background: vis[key] ? `${color}13` : "transparent",
                    color: vis[key] ? color : "#2a4a6a", cursor: "pointer",
                    fontFamily: "'IBM Plex Mono', monospace", transition: "all 0.12s",
                  }}>{label}</button>
                ))}
              </div>

              <div style={{ background: "rgba(255,255,255,0.016)", border: "1px solid rgba(100,220,255,0.07)", borderRadius: 11, padding: "16px 10px 8px" }}>
                <div style={{ fontSize: 9, color: "#1e3a55", letterSpacing: "0.2em", marginBottom: 12, paddingLeft: 6 }}>log₁₀(P) vs. DENSIDAD</div>
                <ResponsiveContainer width="100%" height={270}>
                  <LineChart data={results.chartData} margin={{ top: 4, right: 18, bottom: 22, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 8" stroke="rgba(100,200,255,0.045)" />
                    <XAxis dataKey="rho" tick={{ fill: "#2a4a6a", fontSize: 9 }} tickLine={false} axisLine={{ stroke: "rgba(100,220,255,0.08)" }}
                      label={{ value: "ρ (×10¹⁷ kg/m³)", position: "insideBottom", offset: -13, fill: "#2a4a6a", fontSize: 9 }} />
                    <YAxis tick={{ fill: "#2a4a6a", fontSize: 9 }} tickLine={false} axisLine={{ stroke: "rgba(100,220,255,0.08)" }}
                      label={{ value: "log₁₀(P) [Pa]", angle: -90, position: "insideLeft", fill: "#2a4a6a", fontSize: 9, dy: 48 }} />
                    <Tooltip content={<ChartTip />} />
                    <ReferenceLine x={params.rho_c.toFixed(1)} stroke="rgba(251,146,60,0.28)" strokeDasharray="5 4"
                      label={{ value: "ρ_c", fill: "#fb923c", fontSize: 8, position: "top" }} />
                    {LINE_CFG.map(({ key, label, color, w }) =>
                      vis[key] ? <Line key={key} type="monotone" dataKey={key} name={label} stroke={color} strokeWidth={w} dot={false} strokeOpacity={0.9} animationDuration={400} /> : null
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div style={{ background: "rgba(255,255,255,0.016)", border: "1px solid rgba(100,220,255,0.07)", borderRadius: 11, padding: "16px 10px 8px" }}>
                <div style={{ fontSize: 9, color: "#1e3a55", letterSpacing: "0.2em", marginBottom: 12, paddingLeft: 6 }}>CONTRIBUCIONES RELATIVAS</div>
                <ResponsiveContainer width="100%" height={150}>
                  <AreaChart data={results.chartData} margin={{ top: 4, right: 18, bottom: 18, left: 8 }}>
                    <defs>
                      {[["fermi","#fb923c"],["conf","#c084fc"],["info","#4ade80"]].map(([id,c]) => (
                        <linearGradient key={id} id={`g_${id}`} x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor={c} stopOpacity={0.38} />
                          <stop offset="95%" stopColor={c} stopOpacity={0.04} />
                        </linearGradient>
                      ))}
                    </defs>
                    <CartesianGrid strokeDasharray="3 8" stroke="rgba(100,200,255,0.035)" />
                    <XAxis dataKey="rho" tick={{ fill: "#2a4a6a", fontSize: 9 }} tickLine={false} axisLine={{ stroke: "rgba(100,220,255,0.08)" }} />
                    <YAxis tick={{ fill: "#2a4a6a", fontSize: 9 }} tickLine={false} axisLine={{ stroke: "rgba(100,220,255,0.08)" }} />
                    <Tooltip content={<ChartTip />} />
                    <Area type="monotone" dataKey="logFermi" stroke="#fb923c" strokeWidth={1.5} fill="url(#g_fermi)" dot={false} animationDuration={400} />
                    <Area type="monotone" dataKey="logConf"  stroke="#c084fc" strokeWidth={1.5} fill="url(#g_conf)"  dot={false} animationDuration={400} />
                    <Area type="monotone" dataKey="logInfo"  stroke="#4ade80" strokeWidth={1.5} fill="url(#g_info)"  dot={false} animationDuration={400} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* ── TOV profile tab ── */}
          {tab === "tov" && results && (
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10 }}>
                {[
                  ["Radio R★", tov ? `${tov.R.toFixed(2)} km` : "—", "#64dcff"],
                  ["Masa M★",  tov ? `${tov.M.toFixed(3)} M☉` : "—", "#fb923c"],
                  ["ρ central",`${params.rho_max.toFixed(1)}×10¹⁷`, "#c084fc"],
                  ["Capas",    tov ? `${tov.profile.length}` : "—", "#4ade80"],
                ].map(([k, v, c]) => (
                  <div key={k} style={{ background: "rgba(100,220,255,0.025)", border: "1px solid rgba(100,220,255,0.06)", borderRadius: 8, padding: "10px 12px" }}>
                    <div style={{ color: "#2a4a6a", fontSize: 9, marginBottom: 4 }}>{k}</div>
                    <div style={{ color: c, fontSize: 13 }}>{v}</div>
                  </div>
                ))}
              </div>

              <div style={{ background: "rgba(255,255,255,0.016)", border: "1px solid rgba(100,220,255,0.07)", borderRadius: 11, padding: "16px 10px 8px" }}>
                <div style={{ fontSize: 9, color: "#1e3a55", letterSpacing: "0.2em", marginBottom: 12, paddingLeft: 6 }}>PERFIL RADIAL — PRESIÓN vs. RADIO</div>
                <ResponsiveContainer width="100%" height={260}>
                  <AreaChart data={results.radialProfile} margin={{ top: 4, right: 18, bottom: 22, left: 8 }}>
                    <defs>
                      <linearGradient id="gRad" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stopColor="#fb923c" stopOpacity={0.5} />
                        <stop offset="100%" stopColor="#64dcff" stopOpacity={0.05} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 8" stroke="rgba(100,200,255,0.045)" />
                    <XAxis dataKey="r" tick={{ fill: "#2a4a6a", fontSize: 9 }} tickLine={false} axisLine={{ stroke: "rgba(100,220,255,0.08)" }}
                      label={{ value: "radio r (km)", position: "insideBottom", offset: -13, fill: "#2a4a6a", fontSize: 9 }} />
                    <YAxis tick={{ fill: "#2a4a6a", fontSize: 9 }} tickLine={false} axisLine={{ stroke: "rgba(100,220,255,0.08)" }}
                      label={{ value: "log₁₀(P) [Pa]", angle: -90, position: "insideLeft", fill: "#2a4a6a", fontSize: 9, dy: 48 }} />
                    <Tooltip content={<ChartTip unit="km" />} />
                    <Area type="monotone" dataKey="logP" stroke="#e2e8f0" strokeWidth={2} fill="url(#gRad)" dot={false} animationDuration={400} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              <div style={{ background: "rgba(255,255,255,0.016)", border: "1px solid rgba(100,220,255,0.07)", borderRadius: 11, padding: "16px 10px 8px" }}>
                <div style={{ fontSize: 9, color: "#1e3a55", letterSpacing: "0.2em", marginBottom: 12, paddingLeft: 6 }}>PERFIL RADIAL — DENSIDAD vs. RADIO</div>
                <ResponsiveContainer width="100%" height={180}>
                  <AreaChart data={results.radialProfile} margin={{ top: 4, right: 18, bottom: 22, left: 8 }}>
                    <defs>
                      <linearGradient id="gRho" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stopColor="#c084fc" stopOpacity={0.5} />
                        <stop offset="100%" stopColor="#c084fc" stopOpacity={0.04} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 8" stroke="rgba(100,200,255,0.035)" />
                    <XAxis dataKey="r" tick={{ fill: "#2a4a6a", fontSize: 9 }} tickLine={false} axisLine={{ stroke: "rgba(100,220,255,0.08)" }}
                      label={{ value: "radio r (km)", position: "insideBottom", offset: -13, fill: "#2a4a6a", fontSize: 9 }} />
                    <YAxis tick={{ fill: "#2a4a6a", fontSize: 9 }} tickLine={false} axisLine={{ stroke: "rgba(100,220,255,0.08)" }} />
                    <Tooltip content={<ChartTip unit="km" />} />
                    <Area type="monotone" dataKey="rho" stroke="#c084fc" strokeWidth={1.8} fill="url(#gRho)" dot={false} animationDuration={400} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* ── M-R tab ── */}
          {tab === "mr" && results && (
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <div style={{ background: "rgba(255,255,255,0.016)", border: "1px solid rgba(100,220,255,0.07)", borderRadius: 11, padding: "16px 10px 8px" }}>
                <div style={{ fontSize: 9, color: "#1e3a55", letterSpacing: "0.2em", marginBottom: 6, paddingLeft: 6 }}>CURVA MASA–RADIO (M–R)</div>
                <div style={{ fontSize: 9, color: "#1e3a55", paddingLeft: 6, marginBottom: 12 }}>
                  cada punto = densidad central distinta ({results.mrCurve.length} soluciones TOV)
                </div>
                <ResponsiveContainer width="100%" height={340}>
                  <LineChart data={results.mrCurve} margin={{ top: 10, right: 22, bottom: 24, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 8" stroke="rgba(100,200,255,0.045)" />
                    <XAxis dataKey="R" type="number" domain={["auto","auto"]} tick={{ fill: "#2a4a6a", fontSize: 9 }} tickLine={false} axisLine={{ stroke: "rgba(100,220,255,0.08)" }}
                      label={{ value: "radio R★ (km)", position: "insideBottom", offset: -13, fill: "#2a4a6a", fontSize: 9 }} />
                    <YAxis dataKey="M" type="number" domain={["auto","auto"]} tick={{ fill: "#2a4a6a", fontSize: 9 }} tickLine={false} axisLine={{ stroke: "rgba(100,220,255,0.08)" }}
                      label={{ value: "masa M★ (M☉)", angle: -90, position: "insideLeft", fill: "#2a4a6a", fontSize: 9, dy: 52 }} />
                    <Tooltip content={<ChartTip unit="km" />} />
                    {/* observational constraint bands */}
                    <ReferenceLine y={2.0}  stroke="rgba(251,146,60,0.4)" strokeDasharray="5 3" label={{ value: "PSR J0740 ~2.1 M☉", fill: "#fb923c", fontSize: 8 }} />
                    <ReferenceLine y={1.4}  stroke="rgba(100,220,255,0.3)" strokeDasharray="5 3" label={{ value: "1.4 M☉ típica", fill: "#64dcff",  fontSize: 8 }} />
                    <Line type="monotone" dataKey="M" stroke="#64dcff" strokeWidth={2.2} dot={{ fill: "#64dcff", r: 3 }} animationDuration={500} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div style={{ background: "rgba(255,255,255,0.016)", border: "1px solid rgba(100,220,255,0.07)", borderRadius: 11, padding: "16px 10px 8px" }}>
                <div style={{ fontSize: 9, color: "#1e3a55", letterSpacing: "0.2em", marginBottom: 12, paddingLeft: 6 }}>MASA vs. DENSIDAD CENTRAL</div>
                <ResponsiveContainer width="100%" height={160}>
                  <LineChart data={results.mrCurve} margin={{ top: 4, right: 22, bottom: 22, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 8" stroke="rgba(100,200,255,0.035)" />
                    <XAxis dataKey="rho_c" tick={{ fill: "#2a4a6a", fontSize: 9 }} tickLine={false} axisLine={{ stroke: "rgba(100,220,255,0.08)" }}
                      label={{ value: "ρ_c (×10¹⁷ kg/m³)", position: "insideBottom", offset: -13, fill: "#2a4a6a", fontSize: 9 }} />
                    <YAxis tick={{ fill: "#2a4a6a", fontSize: 9 }} tickLine={false} axisLine={{ stroke: "rgba(100,220,255,0.08)" }} />
                    <Tooltip content={<ChartTip />} />
                    <Line type="monotone" dataKey="M" stroke="#c084fc" strokeWidth={1.8} dot={false} animationDuration={400} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* ── Star tab ── */}
          {tab === "star" && (
            <div style={{ display: "flex", gap: 22, flexWrap: "wrap", alignItems: "flex-start" }}>
              <div>
                <div style={{ fontSize: 9, color: "#1e3a55", letterSpacing: "0.2em", marginBottom: 12 }}>SECCIÓN TRANSVERSAL</div>
                <StarCrossSection params={params} />
                <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 5 }}>
                  {[
                    ["#ff6b35","Núcleo (quark-gluon?)"],
                    ["#7c3aed","Neutrones superfluidos"],
                    ["#1d4ed8","Corteza interna (cristalina)"],
                    ["#0f766e","Corteza externa"],
                    ["#155e75","Atmósfera (~ mm)"],
                  ].map(([c,l]) => (
                    <div key={l} style={{ display: "flex", alignItems: "center", gap: 7 }}>
                      <div style={{ width: 9, height: 9, borderRadius: 2, background: c, flexShrink: 0 }} />
                      <span style={{ fontSize: 10, color: "#3a5a78" }}>{l}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div style={{ flex: 1, minWidth: 220 }}>
                <div style={{ fontSize: 9, color: "#1e3a55", letterSpacing: "0.2em", marginBottom: 12 }}>PARÁMETROS FÍSICOS</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 9 }}>
                  {[
                    ["Masa",       tov ? `${tov.M.toFixed(3)} M☉` : "—",     "#fb923c"],
                    ["Radio",      tov ? `${tov.R.toFixed(2)} km` : "—",      "#64dcff"],
                    ["ρ central",  `${params.rho_max.toFixed(1)}×10¹⁷`,       "#c084fc"],
                    ["ρ_c confin.",`${params.rho_c.toFixed(1)}×10¹⁷`,         "#fb923c"],
                    ["Compacidad", tov ? (tov.M * MSUN * 2 * G / (tov.R * 1e3 * C * C)).toFixed(3) : "—", "#4ade80"],
                    ["V₀",        `10^${params.V0exp.toFixed(1)} Pa`,          "#c084fc"],
                  ].map(([k,v,c]) => (
                    <div key={k} style={{ background: "rgba(100,220,255,0.025)", border: "1px solid rgba(100,220,255,0.06)", borderRadius: 8, padding: "9px 11px" }}>
                      <div style={{ color: "#2a4a6a", fontSize: 9, marginBottom: 3 }}>{k}</div>
                      <div style={{ color: c, fontSize: 12 }}>{v}</div>
                    </div>
                  ))}
                </div>

                {results?.radialProfile && (
                  <div style={{ marginTop: 14, background: "rgba(255,255,255,0.016)", border: "1px solid rgba(100,220,255,0.07)", borderRadius: 11, padding: "14px 10px 8px" }}>
                    <div style={{ fontSize: 9, color: "#1e3a55", letterSpacing: "0.2em", marginBottom: 10, paddingLeft: 4 }}>PRESIÓN RADIAL</div>
                    <ResponsiveContainer width="100%" height={170}>
                      <AreaChart data={results.radialProfile} margin={{ top: 4, right: 12, bottom: 20, left: 6 }}>
                        <defs>
                          <linearGradient id="gStarP" x1="0" y1="0" x2="1" y2="0">
                            <stop offset="0%" stopColor="#fb923c" stopOpacity={0.5} />
                            <stop offset="100%" stopColor="#64dcff" stopOpacity={0.05} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 8" stroke="rgba(100,200,255,0.04)" />
                        <XAxis dataKey="r" tick={{ fill: "#2a4a6a", fontSize: 8 }} tickLine={false} axisLine={{ stroke: "rgba(100,220,255,0.08)" }}
                          label={{ value: "r (km)", position: "insideBottom", offset: -12, fill: "#2a4a6a", fontSize: 8 }} />
                        <YAxis tick={{ fill: "#2a4a6a", fontSize: 8 }} tickLine={false} axisLine={{ stroke: "rgba(100,220,255,0.08)" }} />
                        <Tooltip content={<ChartTip unit="km" />} />
                        <Area type="monotone" dataKey="logP" stroke="#e2e8f0" strokeWidth={1.8} fill="url(#gStarP)" dot={false} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── Info tab ── */}
          {tab === "info" && (
            <div style={{ maxWidth: 660, display: "flex", flexDirection: "column", gap: 12 }}>
              {[
                {
                  color: "#fb923c", title: "Presión de Degeneración de Fermi",
                  formula: "P_Fermi = K · ρ^(5/3)",
                  detail: "Gas de fermiones no relativistas. K = (3π²)^(2/3) · ℏ²/(5m_n). Es el soporte principal contra el colapso. Escala fuertemente con densidad.",
                },
                {
                  color: "#c084fc", title: "Potencial de Confinamiento",
                  formula: "P_conf = V₀ · [(ρ/ρ_c)² − 1]",
                  detail: "Cero exacto en ρ = ρ_c. Debajo: presión negativa (compresión). Arriba: positiva. Modela la naturaleza ligada de la materia nuclear.",
                },
                {
                  color: "#4ade80", title: "Corrección Cuántica Topológica",
                  formula: "P_info = α · (n·h/2e) · ∇²ln(ρ)",
                  detail: "Término holográfico. Acopla geometría del campo de densidad con cuanta de flujo magnético n·h/2e. Análogo a correcciones de Bohm.",
                },
                {
                  color: "#64dcff", title: "Ecuaciones TOV (Tolman–Oppenheimer–Volkoff)",
                  formula: "dP/dr = -G(ε+P)(m+4πr³P/c²) / [r²c²(1-2Gm/rc²)]\ndm/dr = 4πr²ε/c²",
                  detail: "Las ecuaciones de estructura estelar en relatividad general. Integradas numéricamente desde r→0 hasta P→0. Dan radio y masa reales. La curva M–R se construye barriendo densidades centrales.",
                },
              ].map(({ color, title, formula, detail }) => (
                <div key={title} style={{ background: "rgba(255,255,255,0.018)", border: `1px solid ${color}1a`, borderLeft: `3px solid ${color}`, borderRadius: 10, padding: "14px 16px" }}>
                  <div style={{ color, fontSize: 11, fontWeight: 600, marginBottom: 7 }}>{title}</div>
                  <div style={{ fontFamily: "monospace", fontSize: 10, color: "#5a8aaa", background: "rgba(0,0,0,0.22)", padding: "6px 10px", borderRadius: 5, marginBottom: 8, lineHeight: 1.8 }}>{formula}</div>
                  <p style={{ color: "#2e5070", fontSize: 10, margin: 0, lineHeight: 1.75 }}>{detail}</p>
                </div>
              ))}

              <div style={{ background: "rgba(100,220,255,0.03)", border: "1px solid rgba(100,220,255,0.08)", borderRadius: 10, padding: "13px 16px" }}>
                <div style={{ fontSize: 9, color: "#1e3a55", letterSpacing: "0.2em", marginBottom: 7 }}>LIMITACIONES DEL MODELO</div>
                <div style={{ color: "#2a4a6a", fontSize: 10, lineHeight: 1.85 }}>
                  El perfil de densidad es lineal (no autogravitante). La EoS usa gas de Fermi no relativista. Para densidades &gt; 5×10¹⁷ kg/m³ se necesitaría la EoS relativista (Baym–Pethick–Sutherland) y posiblemente quarks libres. El término P_info es fenomenológico. Los próximos pasos físicos son: EoS relativista, transiciones de fase, y comparación con datos NICER/GW170817.
                </div>
              </div>
            </div>
          )}

        </div>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Space+Mono:wght@700&display=swap');
        * { box-sizing: border-box; margin: 0; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(100,220,255,0.12); border-radius: 2px; }
      `}</style>
    </div>
  );
}
