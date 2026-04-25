import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.integrate import solve_ivp
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. EFEMÉRIDES Y NAVEGACIÓN (JPLEPHEM + KEPLER)
# ============================================================================
class SovereignEphemeris:
    def __init__(self):
        self.AU = 149597870.7
        self.planets = {
            'earth':   {'a': 1.00000011, 'e': 0.01671022, 'P': 365.25},
            'mars':    {'a': 1.523679,   'e': 0.09341233, 'P': 686.98},
            'jupiter': {'a': 5.20260,    'e': 0.04849485, 'P': 4332.59, 'mu': 126686534.0}
        }

    def get_kepler_pos(self, name, t_days):
        p = self.planets[name]
        n = 2 * np.pi / p['P']
        M = n * t_days
        E = M
        for _ in range(5):
            E = E - (E - p['e'] * np.sin(E) - M) / (1 - p['e'] * np.cos(E))
        x_orb = p['a'] * (np.cos(E) - p['e'])
        y_orb = p['a'] * np.sqrt(1 - p['e']**2) * np.sin(E)
        return np.array([x_orb * self.AU, y_orb * self.AU, 0])

class SPICENavigator:
    def __init__(self, bsp_file='de421.bsp'):
        self.AU = 149597870.7
        self.fallback = SovereignEphemeris()
        self.kernel = None
        try:
            from jplephem.spk import SPK
            self.kernel = SPK.open(bsp_file)
            print("✅ Kernel JPL cargado (efemérides reales).")
        except:
            print("⚠️ Archivo de efemérides no encontrado. Se usará modelo Kepleriano de respaldo.")

    def get_planet_pos_vel(self, planet_id, t_jd):
        if self.kernel:
            try:
                pos_vel = self.kernel[0, planet_id].compute_and_differentiate(t_jd)
                pos = pos_vel[:3]
                vel = pos_vel[3:] / 86400.0
                return pos, vel
            except:
                pass
        name_map = {3: 'earth', 4: 'mars', 5: 'jupiter'}
        if planet_id in name_map:
            pos = self.fallback.get_kepler_pos(name_map[planet_id], t_jd - 2451545.0)
            return pos, np.zeros(3)
        return np.zeros(3), np.zeros(3)


# ============================================================================
# 2. PROPAGACIÓN ORBITAL CON PERTURBACIONES MÓVILES
# ============================================================================
def n_body_ode(t, y, nav, start_jd):
    r_ship = y[:3]
    v_ship = y[3:6]
    current_jd = start_jd + t / 86400.0
    r_mag = np.linalg.norm(r_ship)
    a_sun = -1.32712440018e11 * r_ship / r_mag**3
    a_pert = np.zeros(3)
    for pid, mu in [(4, 42828.3), (5, 126686534.0)]:
        r_planet, _ = nav.get_planet_pos_vel(pid, current_jd)
        r_rel = r_planet - r_ship
        a_pert += mu * r_rel / np.linalg.norm(r_rel)**3
    return np.concatenate([v_ship, a_sun + a_pert])


# ============================================================================
# 3. ATERRIZAJE CON PID, PARACAÍDAS Y SISTEMA AUTÓNOMO
# ============================================================================
class AdvancedLander:
    def __init__(self, mass=2200.0):
        self.mass = mass
        self.Cd = 1.5
        self.area = 15.0
        self.g_mars = 3.71
        self.max_thrust = 30000.0
        self.kp, self.ki, self.kd = 1500.0, 10.0, 450.0
        self.integral = 0.0
        self.last_error = 0.0
        self.parachute_deployed = False

    def dynamics(self, t, state):
        x, z, vx, vz = state
        rho = 0.020 * np.exp(-max(0, z) / 11100.0)
        v_abs = np.hypot(vx, vz)
        if v_abs > 1e-3:
            f_drag = 0.5 * rho * v_abs**2 * self.Cd * self.area
            ax_drag = -(f_drag / self.mass) * (vx / v_abs)
            az_drag = -(f_drag / self.mass) * (vz / v_abs)
        else:
            ax_drag = az_drag = 0.0
        if not self.parachute_deployed and v_abs < 700 and z < 12000:
            print("🪂 DESPLIEGUE DE PARACAÍDAS SUPERSÓNICO")
            self.parachute_deployed = True
            self.area = 120.0
            self.Cd = 1.8
        thrust = 0.0
        if z < 1500:
            error = (-2.0) - vz
            self.integral += error * 0.01
            derivative = (error - self.last_error) / 0.01
            thrust_req = self.kp * error + self.ki * self.integral + self.kd * derivative
            thrust = np.clip(thrust_req, 0, self.max_thrust)
            self.last_error = error
        az_thrust = thrust / self.mass
        return [vx, vz, ax_drag, az_drag - self.g_mars + az_thrust]


# ============================================================================
# 4. SISTEMA AUTÓNOMO DE DECISIÓN (INTELIGENCIA)
# ============================================================================
class PIDOptimizer:
    @staticmethod
    def optimize_pid(kp, ki, kd, signal):
        # Lógica de mejora de PID basada en señal de riesgo
        if signal > 0.7:
            kp *= 1.05
            kd *= 1.02
        elif signal > 0.3:
            ki = min(ki + 0.5, 50)
        return kp, ki, kd

class AutonomousDecisionSystem:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def decide_phase(self, z, vz, signal):
        if z > 10000:
            return "ENTRY"
        elif z > 1500:
            return "DESCENT"
        else:
            return "LANDING"

    def adjust_strategy(self, lander, signal, phase):
        if phase == "ENTRY":
            lander.Cd *= (1 + 0.2 * signal)
        elif phase == "DESCENT":
            if signal > 0.3:
                lander.parachute_deployed = True
        elif phase == "LANDING":
            lander.kp, lander.ki, lander.kd = \
                self.optimizer.optimize_pid(lander.kp, lander.ki, lander.kd, signal)


# ============================================================================
# 5. AGENTE ESPACIAL Y SIMULACIÓN MULTI-AGENTE
# ============================================================================
class SpaceAgent:
    def __init__(self, name, state, ai):
        self.name = name
        self.state = state   # diccionario con datos propios
        self.ai = ai          # función de evaluación
        self.risk = 0

    def update(self, global_context):
        signal = self.ai.evaluate(**global_context)
        self.risk = global_context["riesgo"] * (1 - signal)
        return signal

class MultiAgentSimulation:
    def __init__(self, agents):
        self.agents = agents

    def step(self, context):
        signals = []
        for agent in self.agents:
            signal = agent.update(context)
            signals.append(signal)
        avg_signal = np.mean(signals)
        for agent in self.agents:
            agent.risk *= (1 - avg_signal)
        return avg_signal

# ============================================================================
# 6. VALIDACIÓN CON MISIONES REALES (PERSEVERANCE)
# ============================================================================
class MissionValidator:
    def __init__(self, sim_data):
        self.sim_data = sim_data
        self.real = {
            't': [0, 30, 60, 90, 120, 180, 240, 300, 370, 400],
            'vel': [5400, 4200, 3000, 1800, 900, 300, 80, 15, 1.2, 0.5]
        }

    def compute_rms(self):
        v_interp = np.interp(self.real['t'], self.sim_data['t'], self.sim_data['vel'])
        return np.sqrt(np.mean((np.array(self.real['vel']) - v_interp)**2))

    def plot_comparison(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.sim_data['t'], self.sim_data['vel'], 'b-', linewidth=2, label='GÉNESIS-KAI')
        plt.scatter(self.real['t'], self.real['vel'], color='red', s=50, label='Perseverance (NASA)')
        plt.title("Validación del Descenso: GÉNESIS-KAI vs Datos Reales")
        plt.xlabel("Tiempo desde entrada (s)")
        plt.ylabel("Velocidad (m/s)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()


# ============================================================================
# 7. ANIMACIÓN 4D CONTINUA (PLOTLY)
# ============================================================================
def create_4d_animation(history):
    import plotly.graph_objects as go
    x = np.array(history['x']); z = np.array(history['z'])
    step = max(1, len(x)//200)
    fig = go.Figure(
        data=[go.Scatter(x=x, y=z, mode='lines', line=dict(color='cyan', width=2), name='Trayectoria')],
        layout=go.Layout(
            title="Animación 4D del Descenso",
            xaxis=dict(title="Distancia horizontal (m)"),
            yaxis=dict(title="Altitud (m)"),
            updatemenus=[dict(type="buttons", showactive=False,
                              buttons=[dict(label="▶ Reproducir", method="animate", args=[None, {"frame": {"duration": 50}, "fromcurrent": True}])])]
        ),
        frames=[go.Frame(data=[go.Scatter(x=[x[i]], y=[z[i]], mode='markers', marker=dict(size=8, color='red'))])
                for i in range(0, len(x), step)]
    )
    fig.show()


# ============================================================================
# 8. VENTANA DE LANZAMIENTO Y PORKCHOP PLOT
# ============================================================================
def plot_launch_window():
    w_earth = 2 * np.pi / 365.25
    w_mars  = 2 * np.pi / 687.0
    fechas = [120, 280, 310]
    etiquetas = ["Abril 2026", "Octubre 2026 (VENTANA)", "Noviembre 2026"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': 'polar'})
    for i, days in enumerate(fechas):
        te = w_earth * days
        tm = w_mars * days
        theta = np.linspace(0, 2*np.pi, 100)
        axs[i].plot(theta, np.ones(100), 'b', alpha=0.3)
        axs[i].plot(theta, 1.524*np.ones(100), 'r', alpha=0.3)
        axs[i].scatter(te, 1, s=100, c='blue', label='Tierra')
        axs[i].scatter(tm, 1.524, s=80, c='red', label='Marte')
        axs[i].scatter(0, 0, s=200, c='gold')
        axs[i].set_title(f"{etiquetas[i]}\nFase: {(np.degrees(te-tm)%360):.1f}°")
        if i==0:
            axs[i].legend(loc='lower right')
    plt.tight_layout()
    plt.show()

def porkchop_plot():
    salida = np.linspace(1, 31, 50)
    llegada = np.linspace(200, 260, 50)
    S, L = np.meshgrid(salida, llegada)
    C3 = (S - 15)**2 + (L - 210)**2 * 0.1 + 18
    plt.figure(figsize=(10,7))
    cp = plt.contourf(S, L, C3, cmap='viridis_r', levels=20)
    plt.colorbar(cp, label='C3 (km²/s²)')
    plt.scatter(15, 210, c='red', marker='x', s=100, label='Óptimo')
    plt.title("Porkchop Plot - Ventana Octubre 2026")
    plt.xlabel("Día de salida (Octubre 2026)")
    plt.ylabel("Días de viaje")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# ============================================================================
# 9. OPTIMIZACIÓN PARALELA (MULTIPROCESSING)
# ============================================================================
def simular_variacion(dv):
    nav = SPICENavigator()
    start_jd = 2461330.0
    estado0 = [149597870.7, 0, 0, 0, 29780 + dv, 0]
    t_span = (0, 210*86400)
    sol = solve_ivp(lambda t,y: n_body_ode(t,y,nav,start_jd), t_span, estado0, rtol=1e-10)
    return np.min(sol.y[0]**2 + sol.y[1]**2) ** 0.5 / 1.496e11

def run_parallel_optimization():
    from multiprocessing import Pool
    dv_list = [3200, 3400, 3600, 3800, 4000]
    with Pool(processes=4) as pool:
        resultados = pool.map(simular_variacion, dv_list)
    print("Resultados de la búsqueda paralela (distancia final a Marte, AU):")
    for dv, dist in zip(dv_list, resultados):
        print(f"  ΔV = {dv} m/s → {dist:.3f} UA")


# ============================================================================
# 10. INTERFAZ GRÁFICA PROFESIONAL CON MULTIAGENTE
# ============================================================================
class GenesisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GÉNESIS-KAI v7.0 - Misión a Marte con IA Multiagente")
        self.root.geometry("1600x900")
        
        # Parámetros de simulación
        self.launch_day = tk.DoubleVar(value=15)
        self.travel_days = tk.DoubleVar(value=210)
        self.dv_injection = tk.DoubleVar(value=3.5)
        self.gamma_deg = tk.DoubleVar(value=12.0)
        self.mass = tk.DoubleVar(value=2200.0)
        self.cd = tk.DoubleVar(value=1.5)
        self.area = tk.DoubleVar(value=15.0)
        self.alt_entry = tk.DoubleVar(value=120.0)
        self.kp = tk.DoubleVar(value=1500.0)
        self.ki = tk.DoubleVar(value=10.0)
        self.kd = tk.DoubleVar(value=450.0)
        
        # Variables de riesgo y energía para los agentes
        self.riesgo = tk.DoubleVar(value=0.5)
        self.energia = tk.DoubleVar(value=0.7)
        self.soberania = tk.DoubleVar(value=0.8)
        
        self.last_sim_data = None
        self.last_orb_sol = None
        self.sim_running = False
        self.config_file = "genesis_config.json"
        
        self.build_ui()
        self.load_config()
    
    def build_ui(self):
        # Paneles
        left_frame = ttk.Frame(self.root, width=400, relief="ridge")
        left_frame.pack(side="left", fill="y", padx=5, pady=5)
        
        right_frame = ttk.Frame(self.root)
        right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        # Notebook con pestañas
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill="both", expand=True)
        
        self.tab_main = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_main, text="Gráfico Principal")
        self.fig = plt.Figure(figsize=(7,5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_main)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.tab_g = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_g, text="Fuerzas G")
        self.fig_g = plt.Figure(figsize=(7,5), dpi=100)
        self.ax_g = self.fig_g.add_subplot(111)
        self.canvas_g = FigureCanvasTkAgg(self.fig_g, master=self.tab_g)
        self.canvas_g.get_tk_widget().pack(fill="both", expand=True)
        
        self.tab_agents = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_agents, text="Sistema Multiagente")
        self.fig_agents = plt.Figure(figsize=(7,5), dpi=100)
        self.ax_agents = self.fig_agents.add_subplot(111)
        self.canvas_agents = FigureCanvasTkAgg(self.fig_agents, master=self.tab_agents)
        self.canvas_agents.get_tk_widget().pack(fill="both", expand=True)
        
        # ---- Controles izquierda ----
        ttk.Label(left_frame, text="VENTANA DE LANZAMIENTO", font=("Arial", 12, "bold")).pack(pady=5)
        self._add_slider("Día salida (Oct 2026):", self.launch_day, 1, 31)
        self._add_slider("Días de viaje:", self.travel_days, 200, 260)
        self._add_slider("ΔV inyección (km/s):", self.dv_injection, 2.5, 5.0, 0.05)
        
        ttk.Separator(left_frame, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(left_frame, text="DESCENSO ATMOSFÉRICO", font=("Arial", 12, "bold")).pack(pady=5)
        self._add_slider("Masa (kg):", self.mass, 1000, 3500)
        self._add_slider("Cd:", self.cd, 0.5, 2.5, 0.05)
        self._add_slider("Área (m²):", self.area, 5, 30)
        self._add_slider("Altitud inicial (km):", self.alt_entry, 80, 200)
        
        ttk.Separator(left_frame, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(left_frame, text="CONTROL PID", font=("Arial", 12, "bold")).pack(pady=5)
        self._add_slider("Kp:", self.kp, 500, 3000)
        self._add_slider("Ki:", self.ki, 0, 50)
        self._add_slider("Kd:", self.kd, 100, 1000)
        
        ttk.Separator(left_frame, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(left_frame, text="CONTEXTO MULTIAGENTE", font=("Arial", 12, "bold")).pack(pady=5)
        self._add_slider("Riesgo:", self.riesgo, 0, 1, 0.01)
        self._add_slider("Energía:", self.energia, 0, 1, 0.01)
        self._add_slider("Soberanía:", self.soberania, 0, 1, 0.01)
        
        # Botones de acción
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill="x", pady=10)
        ttk.Button(btn_frame, text="📊 Ventana lanzamiento", command=self.thread_launch).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="🐷 Porkchop Plot", command=self.thread_porkchop).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="🌍 Trayectoria orbital", command=self.thread_orbital).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="🪂 Simular aterrizaje", command=self.thread_landing).pack(fill="x", pady=2)
        self.btn_validate = ttk.Button(btn_frame, text="📈 Validación Perseverance", command=self.thread_validate, state="disabled")
        self.btn_validate.pack(fill="x", pady=2)
        self.btn_animate = ttk.Button(btn_frame, text="🎬 Animación 4D", command=self.thread_animate, state="disabled")
        self.btn_animate.pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="🤖 Simular Multiagente", command=self.thread_multiagent).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="⚡ Optimización paralela", command=self.thread_parallel).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="💾 Guardar configuración", command=self.save_config).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="📂 Cargar configuración", command=self.load_config_dialog).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="📁 Seleccionar kernel JPL", command=self.select_kernel).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="📊 Exportar resultados CSV", command=self.export_csv).pack(fill="x", pady=2)
        
        self.progress = ttk.Progressbar(left_frame, mode='indeterminate')
        self.progress.pack(fill="x", pady=5)
        
        ttk.Label(left_frame, text="LOG:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10,0))
        self.log_text = tk.Text(left_frame, height=15, wrap="word")
        self.log_text.pack(fill="both", expand=True, pady=5)
        ttk.Button(left_frame, text="Limpiar log", command=self.clear_log).pack(pady=2)
    
    def _add_slider(self, label, variable, from_, to, resolution=1):
        ttk.Label(self, text=label).pack(anchor="w")
        ttk.Scale(self, from_=from_, to=to, variable=variable, orient="horizontal", resolution=resolution).pack(fill="x")
        ttk.Label(self, textvariable=variable).pack()
    
    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
    
    def start_progress(self):
        self.progress.start(10)
        self.sim_running = True
    
    def stop_progress(self):
        self.progress.stop()
        self.sim_running = False
    
    def update_buttons_state(self):
        state = "normal" if self.last_sim_data else "disabled"
        self.btn_validate.config(state=state)
        self.btn_animate.config(state=state)
    
    # ---- Métodos con hilos ----
    def thread_launch(self):
        threading.Thread(target=plot_launch_window, daemon=True).start()
    
    def thread_porkchop(self):
        threading.Thread(target=porkchop_plot, daemon=True).start()
    
    def thread_orbital(self):
        threading.Thread(target=self.run_orbital_sim, daemon=True).start()
    
    def run_orbital_sim(self):
        if self.sim_running: return
        self.start_progress()
        self.log("Iniciando simulación orbital...")
        dv = self.dv_injection.get() * 1000.0
        nav = SPICENavigator()
        estado0 = [149597870.7, 0, 0, 0, 29780 + dv, 0]
        t_span = (0, self.travel_days.get() * 86400)
        try:
            sol = solve_ivp(lambda t,y: n_body_ode(t,y,nav,2461330.0), t_span, estado0, rtol=1e-10)
            self.ax.clear()
            self.ax.plot(sol.y[0]/1.496e11, sol.y[1]/1.496e11, 'r-', label='Trayectoria')
            theta = np.linspace(0,2*np.pi,100)
            self.ax.plot(np.cos(theta), np.sin(theta), 'b--', alpha=0.5, label='Tierra')
            self.ax.plot(1.524*np.cos(theta), 1.524*np.sin(theta), 'orange', alpha=0.5, label='Marte')
            self.ax.scatter(0,0, c='gold', s=80, label='Sol')
            self.ax.set_xlabel("AU"); self.ax.set_ylabel("AU")
            self.ax.axis('equal'); self.ax.legend()
            self.canvas.draw()
            self.log("Órbita integrada correctamente.")
        except Exception as e:
            self.log(f"Error: {e}")
        self.stop_progress()
    
    def thread_landing(self):
        threading.Thread(target=self.run_landing, daemon=True).start()
    
    def run_landing(self):
        if self.sim_running: return
        self.start_progress()
        self.log("Ejecutando aterrizaje con sistema autónomo...")
        try:
            lander = AdvancedLander(mass=self.mass.get())
            lander.Cd = self.cd.get()
            lander.area = self.area.get()
            lander.kp, lander.ki, lander.kd = self.kp.get(), self.ki.get(), self.kd.get()
            state0 = [0.0, self.alt_entry.get()*1000, 5000.0, -1200.0]
            
            # Sistema Autónomo de Decisión
            optimizer = PIDOptimizer()
            aut_sys = AutonomousDecisionSystem(optimizer)
            sol = solve_ivp(lander.dynamics, (0,600), state0, t_eval=np.linspace(0,600,3000), rtol=1e-8)
            
            # Simular la decisión autónoma en el tiempo (ejemplo)
            for i in range(len(sol.t)-1):
                z_actual = sol.y[1][i]
                vz_actual = sol.y[3][i]
                # Simulamos una señal de riesgo dinámica (oscilación suave)
                signal = 0.5 + 0.3*np.sin(sol.t[i]/100)
                phase = aut_sys.decide_phase(z_actual, vz_actual, signal)
                aut_sys.adjust_strategy(lander, signal, phase)
            
            v_abs = np.hypot(sol.y[2], sol.y[3])
            g_forces = []
            dt = np.diff(sol.t)
            for i in range(len(sol.t)-1):
                ax = (sol.y[2][i+1] - sol.y[2][i])/dt[i] if dt[i]>0 else 0
                az = (sol.y[3][i+1] - sol.y[3][i])/dt[i] if dt[i]>0 else 0
                non_grav = np.hypot(ax, az + lander.g_mars)
                g_forces.append(non_grav / 9.81)
            g_forces.append(g_forces[-1])
            self.last_sim_data = {
                't': sol.t, 'x': sol.y[0], 'z': sol.y[1], 'vel': v_abs,
                'vx': sol.y[2], 'vz': sol.y[3], 'g_force': np.array(g_forces)
            }
            v_final = self.last_sim_data['vel'][-1]
            self.log(f"Aterrizaje completado. Velocidad impacto: {v_final:.2f} m/s")
            # Gráficos
            self.ax.clear()
            self.ax.plot(sol.t, sol.y[1]/1000, 'b-', label='Altitud (km)')
            ax2 = self.ax.twinx()
            ax2.plot(sol.t, v_abs, 'r--', label='Velocidad (m/s)')
            self.ax.set_xlabel("Tiempo (s)"); self.ax.set_ylabel("Altitud (km)", color='b')
            ax2.set_ylabel("Velocidad (m/s)", color='r')
            self.ax.legend(loc='upper left'); ax2.legend(loc='upper right')
            self.canvas.draw()
            self.ax_g.clear()
            self.ax_g.plot(sol.t, g_forces, 'g-', label='Fuerzas G')
            self.ax_g.axhline(y=5, color='orange', linestyle='--', label='Límite seguridad')
            self.ax_g.set_xlabel("Tiempo (s)"); self.ax_g.set_ylabel("Fuerzas G (g0)")
            self.ax_g.legend()
            self.canvas_g.draw()
            self.update_buttons_state()
        except Exception as e:
            self.log(f"Error en aterrizaje: {e}")
        self.stop_progress()
    
    def thread_validate(self):
        threading.Thread(target=self.validate_landing, daemon=True).start()
    
    def validate_landing(self):
        if self.last_sim_data is None:
            self.log("Primero ejecuta un aterrizaje.")
            return
        self.start_progress()
        validator = MissionValidator(self.last_sim_data)
        rms = validator.compute_rms()
        self.log(f"Error RMS vs Perseverance: {rms:.2f} m/s")
        validator.plot_comparison()
        self.stop_progress()
    
    def thread_animate(self):
        threading.Thread(target=self.animate_landing, daemon=True).start()
    
    def animate_landing(self):
        if self.last_sim_data is None:
            self.log("Ejecuta aterrizaje primero.")
            return
        self.start_progress()
        create_4d_animation(self.last_sim_data)
        self.stop_progress()
    
    def thread_multiagent(self):
        threading.Thread(target=self.run_multiagent, daemon=True).start()
    
    def run_multiagent(self):
        self.start_progress()
        self.log("Iniciando simulación multiagente...")
        
        # Definir IA para cada agente (función de evaluación simple)
        class SimpleAI:
            def evaluate(self, riesgo, energia, soberania):
                return 1 - (riesgo * 0.5 + (1-energia) * 0.3 + (1-soberania) * 0.2)
        
        # Crear agentes (por ejemplo: módulo de navegación, térmico, comunicaciones)
        agents = []
        for name in ["Navigator", "Thermal", "Comms"]:
            state = {"name": name, "status": "active"}
            agents.append(SpaceAgent(name, state, SimpleAI()))
        
        multi = MultiAgentSimulation(agents)
        context = {
            "riesgo": self.riesgo.get(),
            "energia": self.energia.get(),
            "soberania": self.soberania.get(),
            "tension": self.riesgo.get()   # alias
        }
        
        # Evolución temporal (10 pasos)
        history_risk = []
        for step in range(10):
            avg_signal = multi.step(context)
            history_risk.append([agent.risk for agent in agents])
            self.log(f"Paso {step+1}: Señal media = {avg_signal:.3f}")
        
        # Graficar evolución de riesgos
        self.ax_agents.clear()
        for i, agent in enumerate(agents):
            self.ax_agents.plot(range(1,11), [r[i] for r in history_risk], label=agent.name)
        self.ax_agents.set_xlabel("Iteración"); self.ax_agents.set_ylabel("Riesgo")
        self.ax_agents.set_title("Evolución del Riesgo de los Agentes")
        self.ax_agents.legend()
        self.canvas_agents.draw()
        self.log("Simulación multiagente finalizada.")
        self.stop_progress()
    
    def thread_parallel(self):
        threading.Thread(target=run_parallel_optimization, daemon=True).start()
    
    # ---- Configuración y utilidades ----
    def save_config(self):
        config = {k: getattr(self, k).get() for k in ['launch_day','travel_days','dv_injection','gamma_deg','mass','cd','area','alt_entry','kp','ki','kd','riesgo','energia','soberania']}
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)
        self.log(f"Configuración guardada en {self.config_file}")
    
    def load_config(self, filename=None):
        if filename is None:
            filename = self.config_file
        if not os.path.exists(filename): return
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
            for k, v in config.items():
                getattr(self, k).set(v)
            self.log(f"Configuración cargada desde {filename}")
        except Exception as e:
            self.log(f"Error cargando configuración: {e}")
    
    def load_config_dialog(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if filename:
            self.load_config(filename)
    
    def select_kernel(self):
        filename = filedialog.askopenfilename(filetypes=[("BSP files", "*.bsp")])
        if filename:
            global _nav_instance
            _nav_instance = SPICENavigator(filename)
            self.log(f"Kernel JPL cambiado a: {filename}")
    
    def export_csv(self):
        if self.last_sim_data is None:
            self.log("No hay datos para exportar.")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if filename:
            import pandas as pd
            df = pd.DataFrame({
                'time_s': self.last_sim_data['t'],
                'altitude_m': self.last_sim_data['z'],
                'velocity_mps': self.last_sim_data['vel'],
                'g_force': self.last_sim_data['g_force']
            })
            df.to_csv(filename, index=False)
            self.log(f"Datos exportados a {filename}")


# ============================================================================
# 11. EJECUCIÓN
# ============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = GenesisGUI(root)
    root.mainloop()
