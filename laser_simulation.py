import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.widgets import Button, RadioButtons, Slider

# ── Display toggles ─────────────────────────────────────────────────────────
SHOW_DETECTED_MARKERS = False   # show t₁+d₁/c and t₂+d₂/c arrival markers

# ── Physical parameters ──────────────────────────────────────────────────────
d      = 3.0          # laser → wall distance  [m]
c      = 3e8          # speed of light  [m/s]
x_det  = d/2          # Detector x-position along beam  [m]
y_beam = 0.0          # beam axis y  [m]
y_det  = -0.65        # Detector y (bottom of scene)  [m]
R_wall = 0.65         # wall reflectivity

s_sc   = 0.38         # scatter spatial scale  [m]   (controls halo width)
s_bm   = 0.022        # direct beam core width  [m]
s_pls  = 0.2          # pulse longitudinal sigma  [m]
t_c0   = 1.0e-9       # laser turn-on time constant [s]
pulse_sigma_t0 = s_pls / c  # pulse longitudinal sigma [s]
params = {'beam_width': s_bm, 'pulse_sigma_t': pulse_sigma_t0, 'turn_on_tc': t_c0}

# ── Simulation grid ───────────────────────────────────────────────────────────
Nx, Ny  = 200, 110
xg = np.linspace(-0.2,  d + 0.2, Nx)
yg = np.linspace(-0.75, 0.42,    Ny)
XX, YY  = np.meshgrid(xg, yg)
wall_mask = XX <= d
dx = xg[1] - xg[0]
x_line_mask = (xg >= 0.0) & (xg <= d)
scatter_y = np.exp(-0.5 * (yg / s_sc)**2)
scatter_kernel = np.exp(-0.5 * ((xg[:, None] - xg[None, :]) / s_sc)**2) * dx

n_frames = 80
t_arr    = np.linspace(0, 29e-9, n_frames)
t_ns     = t_arr * 1e9
sample_dt = 1e-9
t_disc_arr = np.arange(0.0, t_arr[-1] + 0.5 * sample_dt, sample_dt)
t_disc_ns = t_disc_arr * 1e9

line_r2 = None
t1_true = None
t2_true = None
dt_true = None
d_1 = None
d_2 = None
t1_det = None
t2_det = None
dx_baseline = None
c_geom = None

def recompute_detector_geometry():
    """Update Detector-dependent geometry/timing values when Detector x changes."""
    global line_r2, t1_true, t2_true, dt_true, d_1, d_2, t1_det, t2_det
    global dx_baseline, c_geom
    line_r2 = (xg - x_det)**2 + y_det**2

    t1_true = x_det / c
    t2_true = (2 * d - x_det) / c
    dt_true = t2_true - t1_true
    dx_baseline = d - x_det
    c_geom = 2.0 * dx_baseline / dt_true

    dist_laser_to_det = np.sqrt(x_det**2 + (y_det - y_beam)**2)
    d_1 = dist_laser_to_det - x_det

    dist_wall_to_det = np.sqrt((d - x_det)**2 + (y_det - y_beam)**2)
    d_2 = dist_wall_to_det - abs(d - x_det)

    t1_det = t1_true + d_1 / c
    t2_det = t2_true + d_2 / c

recompute_detector_geometry()

# ── Scatter kernel ────────────────────────────────────────────────────────────
def lorentz2d(bx, amp=1.0):
    """Lorentzian (1/r²-like) scatter halo from a point at (bx, 0)."""
    r2 = (XX - bx)**2 + YY**2
    return amp * s_sc**2 / (r2 + s_sc**2)

def beam_glow(x_lo, x_hi, amp=1.0):
    """Analytically integrated Gaussian scatter from beam segment [x_lo, x_hi]."""
    if x_hi <= x_lo:
        return np.zeros((Ny, Nx))
    scatter_scale = s_sc
    sq2 = np.sqrt(2) * scatter_scale
    gy  = np.exp(-0.5 * (YY / scatter_scale)**2)
    gx  = (erf((XX - x_lo) / sq2) - erf((XX - x_hi) / sq2)).clip(0)
    return amp * gy * gx

def beam_core(x_lo, x_hi, amp=1.0):
    """Direct bright beam line between x_lo and x_hi."""
    beam_width = params['beam_width']
    mask = (XX >= x_lo) & (XX <= x_hi)
    return amp * np.exp(-0.5 * (YY / beam_width)**2) * mask

def pulse_spot(cx, amp=1.0):
    """Narrow Gaussian spot at (cx, 0) for direct beam core."""
    pulse_length = c * params['pulse_sigma_t']
    beam_width = params['beam_width']
    return amp * np.exp(-0.5 * ((XX - cx) / pulse_length)**2) \
               * np.exp(-0.5 * (YY / beam_width)**2)

def source_ramp(t_emit):
    """Laser turn-on envelope I/I0 = 1 - exp(-t/t_c), clipped to t >= 0."""
    tc = params['turn_on_tc']
    t_positive = np.maximum(t_emit, 0.0)
    return 1.0 - np.exp(-t_positive / tc)

def laser_intensity_trace():
    """Source intensity trace evaluated at the laser output versus time."""
    return source_ramp(t_arr)

def glow_from_profile(profile, amp=1.0):
    """Diffuse scatter from a longitudinal intensity profile along the beam."""
    x_blur = scatter_kernel @ profile
    return amp * scatter_y[:, None] * x_blur[None, :]

def forward_causal_mask(t):
    """True causal mask for forward scatter: circular light cone from laser (0,0)."""
    r = np.sqrt(XX**2 + YY**2)
    return r <= c * t

def reflected_causal_mask(t):
    """True causal mask for reflected scatter: circular light cone from wall (d,0)."""
    t_r = d / c
    if t <= t_r:
        return np.zeros_like(XX, dtype=bool)
    r = np.sqrt((XX - d)**2 + YY**2)
    return r <= c * (t - t_r)

def core_from_profile(profile, amp=1.0):
    """Direct beam core from a longitudinal intensity profile."""
    beam_width = params['beam_width']
    return amp * np.exp(-0.5 * (YY / beam_width)**2) * profile[None, :]

def det_signal_from_profile(profile):
    """Numerical Detector signal from a distributed beam-axis intensity profile."""
    return float(np.sum(profile / np.maximum(line_r2, 1e-4)) * dx)

def det_signal_from_map(field):
    """Detector signal sampled directly from the rendered 2D intensity map near the Detector."""
    # Finite Detector aperture (Gaussian weighting) gives a stable, pixel-based readout.
    aperture_sigma = 0.04
    r2 = (XX - x_det)**2 + (YY - y_det)**2
    weights = np.exp(-0.5 * r2 / (aperture_sigma**2))
    wsum = np.sum(weights)
    if wsum <= 1e-12:
        return 0.0
    return float(np.sum(field * weights) / wsum)

def forward_beam_profile(t):
    """Beam profile with retarded-time source ramp during turn-on."""
    retarded_time = t - xg / c
    illuminated = (xg >= 0.0) & (xg <= min(c * t, d))
    return source_ramp(retarded_time) * illuminated

def reflected_beam_profile(t):
    """Reflected beam profile with retarded-time source ramp."""
    t_r = d / c
    if t <= t_r:
        return np.zeros_like(xg)
    cr = d - c * (t - t_r)
    retarded_time = t - (2 * d - xg) / c
    illuminated = (xg >= max(cr, 0.0)) & (xg <= d)
    return R_wall * source_ramp(retarded_time) * illuminated

def forward_pulse_profile(t):
    """Forward pulse profile with turn-on envelope applied at emission time."""
    cx = c * t
    pulse_length = c * params['pulse_sigma_t']
    # Launch the pulse after the turn-on transient so its peak is not suppressed.
    pulse_launch_delay = 3.0 * params['turn_on_tc']
    retarded_time = t - xg / c + pulse_launch_delay
    envelope = np.exp(-0.5 * ((xg - cx) / pulse_length)**2)
    return source_ramp(retarded_time) * envelope * x_line_mask

def reflected_pulse_profile(t):
    """Reflected pulse profile with turn-on envelope applied at emission time."""
    t_r = d / c
    if t <= t_r:
        return np.zeros_like(xg)
    cr = d - c * (t - t_r)
    pulse_length = c * params['pulse_sigma_t']
    pulse_launch_delay = 3.0 * params['turn_on_tc']
    retarded_time = t - (2 * d - xg) / c + pulse_launch_delay
    envelope = np.exp(-0.5 * ((xg - cr) / pulse_length)**2)
    return R_wall * source_ramp(retarded_time) * envelope * x_line_mask

# ── Detector signal helpers ────────────────────────────────────────────────────────
def det_inv_sq(cx):
    """1/r² from moving point source at (cx,0) to Detector."""
    r2 = (cx - x_det)**2 + y_det**2
    return 1.0 / max(r2, 1e-4)

def det_beam_integral(x_lo, x_hi):
    """∫_{x_lo}^{x_hi} 1/r² dx'  (analytic arctan form)."""
    yp = abs(y_det)
    return (np.arctan((x_hi - x_det) / yp)
           - np.arctan((x_lo - x_det) / yp)) / yp

# ── Per-frame computation — retarded-time model ───────────────────────────────
def compute_retarded(t, mode):
    """Retarded-time scattering field with circular wavefronts.

    Glow (diffuse Rayleigh scatter): circular wavefront expanding from the
    source point.  Iso-delay surfaces r = ct are circles, so the bright halo
    naturally forms a ring that grows outward.  Causality is automatic —
    source_ramp / Gaussian evaluate to zero where t_ret < 0.

      Forward glow:   t_ret = t − sqrt(X²+Y²) / c
                      → circle of radius ct centred at laser origin (0, 0)
      Reflected glow: t_ret = t − d/c − sqrt((X−d)²+Y²) / c
                      → circle expanding from wall (d, 0), starts at t = d/c

    Beam core (collimated): uses the along-axis travel time X/c so the
    bright beam line appears correctly as it sweeps down the x-axis.
    """
    # ── glow wavefront distances (give circular iso-delay contours) ──────────
    r_fwd = np.sqrt(XX**2 + YY**2)
    r_ref = np.sqrt((XX - d)**2 + YY**2)

    t_ret_glow_fwd = t - r_fwd / c
    t_ret_glow_ref = t - (d / c) - r_ref / c

    # ── core beam along-axis retarded times ──────────────────────────────────
    t_ret_core_fwd = t - XX / c
    t_ret_core_ref = t - (2.0 * d - XX) / c

    glow    = np.exp(-0.5 * (YY / s_sc)**2)
    core    = np.exp(-0.5 * (YY / params['beam_width'])**2)
    on_axis = (XX >= 0.0) & (XX <= d)          # core only where beam travels

    if mode == 'pulse':
        sigma_t = params['pulse_sigma_t']
        f_glow_fwd = np.exp(-0.5 * (t_ret_glow_fwd / sigma_t)**2)
        f_core_fwd = np.exp(-0.5 * (t_ret_core_fwd / sigma_t)**2) * on_axis
        f_glow_ref = R_wall * np.exp(-0.5 * (t_ret_glow_ref / sigma_t)**2)
        f_core_ref = R_wall * np.exp(-0.5 * (t_ret_core_ref / sigma_t)**2) * on_axis
    else:                                           # beam
        f_glow_fwd = source_ramp(t_ret_glow_fwd)
        f_core_fwd = source_ramp(t_ret_core_fwd) * on_axis
        f_glow_ref = R_wall * source_ramp(t_ret_glow_ref)
        f_core_ref = R_wall * source_ramp(t_ret_core_ref) * on_axis

    F = (0.80 * glow * (f_glow_fwd + f_glow_ref) +
         0.55 * core * (f_core_fwd + f_core_ref))
    F *= wall_mask
    v = det_signal_from_map(F)
    return np.clip(F, 0, 2.0), float(v)


def compute_pulse(t):
    return compute_retarded(t, 'pulse')


def compute_beam(t):
    return compute_retarded(t, 'beam')

def precompute_mode_frames(compute_fn, label):
    print(f"Precomputing {label} frames…", end=" ", flush=True)
    intensities, signal = [], []
    for t in t_arr:
        intensity, value = compute_fn(t)
        intensities.append(intensity)
        signal.append(value)
    vmax = max(signal) or 1
    signal = [value / vmax for value in signal]
    print("done.")
    return intensities, signal

def normalised_derivative(signal, time_axis):
    """Normalised first derivative d(signal)/dt for overlay plotting."""
    ds_dt = np.gradient(np.asarray(signal), np.asarray(time_axis))
    peak = np.max(np.abs(ds_dt))
    if peak <= 1e-12:
        return np.zeros_like(ds_dt)
    return ds_dt / peak

def sample_signal_1ns(signal):
    """Resample a signal onto 1 ns sampling points for detector-limited mode."""
    return np.interp(t_disc_arr, t_arr, np.asarray(signal))

def recompute_all_frames():
    pulse_intensity, pulse_signal = precompute_mode_frames(compute_pulse, 'pulse')
    beam_intensity, beam_signal = precompute_mode_frames(compute_beam, 'beam')
    pulse_derivative = normalised_derivative(pulse_signal, t_arr)
    beam_derivative = normalised_derivative(beam_signal, t_arr)

    pulse_signal_disc = sample_signal_1ns(pulse_signal)
    beam_signal_disc = sample_signal_1ns(beam_signal)
    pulse_derivative_disc = normalised_derivative(pulse_signal_disc, t_disc_arr)
    beam_derivative_disc = normalised_derivative(beam_signal_disc, t_disc_arr)

    return (
        pulse_intensity,
        pulse_signal,
        pulse_derivative,
        pulse_signal_disc,
        pulse_derivative_disc,
        beam_intensity,
        beam_signal,
        beam_derivative,
        beam_signal_disc,
        beam_derivative_disc,
    )

# ── Precompute both mode frame sets ──────────────────────────────────────────
pI, pV, pD, pV_disc, pD_disc, bI, bV, bD, bV_disc, bD_disc = recompute_all_frames()
src_trace_cont = laser_intensity_trace()
src_trace_disc = source_ramp(t_disc_arr)

# ── Figure layout ─────────────────────────────────────────────────────────────
BG = '#0d0d1a'; TX = '#ccccdd'
fig = plt.figure(figsize=(13.5, 7.2), facecolor=BG)
fig.suptitle("PE3 Speed of Light Experiment — Single-Pixel Detector Simulation",
             color='white', fontsize=11, y=0.985)

ax_map = fig.add_axes([0.05, 0.10, 0.60, 0.82])   # heatmap
ax_sig = fig.add_axes([0.70, 0.10, 0.28, 0.26])   # Detector signal
ax_src = fig.add_axes([0.70, 0.43, 0.28, 0.14])   # source intensity
ax_rad = fig.add_axes([0.70, 0.614, 0.28, 0.316],   # laser settings
                      facecolor='#12122a')
ax_save_btn = fig.add_axes([0.70,   0.010, 0.135, 0.035])  # save PNG button
ax_csv_btn  = fig.add_axes([0.843,  0.010, 0.137, 0.035])  # save CSV button

for ax in (ax_map, ax_sig, ax_src):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TX, labelsize=8.5)
    for sp in ax.spines.values(): sp.set_edgecolor('#334')

# ── Heatmap ───────────────────────────────────────────────────────────────────
heatmap_vmax = 1.5
im = ax_map.imshow(
    bI[0], extent=[xg[0], xg[-1], yg[0], yg[-1]],
    aspect='auto', origin='lower', cmap='inferno',
    norm=Normalize(vmin=0.0, vmax=heatmap_vmax), interpolation='bilinear'
)
ax_map.axvline(d, color='#00cfff', lw=2.5, zorder=3, label='Wall')
ax_map.axvline(0, color='#39ff14', lw=1.5, ls='--', alpha=0.6, zorder=3, label='Laser')
ax_map.axhline(y_beam, color='white', lw=0.4, ls=':', alpha=0.18)

# Detector at the bottom
det_marker, = ax_map.plot(
    x_det, y_det, 'D', ms=11, color='#ff6060',
    markeredgecolor='white', markeredgewidth=0.7, zorder=6
)
det_label = ax_map.annotate(
    'Detector', xy=(x_det, y_det),
    xytext=(x_det + 0.12, y_det + 0.04),
    color='#ff6060', fontsize=9, fontweight='bold', zorder=7
)
# Dashed guide line from Detector to beam axis
det_guide, = ax_map.plot(
    [x_det, x_det], [y_det + 0.08, y_beam - 0.03],
    color='#ff6060', ls=':', lw=1.0, alpha=0.40
)

cb = fig.colorbar(im, ax=ax_map, fraction=0.018, pad=0.01)
cb.ax.yaxis.set_tick_params(color=TX)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=TX, fontsize=8)

ax_map.set_xlabel('x (m)', color=TX, fontsize=9)
ax_map.set_ylabel('y (m)', color=TX, fontsize=9)
ax_map.legend(loc='upper right', fontsize=8, framealpha=0.20,
              labelcolor=TX, facecolor='#1a1a2e')

tlbl = ax_map.text(0.02, 0.96, '', transform=ax_map.transAxes,
                   color='white', fontsize=9.5, fontweight='bold',
                   bbox=dict(facecolor=BG, alpha=0.55, pad=3))
mlbl = ax_map.text(0.50, 0.96, 'Mode: Beam', transform=ax_map.transAxes,
                   color='#66ccff', fontsize=9, ha='center',
                   bbox=dict(facecolor=BG, alpha=0.55, pad=3))
map_title = ax_map.set_title('Top-down view: laser scatter',
                              color=TX, fontsize=10, pad=5)

# ── Detector signal plot ───────────────────────────────────────────────────────────
ax_sig.set_xlim(0, t_ns[-1]); ax_sig.set_ylim(-1.05, 1.22)
ax_sig.set_xlabel('Time (ns)', color=TX, fontsize=9)
ax_sig.set_ylabel('Detector signal', color=TX, fontsize=9)
ax_sig.axhline(0, color='#334', lw=0.7)

t1_true_line = ax_sig.axvline(t1_true * 1e9, color='#ffdd44', lw=1.2, ls='--', alpha=0.65)
t2_true_line = ax_sig.axvline(t2_true * 1e9, color='#ff8844', lw=1.2, ls='--', alpha=0.65)
t1_det_line = ax_sig.axvline(t1_det * 1e9, color='#ffe88a', lw=1.2, ls=':', alpha=0.85,
                             visible=SHOW_DETECTED_MARKERS)
t2_det_line = ax_sig.axvline(t2_det * 1e9, color='#ffb07a', lw=1.2, ls=':', alpha=0.85,
                             visible=SHOW_DETECTED_MARKERS)
t1_true_text = ax_sig.text(t1_true*1e9 + 0.10, 1.10, 't₁', color='#ffdd44', fontsize=9)
t2_true_text = ax_sig.text(t2_true*1e9 + 0.10, 1.10, 't₂', color='#ff8844', fontsize=9)
t1_det_text = ax_sig.text(t1_det*1e9 + 0.10, 1.00, 't₁ + d₁/c', color='#ffe88a', fontsize=8.5,
                          visible=SHOW_DETECTED_MARKERS)
t2_det_text = ax_sig.text(t2_det*1e9 + 0.10, 1.00, 't₂ + d₂/c', color='#ffb07a', fontsize=8.5,
                          visible=SHOW_DETECTED_MARKERS)
sig_line, = ax_sig.plot([], [], color='#66ccff', lw=1.8, label='Signal')
dsig_line, = ax_sig.plot([], [], color='#c77dff', lw=1.2, ls='--', alpha=0.95,
                         label='d(signal)/dt (norm)')
sig_legend = ax_sig.legend(loc='lower right', fontsize=7.5, framealpha=0.20,
                          facecolor='#1a1a2e')
sig_title = ax_sig.set_title('Detector readout  (Beam)', color=TX, fontsize=9.5, pad=4)

# ── Laser source intensity plot ──────────────────────────────────────────────
ax_src.set_xlim(0, t_ns[-1]); ax_src.set_ylim(-0.05, 1.05)
ax_src.set_xlabel('Time  (ns)', color=TX, fontsize=9)
ax_src.axhline(0, color='#334', lw=0.7)
src_line, = ax_src.plot(t_ns, laser_intensity_trace(), color='#8bd450', lw=1.8)
src_cursor = ax_src.axvline(t_ns[0], color='#8bd450', lw=1.0, ls='--', alpha=0.75)
src_title = ax_src.set_title('Laser source intensity', color=TX, fontsize=9.5, pad=4)

# ── Radio buttons ─────────────────────────────────────────────────────────────
ax_rad.set_xticks([])
ax_rad.set_yticks([])
for sp in ax_rad.spines.values(): sp.set_edgecolor('#445')
ax_mode   = ax_rad.inset_axes([0.04, 0.82, 0.92, 0.11], facecolor='#12122a')
ax_sample = ax_rad.inset_axes([0.04, 0.67, 0.92, 0.11], facecolor='#12122a')
ax_beam   = ax_rad.inset_axes([0.04, 0.52, 0.92, 0.09], facecolor='#12122a')
ax_pulse  = ax_rad.inset_axes([0.04, 0.39, 0.92, 0.09], facecolor='#12122a')
ax_detx   = ax_rad.inset_axes([0.04, 0.26, 0.92, 0.09], facecolor='#12122a')
ax_tc     = ax_rad.inset_axes([0.04, 0.13, 0.92, 0.09], facecolor='#12122a')
ax_speed  = ax_rad.inset_axes([0.04, 0.00, 0.92, 0.09], facecolor='#12122a')

radio = RadioButtons(ax_mode, ['Beam', 'Pulse'], activecolor='#ffdd44')
for lbl in radio.labels:
    lbl.set_color(TX); lbl.set_fontsize(9)
sampling_radio = RadioButtons(ax_sample, ['Continuous', 'Discrete (1 ns)'], activecolor='#8bd450')
for lbl in sampling_radio.labels:
    lbl.set_color(TX); lbl.set_fontsize(8.5)
ax_rad.set_title('Laser settings', color=TX, fontsize=9.5, pad=6)

beam_slider = Slider(
    ax_beam, '', 0.005, 0.050,
    valinit=params['beam_width'], valstep=0.001,
    valfmt='%.3g', color='#ffb000'
)
ax_beam.text(0.0, 1.05, 'Beam width', transform=ax_beam.transAxes,
    ha='left', va='bottom', color=TX, fontsize=8, clip_on=False)
beam_val_text = ax_beam.text(1.0, 1.05, f"{params['beam_width']:.3f} m",
    transform=ax_beam.transAxes, ha='right', va='bottom',
    color=TX, fontsize=8, clip_on=False)
beam_slider.valtext.set_visible(False)

pulse_slider = Slider(
    ax_pulse, '', 0.05, 3.00,
    valinit=params['pulse_sigma_t'] * 1e9, valstep=0.01,
    valfmt='%.3g', color='#fe6100'
)
ax_pulse.text(0.0, 1.05, 'Pulse width', transform=ax_pulse.transAxes,
    ha='left', va='bottom', color=TX, fontsize=8, clip_on=False)
pulse_val_text = ax_pulse.text(1.0, 1.05, f"{params['pulse_sigma_t']*1e9:.2f} ns",
    transform=ax_pulse.transAxes, ha='right', va='bottom',
    color=TX, fontsize=8, clip_on=False)
pulse_slider.valtext.set_visible(False)

detector_slider = Slider(
    ax_detx, '', 0.0, d,
    valinit=x_det, valstep=0.01,
    valfmt='%.3g', color='#dc267f'
)
ax_detx.text(0.0, 1.05, 'Detector x position', transform=ax_detx.transAxes,
    ha='left', va='bottom', color=TX, fontsize=8, clip_on=False)
detx_val_text = ax_detx.text(1.0, 1.05, f'{x_det:.2f} m',
    transform=ax_detx.transAxes, ha='right', va='bottom',
    color=TX, fontsize=8, clip_on=False)
detector_slider.valtext.set_visible(False)

tc_slider = Slider(
    ax_tc, '', 0.1, 10.0,
    valinit=params['turn_on_tc'] * 1e9, valstep=0.1,
    valfmt='%.3g', color='#8bd450'
)
ax_tc.text(0.0, 1.05, 'Turn-on time t₀', transform=ax_tc.transAxes,
    ha='left', va='bottom', color=TX, fontsize=8, clip_on=False)
tc_val_text = ax_tc.text(1.0, 1.05, f"{params['turn_on_tc']*1e9:.1f} ns",
    transform=ax_tc.transAxes, ha='right', va='bottom',
    color=TX, fontsize=8, clip_on=False)
tc_slider.valtext.set_visible(False)

speed_slider = Slider(
    ax_speed, '', 0.0, 2.0,
    valinit=1.0, valstep=0.05,
    valfmt='%.3g', color='#648fff'
)
ax_speed.text(0.0, 1.05, 'Simulation speed', transform=ax_speed.transAxes,
    ha='left', va='bottom', color=TX, fontsize=8, clip_on=False)
speed_val_text = ax_speed.text(1.0, 1.05, '1.00×',
    transform=ax_speed.transAxes, ha='right', va='bottom',
    color=TX, fontsize=8, clip_on=False)
speed_slider.valtext.set_visible(False)
for axis in (ax_mode, ax_sample, ax_beam, ax_pulse, ax_detx, ax_tc, ax_speed):
    axis.set_facecolor('#12122a')
    axis.tick_params(colors=TX, labelsize=8)
    for sp in axis.spines.values(): sp.set_edgecolor('#445')

save_btn = Button(ax_save_btn, 'Save PNG', color='#1a1a2e', hovercolor='#252550')
save_btn.label.set_color(TX)
save_btn.label.set_fontsize(8.5)
ax_save_btn.set_facecolor('#1a1a2e')
for sp in ax_save_btn.spines.values(): sp.set_edgecolor('#445')

csv_btn = Button(ax_csv_btn, 'Save CSV', color='#1a1a2e', hovercolor='#252550')
csv_btn.label.set_color(TX)
csv_btn.label.set_fontsize(8.5)
ax_csv_btn.set_facecolor('#1a1a2e')
for sp in ax_csv_btn.spines.values(): sp.set_edgecolor('#445')

def refresh_detector_overlays():
    """Refresh Detector marker and timing overlays after Detector x-position changes."""
    t1_ns = t1_true * 1e9
    t2_ns = t2_true * 1e9
    t1_det_ns = t1_det * 1e9
    t2_det_ns = t2_det * 1e9

    det_marker.set_xdata([x_det])
    det_guide.set_xdata([x_det, x_det])
    det_label.xy = (x_det, y_det)
    det_label.set_position((x_det + 0.12, y_det + 0.04))

    t1_true_line.set_xdata([t1_ns, t1_ns])
    t2_true_line.set_xdata([t2_ns, t2_ns])
    t1_det_line.set_xdata([t1_det_ns, t1_det_ns])
    t2_det_line.set_xdata([t2_det_ns, t2_det_ns])

    t1_true_text.set_position((t1_ns + 0.10, 1.10))
    t2_true_text.set_position((t2_ns + 0.10, 1.10))
    t1_det_text.set_position((t1_det_ns + 0.10, 1.00))
    t2_det_text.set_position((t2_det_ns + 0.10, 1.00))


# ── Animation ─────────────────────────────────────────────────────────────────
state = {
    'mode': 'Beam',
    'sampling_mode': 'Continuous',
    'frame_index': 0,
    'playback_position': 0.0,
    'playback_rate': 1.0,
}


def render_frame(i):
    state['frame_index'] = i
    mode = state['mode']
    sampling_mode = state['sampling_mode']
    discrete_mode = sampling_mode == 'Discrete (1 ns)'

    if discrete_mode:
        disc_idx = int(np.searchsorted(t_disc_arr, t_arr[i], side='right') - 1)
        disc_idx = max(disc_idx, 0)
        x_sig = t_disc_ns[:disc_idx+1]
        x_src = t_disc_ns
        src_y = src_trace_disc
        src_cursor_x = t_disc_ns[disc_idx]
        sig_line.set_linestyle('None')
        sig_line.set_marker('o')
        sig_line.set_markersize(3.0)
        dsig_line.set_linestyle('None')
        dsig_line.set_marker('s')
        dsig_line.set_markersize(2.8)
        src_line.set_linestyle('None')
        src_line.set_marker('o')
        src_line.set_markersize(2.6)
    else:
        x_sig = t_ns[:i+1]
        x_src = t_ns
        src_y = src_trace_cont
        src_cursor_x = t_ns[i]
        sig_line.set_linestyle('-')
        sig_line.set_marker('None')
        dsig_line.set_linestyle('--')
        dsig_line.set_marker('None')
        src_line.set_linestyle('-')
        src_line.set_marker('None')

    if mode == 'Pulse':
        im.set_data(pI[i])
        if discrete_mode:
            sig_line.set_data(x_sig, pV_disc[:len(x_sig)])
            dsig_line.set_data(x_sig, pD_disc[:len(x_sig)])
        else:
            sig_line.set_data(x_sig, pV[:i+1])
            dsig_line.set_data(x_sig, pD[:i+1])
        mlbl.set_text('Mode: Pulse')
        sig_title.set_text('Detector readout  (Pulse)')
        sig_line.set_color('#ffdd44')
        dsig_line.set_color('#38d9a9')
        for handle, txt, col in zip(sig_legend.legend_handles, sig_legend.get_texts(), ['#ffdd44', '#38d9a9']):
            handle.set_color(col)
            txt.set_color(col)
    else:
        im.set_data(bI[i])
        if discrete_mode:
            sig_line.set_data(x_sig, bV_disc[:len(x_sig)])
            dsig_line.set_data(x_sig, bD_disc[:len(x_sig)])
        else:
            sig_line.set_data(x_sig, bV[:i+1])
            dsig_line.set_data(x_sig, bD[:i+1])
        mlbl.set_text('Mode: Beam')
        sig_title.set_text('Detector readout  (Beam)')
        sig_line.set_color('#66ccff')
        dsig_line.set_color('#c77dff')
        for handle, txt, col in zip(sig_legend.legend_handles, sig_legend.get_texts(), ['#66ccff', '#c77dff']):
            handle.set_color(col)
            txt.set_color(col)

    tlbl.set_text(f't = {t_ns[i]:.2f} ns')
    src_line.set_data(x_src, src_y)
    src_cursor.set_xdata([src_cursor_x, src_cursor_x])
    if discrete_mode:
        src_title.set_text('Laser source intensity  (Discrete 1 ns)')
    else:
        src_title.set_text('Laser source intensity')

    return im, sig_line, dsig_line, src_line, src_cursor, tlbl, mlbl, map_title, src_title

def on_radio(label):
    state['mode'] = label
    render_frame(state['frame_index'])
    fig.canvas.draw_idle()

def on_laser_shape_change(_value):
    global pI, pV, pD, pV_disc, pD_disc, bI, bV, bD, bV_disc, bD_disc
    global src_trace_cont, src_trace_disc
    params['beam_width'] = beam_slider.val
    params['pulse_sigma_t'] = pulse_slider.val * 1e-9
    params['turn_on_tc'] = tc_slider.val * 1e-9
    beam_val_text.set_text(f'{beam_slider.val:.3f} m')
    pulse_val_text.set_text(f'{pulse_slider.val:.2f} ns')
    tc_val_text.set_text(f'{tc_slider.val:.1f} ns')
    pI, pV, pD, pV_disc, pD_disc, bI, bV, bD, bV_disc, bD_disc = recompute_all_frames()
    src_trace_cont = laser_intensity_trace()
    src_trace_disc = source_ramp(t_disc_arr)
    render_frame(state['frame_index'])
    fig.canvas.draw_idle()

def on_detector_change(_value):
    global x_det, pI, pV, pD, pV_disc, pD_disc, bI, bV, bD, bV_disc, bD_disc
    x_det = detector_slider.val
    detx_val_text.set_text(f'{x_det:.2f} m')
    recompute_detector_geometry()
    refresh_detector_overlays()
    pI, pV, pD, pV_disc, pD_disc, bI, bV, bD, bV_disc, bD_disc = recompute_all_frames()
    render_frame(state['frame_index'])
    fig.canvas.draw_idle()

def on_sampling_change(label):
    state['sampling_mode'] = label
    render_frame(state['frame_index'])
    fig.canvas.draw_idle()

def on_speed_change(_value):
    state['playback_rate'] = speed_slider.val
    speed_val_text.set_text(f'{speed_slider.val:.2f}×')
    fig.canvas.draw_idle()

def on_save_detector(_event):
    import datetime, threading, time
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = rf't:\downloads\det_signal_{timestamp}.png'

    mode = state['mode']
    sampling_mode = state['sampling_mode']
    discrete_mode = sampling_mode == 'Discrete (1 ns)'

    if discrete_mode:
        x_full = t_disc_ns
        V_full = pV_disc if mode == 'Pulse' else bV_disc
        D_full = pD_disc if mode == 'Pulse' else bD_disc
        ls_sig, ls_dsig, mk_sig, mk_dsig = 'None', 'None', 'o', 's'
    else:
        x_full = t_ns
        V_full = pV if mode == 'Pulse' else bV
        D_full = pD if mode == 'Pulse' else bD
        ls_sig, ls_dsig, mk_sig, mk_dsig = '-', '--', 'None', 'None'

    sc = sig_line.get_color()
    dc = dsig_line.get_color()

    fig2, ax2 = plt.subplots(figsize=(8, 4), facecolor=BG)
    ax2.set_facecolor(BG)
    ax2.axhline(0, color='#334', lw=0.7)
    ax2.axvline(t1_true * 1e9, color='#ffdd44', lw=1.2, ls='--', alpha=0.65)
    ax2.axvline(t2_true * 1e9, color='#ff8844', lw=1.2, ls='--', alpha=0.65)
    if SHOW_DETECTED_MARKERS:
        ax2.axvline(t1_det  * 1e9, color='#ffe88a', lw=1.2, ls=':',  alpha=0.85)
        ax2.axvline(t2_det  * 1e9, color='#ffb07a', lw=1.2, ls=':',  alpha=0.85)
    ax2.plot(x_full, V_full, color=sc, lw=1.8,
             ls=ls_sig, marker=mk_sig, markersize=3.0, label='Signal')
    ax2.plot(x_full, D_full, color=dc, lw=1.2, alpha=0.95,
             ls=ls_dsig, marker=mk_dsig, markersize=2.8, label='d(signal)/dt  (norm.)')
    ax2.text(t1_true*1e9 + 0.15, 1.10, 't\u2081',        color='#ffdd44', fontsize=10)
    ax2.text(t2_true*1e9 + 0.15, 1.10, 't\u2082',        color='#ff8844', fontsize=10)
    if SHOW_DETECTED_MARKERS:
        ax2.text(t1_det *1e9 + 0.15, 1.00, 't\u2081 + d\u2081/c', color='#ffe88a', fontsize=9)
        ax2.text(t2_det *1e9 + 0.15, 1.00, 't\u2082 + d\u2082/c', color='#ffb07a', fontsize=9)
    ax2.set_xlim(0, t_ns[-1])
    ax2.set_ylim(-1.05, 1.22)
    ax2.set_xlabel('Time (ns)', color=TX, fontsize=11)
    ax2.set_ylabel('Detector signal', color=TX, fontsize=11)
    ax2.set_title(f'Detector readout  ({mode})', color=TX, fontsize=12, pad=5)
    ax2.tick_params(colors=TX, labelsize=10)
    for sp in ax2.spines.values(): sp.set_edgecolor('#445')
    leg = ax2.legend(loc='lower right', fontsize=9.5, framealpha=0.20, facecolor='#1a1a2e')
    for txt in leg.get_texts(): txt.set_color(TX)
    fig2.tight_layout()
    fig2.savefig(fname, dpi=200, facecolor=BG)
    plt.close(fig2)

    save_btn.label.set_text('Saved!')
    fig.canvas.draw_idle()
    def _reset():
        time.sleep(2.5)
        save_btn.label.set_text('Save PNG')
        fig.canvas.draw_idle()
    threading.Thread(target=_reset, daemon=True).start()

save_btn.on_clicked(on_save_detector)

def on_save_csv(_event):
    import datetime, csv, threading, time
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    mode = state['mode']
    V_disc = pV_disc if mode == 'Pulse' else bV_disc
    D_disc = pD_disc if mode == 'Pulse' else bD_disc
    fname = rf't:\downloads\det_discrete_{mode.lower()}_{timestamp}.csv'
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_ns', 'signal', 'derivative_norm'])
        for t, v, dv in zip(t_disc_ns, V_disc, D_disc):
            writer.writerow([f'{t:.6g}', f'{v:.8g}', f'{dv:.8g}'])
    csv_btn.label.set_text('Saved!')
    fig.canvas.draw_idle()
    def _reset():
        time.sleep(2.5)
        csv_btn.label.set_text('Save CSV')
        fig.canvas.draw_idle()
    threading.Thread(target=_reset, daemon=True).start()

csv_btn.on_clicked(on_save_csv)

radio.on_clicked(on_radio)
sampling_radio.on_clicked(on_sampling_change)
beam_slider.on_changed(on_laser_shape_change)
pulse_slider.on_changed(on_laser_shape_change)
detector_slider.on_changed(on_detector_change)
tc_slider.on_changed(on_laser_shape_change)
speed_slider.on_changed(on_speed_change)

def update(_frame):
    frame_index = int(state['playback_position'])
    artists = render_frame(frame_index)
    state['playback_position'] = (
        state['playback_position'] + state['playback_rate']
    ) % n_frames
    return artists

ani = animation.FuncAnimation(
    fig, update, frames=n_frames,
    interval=50, blit=False, repeat=True
)

plt.show()
