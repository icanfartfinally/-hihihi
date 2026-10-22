"""
Meissner Effect Magnetic Levitation Simulator
---------------------------------------------
An interactive desktop simulation calculating and visualizing the magnetic
repulsive force F(d) = (k * M^2) / d^3 and equilibrium levitation height of
a permanent magnet above a superconductor.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle


# =============================================================================
# 1. PHYSICAL CONSTANTS & DEFAULT PARAMETERS
# =============================================================================
G_ACCEL = 9.81          # Gravitational acceleration (m/s^2)
DEFAULT_M = 1.00        # Relative magnetic strength factor (dimensionless)
DEFAULT_MASS_G = 25.0   # Mass of the permanent magnet (grams)
DEFAULT_K = 2.0e-6      # Baseline geometric-magnetic coupling constant (N * m^3)

# Distance domain for evaluation: 2 mm to 30 mm (converted to meters)
D_MIN_MM = 2.0
D_MAX_MM = 30.0
Y_MAX_N = 2.5           # Maximum force displayed on the Y-axis (Newtons)

d_meters = np.linspace(D_MIN_MM * 1e-3, D_MAX_MM * 1e-3, 600)
d_mm = d_meters * 1e3


# =============================================================================
# 2. CORE PHYSICS FUNCTIONS
# =============================================================================
def calculate_levitation_force(d: np.ndarray, mag_strength: float, k_const: float) -> np.ndarray:
    """
    Calculate the repulsive Meissner force using the inverse-cube relationship:
        F(d) = (k * M^2) / d^3
    """
    effective_k = k_const * (mag_strength ** 2)
    return effective_k / np.power(d, 3)


def calculate_equilibrium_gap(mag_strength: float, mass_grams: float, k_const: float):
    """
    Solve F_mag(d_eq) = m * g for the stable equilibrium levitation gap d_eq:
        d_eq = ((k * M^2) / (m * g))^(1/3)

    Returns:
        tuple: (Equilibrium gap in mm, Gravitational weight in N)
    """
    mass_kg = mass_grams * 1e-3
    weight_n = mass_kg * G_ACCEL
    effective_k = k_const * (mag_strength ** 2)
    d_eq_meters = np.cbrt(effective_k / weight_n)
    return float(d_eq_meters * 1e3), float(weight_n)


def print_physics_summary() -> None:
    """Print an ASCII-safe physics summary to the console upon launching."""
    summary = (
        "\n" + "=" * 76 + "\n"
        "  MEISSNER EFFECT MAGNETIC LEVITATION -- PHYSICS SUMMARY\n"
        + "=" * 76 + "\n"
        "1. Flux Expulsion (Meissner-Ochsenfeld Effect):\n"
        "   When a superconductor is cooled below its critical temperature (T_c),\n"
        "   persistent surface screening currents expel external magnetic flux\n"
        "   so that the internal magnetic field vanishes (B = 0).\n\n"
        "2. Method of Images & Force Law:\n"
        "   To satisfy B_normal = 0 at the superconducting boundary, a magnet at\n"
        "   height +d induces a virtual 'mirror-image' dipole at depth -d inside\n"
        "   the superconductor with opposing polarity, generating pure repulsion.\n"
        "   In the finite-geometry inverse-cube regime, the repulsive force is:\n"
        "       F_mag(d) = (k * M^2) / d^3\n"
        "   Because of the 1/d^3 dependence, halving the gap distance increases\n"
        "   the upward repulsive force by a factor of 8.\n\n"
        "3. Stable Levitation Equilibrium:\n"
        "   The magnet floats at the equilibrium gap d_eq where upward magnetic\n"
        "   repulsion balances downward weight (F_mag = m * g):\n"
        "       d_eq = [ (k * M^2) / (m * g) ]^(1/3)\n"
        + "=" * 76 + "\n"
    )
    print(summary)


# =============================================================================
# 3. MATPLOTLIB GUI & REAL-TIME INTERACTIVE VISUALIZATION
# =============================================================================
def run_interactive_app() -> None:
    """Initialize the Matplotlib figure, plots, sliders, and callbacks."""
    print_physics_summary()

    # Create main figure with two side-by-side panels
    fig, (ax_plot, ax_schem) = plt.subplots(
        1, 2, figsize=(12, 6.8), gridspec_kw={"width_ratios": [1.6, 1.0]}
    )

    # Safely set window title if supported by the current Matplotlib backend
    if fig.canvas.manager is not None:
        try:
            fig.canvas.manager.set_window_title("Meissner Effect Magnetic Levitation Simulator")
        except AttributeError:
            pass

    plt.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.28, wspace=0.26)

    # Initial calculations
    f_initial = calculate_levitation_force(d_meters, DEFAULT_M, DEFAULT_K)
    d_eq_mm, weight_n = calculate_equilibrium_gap(DEFAULT_M, DEFAULT_MASS_G, DEFAULT_K)

    # -------------------------------------------------------------------------
    # Panel 1: Force vs. Gap Distance Curve
    # -------------------------------------------------------------------------
    (line_force,) = ax_plot.plot(
        d_mm, f_initial, color="#1f77b4", lw=2.5,
        label=r"Meissner Repulsion $F_{\mathrm{mag}}(d) = \frac{k M^2}{d^3}$"
    )
    (line_weight,) = ax_plot.plot(
        [D_MIN_MM, D_MAX_MM], [weight_n, weight_n], color="#d62728", ls="--", lw=1.8,
        label=r"Gravitational Load $F_g = m g$"
    )
    (point_eq,) = ax_plot.plot(
        [d_eq_mm], [weight_n], marker="o", markersize=9,
        color="#2ca02c", ls="None", zorder=5,
        label=r"Equilibrium Point $(d_{\mathrm{eq}}, F_g)$"
    )
    (vline_eq,) = ax_plot.plot(
        [d_eq_mm, d_eq_mm], [0.0, weight_n], color="#2ca02c", ls=":", lw=1.5, alpha=0.85
    )

    # Dynamic telemetry box on the plot
    status_text = ax_plot.text(
        0.96, 0.68, "", transform=ax_plot.transAxes, fontsize=9.5,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc")
    )

    ax_plot.set_title("Magnetic Repulsive Force vs. Gap Distance", fontsize=12, fontweight="bold", pad=10)
    ax_plot.set_xlabel("Gap Distance $d$ (mm)", fontsize=11)
    ax_plot.set_ylabel("Force $F$ (Newtons)", fontsize=11)
    ax_plot.set_xlim(D_MIN_MM, D_MAX_MM)
    ax_plot.set_ylim(0.0, Y_MAX_N)
    ax_plot.grid(True, which="major", linestyle="-", alpha=0.5)
    ax_plot.grid(True, which="minor", linestyle=":", alpha=0.25)
    ax_plot.minorticks_on()
    ax_plot.legend(loc="upper right", fontsize=9.5, framealpha=0.95)

    # -------------------------------------------------------------------------
    # Panel 2: Physical Schematic (Method of Images)
    # -------------------------------------------------------------------------
    ax_schem.set_title("Real-Time Levitation & Image Dipole", fontsize=12, fontweight="bold", pad=10)
    ax_schem.set_xlim(-10, 10)
    ax_schem.set_ylim(-32, 32)
    ax_schem.set_ylabel("Vertical Position $z$ (mm)", fontsize=11)
    ax_schem.set_xticks([])
    ax_schem.grid(True, axis="y", linestyle=":", alpha=0.4)

    # Superconductor bulk region (z <= 0)
    sc_zone = Rectangle((-10, -32), 20, 32, facecolor="#e3f2fd", edgecolor="none", zorder=0)
    ax_schem.add_patch(sc_zone)
    ax_schem.axhline(0, color="#0d47a1", lw=2.5, zorder=3)
    ax_schem.text(
        0, -2.8, r"Superconductor Surface ($z = 0$, $\mathbf{B}_{\mathrm{in}} = 0$)",
        ha="center", va="top", fontsize=8.5, fontweight="bold", color="#0d47a1"
    )

    # Permanent magnet (above z = 0) and Mirror-Image dipole (below z = 0)
    mag_width, mag_height = 7.0, 4.0
    real_mag = Rectangle(
        (-mag_width / 2, d_eq_mm), mag_width, mag_height,
        facecolor="#ff7043", edgecolor="#bf360c", lw=1.8, zorder=4
    )
    image_mag = Rectangle(
        (-mag_width / 2, -d_eq_mm - mag_height), mag_width, mag_height,
        facecolor="#ffab91", edgecolor="#bf360c", lw=1.5, ls="--", alpha=0.65, zorder=2
    )
    ax_schem.add_patch(real_mag)
    ax_schem.add_patch(image_mag)

    label_real = ax_schem.text(
        0, d_eq_mm + mag_height / 2, "Permanent Magnet\n(N -> S)",
        ha="center", va="center", fontsize=8, fontweight="bold", color="white", zorder=5
    )
    label_image = ax_schem.text(
        0, -d_eq_mm - mag_height / 2, "Virtual Image Dipole\n(S <- N)",
        ha="center", va="center", fontsize=8, style="italic", color="#4e342e", zorder=3
    )

    # -------------------------------------------------------------------------
    # Interactive Sliders & Reset Button
    # -------------------------------------------------------------------------
    ax_slider_m = plt.axes([0.14, 0.14, 0.65, 0.03], facecolor="#eef2f7")
    ax_slider_mass = plt.axes([0.14, 0.09, 0.65, 0.03], facecolor="#eef2f7")
    ax_slider_k = plt.axes([0.14, 0.04, 0.65, 0.03], facecolor="#eef2f7")
    ax_btn_reset = plt.axes([0.83, 0.08, 0.11, 0.05])

    slider_m = Slider(
        ax=ax_slider_m, label="Magnetic Strength ($M$)",
        valmin=0.3, valmax=2.5, valinit=DEFAULT_M, valstep=0.05, color="#1f77b4"
    )
    slider_mass = Slider(
        ax=ax_slider_mass, label="Magnet Mass ($g$)",
        valmin=5.0, valmax=100.0, valinit=DEFAULT_MASS_G, valstep=1.0, color="#d62728"
    )
    slider_k = Slider(
        ax=ax_slider_k, label=r"Coupling $k$ ($\mu\mathrm{N}\cdot\mathrm{m}^3$)",
        valmin=0.5, valmax=5.0, valinit=DEFAULT_K * 1e6, valstep=0.1, color="#2ca02c"
    )
    btn_reset = Button(ax_btn_reset, "Reset", color="#f0f0f0", hovercolor="#dcdcdc")

    def update_simulation(_=None) -> None:
        """Callback to recalculate physics and update plots in real time."""
        m_val = float(slider_m.val)
        mass_g = float(slider_mass.val)
        k_val = float(slider_k.val) * 1e-6  # Convert from uN*m^3 to N*m^3

        # Update force curve and equilibrium coordinates
        f_new = calculate_levitation_force(d_meters, m_val, k_val)
        d_eq_new_mm, weight_new_n = calculate_equilibrium_gap(m_val, mass_g, k_val)

        line_force.set_ydata(f_new)
        line_weight.set_ydata([weight_new_n, weight_new_n])
        point_eq.set_data([d_eq_new_mm], [weight_new_n])
        vline_eq.set_data([d_eq_new_mm, d_eq_new_mm], [0.0, weight_new_n])

        # Update magnet and image positions in schematic
        real_mag.set_y(d_eq_new_mm)
        image_mag.set_y(-d_eq_new_mm - mag_height)
        label_real.set_position((0, d_eq_new_mm + mag_height / 2))
        label_image.set_position((0, -d_eq_new_mm - mag_height / 2))

        # Check if equilibrium is within plotted range
        if D_MIN_MM <= d_eq_new_mm <= D_MAX_MM:
            state_str = "STABLE LEVITATION"
        elif d_eq_new_mm < D_MIN_MM:
            state_str = "NEAR SURFACE CONTACT (< 2 mm)"
        else:
            state_str = "HIGH ALTITUDE (> 30 mm)"

        status_text.set_text(
            f"State: {state_str}\n"
            f"Equilibrium Gap ($d_{{eq}}$): {d_eq_new_mm:.2f} mm\n"
            f"Mirror Image Depth: -{d_eq_new_mm:.2f} mm\n"
            f"Weight / Load ($F_g$): {weight_new_n:.3f} N\n"
            f"Effective $k M^2$: {k_val * (m_val**2):.2e} N*m^3"
        )

        fig.canvas.draw_idle()

    def reset_sliders(_event) -> None:
        """Reset all sliders to their initial physical parameters."""
        slider_m.reset()
        slider_mass.reset()
        slider_k.reset()

    # Register callbacks
    slider_m.on_changed(update_simulation)
    slider_mass.on_changed(update_simulation)
    slider_k.on_changed(update_simulation)
    btn_reset.on_clicked(reset_sliders)

    # Keep widget references attached to the figure to prevent garbage collection
    fig._widgets = (slider_m, slider_mass, slider_k, btn_reset)

    update_simulation()
    plt.show()


if __name__ == "__main__":
    run_interactive_app()