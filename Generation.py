#gemini review random change from V1.1
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line_aa
from skimage.morphology import dilation, footprint_rectangle
import os

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
FIXED_WIDTH_CM = 20.0 
FIXED_HEIGHT_CM = 5
def check_constraints(L, alpha_deg, l, r):
    """All lengths in the same unit (cm). Returns (C1_ok, C2_ok, Gx_prime)."""
    a = np.radians(alpha_deg)
    Oy_prime = l * np.sin(2 * a) - r * np.cos(2 * a)
    if l * np.cos(2 * a) > L:
        return False, False, None
    if abs(Oy_prime) > r:
        return False, False, None
    disc = r**2 - Oy_prime**2
    Gx_prime = l * np.cos(2 * a) + r * np.sin(2 * a) + np.sqrt(disc)
    return True, Gx_prime <= L, Gx_prime


def build_tesla_geometry(L, alpha_deg, l, r, offset_x=0.0):
    """
    Build one unit-cell of the Tesla valve geometry.
    All lengths in cm.
    offset_x shifts the whole pattern along x (for tiling n repetitions).
    Returns a dict of 2-D world points (cm).
    """
    a = np.radians(alpha_deg)
    d_pos     = np.array([ np.cos(a),  np.sin(a)])
    d_neg     = np.array([ np.cos(a), -np.sin(a)])
    perp_down = np.array([ np.sin(a), -np.cos(a)])
    perp_up   = np.array([ np.sin(a),  np.cos(a)])

    ox = np.array([offset_x, 0.0])

    A   = np.array([0.0, 0.0]) + ox
    Bp  = A  + L * d_pos
    Ap  = Bp + L * d_neg
    fin = Ap + L * d_pos

    C  = Bp + l * d_pos
    O1 = C  + r * perp_down

    Oy_p = l * np.sin(2 * a) - r * np.cos(2 * a)
    disc = r**2 - Oy_p**2
    Gx_p = l * np.cos(2 * a) + r * np.sin(2 * a) + np.sqrt(max(disc, 0))
    G  = Bp + Gx_p * d_neg

    D  = Ap + l * d_neg
    O2 = D  + r * perp_up
    H  = Ap + Gx_p * d_pos

    return dict(A=A, Bp=Bp, Ap=Ap, fin=fin, C=C, O1=O1, G=G, D=D, O2=O2, H=H)


# ---------------------------------------------------------------------------
# Main mask builder
# ---------------------------------------------------------------------------

def tesla_valve_mask(
    L,                      # cm — length of each diagonal segment
    alpha_deg,              # degrees — half-angle of the main channel
    l,                      # cm — loop arm length
    r,                      # cm — loop radius
    n=1,                    # number of repeated upper+lower loop patterns
    physical_length_cm=20,  # cm — total x-extent of the output matrix
    ppcm=20,                # pixels per cm (controls resolution)
    tube_width_cm=0.3,      # cm — physical width of the carved tube
    margin_cm=0.5,          # cm — blank border around geometry
):
    """
    Returns a boolean mask (True = wall, False = fluid) whose x-dimension
    always represents `physical_length_cm` centimetres regardless of L, l, r.

    Parameters
    ----------
    L, l, r         : geometry lengths in **cm**
    alpha_deg       : angle in degrees
    n               : number of repetitions of the upper+lower loop block
    physical_length_cm : total physical x-size of the output image (cm)
    ppcm            : pixels per centimetre (resolution)
    tube_width_cm   : tube/channel width in cm
    margin_cm       : whitespace border in cm
    """
    C1, C2, _ = check_constraints(L, alpha_deg, l, r)
    if not (C1 and C2):
        raise ValueError(f"Invalid geometry: C1={C1}, C2={C2}")

    # ------------------------------------------------------------------
    # Build ALL n repetitions, collecting every world point for bounding
    # ------------------------------------------------------------------
    # The x-stride between successive patterns is the x-projection of 2L*d_pos
    a = np.radians(alpha_deg)
    stride_x = 2 * L * np.cos(a)          # x advance per one (Bp→Ap→fin) cell

    all_cells = []
    for i in range(n):
        cell = build_tesla_geometry(L, alpha_deg, l, r, offset_x=i * stride_x)
        all_cells.append(cell)

    # Bounding box of all cells (world coordinates, cm)
    all_pts = np.vstack([np.array(list(cell.values())) for cell in all_cells])
    xmin_geom, ymin_geom = all_pts.min(axis=0)
    xmax_geom, ymax_geom = all_pts.max(axis=0)

    # ------------------------------------------------------------------
    # Fixed physical domain: x ∈ [0, physical_length_cm]
    # y is centred on the geometry + margin
    # ------------------------------------------------------------------
    x_domain = physical_length_cm
    y_geom   = (ymax_geom - ymin_geom) + 2 * margin_cm
    y_domain = FIXED_HEIGHT_CM#y_geom                  # !!!!!!CHANGED THIS TO FIXED HEIGHT (cm)!!!!!!!!!!

    # Pixel grid dimensions
    nx = int(round(x_domain * ppcm))
    ny = int(round(y_domain * ppcm))

    # World→pixel transform
    # x=0 maps to col=0, x=physical_length_cm maps to col=nx-1
    # We centre the geometry horizontally inside the domain
    x_offset = (physical_length_cm - (xmax_geom - xmin_geom)) / 2 - xmin_geom
    y_offset  = margin_cm - ymin_geom   # bottom of geometry sits margin_cm from bottom
#latest addon
    y_geom_center = (ymax_geom + ymin_geom) / 2
    y_offset = (FIXED_HEIGHT_CM / 2) - y_geom_center
    scale = ppcm   # 1 cm = ppcm pixels (uniform — no aspect distortion)

    def w2g(pt):
        col = int(round((pt[0] + x_offset) * scale))
        row = int(round((ny - 1) - (pt[1] + y_offset) * scale))
        return np.clip(row, 0, ny - 1), np.clip(col, 0, nx - 1)

    # ------------------------------------------------------------------
    # Canvas
    # ------------------------------------------------------------------
    canvas = np.zeros((ny, nx), dtype=np.float32)
    tube_px = max(1, int(round(tube_width_cm * ppcm)))
    fp = footprint_rectangle((tube_px, tube_px))

    def carve_segment(p0, p1):
        r0, c0 = w2g(p0)
        r1, c1 = w2g(p1)
        tmp = np.zeros((ny, nx), dtype=np.float32)
        rr, cc, _ = line_aa(r0, c0, r1, c1)
        tmp[np.clip(rr, 0, ny-1), np.clip(cc, 0, nx-1)] = 1.0
        return dilation(tmp > 0, fp).astype(np.float32)

    def carve_arc(O_world, r_world, angle_start_deg, angle_end_deg,
                  clockwise=False, n_points=500):
        a_s = np.radians(angle_start_deg)
        a_e = np.radians(angle_end_deg)
        if clockwise:
            if a_e > a_s: a_e -= 2 * np.pi
        else:
            if a_e < a_s: a_e += 2 * np.pi
        angles = np.linspace(a_s, a_e, n_points)
        tmp = np.zeros((ny, nx), dtype=np.float32)
        for ang in angles:
            pt = O_world + r_world * np.array([np.cos(ang), np.sin(ang)])
            ro, co = w2g(pt)
            tmp[ro, co] = 1.0
        return dilation(tmp > 0, fp).astype(np.float32)

    # ------------------------------------------------------------------
    # Draw each cell
    # ------------------------------------------------------------------
    for i, pts in enumerate(all_cells):
        # Main channel segments
        if i == 0:
            pass   # pts['A']→pts['Bp'] is the inlet — skip as in original
        canvas += carve_segment(pts['A'], pts['Bp'])#####inlet segment
        canvas += carve_segment(pts['Bp'], pts['Ap'])
        canvas += carve_segment(pts['Ap'], pts['fin'])

        # Upper loop (clockwise arc)
        canvas += carve_segment(pts['Bp'], pts['C'])
        O1, C_pt, G_pt = pts['O1'], pts['C'], pts['G']
        a_C = np.degrees(np.arctan2(C_pt[1] - O1[1], C_pt[0] - O1[0]))
        a_G = np.degrees(np.arctan2(G_pt[1] - O1[1], G_pt[0] - O1[0]))
        canvas += carve_arc(O1, r, a_C, a_G, clockwise=True)

        # Lower loop (counter-clockwise arc)
        canvas += carve_segment(pts['Ap'], pts['D'])
        O2, D_pt, H_pt = pts['O2'], pts['D'], pts['H']
        a_D = np.degrees(np.arctan2(D_pt[1] - O2[1], D_pt[0] - O2[0]))
        a_H = np.degrees(np.arctan2(H_pt[1] - O2[1], H_pt[0] - O2[0]))
        canvas += carve_arc(O2, r, a_D, a_H, clockwise=False)
############################
# ------------------------------------------------------------------
    # Ajout des tubes de liaison parfaitement alignés
    # ------------------------------------------------------------------
    
    # 1. Tube d'entrée (Gauche -> Point A)
    p_start = all_cells[0]['A']
    # On crée un point sur le bord gauche (colonne 0) à la MÊME hauteur que A
    # Pour le monde réel, x = -x_offset correspond à la colonne 0 du masque
    edge_left = np.array([-x_offset, p_start[1]])
    canvas += carve_segment(edge_left, p_start)

    # 2. Tube de sortie (Point fin -> Droite)
    p_end = all_cells[-1]['fin']
    # On crée un point sur le bord droit (dernière colonne) à la MÊME hauteur que fin
    edge_right = np.array([physical_length_cm - x_offset, p_end[1]])
    canvas += carve_segment(p_end, edge_right)
############################
    mask = canvas == 0   # True = wall (uncarved), False = fluid

    # Return mask + metadata useful for downstream processing
    info = dict(
        nx=nx, ny=ny,
        ppcm=ppcm,
        physical_length_cm=physical_length_cm,
        y_domain_cm=y_domain,
        tube_width_px=tube_px,
    )
    return mask, info


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
def get_tesla_array(L, l, r, a, ppcm, n=1):
    L_bound = 10/(3*np.cos(a))
    if L is None:
        L=np.random.uniform(1, L_bound)
        L=0.75*L_bound #CRAZY SHIT REDUCING TO 3 PARAMETER
        l_bound = 0.8*L
    r_bound = 0.5*L
    if l is None:
        l=np.random.uniform(0.1, l_bound)
    if r is None:
        r = np.random.uniform(0.1, r_bound)
    """
    Returns an array of a FIXED size. 
    If the geometry exceeds these dimensions, returns None.
    """
    # 1. SET YOUR BOUNDARIES (in cm)
    # These determine the constant pixel resolution of your GA
    
    # 2. Pre-calculate the geometry extent to see if it fits redondant???
    alpha_rad = np.radians(a)
    stride_x = 2 * L * np.cos(alpha_rad)
    total_x_geom = (stride_x * n) + (r + l) # Approximate max width
    
    # Simple check: If it's physically impossible to fit in our 'box'
    if total_x_geom > FIXED_WIDTH_CM:
        return None, None, None

    try:
        # We pass the FIXED physical dimensions to your original function
        mask, info = tesla_valve_mask(
            L=L,
            alpha_deg=a,
            l=l,
            r=r,
            n=n, 
            ppcm=ppcm,
            physical_length_cm=FIXED_WIDTH_CM, 
            tube_width_cm=0.25,
            margin_cm=1.0 
        )
        
        # NOTE: Your original script calculates y_domain dynamically.
        # To force the HEIGHT to be fixed as well, we would need to 
        # slightly tweak the 'ny' calculation inside tesla_valve_mask.
        
        return mask, info, (L, l, r, a)  # Return geometry params for reference
    except ValueError:
        return None, None, None
def display_valve(valve_array, info):
    """
    Displays the array without distortion.
    """
    if valve_array is None:
        print("Geometry invalid: Check constraints (L, l, r, a)")
        return

    plt.figure(figsize=(12, 4))
    
    # ~mask flips it so the channel is dark and walls are light (easier to see)
    plt.imshow(~valve_array, cmap='Greys', origin='upper', aspect='equal')
    
    plt.title(f"Tesla Valve Mask | {info['nx']}x{info['ny']} px")
    plt.axis('off') # Hides the pixel numbers for a cleaner look
    plt.show()

#display_valve(get_tesla_array(unit = 3,L=3, l=1, r=1, a=25, ppcm=100))
#my_mask, my_info = get_tesla_array(L=3, l=2, r=1, a=25, ppcm=50, n=3)
#display_valve(my_mask, my_info)

i = 0
while i < 5:
    #my_mask, my_info, geometry_params = get_tesla_array(L=None, l=None, r=None, a=np.random.uniform(10,60), ppcm=30, n=3)#10 60
    my_mask, my_info, geometry_params = get_tesla_array(L=3, l=1, r=1, a=20, ppcm=30, n=3)
    display_valve(my_mask, my_info)
    if my_mask is not None:
        i += 1
        print(geometry_params)
    else:
        print("Invalid parameters")