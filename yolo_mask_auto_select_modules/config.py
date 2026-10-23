"""Central settings for the YOLO/LaMa mask-selection pipeline."""
from pathlib import Path

# ==============================================================================
# 📁 ZENTRALE PFADE & QUELLDATEIEN (HIER BEARBEITEN)
# ==============================================================================
# Wurzelverzeichnis des Projekts (automatisch erkannt)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 1. Pfad zur CSV-Datei (bereits korrekt angepasst)
DEFAULT_ANCHORS_CSV = Path("/home/secret/Downloads/yabai proj/processed_output/anchors.csv")

# 2. Pfad zum Quellbilder-Ordner
DEFAULT_IMAGES_DIR = Path.home() / "Downloads/yabai proj/yabaiproject - Copy/mit_kanji"

# 3. Pfad zum YOLO-Checkpoint / Model-Weights
DEFAULT_WEIGHTS = Path.home() / ".cache/cleanup-models/manga-inpainter-best.pt"

# 4. Standard-Ausgabeordner für die Gewinner und Reports
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "yolo_mask_winners"


# ==============================================================================
# ⚙️ BATCH- & SPEICHER-EINSTELLUNGEN FÜR GROSSE MENGEN (700+ BILDER)
# ==============================================================================
# Wie viele Bilder gleichzeitig an LaMa geschickt werden (schützt /tmp und RAM):
LAMA_BATCH_SIZE = 25

# Hauptschalter für Debugging:
DEBUG_SAVE_CANDIDATES = True

# Granulare Schalter für Debug-Dateien:
DEBUG_SAVE_CLEANED_IMAGES = True       # Bereinigte PNGs speichern (in cleaned_images/)
DEBUG_SAVE_COMPARISON_SHEET = True     # Vergleichs-JPG speichern (in contact_sheets/)
DEBUG_SAVE_RAW_MASK_OVERLAYS = False   # Magenta-Overlays (spart viel Speicherplatz!)
DEBUG_SAVE_SIDE_BY_SIDE_SHEET = False  # Doppelte Side-by-Side Sheets


# ==============================================================================
# 🎯 PIPELINE- & ROUTING-EINSTELLUNGEN
# ==============================================================================
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_DISTANCES = "0,2,4,6,8,10"

# Alpha-Postprocessing für transparente Sprites
POSTPROCESS_REMOVE_SMALL_ALPHA_ISLANDS = True
ALPHA_SMALL_OBJECT_MIN_TRANSPARENT_RATIO = 0.40
ALPHA_SMALL_OBJECT_MAX_PIXELS = 64
ALPHA_VISIBILITY_THRESHOLD = 16

# Schneller Fallback für dünn besetzte transparente Sprites
ALPHA_SPARSE_BOX_FALLBACK_ENABLED = True
ALPHA_SPARSE_BOX_FALLBACK_MIN_TRANSPARENT_RATIO = 0.40
ALPHA_SPARSE_BOX_FALLBACK_MAX_BOXES = 2

# Schwellenwerte für die Beurteilung von Body-Box Abdeckungen im Audit
BODY_BOX_MIN_OVERLAP_PIXELS = 12
BODY_BOX_MIN_OVERLAP_FRACTION = 0.002
BODY_BOX_MIN_DETAIL_ENERGY_COVERAGE = 0.35
BODY_BOX_MIN_HIGH_CONTRAST_COVERAGE = 0.30
BODY_BOX_MIN_HIGH_CONTRAST_PIXELS = 8

# Gewichtungen für Hintergrund- und Strukturerhalt
STRUCTURE_WEIGHTS = {
    "sprite_contour_displacement_ratio": 0.45,
    "sprite_contour_orientation_loss_ratio": 0.15,
    "sprite_texture_distortion_ratio": 0.10,
    "sprite_color_shift_ratio": 0.10,
    "boundary_color_artifact_ratio": 0.05,
    "boundary_geometry_artifact_ratio": 0.05,
    "external_edge_destruction_score": 0.10,  # <-- NEU: Bestraft Zerstörung von UI-Elementen außerhalb der Box
}

TEXT_SCORE_TOLERANCE_FLOOR = 0.02
TEXT_SCORE_TOLERANCE_CEILING = 0.035
TEXT_SCORE_TOLERANCE_RANGE_FRACTION = 0.05
# Minimaler Spielraum gegen harte Grenzfälle bei der Textscore-Zulässigkeit.
TEXT_SCORE_BOUNDARY_SLACK = 0.0001
INK_STRENGTH_THRESHOLD = 0.28
# The T38 contrast-mask candidate joins the normal LaMa/scoring competition.
# T38 was selected from the previously inspected hard-image sweep.
CONTRAST_MASK_THRESHOLDS = (0.38,)
CONTRAST_MASK_PADDING_PX = 3
TEXT_CORRIDOR_DILATION_PX = 3
CONTOUR_MATCH_RADIUS_PX = 3

OUTLINE_LEAK_TEXT_WEIGHT = 0.15
OUTLINE_LEAK_DETAIL_THRESHOLD = 0.28
OUTLINE_LEAK_ABSOLUTE_DENSITY_FULL_SCALE = 0.12

# Box-Only Rescue Einstellungen
BOX_ONLY_RESCUE_ENABLED = True
BOX_ONLY_RESCUE_TEXT_ADVANTAGE = 0.006
BOX_ONLY_RESCUE_TEXT_ADVANTAGE_WITH_LEAK = 0.003
BOX_ONLY_RESCUE_MIN_YOLO_LEAK_SCORE = 0.08
BOX_ONLY_RESCUE_MIN_YOLO_INSIDE_RESIDUAL = 0.10
BOX_ONLY_RESCUE_INSIDE_DETAIL_ADVANTAGE = 0.025

# Sekundäre Auswahlregel: kleine verbliebene Textreste nur bei nahezu gleichem Struktur-Score
# Ein saubererer Kandidat darf den strukturell besten Kandidaten nur innerhalb dieser Toleranz ersetzen.
RESIDUAL_MICRO_COMPONENT_MAX_AREA = 24
RESIDUAL_SPOT_STRUCTURE_TOLERANCE = 0.05
RESIDUAL_SPOT_MIN_AREA_REDUCTION_PX = 2.0
# Micro-Component-Tiebreak darf die Sprite-Kontur nicht deutlich verschlechtern.
RESIDUAL_SPOT_MAX_CONTOUR_DISPLACEMENT_INCREASE = 0.05

# Multi-Scale Blob-Erkennung gegen Zensurbalken & Verfärbungen
RESIDUAL_BLOB_VISUAL_WEIGHT = 0.30
RESIDUAL_BLOB_LUMA_SCALE = 32.0
RESIDUAL_BLOB_CHROMA_SCALE = 28.0
RESIDUAL_BLOB_RING_RADIUS_PX = 8
RESIDUAL_BLOB_RING_BASELINE_FACTOR = 0.75
RESIDUAL_BLOB_FULL_SCALE = 0.18

# Smart-Ink Einstellungen
SMART_INK_INSET_PX = 2
SMART_INK_CONTRAST_THRESHOLD = 0.16
SMART_INK_DILATION_RADIUS = 2

# Exportiert automatisch alle Variablen in Großbuchstaben
__all__ = [name for name in globals() if name.isupper()]