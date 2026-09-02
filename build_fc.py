from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
L = prs.slide_layouts[6]
C_DARK = RGBColor(15, 23, 42)
C_WHITE = RGBColor(255, 255, 255)
C_NAVY = RGBColor(30, 27, 75)
C_PRI = RGBColor(79, 70, 229)
C_CYAN = RGBColor(6, 182, 212)
C_EM = RGBColor(16, 185, 129)
C_AM = RGBColor(245, 158, 11)
C_ROSE = RGBColor(239, 68, 68)
C_BG_L = RGBColor(248, 250, 252)
C_TEXT = RGBColor(30, 41, 59)
C_MUTED = RGBColor(100, 116, 139)
