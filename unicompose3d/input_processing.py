"""Generic target cutouts and conservative input diagnostics, no category rules."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw


def image_quality(path):
    with Image.open(path) as source:
        rgb = np.array(source.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Comparable blur metric at a bounded reference scale, not native pixels.
    ratio = min(1, 512 / max(gray.shape))
    reference = cv2.resize(gray, None, fx=ratio, fy=ratio) if ratio < 1 else gray
    sharpness = float(cv2.Laplacian(reference, cv2.CV_64F).var())
    warnings = []
    if min(rgb.shape[:2]) < 256:
        warnings.append("Low-resolution image; use a clearer original.")
    if sharpness < 35:
        warnings.append("The image may be blurred or lack detail; inspect it before use.")
    if float((gray < 8).mean()) > 0.35 or float((gray > 247).mean()) > 0.65:
        warnings.append("Large underexposed or overexposed areas may lose detail.")
    return {"sharpness": round(sharpness, 1), "warnings": warnings}


def make_cutout(image_path, mask_path, destination):
    """Preserve holes and disconnected product parts; do not fill/smooth masks."""
    with Image.open(image_path) as source, Image.open(mask_path) as mask_image:
        image = source.convert("RGBA")
        if mask_image.size != image.size:
            raise ValueError("Mask dimensions do not match the original image.")
        mask = np.array(mask_image.convert("L"))
        alpha = np.array(image.getchannel("A"))
    mask = np.minimum(mask, alpha)
    foreground = mask > 127
    if foreground.sum() < max(16, foreground.size * 0.0005):
        raise ValueError("The mask is empty or too small. Redraw the box or add keep points.")
    image.putalpha(Image.fromarray(mask))
    # Keep original frame coordinates for the mask editor and multi-view input.
    # Hunyuan's own image processor handles centering/padding from alpha.
    image.save(destination, "PNG")
    warnings = []
    if foreground[[0, -1], :].any() or foreground[:, [0, -1]].any():
        warnings.append("The object touches the image edge and may be cropped.")
    if foreground.mean() > 0.96:
        warnings.append("The mask retains almost the whole image. Check for background or the wrong target.")
    return {"foreground_fraction": round(float(foreground.mean()), 4), "warnings": warnings}




def background_region(rgb, alpha, seed, tolerance):
    """CPU GrabCut expands a background seed across multiple background colours."""
    height,width=alpha.shape
    scale=min(1.,640/max(height,width))
    size=(max(1,round(width*scale)),max(1,round(height*scale)))
    image=cv2.resize(rgb,size,interpolation=cv2.INTER_AREA) if scale<1 else rgb.copy()
    foreground=cv2.resize((alpha>127).astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0 if scale<1 else alpha>127
    x=min(size[0]-1,round(seed[0]*scale));y=min(size[1]-1,round(seed[1]*scale))
    # Preserve the confidently selected outside edge of the object. The unknown
    # interior may contain a hole with several background colours/shadows.
    eroded=cv2.erode(foreground.astype(np.uint8),np.ones((3,3),np.uint8))>0
    boundary=foreground & ~eroded
    mask=np.where(foreground,cv2.GC_PR_FGD,cv2.GC_BGD).astype(np.uint8)
    mask[boundary]=cv2.GC_FGD
    cv2.circle(mask,(x,y),max(1,round(min(size)*.004)),int(cv2.GC_BGD),-1)
    if np.count_nonzero(mask==cv2.GC_FGD)<4 or np.count_nonzero(mask==cv2.GC_BGD)<4:
        return None
    try:
        cv2.grabCut(image,mask,None,np.zeros((1,65),np.float64),
                    np.zeros((1,65),np.float64),2,cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    background=((mask==cv2.GC_BGD)|(mask==cv2.GC_PR_BGD)).astype(np.uint8)
    _,labels=cv2.connectedComponents(background,connectivity=4)
    label=labels[y,x]
    if not label:
        return None
    selected=(labels==label).astype(np.uint8)
    if scale<1:
        selected=cv2.resize(selected,(width,height),interpolation=cv2.INTER_NEAREST)
    return selected>0


def brush_mask(original, cutout, strokes, destination):
    """Edit alpha in the original frame; never infer holes from RGB brightness."""
    with Image.open(original) as source, Image.open(cutout) as current:
        if source.size != current.size:
            raise ValueError("Image and mask dimensions differ.")
        alpha = current.convert("RGBA").getchannel("A")
        width, height = alpha.size
    with Image.open(original) as source:
        rgb = np.asarray(source.convert("RGB"))
    draw = ImageDraw.Draw(alpha)
    region_pixels = 0
    for stroke in strokes:
        radius = max(1, round(stroke.radius * min(width, height)))
        points = [(min(width - 1, round(p[0] * width)), min(height - 1, round(p[1] * height))) for p in stroke.points]
        if stroke.mode == "region":
            if stroke.keep:
                raise ValueError("Region selection is only available for background removal.")
            # User supplies the background seed. Fixed-range Lab flood fill
            # follows a connected colour region, never the entire alpha component.
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            flood = np.zeros((height + 2, width + 2), dtype=np.uint8)
            seed = points[0]
            cv2.floodFill(lab.copy(), flood, seed, (0, 0, 0),
                         (stroke.tolerance,) * 3, (stroke.tolerance,) * 3,
                         4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
            region = flood[1:-1, 1:-1] != 0
            values = np.asarray(alpha).copy()
            expanded = background_region(rgb, values, seed, stroke.tolerance)
            # Prefer the multi-colour region only if it respects the same area
            # guard. Fixed-range fill remains available for ambiguous refinement.
            if expanded is not None:
                expanded_count = int((expanded & (values>127)).sum())
                if expanded_count <= int((values>127).sum())*.60:
                    region |= expanded
            affected = int((region & (values > 127)).sum())
            before = int((values > 127).sum())
            if before and affected > before * .60:
                raise ValueError("Selection spreads into too many objects. Reduce sensitivity or erase individual strokes; the previous mask is retained.")
            values[region] = 0
            region_pixels += affected
            alpha = Image.fromarray(values)
            draw = ImageDraw.Draw(alpha)
            continue
        fill = 255 if stroke.keep else 0
        if len(points) > 1:
            draw.line(points, fill=fill, width=2 * radius)
        for x, y in points:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
    temporary = destination.with_suffix(".mask.png")
    try:
        alpha.save(temporary)
        return {**make_cutout(original, temporary, destination), "region_pixels_removed": region_pixels}
    finally:
        temporary.unlink(missing_ok=True)
