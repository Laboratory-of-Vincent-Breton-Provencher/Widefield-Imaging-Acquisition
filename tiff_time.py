import tifffile

def get_bofTime_us(tiff_path):
    with tifffile.TiffFile(tiff_path) as tif:
        desc = tif.pages[0].description
        i0 = desc.find('bofTime')
        if i0 == -1:
            raise ValueError(f"'bofTime' not found in {tiff_path}")
        a = desc[i0:].find('=')
        b = desc[i0:].find('us')
        ts = float(desc[i0+a+1:i0+b])
        return ts  # en microsecondes

# Exemple d'utilisation
f1 = 'ss_single_1.tiff'
f2 = 'ss_single_66799.tiff'

ts1 = get_bofTime_us(f1)
ts2 = get_bofTime_us(f2)

delta_us = abs(ts2 - ts1)
delta_s = delta_us * 1e-6

print(f"Écart de temps : {delta_us} µs ({delta_s} secondes)")